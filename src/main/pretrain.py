import json

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from config import PATHS, MODEL, PRETRAIN
from common import get_device, configure_backends, amp_dtype, autocast_ctx, load_or_init, evaluate, save_checkpoint, get_lr, build_token_bytes, tokenizer_fingerprint
from dataset import ShardedPretrainDataset, ResumableSampler
from tokenizer import Tokenizer


def load_tokenizer() -> Tokenizer:
    if not PATHS.tok.exists():
        raise FileNotFoundError(f'[tokenizer]: {PATHS.tok} not found; the tokenizer is built by prepare.py, run `python3 prepare.py` first')
    return Tokenizer.load(PATHS.tok)


def load_shard_meta(tokenizer: Tokenizer) -> dict:
    if not PATHS.shard_meta.exists():
        raise FileNotFoundError(f'[shards]: {PATHS.shard_meta} not found; run `python3 prepare.py` to build the token shards')
    meta = json.loads(PATHS.shard_meta.read_text(encoding = 'utf-8'))
    fingerprint = tokenizer_fingerprint(PATHS.tok)
    if meta.get('tokenizer') != fingerprint:
        raise ValueError(f'[shards]: shards were built with tokenizer {meta.get("tokenizer")} but {PATHS.tok} is now {fingerprint}; the token ids mean different things now, rerun `python3 prepare.py --force`')
    if meta.get('vocab_size') != tokenizer.vocab_size:
        raise ValueError(f'[shards]: shards say vocab {meta.get("vocab_size")} but the tokenizer says {tokenizer.vocab_size}; rerun `python3 prepare.py --force`')
    return meta


def main() -> None:
    device = get_device()
    configure_backends()
    dtype = amp_dtype(device)
    tokenizer = load_tokenizer()
    token_bytes = build_token_bytes(tokenizer)
    meta = load_shard_meta(tokenizer)

    grad_accum = PRETRAIN.grad_accum(MODEL.block_size)
    workers = PRETRAIN.num_workers if device.type == 'cuda' else 0
    pin = device.type == 'cuda'

    train_ds = ShardedPretrainDataset(PATHS.shard_dir, MODEL.block_size, prefix = 'train')
    val_ds = ShardedPretrainDataset(PATHS.shard_dir, MODEL.block_size, prefix = 'val')
    sampler = ResumableSampler(len(train_ds), PRETRAIN.shuffle_seed)
    train_loader = DataLoader(train_ds, batch_size = PRETRAIN.micro_batch_size, sampler = sampler, num_workers = workers, pin_memory = pin, drop_last = True)
    val_loader = DataLoader(val_ds, batch_size = PRETRAIN.micro_batch_size, shuffle = False, num_workers = workers, pin_memory = pin)

    for split, dataset in [('train', train_ds), ('val', val_ds)]:
        if dataset.total_tokens != meta[split]['total_tokens']:
            raise ValueError(f'[shards]: {split} shards hold {dataset.total_tokens} tokens but {PATHS.shard_meta} says {meta[split]["total_tokens"]}; a shard file was added or deleted, rerun `python3 prepare.py --force`')
    print(f'[data]: train {train_ds.total_tokens} / val {val_ds.total_tokens} tokens | '
          f'{len(train_ds.shard_paths)} + {len(val_ds.shard_paths)} shards | '
          f'{len(train_loader) // grad_accum} steps per epoch')
    print(f'[data]: mix ' + ', '.join(f'{d} {share * 100:.1f}%' for d, share in sorted(meta['train']['shares'].items())))
    if meta['train']['exhausted']:
        print(f'[data]: warning, these domains ran out of documents: {meta["train"]["exhausted"]}')
    total_steps, warmup_steps = PRETRAIN.schedule()
    print(f'[schedule]: budget {PRETRAIN.total_tokens} tokens = {total_steps} steps '
        f'({PRETRAIN.micro_batch_size} x {MODEL.block_size} x {grad_accum} = {PRETRAIN.tokens_per_step} tokens per step), warmup {warmup_steps}')
    print(f'[runtime]: device {device.type} | autocast {dtype if dtype is not None else "off (fp32)"} | dataloader workers {workers}')

    state = load_or_init(PATHS.pretrain_ckpt, tokenizer.vocab_size, device, lr = PRETRAIN.lr)
    model = state.model
    cfg = state.cfg
    optimizer = state.optimizer
    start_epoch = state.start_epoch
    glob_step = state.glob_step
    samples_seen = state.samples_seen

    print(f'[parameters]: the paramters of model is {sum(p.numel() for p in model.parameters())}')
    if glob_step >= total_steps:
        print(f'[schedule]: warning, glob_step {glob_step} already reached the budget {total_steps}, lr is pinned at the minimum')

    for epoch in range(start_epoch, start_epoch + PRETRAIN.epoches_per_run):
        sampler.set_epoch(epoch, samples_seen)
        model.train()
        optimizer.zero_grad(set_to_none = True)
        loss_accum = torch.zeros((), device = device)
        micro_step = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device, non_blocking = pin)
            labels = batch['labels'].to(device, non_blocking = pin)
            with autocast_ctx(device, dtype):
                logits = model(input_ids)
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits.reshape(B * T, V),
                    labels.reshape(B * T),
                )
            (loss / grad_accum).backward()
            loss_accum += loss.detach()
            samples_seen += input_ids.size(0)
            micro_step += 1
            if micro_step % grad_accum != 0:
                continue

            lr = get_lr(glob_step, PRETRAIN.lr, warmup_steps, total_steps, PRETRAIN.min_lr_ratio)
            for group in optimizer.param_groups:
                group['lr'] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), PRETRAIN.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
            train_loss = (loss_accum / grad_accum).item()
            loss_accum.zero_()
            glob_step += 1

            if glob_step % PRETRAIN.eval_every == 0:
                res = evaluate(model, val_loader, device, token_bytes)
                print(f'epoch {epoch:>4} | step {glob_step:>6} | lr {lr:.2e} | train loss = {train_loss:.4f} | val loss = {res.loss:.4f} | bpb = {res.bpb:.4f}')
            if glob_step % PRETRAIN.save_every == 0:
                save_checkpoint(model, cfg, optimizer, epoch, glob_step, PATHS.pretrain_ckpt, samples_seen)
        samples_seen = 0
        res = evaluate(model, val_loader, device, token_bytes)
        print(f'[epoch end] epoch {epoch:>4} | step {glob_step:>6} | val loss = {res.loss:.4f} | bpb = {res.bpb:.4f}')
        save_checkpoint(model, cfg, optimizer, epoch + 1, glob_step, PATHS.pretrain_ckpt, 0)

    return


if __name__ == '__main__':
    main()