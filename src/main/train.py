import argparse
import torch
import torch.nn.functional as F
import json

from torch.utils.data import DataLoader
from pathlib import Path

from config import PATHS, MODEL, SFT
from common import get_device, configure_backends, amp_dtype, autocast_ctx, load_for_sft, evaluate, save_checkpoint, get_lr, build_token_bytes
from dataset import SFTDataset, sft_collate, IGNORE_INDEX
from tokenizer import Tokenizer

def get_tokenizer() -> Tokenizer:
    if not PATHS.tok.exists():
        raise FileNotFoundError(f'[file no found]: the tokenizer was got by pretraining, please pretrain the tokenizer first')
    return Tokenizer.load(PATHS.tok)

def get_conversation(p: Path) -> list:
    if not p.exists():
        raise FileNotFoundError(f'[sft data]: {p} does not exist; pass --data with a jsonl file where every line is {{"messages": [...]}}')
    conversation = []
    with open(p, errors = 'replace', encoding = 'utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'[sft data]: {p}:{lineno} is not valid JSON: {e}')
            messages = record.get('messages')
            if not isinstance(messages, list) or not messages:
                raise ValueError(f'[sft data]: {p}:{lineno} has no non-empty "messages" list, keys are {sorted(record)}')
            conversation.append(messages)
    return conversation

def main() -> None:
    parser = argparse.ArgumentParser(description = 'supervised fine-tuning on top of a base model')
    parser.add_argument('--data', default = str(PATHS.sft_data), help = 'jsonl file, one {"messages": [...]} per line')
    parser.add_argument('--base', default = str(PATHS.pretrain_ckpt), help = 'base weights, either a pretraining checkpoint or an export_model.py artifact')
    parser.add_argument('--out', default = str(PATHS.sft_ckpt))
    args = parser.parse_args()

    device = get_device()
    configure_backends()
    dtype = amp_dtype(device)
    tokenizer = get_tokenizer()
    token_bytes = build_token_bytes(tokenizer)
    conversation = get_conversation(Path(args.data))
    if len(conversation) < 2:
        raise ValueError(f'[length error]: SFT needs at least 2 conversations (got {len(conversation)}): 1 for training and 1 for validation')
    train_conversation = conversation[:-1]
    val_conversation = conversation[-1:]
    train_ds = SFTDataset(train_conversation, tokenizer, MODEL.block_size)
    val_ds = SFTDataset(val_conversation, tokenizer, MODEL.block_size)
    train_loader = DataLoader(train_ds, batch_size = SFT.batch_size, shuffle = True, collate_fn=sft_collate)
    val_loader = DataLoader(val_ds, batch_size = SFT.batch_size, shuffle = False, collate_fn=sft_collate)
    state = load_for_sft(Path(args.out), Path(args.base), tokenizer.vocab_size, device, SFT.lr)
    cfg = state.cfg
    model = state.model
    optimizer = state.optimizer
    start_epoch = state.start_epoch
    glob_step = state.glob_step

    for epoch in range(start_epoch, start_epoch + SFT.epoches_per_run):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            lr = get_lr(glob_step, SFT.lr, SFT.warmup_steps, SFT.total_steps, SFT.min_lr_ratio)
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad()
            with autocast_ctx(device, dtype):
                logits = model(input_ids)
                B, T, V = logits.shape
                loss = F.cross_entropy(
                    logits.reshape(B * T, V),
                    labels.reshape(B * T),
                    ignore_index=IGNORE_INDEX,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), SFT.grad_clip)
            optimizer.step()
            glob_step += 1

            if glob_step % SFT.eval_every == 0:
                res = evaluate(model, val_loader, device, token_bytes)
                print(f'epoch {epoch:>4} | step {glob_step:>6} | lr {lr:.2e} | train loss = {loss.item():.4f} | val loss = {res.loss:.4f} | bpb = {res.bpb:.4f}')
        res = evaluate(model, val_loader, device, token_bytes)
        print(f'[epoch end] epoch {epoch:>4} | step {glob_step:>6}| train loss = {loss.item():.4f} | val loss = {res.loss:.4f} | bpb = {res.bpb:.4f}')
        
        save_checkpoint(model, cfg, optimizer, epoch + 1, glob_step, Path(args.out))

    return


if __name__ == '__main__':
    main()
        

