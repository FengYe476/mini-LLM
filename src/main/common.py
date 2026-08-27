import os
import torch
import torch.nn.functional as F
import math
import hashlib

from pathlib import Path
from contextlib import nullcontext
from dataclasses import dataclass, asdict

from model import NanoGPT, GPTConfig
from dataset import IGNORE_INDEX
from config import MODEL
from tokenizer import Tokenizer, SPECIAL_TOKENS

@dataclass
class TrainState:
    model: NanoGPT
    optimizer: torch.optim.AdamW
    cfg: GPTConfig
    start_epoch: int
    glob_step: int
    samples_seen: int = 0

@dataclass(frozen = True)
class EvalResult:
    loss: float
    bpb: float
    n_tokens: int
    n_bytes: float

def tokenizer_fingerprint(tok_path: Path) -> str:
    return hashlib.sha256(tok_path.read_bytes()).hexdigest()[:16]

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def configure_backends() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return

def amp_dtype(device: torch.device) -> torch.dtype|None:
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None

def autocast_ctx(device: torch.device, dtype: torch.dtype|None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type = device.type, dtype = dtype)

def build_gpt_config(vocab_size: int) -> GPTConfig:
    return GPTConfig(vocab_size = vocab_size, **asdict(MODEL))

def save_checkpoint(model: NanoGPT, cfg: GPTConfig, optimizer: torch.optim.AdamW, epoch: int, glob_step: int, path: Path, samples_seen: int = 0) -> None:
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'cfg': asdict(cfg),
        'epoch': epoch,
        'glob_step': glob_step,
        'samples_seen': samples_seen
    }
    path.parent.mkdir(parents = True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    torch.save(
        payload,
        temp
    )
    os.replace(temp, path)
    temp.unlink(missing_ok=True)
    return

def build_token_bytes(tokenizer: Tokenizer) -> torch.Tensor:
    table = torch.zeros(tokenizer.vocab_size, dtype = torch.int64)
    for token_id in range(tokenizer.vocab_size):
        if token_id in tokenizer.id_to_special:
            continue
        piece = tokenizer.vocab.get(token_id)
        if piece is None:
            raise KeyError(f'[token bytes]: the token id {token_id} is neither a special token nor in vocab (vocab size = {tokenizer.vocab_size})')
        table[token_id] = len(piece)
    return table


def evaluate(model: NanoGPT, val_loader, device: torch.device, token_bytes: torch.Tensor|None = None) -> EvalResult:
    model.eval()
    loss_sum = 0
    token_counts = 0
    bytes_counts = 0
    if token_bytes is not None:
        token_bytes = token_bytes.to(device)
    dtype = amp_dtype(device)
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            with autocast_ctx(device, dtype):
                logits = model(input_ids)
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V).float(),
                labels.reshape(B * T),
                ignore_index= IGNORE_INDEX,
                reduction = 'sum'
            ).item()
            loss_sum += loss
            token_counts += (batch['labels'] != IGNORE_INDEX).sum().item()
            if token_bytes is not None:
                valid = labels[labels != IGNORE_INDEX]
                bytes_counts += token_bytes[valid].sum().item()
    model.train()
    loss_mean = loss_sum / token_counts
    bpb = (loss_sum / math.log(2) / bytes_counts if bytes_counts > 0 else float('nan'))
    return EvalResult(
        loss = loss_mean,
        bpb = bpb,
        n_tokens = token_counts,
        n_bytes = bytes_counts
    )

def load_or_init(ckpt_path: Path, vocab_size: int, device: torch.device, lr: float) -> TrainState:
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, weights_only=True, map_location='cpu')
        cfg = GPTConfig(**checkpoint['cfg'])
        assert cfg.vocab_size == vocab_size, f'[vocab invalid]: the vocab size between config({cfg.vocab_size}) and tokenizer({vocab_size}) is different'
        want = build_gpt_config(vocab_size)
        if cfg != want:
            diff = {k: f'{v} -> {getattr(want, k)}' for k, v in asdict(cfg).items() if v != getattr(want, k)}
            raise ValueError(f'[model changed]: {ckpt_path} was trained with a different architecture {diff}; delete {ckpt_path} and pretrain from scratch')
        model = NanoGPT(cfg = cfg)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        glob_step = checkpoint['glob_step']
        samples_seen = checkpoint.get('samples_seen', 0)
        print(f'[resume]: epoch {start_epoch}/ step {glob_step} / {samples_seen} samples into the epoch')
    else:
        cfg = build_gpt_config(vocab_size)
        model = NanoGPT(cfg = cfg)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
        start_epoch = 0
        glob_step = 0
        samples_seen = 0
        print(f'[new]: epoch {start_epoch}/ step {glob_step}')

    return TrainState(
        model = model,
        cfg = cfg,
        optimizer=optimizer,
        start_epoch = start_epoch,
        glob_step=glob_step,
        samples_seen=samples_seen
    )

def load_model(ckpt_path: Path, vocab_size: int, device: torch.device) -> tuple[NanoGPT, GPTConfig]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f'[file no found]: cannot found the checkpoint in {ckpt_path}')
    checkpoint = torch.load(ckpt_path, weights_only=True, map_location='cpu')
    cfg = GPTConfig(**checkpoint['cfg'])
    assert cfg.vocab_size == vocab_size, f'[vocab size]: the vocab size between config({cfg.vocab_size}) and tokenizer({vocab_size}) is differnt'
    model = NanoGPT(cfg)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    model.eval()
    print(f'[evaluation]: epoch {checkpoint["epoch"]} | step {checkpoint["glob_step"]}')
    return model, cfg


def load_for_sft(sft_ckpt_path: Path, base_ckpt_path: Path, vocab_size: int, device: torch.device, lr: float) -> TrainState:
    if sft_ckpt_path.exists():
        checkpoint = torch.load(sft_ckpt_path, weights_only=True, map_location='cpu')
        cfg = GPTConfig(**checkpoint['cfg'])
        assert cfg.vocab_size == vocab_size, f'[vocab invalid]: the vocab size between config ({cfg.vocab_size}) and tokenizer ({vocab_size}) is different'
        model = NanoGPT(cfg)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        glob_step = checkpoint['glob_step']
        print(f'[resume]: epoch {start_epoch} | step {glob_step}')

    elif base_ckpt_path.exists():
        checkpoint = torch.load(base_ckpt_path, weights_only=True, map_location='cpu')
        cfg = GPTConfig(**checkpoint['cfg'])
        assert cfg.vocab_size == vocab_size, f'[vocab invalid]: the vocab size between config ({cfg.vocab_size}) and tokenizer ({vocab_size}) is different'
        model = NanoGPT(cfg)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
        start_epoch = 0
        glob_step = 0
        print(f'[sft from base]: base epoch: {checkpoint["epoch"]} | base step {checkpoint["glob_step"]}')
    else:
        raise FileNotFoundError(f'[file not found]: SFT must train from base, please pretrain at first and then give {base_ckpt_path}')

    return TrainState(
        model = model,
        cfg = cfg,
        optimizer = optimizer,
        start_epoch=start_epoch,
        glob_step=glob_step
    )

def get_lr(step: int, base_lr: float, warmup_step: int, total_steps: int, min_lr_ratio: float) -> float:
    if total_steps <= warmup_step:
        raise ValueError(f'[step errors]: the total step({total_steps}) is less than warmup_steps({warmup_step})')
    
    min_lr = base_lr * min_lr_ratio
    if step < warmup_step:
        return base_lr * (step + 1) / warmup_step
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_step) / (total_steps - warmup_step)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (base_lr - min_lr)
        
    






