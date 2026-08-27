import argparse
import torch

from pathlib import Path

from config import PATHS
from common import tokenizer_fingerprint
from model import GPTConfig
from tokenizer import Tokenizer

EXPORT_FORMAT = 1
DTYPES = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}


def build_payload(checkpoint: dict, dtype: torch.dtype, tok_path: Path) -> dict:
    if 'model' not in checkpoint or 'cfg' not in checkpoint:
        raise KeyError(f'[export]: checkpoint is missing "model" or "cfg"; keys are {sorted(checkpoint)}')
    cfg = GPTConfig(**checkpoint['cfg'])
    tokenizer = Tokenizer.load(tok_path)
    if cfg.vocab_size != tokenizer.vocab_size:
        raise ValueError(f'[export]: checkpoint vocab ({cfg.vocab_size}) does not match {tok_path} ({tokenizer.vocab_size}); exporting them together would ship a model whose token ids mean nothing')
    weights = {k: v.to(dtype) for k, v in checkpoint['model'].items()}
    return {
        'export_format': EXPORT_FORMAT,
        'model': weights,
        'cfg': checkpoint['cfg'],
        'epoch': checkpoint.get('epoch', 0),
        'glob_step': checkpoint.get('glob_step', 0),
        'dtype': str(dtype).replace('torch.', ''),
        'tokenizer': tokenizer_fingerprint(tok_path),
        'vocab_size': tokenizer.vocab_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description = 'strip optimizer state from a training checkpoint and write a distributable base model')
    parser.add_argument('--ckpt', default = str(PATHS.pretrain_ckpt))
    parser.add_argument('--tok', default = str(PATHS.tok))
    parser.add_argument('--out', default = 'data/mini-llm-base.pt')
    parser.add_argument('--dtype', choices = sorted(DTYPES), default = 'fp32')
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    tok_path = Path(args.tok)
    out_path = Path(args.out)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'[export]: no checkpoint at {ckpt_path}')
    if not tok_path.exists():
        raise FileNotFoundError(f'[export]: no tokenizer at {tok_path}')

    checkpoint = torch.load(ckpt_path, weights_only = True, map_location = 'cpu')
    payload = build_payload(checkpoint, DTYPES[args.dtype], tok_path)

    out_path.parent.mkdir(parents = True, exist_ok = True)
    temp = out_path.with_name(out_path.name + '.tmp')
    torch.save(payload, temp)
    temp.replace(out_path)

    params = sum(v.numel() for v in payload['model'].values())
    before = ckpt_path.stat().st_size
    after = out_path.stat().st_size
    print(f'[export]: {ckpt_path} -> {out_path}')
    print(f'[export]: epoch {payload["epoch"]} | step {payload["glob_step"]} | vocab {payload["vocab_size"]} | tokenizer {payload["tokenizer"]}')
    print(f'[export]: {params} parameters as {payload["dtype"]}')
    print(f'[export]: {before / 1e6:.0f} MB -> {after / 1e6:.0f} MB ({(1 - after / before) * 100:.0f}% smaller, optimizer state dropped)')
    return


if __name__ == '__main__':
    main()
