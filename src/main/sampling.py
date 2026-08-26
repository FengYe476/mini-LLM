import torch
import torch.nn.functional as F

from typing import Any

from config import PATHS
from common import load_model, get_device
from tokenizer import Tokenizer
from model import NanoGPT, GPTConfig, KVCache


def _to_tensor(model: NanoGPT, ids: list[int]) -> torch.Tensor:
    device = next(model.parameters()).device
    return torch.tensor(ids, dtype = torch.long).view(1, -1).to(device)

def _sample_loop(model: NanoGPT, cfg: GPTConfig, token_ids: torch.Tensor, max_new_tokens: int, temperature: float, top_k: int|None, stop_ids: set[int]|None, use_cache: bool = True) -> list[int]:
    if use_cache and token_ids.size(1) + max_new_tokens > cfg.block_size:
        raise ValueError(f'[cache error]:prompt({token_ids.size(1)}) + max_new_tokens({max_new_tokens}) exceeds block_size({cfg.block_size}); shorten the prompt or pass use_cache = False')
    if temperature <= 0:
        raise ValueError(f'[temp errors]: temperature must be > 0, got {temperature}')
    if max_new_tokens <= 0:
        raise ValueError(f'[token errors]: max new token must be > 0, got {max_new_tokens}')
    if top_k is not None and top_k <= 0:
        raise ValueError(f'[topk errors]: top k must be > 0, got {top_k}')
    new_ids = []
    was_training = model.training
    model.eval()
    cache = KVCache(cfg.n_layer) if use_cache else None
    step_input = token_ids
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if use_cache:
                    logits = model(step_input, cache)
                else:    
                    logits = model(token_ids[:, -cfg.block_size:])
                logits = logits[:, -1, :]
                logits = logits/temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                probs = F.softmax(
                    logits,
                    dim = -1
                )
                next_ids = torch.multinomial(
                    probs,
                    num_samples=1
                )
                if stop_ids is not None and next_ids.item() in stop_ids:
                    break
                if use_cache:
                    step_input = next_ids
                else:
                    token_ids = torch.cat([token_ids, next_ids], dim = -1)
                new_ids.append(next_ids.item())
    finally:
        model.train(was_training)
    return new_ids

def generate(model: NanoGPT, cfg: GPTConfig, tokenizer: Tokenizer, prompt: str, max_new_tokens: int, temperature: float = 1.0, top_k: int|None = None, stop_ids: set[int]|None = None, use_cache: bool = True) -> str:
    ids = tokenizer.encode(prompt)
    ids = ids[-(cfg.block_size - max_new_tokens):]
    token_ids = _to_tensor(model, ids)
    new_ids = _sample_loop(model, cfg, token_ids, max_new_tokens, temperature, top_k, stop_ids, use_cache)
    return tokenizer.decode(new_ids)

def chat(model: NanoGPT, cfg: GPTConfig, tokenizer: Tokenizer, messages: list[dict[str, Any]], max_new_tokens: int, temperature: float = 0.8, top_k: int|None = 40, use_cache: bool = True) -> str:
    ids = tokenizer.render_for_generation(messages, cfg.block_size - max_new_tokens)
    stop_ids = {tokenizer.encode_special_token('<|assistant_end|>')}
    token_ids = _to_tensor(model, ids)
    new_ids = _sample_loop(model, cfg, token_ids, max_new_tokens, temperature, top_k, stop_ids, use_cache)
    return tokenizer.decode(new_ids)

def main() -> None:
    device = get_device()
    tokenizer = Tokenizer.load(PATHS.tok)

    model, cfg = load_model(PATHS.pretrain_ckpt, tokenizer.vocab_size, device)
    prompt = 'QUEEN ELIZABETH:\n'
    for temperature, top_k in [(0.8, None), (0.8, 40), (0.2, 40)]:
        print(f'\n=== base | temperature={temperature} | top_k={top_k} ===')
        print(generate(model, cfg, tokenizer, prompt, max_new_tokens = 200,
                       temperature = temperature, top_k = top_k))

    if not PATHS.sft_ckpt.exists():
        print(f'\n[skip chat]: no sft checkpoint at {PATHS.sft_ckpt}, run train.py first')
        return

    sft_model, sft_cfg = load_model(PATHS.sft_ckpt, tokenizer.vocab_size, device)
    messages = [{'role': 'user', 'content': 'list the files'}]
    print(f'\n=== sft | chat ===')
    answer = chat(sft_model, sft_cfg, tokenizer, messages, max_new_tokens = 100)
    print(f'answer  : {answer!r}')
    print(f'token数 : {len(tokenizer.encode(answer))} (小于100 = 自己吐了 assistant_end 停住了)')
    return

if __name__ == '__main__':
    main()            