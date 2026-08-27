---
license: mit
language:
  - en
library_name: pytorch
pipeline_tag: text-generation
tags:
  - from-scratch
  - gpt
  - rope
  - swiglu
  - rmsnorm
---

# mini-LLM base (132M)

A 132M-parameter decoder-only transformer trained from scratch — tokenizer, data pipeline, model, and training loop all written by hand, with no `transformers` and no `tiktoken`. Source: [FengYe476/mini-LLM](https://github.com/FengYe476/mini-LLM).

This is a **base model**. It has had no instruction tuning, no RLHF, and no safety alignment. It continues text; it does not follow instructions or hold a conversation.

## Architecture

| | |
| --- | --- |
| Parameters | 132.3 M (113.4 M non-embedding) |
| Layers / heads / width | 12 / 12 / 768 |
| Context length | **1024** |
| Vocabulary | 24,576 (256 bytes + 24,274 BPE merges + 46 special tokens) |
| Position encoding | RoPE (θ = 10000) |
| Normalization | RMSNorm, pre-norm |
| Feed-forward | SwiGLU, 4× expansion |
| Embedding | lm_head tied to token embedding |

## Training

One epoch over 5.75B tokens on a single H100 SXM.

| | |
| --- | --- |
| Tokens seen | 5,749,010,432 |
| Optimizer steps | 10,967 at 524,288 tokens each |
| Optimizer | AdamW, lr 1e-3, cosine to 1e-4, 114 warmup steps, grad clip 1.0 |
| Precision | bf16 autocast, fp32 master weights |
| Wall clock | 7.4 hours (~20% MFU) |
| **Final val loss** | **2.2933** |
| **Final val bits-per-byte** | **0.8523** |

Validation loss fell at every one of the 54 evaluation points and stayed within 0.03 of the training loss throughout — a single epoch shows each token exactly once, so there is no overfitting signal to find.

## Training data

A 22 GB corpus mixed from eight domains, tokenized into 5.75B tokens.

| Domain | Target | Actual |
| --- | --- | --- |
| web_edu | 20% | 20.18% |
| code_python | 20% | 20.33% |
| qa_stackexchange | 15% | 16.60% |
| web_dclm | 15% | 15.63% |
| cosmopedia | 12% | 10.64% |
| code_issues | 10% | 10.57% |
| code_shell | 5% | 5.93% |
| terminal_docs | 3% | **0.12%** |

Every domain ran out of documents before the token budget was reached, so the actual mix drifts from the target. `terminal_docs` is 2.88 percentage points short and `qa_stackexchange` absorbed most of the slack. **The model was trained on the actual mix, not the target one.**

Roughly 40% of the corpus is code. Expect the model to be noticeably stronger on code and shell than on prose.

## Limitations

- **Context is 1024 tokens.** Short by modern standards, and a hard limit — RoPE was baked at this length and the model has never seen a longer sequence.
- **English and code only.** Other languages appear only incidentally.
- **No instruction following.** It is a base model. Prompt it with a prefix to continue, not a question to answer.
- **No safety tuning of any kind.** It will reproduce whatever patterns are in the corpus, including undesirable ones.
- **bpb is not comparable across datasets.** The 0.8523 above is measured on this project's own validation split, which is 40% code and therefore more predictable than prose. It is not comparable to bpb published against WikiText or C4.
- **Undertrained relative to modern practice**, though over-trained relative to Chinchilla-optimal (5.75B tokens for 132M parameters is ~43× the parameter count).

## Usage

The weights are a plain PyTorch state dict, not a `transformers` model. Clone the repository for the model definition:

```bash
git clone https://github.com/FengYe476/mini-LLM.git
cd mini-LLM/src/main
```

Then load and sample:

```python
import torch
from model import NanoGPT, GPTConfig
from tokenizer import Tokenizer
from sampling import generate

payload = torch.load('mini-llm-base.pt', weights_only=True, map_location='cpu')
cfg = GPTConfig(**payload['cfg'])
model = NanoGPT(cfg)
model.load_state_dict(payload['model'])
model.eval()

tokenizer = Tokenizer.load('data/tok.json')
print(generate(model, cfg, tokenizer, 'def fibonacci(n):', max_new_tokens=100, temperature=0.8, top_k=40))
```

`data/tok.json` ships with the repository and **must** match this checkpoint — the payload carries a `tokenizer` fingerprint field for exactly that check.

## Fine-tuning

`train.py` fine-tunes this model on your own conversations:

```bash
python3 train.py --data your_data.jsonl --base mini-llm-base.pt --out sft.pt
```

Each line of `your_data.jsonl` is one conversation:

```json
{"messages": [{"role": "user", "content": "what is python"}, {"role": "assistant", "content": "a programming language"}]}
```

Supported roles are `system`, `user`, `assistant`, and `tool`. Only assistant spans are supervised; everything else is masked out of the loss. Tool calls are supported — the tokenizer reserves `<|tool_call_start|>`, `<|tool_result_start|>`, and FIM tokens — with the same `tool_calls` / `tool_call_id` shape the OpenAI API uses.

Read the fine-tuning caveats in the repository README before pointing this at a real dataset. In particular: the SFT loop is deliberately minimal (fp32, no gradient accumulation, checkpoints only at epoch end), and conversations longer than 1024 tokens are truncated to their **tail**, silently dropping everything earlier.

## License

MIT.
