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

# mini-LLM (132M)

A 132M-parameter decoder-only transformer trained from scratch — tokenizer, data pipeline, model, and training loop all written by hand, with no `transformers` and no `tiktoken`. Source: [FengYe476/mini-LLM](https://github.com/FengYe476/mini-LLM).

Two checkpoints are published here: a **base model** from pretraining, and an **instruction-tuned model** fine-tuned on a mix of general conversations and agent trajectories. Neither has had RLHF or any safety alignment.

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
| **Final val loss** | **2.2921** |
| **Final val bits-per-byte** | **0.8518** |

Validation loss fell at every one of the 54 evaluation points and stayed within 0.03 of the training loss throughout — a single epoch shows each token exactly once, so there is no overfitting signal to find.

## Fine-tuning (mini-llm-sft.pt)

Two epochs over 122,012 conversations, 15,252 steps at batch 16.

| | |
| --- | --- |
| General instructions | ~74,000 conversations from UltraChat-200k that fit the 1024-token window |
| **Agent trajectories** | **47,891 windowed samples** |
| **Final val loss** | **0.9610** |
| **Final val bits-per-byte** | **0.5637** |

The agent half is the interesting part. It comes from 1,808 SWE-bench and Terminal-Bench trials, of which 1,329 scored `reward=1` and were kept. Those trajectories run 40,861 tokens at the median against a 1024-token context, and their agent system prompt alone is 1,042 tokens — so the prompt is replaced with a one-line stand-in and each assistant turn becomes its own sample carrying the task description plus the last six messages. The result averages 1,007 tokens per sample at 48% supervised. `tools/build_agent_sft.py` in the repository does this.

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
- **The base model does not follow instructions.** Prompt it with a prefix to continue. The SFT model holds a conversation, but at 132M it is fluent rather than accurate.
- **No safety tuning of any kind.** It will reproduce whatever patterns are in the corpus, including undesirable ones.
- **bpb is not comparable across datasets.** The figures above are measured on this project's own validation split, which is 40% code and therefore more predictable than prose. It is not comparable to bpb published against WikiText or C4.
- **Undertrained relative to modern practice**, though over-trained relative to Chinchilla-optimal (5.75B tokens for 132M parameters is ~43× the parameter count).

## Downloads

| File | What it is |
| --- | --- |
| `mini-llm-base.pt` | base model, first pretraining run |
| `mini-llm-base-run2.pt` | base model, second run (val loss 2.2921) |
| `mini-llm-sft.pt` | instruction-tuned on general + agent data |
| `logs/` | full training logs for every run |

Each file is 529 MB (fp32, optimizer state stripped).

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

from huggingface_hub import hf_hub_download
path = hf_hub_download('ye476/mini-llm-132m', 'mini-llm-sft.pt')   # or 'mini-llm-base.pt'

payload = torch.load(path, weights_only=True, map_location='cpu')
cfg = GPTConfig(**payload['cfg'])
model = NanoGPT(cfg)
model.load_state_dict(payload['model'])
model.eval()

tokenizer = Tokenizer.load('data/tok.json')
# base model: continue a prefix
print(generate(model, cfg, tokenizer, 'def fibonacci(n):', max_new_tokens=100, temperature=0.8, top_k=40))

# sft model: hold a conversation
from sampling import chat
print(chat(model, cfg, tokenizer, [{'role': 'user', 'content': 'list the files'}], max_new_tokens=60))
```

### What to expect

At 132M parameters the model is fluent but not knowledgeable. Asked "what is python" it answers in well-formed English about the wrong language. Asked to list files it correctly produces `ls` in a code block with an explanation — the agent trajectories left a visible mark on its terminal behaviour.

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

Read the fine-tuning caveats in the repository README before pointing this at a real dataset. In particular: the SFT loop is deliberately minimal (no gradient accumulation, checkpoints only at epoch end, and `SFTConfig` needs its step counts matched to your dataset size), and conversations longer than 1024 tokens are truncated to their **tail**, silently dropping everything earlier. `tools/build_agent_sft.py` shows one way to work around that for long trajectories.

## License

MIT.
