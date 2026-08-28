# mini-LLM

**A learning record: building a modern language model from scratch, one technique at a time.**

The goal was not to reimplement GPT-2 but to build the architecture as it is actually written today — RoPE, KV caching, RMSNorm, SwiGLU — and to understand each piece well enough to write it, test it, and prove it correct rather than merely plausible.

A 132M-parameter model in 1,790 lines of Python across ten modules, with three third-party dependencies (`torch`, `numpy`, `regex`). Apart from PyTorch's tensor ops, not a line comes from `transformers` or `tiktoken`. Pretrained for 7.4 hours on 5.75B tokens on a single H100.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/loss-curve-dark.png">
  <img alt="Cross-entropy loss over 10,967 optimizer steps, falling from 4.22 to 2.2933. Training and validation loss stay on top of each other for the whole run." src="docs/loss-curve-light.png">
</picture>

Final validation loss **2.2933**, **0.8523 bits per byte**. The two curves sit on top of each other the whole way, which is what a single epoch looks like: every token is seen exactly once, so there is nothing to memorize and no gap to open.

---

## The model

`model.py`, 197 lines. Decoder-only, 12 layers × 12 heads × 768 wide, 1024 context, 132.3M parameters.

**RoPE — rotary position embeddings** (`model.py:17-30`)
Position enters by rotating Q and K within 2D subspaces rather than adding a learned vector, so attention depends on *relative* distance. The hard part is not the rotation: the cache must be indexed by **absolute** position, `rope_cos[pos:pos + T]`, where `pos` comes from the KV cache. Get that offset wrong and there is no error — every token after the first is rotated as if it sat at position 0, and generation falls apart within a few tokens.

**KV cache** (`model.py:32-36, 100-116`)
Stores K and V per layer so each new token attends to the existing prefix instead of recomputing it. The subtle line is `is_causal = (Q.size(2) == K.size(2))`, which evaluates to **False** during cached decode. That looks like the causal mask has been switched off; it hasn't. With a single query at the end of the sequence, every cached key is already in its past, so masking is unnecessary and applying it would be wrong. A broken cache raises nothing and silently emits garbage you would blame on the model, so T16 pins the cached and uncached paths together logit for logit.

**RMSNorm · SwiGLU · pre-norm** (`model.py:58, 126-140`)
The Llama-era stack, not GPT-2's: normalize by root-mean-square with no mean subtraction and no bias; gate the feed-forward as `silu(gate(x)) * up(x)`; normalize *before* each sublayer so the residual stream stays a clean identity path from embedding to output.

**Weight tying and residual-scaled init** (`model.py:171-176`)
`lm_head` shares the embedding matrix — 18.9M parameters saved, and one vector means the same thing going in and coming out. Residual projections start at `0.02 / sqrt(2 * n_layer)` so summing 12 layers into one stream does not inflate its variance. Getting this wrong raises nothing either; training is just quietly less stable than it should be.

## The tokenizer

`tokenizer.py`, 341 lines. Byte-level BPE, 24,576 tokens = 256 bytes + 24,274 merges + 46 special tokens.

Start with a puzzle. BPE encoding needs to find which pairs can currently be merged, and these two lines produce exactly the same set:

```python
valid = [pair for pair in self.merges if pair in counts]   # A
valid = [pair for pair in counts if pair in self.merges]   # B
```

At production vocab size **B is roughly 200× faster**. `self.merges` holds 24,274 entries while `counts` is only as long as one word — about 10. Same intersection, three orders of magnitude less work, and the gap grows with the vocabulary: 9.4× at 500 merges, 28.6× at 3,000.

That is one of three steps that took encoding from 7 KB/s to 6.78 MB/s:

1. **Regex pre-split** — merges can never cross a word boundary. This is the step that improves *quality*, not just speed: without it `the` and ` quick` can fuse and the vocabulary fills with cross-word garbage.
2. **Iterate the word, not the merge table** — the puzzle above.
3. **Word-level encode cache** — 93% hit rate on real code, capped at 100k entries because code corpora never converge on a fixed vocabulary (roughly 14k new unique words per MB, mostly one-off identifiers and hashes).

| Implementation | Throughput | Compression |
| --- | --- | --- |
| mini-LLM (cold cache) | 5.77 MB/s | 3.81 chars/token |
| mini-LLM (warm cache) | **18.08 MB/s** | 3.81 |
| tiktoken `cl100k_base` (Rust) | 20.03 MB/s | 4.16 |

*2 MB of real Python source, Apple Silicon, single thread.* Pure Python reaching 90% of a Rust implementation, and **92% of `cl100k_base`'s compression from a quarter of its vocabulary** — which also means a much cheaper embedding and lm_head.

The more interesting question is the next one: **what makes me confident A and B are equivalent?** "It got faster" and "it got faster and wrong" look identical in a log. Every optimization here ships with a reference implementation too dumb to be wrong and a test asserting the two agree token for token. That caught three real bugs, including a cache that was read but never written — every result correct, every test green, and no speedup at all.

## Why start here instead of nanochat

[nanochat](https://github.com/karpathy/nanochat) is the better project: more complete, better optimized, written by someone who has done this many times. This one is a gentler on-ramp to the same understanding.

| | mini-LLM | nanochat |
| --- | --- | --- |
| Hardware for a full run | **1 GPU** | an 8×H100 node |
| Wall clock / cost | 7.4 h / ~$25 | ~1.5 h / ~$48 |
| Core modules | **10** | ~35 |
| Distributed training code | **none** | DDP / torchrun |
| Stages covered | tokenizer → pretrain → SFT → sampling | also midtraining, RL, eval |

Renting one H100 is something you can do on a whim; an 8-GPU node is a different kind of decision. More to the point, `pretrain.py` is 122 lines of ordinary PyTorch with no `torchrun` and no rank juggling, readable top to bottom without first learning distributed primitives.

## Quick start

```bash
uv sync
cd src/main

python3 test.py        # 26 consistency tests, no corpus needed
python3 prepare.py     # corpus -> tokenizer -> token shards
python3 pretrain.py    # pretrain
python3 train.py       # SFT
python3 sampling.py    # generate
```

The trained tokenizer ships with the repository (`data/tok.json`), so the throughput benchmark above reproduces without downloading any corpus.

```
src/main/
├── model.py       Transformer (RoPE / RMSNorm / SwiGLU / KV cache)
├── tokenizer.py   BPE tokenizer + chat template
├── dataset.py     domain mixer / shard read-write / SFT dataset
├── prepare.py     corpus -> tokenizer -> shards
├── pretrain.py    pretraining          train.py     SFT
├── sampling.py    sampling and chat    common.py    checkpoints / schedule / bpb
├── config.py      every hyperparameter test.py      26 consistency tests
└── tools/         corpus builder, loss plot
```

## Fine-tuning on the base model

`export_model.py` turns a training checkpoint into a distributable base model, dropping the AdamW state — two thirds of the file and useless for anything but resuming pretraining:

```bash
python3 export_model.py --ckpt data/pretrain_checkpoint.pt --out mini-llm-base.pt
python3 train.py --data your_data.jsonl --base mini-llm-base.pt --out sft.pt
```

One conversation per line, roles `system` / `user` / `assistant` / `tool`, only assistant spans supervised. Tool calls use the OpenAI `tool_calls` / `tool_call_id` shape; `data/sft_toy.jsonl` has a worked example.

**Know the limits before pointing it at real data.** The SFT loop is deliberately minimal: fp32, `batch_size = 2`, no gradient accumulation, checkpoints only at epoch end, and `SFTConfig` tuned for a 6-conversation toy file. And one that bites silently — **a conversation longer than 1024 tokens keeps only its tail**, so the system prompt and the original task scroll out of the window while training proceeds without complaint. A SWE-bench style agent trace runs 40k+ tokens, of which this model sees 2%.

## The training run

One epoch, 10,967 optimizer steps at 524,288 tokens each, on one H100 SXM.

| | |
| --- | --- |
| Corpus | 22 GB raw → 5,750,197,205 train tokens (116 shards) + 2,001,461 val |
| Throughput | 2.44 s/step, 215k tokens/s, ~20% MFU |
| Wall clock / cost | 7.4 hours / ~$25 |
| Final val loss / bpb | **2.2933** / **0.8523** |

Machinery the run needed: gradient accumulation (32 micro-batches into one 524,288-token step), bf16 autocast where CUDA supports it, a cosine schedule counted in optimizer steps, and mid-epoch resumable checkpoints — the unglamorous one, since a 7.4-hour run that can only checkpoint at the end is a run you lose to a single disconnection.

Validation loss fell at every one of the 54 evaluation points and tracked the training loss within 0.03 throughout. The curve fits a power law `L = 48.7·s^-0.596 + 2.12` with R² = 0.9978, and was still descending when the corpus ran out. At ~20% MFU the H100 was underused — most likely the missing `torch.compile` and the size of the logits tensor over a 24,576-entry vocabulary.

**The corpus mix missed its target.** Every domain ran out of documents before the token budget was reached, so `terminal_docs` landed at 0.12% against a 3% target and `qa_stackexchange` absorbed the slack at 16.60% against 15%. The model trained on that real mix, not the intended one. The validation split, being small, hits its target within 0.20pp.

## Tests

26 consistency tests, all passing. Each is named for a specific silent failure it guards against — KV cache divergence (T16), cross-word merges (T18), weighted-vs-naive BPE training (T19), shard-boundary reads (T22), train/inference template drift (T12). Reading them is a fast way to see where this kind of code actually goes wrong.

Three that were caught in practice: tokenizer training text taken from the head of the corpus, so `ROMEO` (163 occurrences in the full text, 0 in the head) got shredded into single letters; an encode cache that was read but never written; and a `zip` that silently truncated because `words` and `freqs` came from different sources.

## Where this stands

**Done** — tokenizer, corpus pipeline, Transformer, pretraining, SFT and sampling code, and one full pretraining run of the 132M model.

**Not done** — `main.py` and `agent.py` are empty; the tokenizer reserves `<|tool_call_start|>` and friends but there is no tool-calling runtime or chat CLI. No SFT run on the new base model yet, and no midtraining, RL, or evaluation harness. The base weights are not in this repository (529 MB).

**Next** — optimizer parameter grouping (AdamW defaults currently decay RMSNorm weights too), `torch.compile`, and a real SFT run.

## References and caveats

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) and [karpathy/nanochat](https://github.com/karpathy/nanochat) — the latter is the main reference for architecture choices and chat template design.
- Every performance number was measured on the hardware named beside it. The baseline behind "990×" is my own first naive implementation, not any mature library, and the "~200×" at 24,274 merges is extrapolated from measured points at 500 and 3,000.
- **bpb is only comparable on the same evaluation data.** The 0.8523 above is measured on this project's own validation split, which is 40% code and therefore more predictable than prose. It is not comparable to bpb published against WikiText or C4.
- The loss curve is plotted from 35 of the run's 54 evaluation points (`docs/train-partial.log`); markers sit on real evaluations, so the two marker-free stretches are exactly where points are missing.

MIT.
