# mini-LLM

**A complete LLM built from scratch, small enough to read in an afternoon and cheap enough to actually run.**

Tokenizer, corpus pipeline, Transformer, pretraining, SFT, sampling — 1,790 lines of Python across ten modules, with three third-party dependencies (`torch`, `numpy`, `regex`). Apart from PyTorch's tensor ops, not a line comes from `transformers` or `tiktoken`. The 132M base model in here was trained end to end on one GPU for 7.4 hours.

This is a **learning repository**. It is for someone who has read how transformers work and now wants to watch every piece get built: how bytes become tokens, how 22 GB of text becomes a training stream, why attention needs a mask, what a KV cache actually caches, and what it feels like when 5.75 billion tokens go through a model you wrote yourself.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/loss-curve-dark.png">
  <img alt="Cross-entropy loss over 10,967 optimizer steps, falling from 4.22 to 2.2933. Training and validation loss stay on top of each other for the whole run." src="docs/loss-curve-light.png">
</picture>

That is the real run: 5.75B tokens, 7.4 hours, one GPU, final validation loss 2.2933 and 0.8523 bits per byte. The two curves sit on top of each other the whole way, which is what a single epoch looks like — every token is seen exactly once, so there is nothing to memorize and no gap to open. The orange jitter is not instability; it is one micro-batch's loss against the blue line's full validation split. Regenerate the plot from any log with `tools/plot_loss.py`.

## Why start here instead of nanochat

[nanochat](https://github.com/karpathy/nanochat) is the better project. It is more complete, better optimized, and written by someone who has done this many times. If you are comfortable reading it, read it.

This repository is a gentler on-ramp to the same understanding:

| | mini-LLM | nanochat |
| --- | --- | --- |
| Hardware for a full run | **1 GPU** | an 8×H100 node |
| Wall clock / cost | 7.4 h / ~$25 | ~1.5 h / ~$48 (~$15 spot) |
| Core modules | **10** | ~35 |
| Distributed training code | **none** | DDP / torchrun |
| Stages covered | tokenizer → pretrain → SFT → sampling | also midtraining, RL, eval harness |

Two of those rows matter more than the rest.

**One GPU, not eight.** Renting a single H100 is something you can do on a whim; an 8-GPU node is a different kind of decision. More importantly, the training loop has no `torchrun`, no DDP, no rank juggling — `pretrain.py` is 122 lines of ordinary PyTorch you can read top to bottom without first learning distributed primitives.

**Ten modules, not thirty-five.** nanochat earns its size by covering RL and a real evaluation harness. That is more to hold in your head at once. Here the whole pipeline fits in `model.py` (197 lines), `tokenizer.py` (341), `dataset.py` (366), `common.py` (234), and `pretrain.py` (122).

nanochat's own answer for laptops is to shrink the model until it fits — "you will not get strong results in this way," as its README puts it. That is the right call for its scale. Here the same code that runs a smoke test on a MacBook's MPS backend is the code that ran the real 5.75B-token job, with a single `bf16` branch between them.

## What makes it a teaching repository

**Everything is hand-written and short enough to modify.** The BPE trainer is a readable merge loop, not a binding to a fast library. When you want to know what changes if the regex pre-split is wrong, you can just break it and look.

**The engineering decisions are shown, not assumed.** The tokenizer went from 7 KB/s to 6.78 MB/s in three steps; the write-up below walks through each one and, more usefully, through how each was proven not to have broken anything.

**The failure modes are documented as tests.** Almost nothing on this path fails loudly — break the KV cache and it silently generates garbage you will misdiagnose as "the model is dumb"; read one token short at a shard boundary and the tail of every shard quietly disappears; swap the tokenizer but keep the old shards and every token id means something different, with no error anywhere. There are 26 tests, each named for the specific silent failure it guards, and reading them is a fast way to learn where this kind of code actually goes wrong. All 26 pass.

**There is one real run, honestly reported.** Not a projection — an actual 7.4-hour job with its throughput, its final bits-per-byte, the corpus mix it really trained on (which missed its target), and the 20% MFU that says the GPU was underused.

---

## Start with a two-line puzzle

BPE encoding needs to find which pairs can currently be merged. These two lines produce exactly the same set:

```python
valid = [pair for pair in self.merges if pair in counts]   # A
valid = [pair for pair in counts if pair in self.merges]   # B
```

At production vocab size, **B is roughly 200x faster than A**.

`self.merges` holds 24274 entries, while `counts` is only as long as one word — about 10 entries. Both compute the same intersection; the amount of work differs by three orders of magnitude. And the gap scales linearly with vocab size: 9.4x at 500 merges, 28.6x at 3000, roughly 200x at 24274.

This is one of three steps that made encoding 990x faster (the others are regex pre-splitting and a word-level cache).

But the more important question is the next one: **what makes me confident A and B are actually equivalent?** "It got faster" and "it got faster and wrong" look identical in the logs. Every optimization here ships with a reference implementation that is too dumb to be wrong, and a test asserting the two agree token for token — see the testing section below.

## Measured

**Tokenizer throughput** (2 MB of real Python source, Apple Silicon, single thread)

| Implementation | Throughput | Compression |
| --- | --- | --- |
| mini-LLM (cold cache) | 5.77 MB/s | 3.81 chars/token |
| mini-LLM (warm cache) | **18.08 MB/s** | 3.81 |
| tiktoken `cl100k_base` (**Rust**) | 20.03 MB/s | 4.16 |

Pure Python reaching 90% of a Rust implementation. The vocabulary is 4x smaller (24.5K vs 100K) for only 8% worse compression — and a 4x smaller vocabulary means a much cheaper embedding and lm_head.

**Corpus** (8-domain mix, full report in `src/main/prepare.log`)

| | |
| --- | --- |
| Raw text | 22 GB |
| Train tokens | 5,750,197,205 (116 shards) |
| Val tokens | 2,001,461 (1 shard) |
| Mix deviation, val | ≤ 0.20 percentage points |
| Mix deviation, train | ≤ 2.88 percentage points (see below) |
| Build time | 5.5 hours |

Target mix: `web_edu` 20% / `code_python` 20% / `web_dclm` 15% / `qa_stackexchange` 15% / `cosmopedia` 12% / `code_issues` 10% / `code_shell` 5% / `terminal_docs` 3%

The validation split hits that target within 0.20pp. The training split does not: every domain ran out of documents before reaching the 6B-token budget, so the mixer had to keep drawing from whatever was left. `terminal_docs` is the casualty — 0.12% actual against a 3% target, a 2.88pp shortfall — and `qa_stackexchange` absorbed most of the slack at 16.60% against 15%. The run below trains on that real mix, not the target one. Fixing it means collecting more `terminal_docs` and rebuilding the shards.

**Model**

| | |
| --- | --- |
| Parameters | 132.3 M (113.4 M non-embedding) |
| Layers / heads / width | 12 / 12 / 768 |
| Context | 1024 |
| Vocabulary | 24576 = 256 bytes + 24274 merges + 46 special tokens |

## Quick start

```bash
uv sync
cd src/main

# run every consistency test (no corpus needed, a few minutes)
python3 test.py

# build the token shards (needs the 8 domain jsonl files under data/corpus/)
python3 prepare.py

# pretrain -> SFT -> sample
python3 pretrain.py
python3 train.py
python3 sampling.py
```

The trained tokenizer ships with the repository (`src/main/data/tok.json`), so the throughput benchmark above reproduces without downloading any corpus.

## Using the base model

The pretrained weights are published separately (they are 529 MB, over what belongs in git). `export_model.py` is what produces them — it strips the AdamW state out of a training checkpoint, which is two thirds of its size and useless for anything but resuming pretraining:

```bash
python3 export_model.py --ckpt data/pretrain_checkpoint.pt --out mini-llm-base.pt
```

The exported payload carries the config, the vocab size, and a sha256 fingerprint of the tokenizer it was trained with, so a mismatched `tok.json` is refused rather than silently producing garbage.

To fine-tune on your own conversations:

```bash
python3 train.py --data your_data.jsonl --base mini-llm-base.pt --out sft.pt
```

One conversation per line:

```json
{"messages": [{"role": "user", "content": "what is python"}, {"role": "assistant", "content": "a programming language"}]}
```

Roles are `system`, `user`, `assistant`, `tool`. Only assistant spans are supervised. Tool calls use the same `tool_calls` / `tool_call_id` shape as the OpenAI API; see `data/sft_toy.jsonl` for a worked example including a `run_bash` call.

**Know these limits before pointing it at a real dataset.** The SFT loop is deliberately minimal and has none of the machinery `pretrain.py` grew:

- fp32 only — no bf16 autocast, so it is several times slower than it needs to be on an A100 or H100
- `batch_size = 2` and no gradient accumulation
- checkpoints written only at epoch end
- `SFTConfig` is tuned for the 6-conversation toy file; `total_steps` and `epoches_per_run` need changing for real data

And one that bites silently: **a conversation longer than 1024 tokens is truncated to its tail.** `render_conversation` keeps the bos token and the last 1023, so the system prompt and the original task scroll out of the window while training proceeds without complaint. Agent trajectories are the common case here — a SWE-bench style trace runs 40k+ tokens, of which this model can see 2%.

## A reading order

If you are here to learn, read the code in the order the data moves through it. Each step is small enough to finish in one sitting, and each has tests you can break on purpose to see what they catch.

1. **`tokenizer.py`** — start at `train()` and `_encode_word()`. Bytes become tokens here, and it is the only stage with no neural network in it, so nothing is hidden behind a matrix multiply. Break the regex in `SPLIT_PATTERN` and watch T18 fail.
2. **`dataset.py`** — `PretrainDataset` first (18 lines, and it shows what a training example *is*: a window of tokens, and the same window shifted by one), then `ShardedPretrainDataset` for how that works when the data does not fit in memory.
3. **`model.py`** — 197 lines, bottom-up: `MLP` → `MultiAttention` → `Block` → `NanoGPT`. The whole transformer is here, including RoPE and the KV cache.
4. **`pretrain.py`** — 122 lines. The training loop: forward, loss, backward, clip, step. Everything else in the file is bookkeeping.
5. **`sampling.py`** — how a trained model produces text, and why the KV cache makes it fast.
6. **`test.py`** — read this last, as a list of the mistakes the code above is defending against.

`common.py` and `config.py` are plumbing; read them when something in the list above references them.

The fastest way to build intuition is step 1 combined with step 6: run `python3 test.py`, then deliberately break something in `tokenizer.py`, and see which test turns red and what it says.

## Pipeline

```
prepare.py    text -> BPE tokenizer -> uint16 token shards
              |-- DomainMixer      deterministic 8-domain proportional scheduling
              |-- ShardWriter      streaming writes (6B tokens will not fit in memory)
              +-- fingerprinting   shards are sha256-bound to the tokenizer; a mismatch refuses to start

pretrain.py   shards -> base model
              |-- memmap random access, windows read across shard boundaries
              |-- cosine + warmup, gradient clipping, resumable checkpoints
              +-- bpb evaluation (vocab-independent, still comparable after a tokenizer change)

train.py      base -> SFT (chat template + loss mask, only assistant spans are supervised)

sampling.py   temperature / top-k / KV cache / stop tokens
```

## Model

`src/main/model.py`. This is a Llama / nanochat era configuration, not vanilla GPT-2:

- **RoPE** rotary position embeddings (`build_rope_cache` / `rotate_half` / `apply_rope`), with correct position offsets under KV caching
- **RMSNorm** with **pre-norm** residuals
- **SwiGLU** feed-forward (gate / up / down)
- **Weight tying** between lm_head and the token embedding
- **Residual-scaled init**: `0.02 / sqrt(2 * n_layer)`
- **SDPA** attention plus a hand-written **KV cache**

## Testing: 26 silent failures

This is the substance of the repository. Every test carries a comment naming the kind of "does not raise, but quietly ruins training" failure it guards against. Several of them are traps that were actually sprung:

| Test | Guards against |
| --- | --- |
| T16 | KV cache must match the naive path logit for logit |
| T18 | Regex pre-split: no cross-word tokens in the vocab, encoding must equal per-word encoding concatenated |
| T19 | Frequency-weighted training must match naive training **merge for merge**, order included |
| T20 | Encode cache: cold equals warm, never persisted, retraining must raise |
| T21 | Tokenizer corpus must be sampled across the whole text (**the ROMEO lesson**, below) |
| T22 | Sharded memmap must match the in-memory version window for window, across shard boundaries |
| T12 | The training template must be a strict prefix of the inference template |
| T14 | "Guard roll-call": statically check that defensive code which only matters later is still on duty |

Three real ones:

- **The ROMEO lesson.** Tokenizer training text was taken from the head of the corpus. `ROMEO` appears 163 times in the full text of tinyshakespeare and 0 times in the head — so it got shredded into single letters and compression collapsed. The fix is segmented random sampling (T21).
- **The optimization that never fired.** The encode cache was written read-only: it looked up entries but never stored them. Every result was correct, every test was green, and it made nothing faster. Only T20's "the cache must be non-empty after encode" catches that.
- **Silent `zip` truncation.** In weighted training, `words` and `freqs` came from different sources with different lengths, and `zip` quietly truncated. T19 reports a divergence at merge 0.

Every test was **mutation tested** — deliberately break the code, confirm the test actually goes red. A test that never goes red is not a test.

## File layout

```
src/main/
├── model.py       Transformer (RoPE / RMSNorm / SwiGLU / KV cache)
├── tokenizer.py   BPE tokenizer + chat template (incl. tool_call and FIM special tokens)
├── dataset.py     domain mixer / shard read-write / SFT dataset
├── prepare.py     corpus -> tokenizer -> shards
├── pretrain.py    pretraining
├── train.py       SFT
├── sampling.py    sampling and chat
├── common.py      checkpoints / lr schedule / bpb evaluation
├── config.py      every hyperparameter
├── test.py        26 consistency tests
└── tools/         corpus download and verification
```

## Where this actually stands

**Done**: tokenizer, corpus pipeline, Transformer, and the full pretraining / SFT / sampling code. 26/26 tests passing. **The 132M base model has been pretrained** — one full epoch over all 5.75B tokens, on a single H100.

**Not done**:

- `main.py` and `agent.py` are empty files. The tokenizer reserves `<|tool_call_start|>` and friends, but the tool-calling runtime and chat CLI are not written yet.
- No SFT run yet on the new base model, and no midtraining / RL / evaluation suite (the CORE, GSM8K tier).
- The base weights are not in this repository (1.59 GB, over GitHub's limits).

## The first pretraining run

One epoch, 10,967 optimizer steps at 524,288 tokens each, on one H100 SXM. The curve is at the top of this README.

| | |
| --- | --- |
| Wall clock | 7.4 hours |
| Throughput | 2.44 s/step, 215k tokens/s, ~20% MFU |
| Final val loss | 2.2933 |
| **Final val bpb** | **0.8523** |
| Checkpoint | 1.59 GB (model + AdamW state) |

Validation bits-per-byte over the run:

| step | 200 | 1200 | 2400 | 4800 | 8000 | 10967 |
| --- | --- | --- | --- | --- | --- | --- |
| val bpb | 1.569 | 1.053 | 0.973 | 0.912 | 0.871 | **0.852** |

Validation loss fell monotonically at every one of the 54 evaluation points and stayed within 0.03 of the training loss throughout — expected, since a single epoch shows each token exactly once. bpb is measured on this repository's own domain mix, which is 40% code and therefore more predictable than prose; it is not comparable to bpb published against WikiText or C4.

At ~20% MFU the H100 was underused. The likely culprits are the absence of `torch.compile` and the size of the logits tensor over a 24,576-entry vocabulary. Both are on the list above.

**Next**:

1. Optimizer parameter grouping (currently AdamW defaults, which applies weight decay to RMSNorm weights too)
2. `torch.compile` for the training step
3. Actually run the first full pretraining

## Pretraining throughput

One epoch is 5.75B tokens at 524,288 tokens per optimizer step, so 10,967 steps. Forward plus backward costs 0.906 GFLOPs/token including attention, which puts one epoch at **5.21 EFLOPs**.

| Configuration | Effective | One epoch on a single A100 |
| --- | --- | --- |
| fp32, no autocast | ~8 TFLOPS | ~7.5 days |
| TF32 only | ~45 TFLOPS | ~32 hours |
| **bf16 autocast (default on A100)** | ~95 TFLOPS | **~15 hours** |
| bf16 + `torch.compile` | ~130 TFLOPS | ~11 hours |

The MFU assumptions are calibrated against nanochat's published speedrun (560M params, 11.2B tokens, 4 hours on 8xH100 works out to 33% MFU); the bf16 row here is 30% of an A100's 312 TFLOPS peak.

bf16 autocast turns on automatically when CUDA reports bf16 support, and TF32 is enabled alongside it. On MPS and CPU the run stays in fp32. Checkpoints are written every 250 steps (2.3% of an epoch) and resume mid-epoch: the checkpoint records how many samples of the current epoch were consumed, and `ResumableSampler` replays the same seeded permutation and skips exactly that many.

## References

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [karpathy/nanochat](https://github.com/karpathy/nanochat) — the main reference for architecture choices and chat template design

## About the numbers

Every performance figure here was measured on a single Apple Silicon machine, with the conditions stated inline. Two caveats worth stating plainly:

- The baseline behind "990x" is **my own first naive implementation** (repeated full-sequence scans over the whole text), not any mature library.
- In the merge-count table, the "~200x" at 24274 merges is extrapolated from the measured 500 and 3000 points. Every other number is measured directly.
- The loss curve is plotted from **35 of the run's 54 evaluation points** (`docs/train-partial.log`); the full log is still on the machine that ran the job. Every plotted value is a measured one — the markers sit on real evaluations, so the two long marker-free stretches between steps 2,400 and 6,200 are exactly where points are missing. The final value and the shape are unaffected.

A few test fixtures deliberately contain non-ASCII text (`'hello 世界 world'`, `'κόσμε'`, and one translation sample in `sft_toy.jsonl`). They are there to exercise the multi-byte UTF-8 paths — T1 asserts a lossless round-trip and T3 asserts that half a character does not crash the decoder. Replacing them with ASCII would silently weaken the tests.
