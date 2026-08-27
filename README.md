# mini-LLM

An LLM training pipeline written from scratch: tokenizer, corpus pipeline, Transformer, pretraining, SFT, sampling. Apart from PyTorch tensor ops, not a single line comes from `transformers` or `tiktoken`.

The code itself is not the hard part. The hard part is that **almost nothing on this path fails loudly**:

- Break the KV cache → no crash, it just silently generates garbage, and you will misdiagnose it as "the model is dumb" rather than "inference is broken"
- Read one token short at a shard boundary → no crash, the tail of every shard just silently disappears
- Break a tokenizer optimization → no crash, the compression ratio just collapses and `    return 1` goes from 4 tokens to 10
- Swap the tokenizer but keep the old shards → no crash, every token id just means something different now

So what this repository is really about is not "I implemented a Transformer". It is that **every one of those silent failures has a test standing guard over it**. Currently 26/26 passing.

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
dojo/              practice floor: rewrite the whole thing from memory, to find out which designs were actually understood
```

## Where this actually stands

**Done**: tokenizer, corpus pipeline (5.75B tokens ready), Transformer, and the full pretraining / SFT / sampling code. 26/26 tests passing.

**Not done — do not misread the above**:

- The production configuration (132M) **has never been trained**. There are no usable base weights in this repository, only records from an early tinyshakespeare toy model (4 layers / 256 wide).
- `main.py` and `agent.py` are empty files. The tokenizer reserves `<|tool_call_start|>` and friends, but the tool-calling runtime and chat CLI are not written yet.
- No midtraining / RL / evaluation suite (the CORE, GSM8K tier).

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

A few test fixtures deliberately contain non-ASCII text (`'hello 世界 world'`, `'κόσμε'`, and one translation sample in `sft_toy.jsonl`). They are there to exercise the multi-byte UTF-8 paths — T1 asserts a lossless round-trip and T3 asserts that half a character does not crash the decoder. Replacing them with ASCII would silently weaken the tests.
