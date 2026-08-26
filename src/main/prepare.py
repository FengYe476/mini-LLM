import json
import argparse
import time

from pathlib import Path

from config import PATHS, PRETRAIN, TOKENIZER, CORPUS
from tokenizer import Tokenizer, SPECIAL_TOKENS, sample_for_training
from dataset import iter_documents, build_token_stream
from common import tokenizer_fingerprint


def collect_tokenizer_text(corpus_dir: Path, weights: dict[str, float], n_chars: int,
                           oversample: int, n_segments: int, seed: int) -> str:
    parts = []
    for domain in sorted(weights):
        budget = int(n_chars * weights[domain])
        if budget < 1:
            continue
        segments = max(1, round(n_segments * weights[domain]))
        if budget // segments < 1:
            segments = budget
        pool, size = [], 0
        for text in iter_documents(corpus_dir/domain, 'train'):
            pool.append(text)
            size += len(text)
            if size >= budget * oversample:
                break
        if not pool:
            raise ValueError(f'[prepare]: domain {domain} produced no documents for tokenizer training')
        picked = sample_for_training('\n'.join(pool), budget, segments, seed)
        parts.append(picked)
        print(f'[tokenizer]: {domain:<18} pool {size:>12} chars -> sampled {len(picked):>10} chars in {segments} segments')
    return '\n'.join(parts)


def get_or_train_tokenizer(corpus_dir: Path, n_merges: int, train_chars: int) -> Tokenizer:
    if PATHS.tok.exists():
        tokenizer = Tokenizer.load(PATHS.tok)
        print(f'[tokenizer]: reuse {PATHS.tok} (vocab {tokenizer.vocab_size})')
        return tokenizer

    vocab_size = 256 + n_merges + len(SPECIAL_TOKENS)
    print(f'[tokenizer]: training a new tokenizer, target vocab {vocab_size} ({n_merges} merges)')
    text = collect_tokenizer_text(corpus_dir, TOKENIZER.weights, train_chars,
                                  TOKENIZER.oversample, TOKENIZER.n_segments, TOKENIZER.sample_seed)
    print(f'[tokenizer]: training on {len(text)} chars, this is the slow part.....')
    start = time.time()
    tokenizer = Tokenizer()
    tokenizer.train(text, vocab_size)
    tokenizer.save(PATHS.tok)
    print(f'[tokenizer]: done in {time.time() - start:.1f}s, vocab {tokenizer.vocab_size}')
    return tokenizer


def shards_are_current(want: dict) -> bool:
    if not PATHS.shard_meta.exists():
        return False
    have = json.loads(PATHS.shard_meta.read_text(encoding = 'utf-8'))
    return all(have.get(key) == value for key, value in want.items())


def report(stats: dict, weights: dict[str, float]) -> None:
    print(f"\n[{stats['split']}]: {stats['total_tokens']} tokens / {stats['total_documents']} documents / {len(stats['shards'])} shards")
    print(f'  {"domain":<20}{"target":>9}{"actual":>9}{"diff":>9}{"tokens":>14}{"docs":>10}')
    for domain in sorted(weights):
        target = weights[domain]
        actual = stats['shares'][domain]
        print(f"  {domain:<20}{target * 100:>8.2f}%{actual * 100:>8.2f}%{(actual - target) * 100:>+8.2f}pp"
              f"{stats['tokens'][domain]:>14}{stats['documents'][domain]:>10}")
    if stats['exhausted']:
        print(f"  [warning]: these domains ran out of documents and dragged the mix off target: {stats['exhausted']}")
    return


def main() -> None:
    parser = argparse.ArgumentParser(description = 'build the pretraining token shards from the text corpus')
    parser.add_argument('--corpus-dir', default = str(CORPUS.corpus_dir))
    parser.add_argument('--target-tokens', type = int, default = CORPUS.target_tokens)
    parser.add_argument('--val-tokens', type = int, default = PRETRAIN.val_tokens)
    parser.add_argument('--merges', type = int, default = TOKENIZER.n_merges)
    parser.add_argument('--train-chars', type = int, default = TOKENIZER.train_chars)
    parser.add_argument('--force', action = 'store_true', help = 'rebuild shards even if they already match the tokenizer')
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f'[prepare]: corpus directory {corpus_dir} does not exist')

    start = time.time()
    tokenizer = get_or_train_tokenizer(corpus_dir, args.merges, args.train_chars)

    want = {
        'tokenizer': tokenizer_fingerprint(PATHS.tok),
        'vocab_size': tokenizer.vocab_size,
        'corpus_dir': str(corpus_dir),
        'target_tokens': args.target_tokens,
        'val_tokens': args.val_tokens,
        'weights': CORPUS.weights,
    }
    if shards_are_current(want) and not args.force:
        have = json.loads(PATHS.shard_meta.read_text(encoding = 'utf-8'))
        print(f"[shards]: already current, {have['train']['total_tokens']} train / {have['val']['total_tokens']} val tokens")
        print(f'[shards]: pass --force to rebuild anyway')
        return

    train_stats = build_token_stream(tokenizer, corpus_dir, CORPUS.weights, PATHS.shard_dir,
                                     PRETRAIN.shard_tokens, args.target_tokens, split = 'train')
    val_stats = build_token_stream(tokenizer, corpus_dir, CORPUS.weights, PATHS.shard_dir,
                                   PRETRAIN.shard_tokens, args.val_tokens, split = 'val')

    payload = {**want, 'train': train_stats, 'val': val_stats}
    temp = PATHS.shard_meta.with_name(PATHS.shard_meta.name + '.tmp')
    PATHS.shard_meta.parent.mkdir(parents = True, exist_ok = True)
    temp.write_text(json.dumps(payload, indent = 2, ensure_ascii = False), encoding = 'utf-8')
    temp.replace(PATHS.shard_meta)

    report(train_stats, CORPUS.weights)
    report(val_stats, CORPUS.weights)
    print(f'\n[prepare]: done in {time.time() - start:.1f}s -> {PATHS.shard_dir}')
    return


if __name__ == '__main__':
    main()