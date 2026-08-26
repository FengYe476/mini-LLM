import os
import bisect
import itertools
import numpy as np
import torch
import json

from torch.utils.data import Dataset, Sampler
from pathlib import Path
from typing import Any, Sequence, Iterator

IGNORE_INDEX = -100
SHARD_DTYPE = np.uint16
SHARD_MAX_ID = int(np.iinfo(SHARD_DTYPE).max)
SHARD_ITEM_SIZE = np.dtype(SHARD_DTYPE).itemsize

def _check_token_range(chunk: np.ndarray) -> None:
    hi, lo = int(chunk.max()), int(chunk.min())
    if hi > SHARD_MAX_ID or lo < 0:
        raise ValueError(f'[shard]: token id out of uint16 range (min={lo}, max={hi}, allowed 0..{SHARD_MAX_ID}); numpy would silently wrap 65536 to 0')
    return

def _write_shard_file(chunk: np.ndarray, path: Path) -> None:
    temp = path.with_name(path.name + '.tmp')
    try:
        chunk.astype(SHARD_DTYPE).tofile(temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok = True)
    return

def _clear_stale_shards(shard_dir: Path, prefix: str) -> None:
    shard_dir.mkdir(parents = True, exist_ok = True)
    for stale in shard_dir.glob(f'{prefix}_*.bin'):
        stale.unlink()
    return

def write_shards(token_ids: Sequence[int], shard_dir: str|Path, shard_tokens: int, prefix: str = 'train') -> list[Path]:
    if shard_tokens < 1:
        raise ValueError(f'[shard]: shard_tokens({shard_tokens}) must be >= 1')
    shard_dir = Path(shard_dir)
    _clear_stale_shards(shard_dir, prefix)

    paths = []
    for start in range(0, len(token_ids), shard_tokens):
        chunk = np.asarray(token_ids[start: start + shard_tokens], dtype = np.int64)
        if chunk.size == 0:
            continue
        _check_token_range(chunk)
        path = shard_dir/f'{prefix}_{len(paths):06d}.bin'
        _write_shard_file(chunk, path)
        paths.append(path)
    if not paths:
        raise ValueError(f'[shard]: token_ids is empty, nothing written')
    return paths

class SFTDataset(Dataset):
    def __init__(self, conversation: Sequence[list[dict[str, Any]]], tokenizer: Any, max_length: int) -> None:
        self.conversation = conversation
        self.tokenizer = tokenizer
        self.max_length = max_length
        return

    def __len__(self) -> int:
        return len(self.conversation)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        conversation = self.conversation[index]
        ids, mask = self.tokenizer.render_conversation(conversation, self.max_length)
        input_ids = ids[:-1]
        labels = [
            token if m == 1 else IGNORE_INDEX
            for token, m in zip(
                ids[1:],
                mask[1:],
                strict=True
            )
        ]
        return {
            'input_ids': torch.tensor(input_ids, dtype = torch.long),
            'labels': torch.tensor(labels, dtype = torch.long)
        }


def sft_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch_max = max(len(item['input_ids']) for item in batch)

    B = len(batch)
    input_ids = torch.full((B, batch_max), 0, dtype = torch.long)
    labels = torch.full((B, batch_max), IGNORE_INDEX, dtype = torch.long)
    
    for i, item in enumerate(batch):
        n = len(item['input_ids'])
        input_ids[i, :n] = item['input_ids']
        labels[i, :n] = item['labels']

    return {
        'input_ids': input_ids,
        'labels': labels
    }


class PretrainDataset(Dataset):
    def __init__(self, token_ids: list[int], block_size: int) -> None:
        if len(token_ids) < block_size + 1:
            raise ValueError(f'[size error]: the block size({block_size}) is greater than length of token_ids({len(token_ids)})')
        self.ids = torch.tensor(token_ids, dtype = torch.long)
        self.block_size = block_size
        return

    def __len__(self) -> int:
        return (len(self.ids) - 1) // self.block_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.block_size
        input_ids = self.ids[start: start + self.block_size]
        labels = self.ids[start + 1:start + self.block_size + 1]
        return {
            'input_ids': input_ids,
            'labels': labels
        }

class ShardedPretrainDataset(Dataset):
    def __init__(self, shard_dir: str|Path, block_size: int, prefix: str = 'train') -> None:
        self.shard_paths = sorted(Path(shard_dir).glob(f'{prefix}_*.bin'))
        if not self.shard_paths:
            raise FileNotFoundError(f'[shard]: no {prefix}_*.bin found in {shard_dir}, run write_shards first')
        self.block_size = block_size
        self.shard_tokens = [path.stat().st_size // SHARD_ITEM_SIZE for path in self.shard_paths]
        self.total_tokens = sum(self.shard_tokens)
        if self.total_tokens < block_size + 1:
            raise ValueError(f'[shard]: total tokens({self.total_tokens}) is less than block_size + 1({block_size + 1})')
        self.cum_tokens = list(itertools.accumulate(self.shard_tokens))
        self._memmap = {}
        return

    def __len__(self) -> int:
        return (self.total_tokens - 1) // self.block_size

    def _get_memmap(self, shard_index: int) -> np.memmap:
        memmap = self._memmap.get(shard_index)
        if memmap is None:
            memmap = np.memmap(self.shard_paths[shard_index], dtype = SHARD_DTYPE, mode = 'r')
            self._memmap[shard_index] = memmap
        return memmap

    def _read(self, start: int, length: int) -> np.ndarray:
        out = np.empty(length, dtype = np.int64)
        filled = 0
        shard_index = bisect.bisect_right(self.cum_tokens, start)
        while filled < length:
            if shard_index >= len(self.shard_paths):
                raise IndexError(f'[shard]: ran out of shards reading {length} tokens from {start} (total={self.total_tokens})')
            base = self.cum_tokens[shard_index - 1] if shard_index else 0
            memmap = self._get_memmap(shard_index)
            local = start + filled - base
            take = min(length - filled, len(memmap) - local)
            if take <= 0:
                raise ValueError(f'[shard]: empty shard {self.shard_paths[shard_index]} blocks reading, shard sizes are {self.shard_tokens}')
            out[filled: filled + take] = memmap[local: local + take]
            filled += take
            shard_index += 1
        return out

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f'[shard]: index {index} out of range (len = {len(self)})')
        window = self._read(index * self.block_size, self.block_size + 1)
        return {
            'input_ids': torch.from_numpy(window[:-1]),
            'labels': torch.from_numpy(window[1:])
        }

class ResumableSampler(Sampler):
    def __init__(self, n_samples: int, seed: int, epoch: int = 0, skip: int = 0) -> None:
        if n_samples < 1:
            raise ValueError(f'[sampler]: n_samples({n_samples}) must be >= 1')
        self.n_samples = n_samples
        self.seed = seed
        self.set_epoch(epoch, skip)
        return

    def set_epoch(self, epoch: int, skip: int = 0) -> None:
        if not 0 <= skip <= self.n_samples:
            raise ValueError(f'[sampler]: skip({skip}) must be in [0, {self.n_samples}]; a checkpoint from a different shard set would land here')
        self.epoch = epoch
        self.skip = skip
        return

    def __len__(self) -> int:
        return self.n_samples - self.skip

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.n_samples, generator = generator).tolist()
        return iter(order[self.skip:])

class ShardWriter:
    def __init__(self, shard_dir: str|Path, shard_tokens: int, prefix: str = 'train') -> None:
        if shard_tokens < 1:
            raise ValueError(f'[shard writer]: shard_tokens({shard_tokens}) must be >= 1')
        self.shard_dir = Path(shard_dir)
        self.shard_tokens = shard_tokens
        self.prefix = prefix
        _clear_stale_shards(self.shard_dir, prefix)
        self.buffer = np.empty(shard_tokens, dtype = SHARD_DTYPE)
        self.filled = 0
        self.paths = []
        self.total_tokens = 0
        self.closed = False
        return

    def _flush(self) -> None:
        if self.filled == 0:
            return
        path = self.shard_dir/f'{self.prefix}_{len(self.paths):06d}.bin'
        _write_shard_file(self.buffer[:self.filled], path)
        self.paths.append(path)
        self.filled = 0
        return

    def add(self, token_ids: Sequence[int]) -> None:
        if self.closed:
            raise ValueError(f'[shard writer]: {self.prefix} writer is already closed, cannot add more tokens')
        chunk = np.asarray(token_ids, dtype = np.int64)
        if chunk.size == 0:
            return
        _check_token_range(chunk)
        chunk = chunk.astype(SHARD_DTYPE)
        self.total_tokens += int(chunk.size)
        pos = 0
        while pos < chunk.size:
            take = min(chunk.size - pos, self.shard_tokens - self.filled)
            self.buffer[self.filled: self.filled + take] = chunk[pos: pos + take]
            self.filled += take
            pos += take
            if self.filled == self.shard_tokens:
                self._flush()
        return

    def close(self) -> list[Path]:
        if not self.closed:
            self._flush()
            self.closed = True
        if not self.paths:
            raise ValueError(f'[shard writer]: no tokens were added for prefix {self.prefix}, nothing written')
        return self.paths

def _list_domain_files(domain_dir: str|Path, split: str) -> list[Path]:
    domain_dir = Path(domain_dir)
    if not domain_dir.is_dir():
        raise FileNotFoundError(f'[corpus]: domain directory {domain_dir} does not exist')
    paths = sorted(domain_dir.glob(f'{split}_*.jsonl'))
    if not paths:
        if list(domain_dir.glob(f'{split}_*.jsonl.zst')):
            raise FileNotFoundError(f'[corpus]: {domain_dir} only holds compressed {split}_*.jsonl.zst; decompress them first (zstd -d), this reader reads plain .jsonl only')
        raise FileNotFoundError(f'[corpus]: no {split}_*.jsonl found in {domain_dir}')
    return paths

def _iter_jsonl(paths: list[Path]) -> Iterator[str]:
    for path in paths:
        with open(path, encoding = 'utf-8') as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f'[corpus]: {path}:{lineno} is not valid JSON: {e}')
                text = doc.get('text')
                if not isinstance(text, str) or not text:
                    raise ValueError(f'[corpus]: {path}:{lineno} has no non-empty string "text" field, keys are {sorted(doc)}')
                yield text

def iter_documents(domain_dir: str|Path, split: str = 'train') -> Iterator[str]:
    return _iter_jsonl(_list_domain_files(domain_dir, split))


class DomainMixer:
    def __init__(self, corpus_dir: str|Path, weights: dict[str, float], split: str = 'train') -> None:
        if not weights:
            raise ValueError(f'[mixer]: weights is empty, nothing to mix')
        bad = {d: w for d, w in weights.items() if w <= 0}
        if bad:
            raise ValueError(f'[mixer]: every weight must be > 0, got {bad}')
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f'[mixer]: weights sum to {total}, must sum to 1.0; got {weights}')

        self.corpus_dir = Path(corpus_dir)
        self.weights = dict(weights)
        self.split = split
        self.readers = {domain: iter_documents(self.corpus_dir/domain, split) for domain in sorted(weights)}
        self.produced = {domain: 0 for domain in self.readers}
        self.documents = {domain: 0 for domain in self.readers}
        self.exhausted = set()
        return

    def _pick(self) -> str|None:
        live = [d for d in self.readers if d not in self.exhausted]
        if not live:
            return None
        return min(live, key = lambda d: (self.produced[d] / self.weights[d], d))

    def next_document(self) -> tuple[str, str]|None:
        while True:
            domain = self._pick()
            if domain is None:
                return None
            try:
                text = next(self.readers[domain])
            except StopIteration:
                self.exhausted.add(domain)
                continue
            self.documents[domain] += 1
            return domain, text

    def credit(self, domain: str, n_tokens: int) -> None:
        if domain not in self.produced:
            raise KeyError(f'[mixer]: unknown domain {domain!r}, known domains are {sorted(self.produced)}')
        self.produced[domain] += n_tokens
        return

    @property
    def total_tokens(self) -> int:
        return sum(self.produced.values())

    @property
    def shares(self) -> dict[str, float]:
        total = self.total_tokens
        return {d: (n / total if total else 0.0) for d, n in self.produced.items()}

def build_token_stream(tokenizer: Any, corpus_dir: str|Path, weights: dict[str, float],
                       shard_dir: str|Path, shard_tokens: int, target_tokens: int,
                       split: str = 'train', bos: str = '<|bos|>') -> dict[str, Any]:
    if target_tokens < 1:
        raise ValueError(f'[build]: target_tokens({target_tokens}) must be >= 1')
    bos_id = tokenizer.encode_special_token(bos)
    mixer = DomainMixer(corpus_dir, weights, split)
    writer = ShardWriter(shard_dir, shard_tokens, prefix = split)

    while writer.total_tokens < target_tokens:
        item = mixer.next_document()
        if item is None:
            break
        domain, text = item
        ids = tokenizer.encode(text, prepend = bos_id)
        writer.add(ids)
        mixer.credit(domain, len(ids))

    paths = writer.close()
    if writer.total_tokens != mixer.total_tokens:
        raise ValueError(f'[build]: writer wrote {writer.total_tokens} tokens but mixer credited {mixer.total_tokens}; the two counters must agree')
    return {
        'split': split,
        'total_tokens': writer.total_tokens,
        'total_documents': sum(mixer.documents.values()),
        'tokens': dict(mixer.produced),
        'documents': dict(mixer.documents),
        'shares': mixer.shares,
        'exhausted': sorted(mixer.exhausted),
        'shards': [path.name for path in paths],
    }