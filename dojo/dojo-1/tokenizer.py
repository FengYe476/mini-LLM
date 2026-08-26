import os
import json
import regex
import random

from typing import Any
from pathlib import Path
from collections import Counter

SPLIT_PATTERN = regex.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
TOKENIZER_FORMAT = 2
ENCODE_CACHE_MAX = 100_000

SPECIAL_TOKENS = [
    '<|bos|>',
    '<|user_start|>',
    '<|user_end|>',
    '<|assistant_start|>',
    '<|assistant_end|>',
    '<|system_start|>',
    '<|system_end|>',
    '<|tool_call_start|>',
    '<|tool_call_end|>',
    '<|tool_result_start|>',
    '<|tool_result_end|>',
    '<|fim_prefix|>',
    '<|fim_suffix|>',
    '<|fim_middle|>',
] + [f'<|reserved_{i}|>' for i in range(32)]

if len(set(SPECIAL_TOKENS)) != len(SPECIAL_TOKENS):
    raise ValueError(f'[special token]: duplicated name in SPEICIAL_TOKEN ({len(SPECIAL_TOKENS)})')

def sample_for_training(content: str, n_chars: int, n_segments: int, seed: int) -> str:
    if n_chars <= 0 or n_segments <= 0:
        raise ValueError(f'[sample]: the n_chars({n_chars}) or n_segments({n_segments}) must be greater than 0')
    seg_len = n_chars // n_segments
    if seg_len <= 1:
        raise ValueError(f'[len error]: the n_chars({n_chars}) is too small for n_segments({n_segments})')
    if len(content) < n_chars:
        return content
    rng = random.Random(seed)
    starts = sorted(rng.randrange(0, len(content) - seg_len) for _ in range(n_segments))
    return ''.join(content[start: start + seg_len] for start in starts)


class Tokenizer:
    def __init__(self) -> None:
        self.vocab_size = 256
        self.vocab = {
            i: bytes([i])
            for i in range(256)
        }
        self.merges = {}
        self.special_token = {}
        self.id_to_special = {}
        self.bos_token_id = None
        self.cache = {}
        return

    def get_stats(self, ids: list[int], counts: dict|None = None, weight: int = 1) -> dict[tuple[int, int], int]:
        counts = counts if counts is not None else {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i+1])
            counts[pair] = counts.get(pair, 0) + weight
        return counts

    def merge(self, ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def _register_special_token(self) -> None:
        token_offset = self.vocab_size
        self.special_token = {
            name: token_offset + i
            for i, name in enumerate(SPECIAL_TOKENS)
        }
        self.id_to_special = {
            i: name
            for name, i in self.special_token.items()
        }
        self.bos_token_id = self.special_token['<|bos|>']
        self.vocab_size = token_offset + len(SPECIAL_TOKENS)
        return

    def train(self, content: str, vocab_size: int) -> None:
        if self.merges:
            raise ValueError(f'[train twice]: the tokenizer has been trained with ({len(self.merges)}) merges, use the new Tokenizer()')
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        if vocab_size_no_special < 256:
            raise ValueError(f'[vocab error]: the vocab size without special token less than 256')
        word_freq = Counter(SPLIT_PATTERN.findall(content))
        try:
            words = [list(w.encode('utf-8')) for w in word_freq]
        except UnicodeEncodeError as e:
            raise ValueError(f'[encode error]: cannot encode the data: {e}')
        freqs = list(word_freq.values())
        num_merges = vocab_size_no_special - self.vocab_size
        for i in range(num_merges):
            counts = {}
            for word, freq in zip(words, freqs):
                counts = self.get_stats(word, counts, freq)
            if not counts:
                break
            pair = max(counts, key = counts.get)
            new_id = self.vocab_size + i
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            words = [self.merge(word, pair, new_id) for word in words]
        self.vocab_size = len(self.vocab)
        self._register_special_token()
        self.cache.clear()
        return

    def encode_special_token(self, name: str) -> int:
        if name not in self.special_token:
            raise ValueError(f'[invalid name]: unknown name:({name})')
        return self.special_token[name]

    def get_bos_token_ids(self) -> int:
        if self.bos_token_id is None:
            raise ValueError(f'[bos error]: bos token has not been initialled')
        return self.bos_token_id

    def _encode_word(self, word: str) -> list[int]:
        cached = self.cache.get(word)
        if cached is not None:
            return cached
        ids = list(word.encode('utf-8'))
        if self.merges:
            while len(ids) >= 2:
                counts = self.get_stats(ids)
                valid_pair = [pair for pair in counts if pair in self.merges]
                if not valid_pair:
                    break
                pair = min(valid_pair, key = lambda p: self.merges[p])
                new_id = self.merges[pair]
                ids = self.merge(ids, pair, new_id)
        if len(self.cache) < ENCODE_CACHE_MAX:
            self.cache[word] =ids
        return ids

    def encode(self, content: str, prepend: int|str|None = None, append: int|str|None = None) -> list[int]:
        if prepend is not None:
            if isinstance(prepend, int):
                prepend_id = prepend
            else:
                prepend_id = self.encode_special_token(prepend)
        if append is not None:
            if isinstance(append, int):
                append_id = append
            else:
                append_id = self.encode_special_token(append)
        try:
            ids = []
            for word in SPLIT_PATTERN.findall(content):
                ids.extend(self._encode_word(word))
        except UnicodeEncodeError as e:
            raise ValueError(f'[encode error]: cannot encode the data: {e}')
        
        if prepend is not None:
            ids.insert(0, prepend_id)
        if append is not None:
            ids.append(append_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        out_bytes = b''
        out_content = []
        for token_id in ids:
            if token_id in self.id_to_special:
                if out_bytes:
                    out_content.append(
                        out_bytes.decode('utf-8', errors = 'replace')
                    )
                out_bytes = b''
                out_content.append(self.id_to_special[token_id])
            else:
                out_bytes += self.vocab[token_id]
        if out_bytes:
            out_content.append(
                out_bytes.decode('utf-8', errors = 'replace')
            )
        return ''.join(out_content)

    def _add_token(self, out_ids: list[int], out_mask: list[int], token_ids: int|list[int], mask_val: int) -> None:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        out_ids.extend(token_ids)
        out_mask.extend([mask_val] * len(token_ids))
        return

    def _add_special(self, out_ids: list[int], out_mask: list[int], token_name: str, mask_val:int) -> None:
        token_ids = self.encode_special_token(token_name)
        self._add_token(out_ids, out_mask, token_ids, mask_val)
        return

    def canonical_json(self, content: Any) -> str:
        return json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

    def _add_content(self, out_ids: list[int], out_mask: list[int], content: Any, mask_val: int) -> None:
        if not content:
            return

        if isinstance(content, str):
            content = content
        else:
            content = self.canonical_json(content)
        self._add_token(out_ids, out_mask, self.encode(content), mask_val)
        return

    def _render_message(self, message: dict[str, Any], supervise: bool) -> tuple[list[int], list[int]]:
        message_id = []
        message_mask = []
        role = message.get('role')
        content = message.get('content')

        if role in {'system', 'developer'}:
            self._add_special(message_id, message_mask, '<|system_start|>', 0)
            self._add_content(message_id, message_mask, content, 0)
            self._add_special(message_id, message_mask, '<|system_end|>', 0)
        elif role == 'user':
            self._add_special(message_id, message_mask, '<|user_start|>', 0)
            self._add_content(message_id, message_mask, content, 0)
            self._add_special(message_id, message_mask, '<|user_end|>', 0)
        elif role == 'assistant':
            self._add_special(message_id, message_mask, '<|assistant_start|>', 0)
            target_mask = 1 if supervise else 0
            self._add_content(message_id, message_mask, content, target_mask)
            tool_calls = message.get('tool_calls') or []
            for tool_call in tool_calls:
                if hasattr(tool_call, 'model_dump'):
                    tool_call = tool_call.model_dump(exclude_none = True)

                tool_call_content = self.canonical_json(tool_call)
                self._add_special(message_id, message_mask, '<|tool_call_start|>', target_mask)
                self._add_token(message_id, message_mask, self.encode(tool_call_content), target_mask)
                self._add_special(message_id, message_mask, '<|tool_call_end|>', target_mask)
            self._add_special(message_id, message_mask, '<|assistant_end|>', target_mask)
        elif role == 'tool':
            payload = {
                'tool_call_id': message.get('tool_call_id'),
                'content': message.get('content')
            }
            tool_result_content = self.canonical_json(payload)
            self._add_special(message_id, message_mask, '<|tool_result_start|>', 0)
            self._add_token(message_id, message_mask, self.encode(tool_result_content), 0)
            self._add_special(message_id, message_mask, '<|tool_result_end|>', 0)
        else:
            raise ValueError(f'[invalid role]: invalid role: {role}')
        return message_id, message_mask

    def render_conversation(self, messages: list[dict[str, Any]], max_token: int) -> tuple[list[int], list[int]]:
        if max_token < 2:
            raise ValueError(f'[token error]: max token is less than 2')
        ids = [self.get_bos_token_ids()]
        mask = [0]
        for message in messages:
            supervise = message.get('role') == 'assistant'
            message_ids, message_mask = self._render_message(message, supervise)
            ids.extend(message_ids)
            mask.extend(message_mask)

        if len(ids) > max_token:
            keep = max_token - 1
            ids = [ids[0]] + ids[-keep:]
            mask = [mask[0]] + mask[-keep:]

        if 1 not in mask:
            raise ValueError(f'[mask error]: there is not learnable information in mask')
        return ids, mask

    def save(self, p: str|Path) -> None:
        p = Path(p)
        payload = {
            'version': TOKENIZER_FORMAT,
            'data': [
                [pair[0], pair[1], new_id] for pair, new_id in self.merges.items()
            ]
        }
        p.parent.mkdir(parents = True, exist_ok=True)
        temp = p.with_name(p.name + '.tmp')
        try:
            temp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), errors = 'replace', encoding = 'utf-8')
            os.replace(temp, p)
        finally:
            temp.unlink(missing_ok=True)
        return

    @classmethod
    def load(cls, p: str|Path) -> 'Tokenizer':
        p = Path(p)
        tok = cls()
        data = json.loads(p.read_text(errors = 'replace', encoding = 'utf-8'))
        version = data.get('version')
        if version != TOKENIZER_FORMAT:
            raise ValueError(f'[format error]: {p} is format v {version} but this code expects v {TOKENIZER_FORMAT}')
        for p0, p1, new_id in data['messages']:
            tok.merges[p0, p1] = new_id
            tok.vocab[new_id] = tok.vocab[p0] + tok.vocab[p1]

        tok.vocab_size = len(tok.vocab)
        tok._register_special_token()
        return tok

    def render_generation(self, messages: list[dict[str, Any]], max_token: int) -> list[int]:
        if max_token < 2:
            raise ValueError(f'[max token]: the max token less than 2')
        ids = [self.get_bos_token_ids()]
        for message in messages:
            message_id, message_mask = self._render_message(message, False)
            ids.extend(message_id)
        ids.append(self.encode_special_token('<|assistant_start|>'))
        if len(ids) > max_token:
            keep = max_token - 1
            ids = [ids[0]] + ids[-keep:]
        return ids
                 
        

        
            
        
        
             


    

    
            
        
            
        
