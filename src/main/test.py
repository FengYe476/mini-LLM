"""
tests.py -- regression tests for tokenizer / dataset / collate

usage: keep next to tokenizer.py and dataset.py, run after touching either:
    python3 tests.py
Only an all-PASS run counts as done. Any FAIL means this change broke an existing contract.
(every entry here maps to a trap we fell into, or nearly did)
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tokenizer import Tokenizer, SPECIAL_TOKENS
from dataset import SFTDataset, sft_collate, IGNORE_INDEX
from model import NanoGPT, GPTConfig
from common import load_for_sft, save_checkpoint
from model import KVCache
from sampling import _sample_loop

results = []
def check(name: str, ok: bool, detail: str = '') -> None:
    results.append((name, bool(ok), detail))

# ---------- shared fixtures ----------
tok = Tokenizer()
# non-ASCII on purpose: exercises the multi-byte UTF-8 path end to end
corpus = ('hello world 你好世界 ' * 30) + ('the quick brown fox jumps over ' * 20)
tok.train(corpus, 256 + len(SPECIAL_TOKENS) + 50)

CONV = [
    {'role': 'system', 'content': 'you are yuki'},
    {'role': 'user', 'content': 'list the files'},
    {'role': 'assistant', 'content': None,
     'tool_calls': [{'id': 'c1', 'type': 'function',
                     'function': {'name': 'run_bash', 'arguments': '{"command":"ls"}'}}]},
    {'role': 'tool', 'tool_call_id': 'c1', 'content': 'a.py\nb.py'},
    {'role': 'assistant', 'content': 'there are two files: a.py and b.py'},
]
SHORT = [
    {'role': 'user', 'content': 'hello there'},
    {'role': 'assistant', 'content': 'hello! how can i help you today?'},
]

# T1 encode->decode must round-trip losslessly (incl. non-ASCII and special tokens)
# non-ASCII fixture: multi-byte round-trip must be lossless
text = 'hello 世界 world'
ids = tok.encode(text, prepend='<|bos|>', append='<|assistant_end|>')
check('T1 encode->decode round-trip', tok.decode(ids) == '<|bos|>' + text + '<|assistant_end|>')

# T2 id_to_special keys must be int (they were inverted once)
check('T2 id_to_special keys are int', isinstance(next(iter(tok.id_to_special)), int))

# T3 invalid byte sequences must not crash (the model will emit half a character someday)
try:
    r = tok.decode([228, 189])
    check('T3 invalid bytes do not crash', True, repr(r))
except Exception as e:
    check('T3 invalid bytes do not crash', False, f'{type(e).__name__}: {e}')

# T4 mask partition: assistant span (excl. start, incl. end) all 1, everything else 0
ids, mask = tok.render_conversation(CONV, 10**9)
as_id = tok.special_token['<|assistant_start|>']
ae_id = tok.special_token['<|assistant_end|>']
inside, errs = False, []
for i, (t, m) in enumerate(zip(ids, mask)):
    if t == as_id:
        if m != 0: errs.append(i)
        inside = True
        continue
    if inside:
        if m != 1: errs.append(i)
        if t == ae_id: inside = False
    elif m != 0:
        errs.append(i)
check('T4 mask partition', not errs, f'supervised tokens={sum(mask)}')

# T5 truncation sweep: never exceed budget, mask truncated in lockstep with ids
full_ids, full_mask = ids, mask
L = len(full_ids)
problems = []
for mt in range(L + 2, 1, -1):
    try:
        i2, m2 = tok.render_conversation(CONV, mt)
    except ValueError:
        continue
    if len(i2) > mt or len(i2) != len(m2):
        problems.append(mt)
        continue
    k = len(i2) - 1
    if k and (i2[1:] != full_ids[-k:] or m2[1:] != full_mask[-k:]):
        problems.append(mt)
check('T5 truncation sweep (within budget, mask in sync)', not problems, str(problems[:3]))

# T6 a user-only conversation must raise (guard direction: was inverted by a double negative once)
try:
    tok.render_conversation([{'role': 'user', 'content': 'hi'}], 10**9)
    check('T6 missing assistant must raise', False, 'it did not raise')
except ValueError:
    check('T6 missing assistant must raise', True)

# T7 max_token < 2 must raise
try:
    tok.render_conversation(CONV, 1)
    check('T7 max_token<2 must raise', False, 'did not raise')
except ValueError:
    check('T7 max_token<2 must raise', True)

# T8 must raise when truncation cuts away the entire supervised span
conv2 = CONV + [{'role': 'user', 'content': 'thanks ' * 40}]
try:
    tok.render_conversation(conv2, 20)
    check('T8 no supervision after truncation must raise', False, 'did not raise')
except ValueError:
    check('T8 no supervision after truncation must raise', True)

# T9 dataset alignment: labels[j] == ids[j+1], and supervision reads mask[j+1] (off by one twice before)
ds = SFTDataset([CONV, SHORT], tok, max_length = 512)
ok = True
for idx, conv in enumerate([CONV, SHORT]):
    ids, mask = tok.render_conversation(conv, 512)
    it = ds[idx]
    expect = [ids[j + 1] if mask[j + 1] == 1 else IGNORE_INDEX for j in range(len(ids) - 1)]
    if it['input_ids'].tolist() != ids[:-1] or it['labels'].tolist() != expect:
        ok = False
check('T9 dataset alignment', ok)

# T10 collate: shape=(B, longest in batch), pad region correct, real region unpolluted
loader = DataLoader(ds, batch_size = 2, shuffle = False, collate_fn = sft_collate)
b = next(iter(loader))
n0, n1 = len(ds[0]['input_ids']), len(ds[1]['input_ids'])
ok = (b['input_ids'].shape == (2, max(n0, n1))
      and bool((b['labels'][1, n1:] == IGNORE_INDEX).all())
      and bool((b['input_ids'][1, n1:] == 0).all())
      and bool((b['input_ids'][1, :n1] == ds[1]['input_ids']).all())
      and bool((b['labels'][1, :n1] == ds[1]['labels']).all()))
check('T10 collate shape and pad region', ok, f'shape={tuple(b["input_ids"].shape)}')

# T11 save/load round-trip: persist then reload, behaviour must be identical token for token
_p = Path('._test_tok.json')
try:
    tok.save(_p)
    tok2 = Tokenizer.load(_p)
    # non-ASCII fixture: the reloaded tokenizer must handle multi-byte identically
    _text = 'hello 世界 world'
    ok = (tok2.encode(_text) == tok.encode(_text)
          and tok2.special_token == tok.special_token
          and tok2.vocab_size == tok.vocab_size
          and tok2.render_conversation(CONV, 512) == tok.render_conversation(CONV, 512))
    check('T11 save/load round-trip', ok)
except Exception as e:
    check('T11 save/load round-trip', False, f'{type(e).__name__}: {e}')
finally:
    _p.unlink(missing_ok = True)

# T12 train/inference template parity: render_for_generation must be a strict prefix of render_conversation
# (the training textbook and the inference exam sheet must be typeset identically; under truncation,
#  bos and the trailing assistant_start must always survive)
_hist = CONV[:-1]                      # half a conversation: stops at the tool result, assistant pending
_gen = tok.render_for_generation(_hist, 10**9)
_full_ids, _ = tok.render_conversation(CONV, 10**9)
_as = tok.special_token['<|assistant_start|>']
ok = (_gen[0] == tok.bos_token_id
      and _gen[-1] == _as
      and _full_ids[:len(_gen)] == _gen)
for _mt in range(len(_gen) + 2, 1, -1):
    _g2 = tok.render_for_generation(_hist, _mt)
    if len(_g2) > _mt or _g2[0] != tok.bos_token_id or _g2[-1] != _as:
        ok = False
        break
check('T12 train/inference template parity (strict prefix)', ok)

# T13 PretrainDataset: window alignment / end-to-end contiguity / rectangular batch / empty-dataset guard
from dataset import PretrainDataset
_pds = PretrainDataset(list(range(1001)), 100)
ok = (len(_pds) == 10
      and _pds[0]['input_ids'].tolist() == list(range(100))
      and _pds[0]['labels'].tolist() == list(range(1, 101))
      and all(_pds[i]['labels'][-1].item() == _pds[i+1]['input_ids'][0].item() for i in range(9)))
_b = next(iter(DataLoader(_pds, batch_size = 4)))
ok = ok and tuple(_b['input_ids'].shape) == (4, 100)
try:
    PretrainDataset(list(range(100)), 100)
    ok = False
except ValueError:
    pass
check('T13 PretrainDataset (windowing/guards)', ok)

# T14 guard roll-call: statically check that the "evaporation-prone" defensive code is still on duty
# (this code does not help anything run today, only helps it not explode later -- which is exactly
#  what you drop first when rewriting from memory, so it gets an attendance check)
_missing = []
_common_src = Path('common.py').read_text(encoding = 'utf-8')
_common_flat = _common_src.replace(' ', '')
if "map_location='cpu'" not in _common_flat:
    _missing.append("common.py: torch.load missing map_location='cpu' (moving a ckpt between machines will blow up)")
if 'assert cfg.vocab_size' not in _common_src:
    _missing.append('common.py: missing vocab-mismatch guard')
if "reduction='sum'" not in _common_flat:
    _missing.append("common.py: evaluate missing reduction='sum' (default mean divides twice, shrinking val loss by n)")
if 'ignore_index=IGNORE_INDEX' not in _common_flat:
    _missing.append('common.py: evaluate missing explicit ignore_index')

_ds_src = Path('dataset.py').read_text(encoding = 'utf-8')
if 'else -100' in _ds_src:
    _missing.append('dataset.py: bare -100, should use the IGNORE_INDEX constant')

_train_src = Path('train.py').read_text(encoding = 'utf-8')
if 'tokenizer.train(' in _train_src:
    _missing.append('train.py: SFT must not train the tokenizer (it would overwrite the real tok.json with a toy vocab)')

for _name in ('train.py', 'pretrain.py'):
    _p = Path(_name)
    if not _p.exists():
        continue
    _src = _p.read_text(encoding = 'utf-8')
    _val_line = next((l for l in _src.splitlines() if 'val_loader' in l and 'DataLoader' in l), '')
    if 'shuffle=False' not in _val_line.replace(' ', ''):
        _missing.append(f'{_name}: val_loader does not pin shuffle=False')
check('T14 guard roll-call (evaporation-prone defences)', not _missing, '; '.join(_missing) if _missing else 'all present')

_base_p = Path('._test_base.pt')
_nope   = Path('._test_no_sft.pt')
_nobase = Path('._test_no_base.pt')
_t15 = []
try:
    # fake base: random weights, and actually step twice so the optimizer accumulates momentum
    # (with empty optimizer state, the "does not inherit momentum" assertion below would pass even if wrong)
    _cfg = GPTConfig(vocab_size = tok.vocab_size, block_size = 32, embedding_dim = 32,
                     n_head = 2, n_layer = 2, dropout = 0.0)
    _m = NanoGPT(cfg = _cfg)
    _opt = torch.optim.AdamW(_m.parameters(), lr = 1e-3)
    for _ in range(2):
        _opt.zero_grad()
        _m(torch.randint(0, tok.vocab_size, (2, 8))).sum().backward()
        _opt.step()
    if not _opt.state_dict()['state']:
        _t15.append('fake base optimizer accumulated no state, the test itself is void')
    save_checkpoint(_m, _cfg, _opt, 9, 999, _base_p)

    # cold start: no sft checkpoint, base exists
    _st = load_for_sft(_nope, _base_p, tok.vocab_size, torch.device('cpu'), 3e-4)
    if _st.start_epoch != 0:
        _t15.append(f'start_epoch should be 0, got {_st.start_epoch} (must not inherit base progress)')
    if _st.glob_step != 0:
        _t15.append(f'glob_step should be 0, got {_st.glob_step} (must not inherit base progress)')
    if _st.optimizer.state_dict()['state']:
        _t15.append('optimizer inherited base momentum (SFT is a different task, it must start fresh)')
    _want, _got = _m.state_dict(), _st.model.state_dict()
    if _want.keys() != _got.keys() or not all(torch.equal(_got[k].cpu(), _want[k]) for k in _want):
        _t15.append('weights differ from base (no knowledge inherited, equivalent to training from scratch)')

    # missing base must raise, must never quietly build a random model
    try:
        load_for_sft(_nope, _nobase, tok.vocab_size, torch.device('cpu'), 3e-4)
        _t15.append('no error when base is missing (would silently start from random weights)')
    except FileNotFoundError:
        pass

    # train.py must actually switch doors: the function is right but the caller was not updated is the classic "half-done" change
    if 'load_or_init' in Path('train.py').read_text(encoding = 'utf-8'):
        _t15.append('train.py still calls load_or_init (SFT would start from random weights)')
except Exception as _e:
    _t15.append(f'{type(_e).__name__}: {_e}')
finally:
    _base_p.unlink(missing_ok = True)
check('T15 SFT cold start from base (inherits weights, not progress)', not _t15, '; '.join(_t15))

# T16 KV cache must match the naive path exactly
# (a broken cache does not raise, it silently generates garbage -- you will misdiagnose it as
#  "the model is dumb" rather than "inference is broken")
from model import KVCache
from sampling import _sample_loop

_t16 = []
try:
    torch.manual_seed(0)
    _c16 = GPTConfig(vocab_size = 64, block_size = 32, embedding_dim = 32,
                     n_head = 2, n_layer = 2, dropout = 0.0)
    _m16 = NanoGPT(cfg = _c16)
    _m16.eval()
    _p16 = torch.randint(0, 64, (1, 5))

    # (1) logits reconciliation: 12 positions in one shot vs prefill 5 then feed 7 one at a time
    _seq16 = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        _full16 = _m16(_seq16)
        _cache16 = KVCache(_c16.n_layer)
        _step16 = [_m16(_seq16[:, :5], _cache16)[:, -1, :]]
        for _i in range(5, 12):
            _step16.append(_m16(_seq16[:, _i:_i + 1], _cache16)[:, -1, :])
    _got16 = torch.cat(_step16, dim = 0)
    _want16 = _full16[0, 4:, :]
    _diff16 = (_got16 - _want16).abs().max().item()
    if _diff16 > 1e-4:
        _t16.append(f'logits differ from naive path: max diff {_diff16:.2e} (float noise ~1e-7, a real bug ~1e-1)')
    if _cache16.pos != 12:
        _t16.append(f'cache.pos should be 12, got {_cache16.pos} (each forward must advance it exactly once)')

    # (2) end to end: greedy (top_k=1) kills the randomness, both paths must agree token for token
    _a16 = _sample_loop(_m16, _c16, _p16.clone(), 20, 1.0, 1, None, use_cache = True)
    _b16 = _sample_loop(_m16, _c16, _p16.clone(), 20, 1.0, 1, None, use_cache = False)
    if _a16 != _b16:
        _n16 = next((_j for _j, (_x, _y) in enumerate(zip(_a16, _b16)) if _x != _y),
                    min(len(_a16), len(_b16)))
        _t16.append(f'greedy generation diverges at token {_n16}: cached {_a16[_n16:_n16 + 3]} vs naive {_b16[_n16:_n16 + 3]}')
    if len(_a16) != 20:
        _t16.append(f'cached path only generated {len(_a16)} tokens (expected 20)')

    # (3) overflow guard must be on duty: the cached path has no sliding window, exceeding block_size must raise, not silently truncate
    try:
        _sample_loop(_m16, _c16, _p16.clone(), 40, 1.0, 1, None, use_cache = True)
        _t16.append('no error when prompt+max_new_tokens exceeds block_size (the cached path cannot slide)')
    except ValueError:
        pass
except Exception as _e:
    _t16.append(f'{type(_e).__name__}: {_e}')
check('T16 KV cache matches naive path (logits/greedy/guard)', not _t16, '; '.join(_t16))

# T17 special token table: append only, never reorder or delete
# (ids come from list order; reorder one = a different vocab = every checkpoint is void, and nothing raises)
_t17 = []
_ORIGINAL_11 = ['<|bos|>', '<|user_start|>', '<|user_end|>', '<|assistant_start|>',
                '<|assistant_end|>', '<|system_start|>', '<|system_end|>',
                '<|tool_call_start|>', '<|tool_call_end|>',
                '<|tool_result_start|>', '<|tool_result_end|>']
if SPECIAL_TOKENS[:11] != _ORIGINAL_11:
    _t17.append('the first 11 special tokens changed (keep the original order; to retire one, rename it to reserved)')
if SPECIAL_TOKENS[11:14] != ['<|fim_prefix|>', '<|fim_suffix|>', '<|fim_middle|>']:
    _t17.append('the FIM trio is missing or reordered')
if sum(1 for _n in SPECIAL_TOKENS if _n.startswith('<|reserved_')) != 32:
    _t17.append('reserved slots are not 32 (they exist so new tokens do not void checkpoints)')
if len(SPECIAL_TOKENS) != 46:
    _t17.append(f'special token count should be 46, got {len(SPECIAL_TOKENS)}')
if len(set(SPECIAL_TOKENS)) != len(SPECIAL_TOKENS):
    _t17.append('duplicate special token name (would leave id_to_special one entry short)')
from config import TOKENIZER
from dataclasses import replace as _replace
if _replace(TOKENIZER, n_merges = 24274).vocab_size != 24576:
    _t17.append('production arithmetic does not equal 24576 (256 bytes + 24274 merges + 46 special tokens)')
check('T17 special token table (append-only/production arithmetic)', not _t17, '; '.join(_t17))

# T18 regex pre-split: full coverage / training never crosses words / encoding must be per-word
# (a character the regex misses vanishes silently from every encoding; forgetting the pre-split in encode
#  lets merges cross word boundaries -- no error, the compression ratio just collapses)
import itertools
from tokenizer import SPLIT_PATTERN

_t18 = []
try:
    for _c in ['def __init__(self, x=1024):\n    return 1\n', '🎉emoji🎉', '\x00\x01',
               'κόσμε', 'a\r\nb', '   \t \n\n', '你好世界', '']:
        if ''.join(SPLIT_PATTERN.findall(_c)) != _c:
            _t18.append(f'split dropped characters: {_c!r}')

    # dedicated tokenizer: the corpus has indentation, making (space, space) a merge -- the only setup that exposes "encode forgot the pre-split"
    _tok18 = Tokenizer()
    _tok18.train(('def f():\n    return 1\n    return 2\n' * 60) + ('a  b  c  d ' * 40),
                 256 + len(SPECIAL_TOKENS) + 40)

    # (1) training side: the vocab must contain no token that crosses a word boundary
    _bad18 = []
    for _tid, _piece in _tok18.vocab.items():
        if _tid < 256:
            continue
        try:
            _str = _piece.decode('utf-8')
        except UnicodeDecodeError:
            continue
        if len(SPLIT_PATTERN.findall(_str)) != 1:
            _bad18.append(_str)
    if _bad18:
        _t18.append(f'vocab contains cross-word token {_bad18[:3]} (train forgot the pre-split)')

    # (2) encoding side: must equal per-word encoding concatenated
    for _text in ['    return 1', 'a  b', 'x  y  z']:
        _ids = _tok18.encode(_text)
        _cat = []
        for _w in SPLIT_PATTERN.findall(_text):
            _cat.extend(_tok18._encode_word(_w))
        if _ids != _cat:
            _t18.append(f'{_text!r}: encode is not per-word ({len(_ids)} tokens) != per-word concat ({len(_cat)} tokens)')
        if _tok18.decode(_ids) != _text:
            _t18.append(f'{_text!r}: lossy round-trip')
        _wb = set(itertools.accumulate(len(_w.encode('utf-8')) for _w in SPLIT_PATTERN.findall(_text)))
        _tb = set(itertools.accumulate(len(_tok18.vocab[_i]) for _i in _ids))
        if not _wb <= _tb:
            _t18.append(f'{_text!r}: token crossed a word boundary (crossing at={sorted(_wb - _tb)[:3]})')

    # (3) old-format files must be rejected, never silently reinterpreted under the new rules
    _p18 = Path('._test_v1.json')
    try:
        _p18.write_text('{"merges": [[104, 101, 256]]}', encoding = 'utf-8')
        Tokenizer.load(_p18)
        _t18.append('a v1 tok.json was silently accepted (its merges were learned under the old split rules)')
    except ValueError:
        pass
    finally:
        _p18.unlink(missing_ok = True)
except Exception as _e:
    _t18.append(f'{type(_e).__name__}: {_e}')
check('T18 regex pre-split (coverage/no cross-word/per-word encode/reject old format)', not _t18, '; '.join(_t18))

# T19 frequency-weighted training must match naive per-word training merge for merge (order included)
# (weighting only replaces "count the same word ten thousand times" with "count it once, times ten thousand";
#  a missing multiply or a changed dedup order silently learns a different vocab)
_t19 = []
try:
    _corpus19 = ('def f(x):\n    return x + 1\n' * 40) + ('hello world hello there 你好世界 ' * 30)
    _n19 = 60
    _V19 = 256 + len(SPECIAL_TOKENS) + _n19

    _fast19 = Tokenizer()
    _fast19.train(_corpus19, _V19)

    # reference implementation: no dedup, count every word instance (the pre-optimisation algorithm), deliberately dumb but obviously correct
    _slow19 = Tokenizer()
    _w19 = [list(_x.encode('utf-8')) for _x in SPLIT_PATTERN.findall(_corpus19)]
    for _i in range(_n19):
        _c19 = {}
        for _word in _w19:
            _slow19.get_stats(_word, _c19)
        if not _c19:
            break
        _pair19 = max(_c19, key = _c19.get)
        _nid19 = _slow19.vocab_size + _i
        _slow19.merges[_pair19] = _nid19
        _slow19.vocab[_nid19] = _slow19.vocab[_pair19[0]] + _slow19.vocab[_pair19[1]]
        _w19 = [_slow19.merge(_word, _pair19, _nid19) for _word in _w19]
    _slow19.vocab_size = len(_slow19.vocab)
    _slow19._register_special_token()

    _a19, _b19 = list(_fast19.merges.items()), list(_slow19.merges.items())
    if _a19 != _b19:
        _k19 = next((_j for _j, (_x, _y) in enumerate(zip(_a19, _b19)) if _x != _y), min(len(_a19), len(_b19)))
        _t19.append(f'merge {_k19} diverges: weighted={_a19[_k19:_k19+1]} naive={_b19[_k19:_k19+1]}')
    if _fast19.vocab != _slow19.vocab:
        _t19.append('vocab mismatch')
    _probe19 = 'def g(y):\n    return y * 2\n'
    if _fast19.encode(_probe19) != _slow19.encode(_probe19):
        _t19.append('same text encodes differently')

    # the weight argument of get_stats must actually take effect
    _chk19 = Tokenizer().get_stats([1, 2, 3], None, 7)
    if _chk19 != {(1, 2): 7, (2, 3): 7}:
        _t19.append(f'get_stats weight had no effect: {_chk19}')
except Exception as _e:
    _t19.append(f'{type(_e).__name__}: {_e}')
check('T19 weighted training equals naive training', not _t19, '; '.join(_t19))

# T20 word-level encode cache: cold==hot / never persisted / retraining must raise
# (the cache is derived from merges; a cache that never fires is merely slow, a stale cache is wrong and silent)
from tokenizer import ENCODE_CACHE_MAX

_t20 = []
try:
    _c20 = ('def f(x):\n    return x + 1\n' * 40) + ('hello world 你好世界 ' * 30)
    _probe20 = 'def g(y):\n    return y * 2\nhello world\n'
    _V20 = 256 + len(SPECIAL_TOKENS) + 60

    _tok20 = Tokenizer()
    _tok20.train(_c20, _V20)
    if _tok20.cache:
        _t20.append(f'cache not cleared after train ({len(_tok20.cache)} entries)')

    _cold20 = _tok20.encode(_probe20)
    if not _tok20.cache:
        _t20.append('cache still empty after encode (the cache never fired at all)')
    _warm20 = _tok20.encode(_probe20)
    if _cold20 != _warm20:
        _k20 = next((_j for _j, (_x, _y) in enumerate(zip(_cold20, _warm20)) if _x != _y), 0)
        _t20.append(f'cold and hot results differ, diverging at token {_k20}')
    if len(_tok20.cache) > ENCODE_CACHE_MAX:
        _t20.append(f'cache exceeded its cap ({len(_tok20.cache)} > {ENCODE_CACHE_MAX})')

    # retraining must raise: num_merges would go negative, range() would be empty, learning nothing without complaint
    try:
        _tok20.train(_c20, _V20)
        _t20.append('retraining did not raise (it would silently learn no merges at all)')
    except ValueError:
        pass

    # persistence: the cache is recomputable derived data, it must not enter the json; a loaded tokenizer must have an empty cache and identical results
    _p20 = Path('._test_cache.json')
    try:
        _tok20.save(_p20)
        import json as _json20
        if 'cache' in _json20.loads(_p20.read_text(encoding = 'utf-8')):
            _t20.append('cache was written into tok.json (it is recomputable derived data)')
        _loaded20 = Tokenizer.load(_p20)
        if _loaded20.cache:
            _t20.append('loaded tokenizer has a non-empty cache')
        if _loaded20.encode(_probe20) != _cold20:
            _t20.append('loaded tokenizer encodes differently from the original')
    finally:
        _p20.unlink(missing_ok = True)
except Exception as _e:
    _t20.append(f'{type(_e).__name__}: {_e}')
check('T20 word-level encode cache (cold==hot/not persisted/rejects retrain)', not _t20, '; '.join(_t20))

# T21 tokenizer training corpus must be sampled randomly (the ROMEO lesson)
# (taking the head = reading only the first slice of the corpus; measured on tinyshakespeare, ROMEO appears
#  163 times in the full text and 0 times in the head, so it got shredded into single letters)
from tokenizer import sample_for_training

_t21 = []
try:
    # synthetic corpus: the rare word only appears in the second half, taking the head must miss it
    _head21 = 'aaa bbb ccc ddd ' * 3000
    _tail21 = 'aaa bbb ZZZRARE ddd ' * 2000
    _corpus21 = _head21 + _tail21
    _n21 = 20_000

    _cut21 = _corpus21[:_n21]
    if 'ZZZRARE' in _cut21:
        _t21.append('the test corpus is wrong: the head already contains the rare word, so it proves nothing')
    _s21 = sample_for_training(_corpus21, _n21, 200, 0)
    if 'ZZZRARE' not in _s21:
        _t21.append('random sampling missed a word that only appears in the second half (sampling did not span the text)')

    # budget: never exceed n_chars
    if len(_s21) > _n21:
        _t21.append(f'sample of {len(_s21)} exceeds the budget of {_n21}')

    # determinism: the same seed must give byte-identical output (otherwise the tokenizer is not reproducible)
    if sample_for_training(_corpus21, _n21, 200, 0) != _s21:
        _t21.append('same seed produced different samples (the tokenizer would not be reproducible)')
    if sample_for_training(_corpus21, _n21, 200, 1) == _s21:
        _t21.append('different seeds produced the same sample (seed had no effect)')

    # must not pollute the global RNG: the random.random() sequence must be unchanged across sampling
    import random as _random21
    _random21.seed(12345)
    _want21 = [_random21.random() for _ in range(3)]
    _random21.seed(12345)
    sample_for_training(_corpus21, _n21, 200, 0)
    if [_random21.random() for _ in range(3)] != _want21:
        _t21.append('sampling polluted the global random state (it would also change the DataLoader shuffle)')

    # a corpus shorter than the budget is returned as-is
    if sample_for_training('short text', _n21, 200, 0) != 'short text':
        _t21.append('corpus shorter than budget was not returned as-is')

    # invalid arguments must raise
    for _a21, _b21 in [(0, 200), (_n21, 0), (10, 200)]:
        try:
            sample_for_training(_corpus21, _a21, _b21, 0)
            _t21.append(f'invalid args (n_chars={_a21}, n_segments={_b21}) did not raise')
        except ValueError:
            pass
except Exception as _e:
    _t21.append(f'{type(_e).__name__}: {_e}')
check('T21 corpus random sampling (spans full text/deterministic/no global pollution)', not _t21, '; '.join(_t21))

# T22 the sharded memmap dataset must match the in-memory version window for window
# (windows must read across shard boundaries; if they do not, the tail of every shard is silently dropped,
#  and the length mismatch does not raise either)
import shutil
from dataset import ShardedPretrainDataset, write_shards, SHARD_MAX_ID

_t22 = []
_d22 = Path('._test_shards')
try:
    _ids22 = list(range(1, 1001))
    _mem22 = PretrainDataset(_ids22, 100)
    for _st22 in [10_000, 300, 101, 7]:
        shutil.rmtree(_d22, ignore_errors = True)
        _paths22 = write_shards(_ids22, _d22, _st22)
        _ds22 = ShardedPretrainDataset(_d22, 100)
        if len(_ds22) != len(_mem22):
            _t22.append(f'shard {_st22}: window count {len(_ds22)} != in-memory {len(_mem22)} (windows do not cross shard boundaries)')
            continue
        for _i22 in range(len(_mem22)):
            if not (torch.equal(_ds22[_i22]['input_ids'], _mem22[_i22]['input_ids'])
                    and torch.equal(_ds22[_i22]['labels'], _mem22[_i22]['labels'])):
                _t22.append(f'shard {_st22}: window {_i22} differs from the in-memory version')
                break
        if _ds22[0]['input_ids'].dtype != torch.long:
            _t22.append(f'dtype should be torch.long, got {_ds22[0]["input_ids"].dtype} (embedding only accepts long)')

    # DataLoader can assemble a rectangular batch
    _b22 = next(iter(DataLoader(ShardedPretrainDataset(_d22, 100), batch_size = 4)))
    if tuple(_b22['input_ids'].shape) != (4, 100):
        _t22.append(f'batch shape {tuple(_b22["input_ids"].shape)} should be (4, 100)')

    # uint16 overflow must raise: numpy silently wraps 65536 to 0
    try:
        write_shards([1, 2, SHARD_MAX_ID + 1], _d22, 1000, prefix = 'ovf')
        _t22.append(f'token id above {SHARD_MAX_ID} did not raise (numpy would silently wrap it to 0)')
    except ValueError:
        pass
    try:
        write_shards([1, -1, 2], _d22, 1000, prefix = 'neg')
        _t22.append('negative token id did not raise')
    except ValueError:
        pass

    # rewriting shards must clear the old files, otherwise old and new are read together
    write_shards(list(range(1, 501)), _d22, 100, prefix = 'stale')
    write_shards(list(range(1, 201)), _d22, 100, prefix = 'stale')
    if ShardedPretrainDataset(_d22, 100, prefix = 'stale').total_tokens != 200:
        _t22.append('rewriting shards left the old files behind (old and new data spliced together)')

    # empty directory / insufficient data must raise
    shutil.rmtree(_d22, ignore_errors = True)
    _d22.mkdir(parents = True, exist_ok = True)
    try:
        ShardedPretrainDataset(_d22, 100)
        _t22.append('empty shard directory did not raise')
    except FileNotFoundError:
        pass
    write_shards(list(range(50)), _d22, 1000)
    try:
        ShardedPretrainDataset(_d22, 100)
        _t22.append('total tokens below block_size+1 did not raise')
    except ValueError:
        pass
except Exception as _e:
    _t22.append(f'{type(_e).__name__}: {_e}')
finally:
    shutil.rmtree(_d22, ignore_errors = True)
check('T22 sharded memmap equals in-memory (boundaries/overflow guard/stale cleanup)', not _t22, '; '.join(_t22))

# T23 the streaming ShardWriter must be byte-identical to one-shot write_shards
# (6B tokens do not fit in memory; but the buffer splicing in "write as you go" misaligns very easily,
#  and a misalignment does not raise, the data is just scrambled)
from dataset import ShardWriter
import random

_t23 = []
_a23, _b23 = Path('._t23_a'), Path('._t23_b')
try:
    _rng23 = random.Random(0)
    _ids23 = [_rng23.randrange(0, 24576) for _ in range(5000)]

    def _chunks23(kind):
        if kind == 'whole':
            return [_ids23]
        if kind == 'single':
            return [[_i] for _i in _ids23]
        _out, _i = [], 0
        while _i < len(_ids23):
            _n = _rng23.randrange(1, 400)
            _out.append(_ids23[_i:_i + _n])
            _i += _n
        return _out

    for _st23 in [10_000, 1000, 333, 7]:
        for _kind23 in ['whole', 'single', 'random']:
            shutil.rmtree(_a23, ignore_errors = True)
            shutil.rmtree(_b23, ignore_errors = True)
            _pa23 = write_shards(_ids23, _a23, _st23)
            _w23 = ShardWriter(_b23, _st23)
            for _c23 in _chunks23(_kind23):
                _w23.add(_c23)
            _pb23 = _w23.close()
            if len(_pa23) != len(_pb23):
                _t23.append(f'shard {_st23}/{_kind23}: file count {len(_pb23)} != {len(_pa23)}')
                continue
            for _x23, _y23 in zip(_pa23, _pb23):
                if _x23.read_bytes() != _y23.read_bytes():
                    _t23.append(f'shard {_st23}/{_kind23}: {_y23.name} differs bytewise from the one-shot write')
                    break
            if _w23.total_tokens != len(_ids23):
                _t23.append(f'shard {_st23}/{_kind23}: total_tokens {_w23.total_tokens} != {len(_ids23)}')

    # no feeding after close
    shutil.rmtree(_b23, ignore_errors = True)
    _w23 = ShardWriter(_b23, 1000)
    _w23.add(_ids23)
    _w23.close()
    try:
        _w23.add([1, 2, 3])
        _t23.append('add still worked after close (those tokens would be silently discarded)')
    except ValueError:
        pass

    # closing without a single token must raise, it must not leave an empty directory that looks like data downstream
    shutil.rmtree(_b23, ignore_errors = True)
    try:
        ShardWriter(_b23, 1000).close()
        _t23.append('closing an empty writer did not raise')
    except ValueError:
        pass

    # the uint16 overflow guard must be on duty on the add side too
    shutil.rmtree(_b23, ignore_errors = True)
    _w23 = ShardWriter(_b23, 1000)
    for _bad23 in [[1, SHARD_MAX_ID + 1], [1, -1]]:
        try:
            _w23.add(_bad23)
            _t23.append(f'ShardWriter.add let an out-of-range token through: {_bad23}')
        except ValueError:
            pass

    # reopening the writer must clear the old shards
    shutil.rmtree(_b23, ignore_errors = True)
    _w23 = ShardWriter(_b23, 100); _w23.add(list(range(500))); _w23.close()
    _w23 = ShardWriter(_b23, 100); _w23.add(list(range(200))); _w23.close()
    if ShardedPretrainDataset(_b23, 50).total_tokens != 200:
        _t23.append('reopening ShardWriter left the old shards behind (old and new data spliced together)')
except Exception as _e:
    _t23.append(f'{type(_e).__name__}: {_e}')
finally:
    shutil.rmtree(_a23, ignore_errors = True)
    shutil.rmtree(_b23, ignore_errors = True)
check('T23 streaming ShardWriter equals one-shot write', not _t23, '; '.join(_t23))

_pre_src = Path('pretrain.py').read_text(encoding = 'utf-8')
for _line in _pre_src.splitlines():
    if 'evaluate(' in _line and 'token_bytes' not in _line:
        _missing.append(f'pretrain.py: evaluate call is missing token_bytes (bpb would be nan) -> {_line.strip()}')

# T24 multi-domain mixer: mix converges / deterministic / exhaustion does not crash / format guards
# (a wrong mix biases everything the model learns, and the loss curve shows nothing unusual)
from dataset import DomainMixer, iter_documents
from config import CORPUS
import json as _json24

_t24 = []
_root24 = Path('._t24_corpus')
_W24 = {'a': 0.5, 'b': 0.3, 'c': 0.2}

def _make24(counts, root = _root24):
    shutil.rmtree(root, ignore_errors = True)
    _r = random.Random(0)
    for _d, _n in counts.items():
        _p = root/_d
        _p.mkdir(parents = True)
        with open(_p/'train_000.jsonl', 'w', encoding = 'utf-8') as _f:
            for _i in range(_n):
                _f.write(_json24.dumps({'text': 'x ' * _r.randrange(50, 500),
                                        'source': _d, 'id': f'{_d}/{_i:09d}'}) + '\n')

try:
    # mix convergence: with enough documents the deviation must be small
    _make24({'a': 4000, 'b': 4000, 'c': 4000})
    _m24 = DomainMixer(_root24, _W24)
    for _ in range(3000):
        _it = _m24.next_document()
        if _it is None:
            break
        _m24.credit(_it[0], len(_it[1]) // 4)
    _dev24 = max(abs(_m24.shares[_d] - _W24[_d]) for _d in _W24)
    if _dev24 > 0.01:
        _t24.append(f'mix deviation {_dev24 * 100:.2f}pp exceeds 1pp: {_m24.shares}')
    if _m24.exhausted:
        _t24.append(f'reported exhaustion despite having enough documents: {_m24.exhausted}')

    # determinism: the same corpus must yield the same document sequence twice
    def _run24():
        _m = DomainMixer(_root24, _W24)
        _seq = []
        for _ in range(500):
            _it = _m.next_document()
            if _it is None:
                break
            _m.credit(_it[0], len(_it[1]) // 4)
            _seq.append((_it[0], len(_it[1])))
        return _seq
    if _run24() != _run24():
        _t24.append('two runs produced different document sequences (the mixer must be deterministic)')

    # exhaustion: do not crash, record it in exhausted, keep the other domains going
    _make24({'a': 5, 'b': 400, 'c': 400})
    _m24 = DomainMixer(_root24, _W24)
    while (_it := _m24.next_document()) is not None:
        _m24.credit(_it[0], len(_it[1]) // 4)
    if 'a' not in _m24.exhausted:
        _t24.append('an exhausted domain was not recorded in exhausted')
    if _m24.documents['a'] != 5:
        _t24.append(f'exhausted domain actually yielded {_m24.documents["a"]} documents, expected 5')
    if _m24.documents['b'] != 400 or _m24.documents['c'] != 400:
        _t24.append('other domains were not drained after one domain ran out')

    # document content and order must pass through untouched
    _make24({'a': 20, 'b': 20, 'c': 20})
    _want24 = [_json24.loads(_l)['text'] for _l in (_root24/'a'/'train_000.jsonl').read_text(encoding = 'utf-8').splitlines()]
    if list(iter_documents(_root24/'a')) != _want24:
        _t24.append('iter_documents returned different content or order than the file')

    # weight guards
    for _bad24 in [{'a': 0.5, 'b': 0.3}, {'a': 1.5, 'b': -0.5}, {}]:
        try:
            DomainMixer(_root24, _bad24)
            _t24.append(f'invalid weight {_bad24} did not raise')
        except ValueError:
            pass

    # format guards: malformed JSON / missing text must both report the file name and line number
    (_root24/'a'/'train_000.jsonl').write_text('{"text": "ok"}\nnot json\n', encoding = 'utf-8')
    try:
        list(iter_documents(_root24/'a'))
        _t24.append('a malformed JSON line did not raise')
    except ValueError as _e:
        if 'train_000.jsonl:2' not in str(_e):
            _t24.append(f'malformed JSON error lacks file name and line number: {_e}')
    (_root24/'a'/'train_000.jsonl').write_text('{"source": "a"}\n', encoding = 'utf-8')
    try:
        list(iter_documents(_root24/'a'))
        _t24.append('a missing text field did not raise')
    except ValueError:
        pass

    # a missing directory / no matching files must raise immediately, not on the first next()
    try:
        iter_documents(_root24/'nope')
        _t24.append('a missing domain directory did not raise immediately')
    except FileNotFoundError:
        pass

    # the mix in config must be valid and sum to 1
    if abs(sum(CORPUS.weights.values()) - 1.0) > 1e-9:
        _t24.append(f'CORPUS weights sum to {sum(CORPUS.weights.values())}, must be 1.0')
    if len(CORPUS.weights) != len(CORPUS.domains):
        _t24.append('CORPUS.domains contains a duplicate domain')
except Exception as _e:
    _t24.append(f'{type(_e).__name__}: {_e}')
finally:
    shutil.rmtree(_root24, ignore_errors = True)
check('T24 multi-domain mixer (mix/determinism/exhaustion/guards)', not _t24, '; '.join(_t24))

# T25 streaming build_token_stream: end to end, token for token, equal to "encode each document then concatenate"
# (read -> encode -> write shards -> read back; a misalignment anywhere in that chain does not raise,
#  the training data is just silently scrambled)
from dataset import build_token_stream

_t25 = []
_c25, _s25 = Path('._t25_corpus'), Path('._t25_shards')
_W25 = {'a': 0.5, 'b': 0.3, 'c': 0.2}

def _make25(counts):
    shutil.rmtree(_c25, ignore_errors = True)
    _r = random.Random(0)
    for _d, _n in counts.items():
        (_c25/_d).mkdir(parents = True)
        with open(_c25/_d/'train_000.jsonl', 'w', encoding = 'utf-8') as _f:
            for _i in range(_n):
                _f.write(_json24.dumps({'text': 'hello world ' * _r.randrange(5, 60),
                                        'source': _d, 'id': f'{_d}/{_i:09d}'}) + '\n')

try:
    _bos25 = tok.encode_special_token('<|bos|>')
    _make25({'a': 2000, 'b': 2000, 'c': 2000})
    shutil.rmtree(_s25, ignore_errors = True)
    _st25 = build_token_stream(tok, _c25, _W25, _s25, 5000, 200_000)

    # stop on target: the overshoot must be at most one document
    if not 200_000 <= _st25['total_tokens'] < 200_000 + 20_000:
        _t25.append(f'total tokens {_st25["total_tokens"]} outside [target, target + one document]')
    if _st25['exhausted']:
        _t25.append(f'reported exhaustion despite a sufficient corpus: {_st25["exhausted"]}')
    if sum(_st25['tokens'].values()) != _st25['total_tokens']:
        _t25.append('per-domain token counts do not sum to the total')

    # mix
    _dev25 = max(abs(_st25['shares'][_d] - _W25[_d]) for _d in _W25)
    if _dev25 > 0.01:
        _t25.append(f'mix deviation {_dev25 * 100:.2f}pp exceeds 1pp')

    # end-to-end reconciliation: the shard token stream == replaying the mixer and encoding each document
    _m25 = DomainMixer(_c25, _W25)
    _want25 = []
    while len(_want25) < _st25['total_tokens']:
        _it25 = _m25.next_document()
        if _it25 is None:
            break
        _ids25 = tok.encode(_it25[1], prepend = _bos25)
        _want25.extend(_ids25)
        _m25.credit(_it25[0], len(_ids25))
    _ds25 = ShardedPretrainDataset(_s25, 128)
    _got25 = _ds25._read(0, _ds25.total_tokens).tolist()
    if _got25 != _want25[:len(_got25)]:
        _k25 = next((_j for _j, (_x, _y) in enumerate(zip(_got25, _want25)) if _x != _y), 0)
        _t25.append(f'shard token stream diverges from per-document encoding at token {_k25}')
    if len(_got25) != _st25['total_tokens']:
        _t25.append(f'read back {len(_got25)} tokens, the stats claim {_st25["total_tokens"]}')

    # every document must start with bos
    if _got25.count(_bos25) != _st25['total_documents']:
        _t25.append(f'bos appears {_got25.count(_bos25)} times != document count {_st25["total_documents"]} (a separator was dropped or doubled)')
    if _got25 and _got25[0] != _bos25:
        _t25.append('the token stream does not start with bos')

    # determinism: rebuilding must produce byte-identical shards
    _bytes25 = [(_s25/_n).read_bytes() for _n in _st25['shards']]
    shutil.rmtree(_s25, ignore_errors = True)
    _st25b = build_token_stream(tok, _c25, _W25, _s25, 5000, 200_000)
    if _st25b['shards'] != _st25['shards'] or [(_s25/_n).read_bytes() for _n in _st25b['shards']] != _bytes25:
        _t25.append('two builds produced different shards (it must be deterministic)')

    # when the corpus runs short: do not crash, report exhaustion, write what there is
    _make25({'a': 20, 'b': 20, 'c': 20})
    shutil.rmtree(_s25, ignore_errors = True)
    _st25c = build_token_stream(tok, _c25, _W25, _s25, 5000, 10_000_000)
    if sorted(_st25c['exhausted']) != ['a', 'b', 'c']:
        _t25.append(f'on exhaustion every domain should be in exhausted, got {_st25c["exhausted"]}')
    if _st25c['total_documents'] != 60:
        _t25.append(f'should have read all 60 documents on exhaustion, got {_st25c["total_documents"]}')

    # invalid arguments
    try:
        build_token_stream(tok, _c25, _W25, _s25, 5000, 0)
        _t25.append('target_tokens=0 did not raise')
    except ValueError:
        pass
except Exception as _e:
    _t25.append(f'{type(_e).__name__}: {_e}')
finally:
    shutil.rmtree(_c25, ignore_errors = True)
    shutil.rmtree(_s25, ignore_errors = True)
check('T25 streaming build_token_stream end-to-end parity', not _t25, '; '.join(_t25))

# T26 tokenizer training mix (final: code+terminal 75% / English 20% / QA 5%)
# (a wrong tokenizer mix silently lowers the compression ratio; a misspelled domain only blows up halfway
#  through prepare, by which point you have already waited tens of minutes)
from prepare import collect_tokenizer_text
from config import TOKENIZER as _TOK26

_t26 = []
_c26 = Path('._t26_corpus')
try:
    if abs(sum(_TOK26.weights.values()) - 1.0) > 1e-9:
        _t26.append(f'tokenizer weights sum to {sum(_TOK26.weights.values())}, must be 1.0')
    if len(_TOK26.weights) != len(_TOK26.domains):
        _t26.append('TOKENIZER.domains contains a duplicate domain')
    _extra26 = set(_TOK26.weights) - set(CORPUS.weights)
    if _extra26:
        _t26.append(f'tokenizer wants to sample domains {sorted(_extra26)} that are not in the corpus, prepare would only fail at read time')

    _code26 = sum(_TOK26.weights.get(_d, 0) for _d in ['code_python', 'code_issues', 'code_shell', 'terminal_docs'])
    _en26 = sum(_TOK26.weights.get(_d, 0) for _d in ['web_edu', 'web_dclm', 'cosmopedia'])
    _qa26 = _TOK26.weights.get('qa_stackexchange', 0)
    if abs(_code26 - 0.75) > 1e-9 or abs(_en26 - 0.20) > 1e-9 or abs(_qa26 - 0.05) > 1e-9:
        _t26.append(f'does not match the final mix: code+terminal {_code26:.3f} (want 0.75) English {_en26:.3f} (want 0.20) QA {_qa26:.3f} (want 0.05)')

    # the sampled per-domain character counts must follow the mix (each domain uses a unique marker char, initials collide)
    shutil.rmtree(_c26, ignore_errors = True)
    _r26 = random.Random(0)
    _mark26 = {_d: chr(ord('A') + _i) for _i, _d in enumerate(sorted(_TOK26.weights))}
    for _d26 in _TOK26.weights:
        (_c26/_d26).mkdir(parents = True)
        with open(_c26/_d26/'train_000.jsonl', 'w', encoding = 'utf-8') as _f26:
            for _i26 in range(400):
                _f26.write(_json24.dumps({'text': _mark26[_d26] * _r26.randrange(200, 800),
                                          'source': _d26, 'id': f'{_d26}/{_i26:09d}'}) + '\n')
    _text26 = collect_tokenizer_text(_c26, _TOK26.weights, 100_000, 4, 100, 0)
    for _d26, _w26 in _TOK26.weights.items():
        _got26 = _text26.count(_mark26[_d26]) / len(_text26)
        if abs(_got26 - _w26) > 0.02:
            _t26.append(f'{_d26} sampled share {_got26 * 100:.1f}% differs from target {_w26 * 100:.1f}% by more than 2pp')
except Exception as _e:
    _t26.append(f'{type(_e).__name__}: {_e}')
finally:
    shutil.rmtree(_c26, ignore_errors = True)
check('T26 tokenizer training mix and sampling', not _t26, '; '.join(_t26))



# ---------- summary ----------
fail = 0
print()
for name, ok, d in results:
    print(f'{"PASS" if ok else "FAIL"}  {name}' + (f'  [{d}]' if d else ''))
    fail += (not ok)
print(f'\n{len(results) - fail}/{len(results)} passed')
sys.exit(1 if fail else 0)