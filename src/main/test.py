"""
tests.py —— tokenizer / dataset / collate 的回归测试

用法: 和 tokenizer.py、dataset.py 放同一目录, 每次改完任何一个文件就跑:
    python3 tests.py
全部 PASS 才算改完。任何 FAIL 都说明这次改动破坏了既有约定。
(这里面每一条都对应我们踩过或差点踩的坑)
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

# ---------- 公共资材 ----------
tok = Tokenizer()
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

# T1 编码解码必须无损往返(含中文和特殊token)
text = 'hello 世界 world'
ids = tok.encode(text, prepend='<|bos|>', append='<|assistant_end|>')
check('T1 encode->decode 往返', tok.decode(ids) == '<|bos|>' + text + '<|assistant_end|>')

# T2 id_to_special 的键必须是 int (曾经反过)
check('T2 id_to_special 键是int', isinstance(next(iter(tok.id_to_special)), int))

# T3 非法字节序列不崩溃 (模型将来会生成半个汉字)
try:
    r = tok.decode([228, 189])
    check('T3 非法字节不崩溃', True, repr(r))
except Exception as e:
    check('T3 非法字节不崩溃', False, f'{type(e).__name__}: {e}')

# T4 mask分区: assistant段(不含start,含end)全1, 其余全0
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
check('T4 mask分区', not errs, f'监督token数={sum(mask)}')

# T5 截断扫描: 永不超预算, mask与ids同步截断
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
check('T5 截断扫描(不超预算,mask同步)', not problems, str(problems[:3]))

# T6 纯user对话必须报错 (守卫方向: 曾被双重否定反转过)
try:
    tok.render_conversation([{'role': 'user', 'content': 'hi'}], 10**9)
    check('T6 无assistant要报错', False, '居然没报错')
except ValueError:
    check('T6 无assistant要报错', True)

# T7 max_token<2 必须报错
try:
    tok.render_conversation(CONV, 1)
    check('T7 max_token<2要报错', False, '没报错')
except ValueError:
    check('T7 max_token<2要报错', True)

# T8 截断把监督段全切掉时必须报错
conv2 = CONV + [{'role': 'user', 'content': 'thanks ' * 40}]
try:
    tok.render_conversation(conv2, 20)
    check('T8 截断后无监督要报错', False, '没报错')
except ValueError:
    check('T8 截断后无监督要报错', True)

# T9 dataset逐位对齐: labels[j] == ids[j+1] 且监督看 mask[j+1] (曾错位过两次)
ds = SFTDataset([CONV, SHORT], tok, max_length = 512)
ok = True
for idx, conv in enumerate([CONV, SHORT]):
    ids, mask = tok.render_conversation(conv, 512)
    it = ds[idx]
    expect = [ids[j + 1] if mask[j + 1] == 1 else IGNORE_INDEX for j in range(len(ids) - 1)]
    if it['input_ids'].tolist() != ids[:-1] or it['labels'].tolist() != expect:
        ok = False
check('T9 dataset逐位对齐', ok)

# T10 collate: 形状=(B,本批最长), pad区正确, 真实区未污染
loader = DataLoader(ds, batch_size = 2, shuffle = False, collate_fn = sft_collate)
b = next(iter(loader))
n0, n1 = len(ds[0]['input_ids']), len(ds[1]['input_ids'])
ok = (b['input_ids'].shape == (2, max(n0, n1))
      and bool((b['labels'][1, n1:] == IGNORE_INDEX).all())
      and bool((b['input_ids'][1, n1:] == 0).all())
      and bool((b['input_ids'][1, :n1] == ds[1]['input_ids']).all())
      and bool((b['labels'][1, :n1] == ds[1]['labels']).all()))
check('T10 collate形状与pad区', ok, f'shape={tuple(b["input_ids"].shape)}')

# T11 save/load round-trip: 落盘再加载, 行为必须逐位一致
_p = Path('._test_tok.json')
try:
    tok.save(_p)
    tok2 = Tokenizer.load(_p)
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

# T12 训练-推理模板一致性: render_for_generation 必须是 render_conversation 的严格前缀
# (训练教材和推理考卷版式一字不差; 截断路径下 bos 和结尾的 assistant_start 必须永远幸存)
_hist = CONV[:-1]                      # 半截对话: 停在 tool 结果, assistant 待生成
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
check('T12 训练-推理模板一致(严格前缀)', ok)

# T13 PretrainDataset: 切窗对齐/首尾相接/矩形batch/空数据集守卫
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
check('T13 PretrainDataset(切窗/守卫)', ok)

# T14 防护考勤: 静态检查"易蒸发"的防御性代码是否在岗
# (这些代码不参与今天跑通, 只参与未来不炸, 默写时最容易丢 -- 所以给它们上考勤)
_missing = []
_common_src = Path('common.py').read_text(encoding = 'utf-8')
_common_flat = _common_src.replace(' ', '')
if "map_location='cpu'" not in _common_flat:
    _missing.append("common.py: torch.load 缺 map_location='cpu'(跨机器搬档会炸)")
if 'assert cfg.vocab_size' not in _common_src:
    _missing.append('common.py: 缺 vocab 连体婴守卫')
if "reduction='sum'" not in _common_flat:
    _missing.append("common.py: evaluate 缺 reduction='sum'(默认mean会二次除, val loss 缩n倍)")
if 'ignore_index=IGNORE_INDEX' not in _common_flat:
    _missing.append('common.py: evaluate 缺显式 ignore_index')

_ds_src = Path('dataset.py').read_text(encoding = 'utf-8')
if 'else -100' in _ds_src:
    _missing.append('dataset.py: 裸-100, 应使用 IGNORE_INDEX 常量')

_train_src = Path('train.py').read_text(encoding = 'utf-8')
if 'tokenizer.train(' in _train_src:
    _missing.append('train.py: SFT 不该训练 tokenizer(会用玩具词表覆盖正式 tok.json)')

for _name in ('train.py', 'pretrain.py'):
    _p = Path(_name)
    if not _p.exists():
        continue
    _src = _p.read_text(encoding = 'utf-8')
    _val_line = next((l for l in _src.splitlines() if 'val_loader' in l and 'DataLoader' in l), '')
    if 'shuffle=False' not in _val_line.replace(' ', ''):
        _missing.append(f'{_name}: val_loader 未固定 shuffle=False')
check('T14 防护考勤(易蒸发防御件)', not _missing, '; '.join(_missing) if _missing else '全员在岗')

_base_p = Path('._test_base.pt')
_nope   = Path('._test_no_sft.pt')
_nobase = Path('._test_no_base.pt')
_t15 = []
try:
    # 造假base: 权重随机, 且真跑两步让 optimizer 攒出动量
    # (optimizer 状态若为空, 下面 "不继承动量" 那条断言就算写错也会假通过)
    _cfg = GPTConfig(vocab_size = tok.vocab_size, block_size = 32, embedding_dim = 32,
                     n_head = 2, n_layer = 2, dropout = 0.0)
    _m = NanoGPT(cfg = _cfg)
    _opt = torch.optim.AdamW(_m.parameters(), lr = 1e-3)
    for _ in range(2):
        _opt.zero_grad()
        _m(torch.randint(0, tok.vocab_size, (2, 8))).sum().backward()
        _opt.step()
    if not _opt.state_dict()['state']:
        _t15.append('假base的optimizer没攒出状态, 测试本身失效')
    save_checkpoint(_m, _cfg, _opt, 9, 999, _base_p)

    # 冷启动: sft档不存在, base存在
    _st = load_for_sft(_nope, _base_p, tok.vocab_size, torch.device('cpu'), 3e-4)
    if _st.start_epoch != 0:
        _t15.append(f'start_epoch应为0实为{_st.start_epoch}(不该继承base进度)')
    if _st.glob_step != 0:
        _t15.append(f'glob_step应为0实为{_st.glob_step}(不该继承base进度)')
    if _st.optimizer.state_dict()['state']:
        _t15.append('optimizer继承了base动量(SFT换了任务, 必须重开)')
    _want, _got = _m.state_dict(), _st.model.state_dict()
    if _want.keys() != _got.keys() or not all(torch.equal(_got[k].cpu(), _want[k]) for k in _want):
        _t15.append('模型权重与base不一致(没继承知识, 等于从零训练)')

    # 无base必须报错, 绝不能悄悄建随机模型
    try:
        load_for_sft(_nope, _nobase, tok.vocab_size, torch.device('cpu'), 3e-4)
        _t15.append('base缺失时没报错(会静默从随机权重开始)')
    except FileNotFoundError:
        pass

    # train.py 必须真的换门: 函数写对了但调用方没换, 是最典型的"改了一半"
    if 'load_or_init' in Path('train.py').read_text(encoding = 'utf-8'):
        _t15.append('train.py仍在用load_or_init(SFT会从随机权重开始)')
except Exception as _e:
    _t15.append(f'{type(_e).__name__}: {_e}')
finally:
    _base_p.unlink(missing_ok = True)
check('T15 SFT从base冷启动(继承权重/不继承进度)', not _t15, '; '.join(_t15))

# T16 KV cache 必须与朴素路径逐位一致
# (缓存写错不报错, 只静默生成垃圾 —— 你会误判成"模型笨", 而不是"推理坏了")
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

    # (1) logits对账: 一次性算12个位置 vs 预填充5个再逐个喂7个
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
        _t16.append(f'logits与朴素路径不一致: 最大差 {_diff16:.2e}(浮点噪声约1e-7, 真错约1e-1)')
    if _cache16.pos != 12:
        _t16.append(f'cache.pos应为12实为{_cache16.pos}(每次forward只该加一次)')

    # (2) 端到端: 贪心(top_k=1)把随机性摁死, 两条路必须逐token一致
    _a16 = _sample_loop(_m16, _c16, _p16.clone(), 20, 1.0, 1, None, use_cache = True)
    _b16 = _sample_loop(_m16, _c16, _p16.clone(), 20, 1.0, 1, None, use_cache = False)
    if _a16 != _b16:
        _n16 = next((_j for _j, (_x, _y) in enumerate(zip(_a16, _b16)) if _x != _y),
                    min(len(_a16), len(_b16)))
        _t16.append(f'贪心生成第{_n16}个token起分叉: 缓存{_a16[_n16:_n16 + 3]} vs 朴素{_b16[_n16:_n16 + 3]}')
    if len(_a16) != 20:
        _t16.append(f'缓存路径只生成{len(_a16)}个token(应为20)')

    # (3) 溢出守卫必须在岗: 缓存路径没有滑窗, 超block_size只能报错不能静默截断
    try:
        _sample_loop(_m16, _c16, _p16.clone(), 40, 1.0, 1, None, use_cache = True)
        _t16.append('prompt+max_new_tokens超block_size时没报错(缓存路径不能滑窗)')
    except ValueError:
        pass
except Exception as _e:
    _t16.append(f'{type(_e).__name__}: {_e}')
check('T16 KV cache与朴素路径一致(logits/贪心/守卫)', not _t16, '; '.join(_t16))

# T17 特殊token表: 只能追加, 永不重排/删除
# (id由列表顺序决定; 动一下顺序=换了词表=全部checkpoint作废, 而且不会报错)
_t17 = []
_ORIGINAL_11 = ['<|bos|>', '<|user_start|>', '<|user_end|>', '<|assistant_start|>',
                '<|assistant_end|>', '<|system_start|>', '<|system_end|>',
                '<|tool_call_start|>', '<|tool_call_end|>',
                '<|tool_result_start|>', '<|tool_result_end|>']
if SPECIAL_TOKENS[:11] != _ORIGINAL_11:
    _t17.append('前11个特殊token被改动(必须原序保留, 要退役就改名成reserved)')
if SPECIAL_TOKENS[11:14] != ['<|fim_prefix|>', '<|fim_suffix|>', '<|fim_middle|>']:
    _t17.append('FIM三件套缺失或顺序变了')
if sum(1 for _n in SPECIAL_TOKENS if _n.startswith('<|reserved_')) != 32:
    _t17.append('预留空位不是32个(留给将来加token而不废checkpoint)')
if len(SPECIAL_TOKENS) != 46:
    _t17.append(f'特殊token总数应为46实为{len(SPECIAL_TOKENS)}')
if len(set(SPECIAL_TOKENS)) != len(SPECIAL_TOKENS):
    _t17.append('特殊token有重名(会让id_to_special反查表少一项)')
from config import TOKENIZER
from dataclasses import replace as _replace
if _replace(TOKENIZER, n_merges = 24274).vocab_size != 24576:
    _t17.append('生产档算式不等于24576(256字节+24274merge+46特殊token)')
check('T17 特殊token表(追加式/生产档算式)', not _t17, '; '.join(_t17))

# T18 正则预切分: 全覆盖 / 训练不跨词 / 编码必须逐词
# (正则漏字符=该字符从编码里静默消失; 编码忘了预切分=merge跨词乱并, 不报错只让压缩比暴跌)
import itertools
from tokenizer import SPLIT_PATTERN

_t18 = []
try:
    for _c in ['def __init__(self, x=1024):\n    return 1\n', '🎉emoji🎉', '\x00\x01',
               'κόσμε', 'a\r\nb', '   \t \n\n', '你好世界', '']:
        if ''.join(SPLIT_PATTERN.findall(_c)) != _c:
            _t18.append(f'切分漏字符: {_c!r}')

    # 专用tokenizer: 语料含缩进, 使(空格,空格)成为一条merge —— 这是唯一能暴露"编码忘预切分"的场景
    _tok18 = Tokenizer()
    _tok18.train(('def f():\n    return 1\n    return 2\n' * 60) + ('a  b  c  d ' * 40),
                 256 + len(SPECIAL_TOKENS) + 40)

    # (1) 训练侧: 词表里不许有跨词边界的token
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
        _t18.append(f'词表含跨词token {_bad18[:3]}(train忘了预切分)')

    # (2) 编码侧: 必须等于逐词编码后拼接
    for _text in ['    return 1', 'a  b', 'x  y  z']:
        _ids = _tok18.encode(_text)
        _cat = []
        for _w in SPLIT_PATTERN.findall(_text):
            _cat.extend(_tok18._encode_word(_w))
        if _ids != _cat:
            _t18.append(f'{_text!r}: encode非逐词({len(_ids)}token) != 逐词拼接({len(_cat)}token)')
        if _tok18.decode(_ids) != _text:
            _t18.append(f'{_text!r}: 往返有损')
        _wb = set(itertools.accumulate(len(_w.encode('utf-8')) for _w in SPLIT_PATTERN.findall(_text)))
        _tb = set(itertools.accumulate(len(_tok18.vocab[_i]) for _i in _ids))
        if not _wb <= _tb:
            _t18.append(f'{_text!r}: token跨越了词边界(越界处={sorted(_wb - _tb)[:3]})')

    # (3) 旧格式档必须拒收, 不能静默用错规则解释merges
    _p18 = Path('._test_v1.json')
    try:
        _p18.write_text('{"merges": [[104, 101, 256]]}', encoding = 'utf-8')
        Tokenizer.load(_p18)
        _t18.append('v1旧格式tok.json被静默接受(merges是按旧切分规则学的)')
    except ValueError:
        pass
    finally:
        _p18.unlink(missing_ok = True)
except Exception as _e:
    _t18.append(f'{type(_e).__name__}: {_e}')
check('T18 正则预切分(全覆盖/训练不跨词/编码逐词/拒旧档)', not _t18, '; '.join(_t18))

# T19 词频加权训练必须与朴素逐词训练逐merge一致(含顺序)
# (加权只是把"同一个词数一万遍"换成"数一遍乘一万"; 权重漏乘/去重顺序变了都会静默学出不同词表)
_t19 = []
try:
    _corpus19 = ('def f(x):\n    return x + 1\n' * 40) + ('hello world hello there 你好世界 ' * 30)
    _n19 = 60
    _V19 = 256 + len(SPECIAL_TOKENS) + _n19

    _fast19 = Tokenizer()
    _fast19.train(_corpus19, _V19)

    # 参照实现: 不去重, 逐个词实例计数(A1-3之前的算法), 故意写得笨但显然正确
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
        _t19.append(f'第{_k19}条merge起分歧: 加权={_a19[_k19:_k19+1]} 朴素={_b19[_k19:_k19+1]}')
    if _fast19.vocab != _slow19.vocab:
        _t19.append('vocab不一致')
    _probe19 = 'def g(y):\n    return y * 2\n'
    if _fast19.encode(_probe19) != _slow19.encode(_probe19):
        _t19.append('同一文本编码结果不一致')

    # get_stats 的 weight 必须真的生效
    _chk19 = Tokenizer().get_stats([1, 2, 3], None, 7)
    if _chk19 != {(1, 2): 7, (2, 3): 7}:
        _t19.append(f'get_stats的weight没生效: {_chk19}')
except Exception as _e:
    _t19.append(f'{type(_e).__name__}: {_e}')
check('T19 词频加权训练与朴素训练等价', not _t19, '; '.join(_t19))

# T20 词级编码缓存: 冷热一致 / 不落盘 / 二次train必须报错
# (缓存是merges的派生物; 缓存没生效只是慢, 缓存脏了则全错而不报错)
from tokenizer import ENCODE_CACHE_MAX

_t20 = []
try:
    _c20 = ('def f(x):\n    return x + 1\n' * 40) + ('hello world 你好世界 ' * 30)
    _probe20 = 'def g(y):\n    return y * 2\nhello world\n'
    _V20 = 256 + len(SPECIAL_TOKENS) + 60

    _tok20 = Tokenizer()
    _tok20.train(_c20, _V20)
    if _tok20.cache:
        _t20.append(f'train结束后缓存未清空({len(_tok20.cache)}条)')

    _cold20 = _tok20.encode(_probe20)
    if not _tok20.cache:
        _t20.append('encode后缓存仍为空(缓存根本没生效)')
    _warm20 = _tok20.encode(_probe20)
    if _cold20 != _warm20:
        _k20 = next((_j for _j, (_x, _y) in enumerate(zip(_cold20, _warm20)) if _x != _y), 0)
        _t20.append(f'冷热结果不一致, 第{_k20}个token起分歧')
    if len(_tok20.cache) > ENCODE_CACHE_MAX:
        _t20.append(f'缓存超过上限({len(_tok20.cache)} > {ENCODE_CACHE_MAX})')

    # 二次train必须报错: num_merges会变负数, range()为空, 一条也学不到且不报错
    try:
        _tok20.train(_c20, _V20)
        _t20.append('二次train没报错(会静默学不到任何merge)')
    except ValueError:
        pass

    # 落盘: 缓存是可重算的派生物, 不该进json; load出来必须空缓存且结果一致
    _p20 = Path('._test_cache.json')
    try:
        _tok20.save(_p20)
        import json as _json20
        if 'cache' in _json20.loads(_p20.read_text(encoding = 'utf-8')):
            _t20.append('cache被写进了tok.json(它是可重算的派生物)')
        _loaded20 = Tokenizer.load(_p20)
        if _loaded20.cache:
            _t20.append('load出来的tokenizer缓存非空')
        if _loaded20.encode(_probe20) != _cold20:
            _t20.append('load后编码结果与原tokenizer不一致')
    finally:
        _p20.unlink(missing_ok = True)
except Exception as _e:
    _t20.append(f'{type(_e).__name__}: {_e}')
check('T20 词级编码缓存(冷热一致/不落盘/拒二次train)', not _t20, '; '.join(_t20))

# T21 tokenizer训练语料随机采样(ROMEO教训)
# (取头部=只读了语料第一部分; tinyshakespeare实测ROMEO全文163次/头部0次, 被切碎成单字母)
from tokenizer import sample_for_training

_t21 = []
try:
    # 合成语料: 稀有词只出现在后半段, 取头部必然漏掉
    _head21 = 'aaa bbb ccc ddd ' * 3000
    _tail21 = 'aaa bbb ZZZRARE ddd ' * 2000
    _corpus21 = _head21 + _tail21
    _n21 = 20_000

    _cut21 = _corpus21[:_n21]
    if 'ZZZRARE' in _cut21:
        _t21.append('测试语料没造对: 头部就含稀有词, 测不出差别')
    _s21 = sample_for_training(_corpus21, _n21, 200, 0)
    if 'ZZZRARE' not in _s21:
        _t21.append('随机采样没捞到只在后半段出现的词(采样没跨越全文)')

    # 预算: 不超n_chars
    if len(_s21) > _n21:
        _t21.append(f'采样结果{len(_s21)}超出预算{_n21}')

    # 确定性: 同seed必须逐字符相同(否则tokenizer无法复现)
    if sample_for_training(_corpus21, _n21, 200, 0) != _s21:
        _t21.append('同seed两次采样结果不同(tokenizer将无法复现)')
    if sample_for_training(_corpus21, _n21, 200, 1) == _s21:
        _t21.append('不同seed采样结果相同(seed没生效)')

    # 不污染全局随机源: 采样前后 random.random() 序列必须不变
    import random as _random21
    _random21.seed(12345)
    _want21 = [_random21.random() for _ in range(3)]
    _random21.seed(12345)
    sample_for_training(_corpus21, _n21, 200, 0)
    if [_random21.random() for _ in range(3)] != _want21:
        _t21.append('采样污染了全局random状态(会连带改变DataLoader的shuffle)')

    # 语料短于预算时原样返回
    if sample_for_training('short text', _n21, 200, 0) != 'short text':
        _t21.append('语料短于预算时未原样返回')

    # 非法参数必须报错
    for _a21, _b21 in [(0, 200), (_n21, 0), (10, 200)]:
        try:
            sample_for_training(_corpus21, _a21, _b21, 0)
            _t21.append(f'非法参数(n_chars={_a21}, n_segments={_b21})没报错')
        except ValueError:
            pass
except Exception as _e:
    _t21.append(f'{type(_e).__name__}: {_e}')
check('T21 训练语料随机采样(跨全文/确定性/不污染全局)', not _t21, '; '.join(_t21))

# T22 分片memmap数据集必须与内存版逐窗口完全一致
# (窗口要跨分片边界读; 不跨就会静默丢掉每个分片尾部的token, 而且长度对不上还不报错)
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
            _t22.append(f'分片{_st22}: 窗口数{len(_ds22)} != 内存版{len(_mem22)}(窗口没跨分片边界)')
            continue
        for _i22 in range(len(_mem22)):
            if not (torch.equal(_ds22[_i22]['input_ids'], _mem22[_i22]['input_ids'])
                    and torch.equal(_ds22[_i22]['labels'], _mem22[_i22]['labels'])):
                _t22.append(f'分片{_st22}: 第{_i22}个窗口与内存版不一致')
                break
        if _ds22[0]['input_ids'].dtype != torch.long:
            _t22.append(f'dtype应为torch.long实为{_ds22[0]["input_ids"].dtype}(embedding只吃long)')

    # DataLoader 拼得出矩形batch
    _b22 = next(iter(DataLoader(ShardedPretrainDataset(_d22, 100), batch_size = 4)))
    if tuple(_b22['input_ids'].shape) != (4, 100):
        _t22.append(f'batch形状{tuple(_b22["input_ids"].shape)}应为(4, 100)')

    # uint16 溢出必须报错: numpy 会把 65536 静默回绕成 0
    try:
        write_shards([1, 2, SHARD_MAX_ID + 1], _d22, 1000, prefix = 'ovf')
        _t22.append(f'token id 超过{SHARD_MAX_ID}没报错(numpy会静默回绕成0)')
    except ValueError:
        pass
    try:
        write_shards([1, -1, 2], _d22, 1000, prefix = 'neg')
        _t22.append('负数token id没报错')
    except ValueError:
        pass

    # 重写分片必须清掉旧文件, 否则新旧混读
    write_shards(list(range(1, 501)), _d22, 100, prefix = 'stale')
    write_shards(list(range(1, 201)), _d22, 100, prefix = 'stale')
    if ShardedPretrainDataset(_d22, 100, prefix = 'stale').total_tokens != 200:
        _t22.append('重写分片时旧文件没清掉(新旧数据被拼在一起)')

    # 空目录/数据不足必须报错
    shutil.rmtree(_d22, ignore_errors = True)
    _d22.mkdir(parents = True, exist_ok = True)
    try:
        ShardedPretrainDataset(_d22, 100)
        _t22.append('空分片目录没报错')
    except FileNotFoundError:
        pass
    write_shards(list(range(50)), _d22, 1000)
    try:
        ShardedPretrainDataset(_d22, 100)
        _t22.append('总token数不足block_size+1时没报错')
    except ValueError:
        pass
except Exception as _e:
    _t22.append(f'{type(_e).__name__}: {_e}')
finally:
    shutil.rmtree(_d22, ignore_errors = True)
check('T22 分片memmap与内存版等价(跨界/溢出守卫/清旧档)', not _t22, '; '.join(_t22))

# T23 流式ShardWriter必须与一次性write_shards逐字节一致
# (6B token无法一次性拿在手里; 但"边喂边写"的缓冲区跨界拼接极易错位, 且错位不报错只是数据乱掉)
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
                _t23.append(f'分片{_st23}/{_kind23}: 文件数 {len(_pb23)} != {len(_pa23)}')
                continue
            for _x23, _y23 in zip(_pa23, _pb23):
                if _x23.read_bytes() != _y23.read_bytes():
                    _t23.append(f'分片{_st23}/{_kind23}: {_y23.name} 与一次性写入的字节不一致')
                    break
            if _w23.total_tokens != len(_ids23):
                _t23.append(f'分片{_st23}/{_kind23}: total_tokens {_w23.total_tokens} != {len(_ids23)}')

    # 关闭后不许再喂
    shutil.rmtree(_b23, ignore_errors = True)
    _w23 = ShardWriter(_b23, 1000)
    _w23.add(_ids23)
    _w23.close()
    try:
        _w23.add([1, 2, 3])
        _t23.append('close后仍能add(这批token会被静默丢弃)')
    except ValueError:
        pass

    # 一个token都没喂就close必须报错, 不能留下空目录让下游以为有数据
    shutil.rmtree(_b23, ignore_errors = True)
    try:
        ShardWriter(_b23, 1000).close()
        _t23.append('空writer close没报错')
    except ValueError:
        pass

    # uint16溢出守卫在add这一侧也必须在岗
    shutil.rmtree(_b23, ignore_errors = True)
    _w23 = ShardWriter(_b23, 1000)
    for _bad23 in [[1, SHARD_MAX_ID + 1], [1, -1]]:
        try:
            _w23.add(_bad23)
            _t23.append(f'ShardWriter.add 放行了越界token {_bad23}')
        except ValueError:
            pass

    # 重开writer必须清掉旧分片
    shutil.rmtree(_b23, ignore_errors = True)
    _w23 = ShardWriter(_b23, 100); _w23.add(list(range(500))); _w23.close()
    _w23 = ShardWriter(_b23, 100); _w23.add(list(range(200))); _w23.close()
    if ShardedPretrainDataset(_b23, 50).total_tokens != 200:
        _t23.append('重开ShardWriter没清掉旧分片(新旧数据被拼在一起)')
except Exception as _e:
    _t23.append(f'{type(_e).__name__}: {_e}')
finally:
    shutil.rmtree(_a23, ignore_errors = True)
    shutil.rmtree(_b23, ignore_errors = True)
check('T23 流式ShardWriter与一次性写入等价', not _t23, '; '.join(_t23))

_pre_src = Path('pretrain.py').read_text(encoding = 'utf-8')
for _line in _pre_src.splitlines():
    if 'evaluate(' in _line and 'token_bytes' not in _line:
        _missing.append(f'pretrain.py: evaluate 调用漏传 token_bytes(bpb会变nan) -> {_line.strip()}')

# T24 多域混合器: 配比收敛 / 确定性 / 域耗尽不崩 / 格式守卫
# (配比错了模型学出来的东西就偏了, 但loss曲线看不出任何异常)
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
    # 配比收敛: 文档充足时误差必须很小
    _make24({'a': 4000, 'b': 4000, 'c': 4000})
    _m24 = DomainMixer(_root24, _W24)
    for _ in range(3000):
        _it = _m24.next_document()
        if _it is None:
            break
        _m24.credit(_it[0], len(_it[1]) // 4)
    _dev24 = max(abs(_m24.shares[_d] - _W24[_d]) for _d in _W24)
    if _dev24 > 0.01:
        _t24.append(f'配比偏差 {_dev24 * 100:.2f}pp 超过1pp: {_m24.shares}')
    if _m24.exhausted:
        _t24.append(f'文档充足却报告耗尽: {_m24.exhausted}')

    # 确定性: 同一份语料两次跑必须逐篇一致
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
        _t24.append('两次运行产出的文档序列不同(混合器必须确定性)')

    # 域耗尽: 不崩, 记入exhausted, 其余域继续
    _make24({'a': 5, 'b': 400, 'c': 400})
    _m24 = DomainMixer(_root24, _W24)
    while (_it := _m24.next_document()) is not None:
        _m24.credit(_it[0], len(_it[1]) // 4)
    if 'a' not in _m24.exhausted:
        _t24.append('文档用完的域没有被记入exhausted')
    if _m24.documents['a'] != 5:
        _t24.append(f'耗尽域实际读到{_m24.documents["a"]}篇, 应为5篇')
    if _m24.documents['b'] != 400 or _m24.documents['c'] != 400:
        _t24.append('某个域耗尽后其余域没有被读完')

    # 文档内容与顺序必须原样透出
    _make24({'a': 20, 'b': 20, 'c': 20})
    _want24 = [_json24.loads(_l)['text'] for _l in (_root24/'a'/'train_000.jsonl').read_text(encoding = 'utf-8').splitlines()]
    if list(iter_documents(_root24/'a')) != _want24:
        _t24.append('iter_documents 读出的文档内容或顺序与文件不一致')

    # 权重守卫
    for _bad24 in [{'a': 0.5, 'b': 0.3}, {'a': 1.5, 'b': -0.5}, {}]:
        try:
            DomainMixer(_root24, _bad24)
            _t24.append(f'非法权重 {_bad24} 没报错')
        except ValueError:
            pass

    # 格式守卫: 非法JSON / 缺text 都要带上文件名和行号
    (_root24/'a'/'train_000.jsonl').write_text('{"text": "ok"}\nnot json\n', encoding = 'utf-8')
    try:
        list(iter_documents(_root24/'a'))
        _t24.append('非法JSON行没报错')
    except ValueError as _e:
        if 'train_000.jsonl:2' not in str(_e):
            _t24.append(f'非法JSON报错没带文件名行号: {_e}')
    (_root24/'a'/'train_000.jsonl').write_text('{"source": "a"}\n', encoding = 'utf-8')
    try:
        list(iter_documents(_root24/'a'))
        _t24.append('缺text字段没报错')
    except ValueError:
        pass

    # 目录不存在 / 无匹配文件 必须立即报错, 不能等到第一次next才炸
    try:
        iter_documents(_root24/'nope')
        _t24.append('域目录不存在时没有立即报错')
    except FileNotFoundError:
        pass

    # config 里的方案丙配比必须合法且加起来是1
    if abs(sum(CORPUS.weights.values()) - 1.0) > 1e-9:
        _t24.append(f'CORPUS配比之和为{sum(CORPUS.weights.values())}, 必须为1.0')
    if len(CORPUS.weights) != len(CORPUS.domains):
        _t24.append('CORPUS.domains里有重名的域')
except Exception as _e:
    _t24.append(f'{type(_e).__name__}: {_e}')
finally:
    shutil.rmtree(_root24, ignore_errors = True)
check('T24 多域混合器(配比/确定性/耗尽/守卫)', not _t24, '; '.join(_t24))

# T25 流式build_token_stream: 端到端与"逐文档编码后拼接"逐token一致
# (读->编码->写分片->再读回, 中间任何一环错位都不报错, 只是训练数据静默乱掉)
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

    # 到量即停: 超出不能多于一篇文档
    if not 200_000 <= _st25['total_tokens'] < 200_000 + 20_000:
        _t25.append(f'总token {_st25["total_tokens"]} 不在[目标, 目标+一篇文档]区间内')
    if _st25['exhausted']:
        _t25.append(f'语料充足却报告耗尽: {_st25["exhausted"]}')
    if sum(_st25['tokens'].values()) != _st25['total_tokens']:
        _t25.append('各域token之和 != 总token')

    # 配比
    _dev25 = max(abs(_st25['shares'][_d] - _W25[_d]) for _d in _W25)
    if _dev25 > 0.01:
        _t25.append(f'配比偏差 {_dev25 * 100:.2f}pp 超过1pp')

    # 端到端对账: 分片token流 == 重放mixer后逐文档encode拼接
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
        _t25.append(f'分片token流与逐文档编码拼接在第{_k25}个token起分歧')
    if len(_got25) != _st25['total_tokens']:
        _t25.append(f'读回{len(_got25)}个token, 统计说{_st25["total_tokens"]}个')

    # 每篇文档都必须以bos开头
    if _got25.count(_bos25) != _st25['total_documents']:
        _t25.append(f'bos出现{_got25.count(_bos25)}次 != 文档数{_st25["total_documents"]}(文档分隔符漏插或多插)')
    if _got25 and _got25[0] != _bos25:
        _t25.append('token流第一个不是bos')

    # 确定性: 重跑一次分片必须逐字节一致
    _bytes25 = [(_s25/_n).read_bytes() for _n in _st25['shards']]
    shutil.rmtree(_s25, ignore_errors = True)
    _st25b = build_token_stream(tok, _c25, _W25, _s25, 5000, 200_000)
    if _st25b['shards'] != _st25['shards'] or [(_s25/_n).read_bytes() for _n in _st25b['shards']] != _bytes25:
        _t25.append('两次构建产出的分片不一致(必须确定性)')

    # 语料不够时: 不崩, 报告耗尽, 有多少写多少
    _make25({'a': 20, 'b': 20, 'c': 20})
    shutil.rmtree(_s25, ignore_errors = True)
    _st25c = build_token_stream(tok, _c25, _W25, _s25, 5000, 10_000_000)
    if sorted(_st25c['exhausted']) != ['a', 'b', 'c']:
        _t25.append(f'语料耗尽时exhausted应为全部域, 实为{_st25c["exhausted"]}')
    if _st25c['total_documents'] != 60:
        _t25.append(f'耗尽时应读完全部60篇, 实为{_st25c["total_documents"]}篇')

    # 非法参数
    try:
        build_token_stream(tok, _c25, _W25, _s25, 5000, 0)
        _t25.append('target_tokens=0 没报错')
    except ValueError:
        pass
except Exception as _e:
    _t25.append(f'{type(_e).__name__}: {_e}')
finally:
    shutil.rmtree(_c25, ignore_errors = True)
    shutil.rmtree(_s25, ignore_errors = True)
check('T25 流式build_token_stream端到端一致', not _t25, '; '.join(_t25))

# T26 tokenizer训练配比(A1定稿: 代码+终端75% / 英文20% / QA5%)
# (tokenizer配比错了会静默降低压缩比; 域名写错则prepare跑到一半才炸, 那时已经等了几十分钟)
from prepare import collect_tokenizer_text
from config import TOKENIZER as _TOK26

_t26 = []
_c26 = Path('._t26_corpus')
try:
    if abs(sum(_TOK26.weights.values()) - 1.0) > 1e-9:
        _t26.append(f'tokenizer配比之和为{sum(_TOK26.weights.values())}, 必须为1.0')
    if len(_TOK26.weights) != len(_TOK26.domains):
        _t26.append('TOKENIZER.domains里有重名的域')
    _extra26 = set(_TOK26.weights) - set(CORPUS.weights)
    if _extra26:
        _t26.append(f'tokenizer要采样的域{sorted(_extra26)}不在语料域里, prepare会在读取时才报错')

    _code26 = sum(_TOK26.weights.get(_d, 0) for _d in ['code_python', 'code_issues', 'code_shell', 'terminal_docs'])
    _en26 = sum(_TOK26.weights.get(_d, 0) for _d in ['web_edu', 'web_dclm', 'cosmopedia'])
    _qa26 = _TOK26.weights.get('qa_stackexchange', 0)
    if abs(_code26 - 0.75) > 1e-9 or abs(_en26 - 0.20) > 1e-9 or abs(_qa26 - 0.05) > 1e-9:
        _t26.append(f'与A1定稿不符: 代码+终端{_code26:.3f}(应0.75) 英文{_en26:.3f}(应0.20) QA{_qa26:.3f}(应0.05)')

    # 实际采样出来的各域字符数必须按配比分配(每域用一个唯一标记字符, 域名首字母会撞车)
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
            _t26.append(f'{_d26} 采样占比{_got26 * 100:.1f}% 与目标{_w26 * 100:.1f}% 相差超过2pp')
except Exception as _e:
    _t26.append(f'{type(_e).__name__}: {_e}')
finally:
    shutil.rmtree(_c26, ignore_errors = True)
check('T26 tokenizer训练配比与采样', not _t26, '; '.join(_t26))



# ---------- 汇总 ----------
fail = 0
print()
for name, ok, d in results:
    print(f'{"PASS" if ok else "FAIL"}  {name}' + (f'  [{d}]' if d else ''))
    fail += (not ok)
print(f'\n{len(results) - fail}/{len(results)} 通过')
sys.exit(1 if fail else 0)