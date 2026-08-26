# mini-LLM

从零手写的 LLM 训练管线：分词器、语料管线、Transformer、预训练、SFT、采样。除了 PyTorch 的张量运算，没有一行来自 `transformers` 或 `tiktoken`。

代码本身不算难。难的是**这条链路上几乎每一处出错都不会报错**：

- KV cache 写错 → 不崩溃，只是静默生成垃圾，你会误判成「模型笨」而不是「推理坏了」
- 分片边界少读一个 token → 不崩溃，只是每个分片尾部的数据静默消失
- 分词器优化改错 → 不崩溃，只是压缩比暴跌，`    return 1` 从 4 个 token 变成 10 个
- 换了分词器却用旧分片 → 不崩溃，只是所有 token id 的含义全变了

所以这个仓库真正想展示的不是「我实现了 Transformer」，而是**每一处静默失败都有一个测试守着**。目前 26/26 通过。

---

## 先看一个两行代码的问题

BPE 编码时需要找出「当前哪些 pair 可以合并」。这两行取出的集合完全相同：

```python
valid = [pair for pair in self.merges if pair in counts]   # A
valid = [pair for pair in counts if pair in self.merges]   # B
```

在生产词表下，**B 比 A 快约 200 倍**。

因为 `self.merges` 有 24274 项，而 `counts` 只有一个词那么长——大约 10 项。两者算出的是同一个交集，工作量却差了三个数量级。而且这个差距随词表线性放大：500 merges 时 9.4×，3000 时 28.6×，24274 时约 200×。

这是分词器编码提速 990 倍的三步之一（另外两步是正则预切分和词级缓存）。

但更重要的是下一个问题：**我凭什么相信 A 和 B 真的等价？** 「变快了」和「变快且变错了」在日志上看起来一模一样。这个仓库对每一步优化都写了一份「笨到不可能写错」的参照实现，用测试断言两者逐 token 一致——详见下面的测试一节。

## 实测

**分词器吞吐**（2 MB 真实 Python 代码语料，Apple Silicon，单线程）

| 实现 | 吞吐 | 压缩比 |
| --- | --- | --- |
| mini-LLM（冷缓存） | 5.77 MB/s | 3.81 字符/token |
| mini-LLM（热缓存） | **18.08 MB/s** | 3.81 |
| tiktoken `cl100k_base`（**Rust**） | 20.03 MB/s | 4.16 |

纯 Python 打到了 Rust 实现的 90%。词表小 4 倍（24.5K vs 100K），压缩比只差 8%——而词表小 4 倍意味着 embedding 和 lm_head 省下大量参数。

**语料**（8 域混合，`src/main/prepare.log` 有完整报告）

| 项 | 值 |
| --- | --- |
| 原始语料 | 22 GB |
| 训练 token | 5,750,197,205（116 个分片） |
| 验证 token | 2,001,461（1 个分片） |
| 配比偏差 | ≤ 0.06 个百分点 |
| 构建耗时 | 5.5 小时 |

域配比：`web_edu` 20% / `code_python` 20% / `web_dclm` 15% / `qa_stackexchange` 15% / `cosmopedia` 12% / `code_issues` 10% / `code_shell` 5% / `terminal_docs` 3%

**模型**

| 项 | 值 |
| --- | --- |
| 参数量 | 132.3 M（非 embedding 113.4 M） |
| 层数 / 头数 / 维度 | 12 / 12 / 768 |
| 上下文 | 1024 |
| 词表 | 24576 = 256 字节 + 24274 merges + 46 特殊 token |

## 快速开始

```bash
uv sync
cd src/main

# 跑一遍全部一致性测试（不需要语料，几分钟）
python3 test.py

# 构建语料分片（需要 data/corpus/ 下的 8 域 jsonl）
python3 prepare.py

# 预训练 → SFT → 采样
python3 pretrain.py
python3 train.py
python3 sampling.py
```

分词器已经训练好并随仓库提供（`src/main/data/tok.json`），所以上面那个吞吐基准不需要任何语料就能复现。

## 管线

```
prepare.py    语料 → BPE 分词器 → uint16 token 分片
              ├── DomainMixer     8 域确定性配比调度
              ├── ShardWriter     流式写盘（6B token 拿不进内存）
              └── 指纹绑定         分片与分词器用 sha256 绑定，换了就拒绝启动

pretrain.py   分片 → 基座模型
              ├── memmap 随机读，窗口跨分片边界
              ├── cosine + warmup、梯度裁剪、断点续训
              └── bpb 评估（不依赖词表，换分词器仍可比）

train.py      基座 → SFT（对话模板 + loss mask，只监督 assistant 段）

sampling.py   temperature / top-k / KV cache / stop token
```

## 模型

`src/main/model.py`，是 Llama / nanochat 那一档的配置，不是原版 GPT-2：

- **RoPE** 旋转位置编码（`build_rope_cache` / `rotate_half` / `apply_rope`），带 KV cache 下的位置偏移
- **RMSNorm** + **Pre-Norm** 残差
- **SwiGLU** 前馈（gate / up / down 三矩阵）
- **权重共享**：lm_head 与 token embedding 绑定
- **残差缩放初始化**：`0.02 / sqrt(2 * n_layer)`
- **SDPA** 注意力 + 手写 **KV cache**

## 测试：26 个静默失败

这是仓库的主体。每个测试的注释都写明了它在防哪一种「不报错但会训废」的错误，其中好几个是真踩过的坑：

| 测试 | 防什么 |
| --- | --- |
| T16 | KV cache 与朴素路径必须逐 logit 一致 |
| T18 | 正则预切分：词表不许有跨词 token，编码必须等于逐词编码后拼接 |
| T19 | 词频加权训练必须与朴素逐实例训练**逐 merge** 一致（含顺序） |
| T20 | 编码缓存：冷热结果一致、不落盘、二次训练必须报错 |
| T21 | 分词器语料必须随机采样全文（**ROMEO 教训**，见下） |
| T22 | 分片 memmap 必须与内存版逐窗口一致，窗口要能跨分片边界 |
| T12 | 训练模板与推理模板必须严格前缀一致 |
| T14 | 「防护考勤」：静态检查那些只在未来才起作用的防御代码还在不在岗 |

几个真实的坑：

- **ROMEO 教训**：分词器训练语料取了开头一段，而 `ROMEO` 在 tinyshakespeare 全文出现 163 次、在开头 0 次——于是它被切成单个字母，压缩比暴跌。修法是分段随机采样（T21）。
- **优化没生效**：编码缓存写成了只查不写。结果完全正确，测试全绿，但一点没加速。只有 T20 那条「encode 之后缓存必须非空」能发现。
- **`zip` 静默截断**：词频加权训练时 `words` 和 `freqs` 来源不同，长度不等，`zip` 默默截断。T19 在第 0 条 merge 就报出分歧。

每条测试都做过**变异测试**——故意把代码改坏，确认测试真的会变红。不会变红的测试等于没写。

## 文件结构

```
src/main/
├── model.py       Transformer（RoPE / RMSNorm / SwiGLU / KV cache）
├── tokenizer.py   BPE 分词器 + 对话模板（含 tool_call / FIM 特殊 token）
├── dataset.py     域混合器 / 分片读写 / SFT 数据集
├── prepare.py     语料 → 分词器 → 分片
├── pretrain.py    预训练
├── train.py       SFT
├── sampling.py    采样与对话
├── common.py      检查点 / 学习率调度 / bpb 评估
├── config.py      全部超参
├── test.py        26 项一致性测试
└── tools/         语料下载与校验
dojo/              默写练习场：不看原代码重写一遍，用来检验哪些设计是真的理解了
```

## 现在到哪一步了

**已完成**：分词器、语料管线（5.75B token 已就绪）、Transformer、预训练/SFT/采样的完整代码，26/26 测试通过。

**未完成，不要误会**：

- 生产配置（132M）**一次都没训练过**。仓库里没有可用的基座权重，只有早期 tinyshakespeare 玩具模型（4 层 / 256 维）的实验记录。
- `main.py` 和 `agent.py` 是空文件。分词器里 `<|tool_call_start|>` 这些 token 已经预留，但工具调用运行时和对话 CLI 还没写。
- 没有 midtrain / RL / 评测集（CORE、GSM8K 那一层）。

**下一步**：

1. 分步保存检查点（现在只在 epoch 末保存，而 1 个 epoch ≈ 35 万步，中途断电全丢）
2. bf16 autocast + 梯度累积（现在 fp32 + 单卡，6B token 跑不完）
3. 优化器参数分组（现在用 AdamW 默认值，对 RMSNorm 权重也施加了 weight decay）
4. 跑通第一次完整预训练

## 参考

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [karpathy/nanochat](https://github.com/karpathy/nanochat) —— 架构选型和对话模板设计主要参考它

## 关于数字

README 里所有性能数字都是在 Apple Silicon 单机上实测的，测量条件已在各处标注。特别说明两点：

- 「990 倍」的基线是**我自己最初的朴素实现**（整段文本反复全序列扫描），不是任何成熟库。
- 表格中 24274 merges 对应的「约 200×」是按 500 / 3000 两个实测点外推的估算值，其余数字均为直接实测。
