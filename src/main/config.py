from pathlib import Path
from dataclasses import dataclass

from tokenizer import SPECIAL_TOKENS

DATA_DIR = Path('data')

@dataclass(frozen = True)
class Paths:
    data_dir: Path = DATA_DIR
    @property
    def corpus(self) -> Path:
        return self.data_dir/'tinyshakespeare.txt'

    @property
    def shard_dir(self) -> Path:
        return self.data_dir/'shards'
    
    @property
    def shard_meta(self) -> Path:
        return self.shard_dir/'meta.json'

    @property
    def tok(self) -> Path:
        return self.data_dir/'tok.json'

    @property
    def pretrain_ckpt(self) -> Path:
        return self.data_dir/'pretrain_checkpoint.pt'

    @property
    def sft_data(self) -> Path:
        return self.data_dir/'sft_toy.jsonl'

    @property
    def sft_ckpt(self) -> Path:
        return self.data_dir/'checkpoint.pt'

@dataclass(frozen = True)
class TokenizerConfig:
    train_chars: int = 10_000_000
    n_merges: int = 24274
    n_segments: int = 200
    sample_seed: int = 0
    oversample: int = 10
    domains: tuple[tuple[str, float], ...] = (
        ('code_python', 0.40),
        ('code_issues', 0.20),
        ('code_shell', 0.10),
        ('terminal_docs', 0.05),
        ('web_edu', 0.10),
        ('web_dclm', 0.06),
        ('cosmopedia', 0.04),
        ('qa_stackexchange', 0.05),
    )

    @property
    def weights(self) -> dict[str, float]:
        return dict(self.domains)

    @property
    def vocab_size(self) -> int:
        return 256 + self.n_merges + len(SPECIAL_TOKENS)

@dataclass(frozen = True)
class ModelConfig:
    block_size: int = 1024
    embedding_dim: int = 768
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.0

@dataclass(frozen = True)
class PretrainConfig:
    batch_size: int = 16
    epoches_per_run: int = 1
    lr: float = 1e-3
    eval_every: int = 200
    total_tokens: int = 6_000_000_000
    warmup_ratio: float = 0.01
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    shard_tokens: int = 50_000_000
    val_tokens: int = 2_000_000

    def schedule(self, block_size: int) -> tuple[int, int]:
        total_steps = max(2, self.total_tokens // (self.batch_size * block_size))
        warmup_steps = max(1, min(int(total_steps * self.warmup_ratio), total_steps - 1))
        return total_steps, warmup_steps

@dataclass(frozen = True)
class SFTConfig:
    batch_size: int = 2
    epoches_per_run: int = 50
    lr: float = 3e-4
    eval_every: int = 10
    warmup_steps: int = 10
    total_steps: int = 150
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0

@dataclass(frozen = True)
class CorpusConfig:
    corpus_dir: Path = DATA_DIR/'corpus'
    target_tokens: int = 6_000_000_000
    domains: tuple[tuple[str, float], ...] = (
        ('web_edu', 0.20),
        ('web_dclm', 0.15),
        ('cosmopedia', 0.12),
        ('code_python', 0.20),
        ('code_shell', 0.05),
        ('code_issues', 0.10),
        ('qa_stackexchange', 0.15),
        ('terminal_docs', 0.03),
    )

    @property
    def weights(self) -> dict[str, float]:
        return dict(self.domains)
    

PATHS = Paths()
MODEL = ModelConfig()
SFT = SFTConfig()
PRETRAIN = PretrainConfig()
TOKENIZER = TokenizerConfig()
CORPUS = CorpusConfig()


