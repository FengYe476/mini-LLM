import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from math import sqrt

@dataclass
class GPTConfig:
    vocab_size: int
    embedding_dim: int
    block_size: int
    n_head: int
    n_layer: int
    dropout: float

def build_rope_cache(block_size: int, head_size: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    if head_size % 2 != 0:
        raise ValueError(f'[rope error]: the head_size ({head_size}) must be even')
    inv_freq = 1.0 / theta ** (torch.arange(0, head_size, 2).float() / head_size)
    angle = torch.outer(torch.arange(block_size).float(), inv_freq)
    angle = torch.cat([angle, angle], dim = -1)
    return angle.cos(), angle.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim = -1)
    return torch.cat([-x2, x1], dim = -1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin

class KVCache:
    def __init__(self, n_layer: int) -> None:
        self.k = [None] * n_layer
        self.v = [None] * n_layer
        self.pos = 0
        return

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.embedding_dim, cfg.embedding_dim * 4)
        self.up = nn.Linear(cfg.embedding_dim, cfg.embedding_dim * 4)
        self.fc2 = nn.Linear(cfg.embedding_dim * 4, cfg.embedding_dim)
        self.mlp_dropout = nn.Dropout(cfg.dropout)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate(x)) * self.up(x)
        x = self.fc2(x)
        x = self.mlp_dropout(x)
        return x

class MultiAttention(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        assert cfg.embedding_dim % cfg.n_head == 0, f'[vocab error]: the embedding_dim({cfg.embedding_dim}) cannot be divided by n_head ({cfg.n_head})'
        self.embedding_dim = cfg.embedding_dim
        self.block_size = cfg.block_size
        self.n_head = cfg.n_head
        self.head_size = cfg.embedding_dim // cfg.n_head
        self.layer_idx = layer_idx
        self.dropout = cfg.dropout
        cos, sin = build_rope_cache(self.block_size, self.head_size)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)

        self.qkv = nn.Linear(cfg.embedding_dim, cfg.embedding_dim * 3)
        self.proj = nn.Linear(cfg.embedding_dim, cfg.embedding_dim)
        self.atten_dropout = nn.Dropout(cfg.dropout)
        return

    def forward(self, x: torch.Tensor, cache: KVCache|None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        Q, K, V = qkv.split(
            self.embedding_dim,
            dim = -1
        )
        Q = Q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        K = K.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        V = V.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        pos = cache.pos if cache is not None else 0
        if pos + T > self.block_size:
            raise ValueError(f'[size error]: pos ({psos}) + T({T}) greater than block size')
        if cache is not None and pos > 0 and T != 1:
            raise ValueError(f'[T error]: T must be 1')
        cos, sin = self.cos[pos: pos + T], self.sin[pos: pos + T]
        Q, K = apply_rope(Q, cos, sin), apply_rope(K, cos, sin)
        if cache is not None:
            if cache.k[self.layer_idx] is not None:
                K = torch.cat([cache.k[self.layer_idx], K], dim = 2)
                V = torch.cat([cache.v[self.layer_idx], V], dim = 2)
            cache.k[self.layer_idx] = K
            cache.v[self.layer_idx] = V
        is_causal = (Q.size(2) == K.size(2))
        out = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, T, C)
        out = self.proj(out)
        out = self.atten_dropout(out)
        return out

class Block(nn.Module):
    def __init__(self, cfg:GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.embedding_dim, eps = 1e-6)
        self.attention = MultiAttention(cfg = cfg, layer_idx= layer_idx)
        self.norm2 = nn.RMSNorm(cfg.embedding_dim, eps = 1e-6)
        self.mlp = MLP(cfg = cfg)
        return

    def forward(self, x: torch.Tensor, cache: KVCache|None = None) -> torch.Tensor:
        x_norm = self.norm1(x)
        atten_out = self.attention(x_norm, cache)
        x = x + atten_out

        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x

class Embedding(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.emb_dropout(x)
        return x

class NanoGPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.embedding = Embedding(cfg = cfg)
        self.block = nn.ModuleList(
            Block(cfg = cfg, layer_idx=i)
            for i in range(cfg.n_layer)
        )
        self.norm_f = nn.RMSNorm(cfg.embedding_dim, eps = 1e-6)
        self.lm_head = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.token_embedding.weight
        self.apply(self._init_weights)
        residual_std = 0.02 / sqrt(2 * cfg.n_layer)
        for name, param in self.named_parameters():
            if name.endswith('fc2.weight') or name.endswith('proj.weight'):
                nn.init.normal_(param, mean = 0.0, std = residual_std)
        return

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
        return

    def forward(self, x: torch.Tensor, cache: KVCache|None = None) -> torch.Tensor:
        B, T = x.shape
        x = self.embedding(x)
        for block in self.block:
            x = block(x, cache)
        x = self.norm_f(x)
        x = self.lm_head(x)
        if cache is not None:
            cache.pos += T
        return x
