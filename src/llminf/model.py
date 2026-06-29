"""A compact GPT-style decoder with first-class KV-cache support.

Deliberately small and readable (nanoGPT lineage). The attention block accepts a
past key/value and returns the updated cache, which is what makes the
KV-cache-on vs -off A/B in `generate.py` an apples-to-apples comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

KVCache = tuple[torch.Tensor, torch.Tensor]


@dataclass
class GPTConfig:
    vocab_size: int = 256          # byte-level
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = True

    @staticmethod
    def small() -> GPTConfig:
        return GPTConfig()

    @staticmethod
    def tiny() -> GPTConfig:
        return GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=128)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: torch.Tensor, past_kv: KVCache | None = None
                ) -> tuple[torch.Tensor, KVCache]:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        present = (k, v)

        # T > 1 => prefill (causal among the new tokens, aligned to the end of k).
        # T == 1 => single-token decode: attend to all cached keys, no mask needed.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), present


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, past_kv: KVCache | None = None
                ) -> tuple[torch.Tensor, KVCache]:
        attn_out, present = self.attn(self.ln_1(x), past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init)

    def _init(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        past_kvs: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[KVCache] | None]:
        B, T = idx.shape
        past_len = past_kvs[0][0].shape[2] if past_kvs is not None else 0
        pos = torch.arange(past_len, past_len + T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)

        presents: list[KVCache] = []
        for i, block in enumerate(self.blocks):
            past = past_kvs[i] if past_kvs is not None else None
            x, present = block(x, past)
            presents.append(present)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, (presents if use_cache else None)

    def num_params(self) -> int:
        # Subtract position embeddings as nanoGPT does for the reported count.
        n = sum(p.numel() for p in self.parameters())
        return n - self.wpe.weight.numel()


def param_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())
