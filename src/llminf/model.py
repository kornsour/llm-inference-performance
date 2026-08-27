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

from .rmsnorm import RMSNorm

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

    def forward(self, x: torch.Tensor, past_kv: KVCache | None = None,
                attn_bias: torch.Tensor | None = None) -> tuple[torch.Tensor, KVCache]:
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

        if attn_bias is not None:
            # `attn_bias` already encodes causality *and* which keys are real vs
            # padding (see `GPT._build_attn_bias`), so no separate `is_causal`.
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias.to(q.dtype))
        else:
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
        self.ln_1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, past_kv: KVCache | None = None,
                attn_bias: torch.Tensor | None = None) -> tuple[torch.Tensor, KVCache]:
        attn_out, present = self.attn(self.ln_1(x), past_kv, attn_bias)
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
        self.ln_f = RMSNorm(cfg.n_embd)
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
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[KVCache] | None]:
        """Run a prefill or single-step decode.

        `attention_mask` is `(B, past_len + T)`, `True`/`1` for a real token and
        `False`/`0` for left-padding — the *whole* key sequence seen so far, not
        just the `T` new tokens. Pass it whenever prompts in the batch were
        left-padded to a common length (see `batching.py`); positions and
        attention both account for the padding, so a shorter prompt in the same
        batch produces exactly the output it would alone. Omit it for a batch
        of same-length, unpadded sequences — the cheaper `is_causal` path below
        is unaffected and behaves exactly as before.
        """
        B, T = idx.shape
        past_len = past_kvs[0][0].shape[2] if past_kvs is not None else 0

        if attention_mask is not None:
            # Position of a real token = count of real tokens at or before it,
            # minus one; padding gets position 0 (unused — those queries' outputs
            # are never read, and no key ever attends to a padded query/key).
            pos = attention_mask.long().cumsum(dim=1) - 1
            pos = pos.clamp(min=0)[:, past_len:past_len + T]
            attn_bias = self._build_attn_bias(attention_mask, past_len, T, idx.device)
        else:
            pos = torch.arange(past_len, past_len + T, device=idx.device)
            attn_bias = None

        x = self.wte(idx) + self.wpe(pos)

        presents: list[KVCache] = []
        for i, block in enumerate(self.blocks):
            past = past_kvs[i] if past_kvs is not None else None
            x, present = block(x, past, attn_bias)
            presents.append(present)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, (presents if use_cache else None)

    @staticmethod
    def _build_attn_bias(attention_mask: torch.Tensor, past_len: int, T: int,
                         device: torch.device) -> torch.Tensor:
        """Additive `(B, 1, T, S)` bias: causal among the `T` new queries, and
        `-inf` for any key position `attention_mask` marks as padding."""
        S = past_len + T
        key_valid = attention_mask.to(device=device, dtype=torch.bool)  # (B, S)
        q_pos = torch.arange(past_len, S, device=device).unsqueeze(1)   # (T, 1)
        k_pos = torch.arange(S, device=device).unsqueeze(0)             # (1, S)
        causal = k_pos <= q_pos                                         # (T, S)
        self_diag = k_pos == q_pos                                      # (T, S)
        # A left-padded query is itself padding for as long as its own row is
        # shorter than the batch max, so `causal & key_valid` can be all-False
        # for it (every key it's allowed to see is also padding) — softmax over
        # an all `-inf` row is NaN. Its output is never read (see `pos` above),
        # so force the diagonal open to keep it a normal, ignorable row instead.
        allowed = (causal.unsqueeze(0) & key_valid.unsqueeze(1)) | self_diag.unsqueeze(0)
        bias = torch.zeros(allowed.shape, dtype=torch.float32, device=device)
        bias = bias.masked_fill(~allowed, float("-inf"))
        return bias.unsqueeze(1)  # (B, 1, T, S), broadcasts over heads

    def num_params(self) -> int:
        # Subtract position embeddings as nanoGPT does for the reported count.
        n = sum(p.numel() for p in self.parameters())
        return n - self.wpe.weight.numel()


def param_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())
