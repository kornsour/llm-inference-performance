import torch

from llminf.model import GPT, GPTConfig, param_bytes
from llminf.rmsnorm import RMSNorm


def test_forward_shapes():
    cfg = GPTConfig.tiny()
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, past = model(idx, use_cache=False)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert past is None


def test_cache_returns_present():
    cfg = GPTConfig.tiny()
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    _, past = model(idx, use_cache=True)
    assert past is not None and len(past) == cfg.n_layer
    k, v = past[0]
    assert k.shape[2] == 8 and v.shape[2] == 8  # cached sequence length


def test_param_count_and_bytes():
    model = GPT(GPTConfig.tiny()).eval()
    assert model.num_params() > 0
    assert param_bytes(model) == sum(p.numel() * 4 for p in model.parameters())


def test_norm_layers_are_fused_rmsnorm():
    """The kernel isn't worth benchmarking if the model never calls it."""
    model = GPT(GPTConfig.tiny()).eval()
    assert isinstance(model.ln_f, RMSNorm)
    for block in model.blocks:
        assert isinstance(block.ln_1, RMSNorm)
        assert isinstance(block.ln_2, RMSNorm)


@torch.no_grad()
def test_all_ones_attention_mask_matches_no_mask():
    """A batch with no padding must be numerically identical whether or not an
    (all-real) attention_mask is passed — the mask path is a strict superset of
    the unmasked one, not a different computation."""
    cfg = GPTConfig.tiny()
    model = GPT(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (3, 10))
    mask = torch.ones(3, 10, dtype=torch.bool)
    logits_plain, _ = model(idx, use_cache=False)
    logits_masked, _ = model(idx, use_cache=False, attention_mask=mask)
    assert torch.allclose(logits_plain, logits_masked, atol=1e-5)


@torch.no_grad()
def test_left_padding_does_not_change_the_real_tokens_output():
    """A padded row's logits at its real-token positions must match the same
    row decoded alone, unpadded — left-padding must be invisible to the tokens
    that follow it, not just avoid crashing/NaN-ing."""
    cfg = GPTConfig.tiny()
    model = GPT(cfg).eval()
    torch.manual_seed(1)
    real = torch.randint(0, cfg.vocab_size, (1, 6))

    solo_logits, _ = model(real, use_cache=False)

    pad = torch.zeros(1, 4, dtype=real.dtype)
    padded = torch.cat([pad, real], dim=1)
    mask = torch.cat([torch.zeros(1, 4, dtype=torch.bool),
                      torch.ones(1, 6, dtype=torch.bool)], dim=1)
    padded_logits, _ = model(padded, use_cache=False, attention_mask=mask)

    assert not torch.isnan(padded_logits).any()
    assert torch.allclose(solo_logits, padded_logits[:, 4:], atol=1e-5)
