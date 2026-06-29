import torch

from llminf.model import GPT, GPTConfig, param_bytes


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
