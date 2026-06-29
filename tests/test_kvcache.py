import torch

from llminf.generate import generate, timed_generate
from llminf.model import GPT, GPTConfig


def _model():
    torch.manual_seed(0)
    return GPT(GPTConfig(n_layer=3, n_head=3, n_embd=96, block_size=128)).eval()


def test_cache_and_no_cache_are_identical():
    """The whole point: KV-cache changes speed, never output."""
    model = _model()
    idx = torch.randint(0, model.cfg.vocab_size, (2, 20))
    out_off = generate(model, idx, max_new_tokens=40, use_cache=False)
    out_on = generate(model, idx, max_new_tokens=40, use_cache=True)
    assert torch.equal(out_off, out_on)


def test_timed_generate_reports_breakdown():
    model = _model()
    idx = torch.randint(0, model.cfg.vocab_size, (1, 16))
    r = timed_generate(model, idx, max_new_tokens=24, use_cache=True)
    assert r["new_tokens"] == 24
    assert len(r["decode_ms"]) == 23
    assert r["tokens_per_s"] > 0
    assert r["ttft_ms"] >= 0
