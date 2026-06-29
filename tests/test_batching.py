import torch

from llminf.batching import batched_generate, throughput
from llminf.model import GPT, GPTConfig


def _model():
    torch.manual_seed(0)
    return GPT(GPTConfig.tiny()).eval()


def test_batched_generate_shape():
    model = _model()
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 12))
    out = batched_generate(model, prompt, batch_size=4, max_new_tokens=10)
    assert out.shape == (4, 22)


def test_batched_rows_match_single():
    """Identical prompts in a batch yield identical greedy continuations."""
    model = _model()
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 12))
    out = batched_generate(model, prompt, batch_size=3, max_new_tokens=8)
    assert torch.equal(out[0], out[1]) and torch.equal(out[1], out[2])


def test_throughput_reports_positive():
    model = _model()
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 12))
    r = throughput(model, prompt, batch_size=2, max_new_tokens=8)
    assert r["tokens_per_s"] > 0
    assert r["total_tokens"] == 16
