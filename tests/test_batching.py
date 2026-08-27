import torch

from llminf.batching import batched_generate, left_pad, make_ragged_prompts, throughput
from llminf.generate import generate
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
    """Identical prompts in a batch must each equal the unbatched generation of
    that same prompt on its own — not merely equal each other, which a bug that
    shifts every row by the same amount would still pass."""
    model = _model()
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 12))
    solo = generate(model, prompt, max_new_tokens=8, use_cache=True)[0]
    out = batched_generate(model, prompt, batch_size=3, max_new_tokens=8)
    for row in out:
        assert torch.equal(row, solo)


def test_throughput_reports_positive():
    model = _model()
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 12))
    r = throughput(model, prompt, batch_size=2, max_new_tokens=8)
    assert r["tokens_per_s"] > 0
    assert r["total_tokens"] == 16


def test_left_pad_places_real_tokens_flush_right():
    prompts = [torch.tensor([[1, 2, 3]]), torch.tensor([[7]]), torch.tensor([[4, 5]])]
    idx, mask = left_pad(prompts, pad_id=0)
    assert idx.shape == (3, 3) and mask.shape == (3, 3)
    assert torch.equal(idx[0], torch.tensor([1, 2, 3]))
    assert torch.equal(mask[0], torch.tensor([True, True, True]))
    assert torch.equal(idx[1], torch.tensor([0, 0, 7]))
    assert torch.equal(mask[1], torch.tensor([False, False, True]))
    assert torch.equal(idx[2], torch.tensor([0, 4, 5]))
    assert torch.equal(mask[2], torch.tensor([False, True, True]))
    # Every row's last column is a real token, whatever its own length was.
    assert mask[:, -1].all()


def test_make_ragged_prompts_spans_the_length_range():
    prompts = make_ragged_prompts(vocab_size=50, batch_size=4, min_len=3, max_len=9, seed=0)
    lengths = sorted(p.shape[-1] for p in prompts)
    assert lengths[0] == 3 and lengths[-1] == 9
    assert len(prompts) == 4


def test_ragged_batch_matches_each_prompt_generated_alone():
    """The acid test for padding + the attention mask: a genuinely ragged batch
    (different content, different lengths, real left-padding) must produce,
    for every row, exactly what that row's own prompt would generate alone —
    padding and the shorter neighbors in the batch must not leak into it."""
    model = _model()
    prompts = make_ragged_prompts(model.cfg.vocab_size, batch_size=5,
                                  min_len=3, max_len=14, seed=7)
    out = batched_generate(model, prompts, max_new_tokens=9)

    T_max = out.shape[1] - 9
    for i, prompt in enumerate(prompts):
        length = prompt.shape[-1]
        solo = generate(model, prompt, max_new_tokens=9, use_cache=True)[0]
        # This row's real tokens + everything generated after them.
        row_tail = out[i, T_max - length:]
        assert torch.equal(row_tail, solo)


def test_ragged_batch_is_order_independent():
    """Padding leaking across rows would show up as a batch-composition
    dependency: the same prompt's continuation changing depending on which
    other (differently-padded) rows share its batch."""
    model = _model()
    prompts = make_ragged_prompts(model.cfg.vocab_size, batch_size=4,
                                  min_len=4, max_len=13, seed=3)
    out_a = batched_generate(model, prompts, max_new_tokens=6)
    out_b = batched_generate(model, list(reversed(prompts)), max_new_tokens=6)
    assert torch.equal(out_a[0], out_b[-1])
    assert torch.equal(out_a[-1], out_b[0])


def test_ragged_throughput_reports_prompt_lengths():
    model = _model()
    prompts = make_ragged_prompts(model.cfg.vocab_size, batch_size=3,
                                  min_len=4, max_len=10, seed=1)
    r = throughput(model, prompts, max_new_tokens=6)
    assert r["batch_size"] == 3
    assert r["prompt_lengths"] == [p.shape[-1] for p in prompts]
    assert r["total_tokens"] == 18
    assert r["tokens_per_s"] > 0
