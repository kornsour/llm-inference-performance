import pytest
import torch

from llminf.bench import (
    BenchConfig,
    bench_kv_cache,
    bench_quantization,
    bench_rmsnorm,
    resident_kv_bytes,
    run_suite,
)
from llminf.model import GPT, GPTConfig

_HAS_ENGINE = len(torch.backends.quantized.supported_engines) > 0 and \
    torch.backends.quantized.supported_engines != ["none"]

# A non-CPU device the int8 model cannot consume — the same failure class as
# CUDA. MPS stands in for it on Apple Silicon; on a CUDA box, cuda itself.
_OFFLOAD = ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else None)


def _model(cfg: GPTConfig) -> GPT:
    torch.manual_seed(0)
    return GPT(cfg).eval()


def test_resident_kv_bytes_matches_the_cache_shapes():
    cfg = GPTConfig(n_layer=3, n_head=3, n_embd=96, block_size=128)
    model = _model(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 16))
    new_tokens = 24
    length = 16 + new_tokens - 1
    expected = 2 * cfg.n_layer * 1 * cfg.n_head * length * (cfg.n_embd // cfg.n_head) * 4
    assert resident_kv_bytes(model, prompt, new_tokens) == expected


def test_kv_cache_memory_is_not_reported_backwards():
    """The KV-cache buys speed by *holding* memory; it must never read as a saving."""
    cfg = GPTConfig(n_layer=3, n_head=3, n_embd=96, block_size=256)
    model = _model(cfg)
    prompt = torch.randint(0, cfg.vocab_size, (1, 32))
    r = bench_kv_cache(model, prompt, new_tokens=32, repeats=2, device=torch.device("cpu"))

    assert r["speedup_x"] > 1.0
    assert r["resident_kv_cache_mb"] > 0
    # Both variants do real work, so neither peak may collapse to ~0 the way the
    # second-measured one used to under an RSS delta.
    off = r["cache_off"]["peak_mem_mb"]
    on = r["cache_on"]["peak_mem_mb"]
    assert off > 0 and on > 0
    assert on > r["resident_kv_cache_mb"]  # the peak contains the cache it holds
    assert 0.1 < on / off < 10.0           # no 141x phantom saving


@pytest.mark.skipif(not _HAS_ENGINE, reason="no quantized engine on this platform")
@pytest.mark.skipif(_OFFLOAD is None, reason="no non-CPU device to test portability")
def test_quantization_bench_survives_a_non_cpu_device():
    """`--device cuda` used to feed CUDA inputs to a CPU-only int8 model."""
    cfg = GPTConfig.tiny()
    dev = torch.device(_OFFLOAD)
    model = _model(cfg).to(dev)
    idx = torch.randint(0, cfg.vocab_size, (1, 16)).to(dev)

    r = bench_quantization(model, idx, new_tokens=8)
    assert r["latency_device"] == "cpu"
    assert r["size_reduction_x"] > 1.0
    assert r["int8_latency_ms_median"] > 0
    # ...and it must leave the caller's model where it found it.
    assert next(model.parameters()).device.type == dev.type


@pytest.mark.skipif(not _HAS_ENGINE, reason="no quantized engine on this platform")
def test_quantization_bench_leaves_the_model_alone_on_cpu():
    cfg = GPTConfig.tiny()
    model = _model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    before = model.wte.weight.clone()
    bench_quantization(model, idx, new_tokens=8)
    assert torch.equal(model.wte.weight, before)
    assert isinstance(model.blocks[0].attn.c_attn, torch.nn.Linear)


def test_bench_config_defaults_are_sane():
    b = BenchConfig()
    assert b.repeats >= 1 and b.new_tokens > 0 and b.batch_sizes[0] == 1


def test_rmsnorm_bench_reports_a_row_per_shape():
    cfg = GPTConfig(n_layer=1, n_head=1, n_embd=16, block_size=32)
    shapes = ((1, 16), (8, 32))
    r = bench_rmsnorm(torch.device("cpu"), cfg, prompt_len=4,
                       repeats=2, inner=2, shapes=shapes)

    assert r["backend"] == "pytorch-reference"  # no CUDA kernel to dispatch to here
    assert len(r["rows"]) == len(shapes)
    for row, (rows, cols) in zip(r["rows"], shapes, strict=True):
        assert row["rows"] == rows and row["cols"] == cols
        assert row["unfused_latency_ms_p50"] >= 0
        assert row["fused_latency_ms_p50"] >= 0
        assert row["unfused_gbps"] is None or row["unfused_gbps"] > 0
        assert row["fused_gbps"] is None or row["fused_gbps"] > 0


def test_rmsnorm_bench_on_cpu_measures_the_same_code_path():
    """Without CUDA, `rmsnorm()` and `rmsnorm_reference()` are the same
    implementation, so this must report parity rather than a fake speedup."""
    cfg = GPTConfig.tiny()
    r = bench_rmsnorm(torch.device("cpu"), cfg, prompt_len=8,
                       repeats=3, inner=3, shapes=((64, 64),))
    assert r["rows"][0]["speedup_x"] == pytest.approx(1.0, rel=0.5)


def test_rmsnorm_bench_defaults_to_shapes_derived_from_the_model():
    cfg = GPTConfig(n_layer=1, n_head=1, n_embd=48, block_size=32)
    r = bench_rmsnorm(torch.device("cpu"), cfg, prompt_len=8, repeats=1, inner=1)
    cols_seen = {row["cols"] for row in r["rows"]}
    assert cfg.n_embd in cols_seen


@pytest.mark.skipif(not _HAS_ENGINE, reason="no quantized engine on this platform")
def test_quantization_latency_is_a_labeled_median_not_a_p50():
    """The two "p50-ish" latency figures in one report must stay distinguishable."""
    cfg = GPTConfig.tiny()
    model = _model(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    r = bench_quantization(model, idx, new_tokens=8)
    assert r["latency_repeats"] == 3
    assert "fp32_latency_ms_median" in r and "int8_latency_ms_median" in r
    # The old, ambiguous field names must not silently reappear.
    assert "fp32_latency_ms_p50" not in r and "int8_latency_ms_p50" not in r


def test_run_suite_env_is_self_describing():
    """`env` must be traceable on its own: no date, git SHA, or real CPU model
    was exactly the gap that made the committed artifact hard to corroborate."""
    cfg = GPTConfig.tiny()
    bcfg = BenchConfig(new_tokens=8, repeats=2, batch_sizes=(1, 2))
    env = run_suite(device="cpu", cfg=cfg, bcfg=bcfg)["env"]

    assert env["cpu_model"] and env["cpu_model"] != "cpu"
    assert env["git_sha"] and env["git_sha"] != "unknown"
    # ISO-8601 UTC timestamp.
    assert env["timestamp_utc"].endswith("+00:00")
    assert isinstance(env["argv"], list) and env["argv"]
