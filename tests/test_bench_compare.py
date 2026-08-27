import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bench_compare  # noqa: E402

from llminf import history  # noqa: E402
from llminf.bench import BenchConfig, run_suite  # noqa: E402
from llminf.model import GPTConfig  # noqa: E402


def _quick_report(**overrides) -> dict:
    """One real `run_suite()` call — timed, so it varies run to run.

    Used to seed exactly one report per test (usually the baseline). Variants
    that must compare deterministically against it go through `_overriding`
    instead of calling this again, so a slow/loaded test machine can't make
    the *other* (untouched) metrics drift far enough from an independently
    re-timed run to flip a "should not breach" assertion on their own.
    """
    cfg = GPTConfig.tiny()
    bcfg = BenchConfig(new_tokens=8, repeats=2, batch_sizes=(1, 2))
    report = run_suite(device="cpu", cfg=cfg, bcfg=bcfg)
    return _overriding(report, **overrides)


def _overriding(report: dict, **overrides) -> dict:
    """A deep copy of `report` with dotted-path fields replaced.

    e.g. `_overriding(report, **{"kv_cache.cache_on.tokens_per_s_mean": 1.0})`.
    """
    report = copy.deepcopy(report)
    for path, value in overrides.items():
        node = report
        *parents, leaf = path.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return report


class _Args:
    """Minimal stand-in for the argparse.Namespace bench_compare.py builds."""

    def __init__(self, **thresholds):
        self.max_throughput_drop_pct = thresholds.get("max_throughput_drop_pct", 20.0)
        self.max_p95_rise_pct = thresholds.get("max_p95_rise_pct", 20.0)
        self.max_mem_rise_pct = thresholds.get("max_mem_rise_pct", 20.0)
        self.max_speedup_drop_pct = thresholds.get("max_speedup_drop_pct", 20.0)


def test_compare_config_quick_is_smaller_and_more_repeats_than_full():
    quick_g, quick_b = bench_compare.compare_config(full=False)
    full_g, full_b = bench_compare.compare_config(full=True)
    assert quick_g.n_layer < full_g.n_layer
    assert quick_b.repeats > 2  # more stable than run_all.py's --quick smoke test


def test_no_baseline_is_not_a_breach(tmp_path):
    db_path = tmp_path / "history.duckdb"
    report = _quick_report()
    config = history.config_from_report(report)
    baseline = history.baseline_for(config, db_path=db_path)  # store doesn't exist yet
    assert baseline is None

    metrics = bench_compare.gated_metrics(_Args())
    table, breached = bench_compare.render_comparison(report, baseline, metrics)
    assert breached is False
    assert "n/a" in table


def test_matching_report_against_its_own_baseline_never_breaches(tmp_path):
    db_path = tmp_path / "history.duckdb"
    report = _quick_report()
    history.append_run(report, db_path=db_path, as_baseline=True)

    config = history.config_from_report(report)
    baseline = history.baseline_for(config, db_path=db_path)
    assert baseline is not None
    assert baseline.is_baseline is True

    metrics = bench_compare.gated_metrics(_Args())
    _, breached = bench_compare.render_comparison(report, baseline, metrics)
    assert breached is False


def test_a_large_throughput_drop_breaches_the_gate(tmp_path):
    db_path = tmp_path / "history.duckdb"
    baseline_report = _quick_report()
    history.append_run(baseline_report, db_path=db_path, as_baseline=True)
    config = history.config_from_report(baseline_report)
    baseline = history.baseline_for(config, db_path=db_path)

    slower = _overriding(baseline_report, **{"kv_cache.cache_on.tokens_per_s_mean":
                                              baseline.kv_cache_on_tokens_per_s * 0.5})

    metrics = bench_compare.gated_metrics(_Args(max_throughput_drop_pct=20.0))
    table, breached = bench_compare.render_comparison(slower, baseline, metrics)
    assert breached is True
    assert "FAIL" in table


def test_a_large_throughput_rise_does_not_breach_the_gate(tmp_path):
    """Only regressions gate — a run that got *faster* than baseline must pass."""
    db_path = tmp_path / "history.duckdb"
    baseline_report = _quick_report()
    history.append_run(baseline_report, db_path=db_path, as_baseline=True)
    config = history.config_from_report(baseline_report)
    baseline = history.baseline_for(config, db_path=db_path)

    faster = _overriding(baseline_report, **{"kv_cache.cache_on.tokens_per_s_mean":
                                              baseline.kv_cache_on_tokens_per_s * 2.0})

    metrics = bench_compare.gated_metrics(_Args(max_throughput_drop_pct=20.0))
    _, breached = bench_compare.render_comparison(faster, baseline, metrics)
    assert breached is False


def test_a_large_p95_rise_breaches_the_gate(tmp_path):
    db_path = tmp_path / "history.duckdb"
    baseline_report = _quick_report()
    history.append_run(baseline_report, db_path=db_path, as_baseline=True)
    config = history.config_from_report(baseline_report)
    baseline = history.baseline_for(config, db_path=db_path)

    slower = _overriding(baseline_report, **{"kv_cache.cache_on.latency.p95_ms":
                                              baseline.kv_cache_on_p95_ms * 3.0})

    metrics = bench_compare.gated_metrics(_Args(max_p95_rise_pct=20.0))
    table, breached = bench_compare.render_comparison(slower, baseline, metrics)
    assert breached is True
    assert "FAIL" in table


def test_a_large_peak_mem_rise_breaches_the_gate(tmp_path):
    db_path = tmp_path / "history.duckdb"
    baseline_report = _quick_report(**{"kv_cache.cache_on.peak_mem_mb": 2.0})
    history.append_run(baseline_report, db_path=db_path, as_baseline=True)
    config = history.config_from_report(baseline_report)
    baseline = history.baseline_for(config, db_path=db_path)
    assert baseline.kv_cache_on_peak_mem_mb == 2.0

    fatter = _overriding(baseline_report, **{"kv_cache.cache_on.peak_mem_mb": 10.0})

    metrics = bench_compare.gated_metrics(_Args(max_mem_rise_pct=20.0))
    table, breached = bench_compare.render_comparison(fatter, baseline, metrics)
    assert breached is True
    assert "FAIL" in table


def test_a_speedup_drop_breaches_the_gate(tmp_path):
    db_path = tmp_path / "history.duckdb"
    baseline_report = _quick_report()
    history.append_run(baseline_report, db_path=db_path, as_baseline=True)
    config = history.config_from_report(baseline_report)
    baseline = history.baseline_for(config, db_path=db_path)

    regressed = _overriding(baseline_report, **{"kv_cache.speedup_x": baseline.kv_cache_speedup_x * 0.3})

    metrics = bench_compare.gated_metrics(_Args(max_speedup_drop_pct=20.0))
    table, breached = bench_compare.render_comparison(regressed, baseline, metrics)
    assert breached is True
    assert "FAIL" in table


def test_ungated_metrics_never_breach_regardless_of_delta(tmp_path):
    """batching/quant/rmsnorm are reported but not gated (see the module docstring)."""
    db_path = tmp_path / "history.duckdb"
    baseline_report = _quick_report()
    history.append_run(baseline_report, db_path=db_path, as_baseline=True)
    config = history.config_from_report(baseline_report)
    baseline = history.baseline_for(config, db_path=db_path)

    collapsed = _overriding(baseline_report, **{"quantization.size_reduction_x": 0.01})

    metrics = bench_compare.gated_metrics(_Args())
    _, breached = bench_compare.render_comparison(collapsed, baseline, metrics)
    assert breached is False


def test_update_baseline_flips_is_baseline_and_is_readable_back(tmp_path):
    db_path = tmp_path / "history.duckdb"
    report = _quick_report()
    run_id = history.append_run(report, db_path=db_path, as_baseline=True)

    config = history.config_from_report(report)
    baseline = history.baseline_for(config, db_path=db_path)
    assert baseline is not None
    assert baseline.run_id == run_id
    assert baseline.is_baseline is True


def test_update_baseline_moves_the_flag_not_duplicates_it(tmp_path):
    """A second --update-baseline for the same config must leave exactly one
    baseline row, on the newest run — not two, and not the old one."""
    db_path = tmp_path / "history.duckdb"
    report = _quick_report()
    first_id = history.append_run(report, db_path=db_path, as_baseline=True)
    second_id = history.append_run(report, db_path=db_path, as_baseline=True)
    assert second_id != first_id

    config = history.config_from_report(report)
    rows = history.matching_runs(config, db_path=db_path, limit=100)
    baseline_rows = [r for r in rows if r.is_baseline]
    assert len(baseline_rows) == 1
    assert baseline_rows[0].run_id == second_id
