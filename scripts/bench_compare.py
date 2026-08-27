"""Gate a benchmark run against its stored baseline.

    python scripts/bench_compare.py                   # quick config (CI default)
    python scripts/bench_compare.py --full             # full config, like `make bench`
    python scripts/bench_compare.py --update-baseline  # record this run as the new baseline

CI builds and tests on every PR, but a performance change that keeps every
test green lands silently — nothing consumes the harness's own metrics
(`tokens_per_s_mean`, `p95_ms`, `peak_mem_mb`) as a gate. This script is that
gate: it runs the benchmark suite (`llminf.bench.run_suite`) and compares four
of the KV-cache section's numbers — the harness's headline optimized-path
metrics — against the row flagged `is_baseline` in
`benchmarks/results/history.duckdb` for the *same configuration key* (device,
model shape, prompt/token counts, torch/Python version — see
`llminf.history.CONFIG_COLUMNS`; comparing across configurations is
meaningless, so a mismatch is reported, not gated):

  - `tokens_per_s_mean` (cache-on) must not fall more than
    `--max-throughput-drop-pct` below baseline
  - `p95_ms` (cache-on) must not rise more than `--max-p95-rise-pct` above
    baseline
  - `peak_mem_mb` (cache-on) must not rise more than `--max-mem-rise-pct`
    above baseline
  - `speedup_x` (the KV-cache win itself, cache-on / cache-off) must not fall
    more than `--max-speedup-drop-pct` below baseline

Batching, quantization, and fused-RMSNorm deltas are reported in the same
table for visibility but are not gated — see `docs/bench-compare.md` for why
the gate is scoped to these four.

Exits non-zero (fails the CI job) only when a baseline exists for this exact
configuration *and* one of the four gated metrics is breached. No baseline
yet for this configuration -> prints a notice and exits 0: there is nothing
to compare against, which is expected the first time a device/model/workload
combination is gated (see `--update-baseline` below).

Default thresholds come from the noise band measured on a GitHub Actions
`ubuntu-latest` CPU runner (`scripts/bench_noise.py`, n=10) for exactly the
`quick` configuration below — see `docs/bench-compare.md` for the measurement
run and the numbers behind them. They are wider for `--full` runs, which have
not been separately calibrated and vary more (also a bigger model, on
whatever machine happens to run `make bench`).

`--update-baseline` records the freshly-run report as the new baseline for
its configuration instead of gating — the one documented way to move the
baseline, e.g. after a measured slowdown that is a deliberate trade-off (a
correctness fix, a memory-for-speed swap). It is a plain flag, not something
that ever happens as a side effect of a normal run:

    python scripts/bench_compare.py --update-baseline           # quick config
    python scripts/bench_compare.py --full --update-baseline    # full config

Then commit the updated `benchmarks/results/history.duckdb` like any other
benchmark artifact. `.github/workflows/bench-baseline.yml`
(`gh workflow run bench-baseline.yml --ref <branch>`) runs the same command on
a GitHub Actions `ubuntu-latest` runner and pushes the result, so the
`quick`-configuration baseline that CI gates against can be re-recorded on the
exact hardware/OS CI itself measures on, not approximated from a laptop.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llminf import history  # noqa: E402
from llminf.bench import BenchConfig, run_suite  # noqa: E402
from llminf.history import RunRow  # noqa: E402
from llminf.model import GPTConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# See docs/bench-compare.md for the full measurement this is based on.
# scripts/bench_noise.py's in-process n=10 sample (one runner, ten
# back-to-back repeats) showed tight noise: max deviation from the mean 0.7%
# (tokens/sec), 3.6% (p95), 0.0% (peak mem — this tiny model's KV-cache
# footprint rounds to a fixed 0.3 MB), 1.8% (speedup_x). But three
# independent CI *job* runs of the identical code/config told a different
# story: two landed within ~1-3% of each other, one differed from both by
# ~28-30% on raw tokens/sec and p95 — GitHub's `ubuntu-latest` label spans
# more than one physical host generation, and which one a job lands on moves
# raw throughput by more than ten in-process repeats on a single host ever
# will. `speedup_x` (cache-on / cache-off, both measured in the same job)
# stayed much closer across all three (2.8-11.2%) — it's a same-run ratio,
# so a host-speed difference mostly cancels out of it. Thresholds below are
# set above the observed *cross-job* spread, not the tighter in-process one:
# comfortable margin over ~30% for the two raw metrics, tighter for the
# ratio and for peak-mem (stable in every sample taken). `full` thresholds
# are wider still — not separately calibrated, and a bigger model on
# whatever machine happens to run `make bench` moves more between samples.
DEFAULT_MAX_THROUGHPUT_DROP_PCT = 45.0
DEFAULT_MAX_P95_RISE_PCT = 45.0
DEFAULT_MAX_MEM_RISE_PCT = 25.0
DEFAULT_MAX_SPEEDUP_DROP_PCT = 25.0
FULL_MAX_THROUGHPUT_DROP_PCT = 55.0
FULL_MAX_P95_RISE_PCT = 55.0
FULL_MAX_MEM_RISE_PCT = 35.0
FULL_MAX_SPEEDUP_DROP_PCT = 35.0


def compare_config(full: bool) -> tuple[GPTConfig, BenchConfig]:
    """The (model, benchmark) configuration `bench_compare.py` measures.

    Deliberately its own configuration, not `run_all.py --quick`'s: that one
    is tuned to be a fast *smoke test* (`repeats=2`), which makes for a very
    shaky p95 to gate on. This one keeps the same tiny model (fast: a whole
    run is a couple of seconds) but takes more samples (`repeats=8`) so the
    p95 this script gates on is not close to "whichever sample happened to be
    slowest of two".
    """
    if full:
        return GPTConfig.small(), BenchConfig()
    return GPTConfig.tiny(), BenchConfig(new_tokens=32, repeats=8, batch_sizes=(1, 2, 4))


class Metric:
    """One row of the comparison table: how to read it off a fresh `report`
    and off a baseline `RunRow`, which direction of movement is bad, and (if
    gated) how many percent of that movement is tolerated."""

    def __init__(
        self, key: str, label: str,
        current: Callable[[dict], float | None],
        baseline: Callable[[RunRow], float | None],
        worse: str,  # "lower" or "higher"
        threshold_pct: float | None,
    ) -> None:
        assert worse in ("lower", "higher")
        self.key, self.label = key, label
        self.current, self.baseline = current, baseline
        self.worse, self.threshold_pct = worse, threshold_pct


def _best_batching_speedup(report: dict) -> float | None:
    return history.best_batching_row(report)[0]


def _best_rmsnorm_speedup(report: dict) -> float | None:
    return history.best_rmsnorm_speedup(report)


def gated_metrics(args: argparse.Namespace) -> list[Metric]:
    return [
        Metric(
            "kv_tokens_per_s", "KV-cache tokens/sec (cache-on)",
            lambda r: r["kv_cache"]["cache_on"]["tokens_per_s_mean"],
            lambda b: b.kv_cache_on_tokens_per_s,
            "lower", args.max_throughput_drop_pct,
        ),
        Metric(
            "kv_p95_ms", "KV-cache p95 latency, ms (cache-on)",
            lambda r: r["kv_cache"]["cache_on"]["latency"]["p95_ms"],
            lambda b: b.kv_cache_on_p95_ms,
            "higher", args.max_p95_rise_pct,
        ),
        Metric(
            "kv_peak_mem_mb", "KV-cache peak mem, MB (cache-on)",
            lambda r: r["kv_cache"]["cache_on"]["peak_mem_mb"],
            lambda b: b.kv_cache_on_peak_mem_mb,
            "higher", args.max_mem_rise_pct,
        ),
        Metric(
            "kv_speedup_x", "KV-cache speedup (cache-on / cache-off)",
            lambda r: r["kv_cache"]["speedup_x"],
            lambda b: b.kv_cache_speedup_x,
            "lower", args.max_speedup_drop_pct,
        ),
        # Reported for visibility (the issue this script closes asks for a
        # PR to show its performance effect "whether or not it trips the
        # gate") but not gated: no noise-band measurement backs a threshold
        # for these yet, and batching/quant in particular have their own
        # multi-row shapes a single scalar diff undersells.
        Metric(
            "batching_speedup_x", "Best batching speedup vs. batch=1",
            _best_batching_speedup, lambda b: b.batching_best_speedup_x,
            "lower", None,
        ),
        Metric(
            "quant_size_reduction_x", "int8 size reduction",
            lambda r: r["quantization"]["size_reduction_x"],
            lambda b: b.quant_size_reduction_x,
            "lower", None,
        ),
        Metric(
            "rmsnorm_speedup_x", "Best fused-RMSNorm speedup",
            _best_rmsnorm_speedup, lambda b: b.rmsnorm_best_speedup_x,
            "lower", None,
        ),
    ]


def _pct_delta(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0 if current == 0 else float("inf")
    return (current - baseline) / baseline * 100.0


def _breach(metric: Metric, current: float | None, baseline: float | None) -> bool:
    if metric.threshold_pct is None or current is None or baseline is None:
        return False
    delta_pct = _pct_delta(current, baseline)
    if metric.worse == "lower":
        return delta_pct < -metric.threshold_pct
    return delta_pct > metric.threshold_pct


def render_comparison(
    report: dict, baseline: RunRow | None, metrics: list[Metric],
) -> tuple[str, bool]:
    """Markdown comparison table + whether any gated metric breached."""
    lines = ["| metric | baseline | current | delta | gate |", "| --- | --- | --- | --- | --- |"]
    breached = False
    for m in metrics:
        current = m.current(report)
        base = m.baseline(baseline) if baseline is not None else None
        if base is None or current is None:
            lines.append(f"| {m.label} | n/a | {current if current is not None else 'n/a'} | — | — |")
            continue
        delta_pct = _pct_delta(current, base)
        is_breach = _breach(m, current, base)
        breached = breached or is_breach
        gate = "—" if m.threshold_pct is None else (
            f"FAIL (limit {m.threshold_pct:g}%)" if is_breach else "ok"
        )
        lines.append(
            f"| {m.label} | {base:.4g} | {current:.4g} | {delta_pct:+.1f}% | {gate} |"
        )
    return "\n".join(lines), breached


def _write_step_summary(heading: str, table_md: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a") as f:
        f.write(f"\n## {heading}\n\n{table_md}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--full", action="store_true",
                     help="use the full `make bench` model/workload instead of the fast "
                          "`quick` one this script defaults to")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--db", type=Path, default=history.DEFAULT_DB_PATH,
                     help="path to the history store (default: benchmarks/results/history.duckdb)")
    ap.add_argument("--update-baseline", action="store_true",
                     help="record this run as the new baseline for its configuration, "
                          "instead of gating")
    ap.add_argument("--max-throughput-drop-pct", type=float, default=None)
    ap.add_argument("--max-p95-rise-pct", type=float, default=None)
    ap.add_argument("--max-mem-rise-pct", type=float, default=None)
    ap.add_argument("--max-speedup-drop-pct", type=float, default=None)
    ap.add_argument("--no-summary", action="store_true",
                     help="skip writing to $GITHUB_STEP_SUMMARY even if it is set "
                          "(mainly for local runs against a real CI checkout)")
    args = ap.parse_args(argv)

    if args.max_throughput_drop_pct is None:
        args.max_throughput_drop_pct = (
            FULL_MAX_THROUGHPUT_DROP_PCT if args.full else DEFAULT_MAX_THROUGHPUT_DROP_PCT)
    if args.max_p95_rise_pct is None:
        args.max_p95_rise_pct = FULL_MAX_P95_RISE_PCT if args.full else DEFAULT_MAX_P95_RISE_PCT
    if args.max_mem_rise_pct is None:
        args.max_mem_rise_pct = FULL_MAX_MEM_RISE_PCT if args.full else DEFAULT_MAX_MEM_RISE_PCT
    if args.max_speedup_drop_pct is None:
        args.max_speedup_drop_pct = (
            FULL_MAX_SPEEDUP_DROP_PCT if args.full else DEFAULT_MAX_SPEEDUP_DROP_PCT)

    gcfg, bcfg = compare_config(args.full)
    report = run_suite(device=args.device, cfg=gcfg, bcfg=bcfg)
    config = history.config_from_report(report)
    label = "full" if args.full else "quick"

    if args.update_baseline:
        run_id = history.append_run(report, db_path=args.db, as_baseline=True)
        print(f"Recorded run_id={run_id} as the new `{label}` baseline for this configuration:")
        for c in history.CONFIG_COLUMNS:
            print(f"  {c}: {config[c]}")
        print(f"\nCommit the updated {args.db} to make this the baseline `make bench-compare` "
              f"gates against.")
        return 0

    baseline = history.baseline_for(config, db_path=args.db)
    metrics = gated_metrics(args)
    table_md, breached = render_comparison(report, baseline, metrics)

    print(f"Configuration ({label}):")
    for c in history.CONFIG_COLUMNS:
        print(f"  {c}: {config[c]}")
    print()

    if baseline is None:
        print(f"No `{label}` baseline recorded for this exact configuration in {args.db} — "
              f"nothing to gate against. Run `make bench-baseline-update` "
              f"(or `--update-baseline{' --full' if args.full else ''}`) to establish one.")
        print()
        print(table_md)
        if not args.no_summary:
            _write_step_summary(
                f"Benchmark comparison ({label}) — no baseline yet, not gated", table_md)
        return 0

    print(f"Comparing against baseline run_id={baseline.run_id} "
          f"({baseline.git_sha}, {baseline.timestamp_utc}):")
    print()
    print(table_md)
    if not args.no_summary:
        heading = f"Benchmark comparison ({label}) vs. baseline run_id={baseline.run_id}"
        if breached:
            heading += " — REGRESSION"
        _write_step_summary(heading, table_md)

    if breached:
        print("\nOne or more gated metrics regressed past their threshold — see the table above.")
        return 1
    print("\nAll gated metrics within threshold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
