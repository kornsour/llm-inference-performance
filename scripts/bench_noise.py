"""Measure run-to-run noise for `bench_compare.py`'s own configuration.

    python scripts/bench_noise.py                # 10 repeats, quick config
    python scripts/bench_noise.py -n 20 --full    # 20 repeats, full config

Runs `scripts/bench_compare.py`'s `quick` (or `--full`) configuration N times
in a single process and reports min/mean/max/stdev for the four metrics
`bench_compare.py` gates on, plus the largest observed deviation from the
mean as a percentage — the number a threshold has to clear to not fail on
noise alone.

This is how `bench_compare.py`'s default `--max-*-pct` thresholds were set:
run this on the same class of machine `make bench-compare` will gate on (a
GitHub Actions `ubuntu-latest` CPU runner, via
`.github/workflows/bench-baseline.yml` — `gh workflow run bench-baseline.yml
--ref <branch>` runs it and posts the numbers to the workflow's job summary),
then pick thresholds comfortably above the observed spread. See
`docs/bench-compare.md` for the measurement this repo's defaults are based on
and for how to redo it if the runner class or the configuration changes.

Purely informational — this never fails/exits non-zero and never touches
`benchmarks/results/history.duckdb`.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bench_compare import compare_config  # noqa: E402

from llminf.bench import run_suite  # noqa: E402

METRICS: tuple[tuple[str, str], ...] = (
    ("tokens_per_s_mean (cache-on)", "tokens_per_s"),
    ("p95_ms (cache-on)", "p95_ms"),
    ("peak_mem_mb (cache-on)", "peak_mem_mb"),
    ("speedup_x", "speedup_x"),
)


def _sample(device: str, full: bool) -> dict[str, float]:
    gcfg, bcfg = compare_config(full)
    report = run_suite(device=device, cfg=gcfg, bcfg=bcfg)
    kv = report["kv_cache"]
    return {
        "tokens_per_s": kv["cache_on"]["tokens_per_s_mean"],
        "p95_ms": kv["cache_on"]["latency"]["p95_ms"],
        "peak_mem_mb": kv["cache_on"]["peak_mem_mb"],
        "speedup_x": kv["speedup_x"],
    }


def _spread_line(label: str, values: list[float]) -> str:
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    max_dev_pct = max(abs(v - mean) for v in values) / mean * 100 if mean else 0.0
    return (f"  {label:<32} min {min(values):>10.4g}  mean {mean:>10.4g}  "
            f"max {max(values):>10.4g}  stdev {stdev:>9.4g}  "
            f"max |dev| from mean: {max_dev_pct:5.1f}%")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-n", "--repeats", type=int, default=10,
                     help="number of full suite runs to sample (default: 10)")
    ap.add_argument("--full", action="store_true",
                     help="use bench_compare.py's `--full` configuration instead of `quick`")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    label = "full" if args.full else "quick"
    print(f"Sampling the `{label}` bench-compare configuration {args.repeats} times "
          f"on device={args.device}...\n", file=sys.stderr)

    samples: dict[str, list[float]] = {key: [] for _, key in METRICS}
    for i in range(args.repeats):
        s = _sample(args.device, args.full)
        for _, key in METRICS:
            samples[key].append(s[key])
        print(f"  run {i + 1}/{args.repeats}: " +
              ", ".join(f"{key}={s[key]:.4g}" for _, key in METRICS), file=sys.stderr)

    lines = [f"## Bench-compare noise (`{label}`, n={args.repeats})", ""]
    for label_, key in METRICS:
        lines.append(_spread_line(label_, samples[key]))
    body = "\n".join(lines)
    print("\n" + body)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("\n" + body.replace("  min", "min").replace("\n  ", "\n- ") + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
