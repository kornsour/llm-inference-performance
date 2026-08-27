"""Print recent benchmark history for the current configuration.

    python scripts/bench_history.py            # last 10 runs
    python scripts/bench_history.py -n 25       # last 25 runs

Reads `benchmarks/results/latest.json` to determine "the current
configuration" (device, device name, torch/Python version, model shape,
benchmark config — everything a fair comparison requires to match), then
queries `benchmarks/results/history.duckdb` for the last N runs sharing that
configuration and prints, per optimization, one row per run plus the spread
(min/mean/max/stdev) across those runs — so a headline ratio like "KV-cache:
4.8x faster" reads as a distribution, not a single sample.

`make bench` appends to the history store on every full run; this script only
reads it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llminf import history  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LATEST_JSON = ROOT / "benchmarks" / "results" / "latest.json"


def _spread(values: list[float | None]) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.3g} (n=1, no spread yet)"
    mean = statistics.mean(vals)
    stdev = statistics.stdev(vals)
    return (f"min {min(vals):.3g} · mean {mean:.3g} · max {max(vals):.3g} "
            f"· stdev {stdev:.3g} (n={len(vals)})")


def _print_section(title: str, rows: list[history.RunRow], row_fmt, values) -> None:
    print(f"\n## {title}\n")
    for r in rows:
        print(f"  {r.git_sha:>8}  {r.timestamp_utc}  {row_fmt(r)}")
    print(f"  spread: {_spread(values(rows))}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--limit", type=int, default=10,
                     help="number of most recent matching runs to show (default: 10)")
    ap.add_argument("--db", type=Path, default=history.DEFAULT_DB_PATH,
                     help="path to the history store (default: benchmarks/results/history.duckdb)")
    args = ap.parse_args(argv)

    if not LATEST_JSON.exists():
        print(f"error: {LATEST_JSON.relative_to(ROOT)} not found — run `make bench` first "
              "to establish the current configuration.", file=sys.stderr)
        return 1

    latest = json.loads(LATEST_JSON.read_text())
    config = history.config_from_report(latest)
    rows = history.matching_runs(config, db_path=args.db, limit=args.limit)

    if not rows:
        print(f"No history yet for the current configuration in "
              f"{args.db.relative_to(ROOT) if args.db.is_relative_to(ROOT) else args.db}.")
        print("Run `make bench` (it appends to the store automatically) to start one.")
        return 0

    m = latest["model"]
    c = latest["config"]
    print(f"Current configuration: {config['device']} ({config['device_name']}), "
          f"torch {config['torch_version']}, python {config['python_version']}, "
          f"{m['params_m']}M params ({m['n_layer']}L/{m['n_head']}H/{m['n_embd']}d), "
          f"prompt {c['prompt_len']} -> {c['new_tokens']} new tokens, {c['repeats']} repeats.")
    print(f"Showing the last {len(rows)} matching run(s), newest first "
          f"(requested {args.limit}).")

    _print_section(
        "KV-cache speedup (tokens/sec, cache-on / cache-off)", rows,
        lambda r: (f"{r.kv_cache_speedup_x:.2f}x  "
                   f"({r.kv_cache_off_tokens_per_s:.1f} -> {r.kv_cache_on_tokens_per_s:.1f} tok/s, "
                   f"p50 {r.kv_cache_off_p50_ms:.1f}ms -> {r.kv_cache_on_p50_ms:.1f}ms)"),
        lambda rs: [r.kv_cache_speedup_x for r in rs],
    )

    _print_section(
        "Batching (best speedup vs. batch size 1)", rows,
        lambda r: f"{r.batching_best_speedup_x:.2f}x at batch {r.batching_best_batch_size}",
        lambda rs: [r.batching_best_speedup_x for r in rs],
    )

    _print_section(
        "int8 quantization (fp32 -> int8 size reduction)", rows,
        lambda r: (f"{r.quant_size_reduction_x:.2f}x smaller, "
                   f"logit MSE {r.quant_logit_mse:.6g}"),
        lambda rs: [r.quant_size_reduction_x for r in rs],
    )

    _print_section(
        "Fused RMSNorm (best speedup across measured shapes)", rows,
        lambda r: f"{r.rmsnorm_best_speedup_x:.2f}x  [{r.rmsnorm_backend}]",
        lambda rs: [r.rmsnorm_best_speedup_x for r in rs],
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
