# Gating CI on a performance comparison

CI builds and tests every PR, but a performance change that keeps every test
green used to land silently — the harness reports the metrics an inference
team tracks (`tokens_per_s_mean`, `p95_ms`, `peak_mem_mb`, the KV-cache
speedup ratio), but nothing consumed them as a gate. `make bench-compare`
(`scripts/bench_compare.py`) is that gate.

## What it does

1. Runs its own small, stable benchmark configuration (`compare_config()` in
   `scripts/bench_compare.py` — the same tiny model as `run_all.py --quick`,
   but `repeats=8` instead of 2, so the p95 it gates on isn't close to
   "whichever of two samples happened to be slowest").
2. Looks up the row flagged `is_baseline` in
   [`benchmarks/results/history.duckdb`](../benchmarks/results/history.duckdb)
   (see [`llminf/history.py`](../src/llminf/history.py)) for the *same
   configuration key* — device, model shape, prompt/token counts, torch/Python
   version. Comparing across configurations is meaningless, so a
   configuration with no baseline yet is reported, not gated (exit 0).
3. Compares four KV-cache-section metrics against that baseline and fails
   (exit 1) if any regresses past its threshold:

   | metric | direction that fails | default threshold (`quick`) |
   | --- | --- | --- |
   | `tokens_per_s_mean` (cache-on) | falls | 10% below baseline |
   | `p95_ms` (cache-on) | rises | 15% above baseline |
   | `peak_mem_mb` (cache-on) | rises | 20% above baseline |
   | `speedup_x` (cache-on / cache-off) | falls | 12% below baseline |

   Batching speedup, int8 size reduction, and fused-RMSNorm speedup are
   reported in the same table for visibility (a PR should show its
   performance effect whether or not it trips the gate) but are **not**
   gated — no noise-band measurement backs a threshold for them yet, and
   each already has its own multi-row shape that a single scalar delta would
   flatten. Extending the gate to them is future work, not deferred by
   accident.

4. Prints the comparison as a markdown table and, in CI, appends it to
   `$GITHUB_STEP_SUMMARY` so it shows up on the workflow run regardless of
   whether the gate passed.

CI runs `python scripts/bench_compare.py` (the `quick` configuration) as a
step in the `test` job, on the Python 3.14 leg only — see `.github/workflows/ci.yml`.
The 3.12 leg doesn't get its own baseline (`python_version` is part of the
configuration key), so running the comparison there would just print "no
baseline yet" on every PR; not worth the extra suite run.

## Why the gate is scoped to the KV-cache section

The issue this closes named exactly these four metrics
(`tokens_per_s_mean`, `p95_ms`, `peak_mem_mb`, "the KV-cache speedup ratio").
They also happen to be the metrics `run_all.py --quick`'s tiny model measures
most stably: batching's throughput scaling and RMSNorm's achieved-GB/s rows
depend on shapes (batch size, hidden width) that move around more between a
noisy CI runner's samples than a single scalar tokens/sec number does. Rather
than gate on numbers with no calibration behind their thresholds, this ships
the well-calibrated core and reports the rest.

## Moving the baseline

The baseline is never picked implicitly — not "the first row", not "the most
recent row". `append_run(..., as_baseline=True)` is the one code path that
sets or moves it, and it is only ever called deliberately:

```bash
make bench-baseline-update                        # quick config (what CI gates on)
ARGS=--full make bench-baseline-update            # full config
# equivalently: uv run python scripts/bench_compare.py --update-baseline [--full]
git add benchmarks/results/history.duckdb
git commit -m "chore(bench): move the bench-compare baseline — <why>"
```

Do this after a measured, *accepted* trade-off — a correctness fix that costs
some throughput, a memory-for-speed swap, a deliberately slower but simpler
implementation — not to make a regression disappear without discussion.

Because the `quick` baseline is what CI actually gates on, it should be
recorded on the same class of machine CI runs on, not approximated from a
laptop: `.github/workflows/bench-baseline.yml` runs
`scripts/bench_compare.py --update-baseline` on a GitHub Actions
`ubuntu-latest` runner and pushes the result back to the branch you dispatch
it against:

```bash
gh workflow run bench-baseline.yml --ref <branch> -f config=quick
```

## How the default thresholds were set

`scripts/bench_noise.py` (`make bench-noise`) runs `bench_compare.py`'s own
configuration N times in a single process and reports the spread (min / mean
/ max / stdev, and the largest deviation from the mean as a percentage) for
each of the four gated metrics — the number a threshold has to clear to not
fail on noise alone, before a regression has to add anything on top of it.

The measurement backing this repo's defaults was taken on a GitHub Actions
`ubuntu-latest` CPU runner — a shared, virtualized machine, noisier than a
quiet laptop and the actual environment `make bench-compare` gates in CI.
`bench-noise.yml`/`bench-baseline.yml` can't be `workflow_dispatch`ed until
they exist on `main` (a GitHub Actions restriction — the workflow file has to
be registered on the default branch first), so for this repo's *first*
calibration and baseline the equivalent commands ran as a one-off job inside
[PR #22](https://github.com/kornsour/llm-inference-performance/pull/22)'s own
CI instead ([run
33120556658](https://github.com/kornsour/llm-inference-performance/actions/runs/33120556658),
job `bootstrap-baseline`, 2026-08-27, `ubuntu-latest`, torch `2.13.0+cu130`,
Python 3.14.7). From here on, re-calibrating is:

```bash
gh workflow run bench-noise.yml --ref <branch> -f repeats=10 -f config=quick
```

Ten repeats of `bench_compare.py`'s `quick` configuration, n=10, measured:

```
tokens_per_s_mean (cache-on)     min       1519  mean       1529  max       1540  stdev     6.404  max |dev| from mean:   0.7%
p95_ms (cache-on)                min      21.06  mean      21.31  max      22.07  stdev    0.2812  max |dev| from mean:   3.6%
peak_mem_mb (cache-on)           min        0.3  mean        0.3  max        0.3  stdev         0  max |dev| from mean:   0.0%
speedup_x                        min       1.42  mean      1.434  max       1.46  stdev   0.01265  max |dev| from mean:   1.8%
```

`peak_mem_mb`'s 0.0% isn't the metric being perfectly stable — it's this
tiny model's KV-cache footprint rounding to a fixed 0.3 MB at the precision
`bench.py` reports, so the real spread underneath is invisible at n=10. The
`--max-mem-rise-pct` default is set wider than the others for exactly that
reason: a threshold this metric's own noise sample can't actually justify
needs the extra margin.

Thresholds in `scripts/bench_compare.py` are set well above the observed
spread — roughly 4-15x the largest deviation from the mean, tighter where
the signal was clean (throughput, speedup) and wider where rounding could be
hiding real movement (peak mem) — not right at it: a gate that fails on
noise half the time trains people to ignore it, which is worse than not
having one. If the runner class changes (GitHub changes the `ubuntu-latest`
image, the model/workload shape in `compare_config()` changes) re-run
`bench-noise.yml` and revisit the `DEFAULT_MAX_*_PCT` constants at the top of
`scripts/bench_compare.py`.

## `full`-configuration thresholds

`--full` (`make bench-compare-full`) gates the same four metrics on the full
`make bench` model/workload instead of the fast one — useful for a local
before/after check on a dev machine, after recording a local baseline with
`ARGS=--full make bench-baseline-update`. Its default thresholds
(`FULL_MAX_*_PCT` in `scripts/bench_compare.py`) are wider and have **not**
been separately calibrated with `bench_noise.py --full` — a bigger model on
whatever machine happens to run `make bench` varies more, and this path isn't
what CI gates on. Treat `--full`'s gate as a coarse local sanity check, not a
tuned threshold; recalibrate with `ARGS=--full make bench-noise` before
relying on it for anything stricter.
