"""Append-only history of benchmark runs, in a local DuckDB file.

`benchmarks/run_all.py` used to write a single `benchmarks/results/latest.json`
that every run overwrote — so the repo held exactly one measurement at a time.
A regression was invisible (the slower number just replaced the faster one),
and a headline ratio like "KV-cache: 4.8x faster" was a single sample from a
single machine with no record of how much it moves run to run.

This module appends every full (non-`--quick`) run's report to
`benchmarks/results/history.duckdb` instead of discarding it. DuckDB is a
single embedded file — no server, ships as a pure-Python wheel, and is happy
reading/writing the same file that `run_all.py` produces. `latest.json` is
left exactly as it was: this store is additive, not a replacement.

Each row is keyed by the dimensions that make two runs comparable: git commit
(which run produced these numbers), device, device name, torch/Python
version, model shape (n_layer/n_head/n_embd/params_m), and benchmark config
(prompt_len/new_tokens/repeats). That is a genuinely high-cardinality key —
comparisons across rows are only meaningful when every column but git_sha
(and the metrics themselves) matches; see `matching_runs` and
`scripts/bench_history.py`.

At most one row per configuration may be flagged `is_baseline` — the number
`scripts/bench_compare.py` (`make bench-compare`) gates CI against. It is
never picked implicitly (e.g. "the first row" or "the most recent row"):
`append_run(..., as_baseline=True)` — `make bench-baseline-update` — is the
one documented way to set or move it, and doing so is a deliberate act, not
something a normal `make bench` run does as a side effect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb

DEFAULT_DB_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "results" / "history.duckdb"
)

# Columns that define a "configuration" — two rows are comparable only when
# all of these match. `git_sha` is deliberately excluded: it is the axis a
# regression shows up on, so filtering by it would hide the very thing this
# store exists to reveal.
CONFIG_COLUMNS = (
    "device", "device_name", "torch_version", "python_version",
    "n_layer", "n_head", "n_embd", "params_m",
    "prompt_len", "new_tokens", "repeats",
)

_CREATE_SEQUENCE = "CREATE SEQUENCE IF NOT EXISTS run_id_seq START 1"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id BIGINT PRIMARY KEY DEFAULT nextval('run_id_seq'),
    recorded_at TIMESTAMP DEFAULT current_timestamp,
    timestamp_utc VARCHAR,
    git_sha VARCHAR,
    device VARCHAR,
    device_name VARCHAR,
    torch_version VARCHAR,
    python_version VARCHAR,
    n_layer INTEGER,
    n_head INTEGER,
    n_embd INTEGER,
    params_m DOUBLE,
    prompt_len INTEGER,
    new_tokens INTEGER,
    repeats INTEGER,
    kv_cache_off_tokens_per_s DOUBLE,
    kv_cache_on_tokens_per_s DOUBLE,
    kv_cache_off_p50_ms DOUBLE,
    kv_cache_off_p95_ms DOUBLE,
    kv_cache_on_p50_ms DOUBLE,
    kv_cache_on_p95_ms DOUBLE,
    kv_cache_off_peak_mem_mb DOUBLE,
    kv_cache_on_peak_mem_mb DOUBLE,
    kv_cache_speedup_x DOUBLE,
    resident_kv_cache_mb DOUBLE,
    batching_best_speedup_x DOUBLE,
    batching_best_batch_size INTEGER,
    quant_size_reduction_x DOUBLE,
    quant_fp32_latency_ms_median DOUBLE,
    quant_int8_latency_ms_median DOUBLE,
    quant_logit_mse DOUBLE,
    rmsnorm_backend VARCHAR,
    rmsnorm_best_speedup_x DOUBLE,
    report_json VARCHAR,
    is_baseline BOOLEAN NOT NULL DEFAULT FALSE
)
"""

# Columns added after the table first shipped (#20). `_ensure_schema` adds
# these to a pre-existing `runs` table via `ALTER TABLE ... ADD COLUMN`, so an
# already-committed `history.duckdb` (rows recorded before this migration
# existed) keeps every old row — they just read back with NULL for the
# columns they predate, rather than needing a destructive rebuild. DuckDB's
# `ADD COLUMN` doesn't accept `NOT NULL`/`DEFAULT` constraints (unlike
# `CREATE TABLE`, which does — see `_CREATE_TABLE`), so `is_baseline` is
# backfilled to `FALSE` for old rows in a follow-up `UPDATE` instead.
_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("kv_cache_off_peak_mem_mb", "DOUBLE"),
    ("kv_cache_on_peak_mem_mb", "DOUBLE"),
    ("is_baseline", "BOOLEAN"),
)

_INSERT = """
INSERT INTO runs (
    timestamp_utc, git_sha, device, device_name, torch_version, python_version,
    n_layer, n_head, n_embd, params_m, prompt_len, new_tokens, repeats,
    kv_cache_off_tokens_per_s, kv_cache_on_tokens_per_s,
    kv_cache_off_p50_ms, kv_cache_off_p95_ms, kv_cache_on_p50_ms, kv_cache_on_p95_ms,
    kv_cache_off_peak_mem_mb, kv_cache_on_peak_mem_mb,
    kv_cache_speedup_x, resident_kv_cache_mb,
    batching_best_speedup_x, batching_best_batch_size,
    quant_size_reduction_x, quant_fp32_latency_ms_median, quant_int8_latency_ms_median,
    quant_logit_mse, rmsnorm_backend, rmsnorm_best_speedup_x, report_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
RETURNING run_id
"""


@dataclass(frozen=True)
class RunRow:
    """One `runs` row, as a typed view over what `bench_history.py` prints.

    Mirrors the `runs` table columns rather than the nested `report` dict —
    callers that want the full report can still parse `report_json`.
    """

    run_id: int
    recorded_at: str
    timestamp_utc: str
    git_sha: str
    device: str
    device_name: str
    torch_version: str
    python_version: str
    n_layer: int
    n_head: int
    n_embd: int
    params_m: float
    prompt_len: int
    new_tokens: int
    repeats: int
    kv_cache_off_tokens_per_s: float
    kv_cache_on_tokens_per_s: float
    kv_cache_off_p50_ms: float
    kv_cache_off_p95_ms: float
    kv_cache_on_p50_ms: float
    kv_cache_on_p95_ms: float
    kv_cache_off_peak_mem_mb: float
    kv_cache_on_peak_mem_mb: float
    kv_cache_speedup_x: float
    resident_kv_cache_mb: float
    batching_best_speedup_x: float
    batching_best_batch_size: int
    quant_size_reduction_x: float
    quant_fp32_latency_ms_median: float
    quant_int8_latency_ms_median: float
    quant_logit_mse: float
    rmsnorm_backend: str
    rmsnorm_best_speedup_x: float
    report_json: str
    is_baseline: bool


_COLUMNS = tuple(RunRow.__dataclass_fields__)  # table column order == field order


def _ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_CREATE_SEQUENCE)
    con.execute(_CREATE_TABLE)
    existing = {row[1] for row in con.execute("PRAGMA table_info('runs')").fetchall()}
    for name, ddl_type in _MIGRATIONS:
        if name not in existing:
            con.execute(f"ALTER TABLE runs ADD COLUMN {name} {ddl_type}")
    if "is_baseline" not in existing:
        # `ADD COLUMN` alone leaves no default, so every insert on a migrated
        # table would otherwise write NULL forever, not just for the rows
        # that predate the column.
        con.execute("ALTER TABLE runs ALTER COLUMN is_baseline SET DEFAULT FALSE")
        con.execute("UPDATE runs SET is_baseline = FALSE WHERE is_baseline IS NULL")


def best_batching_row(report: dict) -> tuple[float | None, int | None]:
    rows = report["batching"]["rows"]
    if not rows:
        return None, None
    best = max(rows, key=lambda r: r["speedup_vs_b1"] or 0.0)
    return best["speedup_vs_b1"], best["batch_size"]


def best_rmsnorm_speedup(report: dict) -> float | None:
    rows = report["rmsnorm"]["rows"]
    if not rows:
        return None
    return max(r["speedup_x"] for r in rows if r["speedup_x"] is not None)


def append_run(report: dict, db_path: Path = DEFAULT_DB_PATH, as_baseline: bool = False) -> int:
    """Append one `run_suite()` report as a row in the DuckDB history store.

    Idempotent in the sense that matters: every call adds a new row (this is
    an append-only log of measurements, not a keyed upsert), so calling it
    twice for the same commit records two samples rather than one — which is
    exactly what lets `bench-history` show run-to-run spread at a fixed
    commit as well as drift across commits.

    `as_baseline=True` additionally (in the same connection) clears
    `is_baseline` on every other row sharing this run's `CONFIG_COLUMNS` and
    sets it on the new row — see `baseline_for` and `scripts/bench_compare.py`
    (`make bench-baseline-update` is the CLI for this). Returns the new row's
    `run_id`.
    """
    env, model, cfg = report["env"], report["model"], report["config"]
    kv = report["kv_cache"]
    q = report["quantization"]
    rn = report["rmsnorm"]
    batching_speedup, batching_batch_size = best_batching_row(report)

    values = (
        env["timestamp_utc"], env["git_sha"], env["device"], env["device_name"],
        env["torch"], env["python"],
        model["n_layer"], model["n_head"], model["n_embd"], model["params_m"],
        cfg["prompt_len"], cfg["new_tokens"], cfg["repeats"],
        kv["cache_off"]["tokens_per_s_mean"], kv["cache_on"]["tokens_per_s_mean"],
        kv["cache_off"]["latency"]["p50_ms"], kv["cache_off"]["latency"]["p95_ms"],
        kv["cache_on"]["latency"]["p50_ms"], kv["cache_on"]["latency"]["p95_ms"],
        kv["cache_off"]["peak_mem_mb"], kv["cache_on"]["peak_mem_mb"],
        kv["speedup_x"], kv["resident_kv_cache_mb"],
        batching_speedup, batching_batch_size,
        q["size_reduction_x"], q["fp32_latency_ms_median"], q["int8_latency_ms_median"],
        q["logit_mse"], rn["backend"], best_rmsnorm_speedup(report),
        json.dumps(report),
    )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        _ensure_schema(con)
        run_id = con.execute(_INSERT, values).fetchone()[0]
        if as_baseline:
            where = " AND ".join(f"{c} = ?" for c in CONFIG_COLUMNS)
            config = config_from_report(report)
            params = [config[c] for c in CONFIG_COLUMNS]
            con.execute(f"UPDATE runs SET is_baseline = FALSE WHERE {where}", params)
            con.execute("UPDATE runs SET is_baseline = TRUE WHERE run_id = ?", [run_id])
    return run_id


def matching_runs(config: dict, db_path: Path = DEFAULT_DB_PATH, limit: int = 10) -> list[RunRow]:
    """The last `limit` rows whose `CONFIG_COLUMNS` match `config`, newest first.

    `config` takes the same key names as `RunRow`/the table columns (see
    `config_from_report`) — typically the current `latest.json`'s env/model/
    config, so "the current configuration" means "whatever `make bench` last
    measured on this device with this model and workload shape".
    """
    if not db_path.exists():
        return []
    where = " AND ".join(f"{c} = ?" for c in CONFIG_COLUMNS)
    params = [config[c] for c in CONFIG_COLUMNS]
    cols = ", ".join(_COLUMNS)
    sql = (
        f"SELECT {cols} FROM runs WHERE {where} "
        f"ORDER BY recorded_at DESC, run_id DESC LIMIT ?"
    )
    # Not `read_only=True`: a `history.duckdb` committed before a schema
    # migration (see `_MIGRATIONS`) needs `_ensure_schema` to run here too, or
    # every read against it fails with "column not found" until something
    # happens to open it for a write first.
    with duckdb.connect(str(db_path)) as con:
        _ensure_schema(con)
        rows = con.execute(sql, [*params, limit]).fetchall()
    # `recorded_at` comes back as a `datetime.datetime`; stringify it so every
    # `RunRow` field is the plain str/int/float its annotation promises.
    return [RunRow(*row[:1], str(row[1]), *row[2:]) for row in rows]


def baseline_for(config: dict, db_path: Path = DEFAULT_DB_PATH) -> RunRow | None:
    """The row flagged `is_baseline` for `config`, or `None` if this exact
    configuration has never had one set.

    `None` is the expected, non-error answer for any configuration nobody has
    run `make bench-baseline-update` for yet (a new device, a changed model
    shape, CI's first run of a workload it hasn't gated before) — callers
    like `scripts/bench_compare.py` treat it as "nothing to gate against",
    not a failure.
    """
    if not db_path.exists():
        return None
    where = " AND ".join(f"{c} = ?" for c in CONFIG_COLUMNS)
    params = [config[c] for c in CONFIG_COLUMNS]
    cols = ", ".join(_COLUMNS)
    sql = (
        f"SELECT {cols} FROM runs WHERE is_baseline AND {where} "
        f"ORDER BY recorded_at DESC, run_id DESC LIMIT 1"
    )
    with duckdb.connect(str(db_path)) as con:  # see `matching_runs` on read_only
        _ensure_schema(con)
        row = con.execute(sql, params).fetchone()
    if row is None:
        return None
    return RunRow(*row[:1], str(row[1]), *row[2:])


def config_from_report(report: dict) -> dict:
    """Extract the `CONFIG_COLUMNS` key from a `run_suite()` report (or
    `latest.json`'s parsed content, same shape)."""
    env, model, cfg = report["env"], report["model"], report["config"]
    return {
        "device": env["device"],
        "device_name": env["device_name"],
        "torch_version": env["torch"],
        "python_version": env["python"],
        "n_layer": model["n_layer"],
        "n_head": model["n_head"],
        "n_embd": model["n_embd"],
        "params_m": model["params_m"],
        "prompt_len": cfg["prompt_len"],
        "new_tokens": cfg["new_tokens"],
        "repeats": cfg["repeats"],
    }
