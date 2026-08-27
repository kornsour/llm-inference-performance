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
    report_json VARCHAR
)
"""

_INSERT = """
INSERT INTO runs (
    timestamp_utc, git_sha, device, device_name, torch_version, python_version,
    n_layer, n_head, n_embd, params_m, prompt_len, new_tokens, repeats,
    kv_cache_off_tokens_per_s, kv_cache_on_tokens_per_s,
    kv_cache_off_p50_ms, kv_cache_off_p95_ms, kv_cache_on_p50_ms, kv_cache_on_p95_ms,
    kv_cache_speedup_x, resident_kv_cache_mb,
    batching_best_speedup_x, batching_best_batch_size,
    quant_size_reduction_x, quant_fp32_latency_ms_median, quant_int8_latency_ms_median,
    quant_logit_mse, rmsnorm_backend, rmsnorm_best_speedup_x, report_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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


_COLUMNS = tuple(RunRow.__dataclass_fields__)  # table column order == field order


def _ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_CREATE_SEQUENCE)
    con.execute(_CREATE_TABLE)


def _best_batching_row(report: dict) -> tuple[float | None, int | None]:
    rows = report["batching"]["rows"]
    if not rows:
        return None, None
    best = max(rows, key=lambda r: r["speedup_vs_b1"] or 0.0)
    return best["speedup_vs_b1"], best["batch_size"]


def _best_rmsnorm_speedup(report: dict) -> float | None:
    rows = report["rmsnorm"]["rows"]
    if not rows:
        return None
    return max(r["speedup_x"] for r in rows if r["speedup_x"] is not None)


def append_run(report: dict, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Append one `run_suite()` report as a row in the DuckDB history store.

    Idempotent in the sense that matters: every call adds a new row (this is
    an append-only log of measurements, not a keyed upsert), so calling it
    twice for the same commit records two samples rather than one — which is
    exactly what lets `bench-history` show run-to-run spread at a fixed
    commit as well as drift across commits.
    """
    env, model, cfg = report["env"], report["model"], report["config"]
    kv = report["kv_cache"]
    q = report["quantization"]
    rn = report["rmsnorm"]
    batching_speedup, batching_batch_size = _best_batching_row(report)

    values = (
        env["timestamp_utc"], env["git_sha"], env["device"], env["device_name"],
        env["torch"], env["python"],
        model["n_layer"], model["n_head"], model["n_embd"], model["params_m"],
        cfg["prompt_len"], cfg["new_tokens"], cfg["repeats"],
        kv["cache_off"]["tokens_per_s_mean"], kv["cache_on"]["tokens_per_s_mean"],
        kv["cache_off"]["latency"]["p50_ms"], kv["cache_off"]["latency"]["p95_ms"],
        kv["cache_on"]["latency"]["p50_ms"], kv["cache_on"]["latency"]["p95_ms"],
        kv["speedup_x"], kv["resident_kv_cache_mb"],
        batching_speedup, batching_batch_size,
        q["size_reduction_x"], q["fp32_latency_ms_median"], q["int8_latency_ms_median"],
        q["logit_mse"], rn["backend"], _best_rmsnorm_speedup(report),
        json.dumps(report),
    )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        _ensure_schema(con)
        con.execute(_INSERT, values)


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
    with duckdb.connect(str(db_path), read_only=True) as con:
        rows = con.execute(sql, [*params, limit]).fetchall()
    # `recorded_at` comes back as a `datetime.datetime`; stringify it so every
    # `RunRow` field is the plain str/int/float its annotation promises.
    return [RunRow(*row[:1], str(row[1]), *row[2:]) for row in rows]


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
