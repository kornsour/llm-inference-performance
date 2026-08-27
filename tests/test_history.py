import copy

from llminf import history
from llminf.bench import BenchConfig, run_suite
from llminf.model import GPTConfig


def _report() -> dict:
    cfg = GPTConfig.tiny()
    bcfg = BenchConfig(new_tokens=8, repeats=2, batch_sizes=(1, 2))
    return run_suite(device="cpu", cfg=cfg, bcfg=bcfg)


def test_append_run_creates_the_store_and_a_row(tmp_path):
    db_path = tmp_path / "history.duckdb"
    assert not db_path.exists()

    history.append_run(_report(), db_path=db_path)

    assert db_path.exists()
    # `_report()` is deterministic (fixed seeds), so the config key from a
    # freshly-built report matches the one just appended.
    rows = history.matching_runs(history.config_from_report(_report()), db_path=db_path)
    assert len(rows) == 1
    row = rows[0]
    assert row.kv_cache_speedup_x is not None
    assert row.git_sha and row.git_sha != "unknown"


def test_append_run_is_additive_not_overwriting(tmp_path):
    """Unlike latest.json, two runs must both survive as separate rows."""
    db_path = tmp_path / "history.duckdb"
    report = _report()

    history.append_run(report, db_path=db_path)
    history.append_run(report, db_path=db_path)

    rows = history.matching_runs(history.config_from_report(report), db_path=db_path, limit=100)
    assert len(rows) == 2
    # Newest first.
    assert rows[0].run_id > rows[1].run_id


def test_matching_runs_only_returns_the_matching_configuration(tmp_path):
    db_path = tmp_path / "history.duckdb"
    report = _report()
    history.append_run(report, db_path=db_path)

    other = copy.deepcopy(report)
    other["config"]["new_tokens"] = 999  # a different workload shape
    history.append_run(other, db_path=db_path)

    rows = history.matching_runs(history.config_from_report(report), db_path=db_path, limit=100)
    assert len(rows) == 1
    assert rows[0].new_tokens == report["config"]["new_tokens"]


def test_matching_runs_respects_the_limit(tmp_path):
    db_path = tmp_path / "history.duckdb"
    report = _report()
    for _ in range(5):
        history.append_run(report, db_path=db_path)

    rows = history.matching_runs(history.config_from_report(report), db_path=db_path, limit=3)
    assert len(rows) == 3


def test_matching_runs_on_a_missing_store_returns_empty(tmp_path):
    db_path = tmp_path / "does-not-exist.duckdb"
    rows = history.matching_runs(history.config_from_report(_report()), db_path=db_path)
    assert rows == []


def test_git_sha_is_not_part_of_the_matching_configuration(tmp_path):
    """Two runs of different commits, same everything else, must both surface —
    that is what makes a regression between commits visible."""
    db_path = tmp_path / "history.duckdb"
    report = _report()
    other_commit = copy.deepcopy(report)
    other_commit["env"]["git_sha"] = "deadbee"

    history.append_run(report, db_path=db_path)
    history.append_run(other_commit, db_path=db_path)

    rows = history.matching_runs(history.config_from_report(report), db_path=db_path, limit=100)
    assert len(rows) == 2
    assert {r.git_sha for r in rows} == {report["env"]["git_sha"], "deadbee"}


def test_report_json_round_trips(tmp_path):
    import json

    db_path = tmp_path / "history.duckdb"
    report = _report()
    history.append_run(report, db_path=db_path)

    rows = history.matching_runs(history.config_from_report(report), db_path=db_path)
    stored = json.loads(rows[0].report_json)
    assert stored["kv_cache"]["speedup_x"] == report["kv_cache"]["speedup_x"]
