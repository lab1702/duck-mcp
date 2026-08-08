"""The five tools, exercised against real csv / parquet / ndjson files."""

from __future__ import annotations

import os

import pytest

from duckdb_mcp.config import HARD_MAX_ROWS
from duckdb_mcp.db import ReadOnlyViolation
from duckdb_mcp.tools import (
    ToolError,
    describe_file,
    list_files,
    preview_file,
    profile_columns,
    run_query,
)


@pytest.fixture(params=["sales.csv", "sales.parquet"])
def sales(request, data_dir):
    return (data_dir / request.param).as_posix()


def body_rows(rendered: str) -> int:
    """Table lines minus the header and the divider."""
    return sum(1 for line in rendered.splitlines() if line.startswith("| ")) - 2


def row_for(rendered: str, column: str) -> str:
    return next(line for line in rendered.splitlines() if line.startswith(f"| {column} "))


def test_query_reads_a_file(session, settings, sales):
    out = run_query(session, settings, f"SELECT region, sum(amount) AS total FROM '{sales}' GROUP BY 1 ORDER BY 1")
    assert "| region | total |" in out
    assert "East" in out and "560.5" in out


def test_query_rejects_writes(session, settings):
    with pytest.raises(ReadOnlyViolation):
        run_query(session, settings, "CREATE TABLE t AS SELECT 1")


def test_query_caps_rows_and_says_so(session, settings):
    out = run_query(session, settings, "SELECT * FROM range(100)", max_rows=5)
    assert body_rows(out) == 5
    assert "more available" in out


def test_row_cap_never_exceeds_the_hard_maximum():
    from duckdb_mcp.tools import _clamp_rows

    assert _clamp_rows(None, 500) == 500
    assert _clamp_rows(1_000_000, 500) == HARD_MAX_ROWS
    assert _clamp_rows(None, 1_000_000) == HARD_MAX_ROWS  # the server default too
    assert _clamp_rows(0, 500) == 1


def test_query_without_cap_note_when_all_rows_fit(session, settings):
    out = run_query(session, settings, "SELECT * FROM range(3)", max_rows=10)
    assert "more available" not in out


def test_query_supports_explain(session, settings):
    # EXPLAIN cannot be wrapped in a subquery; it runs unwrapped instead.
    out = run_query(session, settings, "EXPLAIN SELECT 1")
    assert "explain_value" in out
    assert body_rows(out) >= 1


def test_query_rejects_explain_analyze_write(session, settings, tmp_path):
    target = tmp_path / "pwned.csv"
    with pytest.raises(ReadOnlyViolation):
        run_query(
            session,
            settings,
            f"EXPLAIN ANALYZE COPY (SELECT 1 AS x) TO '{target.as_posix()}'",
        )
    assert not target.exists()


def test_query_pushes_the_cap_into_duckdb(session, settings, monkeypatch):
    """The statement runs as written, with limit+1 handed to DuckDB itself."""
    real_execute = session.execute
    calls: list[tuple[str, int | None, int | None]] = []

    def spy(sql, **kwargs):
        calls.append((sql, kwargs.get("max_rows"), kwargs.get("limit")))
        return real_execute(sql, **kwargs)

    monkeypatch.setattr(session, "execute", spy)
    out = run_query(session, settings, "SELECT * FROM range(100)", max_rows=5)
    assert body_rows(out) == 5
    assert "more available" in out
    assert calls == [("SELECT * FROM range(100)", 6, 6)]


def test_query_runs_explain_without_a_pushed_limit(session, settings, monkeypatch):
    """EXPLAIN yields a plan, not a relation, so only the fetch-side cap applies."""
    calls: list[int | None] = []
    real_execute = session.execute

    def spy(sql, **kwargs):
        calls.append(kwargs.get("limit"))
        return real_execute(sql, **kwargs)

    monkeypatch.setattr(session, "execute", spy)
    run_query(session, settings, "EXPLAIN SELECT 1", max_rows=5)
    assert calls == [None]


def test_query_keeps_duplicate_column_names(session, settings):
    """Capping must not rename columns the way a subquery wrap would."""
    out = run_query(session, settings, "SELECT 1 AS a, 2 AS a")
    assert out.splitlines()[0] == "| a | a |"


def test_query_respects_a_statements_own_limit(session, settings):
    out = run_query(session, settings, "SELECT * FROM range(100) LIMIT 3", max_rows=50)
    assert body_rows(out) == 3
    assert "more available" not in out


@pytest.mark.parametrize(
    "sql", ["DESCRIBE SELECT 1 AS a", "SUMMARIZE SELECT 1 AS a", "SHOW TABLES", "VALUES (1),(2)"]
)
def test_query_handles_statements_invalid_as_a_subquery(session, settings, sql):
    run_query(session, settings, sql)


def test_query_reports_a_timeout_as_a_tool_error(settings):
    """A timeout is a user-facing outcome, so it arrives as a ToolError."""
    from duckdb_mcp.db import DuckDBSession

    db = DuckDBSession(default_timeout=0.3, setup=False)
    try:
        with pytest.raises(ToolError, match="time limit"):
            run_query(db, settings, "SELECT count(*) FROM range(200000000000)")
    finally:
        db.close()


def test_query_surfaces_sql_errors(session, settings):
    with pytest.raises(ToolError, match="nope"):
        run_query(session, settings, "SELECT * FROM nope_missing_table")


def test_describe_file(session, settings, sales):
    out = describe_file(session, settings, sales)
    assert "4 columns" in out
    assert "5 rows" in out
    assert "region" in out and "VARCHAR" in out


def test_describe_file_can_skip_row_count(session, settings, sales):
    out = describe_file(session, settings, sales, include_row_count=False)
    assert "rows" not in out.splitlines()[0]


def test_describe_file_glob(session, settings, data_dir):
    out = describe_file(session, settings, (data_dir / "*.parquet").as_posix())
    assert "4 columns" in out


def test_describe_ndjson(session, settings, data_dir):
    out = describe_file(session, settings, (data_dir / "events.ndjson").as_posix())
    assert "2 columns" in out


def test_describe_file_reports_truncated_schema(session, settings, tmp_path):
    wide = tmp_path / "wide.csv"
    names = [f"c{i}" for i in range(200)]
    wide.write_text(",".join(names) + "\n" + ",".join("1" for _ in names) + "\n", encoding="utf-8")

    settings.max_bytes = 400
    out = describe_file(session, settings, wide.as_posix())
    assert "200 columns" in out
    shown = body_rows(out)
    assert shown < 200
    assert f"showing {shown} of 200 columns" in out


def test_describe_missing_file(session, settings, data_dir):
    with pytest.raises(ToolError):
        describe_file(session, settings, (data_dir / "absent.parquet").as_posix())


def test_preview_file(session, settings, sales):
    out = preview_file(session, settings, sales, rows=2)
    assert "first 2 rows" in out
    assert body_rows(out) == 2


def test_preview_shows_nulls(session, settings, sales):
    assert "NULL" in preview_file(session, settings, sales, rows=10)


def test_preview_header_matches_the_table_after_byte_truncation(session, settings, sales):
    settings.max_bytes = 160
    out = preview_file(session, settings, sales, rows=20)
    shown = body_rows(out)
    assert f"first {shown} rows" in out
    assert "truncated" in out


def test_list_files_filters_to_data_files(session, settings, data_dir):
    out = list_files(session, settings, data_dir.as_posix())
    assert "sales.csv" in out and "sales.parquet" in out and "events.ndjson" in out
    assert "notes.md" not in out


def test_list_files_can_include_everything(session, settings, data_dir):
    out = list_files(session, settings, data_dir.as_posix(), data_files_only=False)
    assert "notes.md" in out


def test_list_files_pattern(session, settings, data_dir):
    out = list_files(session, settings, data_dir.as_posix(), pattern="*.parquet")
    assert "sales.parquet" in out and "sales.csv" not in out


def test_list_files_uses_forward_slashes(session, settings, data_dir):
    out = list_files(session, settings, data_dir.as_posix())
    assert "\\" not in out


def test_list_files_reports_empty(session, settings, tmp_path):
    assert "No matching files" in list_files(session, settings, tmp_path.as_posix())


def test_list_files_stats_only_the_rows_it_renders(session, settings, tmp_path, monkeypatch):
    """Listing a huge directory must not cost one syscall per file."""
    for i in range(40):
        (tmp_path / f"f{i:03d}.csv").write_text("a\n1\n", encoding="utf-8")

    stats: list[str] = []
    real_stat = os.stat
    monkeypatch.setattr(os, "stat", lambda p, *a, **k: (stats.append(p), real_stat(p))[1])

    settings.max_rows = 5
    out = list_files(session, settings, tmp_path.as_posix())
    assert len(stats) == 5
    assert "40 file(s)" in out
    assert "showing 5 of 40 rows" in out


def test_list_files_does_not_double_the_root_separator(session, settings, monkeypatch):
    seen: list[str] = []
    real_execute = session.execute

    def spy(sql, **kwargs):
        seen.append(sql)
        return real_execute(sql, **kwargs)

    monkeypatch.setattr(session, "execute", spy)
    list_files(session, settings, "/")
    assert "'/*'" in seen[0] and "'//*'" not in seen[0]


def test_profile_columns(session, settings, sales):
    out = profile_columns(session, settings, sales)
    assert "5 rows" in out
    assert "4 column(s) profiled" in out
    header = out.splitlines()[2]
    assert header.startswith("| column | type | nulls | approx_distinct | min | max |")
    assert "1 (20.0%)" in row_for(out, "amount")  # one NULL out of five
    assert body_rows(out) == 4


def test_profile_columns_top_values(session, settings, sales):
    out = profile_columns(session, settings, sales, top_k=3)
    assert "East (2)" in row_for(out, "region")


def test_profile_columns_subset(session, settings, sales):
    out = profile_columns(session, settings, sales, columns=["region"])
    assert "1 column(s) profiled" in out
    assert "| amount " not in out


def test_profile_columns_unknown_column(session, settings, sales):
    with pytest.raises(ToolError, match="Unknown column"):
        profile_columns(session, settings, sales, columns=["nope"])


def test_profile_columns_no_top_values(session, settings, sales):
    out = profile_columns(session, settings, sales, top_k=0)
    assert "top_values" not in out


def test_profile_columns_scans_the_file_twice_at_most(session, settings, sales, monkeypatch):
    """One aggregate pass plus one top-values pass, however many columns there are."""
    real_execute = session.execute
    scans: list[str] = []

    def spy(sql, **kwargs):
        if sales in sql and "DESCRIBE" not in sql:
            scans.append(sql)
        return real_execute(sql, **kwargs)

    monkeypatch.setattr(session, "execute", spy)
    profile_columns(session, settings, sales, top_k=3)
    assert len(scans) == 2


def test_profile_columns_reports_truncated_table(session, settings, sales):
    settings.max_bytes = 250
    out = profile_columns(session, settings, sales)
    shown = body_rows(out)
    assert shown < 4
    assert f"showing {shown} of 4 columns" in out


def test_join_across_formats(session, settings, data_dir):
    csv = (data_dir / "sales.csv").as_posix()
    parquet = (data_dir / "sales.parquet").as_posix()
    out = run_query(
        session,
        settings,
        f"SELECT count(*) AS n FROM '{csv}' a JOIN '{parquet}' b USING (id)",
    )
    assert "| 5 |" in out
