"""The five tools, exercised against real csv / parquet / ndjson files."""

from __future__ import annotations

import pytest

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


def test_query_without_cap_note_when_all_rows_fit(session, settings):
    out = run_query(session, settings, "SELECT * FROM range(3)", max_rows=10)
    assert "more available" not in out


def test_query_supports_explain(session, settings):
    # EXPLAIN cannot be wrapped in a subquery; the fallback path handles it.
    out = run_query(session, settings, "EXPLAIN SELECT 1")
    assert "explain_value" in out
    assert body_rows(out) >= 1


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


def test_describe_missing_file(session, settings, data_dir):
    with pytest.raises(ToolError):
        describe_file(session, settings, (data_dir / "absent.parquet").as_posix())


def test_preview_file(session, settings, sales):
    out = preview_file(session, settings, sales, rows=2)
    assert "first 2 rows" in out
    assert body_rows(out) == 2


def test_preview_shows_nulls(session, settings, sales):
    assert "NULL" in preview_file(session, settings, sales, rows=10)


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


def test_join_across_formats(session, settings, data_dir):
    csv = (data_dir / "sales.csv").as_posix()
    parquet = (data_dir / "sales.parquet").as_posix()
    out = run_query(
        session,
        settings,
        f"SELECT count(*) AS n FROM '{csv}' a JOIN '{parquet}' b USING (id)",
    )
    assert "| 5 |" in out
