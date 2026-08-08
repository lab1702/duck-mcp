"""Read-only enforcement, path handling and the query timeout."""

from __future__ import annotations

import time

import duckdb
import pytest

from duckdb_mcp.db import (
    DuckDBSession,
    QueryTimeout,
    ReadOnlyViolation,
    TimeBudget,
    assert_read_only,
    quote_ident,
    source_expr,
    sql_string,
)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "  select 1  ",
        "WITH a AS (SELECT 1) SELECT * FROM a",
        "FROM range(3)",
        "DESCRIBE SELECT 1",
        "SUMMARIZE SELECT 1",
        "SHOW TABLES",
        "EXPLAIN SELECT 1",
    ],
)
def test_read_only_allows_reads(session, sql):
    assert_read_only(session.connection, sql)


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE t(i INT)",
        "INSERT INTO t VALUES (1)",
        "DROP TABLE t",
        "COPY (SELECT 1) TO 'out.csv'",
        "ATTACH 'other.db'",
        "SET memory_limit='1GB'",
        "INSTALL httpfs",
        "UPDATE t SET i = 2",
        "DELETE FROM t",
    ],
)
def test_read_only_rejects_writes(session, sql):
    with pytest.raises(ReadOnlyViolation):
        assert_read_only(session.connection, sql)


@pytest.mark.parametrize(
    "sql",
    [
        # EXPLAIN ANALYZE actually runs what it wraps, so DuckDB reporting the
        # statement kind as EXPLAIN is not enough to call it read-only.
        "EXPLAIN ANALYZE CREATE TABLE evil AS SELECT 42",
        "EXPLAIN ANALYZE COPY (SELECT 1 AS x) TO 'pwned.csv'",
        "EXPLAIN ANALYZE INSERT INTO t VALUES (1)",
        "EXPLAIN ANALYZE INSTALL httpfs",
        "explain\n analyze\n select 1",
        "EXPLAIN (ANALYZE) SELECT 1",
        "EXPLAIN (ANALYZE, FORMAT json) SELECT 1",
        # Plain EXPLAIN only plans, but a write here is a mistake regardless.
        "EXPLAIN CREATE TABLE evil AS SELECT 42",
        "EXPLAIN COPY (SELECT 1 AS x) TO 'pwned.csv'",
    ],
)
def test_read_only_rejects_explain_wrapping_a_write(session, sql):
    with pytest.raises(ReadOnlyViolation):
        assert_read_only(session.connection, sql)


def test_explain_analyze_does_not_run_its_statement(session):
    with pytest.raises(ReadOnlyViolation, match="ANALYZE"):
        assert_read_only(session.connection, "EXPLAIN ANALYZE CREATE TABLE sentinel AS SELECT 42")
    with pytest.raises(duckdb.CatalogException):
        session.execute("SELECT * FROM sentinel")


def test_read_only_allows_explain_options_without_analyze(session):
    assert assert_read_only(session.connection, "EXPLAIN (FORMAT json) SELECT 1") == "EXPLAIN"


def test_assert_read_only_returns_the_kind(session):
    assert assert_read_only(session.connection, "SELECT 1") == "SELECT"


def test_read_only_rejects_multiple_statements(session):
    with pytest.raises(ReadOnlyViolation, match="one statement"):
        assert_read_only(session.connection, "SELECT 1; DROP TABLE t")


def test_read_only_rejects_write_smuggled_after_select(session):
    with pytest.raises(ReadOnlyViolation):
        assert_read_only(session.connection, "SELECT 1; CREATE TABLE evil(i INT)")


def test_empty_sql_rejected(session):
    with pytest.raises(ValueError):
        assert_read_only(session.connection, "   ")


def test_execute_enforces_read_only(session):
    with pytest.raises(ReadOnlyViolation):
        session.execute("CREATE TABLE t(i INT)")


def test_execute_returns_rows(session):
    columns, rows = session.execute("SELECT 1 AS a, 'x' AS b")
    assert columns == ["a", "b"]
    assert rows == [(1, "x")]


def test_execute_caps_fetched_rows(session):
    _, rows = session.execute("SELECT * FROM range(1000)", max_rows=3)
    assert len(rows) == 3


def test_execute_max_rows_does_not_materialise_everything(session):
    # DuckDB streams, so this returns immediately rather than building 10^9 rows.
    _, rows = session.execute("SELECT i FROM range(1000000000) t(i)", max_rows=2, timeout=10)
    assert rows == [(0,), (1,)]


def test_time_budget_shrinks_and_never_reaches_zero():
    budget = TimeBudget(5.0)
    first = budget.remaining()
    assert 0 < first <= 5.0
    assert budget.remaining() <= first

    spent = TimeBudget(-1.0)
    assert spent.remaining() == 0.0  # not a deadline at all: unlimited


def test_time_budget_is_spent_by_elapsed_time():
    budget = TimeBudget(0.4)
    time.sleep(0.25)
    # A second statement inherits what is left, so the configured limit caps the
    # whole tool call rather than each statement in it.
    assert budget.remaining() < 0.2
    time.sleep(0.25)
    assert budget.remaining() == pytest.approx(0.001)


def test_session_budget_uses_the_configured_timeout():
    db = DuckDBSession(default_timeout=0.3, setup=False)
    try:
        assert db.budget().remaining() <= 0.3
        with pytest.raises(QueryTimeout):
            db.execute(
                "SELECT count(*) FROM range(200000000000)", timeout=db.budget().remaining()
            )
    finally:
        db.close()


def test_timeout_cancels_long_query():
    db = DuckDBSession(default_timeout=0.2, setup=False)
    try:
        with pytest.raises(QueryTimeout):
            db.execute("SELECT count(*) FROM range(200000000000)")
    finally:
        db.close()


def test_quoting_escapes_delimiters():
    assert sql_string("a'b") == "'a''b'"
    assert quote_ident('a"b') == '"a""b"'


@pytest.mark.parametrize(
    "path,expected",
    [
        ("data/sales.parquet", "'data/sales.parquet'"),
        ("data/sales.csv", "'data/sales.csv'"),
        ("data/sales.csv.gz", "'data/sales.csv.gz'"),
        ("data/*.parquet", "'data/*.parquet'"),
        ("s3://bucket/x.parquet", "'s3://bucket/x.parquet'"),
        ("data/events.ndjson", "read_json_auto('data/events.ndjson')"),
        ("data/raw.txt", "read_csv('data/raw.txt')"),
        ("book.xlsx", "read_xlsx('book.xlsx')"),
    ],
)
def test_source_expr(path, expected):
    assert source_expr(path) == expected


def test_source_expr_rejects_empty():
    with pytest.raises(ValueError):
        source_expr("   ")
