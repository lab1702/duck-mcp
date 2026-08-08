"""Read-only enforcement, path handling and the query timeout."""

from __future__ import annotations

import pytest

from duckdb_mcp.db import (
    DuckDBSession,
    QueryTimeout,
    ReadOnlyViolation,
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
