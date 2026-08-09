"""Statement kinds, path handling and the query timeout."""

from __future__ import annotations

import time

import pytest

from duckdb_mcp.db import (
    DuckDBSession,
    QueryTimeout,
    TimeBudget,
    capability_hint,
    quote_ident,
    source_expr,
    sql_string,
    statement_kind,
)


@pytest.mark.parametrize(
    ("sql", "kind"),
    [
        ("SELECT 1", "SELECT"),
        ("  select 1  ", "SELECT"),
        ("WITH a AS (SELECT 1) SELECT * FROM a", "SELECT"),
        ("FROM range(3)", "SELECT"),
        # DuckDB calls all of these SELECT, which is what matters: each yields
        # a relation, so a row cap can be pushed into it.
        ("DESCRIBE SELECT 1", "SELECT"),
        ("SUMMARIZE SELECT 1", "SELECT"),
        ("SHOW TABLES", "SELECT"),
        ("VALUES (1)", "SELECT"),
        ("EXPLAIN SELECT 1", "EXPLAIN"),
        ("CREATE TABLE t(i INT)", "CREATE"),
        ("INSERT INTO t VALUES (1)", "INSERT"),
        ("COPY (SELECT 1) TO 'out.csv'", "COPY"),
        ("SET memory_limit='1GB'", "SET"),
        # The last statement is the one whose result comes back.
        ("CREATE TABLE t(i INT); SELECT 1", "SELECT"),
    ],
)
def test_statement_kind(session, sql, kind):
    assert statement_kind(session.connection, sql) == kind


def test_empty_sql_rejected(session):
    with pytest.raises(ValueError):
        statement_kind(session.connection, "   ")


def test_unparseable_sql_rejected(session):
    with pytest.raises(ValueError, match="Could not parse"):
        statement_kind(session.connection, "SELECT FROM WHERE")


def test_execute_writes(session):
    """Writes reach DuckDB; the connection outlives the call, so they persist."""
    try:
        session.execute("CREATE TABLE persisted(i INT)")
        session.execute("INSERT INTO persisted VALUES (1), (2)")
        assert session.execute("SELECT sum(i) FROM persisted")[1] == [(3,)]
    finally:
        session.execute("DROP TABLE IF EXISTS persisted")


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


def test_no_memory_limit_is_set_by_default():
    """DuckDB's own per-instance limit is right for the one instance we run."""
    db = DuckDBSession(default_timeout=30.0, setup=False)
    try:
        assert "memory_limit" not in db.capabilities
    finally:
        db.close()


def test_memory_limit_is_applied_when_asked_for():
    """For the multi-process case: several servers, each with its own ceiling."""
    db = DuckDBSession(default_timeout=30.0, memory_limit="512MB", setup=False)
    try:
        assert db.capabilities["memory_limit"] == "ok"
        _, rows = db.execute("SELECT current_setting('memory_limit')")
        assert rows[0][0] == "488.2 MiB"  # 512MB is 512 * 1000^2, reported back in MiB
    finally:
        db.close()


def test_bad_memory_limit_is_reported_not_fatal():
    db = DuckDBSession(default_timeout=30.0, memory_limit="not-a-size", setup=False)
    try:
        assert db.capabilities["memory_limit"].startswith("unavailable")
        assert db.execute("SELECT 1")[1] == [(1,)]
    finally:
        db.close()


def test_execute_pushes_a_limit_without_renaming_columns(session):
    columns, rows = session.execute("SELECT 1 AS a, 2 AS a", limit=5)
    assert columns == ["a", "a"]
    assert rows == [(1, 2)]


def test_execute_limit_streams(session):
    # Pushed into DuckDB, so this returns at once rather than building 10^9 rows.
    _, rows = session.execute("SELECT i FROM range(1000000000) t(i)", limit=3, timeout=10)
    assert rows == [(0,), (1,), (2,)]


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


# What the server managed to set up at startup, in the three states that matter.
ALL_OK = {"httpfs": "ok", "excel": "ok", "s3_credential_chain": "ok"}
NO_CREDENTIALS = {"httpfs": "ok", "excel": "ok", "s3_credential_chain": "unavailable: no chain"}
NO_HTTPFS = {"httpfs": "unavailable: offline", "excel": "ok"}


@pytest.mark.parametrize(
    ("capabilities", "sql", "message", "expected"),
    [
        (NO_HTTPFS, "SELECT * FROM 'https://h/x.csv'", "HTTP error", "httpfs"),
        (NO_HTTPFS, "SELECT * FROM 's3://b/k.parquet'", "HTTP error", "httpfs"),
        ({}, "SELECT * FROM 's3://b/k.parquet'", "HTTP 403", "httpfs"),
        ({"excel": "unavailable: offline"}, "read_xlsx('b.xlsx')", "IO Error", "excel"),
        (NO_CREDENTIALS, "SELECT * FROM 's3://b/k.parquet'", "HTTP 403 AccessDenied", "AWS"),
    ],
)
def test_capability_hint_explains_a_startup_failure(capabilities, sql, message, expected):
    hint = capability_hint(capabilities, sql, message)
    assert hint is not None and expected in hint


@pytest.mark.parametrize(
    ("capabilities", "sql", "message"),
    [
        # Everything loaded, so the failure is about the data, not the setup.
        (ALL_OK, "SELECT * FROM 'https://h/x.csv'", "HTTP 404 Not Found"),
        (ALL_OK, "SELECT * FROM 's3://b/k.parquet'", "HTTP 403 AccessDenied"),
        # Missing rather than refused: credentials would not have helped.
        (NO_CREDENTIALS, "SELECT * FROM 's3://b/k.parquet'", "HTTP 404 Not Found"),
        # A local read cannot be httpfs's fault however broken httpfs is.
        (NO_HTTPFS, "SELECT * FROM 'a.csv'", "IO Error: no such file"),
    ],
)
def test_capability_hint_stays_quiet_otherwise(capabilities, sql, message):
    assert capability_hint(capabilities, sql, message) is None


def test_capability_hint_names_the_startup_reason():
    """The recorded reason is the part DuckDB's own error cannot know."""
    hint = capability_hint(NO_HTTPFS, "SELECT * FROM 'https://h/x.csv'", "HTTP error")
    assert "offline" in hint
    assert "restart" in hint
