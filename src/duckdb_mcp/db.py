"""DuckDB connection setup, read-only enforcement and query execution."""

from __future__ import annotations

import re
import sys
import threading
from typing import Any, Iterable, Sequence

import duckdb

# Statement kinds a strictly read-only server will run. Everything else --
# INSERT, CREATE, COPY (which writes files), ATTACH, SET, PRAGMA, ... -- is
# rejected before it reaches DuckDB.
ALLOWED_STATEMENTS = frozenset({"SELECT", "EXPLAIN"})

_LEADING_KEYWORD = re.compile(r"^\s*([a-zA-Z_]+)")

# Fallback keyword allowlist, used only when the installed DuckDB is too old to
# expose extract_statements().
_READ_ONLY_KEYWORDS = frozenset({"select", "with", "from", "describe", "explain", "summarize", "show", "table", "values"})

_COMPRESSION_SUFFIXES = (".gz", ".zst", ".bz2", ".br")

_CSV_EXTS = {".csv", ".tsv", ".tab", ".txt"}
_JSON_EXTS = {".json", ".ndjson", ".jsonl"}
_PARQUET_EXTS = {".parquet", ".pq"}
_EXCEL_EXTS = {".xlsx", ".xlsm", ".xls"}

READABLE_EXTS = _CSV_EXTS | _JSON_EXTS | _PARQUET_EXTS | _EXCEL_EXTS


class ReadOnlyViolation(ValueError):
    """Raised when a statement would modify data, files or server state."""


class QueryTimeout(RuntimeError):
    """Raised when a statement is interrupted after exceeding the time limit."""


def sql_string(value: str) -> str:
    """Quote a value as a SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def quote_ident(name: str) -> str:
    """Quote a value as a SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


def _extension(path: str) -> str:
    cleaned = path.split("?", 1)[0].rstrip("/")
    lowered = cleaned.lower()
    for suffix in _COMPRESSION_SUFFIXES:
        if lowered.endswith(suffix):
            lowered = lowered[: -len(suffix)]
            break
    dot = lowered.rfind(".")
    slash = max(lowered.rfind("/"), lowered.rfind("\\"))
    if dot == -1 or dot < slash:
        return ""
    return lowered[dot:]


def source_expr(path: str) -> str:
    """Build the FROM-clause expression that reads ``path``.

    DuckDB's replacement scan already handles a bare ``'file.parquet'`` literal
    for the common formats; the explicit readers are here for the extensions it
    does not recognise (``.txt``, ``.jsonl``, Excel workbooks).
    """
    path = path.strip()
    if not path:
        raise ValueError("path must not be empty")
    literal = sql_string(path)
    ext = _extension(path)
    if ext in _EXCEL_EXTS:
        return f"read_xlsx({literal})"
    if ext in {".tsv", ".tab", ".txt"}:
        return f"read_csv({literal})"
    if ext in {".ndjson", ".jsonl"}:
        return f"read_json_auto({literal})"
    if ext == ".pq":
        return f"read_parquet({literal})"
    return literal


def statement_kinds(con: duckdb.DuckDBPyConnection, sql: str) -> list[str]:
    """Return the DuckDB statement kind of each statement in ``sql``."""
    extract = getattr(con, "extract_statements", None)
    if extract is None:  # pragma: no cover - depends on DuckDB version
        keyword = _LEADING_KEYWORD.match(sql)
        word = keyword.group(1).lower() if keyword else ""
        return ["SELECT" if word in _READ_ONLY_KEYWORDS else word.upper() or "INVALID"]
    kinds = []
    for statement in extract(sql):
        raw = getattr(statement.type, "name", str(statement.type))
        kinds.append(raw.replace("_STATEMENT", "").upper())
    return kinds


def assert_read_only(con: duckdb.DuckDBPyConnection, sql: str) -> None:
    """Reject anything that is not a single read-only statement."""
    if not sql or not sql.strip():
        raise ValueError("sql must not be empty")
    try:
        kinds = statement_kinds(con, sql)
    except duckdb.Error as exc:
        raise ValueError(f"Could not parse SQL: {exc}") from exc
    if not kinds:
        raise ValueError("sql contains no statement")
    if len(kinds) > 1:
        raise ReadOnlyViolation(
            f"Only one statement per call is allowed (got {len(kinds)}: {', '.join(kinds)})."
        )
    kind = kinds[0]
    if kind not in ALLOWED_STATEMENTS:
        raise ReadOnlyViolation(
            f"This server is read-only; {kind} statements are not allowed. "
            "Use SELECT (DESCRIBE / SUMMARIZE / SHOW are fine) or EXPLAIN."
        )


class DuckDBSession:
    """One in-memory DuckDB instance shared by every tool call.

    Each call runs on its own cursor so concurrent tool invocations do not
    trample each other, and so a timeout can interrupt one call in isolation.
    """

    def __init__(self, *, default_timeout: float = 120.0, setup: bool = True) -> None:
        self._con = duckdb.connect(":memory:")
        self.default_timeout = default_timeout
        self.capabilities: dict[str, str] = {}
        if setup:
            self._configure()

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._con

    def close(self) -> None:
        self._con.close()

    def _try(self, name: str, statements: Iterable[str]) -> bool:
        for statement in statements:
            try:
                self._con.execute(statement)
            except duckdb.Error as exc:
                self.capabilities[name] = f"unavailable: {format_duckdb_error(exc)[:120]}"
                return False
        self.capabilities[name] = "ok"
        return True

    def _configure(self) -> None:
        self._try("autoload", [
            "SET autoinstall_known_extensions=true",
            "SET autoload_known_extensions=true",
        ])
        # httpfs powers https:// and s3:// paths; excel powers .xlsx. Both are
        # best-effort: an offline machine still gets a working local server.
        httpfs = self._try("httpfs", ["INSTALL httpfs", "LOAD httpfs"])
        self._try("excel", ["INSTALL excel", "LOAD excel"])
        if httpfs:
            # Picks up AWS env vars, ~/.aws/credentials and instance roles. DuckDB
            # validates the chain at CREATE time, so this fails on a machine with no
            # AWS credentials at all -- exactly the case where it would not help.
            self._try("s3_credential_chain", [
                "CREATE OR REPLACE SECRET duckdb_mcp_aws (TYPE s3, PROVIDER credential_chain)"
            ])

    def execute(
        self,
        sql: str,
        *,
        timeout: float | None = None,
        read_only: bool = True,
    ) -> tuple[list[str], list[tuple[Any, ...]]]:
        """Run ``sql`` and return ``(column_names, rows)``."""
        if read_only:
            assert_read_only(self._con, sql)
        limit = self.default_timeout if timeout is None else timeout
        cursor = self._con.cursor()
        interrupted = threading.Event()

        def _interrupt() -> None:
            interrupted.set()
            try:
                cursor.interrupt()
            except Exception:  # pragma: no cover - cursor already finished
                pass

        timer = threading.Timer(limit, _interrupt) if limit and limit > 0 else None
        try:
            if timer is not None:
                timer.daemon = True
                timer.start()
            result = cursor.execute(sql)
            columns = [d[0] for d in result.description] if result.description else []
            rows = result.fetchall()
            return columns, rows
        except duckdb.Error as exc:
            if interrupted.is_set():
                raise QueryTimeout(
                    f"Query exceeded the {limit:g}s time limit and was cancelled."
                ) from exc
            raise
        finally:
            if timer is not None:
                timer.cancel()
            cursor.close()

    def scalar(self, sql: str, *, timeout: float | None = None) -> Any:
        _, rows = self.execute(sql, timeout=timeout)
        return rows[0][0] if rows and rows[0] else None

    def column_names(self, source: str, *, timeout: float | None = None) -> list[str]:
        columns, _ = self.execute(f"SELECT * FROM {source} LIMIT 0", timeout=timeout)
        return columns


def format_duckdb_error(exc: duckdb.Error) -> str:
    """Flatten a DuckDB error into a single readable line."""
    text = str(exc).strip()
    return " ".join(text.split())


def log(message: str) -> None:
    """Write a diagnostic line to stderr (stdout is the MCP transport)."""
    print(f"[duckdb-mcp] {message}", file=sys.stderr, flush=True)


def head(rows: Sequence[tuple[Any, ...]], n: int) -> list[tuple[Any, ...]]:
    return list(rows[:n])
