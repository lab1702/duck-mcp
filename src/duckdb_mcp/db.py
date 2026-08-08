"""DuckDB connection setup, read-only enforcement and query execution."""

from __future__ import annotations

import re
import sys
import threading
import time
from typing import Any, Iterable

import duckdb

# Statement kinds a strictly read-only server will run. Everything else --
# INSERT, CREATE, COPY (which writes files), ATTACH, SET, mutating PRAGMAs,
# ... -- is rejected before it reaches DuckDB. EXPLAIN is allowed only after
# the statement it wraps has itself been checked; see assert_read_only.
ALLOWED_STATEMENTS = frozenset({"SELECT", "EXPLAIN"})

_LEADING_KEYWORD = re.compile(r"^\s*([a-zA-Z_]+)")

# Splits `EXPLAIN [ANALYZE | (opt, ...)] <statement>` into its option list (if
# any) and the statement being explained.
_EXPLAIN_PREFIX = re.compile(
    r"^\s*EXPLAIN\s*(?:\(([^)]*)\)|\b(ANALYZE)\b)?\s*",
    re.IGNORECASE,
)

# Fallback keyword allowlist, used only when the installed DuckDB is too old to
# expose extract_statements(). EXPLAIN is deliberately absent: it is reported
# as its own kind so the ANALYZE check below still runs.
_READ_ONLY_KEYWORDS = frozenset({"select", "with", "from", "describe", "summarize", "show", "table", "values"})

# A statement handed a fully spent time budget still needs a non-zero timeout,
# because zero means "no limit" to DuckDBSession.execute.
_MIN_TIME_SLICE = 0.001

_COMPRESSION_SUFFIXES = (".gz", ".zst", ".bz2", ".br")

CSV_EXTS = {".csv", ".tsv", ".tab", ".txt"}
_JSON_EXTS = {".json", ".ndjson", ".jsonl"}
_PARQUET_EXTS = {".parquet", ".pq"}
_EXCEL_EXTS = {".xlsx", ".xlsm", ".xls"}

READABLE_EXTS = CSV_EXTS | _JSON_EXTS | _PARQUET_EXTS | _EXCEL_EXTS

# Container formats: readable, but not as lines of text. A raw peek at one
# gets a CSV-parser error rather than anything useful, so callers refuse early.
BINARY_EXTS = _PARQUET_EXTS | _EXCEL_EXTS


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


def extension_of(path: str) -> str:
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
    ext = extension_of(path)
    if ext in _EXCEL_EXTS:
        return f"read_xlsx({literal})"
    if ext in {".tsv", ".tab", ".txt"}:
        return f"read_csv({literal})"
    if ext in {".ndjson", ".jsonl"}:
        return f"read_json_auto({literal})"
    if ext == ".pq":
        return f"read_parquet({literal})"
    return literal


def _strip_leading_comments(sql: str) -> str:
    """Drop leading whitespace and SQL comments.

    The checks below anchor on the first keyword, and a statement may
    legitimately open with a comment -- models write them. Block comments nest
    in DuckDB, so this counts depth rather than scanning for the first ``*/``.
    """
    pos, length = 0, len(sql)
    while pos < length:
        if sql[pos].isspace():
            pos += 1
        elif sql.startswith("--", pos):
            newline = sql.find("\n", pos)
            pos = length if newline == -1 else newline + 1
        elif sql.startswith("/*", pos):
            depth, scan = 1, pos + 2
            while scan < length and depth:
                if sql.startswith("/*", scan):
                    depth, scan = depth + 1, scan + 2
                elif sql.startswith("*/", scan):
                    depth, scan = depth - 1, scan + 2
                else:
                    scan += 1
            if depth:
                # Unterminated: hand it back untouched and let the parser object.
                return sql[pos:]
            pos = scan
        else:
            break
    return sql[pos:]


def statement_kinds(con: duckdb.DuckDBPyConnection, sql: str) -> list[str]:
    """Return the DuckDB statement kind of each statement in ``sql``."""
    extract = getattr(con, "extract_statements", None)
    if extract is None:  # pragma: no cover - depends on DuckDB version
        keyword = _LEADING_KEYWORD.match(_strip_leading_comments(sql))
        word = keyword.group(1).lower() if keyword else ""
        if word == "explain":
            return ["EXPLAIN"]
        return ["SELECT" if word in _READ_ONLY_KEYWORDS else word.upper() or "INVALID"]
    kinds = []
    for statement in extract(sql):
        raw = getattr(statement.type, "name", str(statement.type))
        kinds.append(raw.replace("_STATEMENT", "").upper())
    return kinds


def assert_read_only(con: duckdb.DuckDBPyConnection, sql: str, _depth: int = 0) -> str:
    """Reject anything that is not a single read-only statement.

    Returns the DuckDB statement kind, which callers use to decide how the
    statement can be run.
    """
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
    if kind == "EXPLAIN" and _depth == 0:
        _assert_explain_read_only(con, sql)
    return kind


def _assert_explain_read_only(con: duckdb.DuckDBPyConnection, sql: str) -> None:
    """Check the statement an EXPLAIN wraps.

    DuckDB reports `EXPLAIN ANALYZE <anything>` as a plain EXPLAIN, but ANALYZE
    *runs* the statement it wraps -- `EXPLAIN ANALYZE COPY (...) TO 'f.csv'`
    would write a file. Plain EXPLAIN only plans, but a write there is a
    mistake either way, so both are rejected.
    """
    body = _strip_leading_comments(sql)
    match = _EXPLAIN_PREFIX.match(body)
    if match is None:
        # The parser already called this an EXPLAIN, so the only way here is
        # text the comment stripper could not get past -- an unterminated block
        # comment, say. Refuse rather than guess at what would run.
        raise ReadOnlyViolation("Could not determine what this EXPLAIN wraps; refusing to run it.")
    options, analyze = match.group(1), match.group(2)
    if analyze or (options and re.search(r"\bANALYZE\b", options, re.IGNORECASE)):
        raise ReadOnlyViolation(
            "EXPLAIN ANALYZE executes the statement it wraps, so it is not read-only. "
            "Use plain EXPLAIN for the query plan."
        )
    inner = body[match.end() :].strip().rstrip(";").strip()
    if not inner:
        raise ReadOnlyViolation("EXPLAIN needs a statement to explain.")
    assert_read_only(con, inner, _depth=1)


class TimeBudget:
    """One wall-clock allowance shared by every statement in a tool call.

    ``DuckDBSession.execute`` applies its timeout per statement, so a tool that
    issues several queries would otherwise be handed a fresh full timeout for
    each of them -- turning the configured limit into a per-statement figure
    rather than the server-wide cap it is documented to be.
    """

    __slots__ = ("_deadline",)

    def __init__(self, seconds: float | None) -> None:
        self._deadline = time.monotonic() + seconds if seconds and seconds > 0 else None

    def remaining(self) -> float:
        """Seconds left, or ``0.0`` meaning unlimited -- what ``execute`` expects."""
        if self._deadline is None:
            return 0.0
        return max(self._deadline - time.monotonic(), _MIN_TIME_SLICE)


class DuckDBSession:
    """One in-memory DuckDB instance shared by every tool call.

    Each call runs on its own cursor so concurrent tool invocations do not
    trample each other, and so a timeout can interrupt one call in isolation.
    """

    def __init__(
        self,
        *,
        default_timeout: float = 120.0,
        memory_limit: str | None = None,
        setup: bool = True,
    ) -> None:
        self._con = duckdb.connect(":memory:")
        self.default_timeout = default_timeout
        self.capabilities: dict[str, str] = {}
        if memory_limit:
            # Left alone, DuckDB sizes this against system RAM, which is right
            # for the one instance this server runs. Overriding it is for the
            # case DuckDB cannot see: several server processes on one machine,
            # each with its own independent ceiling. Not part of _configure,
            # because extensions are best-effort and a stated limit is not.
            self._try("memory_limit", [f"SET memory_limit={sql_string(memory_limit)}"])
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

    def budget(self, seconds: float | None = None) -> TimeBudget:
        """A time allowance for one tool call, defaulting to the server limit."""
        return TimeBudget(self.default_timeout if seconds is None else seconds)

    def execute(
        self,
        sql: str,
        *,
        timeout: float | None = None,
        read_only: bool = True,
        max_rows: int | None = None,
        limit: int | None = None,
    ) -> tuple[list[str], list[tuple[Any, ...]]]:
        """Run ``sql`` and return ``(column_names, rows)``.

        ``max_rows`` caps how many rows are pulled into Python. DuckDB streams
        the result, so this keeps an unbounded ``SELECT *`` from materialising
        even when no LIMIT could be pushed into the statement itself.

        ``limit`` additionally pushes the cap into DuckDB, so the optimizer can
        turn an ``ORDER BY`` into a top-N and skip parquet row groups. It goes
        through the relational API rather than wrapping the statement in
        ``SELECT * FROM (...)``: wrapping makes a subquery, and a subquery's
        output names must be unique, so ``SELECT 1 AS a, 2 AS a`` would come
        back as ``a, a_1``. It also keeps DESCRIBE / SUMMARIZE / SHOW working,
        which are not valid in a subquery position.
        """
        if read_only:
            assert_read_only(self._con, sql)
        time_limit = self.default_timeout if timeout is None else timeout
        cursor = self._con.cursor()
        interrupted = threading.Event()

        def _interrupt() -> None:
            interrupted.set()
            try:
                cursor.interrupt()
            except Exception:  # pragma: no cover - cursor already finished
                pass

        timer = threading.Timer(time_limit, _interrupt) if time_limit and time_limit > 0 else None
        try:
            if timer is not None:
                timer.daemon = True
                timer.start()
            if limit is None:
                result = cursor.execute(sql)
                columns = [d[0] for d in result.description] if result.description else []
            else:
                relation = cursor.sql(sql)
                if relation is None:  # pragma: no cover - callers validate first
                    raise ValueError("Statement returned no result to limit.")
                result = relation.limit(limit)
                columns = list(result.columns)
            rows = result.fetchall() if max_rows is None else result.fetchmany(max_rows)
            return columns, rows
        except duckdb.Error as exc:
            if interrupted.is_set():
                raise QueryTimeout(
                    f"Query exceeded the {time_limit:g}s time limit and was cancelled."
                ) from exc
            raise
        finally:
            if timer is not None:
                timer.cancel()
            cursor.close()


def format_duckdb_error(exc: Exception) -> str:
    """Flatten a DuckDB error (or a QueryTimeout) into a single readable line."""
    text = str(exc).strip()
    return " ".join(text.split())


def log(message: str) -> None:
    """Write a diagnostic line to stderr (stdout is the MCP transport)."""
    print(f"[duckdb-mcp] {message}", file=sys.stderr, flush=True)
