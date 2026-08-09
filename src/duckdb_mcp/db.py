"""DuckDB connection setup and query execution."""

from __future__ import annotations

import sys
import threading
import time
from typing import Any, Iterable

import duckdb

# A statement handed a fully spent time budget still needs a non-zero timeout,
# because zero means "no limit" to DuckDBSession.execute.
_MIN_TIME_SLICE = 0.001

_COMPRESSION_SUFFIXES = (".gz", ".zst", ".bz2", ".br")

CSV_EXTS = {".csv", ".tsv", ".tab", ".txt"}
JSON_EXTS = {".json", ".ndjson", ".jsonl"}
PARQUET_EXTS = {".parquet", ".pq"}
EXCEL_EXTS = {".xlsx", ".xlsm", ".xls"}

READABLE_EXTS = CSV_EXTS | JSON_EXTS | PARQUET_EXTS | EXCEL_EXTS

# Path prefixes that only work once httpfs is loaded.
REMOTE_SCHEMES = (
    "s3://", "gs://", "gcs://", "r2://", "az://", "azure://", "abfss://", "http://", "https://",
)

# Signs that a remote read was refused rather than merely missing. DuckDB
# rewrites s3:// to the bucket's https:// form before reporting, so the scheme
# has to be recognised in the statement rather than in the message.
_AUTH_SIGNALS = (
    "403", "401", "accessdenied", "unauthorized", "signaturedoesnotmatch",
    "invalidaccesskeyid", "credential",
)

# The reader that takes multi-file options such as union_by_name, per format.
MULTI_FILE_READERS = (
    [(ext, "read_csv") for ext in CSV_EXTS]
    + [(ext, "read_json_auto") for ext in JSON_EXTS]
    + [(ext, "read_parquet") for ext in PARQUET_EXTS]
)

# Container formats: readable, but not as lines of text. A raw peek at one
# gets a CSV-parser error rather than anything useful, so callers refuse early.
BINARY_EXTS = PARQUET_EXTS | EXCEL_EXTS

# Formats known not to be parquet. An unrecognised extension is not on this
# list: parquet turns up under .parq, .snappy and no extension at all, so those
# are worth attempting rather than refusing.
NON_PARQUET_EXTS = READABLE_EXTS - PARQUET_EXTS


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
    if ext in EXCEL_EXTS:
        return f"read_xlsx({literal})"
    if ext in {".tsv", ".tab", ".txt"}:
        return f"read_csv({literal})"
    if ext in {".ndjson", ".jsonl"}:
        return f"read_json_auto({literal})"
    if ext == ".pq":
        return f"read_parquet({literal})"
    return literal


def statement_kind(con: duckdb.DuckDBPyConnection, sql: str) -> str:
    """The DuckDB statement kind of ``sql`` -- of its last statement, if several.

    The last one is the one whose result comes back, and its kind decides how
    the statement is run: only a SELECT yields a relation a row cap can be
    pushed into.
    """
    if not sql or not sql.strip():
        raise ValueError("sql must not be empty")
    try:
        statements = con.extract_statements(sql)
    except duckdb.Error as exc:
        raise ValueError(f"Could not parse SQL: {exc}") from exc
    if not statements:
        raise ValueError("sql contains no statement")
    raw = getattr(statements[-1].type, "name", str(statements[-1].type))
    return raw.replace("_STATEMENT", "").upper()


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
        which are not valid in a subquery position. Pass it only for a
        statement that yields a relation -- DDL and DML do not.
        """
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


def capability_hint(capabilities: dict[str, str], sql: str, message: str) -> str | None:
    """Explain a failure that something missing at startup is responsible for.

    Extensions and credentials are set up once, when the server starts, and a
    machine that was offline or unconfigured then produces read errors later
    that say nothing about the real cause. The startup outcome is recorded in
    ``DuckDBSession.capabilities``; this is where it gets used.
    """
    lowered = sql.lower()

    if any(scheme in lowered for scheme in REMOTE_SCHEMES):
        state = capabilities.get("httpfs", "not attempted")
        if state != "ok":
            return (
                f"The httpfs extension is not available ({state}), and remote paths need "
                "it. It is installed when the server starts and needs network access at "
                "that moment, so restart the server once it has some."
            )

    if any(ext in lowered for ext in EXCEL_EXTS):
        state = capabilities.get("excel", "not attempted")
        if state != "ok":
            return (
                f"The excel extension is not available ({state}), and .xlsx workbooks need "
                "it. It is installed at startup; restart the server with network access. "
                "Exporting the sheet to CSV also works."
            )

    if "s3://" in lowered and any(signal in message.lower() for signal in _AUTH_SIGNALS):
        state = capabilities.get("s3_credential_chain", "not attempted")
        if state != "ok":
            return (
                f"No AWS credentials were resolved when the server started ({state}), so "
                "this bucket is being read anonymously — which is what a 403 here usually "
                "means. DuckDB resolves credentials once at startup, so setting them now "
                "requires a restart."
            )
    return None


def log(message: str) -> None:
    """Write a diagnostic line to stderr (stdout is the MCP transport)."""
    print(f"[duckdb-mcp] {message}", file=sys.stderr, flush=True)
