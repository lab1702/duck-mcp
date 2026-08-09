"""MCP server wiring: exposes the DuckDB tools over stdio."""

from __future__ import annotations

import argparse
import threading
from typing import Any, Callable, Sequence

import anyio.to_thread
from mcp.server.mcpserver import MCPServer

from . import __version__, tools
from .config import HARD_MAX_ROWS, Settings
from .db import DuckDBSession, log

INSTRUCTIONS = """\
Query local and remote data files with DuckDB SQL. Nothing is ever written:
only SELECT and EXPLAIN statements run, one per call.

Paths are passed straight to DuckDB, so anything it can read works: local
paths, globs (`data/*.parquet`, `data/**/*.csv`), `https://` URLs and `s3://`
URIs (AWS credentials are picked up from the usual environment). CSV, Parquet,
JSON/NDJSON and Excel workbooks are all recognised by extension.

A typical flow: `list_files` to see what is there, `describe_file` for the
schema, `preview_file` for real values, `profile_columns` when you need null
rates and cardinality, then `query` to answer the question. In `query`, refer
to files by quoting the path: `SELECT * FROM 'data/sales.parquet'`.

`preview_file` shows the first rows, so reach for `sample_rows` instead when a
file is ordered by time or partition and its head is not representative. If a
file's parsed output looks wrong -- odd column names, one column holding
everything, fewer rows than expected -- `inspect_raw` shows the underlying text
and what the CSV sniffer made of it; the other tools cannot, because they all
read through that same parser. For parquet, `parquet_metadata` reads the footer
rather than the data: row counts, per-column size and compression, and whether
a filter can skip row groups, all without a scan.

Before trusting a query over a glob of many files, `compare_schemas` checks
that they agree. Files that disagree are not reliably an error: DuckDB adopts
the first file's schema, silently dropping a column the others add and silently
narrowing values its types cannot hold.

Before aggregating over a join, `check_join` says what it will do -- a right
side with repeated keys duplicates left rows and inflates every sum, again
without an error. `find_value` locates which column and file hold a value when
the schema is not yet known.
"""

settings = Settings.from_env()
_session: DuckDBSession | None = None
# Built in a worker thread, so two first calls could arrive at once.
_session_lock = threading.Lock()

mcp = MCPServer("duckdb", instructions=INSTRUCTIONS, version=__version__)


def get_session() -> DuckDBSession:
    """Create the shared in-memory DuckDB instance on first use.

    Blocking, and on the first call substantially so: setup installs httpfs and
    excel and resolves AWS credentials, all over the network. Only ever called
    from ``_call``'s worker thread for that reason.
    """
    global _session
    with _session_lock:
        if _session is None:
            _session = DuckDBSession(
                default_timeout=settings.timeout_seconds, memory_limit=settings.memory_limit
            )
            log(f"duckdb-mcp {__version__} ready; extensions: {_session.capabilities}")
        return _session


async def _call(fn: Callable[..., str], *args: Any, **kwargs: Any) -> str:
    """Run a blocking tool implementation off the event loop.

    The session is resolved inside that thread too, which is why this closes
    over the call rather than binding it with ``partial``: an argument -- even
    to ``partial`` -- is evaluated on the event loop, so the first tool call
    would stall the whole server, pings included, for as long as installing
    extensions over the network takes.
    """

    def run() -> str:
        return fn(get_session(), settings, *args, **kwargs)

    return await anyio.to_thread.run_sync(run)


@mcp.tool()
async def query(sql: str, max_rows: int | None = None) -> str:
    """Run a read-only DuckDB SQL statement and return the rows as a markdown table.

    Only a single SELECT (including WITH/DESCRIBE/SUMMARIZE/SHOW) or EXPLAIN
    statement is accepted; anything that writes data, files or settings is
    rejected. Read files by quoting their path in the FROM clause, e.g.
    `SELECT region, sum(amount) FROM 'data/sales.parquet' GROUP BY 1`.

    Args:
        sql: The statement to run.
        max_rows: Row cap for this call (defaults to the server's limit).
    """
    return await _call(tools.run_query, sql, max_rows)


@mcp.tool()
async def describe_file(path: str, include_row_count: bool = True) -> str:
    """Return the column names and types of a data file, plus its row count.

    Works for any path DuckDB can read -- csv, parquet, json/ndjson, xlsx,
    compressed variants, globs matching many files, http(s) URLs and s3 URIs.

    Args:
        path: File path, glob or URL, e.g. 'data/sales_*.parquet'.
        include_row_count: Set False to skip counting rows on very large inputs.
    """
    return await _call(tools.describe_file, path, include_row_count)


@mcp.tool()
async def preview_file(path: str, rows: int = 20) -> str:
    """Show the first rows of a data file so you can see real values.

    Args:
        path: File path, glob or URL.
        rows: How many rows to show (default 20).
    """
    return await _call(tools.preview_file, path, rows)


@mcp.tool()
async def sample_rows(path: str, rows: int = 20, seed: int | None = None) -> str:
    """Show a random sample of rows from a data file.

    Prefer this to preview_file whenever the file is written in time or
    partition order, where the first rows are all one date or one category and
    generalising from them is misleading. This scans the whole file.

    Args:
        path: File path, glob or URL.
        rows: How many rows to sample (default 20).
        seed: Fix the draw so repeated calls return the same rows.
    """
    return await _call(tools.sample_rows, path, rows, seed)


@mcp.tool()
async def inspect_raw(path: str, lines: int = 20) -> str:
    """Show a text file's raw lines, before any CSV/JSON parsing, together with
    what DuckDB's CSV sniffer detected.

    Use this when a file's parsed output looks wrong -- unexpected column
    names, a missing header, far fewer rows than expected, everything in one
    column. Those come from the parser mis-reading the file rather than from
    the data, and no other tool can show it: they all go through that parser.
    Not for parquet or Excel, which are binary.

    Args:
        path: File path or URL of a text file (csv, tsv, json, ndjson, txt).
        lines: How many lines to show (default 20).
    """
    return await _call(tools.inspect_raw, path, lines)


@mcp.tool()
async def list_files(
    path: str = ".",
    pattern: str = "*",
    recursive: bool = False,
    data_files_only: bool = True,
) -> str:
    """List data files in a directory (local, or an s3:// prefix).

    Args:
        path: Directory or bucket prefix to list.
        pattern: Glob pattern for the file name, e.g. '*.parquet'.
        recursive: Descend into subdirectories.
        data_files_only: Keep only extensions DuckDB can read; set False to see everything.
    """
    return await _call(tools.list_files, path, pattern, recursive, data_files_only)


@mcp.tool()
async def profile_columns(
    path: str,
    columns: list[str] | None = None,
    top_k: int = 5,
) -> str:
    """Profile a file's columns: null counts, approximate distinct counts, min/max
    and the most frequent values of low-cardinality columns.

    This scans the whole file, so prefer describe_file when you only need types.

    Args:
        path: File path, glob or URL.
        columns: Restrict to these columns (default: all).
        top_k: Number of most-frequent values to show; 0 to skip that pass.
    """
    return await _call(tools.profile_columns, path, columns, top_k)


@mcp.tool()
async def parquet_metadata(path: str, row_groups: bool = False) -> str:
    """Report a parquet file's physical layout from its footer, without scanning it.

    Use this to answer questions a scan would be a poor way to answer: how many
    rows a file holds, which columns dominate its size, how well each
    compresses, and whether a range filter on a column can skip row groups
    (the `row_group_order` column). Also flags layouts that make scans
    expensive -- undersized row groups, or a glob of many small files.

    Args:
        path: Parquet file path, glob or URL.
        row_groups: Also list each row group's row count and size.
    """
    return await _call(tools.parquet_metadata, path, row_groups)


@mcp.tool()
async def compare_schemas(path: str, max_files: int | None = None) -> str:
    """Compare the schemas of the files a glob matches and report what a plain
    read does about any differences.

    Use this before trusting a query over a multi-file dataset, and whenever
    one looks wrong -- a column that should be there is missing, or values
    arrive rounded. DuckDB takes its schema from the first file and reconciles
    the others against it, so a column the first file lacks is dropped and a
    value its type cannot hold is narrowed, both without an error.

    Args:
        path: Glob matching the files to compare, e.g. 'data/*.parquet'.
        max_files: How many files to read schemas from (default 100, spread
            across the glob rather than taken from the front).
    """
    return await _call(tools.compare_schemas, path, max_files)


@mcp.tool()
async def check_join(
    left: str,
    right: str,
    left_on: list[str],
    right_on: list[str] | None = None,
) -> str:
    """Check what joining two files on these keys will do, before running it.

    Run this before any join you are about to aggregate over. If the right side
    holds more than one row per key, the join silently duplicates left rows and
    every sum over the result is inflated -- no error, and a total that still
    looks reasonable. Also reports how many rows on each side match nothing, so
    you can tell whether an inner join is quietly dropping data.

    Args:
        left: Left file path, glob or URL.
        right: Right file path, glob or URL.
        left_on: Key column(s) in the left file.
        right_on: Key column(s) in the right file (defaults to left_on).
    """
    return await _call(tools.check_join, left, right, left_on, right_on)


@mcp.tool()
async def find_value(
    path: str,
    value: str,
    columns: list[str] | None = None,
    exact: bool = False,
) -> str:
    """Find which columns, and which files, contain a value.

    Use this when you know a value but not where it lives -- locating a join
    key across unfamiliar files, or confirming an identifier is present at all.
    Every column is compared as text, so numbers, dates and nested values are
    all searched. This scans the whole file.

    Args:
        path: File path, glob or URL.
        value: The value to look for.
        columns: Restrict the search to these columns (default: all).
        exact: Match the whole value instead of any substring of it.
    """
    return await _call(tools.find_value, path, value, columns, exact)


@mcp.tool()
async def check_coverage(path: str, column: str, granularity: str | None = None) -> str:
    """Find missing and repeated values in a column that should run in regular steps.

    Use this on a date, timestamp or sequence column before totalling or
    averaging over it. A table missing three days still sums perfectly happily,
    just short, and one where a day was loaded twice sums too high; neither
    shows up as an error or in a row count. Reports where the holes are, how
    much each hides, and which values repeat.

    Args:
        path: File path, glob or URL.
        column: The date, timestamp or numeric column expected to be regular.
        granularity: Bucket a timestamp first -- year, quarter, month, week,
            day, hour, minute or second. Needed when timestamps carry a time
            of day but the data is really per-day.
    """
    return await _call(tools.check_coverage, path, column, granularity)


def _positive_int(text: str) -> int:
    """An argparse type that rejects what would otherwise break every query."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} must be 1 or greater")
    return value


def _non_negative_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} must not be negative (0 means no limit)")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="duckdb-mcp",
        description="Read-only DuckDB MCP server for CSV, Parquet, JSON and Excel files.",
    )
    parser.add_argument("--version", action="version", version=f"duckdb-mcp {__version__}")
    parser.add_argument(
        "--max-rows",
        type=_positive_int,
        default=None,
        help=f"Default row cap per query, at most {HARD_MAX_ROWS} (env DUCKDB_MCP_MAX_ROWS).",
    )
    parser.add_argument(
        "--timeout",
        type=_non_negative_float,
        default=None,
        help="Seconds before a query is cancelled, 0 for no limit (env DUCKDB_MCP_TIMEOUT).",
    )
    parser.add_argument(
        "--max-bytes",
        type=_positive_int,
        default=None,
        help="Approximate cap on result text size (env DUCKDB_MCP_MAX_BYTES).",
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        default=None,
        help="Override DuckDB's memory ceiling, e.g. '4GB' (env DUCKDB_MCP_MEMORY_LIMIT).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_rows is not None:
        settings.max_rows = args.max_rows
    if args.timeout is not None:
        settings.timeout_seconds = args.timeout
    if args.max_bytes is not None:
        settings.max_bytes = args.max_bytes
    if args.memory_limit is not None:
        settings.memory_limit = args.memory_limit
    mcp.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
