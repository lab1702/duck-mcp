"""Tool implementations. Pure functions over a DuckDBSession, so they are
directly testable without going through the MCP transport."""

from __future__ import annotations

import datetime as dt
import os
from typing import Any, Sequence

import duckdb

from .config import Settings
from .db import (
    READABLE_EXTS,
    DuckDBSession,
    format_duckdb_error,
    quote_ident,
    source_expr,
    sql_string,
)
from .formatting import format_cell, render_result, to_markdown_table

HARD_MAX_ROWS = 10_000
PROFILE_MAX_COLUMNS = 60
PROFILE_TOP_VALUE_COLUMNS = 20
PROFILE_TOP_VALUE_DISTINCT_LIMIT = 50

_NESTED_TYPE_PREFIXES = ("STRUCT", "MAP", "UNION", "LIST")


class ToolError(RuntimeError):
    """A user-facing failure; the message is returned to the model verbatim."""


def _clamp_rows(requested: int | None, default: int) -> int:
    if requested is None:
        return default
    return max(1, min(int(requested), HARD_MAX_ROWS))


def _is_data_file(name: str) -> bool:
    """True when DuckDB is likely to be able to read ``name`` directly."""
    stem = name.split("?", 1)[0].lower()
    for compression in (".gz", ".zst", ".bz2", ".br"):
        if stem.endswith(compression):
            stem = stem[: -len(compression)]
            break
    return os.path.splitext(stem)[1] in READABLE_EXTS


def _is_nested(column_type: str) -> bool:
    upper = (column_type or "").upper()
    return upper.endswith("[]") or upper.startswith(_NESTED_TYPE_PREFIXES) or upper == "JSON"


def _run(session: DuckDBSession, sql: str, *, read_only: bool = False):
    try:
        return session.execute(sql, read_only=read_only)
    except duckdb.Error as exc:
        raise ToolError(format_duckdb_error(exc)) from exc


def _describe(session: DuckDBSession, source: str) -> list[tuple[str, str, str]]:
    _, rows = _run(session, f"DESCRIBE SELECT * FROM {source}")
    described = []
    for row in rows:
        name = row[0]
        ctype = row[1] if len(row) > 1 else ""
        nullable = row[2] if len(row) > 2 else ""
        described.append((name, ctype, nullable))
    return described


# --------------------------------------------------------------------------
# query
# --------------------------------------------------------------------------


def run_query(
    session: DuckDBSession,
    settings: Settings,
    sql: str,
    max_rows: int | None = None,
) -> str:
    limit = _clamp_rows(max_rows, settings.max_rows)
    # Validate the statement the caller actually wrote, before it is wrapped.
    from .db import assert_read_only

    assert_read_only(session.connection, sql)

    inner = sql.strip().rstrip(";").strip()
    # Fetching limit+1 rows tells us whether more exist without counting them,
    # and keeps a `SELECT *` over a huge file from materialising in Python.
    wrapped = f"SELECT * FROM (\n{inner}\n) AS _duckdb_mcp_q LIMIT {limit + 1}"
    try:
        columns, rows = session.execute(wrapped, read_only=False)
    except duckdb.Error:
        # EXPLAIN, DESCRIBE and duplicate output names cannot be wrapped in a
        # subquery; run the original and let its error surface if it is broken.
        columns, rows = _run(session, inner)

    return render_result(
        columns,
        rows,
        max_rows=limit,
        max_bytes=settings.max_bytes,
        empty_note="(0 rows)",
    )


# --------------------------------------------------------------------------
# describe_file
# --------------------------------------------------------------------------


def describe_file(
    session: DuckDBSession,
    settings: Settings,
    path: str,
    include_row_count: bool = True,
) -> str:
    source = source_expr(path)
    described = _describe(session, source)
    if not described:
        raise ToolError(f"{path} has no readable columns.")

    header = f"**{path}** — {len(described)} columns"
    if include_row_count:
        count = _run(session, f"SELECT count(*) FROM {source}")[1][0][0]
        header += f", {int(count):,} rows"

    table, _ = to_markdown_table(
        ["column_name", "column_type", "nullable"],
        described,
        max_bytes=settings.max_bytes,
    )
    return f"{header}\n\n{table}"


# --------------------------------------------------------------------------
# preview_file
# --------------------------------------------------------------------------


def preview_file(
    session: DuckDBSession,
    settings: Settings,
    path: str,
    rows: int | None = None,
) -> str:
    limit = _clamp_rows(rows, 20)
    source = source_expr(path)
    columns, fetched = _run(session, f"SELECT * FROM {source} LIMIT {limit + 1}")
    return render_result(
        columns,
        fetched,
        max_rows=limit,
        max_bytes=settings.max_bytes,
        header=f"**{path}** — first {min(limit, len(fetched))} rows",
        empty_note="(file is empty)",
    )


# --------------------------------------------------------------------------
# list_files
# --------------------------------------------------------------------------


def list_files(
    session: DuckDBSession,
    settings: Settings,
    path: str = ".",
    pattern: str = "*",
    recursive: bool = False,
    data_files_only: bool = True,
) -> str:
    base = (path or ".").replace("\\", "/").rstrip("/")
    if not base:
        base = "/"
    glob_pattern = f"{base}/**/{pattern}" if recursive else f"{base}/{pattern}"
    _, rows = _run(session, f"SELECT file FROM glob({sql_string(glob_pattern)}) ORDER BY file")
    # glob() hands back native separators on Windows; forward slashes are what the
    # caller will paste back into a query.
    files = [row[0].replace("\\", "/") for row in rows]

    if data_files_only:
        files = [f for f in files if _is_data_file(f)]

    if not files:
        return f"No matching files under {glob_pattern}"

    table_rows: list[tuple[Any, ...]] = []
    for name in files:
        size = ""
        modified = ""
        if "://" not in name:
            try:
                stat = os.stat(name)
                size = f"{stat.st_size:,}"
                modified = dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            except OSError:
                pass
        table_rows.append((name, size, modified))

    return render_result(
        ["file", "size_bytes", "modified"],
        table_rows,
        max_rows=settings.max_rows,
        max_bytes=settings.max_bytes,
        header=f"**{glob_pattern}** — {len(files)} file(s)",
    )


# --------------------------------------------------------------------------
# profile_columns
# --------------------------------------------------------------------------


def profile_columns(
    session: DuckDBSession,
    settings: Settings,
    path: str,
    columns: Sequence[str] | None = None,
    top_k: int = 5,
) -> str:
    source = source_expr(path)
    described = _describe(session, source)
    types = {name: ctype for name, ctype, _ in described}

    if columns:
        missing = [c for c in columns if c not in types]
        if missing:
            raise ToolError(
                f"Unknown column(s): {', '.join(missing)}. Available: {', '.join(types)}"
            )
        selected = list(columns)
    else:
        selected = [name for name, _, _ in described]

    dropped = 0
    if len(selected) > PROFILE_MAX_COLUMNS:
        dropped = len(selected) - PROFILE_MAX_COLUMNS
        selected = selected[:PROFILE_MAX_COLUMNS]

    # One scan of the file computes every column's statistics at once.
    exprs = ["count(*)"]
    plan: list[tuple[str, bool]] = []
    for name in selected:
        ident = quote_ident(name)
        nested = _is_nested(types.get(name, ""))
        exprs.append(f"count({ident})")
        if nested:
            exprs.extend(["NULL", "NULL", "NULL"])
        else:
            exprs.extend([
                f"approx_count_distinct({ident})",
                f"min({ident})::VARCHAR",
                f"max({ident})::VARCHAR",
            ])
        plan.append((name, nested))

    _, agg_rows = _run(session, f"SELECT {', '.join(exprs)} FROM {source}")
    values = list(agg_rows[0]) if agg_rows else []
    total = int(values[0]) if values else 0

    records: list[list[Any]] = []
    distincts: dict[str, int | None] = {}
    for index, (name, nested) in enumerate(plan):
        offset = 1 + index * 4
        non_null, distinct, minimum, maximum = values[offset : offset + 4]
        nulls = total - int(non_null or 0)
        null_pct = f"{(nulls / total * 100):.1f}%" if total else "—"
        distincts[name] = None if distinct is None else int(distinct)
        records.append([
            name,
            types.get(name, ""),
            f"{nulls:,} ({null_pct})",
            "—" if distinct is None else f"~{int(distinct):,}",
            "—" if nested else format_cell(minimum),
            "—" if nested else format_cell(maximum),
        ])

    header_cols = ["column", "type", "nulls", "approx_distinct", "min", "max"]

    if top_k and top_k > 0:
        candidates = [
            name
            for name, nested in plan
            if not nested
            and distincts.get(name) is not None
            and 0 < distincts[name] <= PROFILE_TOP_VALUE_DISTINCT_LIMIT
        ][:PROFILE_TOP_VALUE_COLUMNS]
        top_values = {name: _top_values(session, source, name, top_k) for name in candidates}
        if top_values:
            header_cols.append("top_values")
            for record in records:
                record.append(top_values.get(record[0], "—"))

    notes = []
    if dropped:
        notes.append(f"{dropped} further column(s) not profiled")
    header = f"**{path}** — {total:,} rows, {len(records)} column(s) profiled"
    if notes:
        header += " (" + "; ".join(notes) + ")"

    table, _ = to_markdown_table(header_cols, records, max_bytes=settings.max_bytes)
    return f"{header}\n\n{table}"


def _top_values(session: DuckDBSession, source: str, column: str, top_k: int) -> str:
    ident = quote_ident(column)
    sql = (
        f"SELECT {ident}::VARCHAR AS v, count(*) AS n FROM {source} "
        f"GROUP BY 1 ORDER BY n DESC, v LIMIT {int(top_k)}"
    )
    try:
        _, rows = session.execute(sql, read_only=False)
    except duckdb.Error:
        return "—"
    return ", ".join(f"{format_cell(v)} ({int(n):,})" for v, n in rows)
