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
    TimeBudget,
    assert_read_only,
    format_duckdb_error,
    quote_ident,
    source_expr,
    sql_string,
)
from .formatting import format_cell, render_result, to_markdown_table, truncation_note

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


def _run(
    session: DuckDBSession,
    sql: str,
    *,
    read_only: bool = False,
    budget: TimeBudget | None = None,
    max_rows: int | None = None,
):
    try:
        return session.execute(
            sql,
            read_only=read_only,
            timeout=None if budget is None else budget.remaining(),
            max_rows=max_rows,
        )
    except duckdb.Error as exc:
        raise ToolError(format_duckdb_error(exc)) from exc


def _describe(
    session: DuckDBSession, source: str, *, budget: TimeBudget | None = None
) -> list[tuple[str, str, str]]:
    _, rows = _run(session, f"DESCRIBE SELECT * FROM {source}", budget=budget)
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
    kind = assert_read_only(session.connection, sql)
    budget = session.budget()

    inner = sql.strip().rstrip(";").strip()
    if kind == "EXPLAIN":
        # EXPLAIN is not a valid subquery, so it runs as written and the cap is
        # applied while fetching instead.
        columns, rows = _run(session, inner, budget=budget, max_rows=limit + 1)
    else:
        # Fetching limit+1 rows tells us whether more exist without counting
        # them, and pushes the cap down into DuckDB rather than into Python.
        wrapped = f"SELECT * FROM (\n{inner}\n) AS _duckdb_mcp_q LIMIT {limit + 1}"
        try:
            columns, rows = session.execute(
                wrapped, read_only=False, timeout=budget.remaining(), max_rows=limit + 1
            )
        except duckdb.ParserException:
            # Valid on its own but not as a subquery. Only parse errors get a
            # retry: a statement that failed while running has already scanned
            # its input once, and running it again would just pay that cost
            # twice before surfacing the same error.
            columns, rows = _run(session, inner, budget=budget, max_rows=limit + 1)
        except duckdb.Error as exc:
            raise ToolError(format_duckdb_error(exc)) from exc

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
    budget = session.budget()
    described = _describe(session, source, budget=budget)
    if not described:
        raise ToolError(f"{path} has no readable columns.")

    header = f"**{path}** — {len(described)} columns"
    if include_row_count:
        count = _run(session, f"SELECT count(*) FROM {source}", budget=budget)[1][0][0]
        header += f", {int(count):,} rows"

    table, emitted = to_markdown_table(
        ["column_name", "column_type", "nullable"],
        described,
        max_bytes=settings.max_bytes,
    )
    parts = [header, table]
    note = truncation_note(emitted, len(described), settings.max_bytes, unit="columns")
    if note:
        parts.append(note)
    return "\n\n".join(parts)


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
        # {rows} is filled in with the row count that survives the byte cap.
        header=f"**{path}** — first {{rows}} rows",
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
    budget = session.budget()
    described = _describe(session, source, budget=budget)
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

    # One scan of the file computes every column's statistics at once; the
    # optional top-values pass below adds a second, and that is the whole cost.
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

    _, agg_rows = _run(session, f"SELECT {', '.join(exprs)} FROM {source}", budget=budget)
    values = list(agg_rows[0]) if agg_rows else []
    if len(values) != 1 + 4 * len(plan):
        raise ToolError(f"Could not compute column statistics for {path}.")
    total = int(values[0] or 0)

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
        top_values = _top_values(session, source, candidates, top_k, budget=budget)
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

    table, emitted = to_markdown_table(header_cols, records, max_bytes=settings.max_bytes)
    parts = [header, table]
    note = truncation_note(emitted, len(records), settings.max_bytes, unit="columns")
    if note:
        parts.append(note)
    return "\n\n".join(parts)


def _top_values(
    session: DuckDBSession,
    source: str,
    columns: Sequence[str],
    top_k: int,
    *,
    budget: TimeBudget,
) -> dict[str, str]:
    """Most frequent values for several columns, in a single scan of the file.

    Only columns the aggregate pass found to be low-cardinality get here, so
    each ``histogram()`` stays small. Doing them together keeps profile_columns
    to two scans of the file rather than one per column, which matters when the
    file is an https:// or s3:// URL that would otherwise be refetched.
    NULLs are left out; the null count already has its own column.
    """
    if not columns:
        return {}
    exprs = ", ".join(f"histogram({quote_ident(name)}::VARCHAR)" for name in columns)
    try:
        _, rows = session.execute(
            f"SELECT {exprs} FROM {source}", read_only=False, timeout=budget.remaining()
        )
    except duckdb.Error:
        return {}
    if not rows:
        return {}
    summaries: dict[str, str] = {}
    for name, counts in zip(columns, rows[0]):
        if not counts:
            continue
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        summaries[name] = ", ".join(f"{format_cell(v)} ({int(n):,})" for v, n in ranked)
    return summaries
