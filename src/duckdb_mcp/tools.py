"""Tool implementations. Pure functions over a DuckDBSession, so they are
directly testable without going through the MCP transport."""

from __future__ import annotations

import datetime as dt
import os
import re
from typing import Any, Sequence

import duckdb

from .config import HARD_MAX_ROWS, Settings
from .db import (
    BINARY_EXTS,
    CSV_EXTS,
    MULTI_FILE_READERS,
    NON_PARQUET_EXTS,
    READABLE_EXTS,
    DuckDBSession,
    QueryTimeout,
    TimeBudget,
    assert_read_only,
    extension_of,
    format_duckdb_error,
    quote_ident,
    source_expr,
    sql_string,
)
from .formatting import (
    escape_invisibles,
    format_cell,
    format_size,
    render_result,
    to_markdown_table,
    truncation_note,
)

PROFILE_MAX_COLUMNS = 60
PROFILE_TOP_VALUE_COLUMNS = 20
PROFILE_TOP_VALUE_DISTINCT_LIMIT = 50

# Raw-peek limits. A line is cut in SQL rather than in Python so that a file
# which is one enormous minified JSON line cannot be pulled into memory whole.
RAW_MAX_LINES = 500
RAW_LINE_CHARS = 2000

# Thresholds for the layout warnings parquet_metadata emits. Both describe the
# same failure: work split so finely that per-unit overhead dominates. The
# figures are conventional rules of thumb, not DuckDB limits.
SMALL_ROW_GROUP_ROWS = 10_000
SMALL_FILE_BYTES = 16 * 1024 * 1024

# compare_schemas reads one schema per file, so its cost is linear in the file
# count. Past this many it compares a spread instead of every file.
COMPARE_MAX_FILES = 100

# Type names are interpolated into a cast to ask DuckDB how two of them
# reconcile, so anything that is not plainly a type name is not interpolated.
# STRUCT(a INTEGER) and DECIMAL(10,2) pass; a quoted field name does not, and
# is reported as unknown rather than guessed at.
_PLAIN_TYPE = re.compile(r"^[A-Za-z0-9_(), \[\]]+$")

_NESTED_TYPE_PREFIXES = ("STRUCT", "MAP", "UNION", "LIST")


class ToolError(RuntimeError):
    """A user-facing failure; the message is returned to the model verbatim."""


def _plural(count: int, noun: str) -> str:
    return noun if count == 1 else noun + "s"


def _clamp_rows(requested: int | None, default: int) -> int:
    """Hold any row cap to ``1..HARD_MAX_ROWS`` -- the server's default included."""
    return max(1, min(int(default if requested is None else requested), HARD_MAX_ROWS))


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
    limit: int | None = None,
):
    try:
        return session.execute(
            sql,
            read_only=read_only,
            timeout=None if budget is None else budget.remaining(),
            max_rows=max_rows,
            limit=limit,
        )
    except (duckdb.Error, QueryTimeout) as exc:
        # A timeout is a user-facing outcome like any query error, so it leaves
        # here as a ToolError rather than as a bare RuntimeError.
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
    kind = assert_read_only(session.connection, sql)
    budget = session.budget()

    inner = sql.strip().rstrip(";").strip()
    # Asking for limit+1 rows tells us whether more exist without counting them.
    # EXPLAIN produces a plan rather than a relation, so only the fetch-side cap
    # applies to it; everything else also gets the cap pushed into DuckDB.
    columns, rows = _run(
        session,
        inner,
        budget=budget,
        max_rows=limit + 1,
        limit=None if kind == "EXPLAIN" else limit + 1,
    )

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
# sample_rows
# --------------------------------------------------------------------------


def sample_rows(
    session: DuckDBSession,
    settings: Settings,
    path: str,
    rows: int | None = None,
    seed: int | None = None,
) -> str:
    """Return a random sample of rows rather than the first ones.

    ``preview_file`` shows the head of a file, which for anything written in
    time or partition order is systematically unrepresentative -- one date, one
    region, and often the oldest and most schema-drifted records in the set.
    """
    limit = _clamp_rows(rows, 20)
    source = source_expr(path)
    # Reservoir sampling gives exactly ``limit`` rows with uniform probability,
    # at the cost of a full scan -- there is no way to draw a fair sample from
    # an unknown row count without one. REPEATABLE makes the draw reproducible,
    # so a follow-up call can revisit the same rows.
    clause = f"USING SAMPLE reservoir({limit} ROWS)"
    if seed is not None:
        clause += f" REPEATABLE ({int(seed)})"
    columns, fetched = _run(
        session, f"SELECT * FROM {source} {clause}", budget=session.budget(), max_rows=limit
    )

    header = f"**{path}** — random sample of {{rows}} rows"
    if len(fetched) < limit:
        # Fewer rows came back than were asked for, so the sample is the file.
        header = f"**{path}** — all {{rows}} rows (fewer than the {limit} sampled)"
    return render_result(
        columns,
        fetched,
        max_rows=limit,
        max_bytes=settings.max_bytes,
        # The sample is already the whole result, so the only way rows go
        # missing here is the byte cap; total_rows says so without implying
        # that more of the sample was available.
        total_rows=len(fetched),
        header=header,
        empty_note="(file is empty)",
    )


# --------------------------------------------------------------------------
# inspect_raw
# --------------------------------------------------------------------------


def inspect_raw(
    session: DuckDBSession,
    settings: Settings,
    path: str,
    lines: int | None = None,
) -> str:
    """Show a file's raw lines, plus what DuckDB's CSV sniffer makes of them.

    Every other tool here goes through the CSV parser, so when auto-detection
    guesses wrong there is nothing to compare its answer against: a file with a
    comment preamble or an unexpected delimiter comes back as plausible-looking
    but wrong data, with no error. This is the one view that does not depend on
    the parser being right.
    """
    ext = extension_of(path)
    if ext in BINARY_EXTS:
        raise ToolError(
            f"{path} is a binary container, not text; reading it as lines would show nothing "
            "useful. Use describe_file or preview_file instead."
        )

    limit = max(1, min(int(20 if lines is None else lines), RAW_MAX_LINES))
    budget = session.budget()
    literal = sql_string(path)
    # One VARCHAR column, quoting and escaping disabled, and a delimiter no
    # text file contains: every physical line arrives intact, including the
    # ones a real parse would reject. LIMIT pushes down, so a 10GB file costs
    # only the first few kilobytes of it.
    #
    # strict_mode=false matters more than it looks: in strict mode a file with
    # mixed \n and \r\n endings fails outright, and a file this tool is worth
    # running on is exactly the file likely to have them. A raw peek that
    # errors on malformed input would be useless at the one job it has.
    reader = (
        f"read_csv({literal}, columns={{'line': 'VARCHAR'}}, header=false, "
        "auto_detect=false, delim=e'\\x07', quote='', escape='', "
        "strict_mode=false, ignore_errors=true)"
    )
    _, fetched = _run(
        session,
        f"SELECT substr(line, 1, {RAW_LINE_CHARS}) FROM {reader}",
        budget=budget,
        max_rows=limit,
        limit=limit,
    )
    if not fetched:
        return f"**{path}** — file is empty (0 lines)"

    numbered: list[str] = []
    width = len(str(len(fetched)))
    used = 0
    for index, row in enumerate(fetched, start=1):
        text = escape_invisibles("" if row[0] is None else str(row[0]))
        line = f"{index:>{width}} | {text}"
        if used + len(line) + 1 > settings.max_bytes and numbered:
            break
        numbered.append(line)
        used += len(line) + 1

    parts = [
        f"**{path}** — first {len(numbered)} line(s), raw",
        "```text\n" + "\n".join(numbered) + "\n```",
    ]
    if len(numbered) < len(fetched):
        parts.append(truncation_note(len(numbered), len(fetched), settings.max_bytes, unit="lines"))

    sniffed = _sniff_csv(session, path, ext, budget=budget)
    if sniffed:
        parts.extend(sniffed)
    return "\n\n".join(part for part in parts if part)


def _sniff_csv(
    session: DuckDBSession, path: str, ext: str, *, budget: TimeBudget
) -> list[str]:
    """What DuckDB's CSV sniffer concluded about ``path``, if it is CSV-ish.

    Reported next to the raw lines so the two can be compared: the sniffer is
    usually right, and when it is not -- a byte-order mark alone is enough to
    fold the header into the first column name -- the disagreement is the whole
    diagnosis. ``Prompt`` is the sniffer's own ``read_csv`` call, which can be
    edited and passed to ``query`` once the raw lines show what it got wrong.
    """
    if ext and ext not in CSV_EXTS:
        return []
    try:
        _, rows = session.execute(
            "SELECT Delimiter, Quote, Escape, Comment, SkipRows, HasHeader, "
            f"len(Columns), Prompt FROM sniff_csv({sql_string(path)}, ignore_errors=true)",
            read_only=False,
            timeout=budget.remaining(),
        )
    except duckdb.Error:
        # Not a CSV after all (ndjson under a .txt name, say). The raw lines
        # above are still the answer, so this is not worth an error.
        return []
    if not rows:
        return []
    delim, quote, escape, comment, skip, header, ncols, prompt = rows[0]
    summary = (
        f"DuckDB's CSV sniffer reads this as: delimiter {delim!r}, quote {quote!r}, "
        f"escape {escape!r}, comment {comment!r}, skip {skip} row(s), "
        f"header row {'yes' if header else 'no'}, {ncols} column(s)."
    )
    parts = [summary]
    if prompt:
        parts.append("```sql\n" + str(prompt).strip() + "\n```")
    parts.append(
        "(If that disagrees with the raw lines above, pass corrected read_csv options "
        "to `query` -- e.g. `delim`, `skip`, `comment`, `header`, `columns`.)"
    )
    return parts


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
    base = (path or ".").replace("\\", "/").rstrip("/") or "/"
    # Keep exactly one separator, so a filesystem root does not become '//*'.
    prefix = base if base.endswith("/") else base + "/"
    glob_pattern = f"{prefix}**/{pattern}" if recursive else f"{prefix}{pattern}"
    _, rows = _run(session, f"SELECT file FROM glob({sql_string(glob_pattern)}) ORDER BY file")
    # glob() hands back native separators on Windows; forward slashes are what the
    # caller will paste back into a query.
    files = [row[0].replace("\\", "/") for row in rows]

    if data_files_only:
        files = [f for f in files if _is_data_file(f)]

    total = len(files)
    if not total:
        return f"No matching files under {glob_pattern}"

    # Stat only the rows that will be rendered. A directory holding a hundred
    # thousand files should not cost a hundred thousand syscalls to show the
    # first few hundred of them.
    shown = _clamp_rows(None, settings.max_rows)
    table_rows: list[tuple[Any, ...]] = []
    for name in files[:shown]:
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
        max_rows=shown,
        max_bytes=settings.max_bytes,
        total_rows=total,
        header=f"**{glob_pattern}** — {total} file(s)",
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


# --------------------------------------------------------------------------
# parquet_metadata
# --------------------------------------------------------------------------

# Whether a column's row groups are stored in ascending order decides whether a
# range filter on it can skip row groups. The comparison has to work for every
# way parquet renders a statistic as text: numerically when both ends parse as
# numbers ('10' >= '9' is false as text but true as numbers), lexicographically
# otherwise -- which is correct both for BYTE_ARRAY, whose statistics parquet
# already orders bytewise, and for the ISO-8601 text DuckDB gives timestamps.
_ASCENDING = """CASE
    WHEN try_cast(lo AS DOUBLE) IS NOT NULL AND try_cast(prev_hi AS DOUBLE) IS NOT NULL
        THEN try_cast(lo AS DOUBLE) >= try_cast(prev_hi AS DOUBLE)
    ELSE lo >= prev_hi
END"""


def parquet_metadata(
    session: DuckDBSession,
    settings: Settings,
    path: str,
    row_groups: bool = False,
) -> str:
    """Report a parquet file's physical layout, read from its footer.

    Answers what a scan cannot cheaply answer: how the bytes are arranged, how
    much each column costs, and whether a filter can skip row groups. Nothing
    is decompressed, so this stays cheap on a file far too large to profile.
    """
    ext = extension_of(path)
    if ext in NON_PARQUET_EXTS:
        raise ToolError(
            f"{path} is not parquet ({ext}), and only parquet carries this metadata. "
            "Use describe_file for its schema, or profile_columns for its contents."
        )

    literal = sql_string(path)
    budget = session.budget()
    try:
        _, summary = _run(
            session,
            "SELECT count(*), sum(num_rows), sum(num_row_groups), sum(file_size_bytes), "
            "any_value(created_by), count(DISTINCT format_version), max(format_version) "
            f"FROM parquet_file_metadata({literal})",
            budget=budget,
        )
    except ToolError as exc:
        # DuckDB answers a glob that matches nothing with an IO error carrying
        # the whole statement behind it. That is noise around an ordinary
        # outcome, so it gets said plainly instead.
        if "No files found" in str(exc):
            raise ToolError(f"No files matched {path}.") from exc
        raise
    if not summary or summary[0][0] in (0, None):
        raise ToolError(f"No parquet files matched {path}.")
    files, rows, groups, size, created_by, version_count, version = summary[0]
    files, rows, groups = int(files), int(rows or 0), int(groups or 0)

    _, columns = _run(
        session,
        f"""WITH meta AS (
                SELECT file_name, path_in_schema AS col, column_id, row_group_id, type,
                       compression, total_compressed_size AS comp,
                       total_uncompressed_size AS uncomp, stats_null_count AS nulls,
                       stats_min_value AS lo, stats_max_value AS hi
                FROM parquet_metadata({literal})
            ),
            stepped AS (
                SELECT *, lag(hi) OVER (
                    PARTITION BY file_name, col ORDER BY row_group_id
                ) AS prev_hi
                FROM meta
            )
            SELECT col, any_value(type),
                   CASE WHEN count(DISTINCT compression) = 1
                        THEN any_value(compression) ELSE 'mixed' END,
                   sum(comp), sum(uncomp), sum(nulls), count(*),
                   count(*) FILTER (lo IS NULL OR hi IS NULL),
                   count(*) FILTER (
                       prev_hi IS NOT NULL AND lo IS NOT NULL AND NOT ({_ASCENDING})
                   )
            FROM stepped GROUP BY col ORDER BY min(column_id)""",
        budget=budget,
    )

    records: list[list[Any]] = []
    for col, ctype, codec, comp, uncomp, nulls, chunks, no_stats, unordered in columns:
        comp, uncomp = int(comp or 0), int(uncomp or 0)
        records.append([
            col,
            ctype,
            codec,
            format_size(comp),
            f"{uncomp / comp:.1f}x" if comp else "—",
            f"{int(nulls or 0):,}",
            _row_group_order(int(chunks), int(no_stats), int(unordered)),
        ])

    header = (
        f"**{path}** — {files:,} parquet file(s), {rows:,} rows, "
        f"{groups:,} row group(s), {format_size(int(size or 0))} on disk"
    )
    detail = f"Written by {created_by or 'unknown'}"
    if version is not None:
        detail += f" (format v{int(version)}{', mixed' if int(version_count or 1) > 1 else ''})"
    detail += ". Row counts come from the footer, so nothing was scanned."

    table, emitted = to_markdown_table(
        ["column", "type", "codec", "compressed", "ratio", "nulls", "row_group_order"],
        records,
        max_bytes=settings.max_bytes,
    )
    parts = [header, detail, table]
    note = truncation_note(emitted, len(records), settings.max_bytes, unit="columns")
    if note:
        parts.append(note)
    parts.append(
        "(`row_group_order` is `ascending` when a column's row-group ranges climb in "
        "file order, so a range filter on it skips row groups; `scattered` when they "
        "overlap, so every group must be read.)"
    )
    parts.extend(_layout_warnings(files, rows, groups, int(size or 0)))
    if row_groups:
        parts.append(_row_group_table(session, settings, literal, budget=budget))
    return "\n\n".join(part for part in parts if part)


def _row_group_order(chunks: int, no_stats: int, unordered: int) -> str:
    """Summarise one column's row-group ordering for the table."""
    if no_stats:
        return "no stats"
    if chunks < 2:
        return "—"  # a single row group is trivially ordered, and prunes nothing
    if unordered == 0:
        return "ascending"
    return f"scattered ({unordered}/{chunks - 1})"


def _layout_warnings(files: int, rows: int, groups: int, size: int) -> list[str]:
    """Flag layouts that make a scan cost more than the data justifies."""
    warnings = []
    if groups > 1 and rows:
        per_group = rows // groups
        if per_group < SMALL_ROW_GROUP_ROWS:
            warnings.append(
                f"Note: row groups average only {per_group:,} rows. Small row groups add "
                "per-group overhead and give the reader less to parallelise over; "
                "128MB-ish groups are the usual target."
            )
    if files > 1 and size and size // files < SMALL_FILE_BYTES:
        warnings.append(
            f"Note: {files:,} files averaging {format_size(size // files)} each. Many small "
            "files cost one open per file, which dominates on object storage."
        )
    return warnings


def _row_group_table(
    session: DuckDBSession, settings: Settings, literal: str, *, budget: TimeBudget
) -> str:
    """Per-row-group rows and bytes, for spotting uneven splits."""
    limit = _clamp_rows(None, settings.max_rows)
    _, rows = _run(
        session,
        "SELECT file_name, row_group_id, row_group_num_rows, row_group_bytes FROM ("
        "  SELECT DISTINCT file_name, row_group_id, row_group_num_rows, row_group_bytes"
        f"  FROM parquet_metadata({literal})"
        ") ORDER BY file_name, row_group_id",
        budget=budget,
        max_rows=limit + 1,
    )
    table_rows = [
        (
            os.path.basename(str(name).replace("\\", "/")),
            group,
            f"{int(count):,}",
            format_size(int(nbytes)),
        )
        for name, group, count, nbytes in rows[:limit]
    ]
    return render_result(
        ["file", "row_group", "rows", "bytes"],
        table_rows,
        max_rows=limit,
        max_bytes=settings.max_bytes,
        total_rows=len(rows),
        header="Row groups:",
    )


# --------------------------------------------------------------------------
# compare_schemas
# --------------------------------------------------------------------------


def compare_schemas(
    session: DuckDBSession,
    settings: Settings,
    path: str,
    max_files: int | None = None,
) -> str:
    """Compare the schemas of every file a glob matches, and say what a plain
    read will do about the differences.

    Reading a glob whose files disagree is not reliably an error. DuckDB takes
    its schema from the first file and reconciles the rest against it, which
    quietly drops a column the first file lacks and quietly narrows a value the
    first file's type cannot hold. Both look like ordinary results.
    """
    budget = session.budget()
    _, listed = _run(
        session, f"SELECT file FROM glob({sql_string(path)}) ORDER BY file", budget=budget
    )
    files = [str(row[0]).replace("\\", "/") for row in listed]
    if not files:
        raise ToolError(f"No files matched {path}.")
    if len(files) == 1:
        return (
            f"**{path}** — matched one file, so there is nothing to compare.\n\n"
            + describe_file(session, settings, files[0], include_row_count=False)
        )

    limit = max(2, min(int(max_files or COMPARE_MAX_FILES), COMPARE_MAX_FILES))
    chosen = _spread(files, limit)

    schemas: dict[str, list[tuple[str, str]]] = {}
    unreadable: list[str] = []
    for name in chosen:
        try:
            described = _describe(session, source_expr(name), budget=budget)
        except ToolError:
            # One corrupt or empty file must not sink the comparison; it is
            # reported alongside the rest, which is itself a useful finding.
            unreadable.append(name)
            continue
        schemas[name] = [(col, ctype) for col, ctype, _ in described]
    if not schemas:
        raise ToolError(f"None of the {len(chosen)} files matching {path} could be read.")

    reference = chosen[0] if chosen[0] in schemas else next(iter(schemas))
    ref_types = dict(schemas[reference])
    signatures = {tuple(cols) for cols in schemas.values()}

    # Column order follows the reference file, then first appearance elsewhere,
    # so the table reads like the schema a query would actually see.
    order: list[str] = [name for name, _ in schemas[reference]]
    for cols in schemas.values():
        for name, _ in cols:
            if name not in order:
                order.append(name)

    records: list[list[Any]] = []
    drops: list[str] = []
    failures: list[str] = []
    narrowings: list[str] = []
    for column in order:
        seen: dict[str, int] = {}
        for cols in schemas.values():
            for name, ctype in cols:
                if name == column:
                    seen[ctype] = seen.get(ctype, 0) + 1
        present = sum(seen.values())
        types = ", ".join(
            f"{ctype} ({count})" if len(seen) > 1 else ctype
            for ctype, count in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        verdict, kind = _drift_verdict(
            session, column, seen, ref_types, present, len(schemas), reference, budget=budget
        )
        {"drop": drops, "fail": failures, "narrow": narrowings}.get(kind, []).append(column)
        records.append([column, f"{present}/{len(schemas)}", types, verdict])

    header = (
        f"**{path}** — {len(files):,} {_plural(len(files), 'file')}, "
        f"{len(signatures)} distinct {_plural(len(signatures), 'schema')} "
        f"across {len(schemas)} compared"
    )
    parts = [header, _compare_scope(files, chosen, schemas, reference, unreadable)]

    table, emitted = to_markdown_table(
        ["column", "files", "types", "a plain read"], records, max_bytes=settings.max_bytes
    )
    parts.append(table)
    note = truncation_note(emitted, len(records), settings.max_bytes, unit="columns")
    if note:
        parts.append(note)

    if len(signatures) == 1:
        parts.append("Every compared file has the same schema; a plain read is safe.")
    else:
        parts.extend(_drift_advice(path, drops, failures, narrowings))
    return "\n\n".join(part for part in parts if part)


def _spread(files: Sequence[str], limit: int) -> list[str]:
    """Pick ``limit`` files, keeping the first and the last.

    Drift tends to track the order files were written in, so the first N of a
    large glob would be the oldest files and would miss exactly the recent
    change worth finding. An even spread keeps the reference file -- the one
    whose schema a plain read adopts -- and still reaches the newest.
    """
    if len(files) <= limit:
        return list(files)
    step = (len(files) - 1) / (limit - 1)
    picked = {0, len(files) - 1}
    picked.update(round(index * step) for index in range(limit))
    return [files[index] for index in sorted(picked)[:limit]]


def _compare_scope(
    files: Sequence[str],
    chosen: Sequence[str],
    schemas: dict[str, list[tuple[str, str]]],
    reference: str,
    unreadable: Sequence[str],
) -> str:
    """State what was compared and which file sets the schema."""
    lines = []
    if len(chosen) < len(files):
        lines.append(
            f"Compared {len(chosen)} of {len(files):,} files, spread across the glob "
            "(raise max_files to compare more)."
        )
    lines.append(
        f"A plain read takes its schema from the first file, "
        f"`{os.path.basename(reference)}`: "
        + ", ".join(f"{name} {ctype}" for name, ctype in schemas[reference])
    )
    if unreadable:
        shown = ", ".join(os.path.basename(name) for name in unreadable[:5])
        lines.append(
            f"{len(unreadable)} {_plural(len(unreadable), 'file')} "
            f"could not be read at all: {shown}"
        )
    return "\n\n".join(lines)


def _drift_verdict(
    session: DuckDBSession,
    column: str,
    seen: dict[str, int],
    ref_types: dict[str, str],
    present: int,
    total: int,
    reference: str,
    *,
    budget: TimeBudget,
) -> tuple[str, str]:
    """What a plain read does to one column, and which warning it belongs to."""
    if column not in ref_types:
        return (f"drops it — absent from {os.path.basename(reference)}", "drop")
    if present < total:
        absent = total - present
        return (f"fails — missing from {absent} compared {_plural(absent, 'file')}", "fail")

    ref_type = ref_types[column]
    others = [ctype for ctype in seen if ctype != ref_type]
    if not others:
        return ("ok", "")

    worst, kind = "", ""
    for other in others:
        relation = _reconcile(session, ref_type, other, budget=budget)
        if relation == "incompatible":
            return (f"fails — cannot read {other} as {ref_type}", "fail")
        if relation == "narrowing":
            worst, kind = f"narrows {other} to {ref_type} — values are lost", "narrow"
        elif not worst:
            worst, kind = f"casts to {ref_type}", ""
    return (worst, kind)


def _reconcile(
    session: DuckDBSession, ref_type: str, other: str, *, budget: TimeBudget
) -> str:
    """How ``other`` fares when forced into ``ref_type``.

    DuckDB's own rules decide it: the common supertype of the two is the type
    that loses nothing, so a reference type that is not the supertype must be
    narrower, and no supertype at all means the read cannot reconcile them.
    """
    if not (_PLAIN_TYPE.match(ref_type) and _PLAIN_TYPE.match(other)):
        return "unknown"
    try:
        _, rows = session.execute(
            f"SELECT typeof(coalesce(NULL::{ref_type}, NULL::{other}))",
            read_only=False,
            timeout=budget.remaining(),
        )
    except duckdb.Error:
        return "incompatible"
    if not rows:
        return "unknown"
    return "widening" if str(rows[0][0]) == ref_type else "narrowing"


def _drift_advice(
    path: str, drops: Sequence[str], failures: Sequence[str], narrowings: Sequence[str]
) -> list[str]:
    """Spell out the silent losses, then the one option that avoids them."""
    advice = []
    if drops:
        advice.append(
            f"Silently dropped: {', '.join(drops)}. These are absent from the first file, "
            "so a plain read returns no such column and reports nothing."
        )
    if narrowings:
        advice.append(
            f"Silently narrowed: {', '.join(narrowings)}. Values are cast down to the first "
            "file's type, so they come back altered rather than rejected."
        )
    if failures:
        advice.append(f"Read fails on: {', '.join(failures)}.")

    reader = _multi_file_reader(path)
    call = (
        f"{reader}({sql_string(path)}, union_by_name=true)"
        if reader
        else f"read_parquet({sql_string(path)}, union_by_name=true)  -- or read_csv "
        "/ read_json_auto, to match the format"
    )
    advice.append(
        "Reconcile every file's columns by name instead:\n\n"
        f"```sql\nSELECT * FROM {call}\n```\n"
        "(`union_by_name` widens each column to a type that fits every file, and fills "
        "absent ones with NULL.)"
    )
    return advice


def _multi_file_reader(path: str) -> str | None:
    """The reader that accepts ``union_by_name`` for this path's format."""
    ext = extension_of(path)
    for candidate, reader in MULTI_FILE_READERS:
        if ext == candidate:
            return reader
    return None


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
