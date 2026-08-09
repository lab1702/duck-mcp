"""Rendering query results as compact markdown tables."""

from __future__ import annotations

import datetime as _dt
import decimal
from typing import Any, Sequence

MAX_CELL_CHARS = 300


def format_cell(value: Any) -> str:
    """Render one value as a single-line markdown table cell."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = repr(value)
    elif isinstance(value, (int, decimal.Decimal, _dt.date, _dt.time, _dt.datetime, _dt.timedelta)):
        text = str(value)
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        text = f"<{len(raw)} bytes>" if len(raw) > 16 else raw.hex()
    else:
        text = str(value)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    if len(text) > MAX_CELL_CHARS:
        text = text[: MAX_CELL_CHARS - 1] + "…"
    return text.replace("|", "\\|")


def escape_invisibles(text: str) -> str:
    """Make characters that a raw peek exists to reveal actually visible.

    A tab where a comma was expected, a stray carriage return or an embedded
    NUL all break a parse without leaving a mark on screen. Printable
    characters -- including non-ASCII text -- are left alone. A byte-order mark
    is rendered too: DuckDB strips a leading one itself, but concatenated
    exports can carry one into the middle of a file, where nothing strips it.

    A literal backslash is doubled, because otherwise the escapes are
    ambiguous: a field holding the two characters ``\\t`` would render exactly
    like one holding a real tab, and telling those apart is the whole job.
    """
    out = []
    for char in text:
        if char == "\\":
            out.append("\\\\")
        elif char == "﻿":
            out.append("<BOM>")
        elif char == "\t":
            out.append("\\t")
        elif char == "\r":
            out.append("\\r")
        elif char == "\n":
            out.append("\\n")
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            out.append(f"\\x{ord(char):02x}")
        else:
            out.append(char)
    return "".join(out)


def format_bytes(count: int) -> str:
    """Render a byte cap for a truncation note, without flooring to ``0KB``."""
    if count >= 1024 * 1024:
        return f"{count / (1024 * 1024):.0f}MB"
    if count >= 1024:
        return f"{count // 1024}KB"
    return f"{count}B"


def format_size(count: int) -> str:
    """Render a measured size, keeping enough precision to compare two of them.

    ``format_bytes`` rounds to whole units, which is right for a stated cap but
    collapses 1.1MB and 1.9MB onto the same "1MB" -- useless in a table whose
    point is which column dominates the file.
    """
    for unit, scale in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if count >= scale:
            return f"{count / scale:.1f}{unit}"
    return f"{count}B"


def truncation_note(emitted: int, total: int, max_bytes: int, unit: str = "rows") -> str | None:
    """Describe a byte-capped table, or ``None`` when nothing was dropped."""
    if emitted >= total:
        return None
    return (
        f"(showing {emitted:,} of {total:,} {unit}; "
        f"output truncated at ~{format_bytes(max_bytes)})"
    )


def to_markdown_table(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    max_bytes: int | None = None,
) -> tuple[str, int]:
    """Render rows as a markdown table.

    Returns the table text and how many rows it actually contains, which is
    fewer than ``len(rows)`` when ``max_bytes`` cuts the output short.
    """
    if not columns:
        return "(no columns)", 0
    header = "| " + " | ".join(format_cell(c) for c in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    size = len(header) + len(divider) + 2
    emitted = 0
    for row in rows:
        line = "| " + " | ".join(format_cell(v) for v in row) + " |"
        if max_bytes is not None and size + len(line) + 1 > max_bytes and emitted:
            break
        lines.append(line)
        size += len(line) + 1
        emitted += 1
    return "\n".join(lines), emitted


def render_result(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    max_rows: int,
    max_bytes: int,
    total_rows: int | None = None,
    header: str | None = None,
    empty_note: str = "(0 rows)",
) -> str:
    """Render a full tool response: optional header, table, truncation notes.

    ``header`` may contain the placeholder ``{rows}``, which is filled in with
    the number of rows the table ends up containing -- the byte cap is applied
    while rendering, so that count is not known until afterwards.
    """
    parts: list[str] = []
    if not rows:
        if header:
            parts.append(header.replace("{rows}", "0"))
        parts.append(empty_note if not columns else to_markdown_table(columns, [])[0])
        if columns:
            parts.append(empty_note)
        return "\n\n".join(parts)

    shown = list(rows[:max_rows])
    table, emitted = to_markdown_table(columns, shown, max_bytes=max_bytes)
    if header:
        parts.append(header.replace("{rows}", f"{emitted:,}"))
    parts.append(table)

    notes: list[str] = []
    if total_rows is not None and emitted < total_rows:
        notes.append(f"showing {emitted:,} of {total_rows:,} rows")
    elif emitted < len(rows) or len(rows) > max_rows:
        notes.append(f"showing first {emitted:,} rows (more available)")
    if emitted < len(shown):
        notes.append(f"output truncated at ~{format_bytes(max_bytes)}")
    if notes:
        parts.append("(" + "; ".join(notes) + ")")
    return "\n\n".join(parts)
