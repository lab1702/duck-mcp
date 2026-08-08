"""Markdown rendering, escaping and truncation."""

from __future__ import annotations

import pytest

from duckdb_mcp.formatting import (
    format_bytes,
    format_cell,
    render_result,
    to_markdown_table,
    truncation_note,
)


def test_format_cell_handles_specials():
    assert format_cell(None) == "NULL"
    assert format_cell(True) == "true"
    assert format_cell("a|b") == "a\\|b"
    assert format_cell("a\nb") == "a b"
    assert format_cell(b"\x00" * 32) == "<32 bytes>"


def test_format_cell_truncates_long_text():
    cell = format_cell("x" * 1000)
    assert len(cell) == 300
    assert cell.endswith("…")


def test_table_shape():
    table, emitted = to_markdown_table(["a", "b"], [(1, 2), (3, 4)])
    assert emitted == 2
    assert table.splitlines() == ["| a | b |", "| --- | --- |", "| 1 | 2 |", "| 3 | 4 |"]


def body_rows(rendered: str) -> int:
    """Table lines minus the header and the divider."""
    return sum(1 for line in rendered.splitlines() if line.startswith("| ")) - 2


def test_render_result_notes_more_rows():
    rows = [(i,) for i in range(11)]
    out = render_result(["a"], rows, max_rows=10, max_bytes=100_000)
    assert body_rows(out) == 10
    assert "more available" in out


def test_render_result_respects_byte_cap():
    rows = [("x" * 100,) for _ in range(100)]
    out = render_result(["a"], rows, max_rows=100, max_bytes=1024)
    assert "truncated" in out
    assert len(out) < 2048


def test_render_result_empty():
    assert "(0 rows)" in render_result(["a"], [], max_rows=10, max_bytes=1000)


@pytest.mark.parametrize(
    "count,expected",
    [(0, "0B"), (1, "1B"), (512, "512B"), (1023, "1023B"), (1024, "1KB"), (200 * 1024, "200KB"),
     (4 * 1024 * 1024, "4MB")],
)
def test_format_bytes_never_floors_to_zero(count, expected):
    assert format_bytes(count) == expected


def test_render_result_reports_small_caps_in_bytes():
    rows = [("x" * 50,) for _ in range(10)]
    out = render_result(["a"], rows, max_rows=10, max_bytes=200)
    assert "~200B" in out
    assert "~0KB" not in out


def test_render_result_header_placeholder_counts_rendered_rows():
    rows = [("x" * 100,) for _ in range(100)]
    out = render_result(["a"], rows, max_rows=100, max_bytes=1024, header="first {rows} rows")
    assert f"first {body_rows(out)} rows" in out


def test_render_result_header_placeholder_is_zero_when_empty():
    out = render_result(["a"], [], max_rows=10, max_bytes=1000, header="first {rows} rows")
    assert "first 0 rows" in out


def test_render_result_leaves_other_braces_alone():
    # DuckDB glob patterns legitimately contain braces.
    out = render_result(["a"], [(1,)], max_rows=10, max_bytes=1000, header="data/{a,b}.csv")
    assert "data/{a,b}.csv" in out


def test_truncation_note():
    assert truncation_note(4, 4, 1024) is None
    assert truncation_note(3, 200, 1024, unit="columns") == (
        "(showing 3 of 200 columns; output truncated at ~1KB)"
    )
