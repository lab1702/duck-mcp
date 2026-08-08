"""Markdown rendering, escaping and truncation."""

from __future__ import annotations

from duckdb_mcp.formatting import format_cell, render_result, to_markdown_table


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
