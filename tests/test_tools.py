"""The tools, exercised against real csv / parquet / ndjson files."""

from __future__ import annotations

import os
import re

import pytest

from duckdb_mcp.config import HARD_MAX_ROWS
from duckdb_mcp.db import ReadOnlyViolation
from duckdb_mcp.tools import (
    ToolError,
    check_join,
    compare_schemas,
    describe_file,
    find_value,
    inspect_raw,
    list_files,
    parquet_metadata,
    preview_file,
    profile_columns,
    run_query,
    sample_rows,
)


@pytest.fixture(params=["sales.csv", "sales.parquet"])
def sales(request, data_dir):
    return (data_dir / request.param).as_posix()


def body_rows(rendered: str) -> int:
    """Table lines minus the header and the divider."""
    return sum(1 for line in rendered.splitlines() if line.startswith("| ")) - 2


def row_for(rendered: str, column: str) -> str:
    return next(line for line in rendered.splitlines() if line.startswith(f"| {column} "))


def test_query_reads_a_file(session, settings, sales):
    out = run_query(session, settings, f"SELECT region, sum(amount) AS total FROM '{sales}' GROUP BY 1 ORDER BY 1")
    assert "| region | total |" in out
    assert "East" in out and "560.5" in out


def test_query_rejects_writes(session, settings):
    with pytest.raises(ReadOnlyViolation):
        run_query(session, settings, "CREATE TABLE t AS SELECT 1")


def test_query_caps_rows_and_says_so(session, settings):
    out = run_query(session, settings, "SELECT * FROM range(100)", max_rows=5)
    assert body_rows(out) == 5
    assert "more available" in out


def test_row_cap_never_exceeds_the_hard_maximum():
    from duckdb_mcp.tools import _clamp_rows

    assert _clamp_rows(None, 500) == 500
    assert _clamp_rows(1_000_000, 500) == HARD_MAX_ROWS
    assert _clamp_rows(None, 1_000_000) == HARD_MAX_ROWS  # the server default too
    assert _clamp_rows(0, 500) == 1


def test_query_without_cap_note_when_all_rows_fit(session, settings):
    out = run_query(session, settings, "SELECT * FROM range(3)", max_rows=10)
    assert "more available" not in out


def test_query_supports_explain(session, settings):
    # EXPLAIN cannot be wrapped in a subquery; it runs unwrapped instead.
    out = run_query(session, settings, "EXPLAIN SELECT 1")
    assert "explain_value" in out
    assert body_rows(out) >= 1


def test_query_rejects_explain_analyze_write(session, settings, tmp_path):
    target = tmp_path / "pwned.csv"
    with pytest.raises(ReadOnlyViolation):
        run_query(
            session,
            settings,
            f"EXPLAIN ANALYZE COPY (SELECT 1 AS x) TO '{target.as_posix()}'",
        )
    assert not target.exists()


def test_query_pushes_the_cap_into_duckdb(session, settings, monkeypatch):
    """The statement runs as written, with limit+1 handed to DuckDB itself."""
    real_execute = session.execute
    calls: list[tuple[str, int | None, int | None]] = []

    def spy(sql, **kwargs):
        calls.append((sql, kwargs.get("max_rows"), kwargs.get("limit")))
        return real_execute(sql, **kwargs)

    monkeypatch.setattr(session, "execute", spy)
    out = run_query(session, settings, "SELECT * FROM range(100)", max_rows=5)
    assert body_rows(out) == 5
    assert "more available" in out
    assert calls == [("SELECT * FROM range(100)", 6, 6)]


def test_query_runs_explain_without_a_pushed_limit(session, settings, monkeypatch):
    """EXPLAIN yields a plan, not a relation, so only the fetch-side cap applies."""
    calls: list[int | None] = []
    real_execute = session.execute

    def spy(sql, **kwargs):
        calls.append(kwargs.get("limit"))
        return real_execute(sql, **kwargs)

    monkeypatch.setattr(session, "execute", spy)
    run_query(session, settings, "EXPLAIN SELECT 1", max_rows=5)
    assert calls == [None]


def test_query_keeps_duplicate_column_names(session, settings):
    """Capping must not rename columns the way a subquery wrap would."""
    out = run_query(session, settings, "SELECT 1 AS a, 2 AS a")
    assert out.splitlines()[0] == "| a | a |"


def test_query_respects_a_statements_own_limit(session, settings):
    out = run_query(session, settings, "SELECT * FROM range(100) LIMIT 3", max_rows=50)
    assert body_rows(out) == 3
    assert "more available" not in out


@pytest.mark.parametrize(
    "sql", ["DESCRIBE SELECT 1 AS a", "SUMMARIZE SELECT 1 AS a", "SHOW TABLES", "VALUES (1),(2)"]
)
def test_query_handles_statements_invalid_as_a_subquery(session, settings, sql):
    run_query(session, settings, sql)


def test_query_reports_a_timeout_as_a_tool_error(settings):
    """A timeout is a user-facing outcome, so it arrives as a ToolError."""
    from duckdb_mcp.db import DuckDBSession

    db = DuckDBSession(default_timeout=0.3, setup=False)
    try:
        with pytest.raises(ToolError, match="time limit"):
            run_query(db, settings, "SELECT count(*) FROM range(200000000000)")
    finally:
        db.close()


def test_query_surfaces_sql_errors(session, settings):
    with pytest.raises(ToolError, match="nope"):
        run_query(session, settings, "SELECT * FROM nope_missing_table")


def test_describe_file(session, settings, sales):
    out = describe_file(session, settings, sales)
    assert "4 columns" in out
    assert "5 rows" in out
    assert "region" in out and "VARCHAR" in out


def test_describe_file_can_skip_row_count(session, settings, sales):
    out = describe_file(session, settings, sales, include_row_count=False)
    assert "rows" not in out.splitlines()[0]


def test_describe_file_glob(session, settings, data_dir):
    out = describe_file(session, settings, (data_dir / "*.parquet").as_posix())
    assert "4 columns" in out


def test_describe_ndjson(session, settings, data_dir):
    out = describe_file(session, settings, (data_dir / "events.ndjson").as_posix())
    assert "2 columns" in out


def test_describe_file_reports_truncated_schema(session, settings, tmp_path):
    wide = tmp_path / "wide.csv"
    names = [f"c{i}" for i in range(200)]
    wide.write_text(",".join(names) + "\n" + ",".join("1" for _ in names) + "\n", encoding="utf-8")

    settings.max_bytes = 400
    out = describe_file(session, settings, wide.as_posix())
    assert "200 columns" in out
    shown = body_rows(out)
    assert shown < 200
    assert f"showing {shown} of 200 columns" in out


def test_describe_missing_file(session, settings, data_dir):
    with pytest.raises(ToolError):
        describe_file(session, settings, (data_dir / "absent.parquet").as_posix())


def test_preview_file(session, settings, sales):
    out = preview_file(session, settings, sales, rows=2)
    assert "first 2 rows" in out
    assert body_rows(out) == 2


def test_preview_shows_nulls(session, settings, sales):
    assert "NULL" in preview_file(session, settings, sales, rows=10)


def test_preview_header_matches_the_table_after_byte_truncation(session, settings, sales):
    settings.max_bytes = 160
    out = preview_file(session, settings, sales, rows=20)
    shown = body_rows(out)
    assert f"first {shown} rows" in out
    assert "truncated" in out


@pytest.fixture
def ordered(tmp_path):
    """2,000 rows in ascending order -- a file whose head is unrepresentative."""
    path = tmp_path / "ordered.csv"
    path.write_text(
        "id\n" + "\n".join(str(i) for i in range(2000)) + "\n", encoding="utf-8"
    )
    return path.as_posix()


@pytest.fixture
def malformed(tmp_path):
    """A BOM, a comment preamble, semicolons and a ragged row.

    Exactly the file the CSV sniffer gets wrong without saying so.
    """
    path = tmp_path / "bad.csv"
    path.write_bytes(
        "﻿# export from system X\n# generated 2026-01-01\n"
        "id;name;amount\n1;alice;5\n2;bob;6;extra\n".encode()
    )
    return path.as_posix()


def test_sample_rows_returns_the_requested_count(session, settings, ordered):
    out = sample_rows(session, settings, ordered, rows=10, seed=1)
    assert body_rows(out) == 10
    assert "random sample of 10 rows" in out


def test_sample_rows_is_not_the_head(session, settings, ordered):
    """The whole point: a sample reaches past the first rows of an ordered file."""
    out = sample_rows(session, settings, ordered, rows=10, seed=1)
    values = [
        int(cell)
        for line in out.splitlines()
        if line.startswith("| ")
        for cell in [line.strip("|").strip()]
        if cell.isdigit()
    ]
    assert len(values) == 10
    assert max(values) > 100  # the head would be 0..9
    assert values != list(range(10))


def test_sample_rows_is_reproducible_with_a_seed(session, settings, ordered):
    first = sample_rows(session, settings, ordered, rows=10, seed=7)
    assert first == sample_rows(session, settings, ordered, rows=10, seed=7)


def test_sample_rows_covers_a_short_file_entirely(session, settings, sales):
    out = sample_rows(session, settings, sales, rows=50, seed=1)
    assert body_rows(out) == 5
    assert "all 5 rows" in out
    assert "more available" not in out


def test_sample_rows_on_an_empty_file(session, settings, tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("a\n", encoding="utf-8")
    assert "file is empty" in sample_rows(session, settings, empty.as_posix())


def test_sample_rows_reports_byte_truncation_without_claiming_more(session, settings, ordered):
    settings.max_bytes = 120
    out = sample_rows(session, settings, ordered, rows=20, seed=1)
    shown = body_rows(out)
    assert shown < 20
    assert f"showing {shown} of 20 rows" in out
    assert "more available" not in out  # a sample has no "next page"


def test_inspect_raw_reveals_what_the_parser_hides(session, settings, malformed):
    """The motivating case: preview_file is silently wrong, inspect_raw is not."""
    parsed = preview_file(session, settings, malformed, rows=10)
    assert "# export from system X" not in parsed

    out = inspect_raw(session, settings, malformed)
    assert "# export from system X" in out
    assert "# generated 2026-01-01" in out
    assert "id;name;amount" in out
    assert "2;bob;6;extra" in out  # the ragged row, intact


def test_inspect_raw_contradicts_a_collapsed_parse(session, settings, tmp_path):
    """A report footer collapses the file to one column; the sniffer says three."""
    path = tmp_path / "report.csv"
    path.write_text(
        "id,name,amount\n1,alice,5\n2,bob,6\n-- end of report --\n", encoding="utf-8"
    )
    assert "1 columns" in describe_file(session, settings, path.as_posix())

    out = inspect_raw(session, settings, path.as_posix())
    assert "-- end of report --" in out
    assert "3 column(s)" in out  # the disagreement that is the whole diagnosis
    assert "ignore_errors=true" in out  # and the sniffer's call, which is the fix


def test_inspect_raw_makes_control_characters_visible(session, settings, tmp_path):
    path = tmp_path / "invisible.csv"
    path.write_bytes(b"a,b\n1\tx,\x00oops\n")
    out = inspect_raw(session, settings, path.as_posix())
    assert "1\\tx,\\x00oops" in out


def test_inspect_raw_survives_mixed_line_endings(session, settings, tmp_path):
    """Strict mode rejects these outright -- and they are why this tool exists."""
    path = tmp_path / "mixed.csv"
    path.write_bytes(b"id,name\r\n1,alice\n2,bob\r\n")
    out = inspect_raw(session, settings, path.as_posix())
    assert "id,name" in out and "2,bob" in out


def test_inspect_raw_numbers_lines_and_caps_them(session, settings, malformed):
    out = inspect_raw(session, settings, malformed, lines=2)
    assert "```text" in out
    assert "1 | # export from system X" in out
    assert "2 | # generated 2026-01-01" in out
    assert "id;name;amount" not in out


def test_inspect_raw_reports_the_sniffers_verdict(session, settings, malformed):
    out = inspect_raw(session, settings, malformed)
    assert "CSV sniffer reads this as" in out
    assert "```sql" in out
    assert "read_csv(" in out


def test_inspect_raw_pushes_the_line_cap_into_duckdb(session, settings, ordered, monkeypatch):
    """A raw peek at a huge file must not read the whole file."""
    real_execute = session.execute
    calls: list[tuple[int | None, int | None]] = []

    def spy(sql, **kwargs):
        if "read_csv" in sql and "sniff" not in sql:
            calls.append((kwargs.get("max_rows"), kwargs.get("limit")))
        return real_execute(sql, **kwargs)

    monkeypatch.setattr(session, "execute", spy)
    inspect_raw(session, settings, ordered, lines=5)
    assert calls == [(5, 5)]


def test_inspect_raw_rejects_binary_formats(session, settings, data_dir):
    with pytest.raises(ToolError, match="binary container"):
        inspect_raw(session, settings, (data_dir / "sales.parquet").as_posix())


def test_inspect_raw_handles_ndjson_without_sniffing_it(session, settings, data_dir):
    out = inspect_raw(session, settings, (data_dir / "events.ndjson").as_posix())
    assert '{"id": 1, "region": "West"}' in out
    assert "CSV sniffer" not in out


def test_inspect_raw_on_an_empty_file(session, settings, tmp_path):
    empty = tmp_path / "nothing.csv"
    empty.write_text("", encoding="utf-8")
    assert "0 lines" in inspect_raw(session, settings, empty.as_posix())


def test_inspect_raw_respects_the_byte_cap(session, settings, ordered):
    settings.max_bytes = 60
    out = inspect_raw(session, settings, ordered, lines=100)
    assert "of 100 lines" in out


def test_list_files_filters_to_data_files(session, settings, data_dir):
    out = list_files(session, settings, data_dir.as_posix())
    assert "sales.csv" in out and "sales.parquet" in out and "events.ndjson" in out
    assert "notes.md" not in out


def test_list_files_can_include_everything(session, settings, data_dir):
    out = list_files(session, settings, data_dir.as_posix(), data_files_only=False)
    assert "notes.md" in out


def test_list_files_pattern(session, settings, data_dir):
    out = list_files(session, settings, data_dir.as_posix(), pattern="*.parquet")
    assert "sales.parquet" in out and "sales.csv" not in out


def test_list_files_uses_forward_slashes(session, settings, data_dir):
    out = list_files(session, settings, data_dir.as_posix())
    assert "\\" not in out


def test_list_files_reports_empty(session, settings, tmp_path):
    assert "No matching files" in list_files(session, settings, tmp_path.as_posix())


def test_list_files_stats_only_the_rows_it_renders(session, settings, tmp_path, monkeypatch):
    """Listing a huge directory must not cost one syscall per file."""
    for i in range(40):
        (tmp_path / f"f{i:03d}.csv").write_text("a\n1\n", encoding="utf-8")

    stats: list[str] = []
    real_stat = os.stat
    monkeypatch.setattr(os, "stat", lambda p, *a, **k: (stats.append(p), real_stat(p))[1])

    settings.max_rows = 5
    out = list_files(session, settings, tmp_path.as_posix())
    assert len(stats) == 5
    assert "40 file(s)" in out
    assert "showing 5 of 40 rows" in out


def test_list_files_does_not_double_the_root_separator(session, settings, monkeypatch):
    seen: list[str] = []
    real_execute = session.execute

    def spy(sql, **kwargs):
        seen.append(sql)
        return real_execute(sql, **kwargs)

    monkeypatch.setattr(session, "execute", spy)
    list_files(session, settings, "/")
    assert "'/*'" in seen[0] and "'//*'" not in seen[0]


def test_profile_columns(session, settings, sales):
    out = profile_columns(session, settings, sales)
    assert "5 rows" in out
    assert "4 column(s) profiled" in out
    header = out.splitlines()[2]
    assert header.startswith("| column | type | nulls | approx_distinct | min | max |")
    assert "1 (20.0%)" in row_for(out, "amount")  # one NULL out of five
    assert body_rows(out) == 4


def test_profile_columns_top_values(session, settings, sales):
    out = profile_columns(session, settings, sales, top_k=3)
    assert "East (2)" in row_for(out, "region")


def test_profile_columns_subset(session, settings, sales):
    out = profile_columns(session, settings, sales, columns=["region"])
    assert "1 column(s) profiled" in out
    assert "| amount " not in out


def test_profile_columns_unknown_column(session, settings, sales):
    with pytest.raises(ToolError, match="Unknown column"):
        profile_columns(session, settings, sales, columns=["nope"])


def test_profile_columns_no_top_values(session, settings, sales):
    out = profile_columns(session, settings, sales, top_k=0)
    assert "top_values" not in out


def test_profile_columns_scans_the_file_twice_at_most(session, settings, sales, monkeypatch):
    """One aggregate pass plus one top-values pass, however many columns there are."""
    real_execute = session.execute
    scans: list[str] = []

    def spy(sql, **kwargs):
        if sales in sql and "DESCRIBE" not in sql:
            scans.append(sql)
        return real_execute(sql, **kwargs)

    monkeypatch.setattr(session, "execute", spy)
    profile_columns(session, settings, sales, top_k=3)
    assert len(scans) == 2


def test_profile_columns_reports_truncated_table(session, settings, sales):
    settings.max_bytes = 250
    out = profile_columns(session, settings, sales)
    shown = body_rows(out)
    assert shown < 4
    assert f"showing {shown} of 4 columns" in out


@pytest.fixture(scope="session")
def layout_dir(tmp_path_factory):
    """Parquet written three ways: sorted, shuffled, and split into small files."""
    import duckdb

    directory = tmp_path_factory.mktemp("layout")
    con = duckdb.connect(":memory:")
    body = """SELECT i AS id,
                     TIMESTAMP '2026-01-01' + INTERVAL (i) SECOND AS ts,
                     printf('cust-%08d', i) AS label,
                     i % 7 AS bucket,
                     NULL::VARCHAR AS always_null
              FROM range(60000) t(i)"""
    con.execute(
        f"COPY ({body}) TO '{(directory / 'sorted.parquet').as_posix()}' (ROW_GROUP_SIZE 10000)"
    )
    con.execute(
        f"COPY ({body} ORDER BY hash(i)) TO "
        f"'{(directory / 'shuffled.parquet').as_posix()}' (ROW_GROUP_SIZE 10000)"
    )
    many = directory / "many"
    many.mkdir()
    for part in range(4):
        con.execute(
            f"COPY (SELECT i FROM range({part * 500}, {part * 500 + 500}) t(i)) "
            f"TO '{(many / f'part{part}.parquet').as_posix()}' (ROW_GROUP_SIZE 250)"
        )
    con.close()
    return directory


def test_parquet_metadata_reads_the_footer_without_scanning(
    session, settings, layout_dir, monkeypatch
):
    """Row counts must come from metadata, never from a count(*) over the data."""
    scanned: list[str] = []
    real_execute = session.execute

    def spy(sql, **kwargs):
        if "parquet_metadata" not in sql and "parquet_file_metadata" not in sql:
            scanned.append(sql)
        return real_execute(sql, **kwargs)

    monkeypatch.setattr(session, "execute", spy)
    out = parquet_metadata(session, settings, (layout_dir / "sorted.parquet").as_posix())
    assert scanned == []
    assert "60,000 rows" in out
    assert "6 row group(s)" in out


def test_parquet_metadata_marks_a_sorted_column_ascending(session, settings, layout_dir):
    out = parquet_metadata(session, settings, (layout_dir / "sorted.parquet").as_posix())
    assert "ascending" in row_for(out, "id")
    assert "ascending" in row_for(out, "ts")  # timestamps compare as ISO text
    assert "ascending" in row_for(out, "label")  # strings compare bytewise
    # A column cycling 0..6 spans every row group, so no filter on it can prune.
    assert "scattered" in row_for(out, "bucket")


def test_parquet_metadata_marks_a_shuffled_file_scattered(session, settings, layout_dir):
    out = parquet_metadata(session, settings, (layout_dir / "shuffled.parquet").as_posix())
    for column in ("id", "ts", "label"):
        assert "scattered" in row_for(out, column)


def test_parquet_metadata_reports_columns_without_statistics(session, settings, layout_dir):
    out = parquet_metadata(session, settings, (layout_dir / "sorted.parquet").as_posix())
    # An all-NULL column has no min/max, so ordering is unknown -- not "ascending".
    assert "no stats" in row_for(out, "always_null")
    assert "ascending" not in row_for(out, "always_null")


def test_parquet_metadata_shows_per_column_size(session, settings, layout_dir):
    out = parquet_metadata(session, settings, (layout_dir / "sorted.parquet").as_posix())
    assert "SNAPPY" in out
    assert "| label |" in out
    # Sizes must keep a decimal, or every mid-sized column reads as the same "1MB".
    assert re.search(r"\d+\.\d+(KB|MB)", out)


def test_parquet_metadata_warns_about_small_files_and_row_groups(session, settings, layout_dir):
    out = parquet_metadata(session, settings, (layout_dir / "many" / "*.parquet").as_posix())
    assert "4 parquet file(s), 2,000 rows" in out
    assert "row groups average only" in out
    assert "4 files averaging" in out


def test_parquet_metadata_stays_quiet_on_a_healthy_layout(session, settings, layout_dir):
    out = parquet_metadata(session, settings, (layout_dir / "sorted.parquet").as_posix())
    assert "Note:" not in out


def test_parquet_metadata_lists_row_groups_on_request(session, settings, layout_dir):
    path = (layout_dir / "sorted.parquet").as_posix()
    assert "row_group" not in parquet_metadata(session, settings, path).lower().split("|")[0]
    out = parquet_metadata(session, settings, path, row_groups=True)
    assert "| file | row_group | rows | bytes |" in out
    assert "sorted.parquet" in out


def test_parquet_metadata_rejects_a_non_parquet_file(session, settings, data_dir):
    with pytest.raises(ToolError, match="not parquet"):
        parquet_metadata(session, settings, (data_dir / "sales.csv").as_posix())


def test_parquet_metadata_reports_no_match_cleanly(session, settings, tmp_path):
    """DuckDB's own error embeds the whole statement; that must not leak."""
    with pytest.raises(ToolError) as excinfo:
        parquet_metadata(session, settings, (tmp_path / "absent.parquet").as_posix())
    assert "No files matched" in str(excinfo.value)
    assert "SELECT" not in str(excinfo.value)


def test_parquet_metadata_truncates_a_wide_schema(session, settings, tmp_path):
    import duckdb

    wide = tmp_path / "wide.parquet"
    cols = ", ".join(f"i AS c{n}" for n in range(120))
    con = duckdb.connect(":memory:")
    con.execute(f"COPY (SELECT {cols} FROM range(10) t(i)) TO '{wide.as_posix()}'")
    con.close()

    settings.max_bytes = 700
    out = parquet_metadata(session, settings, wide.as_posix())
    assert "of 120 columns" in out


def write_parquet(directory, selects):
    """Write one parquet file per SELECT, named a.parquet, b.parquet, ..."""
    import duckdb

    con = duckdb.connect(":memory:")
    for index, select in enumerate(selects):
        target = (directory / f"{chr(97 + index)}.parquet").as_posix()
        con.execute(f"COPY ({select}) TO '{target}'")
    con.close()
    return (directory / "*.parquet").as_posix()


# Each scenario pairs files whose schemas disagree with what a plain read does
# about it. "drop" and "narrow" are the silent ones -- no error, wrong answer.
DRIFT_SCENARIOS = {
    "clean": (["SELECT 1 AS id", "SELECT 2 AS id"], "clean"),
    "widening": (["SELECT 1 AS id, 1.5::DOUBLE AS v", "SELECT 2 AS id, 2 AS v"], "clean"),
    "narrowing": (["SELECT 1 AS id, 10 AS v", "SELECT 2 AS id, 10.5::DOUBLE AS v"], "narrow"),
    "extra_column": (["SELECT 1 AS id", "SELECT 2 AS id, 'gone' AS extra"], "drop"),
    "missing_column": (["SELECT 1 AS id, 'keep' AS only_a", "SELECT 2 AS id"], "fail"),
    "incompatible": (["SELECT 1 AS id, 10 AS v", "SELECT 2 AS id, 'ten' AS v"], "fail"),
}


def drift_prediction(report: str) -> str:
    if "Read fails on:" in report:
        return "fail"
    if "Silently dropped" in report:
        return "drop"
    if "Silently narrowed" in report:
        return "narrow"
    return "clean"


@pytest.mark.parametrize("scenario", list(DRIFT_SCENARIOS))
def test_compare_schemas_predicts_what_a_plain_read_really_does(
    session, settings, tmp_path, scenario
):
    """The tool's claims must match DuckDB, or they are worse than no tool."""
    import duckdb

    selects, expected = DRIFT_SCENARIOS[scenario]
    glob = write_parquet(tmp_path, selects)
    assert drift_prediction(compare_schemas(session, settings, glob)) == expected

    # Now establish what actually happens, independently of the tool.
    con = duckdb.connect(":memory:")
    try:
        union = con.sql(f"SELECT * FROM read_parquet('{glob}', union_by_name=true) ORDER BY id")
        union_columns, union_rows = list(union.columns), union.fetchall()
        try:
            plain = con.sql(f"SELECT * FROM '{glob}' ORDER BY id")
            plain_columns, plain_rows = list(plain.columns), plain.fetchall()
        except duckdb.Error:
            actual = "fail"
        else:
            if set(union_columns) - set(plain_columns):
                actual = "drop"
            elif plain_rows != [
                tuple(row[union_columns.index(col)] for col in plain_columns)
                for row in union_rows
            ]:
                actual = "narrow"
            else:
                actual = "clean"
    finally:
        con.close()
    assert actual == expected


def test_compare_schemas_names_the_dropped_column(session, settings, tmp_path):
    glob = write_parquet(tmp_path, ["SELECT 1 AS id", "SELECT 2 AS id, 'gone' AS extra"])
    out = compare_schemas(session, settings, glob)
    assert "2 distinct schemas" in out
    assert "drops it — absent from a.parquet" in row_for(out, "extra")
    assert "| extra | 1/2 |" in out
    assert "Silently dropped: extra" in out


def test_compare_schemas_names_the_narrowed_column(session, settings, tmp_path):
    glob = write_parquet(
        tmp_path, ["SELECT 1 AS id, 10 AS v", "SELECT 2 AS id, 10.5::DOUBLE AS v"]
    )
    out = compare_schemas(session, settings, glob)
    assert "narrows DOUBLE to INTEGER" in row_for(out, "v")
    assert "Silently narrowed: v" in out


def test_compare_schemas_accepts_a_widening_difference(session, settings, tmp_path):
    """DOUBLE first, INTEGER later loses nothing, so it must not be alarming."""
    glob = write_parquet(
        tmp_path, ["SELECT 1 AS id, 1.5::DOUBLE AS v", "SELECT 2 AS id, 2 AS v"]
    )
    out = compare_schemas(session, settings, glob)
    assert "Silently narrowed" not in out
    assert "Read fails" not in out
    assert "casts to DOUBLE" in row_for(out, "v")


def test_compare_schemas_is_quiet_when_files_agree(session, settings, tmp_path):
    glob = write_parquet(tmp_path, ["SELECT 1 AS id, 'x' AS name", "SELECT 2 AS id, 'y' AS name"])
    out = compare_schemas(session, settings, glob)
    assert "1 distinct schema " in out
    assert "a plain read is safe" in out
    assert "union_by_name" not in out


def test_compare_schemas_handles_nested_types(session, settings, tmp_path):
    glob = write_parquet(
        tmp_path,
        ["SELECT 1 AS id, [1,2] AS tags, {'k': 1} AS meta",
         "SELECT 2 AS id, ['a'] AS tags, {'k': 1} AS meta"],
    )
    out = compare_schemas(session, settings, glob)
    # Nested columns must keep their own names, not their inner field names.
    assert "| tags |" in out and "| meta |" in out
    assert "element" not in out
    assert "INTEGER[]" in row_for(out, "tags")
    assert "ok" in row_for(out, "meta")


def test_compare_schemas_suggests_the_reader_matching_the_format(session, settings, tmp_path):
    csvs = tmp_path / "csvs"
    csvs.mkdir()
    (csvs / "jan.csv").write_text("id,name\n1,x\n", encoding="utf-8")
    (csvs / "feb.csv").write_text("id,name,region\n2,y,west\n", encoding="utf-8")
    out = compare_schemas(session, settings, (csvs / "*.csv").as_posix())
    assert "read_csv(" in out
    assert "read_parquet(" not in out


def test_compare_schemas_spreads_across_a_large_glob(session, settings, tmp_path):
    """Drift tracks write order, so sampling the front would miss a late change."""
    import duckdb

    con = duckdb.connect(":memory:")
    for index in range(40):
        extra = ", 'late' AS added" if index >= 38 else ""
        con.execute(
            f"COPY (SELECT {index} AS id{extra}) TO "
            f"'{(tmp_path / f'p{index:03d}.parquet').as_posix()}'"
        )
    con.close()

    out = compare_schemas(session, settings, (tmp_path / "*.parquet").as_posix(), max_files=6)
    assert "Compared 6 of 40 files" in out
    assert "added" in out  # the last file is always included, so the drift is seen


def test_compare_schemas_reports_a_single_file_plainly(session, settings, data_dir):
    out = compare_schemas(session, settings, (data_dir / "sales.parquet").as_posix())
    assert "nothing to compare" in out
    assert "region" in out  # still shows the schema


def test_compare_schemas_survives_an_unreadable_file(session, settings, tmp_path):
    write_parquet(tmp_path, ["SELECT 1 AS id", "SELECT 2 AS id"])
    (tmp_path / "c.parquet").write_bytes(b"not parquet at all")
    out = compare_schemas(session, settings, (tmp_path / "*.parquet").as_posix())
    assert "could not be read at all: c.parquet" in out
    assert "| id | 2/2 |" in out  # the readable files are still compared


def test_compare_schemas_reports_no_match(session, settings, tmp_path):
    with pytest.raises(ToolError, match="No files matched"):
        compare_schemas(session, settings, (tmp_path / "*.parquet").as_posix())


@pytest.fixture(scope="session")
def join_dir(tmp_path_factory):
    """Order/customer files shaped to exercise each join pathology."""
    import duckdb

    directory = tmp_path_factory.mktemp("joins")
    con = duckdb.connect(":memory:")

    def write(name, select):
        con.execute(f"COPY ({select}) TO '{(directory / name).as_posix()}'")

    # 1,000 orders over 200 customers, five orders each.
    write("orders.parquet",
          "SELECT i AS order_id, (i % 200)+1 AS customer_id, 10.0 AS amount FROM range(1000) t(i)")
    # 200 customers, but 20 of them listed twice -- the fan-out.
    write("dupes.parquet",
          "SELECT i+1 AS id, 'c' AS name FROM range(200) t(i) "
          "UNION ALL SELECT i+1, 'dupe' FROM range(20) t(i)")
    write("clean.parquet", "SELECT i+1 AS id, 'c' AS name FROM range(200) t(i)")
    # Only 100 customers, so half the orders match nothing.
    write("partial.parquet", "SELECT i+1 AS id FROM range(100) t(i)")
    write("orphans.parquet", "SELECT 9000+i AS order_id, 9000+i AS customer_id FROM range(30) t(i)")
    write("nullkeys.parquet", "SELECT i AS id, NULL::INT AS cust FROM range(10) t(i)")
    write("strkeys.parquet", "SELECT i AS id, ('c' || i) AS cust FROM range(10) t(i)")
    con.close()
    return directory


def test_check_join_flags_the_fan_out(session, settings, join_dir):
    """The silent bug: duplicate keys on the right inflate every aggregate."""
    out = check_join(
        session, settings,
        (join_dir / "orders.parquet").as_posix(),
        (join_dir / "dupes.parquet").as_posix(),
        ["customer_id"], ["id"],
    )
    assert "many-to-many" in out
    assert "Rows are duplicated" in out
    assert "1,100 rows rather than 1,000" in out


def test_check_join_predicts_the_row_count_exactly(session, settings, join_dir):
    """The headline number must match the join it declines to run."""
    import duckdb

    left = (join_dir / "orders.parquet").as_posix()
    right = (join_dir / "dupes.parquet").as_posix()
    out = check_join(session, settings, left, right, ["customer_id"], ["id"])

    con = duckdb.connect(":memory:")
    actual = con.sql(
        f"SELECT count(*) FROM '{left}' a JOIN '{right}' b ON a.customer_id = b.id"
    ).fetchone()[0]
    con.close()
    assert f"yields {actual:,} rows" in out


def test_check_join_approves_a_clean_lookup(session, settings, join_dir):
    out = check_join(
        session, settings,
        (join_dir / "orders.parquet").as_posix(),
        (join_dir / "clean.parquet").as_posix(),
        ["customer_id"], ["id"],
    )
    assert "many-to-one (a safe lookup)" in out
    assert "Rows are duplicated" not in out
    assert "match nothing" not in out


def test_check_join_reports_unmatched_rows(session, settings, join_dir):
    out = check_join(
        session, settings,
        (join_dir / "orders.parquet").as_posix(),
        (join_dir / "partial.parquet").as_posix(),
        ["customer_id"], ["id"],
    )
    assert "500 left rows (50.0%) match nothing" in out
    assert "LEFT JOIN" in out


def test_check_join_reports_no_overlap(session, settings, join_dir):
    out = check_join(
        session, settings,
        (join_dir / "orphans.parquet").as_posix(),
        (join_dir / "clean.parquet").as_posix(),
        ["customer_id"], ["id"],
    )
    assert "No rows match at all" in out
    assert "share no values" in out


def test_check_join_explains_an_all_null_key(session, settings, join_dir):
    out = check_join(
        session, settings,
        (join_dir / "nullkeys.parquet").as_posix(),
        (join_dir / "clean.parquet").as_posix(),
        ["cust"], ["id"],
    )
    assert "No rows match at all" in out
    assert "NULL never matches" in out


def test_check_join_explains_a_type_mismatch(session, settings, join_dir):
    """DuckDB's own error names one stray value; the cause is the type pairing."""
    with pytest.raises(ToolError) as excinfo:
        check_join(
            session, settings,
            (join_dir / "strkeys.parquet").as_posix(),
            (join_dir / "clean.parquet").as_posix(),
            ["cust"], ["id"],
        )
    message = str(excinfo.value)
    assert "key types do not match" in message
    assert "VARCHAR" in message and "BIGINT" in message


def test_check_join_supports_composite_keys(session, settings, tmp_path):
    import duckdb

    con = duckdb.connect(":memory:")
    left = (tmp_path / "l.parquet").as_posix()
    right = (tmp_path / "r.parquet").as_posix()
    con.execute(
        f"COPY (SELECT i AS a, i % 3 AS b FROM range(30) t(i)) TO '{left}'"
    )
    con.execute(
        f"COPY (SELECT i AS a, i % 3 AS b FROM range(15) t(i)) TO '{right}'"
    )
    actual = con.sql(
        f"SELECT count(*) FROM '{left}' x JOIN '{right}' y ON x.a=y.a AND x.b=y.b"
    ).fetchone()[0]
    con.close()

    out = check_join(session, settings, left, right, ["a", "b"], ["a", "b"])
    assert f"yields {actual:,} rows" in out
    assert "one-to-one" in out


def test_check_join_rejects_unknown_columns(session, settings, join_dir):
    with pytest.raises(ToolError, match="no column"):
        check_join(
            session, settings,
            (join_dir / "orders.parquet").as_posix(),
            (join_dir / "clean.parquet").as_posix(),
            ["nope"], ["id"],
        )


def test_check_join_rejects_mismatched_key_counts(session, settings, join_dir):
    with pytest.raises(ToolError, match="line up"):
        check_join(
            session, settings,
            (join_dir / "orders.parquet").as_posix(),
            (join_dir / "clean.parquet").as_posix(),
            ["customer_id", "amount"], ["id"],
        )


def test_check_join_defaults_right_keys_to_left(session, settings, tmp_path):
    import duckdb

    con = duckdb.connect(":memory:")
    for name in ("a.parquet", "b.parquet"):
        con.execute(f"COPY (SELECT i AS id FROM range(5) t(i)) TO '{(tmp_path / name).as_posix()}'")
    con.close()
    out = check_join(
        session, settings,
        (tmp_path / "a.parquet").as_posix(), (tmp_path / "b.parquet").as_posix(), ["id"],
    )
    assert "one-to-one" in out


@pytest.fixture(scope="session")
def search_dir(tmp_path_factory):
    import duckdb

    directory = tmp_path_factory.mktemp("search")
    con = duckdb.connect(":memory:")
    con.execute(
        "COPY (SELECT i+1 AS id, 'cust-' || (i+1) AS name, 'ACME 50% off' AS note "
        f"FROM range(20) t(i)) TO '{(directory / 'customers.parquet').as_posix()}'"
    )
    parts = directory / "parts"
    parts.mkdir()
    for index in range(3):
        con.execute(
            f"COPY (SELECT {index}*100+i AS id, "
            f"CASE WHEN {index}=1 AND i=0 THEN 'NEEDLE' ELSE 'x' END AS tag "
            f"FROM range(5) t(i)) TO '{(parts / f'p{index}.parquet').as_posix()}'"
        )
    con.close()
    return directory


def test_find_value_locates_the_column(session, settings, search_dir):
    out = find_value(session, settings, (search_dir / "customers.parquet").as_posix(), "cust-7")
    assert "| name |" in out
    assert "cust-7" in row_for(out, "name")
    assert "| note |" not in out  # columns without a match are left out


def test_find_value_treats_percent_as_a_literal(session, settings, search_dir):
    """'%' is a LIKE wildcard; unescaped it would match every row."""
    path = (search_dir / "customers.parquet").as_posix()
    assert "| note |" in find_value(session, settings, path, "50%")
    assert "No column contains" in find_value(session, settings, path, "%zz%")


def test_find_value_searches_non_text_columns(session, settings, search_dir):
    out = find_value(session, settings, (search_dir / "customers.parquet").as_posix(), "17")
    assert "| id |" in out  # an INTEGER column, matched as text


def test_find_value_exact_mode(session, settings, search_dir):
    path = (search_dir / "customers.parquet").as_posix()
    assert "| name |" in find_value(session, settings, path, "cust-7", exact=True)
    # A substring alone must not satisfy an exact search.
    assert "No column contains" in find_value(session, settings, path, "cust", exact=True)


def test_find_value_reports_which_file(session, settings, search_dir):
    out = find_value(session, settings, (search_dir / "parts" / "*.parquet").as_posix(), "NEEDLE")
    assert "Found in 1 of 3 files" in out
    assert "p1.parquet" in out


def test_find_value_says_when_absent(session, settings, search_dir):
    out = find_value(session, settings, (search_dir / "customers.parquet").as_posix(), "zzz")
    assert "No column contains that value" in out


def test_find_value_restricts_to_named_columns(session, settings, search_dir):
    path = (search_dir / "customers.parquet").as_posix()
    out = find_value(session, settings, path, "7", columns=["name"])
    assert "| name |" in out
    assert "| id |" not in out
    with pytest.raises(ToolError, match="Unknown column"):
        find_value(session, settings, path, "7", columns=["nope"])


def test_find_value_rejects_an_empty_value(session, settings, search_dir):
    with pytest.raises(ToolError, match="must not be empty"):
        find_value(session, settings, (search_dir / "customers.parquet").as_posix(), "")


def test_join_across_formats(session, settings, data_dir):
    csv = (data_dir / "sales.csv").as_posix()
    parquet = (data_dir / "sales.parquet").as_posix()
    out = run_query(
        session,
        settings,
        f"SELECT count(*) AS n FROM '{csv}' a JOIN '{parquet}' b USING (id)",
    )
    assert "| 5 |" in out
