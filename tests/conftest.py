"""Shared fixtures: a session, settings, and a small data directory."""

from __future__ import annotations

import json

import duckdb
import pytest

from duckdb_mcp.config import Settings
from duckdb_mcp.db import DuckDBSession

ROWS = [
    (1, "West", 100.0, "2026-01-01"),
    (2, "East", 250.5, "2026-01-02"),
    (3, "West", None, "2026-01-03"),
    (4, "North", 75.25, "2026-01-04"),
    (5, "East", 310.0, "2026-01-05"),
]


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """A directory holding the same data as csv, parquet and ndjson."""
    directory = tmp_path_factory.mktemp("data")

    csv = directory / "sales.csv"
    lines = ["id,region,amount,order_date"]
    for row in ROWS:
        amount = "" if row[2] is None else row[2]
        lines.append(f"{row[0]},{row[1]},{amount},{row[3]}")
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    con = duckdb.connect(":memory:")
    con.execute(f"COPY (SELECT * FROM read_csv('{csv.as_posix()}')) TO '{(directory / 'sales.parquet').as_posix()}'")
    con.close()

    ndjson = directory / "events.ndjson"
    ndjson.write_text(
        "\n".join(json.dumps({"id": r[0], "region": r[1]}) for r in ROWS) + "\n",
        encoding="utf-8",
    )

    (directory / "notes.md").write_text("not a data file\n", encoding="utf-8")
    return directory


@pytest.fixture(scope="session")
def session():
    db = DuckDBSession(default_timeout=30.0)
    yield db
    db.close()


@pytest.fixture
def settings():
    return Settings(max_rows=500, timeout_seconds=30.0, max_bytes=200 * 1024)
