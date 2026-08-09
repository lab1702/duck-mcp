"""Settings from the environment and the command line, including bad values."""

from __future__ import annotations

import pytest

from duckdb_mcp.config import (
    DEFAULT_MAX_ROWS,
    DEFAULT_MEMORY_LIMIT,
    DEFAULT_TIMEOUT_SECONDS,
    HARD_MAX_ROWS,
    Settings,
)
from duckdb_mcp.server import build_parser


def test_from_env_defaults(monkeypatch):
    for name in ("MAX_ROWS", "TIMEOUT", "MAX_BYTES", "MEMORY_LIMIT"):
        monkeypatch.delenv(f"DUCKDB_MCP_{name}", raising=False)
    settings = Settings.from_env()
    assert settings.max_rows == DEFAULT_MAX_ROWS
    assert settings.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert settings.memory_limit is DEFAULT_MEMORY_LIMIT is None  # DuckDB's own


def test_from_env_reads_overrides(monkeypatch):
    monkeypatch.setenv("DUCKDB_MCP_MAX_ROWS", "50")
    monkeypatch.setenv("DUCKDB_MCP_TIMEOUT", "5.5")
    monkeypatch.setenv("DUCKDB_MCP_MAX_BYTES", "4096")
    monkeypatch.setenv("DUCKDB_MCP_MEMORY_LIMIT", " 2GB ")
    settings = Settings.from_env()
    assert (settings.max_rows, settings.timeout_seconds, settings.max_bytes) == (50, 5.5, 4096)
    assert settings.memory_limit == "2GB"


def test_timeout_of_zero_is_kept(monkeypatch):
    """0 means no time limit, which is a real choice rather than a bad value."""
    monkeypatch.setenv("DUCKDB_MCP_TIMEOUT", "0")
    assert Settings.from_env().timeout_seconds == 0.0


@pytest.mark.parametrize(
    "name,value",
    [
        ("DUCKDB_MCP_MAX_ROWS", "-5"),
        ("DUCKDB_MCP_MAX_ROWS", "0"),
        ("DUCKDB_MCP_MAX_ROWS", "lots"),
        # Clamped to HARD_MAX_ROWS at every call site anyway; being told which
        # figure is in force beats watching 50,000 quietly behave as 10,000.
        ("DUCKDB_MCP_MAX_ROWS", "50000"),
        ("DUCKDB_MCP_TIMEOUT", "-1"),
        ("DUCKDB_MCP_TIMEOUT", "soon"),
        ("DUCKDB_MCP_MAX_BYTES", "-1"),
    ],
)
def test_unusable_env_values_fall_back_and_are_reported(monkeypatch, capsys, name, value):
    """A negative max_rows would otherwise make every query fail with LIMIT -4."""
    monkeypatch.setenv(name, value)
    settings = Settings.from_env()
    assert settings == Settings()
    assert value in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--max-rows", "-5"],
        ["--max-rows", "0"],
        ["--max-rows", "lots"],
        # The help text promises this ceiling, so the parser has to hold it.
        ["--max-rows", str(HARD_MAX_ROWS + 1)],
        ["--timeout", "-1"],
        ["--max-bytes", "0"],
    ],
)
def test_cli_rejects_unusable_values(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_cli_accepts_the_ceiling_itself():
    """The bound is inclusive: 10,000 is the largest cap, not the first bad one."""
    assert build_parser().parse_args(["--max-rows", str(HARD_MAX_ROWS)]).max_rows == HARD_MAX_ROWS


def test_cli_accepts_valid_values():
    args = build_parser().parse_args(
        ["--max-rows", "1000", "--timeout", "0", "--max-bytes", "1024", "--memory-limit", "8GB"]
    )
    assert (args.max_rows, args.timeout, args.max_bytes, args.memory_limit) == (
        1000,
        0.0,
        1024,
        "8GB",
    )
