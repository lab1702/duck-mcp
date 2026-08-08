"""Runtime limits, overridable by environment variable or CLI flag."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MAX_ROWS = 500
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_BYTES = 200 * 1024


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """Caps applied to every tool call.

    A ``max_rows`` argument on a tool overrides the default per call, but
    ``max_bytes`` and ``timeout_seconds`` are hard limits on the server.
    """

    max_rows: int = DEFAULT_MAX_ROWS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_bytes: int = DEFAULT_MAX_BYTES

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            max_rows=_env_int("DUCKDB_MCP_MAX_ROWS", DEFAULT_MAX_ROWS),
            timeout_seconds=_env_float("DUCKDB_MCP_TIMEOUT", DEFAULT_TIMEOUT_SECONDS),
            max_bytes=_env_int("DUCKDB_MCP_MAX_BYTES", DEFAULT_MAX_BYTES),
        )
