"""Runtime limits, overridable by environment variable or CLI flag."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from .db import log

DEFAULT_MAX_ROWS = 500
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_BYTES = 200 * 1024

# None leaves DuckDB's own per-instance limit in place, which is the right
# default: this server runs exactly one instance, which is the case that limit
# is designed for. See Settings.memory_limit for when overriding it earns its
# keep.
DEFAULT_MEMORY_LIMIT: str | None = None

# Ceiling on any row cap, whoever asks for it -- the caller via a tool argument
# or the operator via --max-rows. 10,000 markdown rows is already well past what
# max_bytes will emit, so this exists to stop a runaway cap pulling an arbitrary
# result into memory, not to shape normal output.
HARD_MAX_ROWS = 10_000


def _env_number(
    name: str,
    default: Any,
    cast: Callable[[str], Any],
    minimum: float,
    maximum: float | None = None,
) -> Any:
    """Read a numeric setting, falling back to ``default`` on anything unusable.

    A bad value is reported rather than silently accepted: ``--max-rows -5``
    would otherwise turn every query into a ``LIMIT/OFFSET cannot be negative``
    error pointing at DuckDB instead of at the misconfiguration. A value over
    the ceiling is reported for the same reason -- it is held to
    ``HARD_MAX_ROWS`` at every call site regardless, and being told that beats
    watching a cap of 50,000 quietly behave as 10,000.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        log(f"ignoring {name}={raw!r}: not a number; using {default}")
        return default
    if value < minimum:
        log(f"ignoring {name}={raw!r}: must be at least {minimum}; using {default}")
        return default
    if maximum is not None and value > maximum:
        log(f"ignoring {name}={raw!r}: must be at most {maximum}; using {default}")
        return default
    return value


def _env_int(name: str, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    return int(_env_number(name, default, int, minimum, maximum))


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    return float(_env_number(name, default, float, minimum))


def _env_str(name: str, default: str | None) -> str | None:
    raw = os.environ.get(name)
    return raw.strip() if raw and raw.strip() else default


@dataclass
class Settings:
    """Caps applied to every tool call.

    A ``max_rows`` argument on a tool overrides the default per call (up to
    ``HARD_MAX_ROWS``); ``max_bytes`` and ``timeout_seconds`` are hard limits on
    the server. ``timeout_seconds`` of 0 means no time limit.

    ``memory_limit`` of ``None`` -- the default -- leaves DuckDB's own limit
    alone. It is worth setting only for what DuckDB cannot see: MCP starts one
    server process per client, so several concurrent sessions put several
    DuckDB instances on one machine, each holding its own independent ceiling.
    The value goes straight to ``SET memory_limit``, so it takes a size with a
    unit (``'4GB'``, ``'512MB'``); percentages are not accepted.
    """

    max_rows: int = DEFAULT_MAX_ROWS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_bytes: int = DEFAULT_MAX_BYTES
    memory_limit: str | None = DEFAULT_MEMORY_LIMIT

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            max_rows=_env_int("DUCKDB_MCP_MAX_ROWS", DEFAULT_MAX_ROWS, maximum=HARD_MAX_ROWS),
            timeout_seconds=_env_float("DUCKDB_MCP_TIMEOUT", DEFAULT_TIMEOUT_SECONDS),
            max_bytes=_env_int("DUCKDB_MCP_MAX_BYTES", DEFAULT_MAX_BYTES),
            memory_limit=_env_str("DUCKDB_MCP_MEMORY_LIMIT", DEFAULT_MEMORY_LIMIT),
        )
