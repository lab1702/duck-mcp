"""The README's tool table, checked against the tools actually registered.

Hand-maintaining that table across a few commits let three signatures drift --
a default written as its effective value rather than its real one, and a
parameter left out entirely. It is cheaper to check it than to re-read it.
"""

from __future__ import annotations

import json
import pathlib
import re

import anyio
import pytest

from duckdb_mcp.server import mcp

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"

# Rows look like: | `tool_name(args)` | description |
_ROW = re.compile(r"^\| `(\w+)\(([^`]*)\)` \|", re.MULTILINE)


def documented() -> dict[str, str]:
    return dict(_ROW.findall(README.read_text(encoding="utf-8")))


def _literal(value) -> str:
    """A default, written the way the README writes it.

    Python spells None and the booleans differently from JSON, so those go
    through ``repr``. Not through a lookup table keyed by the values
    themselves: ``0 == False`` and ``1 == True`` hash alike, so a ``top_k=0``
    default -- which the tool's own docstring invites -- would come back as
    ``top_k=False`` and the check would demand the README say so.
    """
    if value is None or isinstance(value, bool):
        return repr(value)
    return json.dumps(value)


def registered() -> dict[str, str]:
    """Each tool's signature, rendered the way the README writes it."""

    async def collect():
        return await mcp.list_tools()

    signatures = {}
    for tool in anyio.run(collect):
        schema = tool.input_schema
        required = set(schema.get("required", []))
        signatures[tool.name] = ", ".join(
            name if name in required else f"{name}={_literal(prop.get('default'))}"
            for name, prop in schema.get("properties", {}).items()
        )
    return signatures


def test_literal_does_not_confuse_zero_with_false():
    """`0 == False` and `1 == True` hash alike, so a lookup keyed by them collides."""
    assert _literal(0) == "0"
    assert _literal(1) == "1"
    assert _literal(False) == "False"
    assert _literal(True) == "True"
    assert _literal(None) == "None"
    assert _literal("*") == '"*"'


def test_readme_documents_every_tool():
    assert set(documented()) == set(registered())


@pytest.mark.parametrize("name", sorted(registered()))
def test_readme_signature_matches(name):
    assert documented()[name] == registered()[name]
