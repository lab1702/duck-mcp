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


def registered() -> dict[str, str]:
    """Each tool's signature, rendered the way the README writes it."""

    async def collect():
        return await mcp.list_tools()

    literals = {None: "None", True: "True", False: "False"}
    signatures = {}
    for tool in anyio.run(collect):
        schema = tool.input_schema
        required = set(schema.get("required", []))
        signatures[tool.name] = ", ".join(
            name
            if name in required
            else f"{name}={literals.get(prop.get('default'), json.dumps(prop.get('default')))}"
            for name, prop in schema.get("properties", {}).items()
        )
    return signatures


def test_readme_documents_every_tool():
    assert set(documented()) == set(registered())


@pytest.mark.parametrize("name", sorted(registered()))
def test_readme_signature_matches(name):
    assert documented()[name] == registered()[name]
