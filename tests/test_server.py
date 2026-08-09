"""How the async tool handlers hand work to a thread."""

from __future__ import annotations

import threading

import anyio

from duckdb_mcp import server


def test_call_resolves_the_session_off_the_event_loop(monkeypatch):
    """Building the session installs extensions over the network.

    On the event loop that blocks the whole server -- pings included -- for
    however long the install takes, which on an offline machine is until the
    connection attempt gives up. Passing ``get_session()`` as an argument, even
    to ``functools.partial``, evaluates it there; only a closure defers it.
    """
    seen: dict[str, int] = {}

    def fake_get_session():
        seen["session"] = threading.get_ident()
        return "session"

    def tool(session, settings, value):
        seen["tool"] = threading.get_ident()
        return f"{session}:{settings.max_rows}:{value}"

    monkeypatch.setattr(server, "get_session", fake_get_session)

    async def main():
        seen["loop"] = threading.get_ident()
        return await server._call(tool, "x")

    result = anyio.run(main)
    assert result == f"session:{server.settings.max_rows}:x"
    assert seen["session"] == seen["tool"] != seen["loop"]


def test_get_session_is_built_once_under_concurrent_first_calls(monkeypatch):
    """It is reached from worker threads now, so the check-then-set needs a lock."""
    built = []

    class FakeSession:
        def __init__(self, **kwargs):
            built.append(kwargs)
            self.capabilities = {}

    monkeypatch.setattr(server, "_session", None)
    monkeypatch.setattr(server, "DuckDBSession", FakeSession)

    barrier = threading.Barrier(8)
    results = []

    def race():
        barrier.wait()
        results.append(server.get_session())

    threads = [threading.Thread(target=race) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(built) == 1
    assert len({id(session) for session in results}) == 1
