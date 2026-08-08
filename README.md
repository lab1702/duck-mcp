# duckdb-mcp

An MCP server that lets an AI assistant explore and query data files with
[DuckDB](https://duckdb.org) — CSV, Parquet, JSON/NDJSON, Excel, compressed
variants, globs, `https://` URLs and `s3://` buckets.

It is **strictly read-only**: only a single `SELECT` (including
`WITH`/`DESCRIBE`/`SUMMARIZE`/`SHOW`) or `EXPLAIN` statement runs per call.
`INSERT`, `CREATE`, `COPY … TO`, `ATTACH`, `SET` and friends are rejected
before they reach DuckDB, so no tool call can modify data, write files or
change server state. `EXPLAIN ANALYZE` is rejected too, because it runs the
statement it wraps.

Read-only is not the same as sandboxed. The server can read anything the user
account running it can read — `SELECT * FROM read_text('~/.aws/credentials')`
is a legitimate read — and it can fetch any URL. Run it as a user whose file
access you are happy to expose, and treat the contents of the files it reads as
untrusted input to whatever model is driving it.

## Install

The only thing to install is [uv](https://docs.astral.sh/uv/). Everything else
is fetched and cached automatically.

```powershell
winget install --id=astral-sh.uv -e     # Windows
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
```

### Claude Code

```bash
claude mcp add -s user duckdb -- uvx --from git+https://github.com/lab1702/duck-mcp duckdb-mcp
```

`-s user` registers the server for every project. Without it, `claude mcp add`
defaults to `local` scope and the server is only available in the directory you
ran the command in — it will not appear under `/mcp` anywhere else. Restart
Claude Code afterwards; `/mcp` reads the config at startup.

### Claude Desktop / other MCP clients

Add to your client's MCP config (`claude_desktop_config.json` for Claude
Desktop):

```json
{
  "mcpServers": {
    "duckdb": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/lab1702/duck-mcp",
        "duckdb-mcp"
      ]
    }
  }
}
```

No clone, no virtualenv, no `pip install`. To pick up a new release, run
`uvx --refresh --from git+https://github.com/lab1702/duck-mcp duckdb-mcp --version`.

## Tools

| Tool | What it does |
| --- | --- |
| `query(sql, max_rows=500)` | Run a read-only SQL statement, returns a markdown table. |
| `describe_file(path, include_row_count=True)` | Column names, types and row count for a file or glob. |
| `preview_file(path, rows=20)` | First N rows, so the model sees real values. |
| `list_files(path, pattern="*", recursive=False)` | List readable data files in a directory or `s3://` prefix. |
| `profile_columns(path, columns=None, top_k=5)` | Null counts, approximate distinct counts, min/max and most-frequent values. |

Files are referenced by path directly in SQL — there is no import or
registration step:

```sql
SELECT region, sum(amount) AS total
FROM 'data/sales_*.parquet'
WHERE order_date >= DATE '2026-01-01'
GROUP BY 1 ORDER BY total DESC
```

Joining across formats works the same way:

```sql
SELECT c.name, sum(s.amount)
FROM 'data/sales.parquet' s
JOIN 'data/customers.csv' c ON c.id = s.customer_id
GROUP BY 1
```

## Remote files

`https://` and `s3://` paths work through DuckDB's `httpfs` extension, which is
installed on first start. S3 authentication uses DuckDB's credential chain, so
`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, `~/.aws/credentials`, `AWS_PROFILE`
and instance roles are all picked up automatically. Public buckets and URLs
need no configuration.

Credentials are resolved once, at startup — if you add them afterwards, restart
the server. On a machine with no AWS credentials at all, that step is skipped
(DuckDB validates the chain when the secret is created) and public buckets still
work. Likewise, if the machine is offline at startup, extension setup is skipped
and local files still work. The server writes a line to stderr on startup
reporting which of `httpfs`, `excel` and the S3 credential chain came up.

## Limits

Defaults, all overridable:

| Setting | Default | Env var | Flag |
| --- | --- | --- | --- |
| Rows per result | 500 | `DUCKDB_MCP_MAX_ROWS` | `--max-rows` |
| Query timeout | 120s | `DUCKDB_MCP_TIMEOUT` | `--timeout` |
| Result text size | 200 KB | `DUCKDB_MCP_MAX_BYTES` | `--max-bytes` |
| Memory ceiling | DuckDB's own | `DUCKDB_MCP_MEMORY_LIMIT` | `--memory-limit` |

Truncated results say so explicitly, e.g. `(showing first 500 rows (more
available))`, including when a schema or profile table is cut short by the
size cap. The timeout covers a whole tool call, not each statement in it, so
a tool that runs several queries still finishes within it; a call that
overruns is cancelled and reported.

No row cap goes above 10,000, however it is set. A timeout of `0` means no
time limit.

The memory ceiling is left to DuckDB by default, which sizes it against system
memory — the right answer for the single instance this server runs. Setting it
is for the case DuckDB cannot see: MCP starts one server process per client, so
several concurrent sessions mean several DuckDB instances on one machine, each
holding its own independent ceiling. `--memory-limit` takes a size with a unit
(`4GB`, `512MB`); percentages are not accepted.

Unusable values are not silently accepted: a bad flag is a startup error, and
a bad environment variable is ignored with a line on stderr.

```json
{
  "mcpServers": {
    "duckdb": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/lab1702/duck-mcp", "duckdb-mcp",
               "--max-rows", "1000", "--timeout", "300"]
    }
  }
}
```

## Development

```bash
git clone https://github.com/lab1702/duck-mcp
cd duck-mcp
uv sync --extra dev
uv run pytest
```

Or without uv:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # .venv/bin/python on macOS/Linux
.venv/Scripts/python -m pytest
```

## License

MIT
