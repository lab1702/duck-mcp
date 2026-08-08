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
| `sample_rows(path, rows=20, seed=None)` | A *random* sample of rows, for files whose head is not representative. |
| `inspect_raw(path, lines=20)` | Raw lines of a text file, before parsing, plus what the CSV sniffer detected. |
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

### Sampling instead of the head

`preview_file` returns the first rows. For a file written in time or partition
order that is systematically unrepresentative — one date, one region, and often
the oldest and most schema-drifted records in the set. `sample_rows` draws a
uniform random sample instead:

```
sample_rows('data/events_2026.parquet', rows=20, seed=42)
```

That costs a full scan, which the head does not, so it is the right tool for
understanding a file rather than for glancing at one. Passing `seed` fixes the
draw, so a follow-up call revisits the same rows.

### When the parse looks wrong

Every other tool reads through DuckDB's CSV/JSON parser, so if auto-detection
misreads a file there is nothing to check its answer against — the result is
plausible-looking but wrong data, with no error. DuckDB's sniffer is good, but
a report footer is enough to defeat it:

```
id,name,amount
1,alice,5
2,bob,6
-- end of report --
```

`describe_file` reports **one** `VARCHAR` column named `id,name,amount`, three
rows, no error. `inspect_raw` shows the lines next to what the sniffer concluded:

```text
1 | id,name,amount
2 | 1,alice,5
3 | 2,bob,6
4 | -- end of report --
```

> DuckDB's CSV sniffer reads this as: delimiter `','`, skip 0 row(s), header
> row yes, **3 column(s)**.

Three columns against the one `describe_file` produced — the disagreement is
the diagnosis. `inspect_raw` also prints the sniffer's own `read_csv` call,
which here is already the fix, and can be pasted straight into `query`:

```sql
FROM read_csv('report.csv', auto_detect=false, delim=',', header=true,
              columns={'id': 'BIGINT', 'name': 'VARCHAR', 'amount': 'BIGINT'},
              ignore_errors=true);   -- returns the 2 real rows
```

`inspect_raw` reads only as many lines as you ask for, so it is safe on a
multi-gigabyte file, and it tolerates input a real parse rejects — mixed line
endings, unterminated quotes, ragged rows. Tabs, carriage returns and other
invisible characters are escaped so they can be seen. It is text-only; parquet
and Excel are rejected with a pointer to `describe_file`.

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
uv sync --extra dev --upgrade
uv run pytest
```

There is no committed `uv.lock`. Dependencies are declared with lower bounds
only, so a fresh install — and `uvx` — resolves to the current `duckdb` and
`mcp` releases. `--upgrade` re-resolves an existing checkout to the latest
versions; run the tests after, since tracking upstream means meeting its
breaking changes early rather than at a pinned upgrade later.

Or without uv:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # .venv/bin/python on macOS/Linux
.venv/Scripts/python -m pytest
```

## License

MIT
