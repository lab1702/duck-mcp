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

No clone, no virtualenv, no `pip install`. `uvx` re-resolves the repository each
time it starts the server, so a new session already runs the current `main` —
there is no separate update step.

The same re-resolution means the server will not start without a network. A
warm cache does not save you, and neither does pinning the URL to a commit:
`uvx` contacts GitHub before it runs anything, and exits with a git error
instead of starting the server. If you need it to work offline, do a
[development](#development) checkout — that runs from a local environment with
nothing to fetch — and point your MCP config at `.venv/Scripts/duckdb-mcp`
(`.venv/bin/duckdb-mcp` on macOS/Linux) instead of `uvx`.

## Tools

Finding and reading data:

| Tool | What it does |
| --- | --- |
| `query(sql, max_rows=None)` | Run a read-only SQL statement, returns a markdown table. Defaults to the server's row cap (500). |
| `list_files(path=".", pattern="*", recursive=False, data_files_only=True)` | List files in a directory or `s3://` prefix. Only readable data formats unless `data_files_only=False`. |
| `describe_file(path, include_row_count=True)` | Column names, types and row count for a file or glob. |
| `preview_file(path, rows=20)` | First N rows, so the model sees real values. |
| `sample_rows(path, rows=20, seed=None)` | A *random* sample of rows, for files whose head is not representative. |
| `profile_columns(path, columns=None, top_k=5)` | Null counts, approximate distinct counts, min/max and most-frequent values. |
| `find_value(path, value, columns=None, exact=False)` | Which columns — and which files — contain a value. |

Checking that an answer is the right one. Most of what goes wrong with a data
file is not an error — it is a plausible number that happens to be wrong. These
four look for that:

| Tool | What it does |
| --- | --- |
| `inspect_raw(path, lines=20)` | Raw lines of a text file, before parsing, plus what the CSV sniffer detected. |
| `compare_schemas(path, max_files=None)` | Compare schemas across a glob and say what a plain read does about the differences. Reads up to 100 files. |
| `check_join(left, right, left_on, right_on=None)` | What a join will do before you run it: fan-out, match rates, orphans. |
| `check_coverage(path, column, granularity=None)` | Missing and repeated values in a column that should run in regular steps. |
| `parquet_metadata(path, row_groups=False)` | Parquet layout from the footer: sizes, compression, row-group pruning — no scan. |

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

### Parquet layout

Everything else here reads data. `parquet_metadata` reads the footer, so it
answers layout questions on a file far too large to profile — and answers
"how many rows?" for free, since parquet records that in metadata:

```
**events/*.parquet** — 12 parquet file(s), 4,500,000 rows, 45 row group(s), 210.5MB on disk

| column | type | codec | compressed | ratio | nulls | row_group_order |
| ------ | ---- | ----- | ---------- | ----- | ----- | --------------- |
| id     | INT64      | SNAPPY | 1.1MB  | 2.0x  | 0       | ascending       |
| ts     | INT64      | SNAPPY | 1.7MB  | 1.3x  | 0       | ascending       |
| label  | BYTE_ARRAY | SNAPPY | 1.4MB  | 3.4x  | 0       | ascending       |
| bucket | INT64      | SNAPPY | 6.7KB  | 16.7x | 0       | scattered (9/9) |
| notes  | BYTE_ARRAY | SNAPPY | 290B   | 0.9x  | 300,000 | no stats        |
```

`row_group_order` is the useful part. A parquet reader skips a row group when
its recorded min/max cannot match the filter, so a range filter is cheap on a
column whose row-group ranges climb through the file (`ascending`) and reads
everything on one whose ranges overlap (`scattered`). Above, filtering on `ts`
prunes; filtering on `bucket` — which cycles through the same seven values in
every row group — cannot. `no stats` means the column records no min/max at
all, so nothing can be inferred either way.

Comparisons are numeric when both bounds parse as numbers and lexicographic
otherwise, which is what parquet itself does for strings and matches the
ISO-8601 text DuckDB gives timestamps.

The tool also flags layouts that make scans cost more than the data warrants —
undersized row groups, or a glob of many small files:

```
Note: row groups average only 1,000 rows. Small row groups add per-group
overhead and give the reader less to parallelise over; 128MB-ish groups are
the usual target.

Note: 4 files averaging 4.3KB each. Many small files cost one open per file,
which dominates on object storage.
```

`row_groups=True` adds a per-row-group table of row counts and sizes, for
finding uneven splits. Actual value ranges are not reported here — footer
statistics are approximate by design; use `profile_columns` for real min/max.

### Schema drift across a glob

Reading `data/*.parquet` when the files disagree is not reliably an error.
DuckDB takes its schema from the first file and reconciles the rest against
it, so two of the four possible outcomes are silent and wrong:

| difference | what a plain read does |
| --- | --- |
| a later file adds a column | **drops it** — no error, the column simply is not there |
| a later file widens a type | **narrows the values** — `10.5` comes back as `10` |
| a later file drops a column | read fails |
| the types cannot reconcile | read fails |

`compare_schemas` reports which of those you are in:

```
**data/*.parquet** — 3 files, 2 distinct schemas across 3 compared

A plain read takes its schema from the first file, `a.parquet`: id INTEGER, amount INTEGER

| column | files | types                   | a plain read                              |
| ------ | ----- | ----------------------- | ----------------------------------------- |
| id     | 3/3   | INTEGER                 | ok                                        |
| amount | 3/3   | INTEGER (2), DOUBLE (1) | narrows DOUBLE to INTEGER — values are lost |
| extra  | 1/3   | VARCHAR                 | drops it — absent from a.parquet          |
```

Whether a difference is harmful depends on direction, so the tool asks DuckDB
rather than guessing: the common supertype of the two types is the one that
loses nothing, so a first file whose type is *not* that supertype must be
narrowing. `DOUBLE` first and `INTEGER` later is fine and reported as such;
`INTEGER` first and `DOUBLE` later loses your decimals.

The fix in every case is to reconcile by name, which the tool prints with the
reader matching your format:

```sql
SELECT * FROM read_parquet('data/*.parquet', union_by_name=true)
```

Schemas are read one file at a time, so cost is linear in the file count.
Above `max_files` (default 100) it compares a spread across the glob rather
than the first N — drift tends to track write order, so the first N files
would be the oldest and would miss exactly the recent change worth finding.
The first and last file are always included. Files that cannot be read at all
are reported rather than skipped silently.

### Checking a join before trusting it

If the right side of a join holds more than one row per key, the join
duplicates left rows and every aggregate over the result is inflated. There is
no error, and the total still looks plausible:

```
sum(amount) alone      = 10000.0
sum(amount) after join = 11000.0     <- 20 duplicate customer ids
```

`check_join` reports that from grouped key counts, without materialising the
join:

```
**orders.parquet** ⋈ **customers.parquet** on customer_id = id

| side  | rows  | distinct keys | max rows per key | matched        | unmatched | null keys |
| ----- | ----- | ------------- | ---------------- | -------------- | --------- | --------- |
| left  | 1,000 | 200           | 5                | 1,000 (100.0%) | 0         | 0         |
| right | 220   | 200           | 2                | 220 (100.0%)   | 0         | 0         |

Relationship: many-to-many — up to 5 left rows and 2 right rows per key.
The join yields 1,100 rows from 1,000 on the left.
```

Direction is what matters: five orders per customer is normal, two customer
rows per id is the bug. So `many-to-one` is reported as a safe lookup while
`many-to-many` gets a warning. The predicted row count is exact — a test
asserts it against the join it declined to run.

It also reports rows that match nothing, so you can see an inner join dropping
half your data, and diagnoses a join returning nothing at all: keys that are
entirely NULL, or key columns whose types cannot be compared.

### Gaps in a series

A daily table missing three days still sums and averages perfectly happily —
the total is just quietly short. One where a day was loaded twice sums too
high. Neither shows up as an error, and neither changes anything a row count
would reveal:

```
count(*) = 29, sum = 290       <- looks entirely fine
```

`check_coverage` infers the step from the data and reports what is absent:

```
29 rows, 28 distinct values from 2026-01-01 to 2026-01-31.
Step looks like 1 day (25 of 27 intervals), so 31 values were expected.

3 missing (9.7% of the expected range) across 2 gaps.

| after      | before     | missing |
| ---------- | ---------- | ------- |
| 2026-01-08 | 2026-01-11 | 2       |
| 2026-01-24 | 2026-01-26 | 1       |

1 repeated value. A sum over this column double-counts them.

| value      | rows |
| ---------- | ---- |
| 2026-01-05 | 2    |
```

The step is inferred rather than assumed, so the same tool covers dates,
timestamps and plain integer sequences. Calendar steps are not constant — a
month is 28 to 31 days — so an interval counts as a gap only once it is half
again the usual step, which finds a missing month without reporting February
as a hole every year.

For timestamps carrying a time of day that really describe a daily series,
pass `granularity='day'` (or `hour`, `month`, …) to bucket them first.

### Finding where a value lives

```
find_value('data/*.parquet', 'NEEDLE')
```

Searches every column as text — numbers, dates and nested values included —
and reports which columns match, with an example, plus which files they are
in. Useful for locating a join key in a dataset whose schema you do not know
yet, which is exactly when you cannot write the query yourself. `%` and `_` in
the search value are matched literally rather than as wildcards.

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

Because all of that happens at startup, a read that fails later says nothing
about the real cause. So when something set up at startup is missing, the error
carries the reason with it:

```
HTTP Error: HTTP GET error reading 'https://bucket.s3.amazonaws.com/data.parquet'
in region '' (HTTP 403 Forbidden) AccessDenied: Access Denied

No AWS credentials were resolved when the server started (unavailable: Secret
Validation Failure: ... Credential Chain: 'config'), so this bucket is being read
anonymously — which is what a 403 here usually means. DuckDB resolves credentials
once at startup, so setting them now requires a restart.
```

The same applies to an `https://` or `s3://` path when `httpfs` failed to
install, and to `.xlsx` when `excel` did. The hint is added only when the
capability is actually missing *and* the failure is the kind it would explain —
a 404 on S3 is a missing object, not a credentials problem, and gets no hint.

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
