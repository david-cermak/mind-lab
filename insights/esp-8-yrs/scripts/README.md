# Scripts

Utilities for collecting git commits across ESP repositories and displaying them.

## Typical workflow

```bash
# 1. Generate commit lists and summary (writes to repo root by default)
./scripts/generate_commit_lists.sh

# 2. Print non-merge commits with colors (~20 seconds total)
./scripts/print_commits.sh
```

## Scripts

### `generate_commit_lists.sh`

Bash wrapper around `generate_commit_lists.py`.

With no arguments, uses the default ESP setup:

- Repos: `ext/esp-idf`, `ext/esp-lwip`, `ext/esp-protocols`
- Emails: `cermak@espressif.com`, `david.cermak@espressif.com`
- Output: repo root (`list.txt`, `list_non_merge.txt`, `summary.md`)

Pass any arguments to override defaults and call the Python script directly:

```bash
./scripts/generate_commit_lists.sh --help
```

### `generate_commit_lists.py`

Main generator. Scans git history on the current ref for each repo (does not check out branches).

Matches commits where the **author or committer** email is in the given list. Deduplicates by commit title across repos (prefers `esp-lwip` > `esp-protocols` > `esp-idf`).

**Outputs:**

| File | Description |
|------|-------------|
| `list.txt` | All deduplicated commits |
| `list_non_merge.txt` | Same list, excluding titles starting with `Merge ` |
| `summary.md` | Area table, counts, and short descriptions |

**Line format:** `{title} {short-sha} {repo-label}`

**Common options:**

```bash
python3 scripts/generate_commit_lists.py \
  --repo path/to/repo:label \
  --ref label=branch-or-ref \
  --email user@example.com \
  --output-dir . \
  --dedupe-priority esp-lwip,esp-protocols,esp-idf \
  --no-summary
```

Requires: `python3`, `git`

### `print_commits.sh`

Prints commits from a list file one line at a time, paced to finish in about 20 seconds by default.

**Colors** (when stdout is a terminal):

- Title — default
- Short SHA — green
- Repo label — red

```bash
./scripts/print_commits.sh                          # list_non_merge.txt, 20s
./scripts/print_commits.sh list.txt 30              # custom file, 30s duration
```

Each line is expected to match: `{title} {8-char-sha} {repo}`
