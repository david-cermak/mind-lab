## News → blog candidate pipeline

This folder has two scripts that work well together:

- **`check_news.py`**: generate a list of news/search results (Markdown or JSON).
- **`select_blog_candidates.py`**: ask an OpenAI‑compatible LLM to pick the best blog post ideas from that news list, using your prior posts as context.

---

## Requirements

- **Python**: 3.9+ recommended.
- **Python packages**:
  - For `check_news.py`:
    - `ddgs`
    - `python-dotenv` (optional, for loading a `.env` file)
  - For `select_blog_candidates.py`:
    - `openai`
    - `python-dotenv` (optional, for loading a `.env` file)

Install (example):

```bash
pip install ddgs python-dotenv openai
```

---

## 1) Generate news items (`check_news.py`)

`check_news.py` does keyword‑combination DuckDuckGo searches and prints results as **Markdown**, **JSON**, or **both**.

### Configuration (keywords)

Set keywords via environment variables or `.env`:

- **`TEXT_KEYWORDS`**: comma/newline-separated “text search” keywords
- **`NEWS_KEYWORDS`**: comma/newline-separated “news search” keywords
- **`NEWS_TARGET_RESULTS`**: total results to collect (default: 100)
- **`TEXT_MAX_KEYWORDS_PER_QUERY`**: max keywords per text query (default: 15)
- **`NEWS_MAX_KEYWORDS_PER_QUERY`**: max keywords per news query (default: 4)

Example `.env`:

```dotenv
TEXT_KEYWORDS=esp32, mqtt, fuzzing, post-quantum, tls, mbedtls
NEWS_KEYWORDS=esp32 security, mqtt broker, protocol fuzzing
NEWS_TARGET_RESULTS=80
```

### Generate a Markdown file for the selector

The selector expects `news2.md`‑style Markdown with sections like:

- `### 1. <title>`
- `- **URL**: ...`
- `- **Summary**: ...`
- `- **Date**: ...` (optional)

Generate that file:

```bash
python scripts/check_news.py -f md -o scripts/output/news2.md
```

---

## 2) Select blog candidates (`select_blog_candidates.py`)

`select_blog_candidates.py`:

- reads a news markdown file (default: `scripts/output/news2.md`)
- reads a “previous posts” file (default: `scripts/output/posts.txt`)
- chunks the news into batches and asks an LLM to select up to N candidates per chunk
- writes `selected.json` next to the news file by default (e.g. `scripts/output/selected.json`)

### OpenAI‑compatible API configuration

Configuration priority is: **CLI args → environment variables → `.env`**.

- **CLI args**:
  - `--base-url`
  - `--model`
  - `--api-key`
- **Environment variables** (same names also work in `.env`):
  - `OPENAI_BASE_URL` (or `BASE_URL`)
  - `OPENAI_MODEL` (or `MODEL`)
  - `OPENAI_API_KEY` (or `API_KEY`)

Example `.env`:

```dotenv
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini
OPENAI_API_KEY=sk-...
```

### Multi-source candidates (new)

A blog candidate can be based on **one** source item or a **combination of multiple** related items.

- **`BLOG_CANDIDATE_MAX_SOURCES`**: max number of source items per candidate (default: 3)
- CLI override: `--max-sources-per-candidate N`

### Hard exclusion: official Espressif docs

The selector will **not** select official Espressif documentation pages (notably anything on `docs.espressif.com`).

### Typical run

```bash
python scripts/select_blog_candidates.py \
  --news-file scripts/output/news2.md \
  --posts-file scripts/output/posts.txt
```

### Outputs (`selected.json`)

The output is JSON:

- **`candidates`**: list of selected ideas
- Each candidate includes:
  - **`item_ids`**: the source ITEM numbers (1..N)
  - **`source_title` / `url` / `link` / `summary`**: arrays aligned with `item_ids`
  - **`title`**: proposed blog post title
  - **`reason`**: why it fits your history
  - **`angle`**: suggested twist / experiment plan

Notes:

- The script auto-fills `source_title`/`url`/`summary` arrays from the original news items to keep them consistent.
- For convenience/back-compat, it also keeps `item_id` as the first entry of `item_ids`.


