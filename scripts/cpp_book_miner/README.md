# cpp_book_miner

Mine 1–4 attention-grabbing, minimal, Godbolt-ready C++ examples per chapter from `insights/programming/cpp.md`.

## Quick start

1) Configure:

```
export OPENAI_API_KEY=sk-...
```

Optional:

```
export OPENAI_BASE_URL=https://api.openai.com/v1
```

2) Run steps from repo root:

```
python -m scripts.cpp_book_miner.main split
python -m scripts.cpp_book_miner.main summarize
python -m scripts.cpp_book_miner.main summary
python -m scripts.cpp_book_miner.main generate
python -m scripts.cpp_book_miner.main render
python -m scripts.cpp_book_miner.main render-summary
python -m scripts.cpp_book_miner.main join --folder citations --ext json
python -m scripts.cpp_book_miner.main export
```

Outputs are written to `artifacts/cpp_mining/*` and exported posts to `insights/programming/{slug}/`.

The `summary` command generates a high-level chapter summary per chapter and writes
`<chapter_id>.summary.json` files into `artifacts/cpp_mining/summaries/`. Each file has:

- `title`: a short, compelling title for the chapter summary
- `learning_objective`: 1–3 sentences describing what the reader should learn
- `summary`: 3–8 paragraphs of prose (100–500 words total) covering the core ideas

The `render-summary` command turns those JSON summaries into human-readable markdown files:

- Input: `artifacts/cpp_mining/summaries/<chapter_id>.summary.json`
- Output: `artifacts/cpp_mining/summaries/<chapter_id>.summary.md`

You can restrict to a specific chapter with `--chapter <id>` or use `--first` to render only the first chapter.

The `join` command lets you combine many per-chapter files into a single file for a given folder and extension:

- Example (citations JSON): `python -m scripts.cpp_book_miner.main join --folder citations --ext json`
  - Input: `artifacts/cpp_mining/citations/*.json`
  - Output: `artifacts/cpp_mining/citations/all.json` (JSON array of per-chapter citation objects)
- Example (citations markdown): `python -m scripts.cpp_book_miner.main join --folder citations --ext md`
  - Input: `artifacts/cpp_mining/citations/*.md`
  - Output: `artifacts/cpp_mining/citations/all.md` (concatenated markdown with `---` separators)

- Example (candidates JSONL): `python -m scripts.cpp_book_miner.main join --folder candidates --ext jsonl`
  - Input: `artifacts/cpp_mining/candidates/*.jsonl`
  - Output: `artifacts/cpp_mining/candidates/all.jsonl` (concatenated JSON Lines)

The `--folder` argument can be a logical name (`citations`, `candidates`, `summaries`, `chapters`, `review`, `refined`)
or a path relative to the repo root. Use `--output some_name.json` to override the default `all.<ext>` name.

## Config

See `scripts/cpp_book_miner/config.yaml`. You can override `--config` with a custom file.

## Notes

- Uses Chat Completions API. Keep chapters reasonably sized; the tool uses the raw chapter markdown content for prompts.
- Candidates are stored per-chapter as JSONL in `artifacts/cpp_mining/candidates/`.


