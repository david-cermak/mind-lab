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
python -m scripts.cpp_book_miner.main generate
python -m scripts.cpp_book_miner.main render
python -m scripts.cpp_book_miner.main export
```

Outputs are written to `artifacts/cpp_mining/*` and exported posts to `insights/programming/{slug}/`.

## Config

See `scripts/cpp_book_miner/config.yaml`. You can override `--config` with a custom file.

## Notes

- Uses Chat Completions API. Keep chapters reasonably sized; the tool uses the raw chapter markdown content for prompts.
- Candidates are stored per-chapter as JSONL in `artifacts/cpp_mining/candidates/`.


