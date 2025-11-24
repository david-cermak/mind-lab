# pdf2anki

Minimal pipeline turning a PDF into an Anki deck (text + image occlusion cards).

## Quick Start
| Step | Command | Result |
| --- | --- | --- |
| 1. Configure env | `cp pdf2anki/.env.example .env && edit` | `.env` with API + defaults |
| 2. Install deps | `python -m venv .venv && source .venv/bin/activate && pip install -r pdf2anki/requirements.txt` | Ready virtualenv |
| 3. Run full pipeline | `python -m pdf2anki.main pdfs/book.pdf --output-dir output --final-apkg book.apkg` | `output/` assets + `book.apkg` deck |

### Running Individual Stages

If the earlier stages already produced intermediates, you can re-run later steps only:

| Stage | Command | Notes |
| --- | --- | --- |
| Extract PDF assets | `python -m pdf2anki.extractors.pdf pdfs/book.pdf --output-dir output` | Produces `output/images`, `output/text`, `output/metadata.jsonl` |
| OCR images | `python -m pdf2anki.extractors.ocr --metadata output/metadata.jsonl --output-dir output/ocr` | Generates one OCR JSON per image |
| Plan cards | `python -m pdf2anki.analysis.strategy --metadata output/metadata.jsonl --ocr-dir output/ocr --output output/plan.json` | Decides how many text/occlusion cards to make |
| Generate text cards only | `python -m pdf2anki.generate_text_cards --output-dir output --final-apkg output/text_cards.apkg` | Builds a deck with text notes only and writes `output/text_cards.json` |
| Vision analysis (debug only) | `python -m pdf2anki.debug_occlusion <image_id> --output-dir output --call-llm` | Test vision LLM semantic grouping (not yet integrated into main pipeline) |
| Regenerate deck only | `python -m pdf2anki.main pdfs/book.pdf --output-dir output --final-apkg book.apkg --skip-ocr --max-text-cards 0` | Reuses existing assets, skips OCR/text cards as needed. Uses spatial grouping for occlusion (distance-based algorithm) |

- **Input:** Any PDF textbook/notes.
- **Output:** Intermediate assets under `output/` and a final `.apkg` deck ready for Anki import.

### Debugging Image Occlusion

To debug a single image card and see what prompts would be sent (without calling LLM):

```bash
python -m pdf2anki.debug_occlusion page3_img0_xref23 --output-dir output
```

This shows:
- OCR tokens found
- Spatial grouping results
- Generated occlusion rectangles
- Vision LLM prompt preview (what would be sent)

To actually call the vision LLM:
```bash
python -m pdf2anki.debug_occlusion page3_img0_xref23 --output-dir output --call-llm
```

### Debugging Text Cards

After running the text-only target, you can preview the generated Q&A pairs without opening the `.apkg`:

```bash
python -m pdf2anki.debug_text_cards --output-dir output
```

Use `--regenerate` to force new LLM calls or `--cards-json /path/to/text_cards.json` to point at a specific cache.
