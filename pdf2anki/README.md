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
| Vision check (optional) | `python -m pdf2anki.analysis.vision --image ...` | Use ad-hoc to inspect diagram suitability |
| Regenerate deck only | `python -m pdf2anki.main pdfs/book.pdf --output-dir output --final-apkg book.apkg --skip-ocr --max-text-cards 0` | Reuses existing assets, skips OCR/text cards as needed |

- **Input:** Any PDF textbook/notes.
- **Output:** Intermediate assets under `output/` and a final `.apkg` deck ready for Anki import.
