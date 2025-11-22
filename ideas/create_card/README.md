# Image Occlusion Card Builder

Utility script that converts OCR output (text tokens with pixel coordinates) into an Anki deck containing a single Image Occlusion card. It performs spatial grouping, optional LLM refinement, generates image-occlusion markup, and packages everything into a `.apkg` that can be imported into Anki.

## Features

- Filters low-confidence OCR tokens (default minimum confidence: 75%).
- Groups nearby tokens into occlusion regions using distance + alignment heuristics.
- Optional LLM refinement (`--use-llm`) to re-cluster tokens when a vision model is available.
- Emits both metadata JSON and a ready-to-import `.apkg` archive (SQLite DB, media map, image).

## Setup

Use the repo’s virtual environment or create a fresh one:

```bash
cd ideas
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # repo-wide deps
pip install -r create_card/requirements.txt  # script-specific extras
```

Set `OPENAI_API_KEY` (or `--llm-api-key`) if you plan to use `--use-llm`.

## Usage

```bash
cd /home/david/repos/mind-lab
source ideas/.venv/bin/activate
python ideas/create_card/create_occlusion_card.py \
  --ocr-json ideas/output/ocr/page3_img0_xref23.json \
  --image ideas/image_page3_img0_xref23.png \
  --output ideas/output/page3_img0_xref23.apkg \
  --json-output ideas/output/page3_img0_xref23.occlusion.json \
  --proximity-threshold 90 \
  --padding 8 \
  --deck-name "Roots Deck" \
  --header-text "Stavba kořene"
```

### Key Flags

- `--ocr-json`: OCR JSON with `tokens[]` entries.
- `--image`: PNG/JPEG referenced by the note.
- `--output`: Destination `.apkg`.
- `--json-output`: Optional metadata file (includes rectangles + markup).
- `--use-llm`: Ask a vision-capable model to regroup tokens (requires API access).
- `--proximity-threshold`: Pixel distance for spatial grouping (default 90).
- `--padding`: Extra pixels around each occlusion box (default 8).

Import the generated `.apkg` into Anki to review the card. Adjust thresholds/padding or re-run with `--use-llm` if the masks are too coarse/fine.*** End Patch

