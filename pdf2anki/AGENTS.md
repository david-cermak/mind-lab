# pdf2anki Developer Manual

## Architecture Overview

The pdf2anki pipeline converts PDF textbooks into Anki flashcards through a modular 7-stage process:

1. **PDF Extraction** - Extract images and text from PDF
2. **OCR Processing** - Run Tesseract OCR on images to get text labels with positions
3. **Content Analysis** - Plan card generation strategy (text vs occlusion)
4. **Vision Analysis** (optional) - Use LLM to semantically group OCR labels
5. **Text Card Generation** - Generate Q&A flashcards from text using LLM
6. **Image Occlusion Generation** - Create occlusion masks from grouped labels
7. **Deck Assembly** - Combine all cards into a single `.apkg` file

## Module Structure

```
pdf2anki/
├── extractors/          # Stage 1-2: PDF & OCR extraction
│   ├── pdf.py          # Extract images/text from PDF using PyMuPDF
│   └── ocr.py          # Run Tesseract OCR, filter low-confidence tokens
├── analysis/           # Stage 3-4: Content analysis & vision
│   ├── strategy.py     # Plan card generation (token counting, segmentation)
│   └── vision.py       # Vision LLM for semantic label grouping
├── generators/         # Stage 5-6: Card generation
│   ├── text.py         # Generate Q&A cards from text using LLM
│   └── occlusion.py    # Spatial grouping + occlusion rectangle generation
├── utils/              # Shared utilities
│   └── anki_db.py      # DeckBuilder class for .apkg creation
├── main.py             # Orchestrator (runs full pipeline)
└── debug_occlusion.py  # Debug tool for single image processing
```

## Key Components

### PDF Extraction (`extractors/pdf.py`)

- Uses PyMuPDF (fitz) to extract images and text per page
- Converts CMYK images to RGB
- Saves images as PNG with IDs: `page{N}_img{M}_xref{X}.png`
- Generates `metadata.jsonl` linking images to page text and positions

**Output:** `output/images/`, `output/text/`, `output/metadata.jsonl`

### OCR Processing (`extractors/ocr.py`)

- Runs Tesseract OCR with Czech language pack (`lang="ces"`)
- Extracts tokens with bounding boxes `[left, top, width, height]`
- Filters tokens below 75% confidence (removes noise like "|", "A7")
- Saves one JSON file per image: `output/ocr/{image_id}.json`

**Key Data Structure:**
```json
{
  "tokens": [
    {"text": "Stavba", "bbox": [748, 20, 173, 30], "confidence": 95.0},
    ...
  ]
}
```

### Content Strategy (`analysis/strategy.py`)

- Estimates token counts for text segments
- Decides card generation strategy:
  - `text_qa`: Generate N Q&A cards from text (1 per ~300 tokens)
  - `image_occlusion`: Create occlusion card if OCR tokens > 3
  - `image_visual`: Skip or describe-only for decorative images
- Outputs `plan.json` with execution plan

### Vision Analysis (`analysis/vision.py`)

**Purpose:** Use vision LLM to semantically group OCR labels for better occlusion masks.

**Status:** Currently available via debug tool only. Not yet integrated into main pipeline (see `main.py` lines 125-127).

**Input:**
- Image path
- Page text context (first 1000 chars)
- OCR tokens with positions

**Prompt Structure:**
1. System: "You are a helpful assistant analyzing educational diagrams. Output valid JSON."
2. User: 
   - Page text context
   - Full OCR tokens JSON with bounding boxes
   - Tasks: (1) Describe diagram, (2) Group labels semantically

**Output:**
```json
{
  "description": "The diagram shows...",
  "groups": [[0, 1, 3, 4, 5], [6, 7, 8], ...],
  "group_labels": ["Title group", "Left labels", ...]
}
```

**Token Estimation:**
- System: ~19 tokens
- Prompt text: ~1370 tokens (varies with OCR count)
- Image: ~85 + (base64_size / 4) tokens
- Total: ~412k tokens for 1.2MB image

### Text Card Generation (`generators/text.py`)

- Uses LLM to generate Q&A pairs from text segments
- Default model: `gpt-4o-mini` (cheaper)
- Output format: `[{"front": "...", "back": "..."}, ...]`
- Respects `PDF2ANKI_BASE_URL` and `OPENAI_API_KEY` from `.env`

### Image Occlusion (`generators/occlusion.py`)

**Spatial Grouping Algorithm:**
1. Filter tokens by confidence (default: ≥75%)
2. Sort tokens by position (top-to-bottom, left-to-right)
3. Group tokens within proximity threshold (default: 90px)
4. Consider alignment (same row/column detection)
5. Calculate bounding boxes with padding (default: 8px)
6. Normalize coordinates to [0,1] range

**Output:**
- Rectangles with pixel and normalized coordinates
- Anki markup: `{{c1::image-occlusion:rect:left=X:top=Y:width=W:height=H:oi=1}}`

**Current Implementation:**
- Uses spatial-only grouping (distance-based algorithm)
- Vision LLM semantic grouping is available via `debug_occlusion.py` but not yet integrated into main pipeline
- Future: Use vision LLM `groups` output to refine spatial groups for better semantic coherence

### Deck Builder (`utils/anki_db.py`)

**DeckBuilder Class:**
- `add_text_note(front, back, tags)` - Add Basic note type
- `add_occlusion_note(markup, image_path, header, back_extra, tags)` - Add Image Occlusion note
- `build()` - Creates SQLite DB + media files + ZIP archive

**Anki Database Schema:**
- `col` table: Collection config, models, decks
- `notes` table: Note content (fields separated by `\x1f`)
- `cards` table: Card instances linked to notes
- `media` JSON: Maps numeric IDs to filenames
- `meta` file: Version info (2 bytes: `\x08\x02`)

**Note Types:**
- Basic: Front/Back fields
- Image Occlusion: Occlusion markup, Image, Header, Back Extra, Comments

## Configuration

### Environment Variables (`.env`)

```bash
OPENAI_API_KEY=sk-...
PDF2ANKI_LLM_MODEL=gpt-4o-mini          # For text generation
PDF2ANKI_VISION_MODEL=gpt-4o            # For vision analysis
PDF2ANKI_BASE_URL=https://...           # Optional: LiteLLM proxy
PDF2ANKI_OUTPUT_DIR=output
PDF2ANKI_FINAL_APKG=result.apkg
PDF2ANKI_MAX_TEXT_CARDS=10
```

### CLI Arguments

**Main Pipeline:**
- `pdf_path` - Input PDF
- `--output-dir` - Intermediate files directory
- `--final-apkg` - Output deck path
- `--skip-ocr` - Reuse existing OCR results
- `--max-text-cards` - Limit text card generation
- `--llm-model`, `--vision-model`, `--api-key`, `--base-url` - Override .env

**Debug Tool:**
- `image_id` - Image ID to debug (e.g., `page3_img0_xref23`)
- `--output-dir` - Where to find metadata/OCR
- `--no-prompt` - Skip prompt preview
- `--call-llm` - Actually call vision LLM (default: preview only)

## Data Flow

```
PDF → [Extract] → images + text + metadata.jsonl
                ↓
              [OCR] → ocr/*.json (tokens with bboxes)
                ↓
            [Strategy] → plan.json (card generation plan)
                ↓
        ┌─────────────┴─────────────┐
        ↓                           ↓
   [Text Gen]                  [Occlusion]
   (LLM Q&A)              (Spatial grouping)
        ↓                           ↓
        └─────────────┬─────────────┘
                      ↓
                [Deck Builder]
                      ↓
                  result.apkg
```

## Debugging

### Single Image Debug

```bash
python -m pdf2anki.debug_occlusion page3_img0_xref23 --output-dir output
```

Shows:
- OCR tokens (filtered by confidence)
- Spatial grouping results
- Generated occlusion rectangles
- Vision LLM prompt preview with token estimates
- (Optional) Actual LLM response with semantic groups

### Common Issues

1. **Token estimation too high** - Image base64 encoding inflates size; actual API usage is lower
2. **Spatial groups too large** - Adjust `proximity_threshold` in `occlusion.py`
3. **Missing labels** - Check OCR confidence threshold (default 75%)
4. **API errors** - Verify `.env` has correct `PDF2ANKI_BASE_URL` and `OPENAI_API_KEY`

## Future Enhancements

- [ ] Use vision LLM groups to refine spatial grouping
- [ ] Multi-page text segmentation for better Q&A context
- [ ] Support for other note types (cloze, basic-optional-reversed)
- [ ] Batch processing with progress tracking
- [ ] Cost estimation before running pipeline

## Dependencies

- `PyMuPDF` - PDF extraction
- `Pillow` - Image processing
- `pytesseract` - OCR
- `openai` - LLM API client
- `python-dotenv` - Environment variable loading

## Testing

Run individual stages to test:
```bash
# Test extraction
python -m pdf2anki.extractors.pdf test.pdf --output-dir test-output

# Test OCR
python -m pdf2anki.extractors.ocr --metadata test-output/metadata.jsonl --output-dir test-output/ocr

# Test single image debug
python -m pdf2anki.debug_occlusion page1_img0_xref14 --output-dir test-output
```

