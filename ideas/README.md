# Extraction Pipeline

This folder now contains a two-step pipeline to capture image, text, and OCR
metadata that can feed downstream flashcard tooling.

## 1. Extract page assets

```
cd ideas
source .venv/bin/activate
python pdf_test.py rostliny.pdf --output-dir output
```

Key outputs under `output/`:

- `images/*.png` – exported images named `page{n}_img{m}_xref{x}.png`
- `text/page_XXXX.txt` – full text per page (zero-padded numbering)
- `metadata.jsonl` – one JSON line per image with:
  - IDs, relative image paths, page numbers, xref references, pixel size
  - Bounding boxes returned by PyMuPDF (`page.get_image_rects`)
  - Plaintext for the page plus the immediately following page
  - Paths to the saved text files so other scripts can link assets

Use `--output-dir` to target a different folder when processing other PDFs.

## 2. OCR the extracted images

Install Tesseract and its Czech language data if you are working with ČJ
documents:

```
sudo apt install tesseract-ocr tesseract-ocr-ces
```

Then run:

```
python ocr_images.py --metadata output/metadata.jsonl --output-dir output/ocr
```

Each JSON file in `output/ocr/` contains:

- `detected_text` – concatenated OCR output
- `tokens[]` – text, confidence, and `[left, top, width, height]` per token
- Original image path and identifier (mirrors `metadata.jsonl`)

Re-run with `--skip-existing` to leave previously processed images untouched.

### OCR quality tips

- Provide the `ces` language pack to reduce errors such as *kofene* → *kořene*:
  `pytesseract.image_to_data(..., lang="ces")` (adjust in `ocr_images.py` if
  needed).
- If diagrams include mixed languages, pass `lang="ces+eng"` to keep English
  labels.
- For hard-to-read scans, consider denoising images (OpenCV, Pillow filters) or
  switching to PaddleOCR/DocTR models that handle accents better.

