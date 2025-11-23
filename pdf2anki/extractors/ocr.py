import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, List

import pytesseract
from PIL import Image
from pytesseract import Output, TesseractNotFoundError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR over extracted images and save token-level metadata."
    )
    parser.add_argument(
        "--metadata",
        default="output/metadata.jsonl",
        help="Path to the metadata JSONL produced by pdf_test.py (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/ocr",
        help="Directory where OCR JSON files will be stored (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip OCR for images that already have a JSON result in the output directory.",
    )
    return parser.parse_args()


def _iter_metadata(metadata_path: Path) -> Iterator[Dict]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def _ensure_tesseract_available() -> None:
    try:
        pytesseract.get_tesseract_version()
    except TesseractNotFoundError as exc:  # pragma: no cover - configuration guard
        raise SystemExit(
            "Tesseract OCR binary not found. Install it (e.g., `sudo apt install tesseract-ocr`)."
        ) from exc


def _ocr_image(image_path: Path) -> Dict:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        data = pytesseract.image_to_data(image, output_type=Output.DICT, lang="ces")

    tokens: List[Dict] = []
    combined_text_parts: List[str] = []
    for idx, text in enumerate(data["text"]):
        cleaned = text.strip()
        if not cleaned:
            continue
        conf_raw = data["conf"][idx]
        try:
            confidence = float(conf_raw)
        except (TypeError, ValueError):
            confidence = -1.0
        bbox = [
            int(data["left"][idx]),
            int(data["top"][idx]),
            int(data["width"][idx]),
            int(data["height"][idx]),
        ]
        tokens.append({"text": cleaned, "confidence": confidence, "bbox": bbox})
        combined_text_parts.append(cleaned)

    return {
        "detected_text": " ".join(combined_text_parts),
        "tokens": tokens,
    }


def run_ocr(metadata_path: Path, output_dir: Path, skip_existing: bool) -> None:
    _ensure_tesseract_available()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = metadata_path.parent
    results = 0
    skipped = 0

    for record in _iter_metadata(metadata_path):
        image_rel_path = record.get("image_path")
        image_id = record.get("id")
        if not image_rel_path or not image_id:
            continue

        image_path = (metadata_dir / image_rel_path).resolve()
        if not image_path.exists():
            print(f"[WARN] Missing image for {image_id}: {image_path}")
            continue

        output_path = output_dir / f"{image_id}.json"
        if skip_existing and output_path.exists():
            skipped += 1
            continue

        ocr_payload = _ocr_image(image_path)
        ocr_payload.update(
            {
                "id": image_id,
                "image_path": str(image_path),
            }
        )
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(ocr_payload, handle, ensure_ascii=False, indent=2)

        results += 1

    print(
        f"OCR complete. Generated {results} files in {output_dir}."
        + (f" Skipped {skipped} existing files." if skipped else "")
    )


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    run_ocr(metadata_path, output_dir, args.skip_existing)


if __name__ == "__main__":
    main()

