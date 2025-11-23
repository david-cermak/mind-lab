import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Set

import fitz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract images and surrounding text from a PDF."
    )
    parser.add_argument(
        "pdf_path",
        nargs="?",
        default="rostliny.pdf",
        help="Path to the source PDF (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where images, text, and metadata will be stored (default: %(default)s)",
    )
    return parser.parse_args()


def _ensure_text_file(
    page_num: int, text: str, destination: Path, written_pages: Set[int]
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    text_path = destination / f"page_{page_num:04d}.txt"
    if page_num not in written_pages:
        text_path.write_text(text, encoding="utf-8")
        written_pages.add(page_num)
    return text_path


def _bbox_list(rects: Iterable[fitz.Rect]) -> List[List[float]]:
    return [[rect.x0, rect.y0, rect.x1, rect.y1] for rect in rects]


def extract_assets(pdf_path: Path, output_dir: Path) -> None:
    doc = fitz.open(pdf_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    text_dir = output_dir / "text"
    metadata_path = output_dir / "metadata.jsonl"

    written_pages: Set[int] = set()
    total_images = 0

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1
            page_text = (page.get_text("text") or "").strip()
            next_page_text = ""
            next_page_number = None
            if page_index + 1 < len(doc):
                next_page = doc[page_index + 1]
                next_page_text = (next_page.get_text("text") or "").strip()
                next_page_number = page_num + 1

            page_text_file = _ensure_text_file(
                page_num, page_text, text_dir, written_pages
            )
            next_page_text_file: Optional[Path] = None
            if next_page_number is not None:
                next_page_text_file = _ensure_text_file(
                    next_page_number, next_page_text, text_dir, written_pages
                )

            images = page.get_images(full=True)
            if not images:
                continue

            for img_index, img in enumerate(images):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                width, height = pix.width, pix.height
                if pix.n >= 5:  # CMYK or similar, convert to RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                image_id = f"page{page_num}_img{img_index}_xref{xref}"
                image_path = images_dir / f"{image_id}.png"
                pix.save(image_path)
                pix = None  # Free memory

                rects = page.get_image_rects(xref)
                record = {
                    "id": image_id,
                    "image_path": str(image_path.relative_to(output_dir)),
                    "page_number": page_num,
                    "next_page_number": next_page_number,
                    "xref": xref,
                    "width": width,
                    "height": height,
                    "bboxes": _bbox_list(rects),
                    "page_text": page_text,
                    "next_page_text": next_page_text,
                    "page_text_path": str(page_text_file.relative_to(output_dir)),
                    "next_page_text_path": str(next_page_text_file.relative_to(output_dir))
                    if next_page_text_file
                    else None,
                }
                metadata_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_images += 1

    doc.close()
    print(f"Saved {total_images} images to {images_dir}")
    print(f"Metadata written to {metadata_path}")


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    extract_assets(pdf_path, output_dir)


if __name__ == "__main__":
    main()
