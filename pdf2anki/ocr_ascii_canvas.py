"""
Render OCR tokens onto a coarse ASCII grid for LLM-friendly prompts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project OCR tokens onto a coarse ASCII grid.")
    parser.add_argument("ocr_json", type=Path, help="Path to OCR JSON file produced by extractors.ocr")
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional image path. If omitted, canvas size is inferred from token bounding boxes.",
    )
    parser.add_argument("--cols", type=int, default=80, help="Number of columns in the ASCII grid (default: %(default)s)")
    parser.add_argument("--rows", type=int, default=40, help="Number of rows in the ASCII grid (default: %(default)s)")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=70.0,
        help="Drop tokens below this OCR confidence (default: %(default)s)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Optional limit on how many tokens to render (sorted by y,x)",
    )
    return parser.parse_args()


def load_tokens(ocr_path: Path, min_conf: float) -> List[dict]:
    data = json.loads(ocr_path.read_text(encoding="utf-8"))
    tokens = []
    for tok in data.get("tokens", []):
        text = (tok.get("text") or "").strip()
        conf = float(tok.get("confidence", 0))
        if not text or conf < min_conf:
            continue
        bbox = tok.get("bbox") or [0, 0, 0, 0]
        tokens.append({"text": text, "confidence": conf, "bbox": bbox})
    tokens.sort(key=lambda t: (t["bbox"][1], t["bbox"][0]))
    return tokens


def infer_dimensions(tokens: List[dict], image_path: Path | None) -> Tuple[int, int]:
    if image_path and image_path.exists():
        with Image.open(image_path) as im:
            return im.size  # (width, height)
    max_x = max((t["bbox"][0] + t["bbox"][2]) for t in tokens) if tokens else 1
    max_y = max((t["bbox"][1] + t["bbox"][3]) for t in tokens) if tokens else 1
    return max_x, max_y


def place_symbol(
    grid: List[List[str]],
    row: int,
    col: int,
    symbol: str,
) -> None:
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if row < 0 or row >= rows:
        return
    if cols <= 0:
        return
    start = max(0, min(col, cols - len(symbol)))
    for offset, ch in enumerate(symbol):
        c = start + offset
        if c >= cols:
            break
        grid[row][c] = ch


def _symbol_with_span(index: int, bbox_width: float, canvas_width: int, cols: int) -> str:
    if canvas_width <= 0:
        canvas_width = 1
    desired = int(max(5, round((bbox_width / canvas_width) * cols)))
    core = f"{index}"
    inner = max(len(core), desired - 2)
    pad_left = (inner - len(core)) // 2
    pad_right = inner - len(core) - pad_left
    return "[" + ("." * pad_left) + core + ("." * pad_right) + "]"


def render_ascii_grid(tokens: List[dict], width: int, height: int, rows: int, cols: int, max_tokens: int | None) -> Tuple[str, List[str]]:
    grid = [[" " for _ in range(cols)] for _ in range(rows)]
    legend: List[str] = []
    usable_tokens = tokens[: max_tokens or len(tokens)]

    for idx, token in enumerate(usable_tokens, start=1):
        left, top, bbox_w, bbox_h = token["bbox"]
        col = min(cols - 1, max(0, int((left + bbox_w / 2) / width * cols)))
        row = min(rows - 1, max(0, int((top + bbox_h / 2) / height * rows)))
        symbol = _symbol_with_span(idx, bbox_w, width, cols)
        place_symbol(grid, row, col, symbol)
        legend.append(f"[{idx}] {token['text']} (conf={token['confidence']:.1f})")

    ascii_lines = ["".join(line).rstrip() for line in grid]
    ascii_block = "\n".join(ascii_lines)
    return ascii_block, legend


def main() -> None:
    args = parse_args()
    tokens = load_tokens(args.ocr_json, args.min_confidence)
    if not tokens:
        print("No tokens to render (after filtering).")
        return

    width, height = infer_dimensions(tokens, args.image)
    ascii_block, legend = render_ascii_grid(tokens, width, height, args.rows, args.cols, args.max_tokens)

    print("ASCII canvas:")
    print("```")
    print(ascii_block)
    print("```")
    print()
    print("Legend:")
    for entry in legend:
        print(f"  {entry}")


if __name__ == "__main__":
    main()

