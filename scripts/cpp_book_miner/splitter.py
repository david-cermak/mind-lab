from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .models import ChapterRef, ChapterIndex


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        text = "chapter"
    return text


def _detect_heading_level(lines: List[str]) -> str:
    has_h2 = any(line.startswith("## ") for line in lines)
    has_h1 = any(line.startswith("# ") for line in lines)
    if has_h2:
        return "##"
    if has_h1:
        return "#"
    # fallback: treat any line starting with 'Chapter' as a heading
    return "CHAPTER"


def _gather_chapter_spans(lines: List[str]) -> List[Tuple[int, int, str]]:
    level = _detect_heading_level(lines)
    headings: List[Tuple[int, str]] = []
    if level in ("#", "##"):
        prefix = f"{level} "
        for i, line in enumerate(lines):
            if line.startswith(prefix):
                title = line[len(prefix) :].strip()
                headings.append((i, title))
    else:
        for i, line in enumerate(lines):
            if re.match(r"^\s*chapter\b", line.strip().lower()):
                title = line.strip()
                headings.append((i, title))
    if not headings:
        # single chapter (whole file)
        return [(0, len(lines), "Full Book")]
    spans: List[Tuple[int, int, str]] = []
    for idx, (start, title) in enumerate(headings):
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        spans.append((start, end, title))
    return spans


def split_book_markdown(
    input_path: Path, chapters_dir: Path, index_path: Path
) -> ChapterIndex:
    chapters_dir.mkdir(parents=True, exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    spans = _gather_chapter_spans(lines)
    index = ChapterIndex()
    for start, end, title in spans:
        slug = _slugify(title)
        chapter_id = slug
        chapter_md_path = chapters_dir / f"{chapter_id}.md"
        # Write chapter markdown
        content = "".join(lines[start:end]).lstrip("\n")
        with open(chapter_md_path, "w", encoding="utf-8") as f:
            f.write(content)
        # Add to index
        index.chapters.append(
            ChapterRef(
                chapter_id=chapter_id,
                title=title,
                start_line=start + 1,
                end_line=end,
                source_path=str(input_path),
                output_markdown_path=str(chapter_md_path),
            )
        )
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index.model_dump(), f, indent=2, ensure_ascii=False)
    return index


