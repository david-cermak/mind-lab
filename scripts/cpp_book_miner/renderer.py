from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .models import ChapterRef, Candidate, ReviewBundle, from_jsonl


def _env(templates_dir: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(enabled_extensions=("j2",)),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_review_markdown(
    chapter: ChapterRef,
    candidates: List[Candidate],
    templates_dir: Path,
    out_path: Path,
) -> None:
    env = _env(templates_dir)
    template = env.get_template("review.md.j2")
    bundle = ReviewBundle(chapter=chapter, candidates=candidates)
    out = template.render(bundle=bundle)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)


def export_post(candidate: Candidate, base_dir: Path, chapter_title: str) -> None:
    slug = candidate.id
    dir_path = base_dir / slug
    dir_path.mkdir(parents=True, exist_ok=True)
    # Write code
    code_path = dir_path / "test.cpp"
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(candidate.code.rstrip() + "\n")
    # Render post
    templates_dir = Path(__file__).parent / "templates"
    env = _env(templates_dir)
    template = env.get_template("post.md.j2")
    post_md = template.render(
        title=candidate.title,
        hook=candidate.hook,
        tags=candidate.tags,
        std=candidate.std,
        chapter_title=chapter_title,
        notes=candidate.notes,
        tuning=candidate.tuning.model_dump() if candidate.tuning else None,
    )
    post_path = dir_path / "post.md"
    with open(post_path, "w", encoding="utf-8") as f:
        f.write(post_md)


