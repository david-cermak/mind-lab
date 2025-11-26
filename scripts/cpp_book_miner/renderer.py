from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .models import ChapterRef, Candidate, ReviewBundle, ChapterCitations, ChapterFullSummary, from_jsonl


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


def render_citations_markdown(
    chapter: ChapterRef,
    citations: ChapterCitations,
    out_path: Path,
    context_lines: int = 5,
) -> None:
    """
    Render citations to a markdown file with citations, line numbers, and context.
    
    Args:
        chapter: Chapter reference containing the path to the chapter file
        citations: ChapterCitations object with extracted citations
        out_path: Path to write the output markdown file
        context_lines: Number of lines before and after the citation to include (default: 5)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read the chapter file to extract actual text
    chapter_lines = []
    try:
        with open(chapter.output_markdown_path, "r", encoding="utf-8") as chapter_file:
            chapter_lines = chapter_file.readlines()
    except FileNotFoundError:
        # Fallback: try relative path from citations dir
        chapter_path = Path(chapter.output_markdown_path)
        if not chapter_path.exists():
            # Try to find it relative to citations dir
            citations_dir = out_path.parent
            # Assuming chapters_dir is sibling to citations_dir
            chapters_dir = citations_dir.parent / "chapters"
            chapter_path = chapters_dir / f"{chapter.chapter_id}.md"
        if chapter_path.exists():
            with open(chapter_path, "r", encoding="utf-8") as chapter_file:
                chapter_lines = chapter_file.readlines()
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Citations from Chapter: {chapter.title}\n\n")
        f.write(f"**Chapter ID:** {chapter.chapter_id}\n\n")
        f.write(f"**Total Citations:** {len(citations.citations)}\n\n")
        f.write("---\n\n")
        
        if not citations.citations:
            f.write("*No citations found in this chapter.*\n")
            return
        
        for i, citation in enumerate(citations.citations, 1):
            f.write(f"## Citation {i}\n\n")
            f.write(f"**Line {citation.line_number}**\n\n")
            f.write(f"> {citation.citation}\n\n")
            
            # Extract actual text from the book around the citation line
            if chapter_lines:
                line_idx = citation.line_number - 1  # Convert to 0-based index
                if 0 <= line_idx < len(chapter_lines):
                    start_idx = max(0, line_idx - context_lines)
                    end_idx = min(len(chapter_lines), line_idx + context_lines + 1)
                    context_text_lines = chapter_lines[start_idx:end_idx]
                    
                    # Find the citation line within the context
                    citation_line_in_context = line_idx - start_idx
                    
                    f.write(f"**Actual Text from Book:**\n\n")
                    f.write("```\n")
                    for j, line in enumerate(context_text_lines):
                        line_num = start_idx + j + 1  # 1-based line number
                        # Highlight the citation line
                        if j == citation_line_in_context:
                            f.write(f"{line_num:4d}|> {line.rstrip()}\n")
                        else:
                            f.write(f"{line_num:4d}|  {line.rstrip()}\n")
                    f.write("```\n\n")
            
            f.write(f"**Context Summary:**\n\n{citation.context}\n\n")
            f.write("---\n\n")


def render_summary_markdown(
    chapter: ChapterRef,
    summary: ChapterFullSummary,
    out_path: Path,
) -> None:
    """
    Render a human-readable markdown summary for a chapter.

    This does not use Jinja; it just formats the JSON into markdown:
    - H1: chapter title
    - H2: Summary title (from the LLM)
    - Learning objective section
    - Summary paragraphs
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        # Chapter heading
        f.write(f"# {chapter.title}\n\n")
        # Summary title
        display_title = summary.title.strip() or chapter.title
        f.write(f"## {display_title}\n\n")
        # Learning objective
        if summary.learning_objective.strip():
            f.write("### Learning objective\n\n")
            f.write(summary.learning_objective.strip() + "\n\n")
        # Summary body
        if summary.summary.strip():
            f.write("### Summary\n\n")
            f.write(summary.summary.strip() + "\n")


