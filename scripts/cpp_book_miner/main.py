from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import re

import yaml

from .models import ChapterIndex, ChapterRef, ChapterSummary, Candidate, ChapterCitations, from_jsonl
from .splitter import split_book_markdown
from .generator import load_config, run_summarization_for_chapter, run_candidates_for_chapter, run_citations_for_chapter
from .renderer import render_review_markdown, export_post, render_citations_markdown
from .generator import run_refine_for_candidates


def _repo_root() -> Path:
    return Path(os.getcwd())


def _resolve_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    base = _repo_root()
    paths = config["paths"]
    return {
        "input_book_path": base / paths["input_book_path"],
        "output_base": base / paths["output_base"],
        "chapters_dir": base / paths["chapters_dir"],
        "summaries_dir": base / paths["summaries_dir"],
        "candidates_dir": base / paths["candidates_dir"],
        "review_dir": base / paths["review_dir"],
        "refined_dir": base / paths["refined_dir"],
        "citations_dir": base / paths["citations_dir"],
    }

def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "part"

def _extract_by_subheading(text: str, subheading: str) -> str:
    lines = text.splitlines(keepends=True)
    heading_regex = re.compile(r"^(#+)\s+(.*)$")
    target = subheading.strip().lower()
    start_idx = None
    level = None
    for i, line in enumerate(lines):
        m = heading_regex.match(line.strip())
        if not m:
            continue
        lvl = len(m.group(1))
        title = m.group(2).strip().lower()
        if title == target:
            start_idx = i
            level = lvl
            break
    if start_idx is None:
        return text
    end_idx = len(lines)
    for j in range(start_idx + 1, len(lines)):
        m = heading_regex.match(lines[j].strip())
        if m:
            next_level = len(m.group(1))
            if next_level <= level:
                end_idx = j
                break
    return "".join(lines[start_idx:end_idx]).lstrip("\n")

def _slice_text_by_lines(text: str, start_line: int, end_line: int) -> str:
    if start_line <= 0 and end_line <= 0:
        return text
    lines = text.splitlines(keepends=True)
    s = max(1, start_line) if start_line > 0 else 1
    e = min(len(lines), end_line) if end_line > 0 else len(lines)
    if s > e:
        s, e = e, s
    return "".join(lines[s - 1 : e]).lstrip("\n")

def _build_suffix(args: "argparse.Namespace") -> str:
    parts: List[str] = []
    if getattr(args, "subheading", ""):
        parts.append(_slugify(args.subheading))
    if getattr(args, "start_line", 0) or getattr(args, "end_line", 0):
        parts.append(f"L{args.start_line or 1}-{args.end_line or 'end'}")
    return ("__" + "_".join(parts)) if parts else ""

def _chapter_text_with_slice(chapter: ChapterRef, args: "argparse.Namespace") -> str:
    with open(chapter.output_markdown_path, "r", encoding="utf-8") as f:
        text = f.read()
    if getattr(args, "subheading", ""):
        text = _extract_by_subheading(text, args.subheading)
    if getattr(args, "start_line", 0) or getattr(args, "end_line", 0):
        text = _slice_text_by_lines(text, getattr(args, "start_line", 0), getattr(args, "end_line", 0))
    return text


def cmd_split(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    paths = _resolve_paths(config)
    index_path = paths["chapters_dir"] / "index.json"
    idx = split_book_markdown(
        input_path=paths["input_book_path"],
        chapters_dir=paths["chapters_dir"],
        index_path=index_path,
    )
    print(f"Wrote {len(idx.chapters)} chapters to {paths['chapters_dir']}")


def _read_index(paths: Dict[str, Path]) -> ChapterIndex:
    index_path = paths["chapters_dir"] / "index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ChapterIndex.model_validate(data)


def cmd_citations(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    # Enable prompt debugging without hitting the network
    config["debug_llm"] = bool(getattr(args, "debug_llm", False))
    config["debug_llm_output"] = bool(getattr(args, "debug_llm_output", False))
    raw_override: str | None = None
    if getattr(args, "raw_file", ""):
        with open(args.raw_file, "r", encoding="utf-8") as f:
            raw_override = f.read()
        # Ensure we don't try to "debug prompt" when using override
        config["debug_llm"] = False
    elif getattr(args, "paste_raw", False):
        raw_override = sys.stdin.read()
        config["debug_llm"] = False
    paths = _resolve_paths(config)
    idx = _read_index(paths)
    # Restrict to a single chapter if requested or when slicing is used
    chapters: List[ChapterRef]
    if args.chapter:
        chapters = [c for c in idx.chapters if c.chapter_id == args.chapter]
    elif args.subheading or args.start_line or args.end_line or args.first:
        chapters = idx.chapters[:1]
    else:
        chapters = idx.chapters
    try:
        for chapter in chapters:
            # Get original chapter text to compute line offset
            with open(chapter.output_markdown_path, "r", encoding="utf-8") as f:
                original_text = f.read()
            text = _chapter_text_with_slice(chapter, args)
            suffix = _build_suffix(args)
            out_id = chapter.chapter_id + suffix
            # Calculate line offset if slicing was used
            line_offset = 0
            if getattr(args, "start_line", 0):
                # Count lines before the slice in original text
                original_lines = original_text.splitlines(keepends=False)
                line_offset = max(0, getattr(args, "start_line", 1) - 1)
            citations = run_citations_for_chapter(
                chapter_id=out_id,
                chapter_title=chapter.title,
                chapter_text=text,
                out_dir=paths["citations_dir"],
                config=config,
                raw_override=raw_override,
                line_offset=line_offset,
            )
            print(f"Extracted {len(citations.citations)} citations from {out_id}")
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return


def cmd_summarize(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    # Enable prompt debugging without hitting the network
    config["debug_llm"] = bool(getattr(args, "debug_llm", False))
    config["debug_llm_output"] = bool(getattr(args, "debug_llm_output", False))
    raw_override: str | None = None
    if getattr(args, "raw_file", ""):
        with open(args.raw_file, "r", encoding="utf-8") as f:
            raw_override = f.read()
        # Ensure we don't try to "debug prompt" when using override
        config["debug_llm"] = False
    elif getattr(args, "paste_raw", False):
        raw_override = sys.stdin.read()
        config["debug_llm"] = False
    paths = _resolve_paths(config)
    idx = _read_index(paths)
    # Restrict to a single chapter if requested or when slicing is used
    chapters: List[ChapterRef]
    if args.chapter:
        chapters = [c for c in idx.chapters if c.chapter_id == args.chapter]
    elif args.subheading or args.start_line or args.end_line or args.first:
        chapters = idx.chapters[:1]
    else:
        chapters = idx.chapters
    try:
        for chapter in chapters:
            text = _chapter_text_with_slice(chapter, args)
            suffix = _build_suffix(args)
            out_id = chapter.chapter_id + suffix
            summary = run_summarization_for_chapter(
                chapter_id=out_id,
                chapter_title=chapter.title,
                chapter_text=text,
                out_dir=paths["summaries_dir"],
                config=config,
                raw_override=raw_override,
            )
            print(f"Summarized {out_id}: {len(summary.topics)} topics")
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return


def _load_summary(paths: Dict[str, Path], chapter_id: str) -> ChapterSummary:
    with open(paths["summaries_dir"] / f"{chapter_id}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return ChapterSummary.model_validate(data)


def _load_citations(paths: Dict[str, Path], chapter_id: str) -> ChapterCitations:
    with open(paths["citations_dir"] / f"{chapter_id}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return ChapterCitations.model_validate(data)


def cmd_generate(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    config["debug_llm"] = bool(getattr(args, "debug_llm", False))
    config["debug_llm_output"] = bool(getattr(args, "debug_llm_output", False))
    raw_override: str | None = None
    if getattr(args, "raw_file", ""):
        with open(args.raw_file, "r", encoding="utf-8") as f:
            raw_override = f.read()
        config["debug_llm"] = False
    elif getattr(args, "paste_raw", False):
        raw_override = sys.stdin.read()
        config["debug_llm"] = False
    paths = _resolve_paths(config)
    idx = _read_index(paths)
    # Restrict to a single chapter if requested or when slicing is used
    chapters: List[ChapterRef]
    if args.chapter:
        chapters = [c for c in idx.chapters if c.chapter_id == args.chapter]
    elif args.subheading or args.start_line or args.end_line or args.first:
        chapters = idx.chapters[:1]
    else:
        chapters = idx.chapters
    try:
        for chapter in chapters:
            text = _chapter_text_with_slice(chapter, args)
            suffix = _build_suffix(args)
            # If slicing, compute topics on-the-fly for this slice
            if suffix:
                summary = run_summarization_for_chapter(
                    chapter_id=chapter.chapter_id + suffix,
                    chapter_title=chapter.title,
                    chapter_text=text,
                    out_dir=paths["summaries_dir"],
                    config=config,
                )
                chapter_id_for_candidates = chapter.chapter_id + suffix
                topics = summary.topics
            else:
                summary = _load_summary(paths, chapter.chapter_id)
                chapter_id_for_candidates = chapter.chapter_id
                topics = summary.topics
            candidates = run_candidates_for_chapter(
                chapter_id=chapter_id_for_candidates,
                chapter_title=chapter.title,
                chapter_text=text,
                topics=topics,
                out_dir=paths["candidates_dir"],
                config=config,
                raw_override=raw_override,
            )
            print(f"Generated {len(candidates)} candidates for {chapter_id_for_candidates}")
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return


def _read_candidates(paths: Dict[str, Path], chapter_id: str) -> List[Candidate]:
    path = paths["candidates_dir"] / f"{chapter_id}.jsonl"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return from_jsonl(f.read(), Candidate)


def cmd_render_review(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    paths = _resolve_paths(config)
    idx = _read_index(paths)
    # Restrict which chapters to render if requested
    chapters: List[ChapterRef]
    if getattr(args, "chapter", ""):
        chapters = [c for c in idx.chapters if c.chapter_id == args.chapter]
    elif getattr(args, "first", False):
        chapters = idx.chapters[:1]
    else:
        chapters = idx.chapters
    for chapter in chapters:
        candidates = _read_candidates(paths, chapter.chapter_id)
        out_path = paths["review_dir"] / f"{chapter.chapter_id}.md"
        templates_dir = Path(__file__).parent / "templates"
        render_review_markdown(
            chapter=chapter, candidates=candidates, templates_dir=templates_dir, out_path=out_path
        )
        print(f"Wrote review bundle: {out_path}")


def cmd_render_citations(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    paths = _resolve_paths(config)
    idx = _read_index(paths)
    # Restrict which chapters to render if requested
    chapters: List[ChapterRef]
    if getattr(args, "chapter", ""):
        chapters = [c for c in idx.chapters if c.chapter_id == args.chapter]
    elif getattr(args, "first", False):
        chapters = idx.chapters[:1]
    else:
        chapters = idx.chapters
    for chapter in chapters:
        try:
            citations = _load_citations(paths, chapter.chapter_id)
        except FileNotFoundError:
            print(f"Warning: No citations found for {chapter.chapter_id}. Run 'citations' command first.", file=sys.stderr)
            continue
        out_path = paths["citations_dir"] / f"{chapter.chapter_id}.md"
        render_citations_markdown(
            chapter=chapter, citations=citations, out_path=out_path
        )
        print(f"Wrote citations markdown: {out_path}")


def cmd_export(args: argparse.Namespace) -> None:
    config = load_config(Path(args.config))
    paths = _resolve_paths(config)
    idx = _read_index(paths)
    base_dir = _repo_root() / "insights" / "programming"
    # Restrict which chapters to export if requested
    chapters: List[ChapterRef]
    if getattr(args, "chapter", ""):
        chapters = [c for c in idx.chapters if c.chapter_id == args.chapter]
    elif getattr(args, "first", False):
        chapters = idx.chapters[:1]
    else:
        chapters = idx.chapters
    for chapter in chapters:
        candidates = _read_candidates(paths, chapter.chapter_id)
        for c in candidates:
            export_post(candidate=c, base_dir=base_dir, chapter_title=chapter.title)
            print(f"Exported {c.id} -> insights/programming/{c.id}/")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cpp_book_miner", description="Mine viral C++ CE examples")
    p.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config.yaml",
    )
    p.add_argument(
        "--debug-llm",
        action="store_true",
        help="Print LLM prompts (system+user) and skip API calls.",
    )
    p.add_argument(
        "--debug-llm-output",
        action="store_true",
        help="Print and persist raw LLM outputs to *.raw.txt files.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("split", help="Split book into chapters").set_defaults(func=cmd_split)
    sp_cit = sub.add_parser("citations", help="Extract interesting quotations and citations per chapter")
    sp_cit.add_argument("--chapter", type=str, default="", help="Restrict to a chapter id")
    sp_cit.add_argument("--first", action="store_true", help="Use first chapter only")
    sp_cit.add_argument("--subheading", type=str, default="", help="Subsection heading to slice inside chapter")
    sp_cit.add_argument("--start-line", type=int, default=0, help="Start line within chapter (1-based)")
    sp_cit.add_argument("--end-line", type=int, default=0, help="End line within chapter (inclusive)")
    sp_cit.add_argument("--debug-llm", action="store_true", help="Print LLM prompts and skip API calls")
    sp_cit.add_argument("--debug-llm-output", action="store_true", help="Print and persist raw LLM outputs")
    sp_cit.add_argument("--paste-raw", action="store_true", help="Read raw LLM output from stdin instead of calling LLM")
    sp_cit.add_argument("--raw-file", type=str, default="", help="Path to file containing raw LLM output to parse")
    sp_cit.set_defaults(func=cmd_citations)
    sp_sum = sub.add_parser("summarize", help="Summarize topics per chapter")
    sp_sum.add_argument("--chapter", type=str, default="", help="Restrict to a chapter id")
    sp_sum.add_argument("--first", action="store_true", help="Use first chapter only")
    sp_sum.add_argument("--subheading", type=str, default="", help="Subsection heading to slice inside chapter")
    sp_sum.add_argument("--start-line", type=int, default=0, help="Start line within chapter (1-based)")
    sp_sum.add_argument("--end-line", type=int, default=0, help="End line within chapter (inclusive)")
    sp_sum.add_argument("--debug-llm", action="store_true", help="Print LLM prompts and skip API calls")
    sp_sum.add_argument("--debug-llm-output", action="store_true", help="Print and persist raw LLM outputs")
    sp_sum.add_argument("--paste-raw", action="store_true", help="Read raw LLM output from stdin instead of calling LLM")
    sp_sum.add_argument("--raw-file", type=str, default="", help="Path to file containing raw LLM output to parse")
    sp_sum.set_defaults(func=cmd_summarize)
    sp_gen = sub.add_parser("generate", help="Generate candidates")
    sp_gen.add_argument("--chapter", type=str, default="", help="Restrict to a chapter id")
    sp_gen.add_argument("--first", action="store_true", help="Use first chapter only")
    sp_gen.add_argument("--subheading", type=str, default="", help="Subsection heading to slice inside chapter")
    sp_gen.add_argument("--start-line", type=int, default=0, help="Start line within chapter (1-based)")
    sp_gen.add_argument("--end-line", type=int, default=0, help="End line within chapter (inclusive)")
    sp_gen.add_argument("--debug-llm", action="store_true", help="Print LLM prompts and skip API calls")
    sp_gen.add_argument("--debug-llm-output", action="store_true", help="Print and persist raw LLM outputs")
    sp_gen.add_argument("--paste-raw", action="store_true", help="Read raw LLM output from stdin instead of calling LLM")
    sp_gen.add_argument("--raw-file", type=str, default="", help="Path to file containing raw LLM output to parse")
    sp_gen.set_defaults(func=cmd_generate)
    sp_render = sub.add_parser("render", help="Render review bundles")
    sp_render.add_argument("--chapter", type=str, default="", help="Restrict to a chapter id")
    sp_render.add_argument("--first", action="store_true", help="Render first chapter only")
    # Accept debug flags here so users can pass them after the subcommand; they are no-ops for render
    sp_render.add_argument("--debug-llm", action="store_true", help=argparse.SUPPRESS)
    sp_render.add_argument("--debug-llm-output", action="store_true", help=argparse.SUPPRESS)
    sp_render.set_defaults(func=cmd_render_review)
    sp_render_cit = sub.add_parser("render-citations", help="Render citations markdown")
    sp_render_cit.add_argument("--chapter", type=str, default="", help="Restrict to a chapter id")
    sp_render_cit.add_argument("--first", action="store_true", help="Render first chapter only")
    # Accept debug flags here so users can pass them after the subcommand; they are no-ops for render
    sp_render_cit.add_argument("--debug-llm", action="store_true", help=argparse.SUPPRESS)
    sp_render_cit.add_argument("--debug-llm-output", action="store_true", help=argparse.SUPPRESS)
    sp_render_cit.set_defaults(func=cmd_render_citations)
    sp_export = sub.add_parser("export", help="Export accepted posts")
    sp_export.add_argument("--chapter", type=str, default="", help="Restrict to a chapter id")
    sp_export.add_argument("--first", action="store_true", help="Export first chapter only")
    # Accept debug flags here so users can pass them after the subcommand; they are no-ops for export
    sp_export.add_argument("--debug-llm", action="store_true", help=argparse.SUPPRESS)
    sp_export.add_argument("--debug-llm-output", action="store_true", help=argparse.SUPPRESS)
    sp_export.set_defaults(func=cmd_export)
    # refine
    def cmd_refine(args: argparse.Namespace) -> None:
        config = load_config(Path(args.config))
        config["debug_llm"] = bool(getattr(args, "debug_llm", False))
        config["debug_llm_output"] = bool(getattr(args, "debug_llm_output", False))
        paths = _resolve_paths(config)
        idx = _read_index(paths)
        # If --chapter provided, restrict, else all
        if args.chapter:
            targets = [c for c in idx.chapters if c.chapter_id == args.chapter]
        elif getattr(args, "first", False):
            targets = idx.chapters[:1]
        else:
            targets = idx.chapters
        try:
            for chapter in targets:
                cands = _read_candidates(paths, chapter.chapter_id)
                if args.ids:
                    ids = set(x.strip() for x in args.ids.split(",") if x.strip())
                    cands = [c for c in cands if c.id in ids]
                out_dir = paths["refined_dir"] / chapter.chapter_id
                refined = run_refine_for_candidates(cands, out_dir, config)
                print(f"Refined {len(refined)} for {chapter.chapter_id} -> {out_dir}")
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return

    pr = sub.add_parser("refine", help="Refine shortlisted candidates")
    pr.add_argument("--chapter", type=str, default="", help="Chapter id to refine (optional)")
    pr.add_argument("--ids", type=str, default="", help="Comma-separated candidate ids (optional)")
    pr.add_argument("--debug-llm", action="store_true", help="Print LLM prompts and skip API calls")
    pr.add_argument("--debug-llm-output", action="store_true", help="Print and persist raw LLM outputs")
    pr.add_argument("--first", action="store_true", help="Refine first chapter only")
    pr.set_defaults(func=cmd_refine)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


