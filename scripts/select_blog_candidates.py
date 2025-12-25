#!/usr/bin/env python3
"""
Select best blog post candidates from news/search results using an OpenAI-compatible API.

The script:
  1. Loads previous posts from a posts definition file (e.g. scripts/output/posts.txt).
  2. Loads news/search results from a markdown file (e.g. scripts/output/news2.md).
  3. Splits the news into chunks of items.
  4. For each chunk, calls the LLM with:
       - An extraction of previous posts
       - Instructions what to choose (up to N best choices per chunk)
       - The current news chunk
  5. Merges and prints all selected candidates.

Configuration is done similarly to scripts/test_openai_api.py:
  Priority: CLI args > Environment variables > .env file
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


def load_env_file(project_root: Path | None = None) -> None:
    """Load .env file if python-dotenv is available (same logic as test_openai_api.py)."""
    if load_dotenv is None:
        return

    if project_root is None:
        current = Path(__file__).resolve().parent
        for parent in [current, current.parent]:
            env_file = parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                return
        load_dotenv()
    else:
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)


def get_config(
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[str, str, str]:
    """
    Get configuration with priority: CLI args > env vars > .env file.

    Returns:
        Tuple of (base_url, model, api_key)
    """
    load_env_file()

    final_base_url = (
        base_url
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("BASE_URL")
    )

    final_model = (
        model
        or os.getenv("OPENAI_MODEL")
        or os.getenv("MODEL")
    )

    final_api_key = (
        api_key
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("API_KEY")
    )

    if not final_base_url:
        raise ValueError(
            "base-url is required. Provide via --base-url, OPENAI_BASE_URL env var, or .env file"
        )
    if not final_model:
        raise ValueError(
            "model is required. Provide via --model, OPENAI_MODEL env var, or .env file"
        )
    if not final_api_key:
        raise ValueError(
            "api-key is required. Provide via --api-key, OPENAI_API_KEY env var, or .env file"
        )

    return final_base_url, final_model, final_api_key


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    # Load .env early so defaults can depend on it (python-dotenv is optional).
    load_env_file()
    parser = argparse.ArgumentParser(
        description="Select best blog post candidates from news/search results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with env vars / .env
  export OPENAI_BASE_URL=https://api.openai.com/v1
  export OPENAI_MODEL=gpt-4
  export OPENAI_API_KEY=sk-...
  python scripts/select_blog_candidates.py

  # Explicit files and provider
  python scripts/select_blog_candidates.py \\
      --news-file scripts/output/news2.md \\
      --posts-file scripts/output/posts.txt \\
      --base-url https://api.openai.com/v1 \\
      --model gpt-4 \\
      --api-key sk-...
        """.strip(),
    )

    parser.add_argument(
        "--base-url",
        dest="base_url",
        metavar="URL",
        help="API base URL (e.g., https://api.openai.com/v1). "
        "Can also use OPENAI_BASE_URL env var or .env file.",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        help="Model name (e.g., gpt-4, gpt-3.5-turbo). "
        "Can also use OPENAI_MODEL env var or .env file.",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        metavar="KEY",
        help="API key. Can also use OPENAI_API_KEY env var or .env file.",
    )
    parser.add_argument(
        "--news-file",
        default="scripts/output/news2.md",
        help="Path to markdown file with news/search results "
        "(default: scripts/output/news2.md).",
    )
    parser.add_argument(
        "--posts-file",
        default="scripts/output/posts.txt",
        help="Path to file with previous posts definitions "
        "(default: scripts/output/posts.txt).",
    )
    parser.add_argument(
        "--items-per-chunk",
        type=int,
        default=20,
        help="Approximate number of news items per LLM chunk (default: 20).",
    )
    parser.add_argument(
        "--max-candidates-per-chunk",
        type=int,
        default=3,
        help="Maximum number of candidates the LLM should return per chunk (default: 3).",
    )
    parser.add_argument(
        "--max-sources-per-candidate",
        type=int,
        default=get_max_sources_per_candidate(default=3),
        help=(
            "Maximum number of source items a single blog candidate can combine "
            "(default: 3; also configurable via BLOG_CANDIDATE_MAX_SOURCES env var)."
        ),
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Optional limit on number of news items to consider (0 = all).",
    )
    parser.add_argument(
        "--raw-json",
        action="store_true",
        help="Print raw merged JSON output instead of a human-readable summary.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help=(
            "Path to write merged candidates as JSON. "
            "Default: selected.json next to the news file."
        ),
    )

    return parser.parse_args(argv)


def load_text(path: str | Path) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8")


def get_max_sources_per_candidate(default: int = 3) -> int:
    """
    Get maximum number of source items a single candidate can reference.

    Controlled via environment variable:
        BLOG_CANDIDATE_MAX_SOURCES
    """
    load_env_file()
    raw = os.getenv("BLOG_CANDIDATE_MAX_SOURCES") or os.getenv("MAX_SOURCES_PER_CANDIDATE") or ""
    if not raw:
        return default
    try:
        value = int(raw)
        if value <= 0:
            return default
        return value
    except ValueError:
        return default


def is_official_espressif_documentation(url: str, title: str = "") -> bool:
    """
    Heuristic filter for official Espressif documentation.

    We intentionally do NOT block all Espressif content (e.g. blog posts, GitHub),
    only the official documentation site(s).
    """
    u = (url or "").strip().lower()
    t = (title or "").strip().lower()
    if not u and not t:
        return False
    # Primary official docs host (ESP-IDF programming guide, API reference, etc.)
    if "docs.espressif.com" in u:
        return True
    # Very common docs phrasing in titles (extra safety if URL missing)
    if "esp-idf programming guide" in t and "espressif" in t:
        return True
    return False


def extract_previous_posts_summary(posts_text: str, max_chars: int = 6000) -> str:
    """
    Prepare a compact summary of previous posts to pass into the prompt.

    Currently this is a simple truncation of the original file content, which
    already contains slug, title, description, and body references.
    """
    text = posts_text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars] + "\n...\n[truncated]"


def _parse_bold_field_line(line: str) -> tuple[str, str] | None:
    """
    Parse lines like:
      - **URL**: https://...
      - **Summary**: ...
      - **Date**: 2025-01-01T...
    """
    if not line.startswith("- **") or "**:" not in line:
        return None
    # Split at the first '**:' occurrence
    prefix, value = line.split("**:", 1)
    # prefix looks like "- **URL"
    key = prefix.replace("- **", "").strip()
    value = value.strip()
    if not key:
        return None
    return key, value


def parse_news_items(markdown: str) -> List[Dict[str, Any]]:
    """
    Parse `news2.md`/`news9.md`-style markdown into a list of item dicts.

    Each item starts with a line like '### 1. Title' and includes following lines
    until the next '### ' heading or end of file.
    """
    lines = markdown.splitlines()
    items: List[Dict[str, Any]] = []
    current_lines: List[str] = []

    for line in lines:
        if line.startswith("### "):
            # Start of a new item
            if current_lines:
                items.append(_parse_single_item_block("\n".join(current_lines).strip()))
                current_lines = []
        if line.startswith("### ") or current_lines:
            current_lines.append(line)

    if current_lines:
        items.append(_parse_single_item_block("\n".join(current_lines).strip()))

    # Filter out possible leading metadata like "## Search results..."
    items = [item for item in items if isinstance(item, dict) and str(item.get("raw", "")).startswith("### ")]
    return items


def _parse_single_item_block(block: str) -> Dict[str, Any]:
    lines = block.splitlines()
    header = lines[0].strip() if lines else ""
    # Expect header like: "### 123. Some title"
    item_id: Optional[int] = None
    title = header
    if header.startswith("### "):
        after = header[4:].strip()
        if ". " in after:
            maybe_num, rest = after.split(". ", 1)
            try:
                item_id = int(maybe_num.strip())
                title = rest.strip()
            except ValueError:
                title = after
        else:
            title = after

    url: str = ""
    summary: str = ""
    date_str: str = ""
    for line in lines[1:]:
        parsed = _parse_bold_field_line(line.strip())
        if not parsed:
            continue
        key, value = parsed
        k = key.lower()
        if k == "url":
            url = value
        elif k == "summary":
            summary = value
        elif k == "date":
            date_str = value

    return {
        "item_id": item_id,
        "title": title,
        "url": url,
        "summary": summary,
        "date": date_str,
        "raw": block,
    }


def _render_item_for_prompt(item: Dict[str, Any]) -> str:
    item_id = item.get("item_id")
    title = item.get("title") or ""
    url = item.get("url") or ""
    summary = item.get("summary") or ""
    date_str = item.get("date") or ""
    parts: List[str] = []
    parts.append(f"### {item_id}. {title}" if item_id is not None else f"### {title}")
    if url:
        parts.append(f"- **URL**: {url}")
    if summary:
        parts.append(f"- **Summary**: {summary}")
    if date_str:
        parts.append(f"- **Date**: {date_str}")
    return "\n".join(parts).strip()


def chunk_items(items: List[Dict[str, Any]], items_per_chunk: int, max_items: int = 0) -> List[Tuple[int, str]]:
    """
    Group items into chunks of approx. `items_per_chunk`.

    Returns list of (chunk_index, chunk_text). Each item within a chunk is
    prefixed with a stable ITEM ID so the LLM can refer to it.
    """
    if max_items > 0:
        items = items[:max_items]

    chunks: List[Tuple[int, str]] = []
    for chunk_idx, start in enumerate(range(0, len(items), items_per_chunk), start=1):
        end = start + items_per_chunk
        sub_items = items[start:end]
        labelled: List[str] = []
        for item in sub_items:
            item_id = item.get("item_id")
            # Fallback to list position if item_id missing
            if item_id is None:
                item_id = start + len(labelled) + 1
            labelled.append(f"ITEM {item_id}:\n{_render_item_for_prompt(item)}")
        chunk_text = "\n\n".join(labelled)
        chunks.append((chunk_idx, chunk_text))
    return chunks


def build_messages(
    previous_posts_summary: str,
    chunk_text: str,
    max_candidates_per_chunk: int,
    max_sources_per_candidate: int,
    today: str,
) -> list[dict[str, Any]]:
    """
    Build chat messages for a single LLM call.

    The prompt includes:
      - previous posts summary
      - clear instructions
      - the current news chunk
    """
    system_content = (
        "You are helping an embedded / firmware developer decide what to write about next.\n"
        f"Today is {today}.\n"
        "You will receive:\n"
        "  1) A description of previous blog posts (topics, titles, and descriptions).\n"
        "  2) A chunk of news/search results, each labelled as ITEM N.\n\n"
        "Your job is to pick up to "
        f"{max_candidates_per_chunk} promising blog-post candidates based on this chunk.\n"
        "A candidate may be based on a single ITEM, or it may combine multiple related ITEMs into\n"
        "one stronger post idea.\n\n"
        "Guidelines:\n"
        "- Only choose items that are recent (roughly within the last few months relative to today).\n"
        "  If an ITEM includes a Date and it looks old, do not pick it.\n"
        "- Prefer items that are connected to previous topics (embedded, C/C++, ESP32, fuzzing,\n"
        "  networking, secure channels, cryptography, TTCN-3, MQTT, Kafka, console/tunnelling, etc.),\n"
        "  but avoid repeating essentially the same post.\n"
        "- Reward novelty, concrete technical depth, and opportunities for experiments, PoCs, or\n"
        "  conference-talk-style writeups.\n"
        "- Ignore obviously off-topic or generic items.\n\n"
        "Hard exclusions:\n"
        "- Do NOT select official Espressif documentation pages (especially anything on docs.espressif.com).\n\n"
        "Output strictly valid JSON with the following structure:\n"
        "{\n"
        '  "candidates": [\n'
        "    {\n"
        f'      "item_ids": [<integer ITEM number>, ...],  // 1 to {max_sources_per_candidate} entries\n'
        '      "title": "short working blog post title",\n'
        '      "source_title": ["title or main line from each chosen news item", ...],\n'
        '      "url": ["url for each chosen news item", ...],\n'
        '      "link": ["same as url (for convenience)", ...],\n'
        '      "summary": ["summary/snippet for each chosen news item", ...],\n'
        '      "reason": "why this is a good fit given previous posts",\n'
        '      "angle": "suggested angle or twist for the post"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        f"Rules:\n"
        f"- item_ids must reference ITEM numbers from this chunk only.\n"
        f"- item_ids length must be between 1 and {max_sources_per_candidate}.\n"
        f"- source_title/url/link/summary arrays must match item_ids length and order.\n"
        "Return an empty list if nothing is suitable.\n"
    )

    user_content = (
        "Previous blog posts (reference only):\n"
        "------------------------------------\n"
        f"{previous_posts_summary}\n\n"
        "News/search results chunk:\n"
        "--------------------------\n"
        f"{chunk_text}\n\n"
        f"Select up to {max_candidates_per_chunk} best candidates from this chunk (recent items only)."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def call_llm_for_chunk(
    client: OpenAI,
    model: str,
    previous_posts_summary: str,
    chunk_text: str,
    max_candidates_per_chunk: int,
    max_sources_per_candidate: int,
    today: str,
) -> Dict[str, Any]:
    """Call the LLM once for a single chunk and parse JSON response."""
    messages = build_messages(
        previous_posts_summary,
        chunk_text,
        max_candidates_per_chunk,
        max_sources_per_candidate=max_sources_per_candidate,
        today=today,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    if not response.choices:
        return {"candidates": []}

    content = response.choices[0].message.content or ""

    # Attempt to parse JSON; if it fails, fall back to empty result.
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "candidates" in parsed and isinstance(parsed["candidates"], list):
            return parsed
    except json.JSONDecodeError:
        pass

    return {"candidates": []}


def merge_candidates(all_chunks_results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge candidates from all chunks, de-duplicating by (item_ids, title)."""
    merged: List[Dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()

    for result in all_chunks_results:
        for c in result.get("candidates", []):
            ids = normalize_item_ids(c)
            # Dedup independent of ordering of ids (but keep original order in output).
            key = (tuple(sorted(ids)), c.get("title"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(c)
    return merged


def normalize_item_ids(candidate: Dict[str, Any]) -> tuple[int, ...]:
    """
    Normalize candidate item references into a stable tuple[int, ...].

    Supports both the new schema ("item_ids": [...]) and legacy ("item_id": int).
    """
    raw_ids = candidate.get("item_ids")
    ids: List[int] = []
    if isinstance(raw_ids, list):
        for x in raw_ids:
            try:
                ids.append(int(x))
            except Exception:
                continue
    else:
        raw_id = candidate.get("item_id")
        try:
            if raw_id is not None:
                ids = [int(raw_id)]
        except Exception:
            ids = []
    # Dedup while keeping order
    seen: set[int] = set()
    out: List[int] = []
    for i in ids:
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return tuple(out)


def print_summary(candidates: List[Dict[str, Any]]) -> None:
    """Print a compact, human-readable summary of merged candidates."""
    if not candidates:
        print("No suitable candidates found.")
        return

    print("Merged blog post candidates:\n")
    for idx, c in enumerate(candidates, start=1):
        item_ids = list(normalize_item_ids(c))
        title = c.get("title") or "<no title>"
        source_title = c.get("source_title") or ""
        url = c.get("link") or c.get("url") or ""
        summary = c.get("summary") or ""
        reason = c.get("reason") or ""
        angle = c.get("angle") or ""

        print(f"{idx}. {title}")
        if item_ids:
            print(f"   - Source ITEM(s): {', '.join(str(x) for x in item_ids)}")
        # Handle either string or list values for backwards/forwards compatibility
        if isinstance(source_title, list):
            for i, st in enumerate(source_title, start=1):
                if st:
                    print(f"   - Source title {i}: {st}")
        elif source_title:
            print(f"   - Source title: {source_title}")
        if isinstance(url, list):
            for i, u in enumerate(url, start=1):
                if u:
                    print(f"   - Link {i}: {u}")
        elif url:
            print(f"   - Link: {url}")
        if isinstance(summary, list):
            for i, s in enumerate(summary, start=1):
                if s:
                    print(f"   - Summary {i}: {s}")
        elif summary:
            print(f"   - Summary: {summary}")
        if reason:
            print(f"   - Why: {reason}")
        if angle:
            print(f"   - Angle: {angle}")
        print()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if OpenAI is None:
        print(
            "Error: openai package is required. Install with: pip install openai",
            file=sys.stderr,
        )
        return 1

    try:
        base_url, model, api_key = get_config(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    news_text = load_text(args.news_file)
    posts_text = load_text(args.posts_file)
    previous_posts_summary = extract_previous_posts_summary(posts_text)

    items = parse_news_items(news_text)
    # Filter out official Espressif documentation from consideration.
    items = [
        it
        for it in items
        if not is_official_espressif_documentation(str(it.get("url") or ""), str(it.get("title") or ""))
    ]

    if not items:
        print("No eligible news items found (after filtering).", file=sys.stderr)
        return 1

    chunks = chunk_items(items, args.items_per_chunk, max_items=args.max_items)
    items_by_id: Dict[int, Dict[str, Any]] = {}
    for idx, item in enumerate(items, start=1):
        item_id = item.get("item_id") or idx
        try:
            items_by_id[int(item_id)] = item
        except Exception:
            continue

    today = dt.date.today().isoformat()

    print(f"Loaded {len(items)} news items, {len(chunks)} chunk(s).", file=sys.stderr)
    print(f"Using model: {model}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    all_results: List[Dict[str, Any]] = []

    for chunk_idx, chunk_text in chunks:
        print(f"Processing chunk {chunk_idx}/{len(chunks)}...", file=sys.stderr)
        try:
            result = call_llm_for_chunk(
                client=client,
                model=model,
                previous_posts_summary=previous_posts_summary,
                chunk_text=chunk_text,
                max_candidates_per_chunk=args.max_candidates_per_chunk,
                max_sources_per_candidate=args.max_sources_per_candidate,
                today=today,
            )
            all_results.append(result)
        except Exception as e:  # pragma: no cover - network/LLM errors
            print(f"Error calling API for chunk {chunk_idx}: {e}", file=sys.stderr)

    merged = merge_candidates(all_results)
    # Enrich merged candidates with original data (url/summary/title) from the news items.
    for c in merged:
        ids = list(normalize_item_ids(c))
        if not ids:
            continue
        # Cap to configured limit (safety in case the model ignores instructions)
        ids = ids[: max(1, int(args.max_sources_per_candidate))]
        # Store back normalized ids
        c["item_ids"] = ids
        # Materialize per-source arrays from original items
        src_titles: List[str] = []
        src_urls: List[str] = []
        src_summaries: List[str] = []
        for item_id in ids:
            src = items_by_id.get(int(item_id))
            if not src:
                src_titles.append("")
                src_urls.append("")
                src_summaries.append("")
                continue
            src_titles.append(str(src.get("title") or ""))
            src_urls.append(str(src.get("url") or ""))
            src_summaries.append(str(src.get("summary") or ""))
        # Populate arrays (override to keep schema consistent)
        c["source_title"] = src_titles
        c["url"] = src_urls
        c["link"] = list(src_urls)
        c["summary"] = src_summaries
        # Optional legacy compatibility: keep a primary item_id
        c["item_id"] = ids[0] if ids else None

    # Determine output file path (default: selected.json next to the news file)
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        news_path = Path(args.news_file)
        output_path = news_path.with_name("selected.json")

    try:
        output_path.write_text(
            json.dumps({"candidates": merged}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote {len(merged)} merged candidates to {output_path}", file=sys.stderr)
    except Exception as e:  # pragma: no cover - IO errors
        print(f"Error writing output file {output_path}: {e}", file=sys.stderr)

    if args.raw_json:
        print(json.dumps({"candidates": merged}, indent=2, ensure_ascii=False))
    else:
        print_summary(merged)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


