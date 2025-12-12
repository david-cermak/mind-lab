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
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

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


def parse_news_items(markdown: str) -> List[str]:
    """
    Parse `news2.md`-style markdown into a list of item blocks.

    Each item starts with a line like '### 1. Title' and includes following lines
    until the next '### ' heading or end of file.
    """
    lines = markdown.splitlines()
    items: List[str] = []
    current: List[str] = []

    for line in lines:
        if line.startswith("### "):
            # Start of a new item
            if current:
                items.append("\n".join(current).strip())
                current = []
        if line.startswith("### ") or current:
            current.append(line)

    if current:
        items.append("\n".join(current).strip())

    # Filter out possible leading metadata like "## Search results..."
    items = [item for item in items if item.startswith("### ")]
    return items


def chunk_items(items: List[str], items_per_chunk: int, max_items: int = 0) -> List[Tuple[int, str]]:
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
        for global_idx, item in enumerate(sub_items, start=start + 1):
            labelled.append(f"ITEM {global_idx}:\n{item}")
        chunk_text = "\n\n".join(labelled)
        chunks.append((chunk_idx, chunk_text))
    return chunks


def build_messages(
    previous_posts_summary: str,
    chunk_text: str,
    max_candidates_per_chunk: int,
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
        "You will receive:\n"
        "  1) A description of previous blog posts (topics, titles, and descriptions).\n"
        "  2) A chunk of news/search results, each labelled as ITEM N.\n\n"
        "Your job is to pick up to "
        f"{max_candidates_per_chunk} of the most promising items in this chunk that would make\n"
        "interesting *new* blog posts or natural follow-ups to existing posts.\n\n"
        "Guidelines:\n"
        "- Prefer items that are connected to previous topics (embedded, C/C++, ESP32, fuzzing,\n"
        "  networking, secure channels, cryptography, TTCN-3, MQTT, Kafka, console/tunnelling, etc.),\n"
        "  but avoid repeating essentially the same post.\n"
        "- Reward novelty, concrete technical depth, and opportunities for experiments, PoCs, or\n"
        "  conference-talk-style writeups.\n"
        "- Ignore obviously off-topic or generic items.\n\n"
        "Output strictly valid JSON with the following structure:\n"
        "{\n"
        '  "candidates": [\n'
        "    {\n"
        '      "item_id": <integer ITEM number>,\n'
        '      "title": "short working blog post title",\n'
        '      "source_title": "title or main line from the news item",\n'
        '      "reason": "why this is a good fit given previous posts",\n'
        '      "angle": "suggested angle or twist for the post"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Return an empty list if nothing is suitable.\n"
    )

    user_content = (
        "Previous blog posts (reference only):\n"
        "------------------------------------\n"
        f"{previous_posts_summary}\n\n"
        "News/search results chunk:\n"
        "--------------------------\n"
        f"{chunk_text}\n\n"
        f"Select up to {max_candidates_per_chunk} best candidates from this chunk."
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
) -> Dict[str, Any]:
    """Call the LLM once for a single chunk and parse JSON response."""
    messages = build_messages(previous_posts_summary, chunk_text, max_candidates_per_chunk)

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
    """Merge candidates from all chunks, de-duplicating by (item_id, title)."""
    merged: List[Dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()

    for result in all_chunks_results:
        for c in result.get("candidates", []):
            key = (c.get("item_id"), c.get("title"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(c)
    return merged


def print_summary(candidates: List[Dict[str, Any]]) -> None:
    """Print a compact, human-readable summary of merged candidates."""
    if not candidates:
        print("No suitable candidates found.")
        return

    print("Merged blog post candidates:\n")
    for idx, c in enumerate(candidates, start=1):
        item_id = c.get("item_id")
        title = c.get("title") or "<no title>"
        source_title = c.get("source_title") or ""
        reason = c.get("reason") or ""
        angle = c.get("angle") or ""

        print(f"{idx}. {title}")
        if item_id is not None:
            print(f"   - Source ITEM: {item_id}")
        if source_title:
            print(f"   - Source title: {source_title}")
        if reason:
            print(f"   - Why: {reason}")
        if angle:
            print(f"   - Angle: {angle}")
        print()


def main(argv: list[str] | None = None) -> int:
    if OpenAI is None:
        print(
            "Error: openai package is required. Install with: pip install openai",
            file=sys.stderr,
        )
        return 1

    args = parse_args(argv)

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

    if not items:
        print("No news items found in the input file.", file=sys.stderr)
        return 1

    chunks = chunk_items(items, args.items_per_chunk, max_items=args.max_items)

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
            )
            all_results.append(result)
        except Exception as e:  # pragma: no cover - network/LLM errors
            print(f"Error calling API for chunk {chunk_idx}: {e}", file=sys.stderr)

    merged = merge_candidates(all_results)

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


