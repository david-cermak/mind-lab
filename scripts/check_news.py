#!/usr/bin/env python3
"""
Check the latest news on a configurable topic using DuckDuckGo search.

Supports configuration via environment variables or .env file.
Priority: Environment variables > .env file > default values

Usage:
    # Using environment variables
    export NEWS_TOPIC="security vulnerabilities"
    python scripts/check_news.py

    # Using .env file (create .env in project root)
    NEWS_TOPIC=security flaws/latest vulnerabilities/issues in embedded systems

    # Run with default topic
    python scripts/check_news.py
"""
from __future__ import annotations

import os
import sys
import time
import json
import random
import re
from itertools import combinations
from pathlib import Path
from typing import Optional
from argparse import ArgumentParser, Namespace

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None


def load_env_file(project_root: Path | None = None) -> None:
    """Load .env file if python-dotenv is available."""
    if load_dotenv is None:
        return
    
    if project_root is None:
        # Try to find project root (where .env might be)
        current = Path(__file__).resolve().parent
        # Go up to find project root (look for .env or common markers)
        for parent in [current, current.parent]:
            env_file = parent / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                return
        # Fallback: try current directory
        load_dotenv()
    else:
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)


def _parse_keywords(raw: str | None) -> list[str]:
    """Parse a comma / newline separated keyword string into a clean list."""
    if not raw:
        return []
    
    # Split on commas or newlines, strip whitespace, drop empties
    parts = re.split(r"[,\n]", raw)
    keywords = [p.strip() for p in parts if p.strip()]
    # Deduplicate but keep order
    seen: set[str] = set()
    unique_keywords: list[str] = []
    for kw in keywords:
        key = kw.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_keywords.append(kw)
    return unique_keywords


def get_keywords_config() -> list[str]:
    """
    Legacy helper to get list of keywords from environment / .env.
    
    Reads (in order of precedence):
    - Environment variables
    - .env file
    
    Expected variable (legacy):
        NEWS_KEYWORDS or KEYWORDS: comma- or newline-separated list of keywords.
    """
    # Load .env file first (lowest priority)
    load_env_file()
    
    raw = os.getenv("NEWS_KEYWORDS") or os.getenv("KEYWORDS")
    return _parse_keywords(raw)


def get_text_keywords_config() -> list[str]:
    """
    Get list of TEXT (general) keywords from environment / .env.
    
    Reads (in order of precedence):
    - TEXT_KEYWORDS
    - KEYWORDS (legacy fallback)
    """
    load_env_file()
    raw = os.getenv("TEXT_KEYWORDS") or os.getenv("KEYWORDS")
    return _parse_keywords(raw)


def get_news_keywords_config() -> list[str]:
    """
    Get list of NEWS-specific keywords from environment / .env.
    
    Reads (in order of precedence):
    - NEWS_KEYWORDS
    - KEYWORDS (legacy fallback)
    """
    load_env_file()
    raw = os.getenv("NEWS_KEYWORDS") or os.getenv("KEYWORDS")
    return _parse_keywords(raw)


def get_target_results(default: int = 100) -> int:
    """Get desired total number of results from env, defaulting to 100."""
    load_env_file()
    raw = os.getenv("NEWS_TARGET_RESULTS") or os.getenv("NEWS_RESULTS") or ""
    if not raw:
        return default
    try:
        value = int(raw)
        if value <= 0:
            return default
        return value
    except ValueError:
        return default


def get_max_keywords_per_query(default: int = 6) -> int:
    """
    Legacy helper: maximum number of keywords to use in a single query.
    
    Controlled via environment variable (legacy):
        NEWS_MAX_KEYWORDS_PER_QUERY
    """
    load_env_file()
    raw = os.getenv("NEWS_MAX_KEYWORDS_PER_QUERY") or ""
    if not raw:
        return default
    try:
        value = int(raw)
        if value <= 0:
            return default
        return value
    except ValueError:
        return default


def get_text_max_keywords_per_query(default: int = 15) -> int:
    """
    Get maximum number of TEXT keywords to use in a single query.
    
    Controlled via environment variable:
        TEXT_MAX_KEYWORDS_PER_QUERY
    """
    load_env_file()
    raw = os.getenv("TEXT_MAX_KEYWORDS_PER_QUERY") or ""
    if not raw:
        return default
    try:
        value = int(raw)
        if value <= 0:
            return default
        return value
    except ValueError:
        return default


def get_news_max_keywords_per_query(default: int = 4) -> int:
    """
    Get maximum number of NEWS keywords to use in a single query.
    
    Controlled via environment variable:
        NEWS_MAX_KEYWORDS_PER_QUERY
    """
    load_env_file()
    raw = os.getenv("NEWS_MAX_KEYWORDS_PER_QUERY") or ""
    if not raw:
        return default
    try:
        value = int(raw)
        if value <= 0:
            return default
        return value
    except ValueError:
        return default


def get_config(topic: str | None = None) -> str:
    """
    Get configuration with priority: env vars > .env file > default.
    
    Returns:
        The news topic to search for
    """
    # Load .env file first (lowest priority)
    load_env_file()
    
    # Get from environment variables (highest priority)
    final_topic = (
        topic
        or os.getenv("NEWS_TOPIC")
    )
    
    # Default value if not set
    if not final_topic:
        final_topic = "security flaws/latest vulnerabilities/issues in embedded systems"
    
    return final_topic


def search_news(topic: str, max_results: int = 5, do_news: bool = True, max_retries: int = 3, retry_delay: int = 5) -> list[dict]:
    """
    Search for news articles on the given topic using DuckDuckGo.
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to return
        max_retries: Maximum number of retry attempts on rate limit
        retry_delay: Delay in seconds between retries
        
    Returns:
        List of dictionaries containing news results with keys:
        - title: Article title
        - url: Article URL
        - snippet: Article snippet/description
        - date: Publication date (if available)
    """
    if DDGS is None:
        raise ImportError(
            "ddgs is required. Install it with: pip install ddgs"
        )
    
    last_error = None
    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                # Search for news articles
                if do_news:
                    results = list(ddgs.news(
                        query=topic,
                        max_results=max_results
                    ))
                else:
                    results = list(ddgs.text(
                        query=topic,
                        max_results=max_results
                    ))

                return results
        except Exception as e:
            error_str = str(e)
            last_error = e
            
            # Check if it's a rate limit error
            if "Ratelimit" in error_str or "202" in error_str or "rate limit" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                    print(f"Rate limited. Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                    time.sleep(wait_time)
                    continue
                else:
                    raise RuntimeError(
                        f"Rate limited after {max_retries} attempts. "
                        f"Please wait a few minutes before trying again. Error: {e}"
                    ) from e
            else:
                # Not a rate limit error, raise immediately
                raise RuntimeError(f"Error searching for news: {e}") from e
    
    # If we get here, all retries failed
    raise RuntimeError(f"Error searching for news after {max_retries} attempts: {last_error}") from last_error


def search_with_keyword_combinations(
    text_keywords: list[str],
    news_keywords: list[str],
    target_results: int = 100,
    text_max_keywords_per_query: int | None = None,
    news_max_keywords_per_query: int | None = None,
    max_retries: int = 3,
    retry_delay: int = 5,
) -> list[dict]:
    """
    Search using combinations of keywords, starting with more specific queries.
    
    The algorithm:
    - Starts with queries that use all keywords, then progressively fewer.
    - For each keyword count k, it generates random combinations of size k.
    - It keeps querying until `target_results` unique results are collected
      (or all combinations are exhausted).
    - Each result is annotated with the query that produced it.
    
    Returns a list of dicts with at least:
        - title
        - url
        - summary
        - date
        - query_text
        - query_keywords
    """
    if not text_keywords and not news_keywords:
        return []
    
    # Work on copies so we can shuffle safely
    base_text_keywords = list(text_keywords)
    base_news_keywords = list(news_keywords)
    random.shuffle(base_text_keywords)
    random.shuffle(base_news_keywords)
    
    # Cap maximum keywords per query for each mode
    if text_max_keywords_per_query is None:
        text_max_keywords_per_query = get_text_max_keywords_per_query(default=15)
    if news_max_keywords_per_query is None:
        news_max_keywords_per_query = get_news_max_keywords_per_query(default=4)
    
    if base_text_keywords:
        text_max_keywords_per_query = max(
            1, min(text_max_keywords_per_query, len(base_text_keywords))
        )
    else:
        text_max_keywords_per_query = 0
    
    if base_news_keywords:
        news_max_keywords_per_query = max(
            1, min(news_max_keywords_per_query, len(base_news_keywords))
        )
    else:
        news_max_keywords_per_query = 0
    
    collected: list[dict] = []
    seen_urls: set[str] = set()
    
    total_target = max(target_results, 1)
    
    # For each level of specificity (k keywords per query), randomly sample
    # combinations instead of generating all of them, to avoid enormous query counts.
    max_queries_per_level = 20
    
    # Track which combinations we've already used for each mode and k
    text_queries_tried: dict[int, set[tuple[str, ...]]] = {}
    news_queries_tried: dict[int, set[tuple[str, ...]]] = {}
    
    # Safety limit to avoid infinite loops if APIs start failing badly
    max_total_queries = 1000
    total_queries_issued = 0
    
    while len(collected) < total_target and total_queries_issued < max_total_queries:
        # Determine which modes are currently usable (have keywords and remaining combos)
        available_modes: list[str] = []
        
        if text_max_keywords_per_query > 0 and base_text_keywords:
            for k in range(text_max_keywords_per_query, 0, -1):
                if len(base_text_keywords) >= k:
                    tried = text_queries_tried.get(k, set())
                    if len(tried) < max_queries_per_level:
                        available_modes.append("text")
                        break
        
        if news_max_keywords_per_query > 0 and base_news_keywords:
            for k in range(news_max_keywords_per_query, 0, -1):
                if len(base_news_keywords) >= k:
                    tried = news_queries_tried.get(k, set())
                    if len(tried) < max_queries_per_level:
                        available_modes.append("news")
                        break
        
        if not available_modes:
            break
        
        # Randomly choose whether this query will be text or news, among available
        mode = random.choice(available_modes)
        
        if mode == "text":
            base_list = base_text_keywords
            max_k = text_max_keywords_per_query
            queries_tried_map = text_queries_tried
            use_news = False
        else:
            base_list = base_news_keywords
            max_k = news_max_keywords_per_query
            queries_tried_map = news_queries_tried
            use_news = True
        
        # Choose the highest k that still has room for new combinations
        k_chosen = None
        for k in range(max_k, 0, -1):
            if len(base_list) < k:
                continue
            tried = queries_tried_map.get(k, set())
            if len(tried) < max_queries_per_level:
                k_chosen = k
                break
        
        if k_chosen is None:
            # No valid k for this mode, try again (other mode may still work)
            continue
        
        # Randomly sample a new combination of size k_chosen
        tried_set = queries_tried_map.setdefault(k_chosen, set())
        combo: tuple[str, ...] | None = None
        attempts = 0
        while attempts < 50:
            candidate = tuple(sorted(random.sample(base_list, k_chosen)))
            if candidate not in tried_set:
                combo = candidate
                tried_set.add(candidate)
                break
            attempts += 1
        
        if combo is None:
            # Could not find a new combination for this k, try again
            continue
        
        if len(collected) >= total_target:
            break
        
        query_keywords = list(combo)
        query_text = " ".join(query_keywords)
        remaining = total_target - len(collected)
        
        try:
            raw_results = search_news(
                topic=query_text,
                do_news=use_news,
                max_results=remaining,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
            total_queries_issued += 1
        except Exception as e:
            print(f"Error searching for '{query_text}': {e}", file=sys.stderr)
            continue
        
        for item in raw_results:
            url = item.get("href") or item.get("url")
            if not url:
                continue
            if url in seen_urls:
                continue
            
            seen_urls.add(url)
            result = {
                "title": item.get("title") or "No title",
                "url": url,
                "summary": item.get("body") or item.get("snippet") or "",
                "date": item.get("date"),
                "query_text": query_text,
                "query_keywords": query_keywords,
                "source": "news" if use_news else "text",
            }
            collected.append(result)
            
            if len(collected) >= total_target:
                break
    
    return collected


def format_results_markdown(results: list[dict]) -> str:
    """Format aggregated search results as Markdown."""
    if not results:
        return "No results found."
    
    lines: list[str] = []
    lines.append(f"## Search results ({len(results)} item(s))\n")
    
    for idx, result in enumerate(results, 1):
        title = result.get("title") or "No title"
        url = result.get("url") or "N/A"
        summary = result.get("summary") or ""
        query_text = result.get("query_text") or ""
        query_keywords = result.get("query_keywords") or []
        date = result.get("date")
        source = result.get("source")
        
        lines.append(f"### {idx}. {title}")
        lines.append("")
        lines.append(f"- **URL**: {url}")
        if summary:
            lines.append(f"- **Summary**: {summary}")
        if query_keywords:
            lines.append(f"- **Query keywords**: {', '.join(query_keywords)}")
        if query_text:
            lines.append(f"- **Query text**: `{query_text}`")
        if source:
            lines.append(f"- **Source**: {source}")
        if date:
            lines.append(f"- **Date**: {date}")
        
        lines.append("")
    
    return "\n".join(lines)


def results_to_json(results: list[dict]) -> str:
    """Serialize results to pretty-printed JSON."""
    return json.dumps(results, indent=2, ensure_ascii=False)


def parse_args(argv: list[str] | None = None) -> Namespace:
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Check latest news/text topics via DuckDuckGo.")
    parser.add_argument(
        "-f",
        "--format",
        dest="output_format",
        choices=["json", "md", "markdown", "both"],
        default="both",
        help="Output format: json, md/markdown, or both (default).",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        help="Optional output file. If omitted, results are printed to stdout.",
    )
    return parser.parse_args(argv)


def main() -> int:
    """Main entry point."""
    # Check if ddgs is available
    if DDGS is None:
        print("Error: ddgs is required.", file=sys.stderr)
        print("Install it with: pip install ddgs", file=sys.stderr)
        return 1
    
    # Parse CLI arguments
    args = parse_args()
    fmt = args.output_format
    output_file = args.output_file
    if fmt == "markdown":
        fmt = "md"
    
    # First, get keyword-based configuration
    try:
        text_keywords = get_text_keywords_config()
        news_keywords = get_news_keywords_config()
        target_results = get_target_results(default=100)
        text_max_keywords_per_query = get_text_max_keywords_per_query(default=15)
        news_max_keywords_per_query = get_news_max_keywords_per_query(default=4)
    except Exception as e:
        print(f"Error loading keyword configuration: {e}", file=sys.stderr)
        return 1
    
    if not text_keywords and not news_keywords:
        print(
            "No keywords configured. Please set TEXT_KEYWORDS and/or NEWS_KEYWORDS (or legacy KEYWORDS) "
            "in your environment or .env file.",
            file=sys.stderr,
        )
        return 1
    
    print(
        f"Using keyword-combination search with "
        f"{len(text_keywords)} text keyword(s) and {len(news_keywords)} news keyword(s)."
    )
    print(f"Target total results: {target_results}")
    print(
        f"Max text keywords per query: {text_max_keywords_per_query}, "
        f"max news keywords per query: {news_max_keywords_per_query}"
    )
    print("Using DuckDuckGo news/text search (random per query, mode-specific configs)...\n")
    
    try:
        results = search_with_keyword_combinations(
            text_keywords=text_keywords,
            news_keywords=news_keywords,
            target_results=target_results,
            text_max_keywords_per_query=text_max_keywords_per_query,
            news_max_keywords_per_query=news_max_keywords_per_query,
        )
    except Exception as e:
        print(f"Error during keyword-combination search: {e}", file=sys.stderr)
        return 1
    
    # Prepare JSON and Markdown representations
    json_str = results_to_json(results)
    md_str = format_results_markdown(results)
    
    # Decide what to output based on format
    if fmt == "json":
        final_output = json_str
    elif fmt == "md":
        final_output = md_str
    else:  # both
        final_output = "=== JSON results ===\n" + json_str + "\n\n=== Markdown results ===\n\n" + md_str
    
    if output_file:
        # Write to file
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(final_output, encoding="utf-8")
        print(f"Wrote results to {out_path}")
    else:
        # Print to stdout
        print(final_output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

