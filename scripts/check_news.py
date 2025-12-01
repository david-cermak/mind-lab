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
from pathlib import Path
from typing import Optional

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


def search_news(topic: str, max_results: int = 10, max_retries: int = 3, retry_delay: int = 5) -> list[dict]:
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
                results = list(ddgs.news(
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


def format_results(results: list[dict]) -> str:
    """Format search results for display."""
    if not results:
        return "No news articles found."
    
    output = []
    output.append(f"\n{'='*80}")
    output.append(f"Found {len(results)} news article(s):")
    output.append(f"{'='*80}\n")
    
    for i, result in enumerate(results, 1):
        output.append(f"{i}. {result.get('title', 'No title')}")
        output.append(f"   URL: {result.get('url', 'N/A')}")
        
        if 'body' in result:
            snippet = result['body']
            # Truncate long snippets
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            output.append(f"   Summary: {snippet}")
        
        if 'date' in result:
            output.append(f"   Date: {result['date']}")
        
        output.append("")
    
    return "\n".join(output)


def main() -> int:
    """Main entry point."""
    # Check if ddgs is available
    if DDGS is None:
        print("Error: ddgs is required.", file=sys.stderr)
        print("Install it with: pip install ddgs", file=sys.stderr)
        return 1
    
    # Get configuration
    try:
        topic = get_config()
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1
    
    print(f"Searching for news on: {topic}")
    print("Using DuckDuckGo search engine...\n")
    
    # Search for news
    try:
        results = search_news(topic, max_results=10)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Display results
    output = format_results(results)
    print(output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

