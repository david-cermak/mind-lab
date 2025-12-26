#!/usr/bin/env python3
"""
Finalize blog candidates selected by scripts/select_blog_candidates.py.

Given a selected.json file (with "candidates"), this script:
  1) Dedupes and visits each candidate's URL(s) and extracts page text (best-effort).
     - Uses crawl4ai (preferred) for HTML extraction.
     - Downloads PDFs to disk and extracts text with pypdf.
     - Uses a local cache file to avoid re-downloading.
  2) Calls an OpenAI-compatible LLM to produce:
     - A blog-ready summary (200–500 words, markdown)
     - novelty_score (0–10)
     - relevance_score (0–10) based on keyword match from .env
     - a short summary (~20 words)
  3) Writes:
     - blog_candidate_<slug>.md per candidate
     - final_report.json with scores + pointers to markdown files

Configuration:
  Priority: CLI args > environment variables > .env file
  - API: OPENAI_BASE_URL/BASE_URL, OPENAI_MODEL/MODEL, OPENAI_API_KEY/API_KEY
  - Relevance keywords:
      BLOG_RELEVANCE_KEYWORDS (preferred), else TEXT_KEYWORDS/NEWS_KEYWORDS/KEYWORDS
  - Content limits:
      BLOG_FINALIZE_MAX_LINKS_PER_CANDIDATE (default: 3)
      BLOG_FINALIZE_MAX_CONTENT_CHARS (default: 20000)
      BLOG_FINALIZE_SOURCES_APPEND_CHARS (default: 8000)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import gc
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


def load_env_file(project_root: Path | None = None) -> None:
    """Load .env file if python-dotenv is available."""
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


def get_api_config(
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[str, str, str]:
    """Get API configuration with priority: CLI args > env vars > .env file."""
    load_env_file()
    final_base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL")
    final_model = model or os.getenv("OPENAI_MODEL") or os.getenv("MODEL")
    final_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not final_base_url:
        raise ValueError("base-url is required (OPENAI_BASE_URL or --base-url)")
    if not final_model:
        raise ValueError("model is required (OPENAI_MODEL or --model)")
    if not final_api_key:
        raise ValueError("api-key is required (OPENAI_API_KEY or --api-key)")
    return final_base_url, final_model, final_api_key


def _parse_keywords(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = re.split(r"[,\n]", raw)
    kws = [p.strip() for p in parts if p.strip()]
    seen: set[str] = set()
    out: list[str] = []
    for kw in kws:
        key = kw.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(kw)
    return out


def get_relevance_keywords() -> list[str]:
    load_env_file()
    raw = (
        os.getenv("BLOG_RELEVANCE_KEYWORDS")
        or os.getenv("BLOG_KEYWORDS")
        or os.getenv("TEXT_KEYWORDS")
        or os.getenv("NEWS_KEYWORDS")
        or os.getenv("KEYWORDS")
    )
    return _parse_keywords(raw)


def get_int_env(name: str, default: int) -> int:
    load_env_file()
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default


def slugify(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t or "candidate"


def normalize_links(candidate: Dict[str, Any]) -> list[str]:
    # Prefer `link`, then `url`
    raw = candidate.get("link")
    if raw is None:
        raw = candidate.get("url")
    links: list[str] = []
    if isinstance(raw, list):
        for x in raw:
            if isinstance(x, str) and x.strip():
                links.append(x.strip())
    elif isinstance(raw, str) and raw.strip():
        links = [raw.strip()]
    # Dedup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for u in links:
        key = u.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _http_fetch(url: str, timeout_seconds: int = 30) -> tuple[str, str]:
    """
    Fetch URL and return (content_type, body_text).
    Uses httpx if available, otherwise urllib.
    """
    try:
        import httpx  # type: ignore
    except Exception:
        httpx = None  # type: ignore
    headers = {
        "User-Agent": "mind-lab-blog-candidate-bot/1.0 (+https://example.invalid)",
        "Accept": "*/*",
    }
    if httpx is not None:
        with httpx.Client(timeout=timeout_seconds, headers=headers, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            return ctype, resp.text
    # urllib fallback
    import urllib.request

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_seconds) as r:  # nosec - user supplied URL
        ctype = r.headers.get("content-type", "") or ""
        data = r.read()
    try:
        return ctype, data.decode("utf-8", errors="replace")
    except Exception:
        return ctype, data.decode(errors="replace")


def _html_to_text(html: str) -> str:
    # Prefer BeautifulSoup if available, else fallback to naive tag stripping.
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        BeautifulSoup = None  # type: ignore
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        # Drop scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        # Normalize whitespace
        lines = [ln.strip() for ln in text.splitlines()]
        lines = [ln for ln in lines if ln]
        return "\n".join(lines)
    # Naive fallback
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _pdf_to_text_from_bytes(pdf_bytes: bytes, max_pages: int = 12) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:
        raise RuntimeError("pypdf is required to extract text from PDFs") from e
    import io

    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            parts.append(t.strip())
    return "\n\n".join(parts).strip()


def _fetch_pdf_bytes(url: str, timeout_seconds: int = 30) -> bytes:
    try:
        import httpx  # type: ignore
    except Exception:
        httpx = None  # type: ignore
    headers = {
        "User-Agent": "mind-lab-blog-candidate-bot/1.0 (+https://example.invalid)",
        "Accept": "application/pdf,*/*",
    }
    if httpx is not None:
        with httpx.Client(timeout=timeout_seconds, headers=headers, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.content
    import urllib.request

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_seconds) as r:  # nosec - user supplied URL
        return r.read()

def _looks_like_pdf(url: str, content_type: str = "") -> bool:
    u = (url or "").strip().lower()
    ct = (content_type or "").strip().lower()
    if u.endswith(".pdf"):
        return True
    if "application/pdf" in ct or ct.startswith("application/pdf"):
        return True
    return False


def _safe_filename_from_url(url: str) -> str:
    # A stable filename for downloads based on URL hash
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"source_{h}"


def fetch_and_extract(
    url: str,
    *,
    out_dir: Path,
    timeout_seconds: int,
    max_pdf_pages: int = 12,
) -> Dict[str, Any]:
    """
    Fetch and extract readable text for a URL.

    Returns dict with:
      - fetched_at: int
      - url: str
      - content_type: str
      - text: str
      - download_path: str (optional; for PDFs)
      - error: str (optional)
    """
    fetched_at = int(time.time())
    u = (url or "").strip()
    if not u:
        return {"fetched_at": fetched_at, "url": u, "content_type": "", "text": "", "error": "empty_url"}

    # Quick path for PDFs based on URL suffix (does NOT require crawl4ai).
    if _looks_like_pdf(u):
        try:
            pdf_bytes = _fetch_pdf_bytes(u, timeout_seconds=timeout_seconds)
            downloads = out_dir / "downloads"
            downloads.mkdir(parents=True, exist_ok=True)
            filename = _safe_filename_from_url(u) + ".pdf"
            pdf_path = downloads / filename
            pdf_path.write_bytes(pdf_bytes)
            text = ""
            err: str | None = None
            try:
                text = _pdf_to_text_from_bytes(pdf_bytes, max_pages=max_pdf_pages)
            except Exception as e:
                # Still consider the PDF "downloaded" even if text extraction fails.
                err = str(e)
            return {
                "fetched_at": fetched_at,
                "url": u,
                "content_type": "application/pdf",
                "text": text,
                "download_path": str(pdf_path),
                **({"error": f"pdf_extract_failed: {err}"} if err else {}),
            }
        except Exception as e:
            return {
                "fetched_at": fetched_at,
                "url": u,
                "content_type": "application/pdf",
                "text": "",
                "error": f"pdf_fetch_failed: {e}",
            }

    # HTML extraction is handled in a single shared crawl4ai session in main()
    # (to avoid per-URL asyncio.run() which can leave subprocess transports alive).
    return {"fetched_at": fetched_at, "url": u, "content_type": "", "text": "", "error": "html_fetch_should_be_batched"}


async def crawl4ai_extract_many(urls: list[str], timeout_seconds: int = 30) -> Dict[str, Dict[str, Any]]:
    """
    Extract many URLs in a single crawl4ai session + event loop.
    This avoids the common 'Event loop is closed' warning caused by repeatedly creating/closing loops.
    """
    from crawl4ai import AsyncWebCrawler  # type: ignore

    results: Dict[str, Dict[str, Any]] = {}
    async with AsyncWebCrawler(verbose=False) as crawler:
        for u in urls:
            try:
                r = await crawler.arun(url=u)
                md = getattr(r, "markdown", None)
                if isinstance(md, str) and md.strip():
                    results[u] = {"content_type": "text/markdown", "text": md.strip()}
                    continue
                cleaned = getattr(r, "cleaned_text", None)
                if isinstance(cleaned, str) and cleaned.strip():
                    results[u] = {"content_type": "text/plain", "text": cleaned.strip()}
                    continue
                html = getattr(r, "html", None)
                if isinstance(html, str) and html.strip():
                    results[u] = {"content_type": "text/html", "text": _html_to_text(html)}
                    continue
                results[u] = {"content_type": "", "text": ""}
            except Exception as e:
                results[u] = {"content_type": "", "text": "", "error": f"crawl_failed: {e}"}
    return results


async def crawl4ai_extract_one(crawler: Any, url: str) -> Dict[str, Any]:
    """
    Extract a single URL using an existing AsyncWebCrawler instance.
    Kept separate so we can reuse the crawler for the entire run.
    """
    u = (url or "").strip()
    if not u:
        return {"content_type": "", "text": "", "error": "empty_url"}
    try:
        r = await crawler.arun(url=u)
        md = getattr(r, "markdown", None)
        if isinstance(md, str) and md.strip():
            return {"content_type": "text/markdown", "text": md.strip()}
        cleaned = getattr(r, "cleaned_text", None)
        if isinstance(cleaned, str) and cleaned.strip():
            return {"content_type": "text/plain", "text": cleaned.strip()}
        html = getattr(r, "html", None)
        if isinstance(html, str) and html.strip():
            return {"content_type": "text/html", "text": _html_to_text(html)}
        return {"content_type": "", "text": ""}
    except Exception as e:
        return {"content_type": "", "text": "", "error": f"crawl_failed: {e}"}


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            # Back-compat: previous versions stored {fetched_at, text}
            return data  # url -> {fetched_at, text, content_type?, download_path?, error?}
    except Exception:
        pass
    return {}


def save_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def build_llm_messages(
    candidate: Dict[str, Any],
    sources: List[Dict[str, str]],
    keywords: List[str],
    min_words: int,
    max_words: int,
) -> list[dict[str, Any]]:
    title = candidate.get("title") or ""
    reason = candidate.get("reason") or ""
    angle = candidate.get("angle") or ""
    kws = ", ".join(keywords[:50])

    system = (
        "You are helping an embedded / firmware developer choose final blog posts.\n"
        "You will be given extracted page contents from 1+ sources that inspired a candidate.\n"
        "Your job is to produce a blog-ready summary and score novelty/relevance.\n"
        "Output strictly valid JSON only.\n"
    )

    user_parts: list[str] = []
    user_parts.append("Candidate:")
    user_parts.append(f"- Title: {title}")
    if reason:
        user_parts.append(f"- Reason: {reason}")
    if angle:
        user_parts.append(f"- Angle: {angle}")
    user_parts.append("")
    user_parts.append("Relevance keywords (for scoring):")
    user_parts.append(kws if kws else "(none provided)")
    user_parts.append("")
    user_parts.append("Sources (extracted content):")
    user_parts.append("----------------------------------------")
    for i, s in enumerate(sources, start=1):
        user_parts.append(f"SOURCE {i}:")
        user_parts.append(f"- URL: {s.get('url','')}")
        st = s.get("source_title", "")
        if st:
            user_parts.append(f"- Source title: {st}")
        summ = s.get("summary", "")
        if summ:
            user_parts.append(f"- Original snippet: {summ}")
        user_parts.append("")
        content = s.get("content", "") or ""
        if not content.strip():
            user_parts.append("(No content extracted; use title/snippet only.)")
        else:
            user_parts.append(content.strip())
        user_parts.append("")
        user_parts.append("----------------------------------------")
    user_parts.append("")
    user_parts.append("Return JSON with this schema:")
    user_parts.append("{")
    user_parts.append('  "novelty_score": 0-10,')
    user_parts.append('  "relevance_score": 0-10,')
    user_parts.append('  "short_summary": "about 20 words",')
    user_parts.append(f'  "blog_summary_markdown": "{min_words}-{max_words} words, markdown",')
    user_parts.append('  "matched_keywords": ["keyword", ...]')
    user_parts.append("}")
    user_parts.append("")
    user_parts.append("Scoring guidance:")
    user_parts.append("- novelty_score: how new/interesting the idea is (not generic, not rehash).")
    user_parts.append("- relevance_score: how strongly it matches the given keywords and embedded/firmware topics.")
    user_parts.append(f"- blog_summary_markdown must be {min_words}-{max_words} words.")
    user = "\n".join(user_parts)

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_llm(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
) -> Dict[str, Any]:
    resp = client.chat.completions.create(model=model, messages=messages)
    if not resp.choices:
        return {}
    content = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _ensure_list_str(x: Any) -> list[str]:
    if isinstance(x, list):
        return [str(i) for i in x]
    if x is None:
        return []
    return [str(x)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    load_env_file()
    p = argparse.ArgumentParser(description="Finalize blog candidates from selected.json")
    p.add_argument("--selected-file", default="", help="Path to selected.json (default: output/selected.json if exists).")
    p.add_argument(
        "--output-dir",
        default="output/final_candidates",
        help="Directory for markdown files and final_report.json (default: output/final_candidates).",
    )
    p.add_argument("--cache-file", default="", help="Optional crawl cache file path (JSON).")
    p.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip OpenAI summary/scoring generation (still crawls and writes markdown/report).",
    )
    p.add_argument("--base-url", dest="base_url", default="", help="API base URL (or OPENAI_BASE_URL).")
    p.add_argument("--model", default="", help="Model name (or OPENAI_MODEL).")
    p.add_argument("--api-key", dest="api_key", default="", help="API key (or OPENAI_API_KEY).")
    p.add_argument(
        "--max-links-per-candidate",
        type=int,
        default=get_int_env("BLOG_FINALIZE_MAX_LINKS_PER_CANDIDATE", 3),
        help="Max deduped links per candidate (default: 3; env BLOG_FINALIZE_MAX_LINKS_PER_CANDIDATE).",
    )
    p.add_argument(
        "--max-content-chars",
        type=int,
        default=get_int_env("BLOG_FINALIZE_MAX_CONTENT_CHARS", 20000),
        help="Max chars of extracted content per source to send to LLM (default: 20000; env BLOG_FINALIZE_MAX_CONTENT_CHARS).",
    )
    p.add_argument(
        "--sources-append-chars",
        type=int,
        default=get_int_env("BLOG_FINALIZE_SOURCES_APPEND_CHARS", 8000),
        help="Max chars of cached source text to append into markdown (default: 8000; env BLOG_FINALIZE_SOURCES_APPEND_CHARS).",
    )
    p.add_argument("--min-words", type=int, default=200, help="Minimum words for blog summary (default: 200).")
    p.add_argument("--max-words", type=int, default=500, help="Maximum words for blog summary (default: 500).")
    p.add_argument("--timeout-seconds", type=int, default=30, help="Fetch timeout per URL (default: 30).")
    return p.parse_args(argv)


def _default_selected_path() -> Path:
    # Prefer repo-root output/selected.json if it exists; else scripts/output/selected.json
    root = Path(os.getcwd())
    p1 = root / "output" / "selected.json"
    if p1.exists():
        return p1
    p2 = root / "scripts" / "output" / "selected.json"
    return p2


def _prepare_run(args: argparse.Namespace) -> tuple[Path, Path, Dict[str, Dict[str, Any]], list[dict[str, Any]], Path, list[str], Any, str]:
    """
    Load inputs and return (selected_path, out_dir, cache, candidates, cache_path, keywords, client, model).
    client may be None when --skip-llm is set.
    """
    if not args.skip_llm and OpenAI is None:
        raise RuntimeError("openai package is required. Install with: pip install openai")

    selected_path = Path(args.selected_file) if args.selected_file else _default_selected_path()
    if not selected_path.exists():
        raise RuntimeError(f"selected.json not found at {selected_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.cache_file) if args.cache_file else (out_dir / "crawl_cache.json")
    cache = load_cache(cache_path)

    keywords = get_relevance_keywords()

    client = None
    model = ""
    if not args.skip_llm:
        try:
            base_url, model, api_key = get_api_config(
                base_url=args.base_url or None,
                model=args.model or None,
                api_key=args.api_key or None,
            )
        except ValueError as e:
            raise RuntimeError(str(e))
        client = OpenAI(base_url=base_url, api_key=api_key)

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    candidates = selected.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("No candidates found in selected.json")
    # Narrow type
    cand_list: list[dict[str, Any]] = [c for c in candidates if isinstance(c, dict)]
    return selected_path, out_dir, cache, cand_list, cache_path, keywords, client, model


async def main_async(argv: list[str] | None = None) -> int:
    """
    Async main used when crawl4ai is available.
    Runs a single event loop for the whole program, and keeps one AsyncWebCrawler alive
    across all candidates to avoid 'Event loop is closed' finalizer warnings.
    """
    args = parse_args(argv)
    try:
        selected_path, out_dir, cache, candidates, cache_path, keywords, client, model = _prepare_run(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Import crawl4ai here; if it fails, we'll fall back to sync behavior below.
    from crawl4ai import AsyncWebCrawler  # type: ignore

    report_items: list[dict[str, Any]] = []

    async with AsyncWebCrawler(verbose=False) as crawler:
        for idx, cand in enumerate(candidates, start=1):
            title = str(cand.get("title") or "").strip()
            slug = slugify(title)[:80]
            md_path = out_dir / f"blog_candidate_{slug}.md"

            links = normalize_links(cand)
            if args.max_links_per_candidate > 0:
                links = links[: args.max_links_per_candidate]

            source_titles = _ensure_list_str(cand.get("source_title"))
            source_summaries = _ensure_list_str(cand.get("summary"))

            sources: list[dict[str, Any]] = []

            # Fetch/crawl missing sources
            for i, url in enumerate(links):
                cached = cache.get(url) or {}
                cached_text = str(cached.get("text") or "")

                if not cached_text and not cached.get("download_path"):
                    print(f"[{idx}/{len(candidates)}] Fetching {url}", file=sys.stderr)
                    if _looks_like_pdf(url):
                        fetched = fetch_and_extract(url, out_dir=out_dir, timeout_seconds=args.timeout_seconds)
                    else:
                        ex = await crawl4ai_extract_one(crawler, url)
                        fetched = {
                            "fetched_at": int(time.time()),
                            "url": url,
                            "content_type": str(ex.get("content_type") or ""),
                            "text": str(ex.get("text") or ""),
                            **({"error": str(ex.get("error"))} if ex.get("error") else {}),
                        }
                    cache[url] = fetched
                    cached = fetched
                    cached_text = str(cached.get("text") or "")

                content = cached_text
                if args.max_content_chars > 0 and len(content) > args.max_content_chars:
                    content = content[: args.max_content_chars] + "\n...[truncated]"

                sources.append(
                    {
                        "url": url,
                        "source_title": source_titles[i] if i < len(source_titles) else "",
                        "summary": source_summaries[i] if i < len(source_summaries) else "",
                        "content": content,
                        "cached": cached,
                    }
                )

            # Persist cache after each candidate
            try:
                save_cache(cache_path, cache)
            except Exception:
                pass

            novelty = None
            relevance = None
            short_summary = ""
            blog_md = ""
            matched_keywords: list[Any] = []
            if not args.skip_llm:
                messages = build_llm_messages(
                    candidate=cand,
                    sources=[
                        {"url": s["url"], "source_title": s["source_title"], "summary": s["summary"], "content": s["content"]}
                        for s in sources
                    ],
                    keywords=keywords,
                    min_words=max(50, int(args.min_words)),
                    max_words=max(int(args.min_words), int(args.max_words)),
                )
                result = call_llm(client=client, model=model, messages=messages) if client is not None else {}
                novelty = result.get("novelty_score")
                relevance = result.get("relevance_score")
                short_summary = str(result.get("short_summary") or "").strip()
                blog_md = str(result.get("blog_summary_markdown") or "").strip()
                mk = result.get("matched_keywords")
                matched_keywords = mk if isinstance(mk, list) else []
            else:
                if sources and sources[0].get("source_title"):
                    short_summary = str(sources[0].get("source_title") or "")[:120]

            # Write markdown (same format as sync path)
            md_lines: list[str] = []
            md_lines.append(f"## {title or 'Blog candidate'}")
            md_lines.append("")
            md_lines.append("### Sources")
            for s in sources:
                u = s.get("url", "")
                st = s.get("source_title", "")
                if st:
                    md_lines.append(f"- [{st}]({u})")
                else:
                    md_lines.append(f"- {u}")
            md_lines.append("")
            md_lines.append("### Scores")
            md_lines.append(f"- **Novelty (0-10)**: {novelty if novelty is not None else 'N/A'}")
            md_lines.append(f"- **Relevance (0-10)**: {relevance if relevance is not None else 'N/A'}")
            if matched_keywords:
                md_lines.append(f"- **Matched keywords**: {', '.join(str(x) for x in matched_keywords)}")
            md_lines.append("")
            md_lines.append("### Blog summary (200–500 words)")
            md_lines.append("")
            md_lines.append(blog_md or ("(LLM skipped.)" if args.skip_llm else "(LLM did not return a blog summary.)"))
            md_lines.append("")
            for i, s in enumerate(sources, start=1):
                cached = s.get("cached") or {}
                cached_text = str(cached.get("text") or "")
                cached_ct = str(cached.get("content_type") or "")
                cached_err = str(cached.get("error") or "")
                download_path = str(cached.get("download_path") or "")
                md_lines.append(f"## Sources ({i})")
                md_lines.append("")
                md_lines.append(f"- **URL**: {s.get('url','')}")
                if cached_ct:
                    md_lines.append(f"- **Content-Type**: {cached_ct}")
                if download_path:
                    md_lines.append(f"- **Downloaded file**: `{download_path}`")
                if cached_err:
                    md_lines.append(f"- **Error**: {cached_err}")
                md_lines.append("")
                if not cached_text.strip():
                    md_lines.append("(No cached text.)")
                    md_lines.append("")
                    continue
                append_text = cached_text
                if args.sources_append_chars > 0 and len(append_text) > args.sources_append_chars:
                    append_text = append_text[: args.sources_append_chars] + "\n...[truncated]"
                md_lines.append("```text")
                md_lines.append(append_text)
                md_lines.append("```")
                md_lines.append("")
            md_path.write_text("\n".join(md_lines), encoding="utf-8")

            report_items.append(
                {
                    "title": title,
                    "item_ids": cand.get("item_ids", []),
                    "novelty_score": novelty,
                    "relevance_score": relevance,
                    "short_summary": short_summary,
                    "markdown_path": str(md_path),
                    "links": links,
                    "cache_file": str(cache_path),
                }
            )

    # Let async cleanup finish before loop potentially closes, then force gc while loop is still alive.
    # (crawl4ai/playwright sometimes leave transports that only finalize under GC)
    await asyncio.sleep(0.05)
    gc.collect()

    final_report = {"generated_at": int(time.time()), "selected_file": str(selected_path), "items": report_items}
    (out_dir / "final_report.json").write_text(json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(report_items)} items to {out_dir / 'final_report.json'}", file=sys.stderr)
    return 0


def _run_coro_keep_loop_open(coro: Any) -> int:
    """
    Run a coroutine in a dedicated event loop but *do not close the loop*.

    Why:
      Some crawl4ai/playwright subprocess transports can be finalized late (during interpreter shutdown).
      If the event loop is closed (as asyncio.run does), their __del__ can throw:
        RuntimeError: Event loop is closed
    Keeping the loop open prevents that noisy shutdown traceback.

    This is a pragmatic script-level workaround; the process is exiting anyway.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(coro)
        # Best-effort: cancel any leftovers and let them drain while loop is still open
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_asyncgens())
        try:
            loop.run_until_complete(loop.shutdown_default_executor())
        except Exception:
            pass
        gc.collect()
        return int(result)
    finally:
        # Intentionally do not loop.close()
        # (avoid BaseSubprocessTransport.__del__ scheduling callbacks on a closed loop)
        pass


def main(argv: list[str] | None = None) -> int:
    """
    Sync entrypoint. If crawl4ai is available, we run the full program under a single asyncio.run().
    Otherwise we fall back to a minimal sync mode that only downloads PDFs and caches crawl4ai errors for HTML.
    """
    args = parse_args(argv)
    try:
        import crawl4ai  # type: ignore
        return _run_coro_keep_loop_open(main_async(argv))
    except Exception:
        # Fallback sync path (no crawl4ai): keeps existing behavior but without spawning subprocesses.
        try:
            selected_path, out_dir, cache, candidates, cache_path, keywords, client, model = _prepare_run(args)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        report_items: list[dict[str, Any]] = []
        for idx, cand in enumerate(candidates, start=1):
            title = str(cand.get("title") or "").strip()
            slug = slugify(title)[:80]
            md_path = out_dir / f"blog_candidate_{slug}.md"
            links = normalize_links(cand)
            if args.max_links_per_candidate > 0:
                links = links[: args.max_links_per_candidate]

            source_titles = _ensure_list_str(cand.get("source_title"))
            source_summaries = _ensure_list_str(cand.get("summary"))
            sources: list[dict[str, Any]] = []

            for i, url in enumerate(links):
                cached = cache.get(url) or {}
                cached_text = str(cached.get("text") or "")
                if not cached_text and not cached.get("download_path"):
                    print(f"[{idx}/{len(candidates)}] Fetching {url}", file=sys.stderr)
                    if _looks_like_pdf(url):
                        fetched = fetch_and_extract(url, out_dir=out_dir, timeout_seconds=args.timeout_seconds)
                    else:
                        # Best-effort HTML fetch+extract when crawl4ai isn't available.
                        try:
                            ctype, body = _http_fetch(url, timeout_seconds=args.timeout_seconds)
                            text = _html_to_text(body)
                            fetched = {
                                "fetched_at": int(time.time()),
                                "url": url,
                                "content_type": ctype or "text/html",
                                "text": text,
                            }
                        except Exception as e:
                            fetched = {
                                "fetched_at": int(time.time()),
                                "url": url,
                                "content_type": "",
                                "text": "",
                                "error": f"html_fetch_failed: {e}",
                            }
                    cache[url] = fetched
                    cached = fetched
                    cached_text = str(cached.get("text") or "")

                content = cached_text
                if args.max_content_chars > 0 and len(content) > args.max_content_chars:
                    content = content[: args.max_content_chars] + "\n...[truncated]"
                sources.append(
                    {
                        "url": url,
                        "source_title": source_titles[i] if i < len(source_titles) else "",
                        "summary": source_summaries[i] if i < len(source_summaries) else "",
                        "content": content,
                        "cached": cached,
                    }
                )

            try:
                save_cache(cache_path, cache)
            except Exception:
                pass

            novelty = None
            relevance = None
            short_summary = ""
            blog_md = ""
            matched_keywords: list[Any] = []
            if not args.skip_llm:
                messages = build_llm_messages(
                    candidate=cand,
                    sources=[
                        {"url": s["url"], "source_title": s["source_title"], "summary": s["summary"], "content": s["content"]}
                        for s in sources
                    ],
                    keywords=keywords,
                    min_words=max(50, int(args.min_words)),
                    max_words=max(int(args.min_words), int(args.max_words)),
                )
                result = call_llm(client=client, model=model, messages=messages) if client is not None else {}
                novelty = result.get("novelty_score")
                relevance = result.get("relevance_score")
                short_summary = str(result.get("short_summary") or "").strip()
                blog_md = str(result.get("blog_summary_markdown") or "").strip()
                mk = result.get("matched_keywords")
                matched_keywords = mk if isinstance(mk, list) else []

            md_lines: list[str] = []
            md_lines.append(f"## {title or 'Blog candidate'}")
            md_lines.append("")
            md_lines.append("### Sources")
            for s in sources:
                u = s.get("url", "")
                st = s.get("source_title", "")
                if st:
                    md_lines.append(f"- [{st}]({u})")
                else:
                    md_lines.append(f"- {u}")
            md_lines.append("")
            md_lines.append("### Scores")
            md_lines.append(f"- **Novelty (0-10)**: {novelty if novelty is not None else 'N/A'}")
            md_lines.append(f"- **Relevance (0-10)**: {relevance if relevance is not None else 'N/A'}")
            if matched_keywords:
                md_lines.append(f"- **Matched keywords**: {', '.join(str(x) for x in matched_keywords)}")
            md_lines.append("")
            md_lines.append("### Blog summary (200–500 words)")
            md_lines.append("")
            md_lines.append(blog_md or ("(LLM skipped.)" if args.skip_llm else "(LLM did not return a blog summary.)"))
            md_lines.append("")
            for i, s in enumerate(sources, start=1):
                cached = s.get("cached") or {}
                cached_text = str(cached.get("text") or "")
                cached_ct = str(cached.get("content_type") or "")
                cached_err = str(cached.get("error") or "")
                download_path = str(cached.get("download_path") or "")
                md_lines.append(f"## Sources ({i})")
                md_lines.append("")
                md_lines.append(f"- **URL**: {s.get('url','')}")
                if cached_ct:
                    md_lines.append(f"- **Content-Type**: {cached_ct}")
                if download_path:
                    md_lines.append(f"- **Downloaded file**: `{download_path}`")
                if cached_err:
                    md_lines.append(f"- **Error**: {cached_err}")
                md_lines.append("")
                if not cached_text.strip():
                    md_lines.append("(No cached text.)")
                    md_lines.append("")
                    continue
                append_text = cached_text
                if args.sources_append_chars > 0 and len(append_text) > args.sources_append_chars:
                    append_text = append_text[: args.sources_append_chars] + "\n...[truncated]"
                md_lines.append("```text")
                md_lines.append(append_text)
                md_lines.append("```")
                md_lines.append("")
            md_path.write_text("\n".join(md_lines), encoding="utf-8")

            report_items.append(
                {
                    "title": title,
                    "item_ids": cand.get("item_ids", []),
                    "novelty_score": novelty,
                    "relevance_score": relevance,
                    "short_summary": short_summary,
                    "markdown_path": str(md_path),
                    "links": links,
                    "cache_file": str(cache_path),
                }
            )

        final_report = {"generated_at": int(time.time()), "selected_file": str(selected_path), "items": report_items}
        (out_dir / "final_report.json").write_text(json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(report_items)} items to {out_dir / 'final_report.json'}", file=sys.stderr)
        return 0

    report_items: list[dict[str, Any]] = []

    for idx, cand in enumerate(candidates, start=1):
        if not isinstance(cand, dict):
            continue
        title = str(cand.get("title") or "").strip()
        slug = slugify(title)[:80]
        md_path = out_dir / f"blog_candidate_{slug}.md"

        links = normalize_links(cand)
        if args.max_links_per_candidate > 0:
            links = links[: args.max_links_per_candidate]

        # Gather per-source metadata arrays (if available)
        source_titles = _ensure_list_str(cand.get("source_title"))
        source_summaries = _ensure_list_str(cand.get("summary"))

        sources: list[dict[str, Any]] = []
        html_urls_to_fetch: list[str] = []
        for i, url in enumerate(links):
            cached = cache.get(url) or {}
            # Back-compat: older caches only have {fetched_at, text}
            cached_text = str(cached.get("text") or "")
            if not cached_text and not cached.get("download_path"):
                # PDFs are handled immediately; HTML is batched below via crawl4ai.
                if _looks_like_pdf(url):
                    print(f"[{idx}/{len(candidates)}] Fetching {url}", file=sys.stderr)
                    fetched = fetch_and_extract(url, out_dir=out_dir, timeout_seconds=args.timeout_seconds)
                    cache[url] = fetched
                    cached = fetched
                    cached_text = str(cached.get("text") or "")
                else:
                    html_urls_to_fetch.append(url)
            # Cap content before sending to model
            content = cached_text
            if args.max_content_chars > 0 and len(content) > args.max_content_chars:
                content = content[: args.max_content_chars] + "\n...[truncated]"
            sources.append(
                {
                    "url": url,
                    "source_title": source_titles[i] if i < len(source_titles) else "",
                    "summary": source_summaries[i] if i < len(source_summaries) else "",
                    "content": content,
                    "cached": cached,
                }
            )

        # Batch-fetch HTML URLs in one crawl4ai session to avoid asyncio loop shutdown issues.
        if html_urls_to_fetch:
            try:
                import asyncio

                # Ensure stable/deduped order
                seen_html: set[str] = set()
                html_urls_to_fetch = [u for u in html_urls_to_fetch if not (u in seen_html or seen_html.add(u))]

                for u in html_urls_to_fetch:
                    print(f"[{idx}/{len(candidates)}] Fetching {u}", file=sys.stderr)
                extracted_map = asyncio.run(crawl4ai_extract_many(html_urls_to_fetch, timeout_seconds=args.timeout_seconds))
                for u in html_urls_to_fetch:
                    ex = extracted_map.get(u, {})
                    entry = {
                        "fetched_at": int(time.time()),
                        "url": u,
                        "content_type": str(ex.get("content_type") or ""),
                        "text": str(ex.get("text") or ""),
                        **({"error": str(ex.get("error"))} if ex.get("error") else {}),
                    }
                    cache[u] = entry
                # Update sources list with newly cached text
                for s in sources:
                    u = s.get("url", "")
                    if u in extracted_map:
                        cached = cache.get(u) or {}
                        s["cached"] = cached
                        txt = str(cached.get("text") or "")
                        if args.max_content_chars > 0 and len(txt) > args.max_content_chars:
                            txt = txt[: args.max_content_chars] + "\n...[truncated]"
                        s["content"] = txt
                # Force GC now, while the loop is still alive, to reduce "event loop is closed" finalizer noise.
                gc.collect()
            except Exception as e:
                # If crawl4ai fails/missing, mark these entries with an error
                for u in html_urls_to_fetch:
                    cache[u] = {
                        "fetched_at": int(time.time()),
                        "url": u,
                        "content_type": "",
                        "text": "",
                        "error": f"crawl4ai_failed: {e}",
                    }

        # Persist cache after we potentially updated it for this candidate
        try:
            save_cache(cache_path, cache)
        except Exception:
            pass

        novelty = None
        relevance = None
        short_summary = ""
        blog_md = ""
        matched_keywords: list[Any] = []
        if not args.skip_llm:
            messages = build_llm_messages(
                candidate=cand,
                sources=[{"url": s["url"], "source_title": s["source_title"], "summary": s["summary"], "content": s["content"]} for s in sources],
                keywords=keywords,
                min_words=max(50, int(args.min_words)),
                max_words=max(int(args.min_words), int(args.max_words)),
            )
            result = call_llm(client=client, model=model, messages=messages) if client is not None else {}
            novelty = result.get("novelty_score")
            relevance = result.get("relevance_score")
            short_summary = str(result.get("short_summary") or "").strip()
            blog_md = str(result.get("blog_summary_markdown") or "").strip()
            mk = result.get("matched_keywords")
            matched_keywords = mk if isinstance(mk, list) else []
        else:
            # Crawl-only mode: keep summaries empty (or minimally informative)
            if sources and sources[0].get("source_title"):
                short_summary = str(sources[0].get("source_title") or "")[:120]
            else:
                short_summary = ""

        # Write markdown
        md_lines: list[str] = []
        md_lines.append(f"## {title or 'Blog candidate'}")
        md_lines.append("")
        md_lines.append("### Sources")
        for s in sources:
            u = s.get("url", "")
            st = s.get("source_title", "")
            if st:
                md_lines.append(f"- [{st}]({u})")
            else:
                md_lines.append(f"- {u}")
        md_lines.append("")
        md_lines.append("### Scores")
        md_lines.append(f"- **Novelty (0-10)**: {novelty if novelty is not None else 'N/A'}")
        md_lines.append(f"- **Relevance (0-10)**: {relevance if relevance is not None else 'N/A'}")
        if matched_keywords:
            md_lines.append(f"- **Matched keywords**: {', '.join(str(x) for x in matched_keywords)}")
        md_lines.append("")
        md_lines.append("### Blog summary (200–500 words)")
        md_lines.append("")
        md_lines.append(blog_md or ("(LLM skipped.)" if args.skip_llm else "(LLM did not return a blog summary.)"))
        md_lines.append("")
        # Append cached source texts at the end
        for i, s in enumerate(sources, start=1):
            cached = s.get("cached") or {}
            cached_text = str(cached.get("text") or "")
            cached_ct = str(cached.get("content_type") or "")
            cached_err = str(cached.get("error") or "")
            download_path = str(cached.get("download_path") or "")
            md_lines.append(f"## Sources ({i})")
            md_lines.append("")
            md_lines.append(f"- **URL**: {s.get('url','')}")
            if cached_ct:
                md_lines.append(f"- **Content-Type**: {cached_ct}")
            if download_path:
                md_lines.append(f"- **Downloaded file**: `{download_path}`")
            if cached_err:
                md_lines.append(f"- **Error**: {cached_err}")
            md_lines.append("")
            if not cached_text.strip():
                md_lines.append("(No cached text.)")
                md_lines.append("")
                continue
            append_text = cached_text
            if args.sources_append_chars > 0 and len(append_text) > args.sources_append_chars:
                append_text = append_text[: args.sources_append_chars] + "\n...[truncated]"
            md_lines.append("```text")
            md_lines.append(append_text)
            md_lines.append("```")
            md_lines.append("")
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

        report_items.append(
            {
                "title": title,
                "item_ids": cand.get("item_ids", []),
                "novelty_score": novelty,
                "relevance_score": relevance,
                "short_summary": short_summary,
                "markdown_path": str(md_path),
                "links": links,
                "cache_file": str(cache_path),
            }
        )

    final_report = {"generated_at": int(time.time()), "selected_file": str(selected_path), "items": report_items}
    (out_dir / "final_report.json").write_text(json.dumps(final_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(report_items)} items to {out_dir / 'final_report.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



