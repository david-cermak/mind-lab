from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any

import httpx
import yaml
import sys

from .models import ChapterSummary, Candidate, to_jsonl
from .prompts import (
    summarization_system_prompt,
    summarization_user_prompt,
    candidate_system_prompt,
    candidate_user_prompt,
    PATTERN_LIBRARY,
    refine_system_prompt,
    refine_user_prompt,
)


class LLMClient:
    def __init__(self, model: str, timeout_seconds: int = 120, base_url: str | None = None) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        # Prefer explicit base_url from config; else env var; else OpenAI default
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.model = model
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> Dict[str, str]:
        if not self.api_key or not self.api_key.strip():
            # Fail early with a clear message instead of letting httpx raise a low-level error
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please export OPENAI_API_KEY (and optionally OPENAI_BASE_URL)."
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": messages,
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _find_matching_bracket(text: str, start_index: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    for i in range(start_index, len(text)):
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _extract_json_segment(text: str) -> str | None:
    """
    Try to extract a valid JSON substring from text:
      1) Code fence ```json ... ``` or ``` ... ```
      2) First balanced {...} or [...]
    """
    s = text.strip()
    # 1) Code fences
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[0].strip().startswith("```"):
            # Drop opening fence line (may include 'json')
            inner_lines = lines[1:]
            # Drop trailing fence if present
            if inner_lines and inner_lines[-1].strip().startswith("```"):
                inner_lines = inner_lines[:-1]
            candidate = "\n".join(inner_lines).strip()
            if candidate:
                return candidate
    # 2) Balanced bracket extraction
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = s.find(open_ch)
        if start != -1:
            end = _find_matching_bracket(s, start, open_ch, close_ch)
            if end != -1:
                return s[start : end + 1]
    return None


def _loads_relaxed_json(text: str) -> Any:
    """
    Parse JSON that may be wrapped in markdown fences or have surrounding prose.
    Raises ValueError if no JSON can be parsed.
    """
    try:
        return json.loads(text)
    except Exception:
        pass
    segment = _extract_json_segment(text)
    if segment is None:
        raise ValueError("No JSON segment found")
    return json.loads(segment)


def run_summarization_for_chapter(
    chapter_id: str,
    chapter_title: str,
    chapter_text: str,
    out_dir: Path,
    config: Dict[str, Any],
) -> ChapterSummary:
    ensure_dir(out_dir)
    llm = LLMClient(
        model=config["llm"]["model"],
        timeout_seconds=config["llm"]["request_timeout_seconds"],
        base_url=config.get("llm", {}).get("base_url"),
    )
    system = summarization_system_prompt()
    user = summarization_user_prompt(
        chapter_title=chapter_title,
        chapter_text=chapter_text,
        max_topics=config["limits"]["max_topics"],
    )
    if config.get("debug_llm", False):
        # Print the exact prompt payload we'd send
        debug_payload = {
            "url": f"{llm.base_url}/chat/completions",
            "model": llm.model,
            "temperature": config["llm"]["temperature_summarize"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        print(json.dumps({"debug_llm": True, "phase": "summarize", "payload": debug_payload}, indent=2))
        # Return an empty summary to keep pipeline shape without external calls
        return ChapterSummary(chapter_id=chapter_id, title=chapter_title, topics=[])
    raw = llm.chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=config["llm"]["temperature_summarize"],
    )
    if config.get("debug_llm_output", False):
        try:
            debug_out_path = out_dir / f"{chapter_id}.raw.txt"
            with open(debug_out_path, "w", encoding="utf-8") as f:
                f.write(raw)
            print(
                json.dumps(
                    {
                        "debug_llm_output": True,
                        "phase": "summarize",
                        "chapter_id": chapter_id,
                        "raw_path": str(debug_out_path),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
        except Exception as e:
            print(f"[debug_llm_output] Failed to persist raw summarize output: {e}", file=sys.stderr)
    try:
        parsed = _loads_relaxed_json(raw)
        topics = [t.strip() for t in parsed.get("topics", []) if isinstance(t, str) and t.strip()]
    except Exception:
        # Fallback: try to split lines if model failed schema
        topics = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
        topics = topics[: config["limits"]["max_topics"]]
    summary = ChapterSummary(chapter_id=chapter_id, title=chapter_title, topics=topics)
    out_path = out_dir / f"{chapter_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary.model_dump(), f, indent=2, ensure_ascii=False)
    return summary


def run_candidates_for_chapter(
    chapter_id: str,
    chapter_title: str,
    chapter_text: str,
    topics: List[str],
    out_dir: Path,
    config: Dict[str, Any],
) -> List[Candidate]:
    ensure_dir(out_dir)
    llm = LLMClient(
        model=config["llm"]["model"],
        timeout_seconds=config["llm"]["request_timeout_seconds"],
        base_url=config.get("llm", {}).get("base_url"),
    )
    system = candidate_system_prompt()
    user = candidate_user_prompt(
        chapter_title=chapter_title,
        chapter_text=chapter_text,
        topics=topics,
        pattern_library=PATTERN_LIBRARY,
        min_candidates=config["limits"]["min_candidates_per_chapter"],
        max_candidates=config["limits"]["max_candidates_per_chapter"],
        std=config["cpp_standard"],
        max_lines=config["limits"]["max_candidate_lines"],
    )
    if config.get("debug_llm", False):
        debug_payload = {
            "url": f"{llm.base_url}/chat/completions",
            "model": llm.model,
            "temperature": config["llm"]["temperature_generate"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        print(json.dumps({"debug_llm": True, "phase": "generate", "payload": debug_payload}, indent=2))
        return []
    raw = llm.chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=config["llm"]["temperature_generate"],
    )
    if config.get("debug_llm_output", False):
        try:
            debug_out_path = out_dir / f"{chapter_id}.raw.txt"
            with open(debug_out_path, "w", encoding="utf-8") as f:
                f.write(raw)
            print(
                json.dumps(
                    {
                        "debug_llm_output": True,
                        "phase": "generate",
                        "chapter_id": chapter_id,
                        "raw_path": str(debug_out_path),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                file=sys.stderr,
            )
        except Exception as e:
            print(f"[debug_llm_output] Failed to persist raw generate output: {e}", file=sys.stderr)
    try:
        data = _loads_relaxed_json(raw)
        if isinstance(data, dict):
            data = data.get("candidates", [])
        items = []
        for obj in data:
            if not isinstance(obj, dict):
                continue
            # Ensure required fields
            obj.setdefault("tags", [])
            obj.setdefault("std", config["cpp_standard"])
            obj["chapter_id"] = chapter_id
            items.append(Candidate.model_validate(obj))
    except Exception:
        items = []
    # Persist JSONL
    out_path = out_dir / f"{chapter_id}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(to_jsonl(items))
        f.write("\n" if items else "")
    return items


def run_refine_for_candidates(
    candidates: List[Candidate],
    out_dir: Path,
    config: Dict[str, Any],
) -> List[Candidate]:
    if not candidates:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = LLMClient(
        model=config["llm"]["model"],
        timeout_seconds=config["llm"]["request_timeout_seconds"],
        base_url=config.get("llm", {}).get("base_url"),
    )
    refined: List[Candidate] = []
    for c in candidates:
        system = refine_system_prompt()
        user = refine_user_prompt(
            title=c.title, hook=c.hook, code=c.code, std=c.std, max_lines=config["limits"]["max_candidate_lines"]
        )
        if config.get("debug_llm", False):
            debug_payload = {
                "url": f"{llm.base_url}/chat/completions",
                "model": llm.model,
                "temperature": 0.2,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
            print(json.dumps({"debug_llm": True, "phase": "refine", "payload": debug_payload}, indent=2))
            refined.append(c)
            continue
        raw = llm.chat(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], temperature=0.2)
        if config.get("debug_llm_output", False):
            try:
                out_file = out_dir / f"{c.id}.raw.txt"
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(raw)
                print(
                    json.dumps(
                        {
                            "debug_llm_output": True,
                            "phase": "refine",
                            "candidate_id": c.id,
                            "raw_path": str(out_file),
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                )
            except Exception as e:
                print(f"[debug_llm_output] Failed to persist raw refine output for {c.id}: {e}", file=sys.stderr)
        try:
            data = _loads_relaxed_json(raw)
            new_title = data.get("title", c.title)
            new_hook = data.get("hook", c.hook)
            new_code = data.get("code", c.code)
            refined.append(
                Candidate(
                    **c.model_dump(exclude={"title", "hook", "code"}),
                    title=new_title,
                    hook=new_hook,
                    code=new_code,
                )
            )
        except Exception:
            refined.append(c)
    # Persist JSONL
    out_path = out_dir / "refined.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(to_jsonl(refined))
        f.write("\n" if refined else "")
    return refined


