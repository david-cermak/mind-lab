#!/usr/bin/env python3
"""
Generate a work report from recent git commits.

Features:
- Collect commits since a given date (and optional author) from a git repo.
- Deduplicate likely backports / repeated commits.
- Optionally call an OpenAI-compatible chat API to:
  - Summarize each commit into 1–5 sentences (`details.md`).
  - Generate a thematic, high-level `report.md` from `details.md`.

The workflow is step-based so you can run collection and inspection
even without any LLM API configured:

- --step collect  : only collect & deduplicate commits, no LLM calls.
- --step details  : generate details.md from stored commits (or on the fly).
- --step report   : generate report.md from existing details.md.
- --step all      : run collect -> details -> report.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class Commit:
    hash: str
    date: str  # ISO or git short date
    subject: str
    body: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate work report from git commits using an OpenAI-compatible API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--since",
        required=True,
        help='Lower bound for commit dates, passed to "git log --since". '
        "Examples: '2024-12-01', '2 weeks ago'",
    )
    parser.add_argument(
        "--author",
        help="Optional author filter passed to git log --author (e.g. email address).",
    )
    parser.add_argument(
        "--repo-path",
        default=".",
        help="Path to the git repository (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where details.md, report.md, and intermediate files are written.",
    )
    parser.add_argument(
        "--step",
        choices=("collect", "details", "report", "all"),
        default="all",
        help="Which step to run: collect commits, generate details, generate report, or all (default: all).",
    )
    parser.add_argument(
        "--dump-llm-input",
        action="store_true",
        help="When used with --step collect, also write the exact markdown prompt blocks "
        "that would be sent to the LLM for commit details.",
    )

    # API-related arguments (mirroring scripts/test_openai_api.py)
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
        help="Model name (e.g., gpt-4, gpt-4.1-mini). "
        "Can also use OPENAI_MODEL env var or .env file.",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        metavar="KEY",
        help="API key. Can also use OPENAI_API_KEY env var or .env file.",
    )

    return parser.parse_args(argv)


def load_env_file(project_root: Optional[Path] = None) -> None:
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


def get_config(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple[str, str, str]:
    """
    Get configuration with priority: CLI args > env vars > .env file.

    Returns:
        Tuple of (base_url, model, api_key)
    """
    # Load .env file first (lowest priority)
    load_env_file()

    final_base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL")
    final_model = model or os.getenv("OPENAI_MODEL") or os.getenv("MODEL")
    final_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

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


def run_git_log(
    repo_path: Path, since: str, author: Optional[str] = None
) -> List[Commit]:
    """
    Call git log and parse commits.

    We use a machine-readable format: each commit on a single line as JSON.
    """
    pretty_format = r'{"hash":"%H","date":"%ad","subject":"%s","body":"%b"}'
    cmd = [
        "git",
        "-C",
        str(repo_path),
        "log",
        "--all",
        f"--since={since}",
        "--date=short",
        f"--pretty=format:{pretty_format}",
    ]
    if author:
        cmd.append(f"--author={author}")

    try:
        output = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, text=True, encoding="utf-8"
        )
    except subprocess.CalledProcessError as e:
        print("Error running git log:", file=sys.stderr)
        print(e.output, file=sys.stderr)
        raise SystemExit(1)

    commits: List[Commit] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            # Best-effort skip if parsing fails
            continue
        commits.append(
            Commit(
                hash=data.get("hash", ""),
                date=data.get("date", ""),
                subject=data.get("subject", ""),
                body=data.get("body", ""),
            )
        )
    return commits


def normalize_subject(subject: str) -> str:
    """Normalize commit subject for deduplication."""
    s = subject.strip().lower()
    # Drop trailing punctuation like '.', '!' etc.
    s = re.sub(r"[.!?]+$", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def deduplicate_commits(commits: Iterable[Commit]) -> List[Commit]:
    """
    Deduplicate commits by normalized subject and body.

    Strategy:
    - Group by (normalized_subject, normalized_body).
    - Keep the most recent commit (by original list order).
    """
    groups: dict[tuple[str, str], Commit] = {}
    for commit in commits:
        key = (normalize_subject(commit.subject), commit.body.strip())
        # later commits in the list are usually more recent; keep last
        groups[key] = commit
    # Preserve original order of appearance of the kept commits
    kept_hashes = {c.hash for c in groups.values()}
    ordered = [c for c in commits if c.hash in kept_hashes]
    # De-duplicate list order while preserving it
    seen: set[str] = set()
    result: List[Commit] = []
    for c in ordered:
        if c.hash in seen:
            continue
        seen.add(c.hash)
        result.append(c)
    return result


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_markdown(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def render_commit_markdown_list(commits: Iterable[Commit]) -> str:
    lines = []
    lines.append("# Deduplicated commits\n")
    for c in commits:
        lines.append(f"## {c.subject}\n")
        lines.append(f"- Hash: `{c.hash}`")
        lines.append(f"- Date: {c.date}")
        if c.body.strip():
            lines.append("")
            lines.append("```")
            lines.append(c.body.strip())
            lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_details_prompt_for_commit(commit: Commit) -> str:
    """
    Build a markdown prompt for summarizing a single commit.
    """
    lines: list[str] = []
    lines.append("# Commit to summarize")
    lines.append("")
    lines.append(f"- Hash: `{commit.hash}`")
    lines.append(f"- Date: {commit.date}")
    lines.append(f"- Subject: {commit.subject}")
    lines.append("")
    lines.append("## Full commit message")
    lines.append("")
    if commit.body.strip():
        lines.append("```")
        lines.append(commit.body.strip())
        lines.append("```")
    else:
        lines.append("_No additional body_")
    lines.append("")
    lines.append("## Task")
    lines.append("")
    lines.append(
        "Summarize this commit in **1–5 concise sentences** focusing on:"
    )
    lines.append("- What changed technically.")
    lines.append("- Why the change was made (bug fix, feature, refactor, docs, CI, etc.).")
    lines.append("- Any user-facing or developer-impacting consequences.")
    lines.append("")
    lines.append(
        "Respond with just the summary sentences in markdown (no extra headings)."
    )
    return "\n".join(lines)


def build_report_prompt_from_details(details_md: str, since: str) -> str:
    lines: list[str] = []
    lines.append("# Work report generation task")
    lines.append("")
    lines.append(f"The following markdown describes commit-level details since **{since}**.")
    lines.append("Use it to generate a high-level work report.")
    lines.append("")
    lines.append("## Commit details")
    lines.append("")
    lines.append("```markdown")
    lines.append(details_md.strip())
    lines.append("```")
    lines.append("")
    lines.append("## Task")
    lines.append("")
    lines.append(
        "Write a concise work report in markdown that groups the work into thematic sections."
    )
    lines.append("Guidelines:")
    lines.append("- Use clear section headings (e.g., Networking, Bug fixes, Documentation, CI, etc.).")
    lines.append("- Highlight key changes, patterns, and impact, not every commit individually.")
    lines.append("- Keep it suitable for a weekly/monthly status report.")
    lines.append("- Do not copy all commit summaries verbatim; synthesize them.")
    return "\n".join(lines)


def init_openai_client(args: argparse.Namespace) -> OpenAI:
    if OpenAI is None:
        print(
            "Error: openai package is required for steps that use the LLM. "
            "Install with: pip install openai",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        base_url, model, api_key = get_config(
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    # Attach model name onto args for later use
    args._resolved_model = model  # type: ignore[attr-defined]
    return client


def step_collect(args: argparse.Namespace) -> list[Commit]:
    repo_path = Path(args.repo_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_output_dir(output_dir)

    print(f"[collect] Using repo: {repo_path}", file=sys.stderr)
    print(f"[collect] Since: {args.since}", file=sys.stderr)
    if args.author:
        print(f"[collect] Author filter: {args.author}", file=sys.stderr)

    commits = run_git_log(repo_path=repo_path, since=args.since, author=args.author)
    if not commits:
        print("[collect] No commits found for the given range/filters.", file=sys.stderr)
        return []

    print(f"[collect] Found {len(commits)} commits.", file=sys.stderr)
    deduped = deduplicate_commits(commits)
    print(f"[collect] {len(deduped)} commits after deduplication.", file=sys.stderr)

    raw_path = output_dir / "commits_raw.json"
    dedup_path = output_dir / "commits_deduped.json"
    md_path = output_dir / "commits_deduped.md"

    write_json(raw_path, [c.to_dict() for c in commits])
    write_json(dedup_path, [c.to_dict() for c in deduped])
    write_markdown(md_path, render_commit_markdown_list(deduped))

    print(f"[collect] Wrote raw commits to {raw_path}", file=sys.stderr)
    print(f"[collect] Wrote deduplicated commits to {dedup_path}", file=sys.stderr)
    print(f"[collect] Wrote human-readable list to {md_path}", file=sys.stderr)

    if args.dump_llm_input:
        llm_input_path = output_dir / "llm_details_input.md"
        blocks: list[str] = []
        for c in deduped:
            blocks.append("---")
            blocks.append(build_details_prompt_for_commit(c))
            blocks.append("")
        write_markdown(llm_input_path, "\n".join(blocks).rstrip() + "\n")
        print(
            f"[collect] Wrote would-be LLM detail prompts to {llm_input_path}",
            file=sys.stderr,
        )

    return deduped


def load_deduped_commits_from_json(output_dir: Path) -> list[Commit]:
    dedup_path = output_dir / "commits_deduped.json"
    if not dedup_path.exists():
        print(
            f"[details] Expected {dedup_path} to exist. "
            "Run with --step collect first or allow recomputation.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    with dedup_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    commits: list[Commit] = []
    for item in data:
        commits.append(
            Commit(
                hash=item.get("hash", ""),
                date=item.get("date", ""),
                subject=item.get("subject", ""),
                body=item.get("body", ""),
            )
        )
    return commits


def step_details(
    args: argparse.Namespace,
    client: OpenAI,
    commits: Optional[list[Commit]] = None,
) -> None:
    output_dir = Path(args.output_dir).resolve()
    ensure_output_dir(output_dir)

    if commits is None:
        print(
            "[details] Loading commits from commits_deduped.json in output-dir.",
            file=sys.stderr,
        )
        commits = load_deduped_commits_from_json(output_dir)

    if not commits:
        print("[details] No commits available for details generation.", file=sys.stderr)
        return

    details_lines: list[str] = []
    details_lines.append(f"# Commit details since {args.since}")
    details_lines.append("")

    model = getattr(args, "_resolved_model", args.model)
    print(
        f"[details] Generating summaries for {len(commits)} commits using model: {model}",
        file=sys.stderr,
    )

    for idx, commit in enumerate(commits, start=1):
        print(
            f"[details] ({idx}/{len(commits)}) Summarizing commit {commit.hash[:10]}",
            file=sys.stderr,
        )
        prompt = build_details_prompt_for_commit(commit)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as e:
            print(f"[details] Error calling API for commit {commit.hash}: {e}", file=sys.stderr)
            continue

        content = ""
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content or ""

        # Write section to details.md
        details_lines.append(f"## {commit.subject}")
        details_lines.append("")
        details_lines.append(f"- Hash: `{commit.hash}`")
        details_lines.append(f"- Date: {commit.date}")
        details_lines.append("")
        if content.strip():
            details_lines.append(content.strip())
        else:
            details_lines.append("_No summary generated (API error)._")
        details_lines.append("")

    details_md_path = output_dir / "details.md"
    write_markdown(details_md_path, "\n".join(details_lines).rstrip() + "\n")
    print(f"[details] Wrote details to {details_md_path}", file=sys.stderr)


def step_report(args: argparse.Namespace, client: OpenAI) -> None:
    output_dir = Path(args.output_dir).resolve()
    details_md_path = output_dir / "details.md"
    if not details_md_path.exists():
        print(
            f"[report] Expected {details_md_path} to exist. "
            "Run with --step details or --step all first.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    with details_md_path.open("r", encoding="utf-8") as f:
        details_md = f.read()

    prompt = build_report_prompt_from_details(details_md=details_md, since=args.since)
    model = getattr(args, "_resolved_model", args.model)
    print(f"[report] Generating report using model: {model}", file=sys.stderr)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as e:
        print(f"[report] Error calling API: {e}", file=sys.stderr)
        raise SystemExit(1)

    content = ""
    if response.choices and len(response.choices) > 0:
        content = response.choices[0].message.content or ""

    report_md_path = output_dir / "report.md"
    if content.strip():
        write_markdown(report_md_path, content.strip() + "\n")
    else:
        write_markdown(
            report_md_path,
            "# Report\n\n_No report generated (empty response from API)._",
        )
    print(f"[report] Wrote report to {report_md_path}", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # For steps that *only* collect, we must not require OpenAI.
    needs_llm = args.step in ("details", "report", "all")
    client: Optional[OpenAI] = None

    if needs_llm:
        client = init_openai_client(args)

    # Step dispatch
    commits_for_details: Optional[list[Commit]] = None

    if args.step in ("collect", "all"):
        commits_for_details = step_collect(args)

    if args.step in ("details", "all"):
        if client is None:
            print(
                "Internal error: LLM client not initialized for details step.",
                file=sys.stderr,
            )
            return 1
        # If all: reuse commits from collection if available; otherwise load from JSON.
        step_details(args, client, commits=commits_for_details)

    if args.step in ("report", "all"):
        if client is None:
            print(
                "Internal error: LLM client not initialized for report step.",
                file=sys.stderr,
            )
            return 1
        step_report(args, client)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


