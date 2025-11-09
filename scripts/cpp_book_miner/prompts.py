from __future__ import annotations

from typing import List

PATTERN_LIBRARY: List[str] = [
    "Strict aliasing / TBAA surprises",
    "Signed overflow is undefined behavior",
    "Dangling lifetime: string_view, ranges views",
    "SSO illusions: addresses and string capacity",
    "ADL and hidden friends",
    "CTAD surprises and deduction guides",
    "Two-phase lookup / template name lookup quirks",
    "Overload resolution ambiguity and decay",
    "Virtual dispatch during constructors / destructors",
    "Static initialization order fiasco",
    "Copy elision / NRVO toggles by trivial changes",
    "Volatile myths (not for threading)",
    "Atomics and memory_order gotchas",
    "constexpr vs consteval distinctions",
    "Exception specifications influence overload sets",
    "ODR / inline variables pitfalls",
    "Bitfield packing and alignment surprises",
    "Struct layout and padding visibility",
    "Aggregate initialization traps",
    "Range-for value vs reference decay",
    "Dangling iterators after container ops",
    "Move-from object surprising states",
]


def summarization_system_prompt() -> str:
    return (
        "You are a C++ language expert. Extract concise, high-signal topics from a chapter. "
        "Prefer pitfalls, surprising behaviors, boundary conditions, subtle rules, and mechanisms "
        "that lend themselves to minimal, single-file examples."
    )


def summarization_user_prompt(chapter_title: str, chapter_text: str, max_topics: int) -> str:
    return (
        f"Chapter: {chapter_title}\n\n"
        "Task: List up to {max_topics} concise bullet topics from this chapter that are likely to yield "
        "attention-grabbing, minimal Godbolt examples. Avoid generic advice.\n\n"
        "Return JSON only: {\"topics\": [\"...\"]}\n\n"
        "Chapter content:\n"
        f"{chapter_text}\n"
    ).replace("{max_topics}", str(max_topics))


def candidate_system_prompt() -> str:
    return (
        "You are a C++ language expert optimizing for viral, minimal Godbolt examples that reveal "
        "surprising truths. Prefer short, standalone snippets. Avoid undefined behavior that cannot "
        "be observed predictably."
    )


def candidate_user_prompt(
    chapter_title: str,
    chapter_text: str,
    topics: List[str],
    pattern_library: List[str],
    min_candidates: int,
    max_candidates: int,
    std: str,
    max_lines: int,
) -> str:
    topics_rendered = "\n".join(f"- {t}" for t in topics[:30])
    patterns_rendered = "\n".join(f"- {p}" for p in pattern_library)
    return (
        f"Chapter: {chapter_title}\n\n"
        f"Constraints:\n"
        f"- Produce {min_candidates}..{max_candidates} candidates\n"
        f"- Single-file, no external deps, minimal includes\n"
        f"- Compile for {std}; keep code to <= {max_lines} lines\n"
        f"- Prefer contrasts visible under -O0 vs -O2 or different compilers\n"
        f"- Avoid fragile UB unless clearly labeled and reliably observable\n\n"
        f"Attention patterns (hints):\n{patterns_rendered}\n\n"
        f"Chapter topics to target:\n{topics_rendered}\n\n"
        "Output STRICT JSON array, each object with keys:\n"
        "{\n"
        "  \"id\": string (short slug),\n"
        "  \"title\": string,\n"
        "  \"hook\": string,\n"
        "  \"tags\": [string],\n"
        "  \"std\": string,\n"
        "  \"code\": string,\n"
        "  \"notes\": string,\n"
        "  \"tuning\": {\"description\": string, \"compilers\": [string], \"flags\": [string]},\n"
        "  \"risks\": string\n"
        "}\n\n"
        "Chapter content:\n"
        f"{chapter_text}\n"
    )

def refine_system_prompt() -> str:
    return (
        "You are a C++ code editor that tightens minimal examples without changing the core idea. "
        "Reduce lines, improve the hook, ensure a single clear point. Preserve compile-ability."
    )


def refine_user_prompt(
    title: str,
    hook: str,
    code: str,
    std: str,
    max_lines: int,
) -> str:
    return (
        f"Goal: rewrite to a crisper minimal example (<= {max_lines} lines) for {std}.\n"
        "Return strict JSON with keys: {\"title\": str, \"hook\": str, \"code\": str}.\n\n"
        f"Current title: {title}\n"
        f"Current hook: {hook}\n"
        "Current code:\n"
        f"{code}\n"
    )


