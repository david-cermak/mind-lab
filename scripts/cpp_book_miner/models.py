from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChapterRef(BaseModel):
    chapter_id: str
    title: str
    start_line: int
    end_line: int
    source_path: str
    output_markdown_path: str


class ChapterIndex(BaseModel):
    chapters: List[ChapterRef] = Field(default_factory=list)


class ChapterSummary(BaseModel):
    chapter_id: str
    title: str
    topics: List[str]


class ChapterFullSummary(BaseModel):
    chapter_id: str
    title: str
    learning_objective: str
    summary: str


class Citation(BaseModel):
    citation: str = Field(description="The exact wording of the quotation/citation")
    line_number: int = Field(description="Line number in the chapter where this citation appears")
    context: str = Field(description="Summary of the context surrounding this citation (max 200 words)")


class ChapterCitations(BaseModel):
    chapter_id: str
    title: str
    citations: List[Citation] = Field(default_factory=list)


class CompilerTuning(BaseModel):
    description: Optional[str] = None
    compilers: Optional[List[str]] = None
    flags: Optional[List[str]] = None


class Candidate(BaseModel):
    id: str
    chapter_id: str
    title: str
    hook: str
    tags: List[str] = Field(default_factory=list)
    std: str
    code: str
    notes: str
    tuning: Optional[CompilerTuning] = None
    risks: Optional[str] = None


class ReviewBundle(BaseModel):
    chapter: ChapterRef
    candidates: List[Candidate]


def to_jsonl(items: List[BaseModel]) -> str:
    return "\n".join(item.model_dump_json() for item in items)


def from_jsonl(lines: str, model: Any) -> List[Any]:
    result: List[Any] = []
    for line in lines.splitlines():
        line = line.strip()
        if not line:
            continue
        result.append(model.model_validate_json(line))
    return result


