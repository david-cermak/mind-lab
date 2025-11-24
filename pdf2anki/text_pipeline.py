"""
Helpers for running the text-card portion of the pipeline independently.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pdf2anki.analysis import strategy
from pdf2anki.generators import text as text_generator

logger = logging.getLogger(__name__)


def ensure_plan(
    metadata_path: Path,
    ocr_dir: Path,
    plan_path: Path,
    force_rebuild: bool = False,
    tokens_per_card: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Make sure a strategy plan exists and return its contents.
    """
    if force_rebuild or not plan_path.exists():
        logger.info(
            "Creating strategy plan (metadata=%s, ocr_dir=%s, plan=%s)",
            metadata_path,
            ocr_dir,
            plan_path,
        )
        strategy.create_strategy_plan(metadata_path, ocr_dir, plan_path, tokens_per_card=tokens_per_card)

    with plan_path.open("r", encoding="utf-8") as handle:
        plan_data = json.load(handle)

    return plan_data.get("plan", [])


def generate_text_cards_from_plan(
    plan: List[Dict[str, Any]],
    output_dir: Path,
    *,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    max_cards: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run the text card generator for every text_qa plan item and return a flat list.
    """
    generated: List[Dict[str, Any]] = []
    total_cards = 0

    for item in plan:
        if item.get("type") != "text_qa":
            continue

        source_text_path = item.get("source_text_path")
        if not source_text_path:
            logger.debug(
                "Skipping text item on page %s: missing source_text_path",
                item.get("page_number"),
            )
            continue

        text_path = Path(source_text_path)
        if not text_path.is_absolute():
            text_path = output_dir / text_path

        if not text_path.exists():
            logger.warning("Text file not found for plan item: %s", text_path)
            continue

        try:
            page_text = text_path.read_text(encoding="utf-8").strip()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to read %s: %s", text_path, exc)
            continue

        if not page_text:
            logger.debug("Skipping %s because page text is empty", text_path)
            continue

        estimated_cards = int(item.get("estimated_cards", 0) or 0)
        if estimated_cards <= 0:
            continue

        if max_cards is not None:
            remaining = max_cards - total_cards
            if remaining <= 0:
                logger.info("Global text card limit (%s) reached", max_cards)
                break
            estimated_cards = min(estimated_cards, remaining)

        logger.info(
            "Generating %s text cards for page %s (%s)",
            estimated_cards,
            item.get("page_number"),
            text_path,
        )

        cards = text_generator.generate_text_cards(
            page_text,
            num_cards=estimated_cards,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

        for card in cards:
            front = (card.get("front") or "").strip()
            back = (card.get("back") or "").strip()
            if not front or not back:
                continue

            generated.append(
                {
                    "front": front,
                    "back": back,
                    "page_number": item.get("page_number"),
                    "text_source": _relativize_path(text_path, output_dir),
                    "context_tokens": item.get("context_tokens"),
                }
            )
            total_cards += 1

        if max_cards is not None and total_cards >= max_cards:
            logger.info("Reached requested max_text_cards (%s)", max_cards)
            break

    return generated


def _relativize_path(target: Path, base: Path) -> str:
    try:
        return str(target.relative_to(base))
    except ValueError:
        return str(target)

