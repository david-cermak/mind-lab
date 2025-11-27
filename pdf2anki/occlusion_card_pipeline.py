"""
Helpers for generating occlusion cards (image_occlusion plan items).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from pdf2anki.analysis import vision
from pdf2anki.generators import occlusion
from pdf2anki.plan_utils import load_metadata_by_image_id

logger = logging.getLogger(__name__)


def generate_occlusion_cards_from_plan(
    plan: List[Dict[str, Any]],
    output_dir: Path,
    metadata_path: Path,
    ocr_dir: Path,
    *,
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    min_confidence: float = 75.0,
    max_cards: Optional[int] = None,
    disable_vision: bool = False,
    occlude_all: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run the occlusion workflow for every image_occlusion plan item.
    Returns a list of dicts describing the cards that should be added to the deck.
    """

    metadata_map = load_metadata_by_image_id(metadata_path)
    output_dir = output_dir.resolve()
    ocr_dir = ocr_dir.resolve()

    generated: List[Dict[str, Any]] = []

    for item in plan:
        if item.get("type") != "image_occlusion":
            continue

        if max_cards is not None and len(generated) >= max_cards:
            logger.info("Reached requested max_occlusion_cards (%s)", max_cards)
            break

        image_id = item.get("image_id")
        rel_image_path = item.get("image_path")
        if not image_id or not rel_image_path:
            logger.debug("Skipping occlusion item missing required fields: %s", item)
            continue

        image_path = Path(rel_image_path)
        if not image_path.is_absolute():
            image_path = output_dir / image_path
        if not image_path.exists():
            logger.warning("Image file not found for %s: %s", image_id, image_path)
            continue

        ocr_path = _resolve_ocr_path(item.get("ocr_source"), image_id, ocr_dir, output_dir)
        if not ocr_path or not ocr_path.exists():
            logger.warning("OCR file not found for %s (source=%s)", image_id, item.get("ocr_source"))
            continue

        try:
            with ocr_path.open("r", encoding="utf-8") as handle:
                ocr_data = json.load(handle)
        except Exception as exc:
            logger.warning("Failed to read OCR data for %s: %s", image_id, exc)
            continue

        tokens = ocr_data.get("tokens", [])
        if not tokens:
            logger.debug("No OCR tokens for %s, skipping", image_id)
            continue

        metadata = metadata_map.get(image_id)
        page_text = metadata.get("page_text", "") if metadata else ""
        page_number = item.get("page_number") or (metadata.get("page_number") if metadata else None)

        with Image.open(image_path) as image:
            width, height = image.size

        semantic_groups = None
        back_extra = None

        if not disable_vision:
            try:
                logger.info("Calling vision LLM for occlusion image %s", image_id)
                vision_result = vision.analyze_image_context(
                    image_path,
                    page_text,
                    tokens,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                )
                if "error" not in vision_result:
                    semantic_groups = vision_result.get("groups")
                    back_extra = vision_result.get("description")
                    if semantic_groups:
                        logger.info("Vision LLM produced %s semantic groups for %s", len(semantic_groups), image_id)
                else:
                    logger.warning("Vision LLM error for %s: %s", image_id, vision_result.get("error"))
            except Exception as exc:
                logger.warning("Vision call failed for %s: %s", image_id, exc)

        occlusion_data = occlusion.generate_occlusion_card_data(
            ocr_data,
            width,
            height,
            min_confidence=min_confidence,
            semantic_groups=semantic_groups,
            occlude_all=occlude_all,
        )

        rectangles = occlusion_data.get("rectangles") or []
        if not rectangles:
            logger.warning("No rectangles generated for %s, skipping card", image_id)
            continue

        generated.append(
            {
                "image_id": image_id,
                "image_path": str(_relativize_path(image_path, output_dir)),
                "markup": occlusion_data["markup"],
                "header": f"Diagram (Page {page_number})" if page_number else "Diagram",
                "back_extra": back_extra,
                "grouping_method": occlusion_data.get("grouping_method"),
                "group_count": occlusion_data.get("group_count"),
                "semantic_groups_used": bool(semantic_groups),
            }
        )

    return generated


def _resolve_ocr_path(
    ocr_source: Optional[str],
    image_id: str,
    ocr_dir: Path,
    output_dir: Path,
) -> Optional[Path]:
    if ocr_source:
        candidate = Path(ocr_source)
        resolved = _normalize_candidate(candidate, ocr_dir, output_dir)
        if resolved and resolved.exists():
            return resolved

    default_candidate = ocr_dir / f"{image_id}.json"
    return default_candidate if default_candidate.exists() else None


def _normalize_candidate(candidate: Path, ocr_dir: Path, output_dir: Path) -> Optional[Path]:
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        fallback = ocr_dir / candidate.name
        return fallback if fallback.exists() else None

    text = str(candidate)
    if text.startswith("output/"):
        candidate = output_dir / text[7:]
    elif text.startswith("ocr/"):
        candidate = output_dir / text
    else:
        candidate = output_dir / candidate

    return candidate


def _relativize_path(target: Path, base: Path) -> Path:
    try:
        return target.relative_to(base)
    except ValueError:
        return target

