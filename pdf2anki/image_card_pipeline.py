"""
Helpers for generating image-only cards (image_visual plan items).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pdf2anki.analysis import vision
from pdf2anki.plan_utils import load_metadata_by_image_id

logger = logging.getLogger(__name__)


def generate_image_only_cards_from_plan(
    plan: List[Dict[str, Any]],
    output_dir: Path,
    metadata_path: Path,
    *,
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    language: str = "cs",
    max_cards: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run the image description workflow for every image_visual plan item.
    Returns a list of dicts with image metadata and the generated description.
    """

    metadata_map = load_metadata_by_image_id(metadata_path)
    output_dir = output_dir.resolve()

    generated: List[Dict[str, Any]] = []

    for item in plan:
        if item.get("type") != "image_visual":
            continue

        if max_cards is not None and len(generated) >= max_cards:
            logger.info("Reached requested max_image_cards (%s)", max_cards)
            break

        rel_image_path = item.get("image_path")
        if not rel_image_path:
            logger.debug("Skipping image_visual item missing image_path: %s", item)
            continue

        image_path = Path(rel_image_path)
        if not image_path.is_absolute():
            image_path = output_dir / image_path

        if not image_path.exists():
            logger.warning("Image not found for item %s: %s", item.get("image_id"), image_path)
            continue

        image_id = item.get("image_id")
        metadata = metadata_map.get(image_id) if image_id else None
        page_text = metadata.get("page_text", "") if metadata else ""

        logger.info("Describing image %s (%s)", image_id, image_path)

        vision_result = vision.describe_image_only(
            image_path,
            page_text=page_text,
            language=language,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

        if "error" in vision_result:
            logger.warning("Vision description failed for %s: %s", image_id, vision_result["error"])
            continue

        description = (vision_result.get("description") or "").strip()
        if not description:
            logger.warning("Vision description empty for %s", image_id)
            continue

        generated.append(
            {
                "image_id": image_id,
                "image_path": str(_relativize_path(image_path, output_dir)),
                "description": description,
                "page_number": item.get("page_number") or (metadata.get("page_number") if metadata else None),
                "language": language,
            }
        )

    return generated


def _relativize_path(target: Path, base: Path) -> Path:
    try:
        return target.relative_to(base)
    except ValueError:
        return target

