"""
Helpers shared by CLI targets that operate on plan + metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def load_metadata_by_image_id(metadata_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Read metadata.jsonl and return a mapping of image_id -> metadata record.
    """
    mapping: Dict[str, Dict[str, Any]] = {}

    if not metadata_path.exists():
        logger.warning("Metadata path does not exist: %s", metadata_path)
        return mapping

    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid metadata line: %s", exc)
                continue

            image_id = record.get("id")
            if image_id:
                mapping[image_id] = record

    return mapping

