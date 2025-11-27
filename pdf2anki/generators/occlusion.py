"""
Generate occlusion masks and markup.
Wraps the functionality of the original create_occlusion_card.py script.
"""

import json
import logging
import math
import os
import re
import uuid
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

BBox = Tuple[int, int, int, int]


@dataclass
class OcrToken:
    index: int
    text: str
    bbox: BBox
    confidence: float = 0.0


@dataclass
class TokenGroup:
    tokens: List[OcrToken] = field(default_factory=list)
    bbox: BBox = (0, 0, 0, 0)

    def add(self, token: OcrToken) -> None:
        self.tokens.append(token)
        if len(self.tokens) == 1:
            self.bbox = token.bbox
        else:
            self.bbox = merge_bboxes(self.bbox, token.bbox)

    @property
    def label(self) -> str:
        return " ".join(tok.text for tok in self.tokens).strip()


def merge_bboxes(a: BBox, b: BBox) -> BBox:
    left_a, top_a, width_a, height_a = a
    left_b, top_b, width_b, height_b = b
    right_a, bottom_a = left_a + width_a, top_a + height_a
    right_b, bottom_b = left_b + width_b, top_b + height_b
    left = min(left_a, left_b)
    top = min(top_a, top_b)
    right = max(right_a, right_b)
    bottom = max(bottom_a, bottom_b)
    return (left, top, right - left, bottom - top)


def bbox_gap(a: BBox, b: BBox) -> float:
    left_a, top_a, width_a, height_a = a
    left_b, top_b, width_b, height_b = b
    right_a, bottom_a = left_a + width_a, top_a + height_a
    right_b, bottom_b = left_b + width_b, top_b + height_b

    if right_a < left_b:
        horizontal = left_b - right_a
    elif right_b < left_a:
        horizontal = left_a - right_b
    else:
        horizontal = 0

    if bottom_a < top_b:
        vertical = top_b - bottom_a
    elif bottom_b < top_a:
        vertical = top_a - bottom_b
    else:
        vertical = 0

    return math.hypot(horizontal, vertical)


def _aligned(a: BBox, b: BBox, tolerance: float = 0.6) -> bool:
    """Return True if boxes roughly share the same row or column."""
    left_a, top_a, width_a, height_a = a
    left_b, top_b, width_b, height_b = b
    center_a = (left_a + width_a / 2, top_a + height_a / 2)
    center_b = (left_b + width_b / 2, top_b + height_b / 2)
    avg_height = (height_a + height_b) / 2
    avg_width = (width_a + width_b) / 2
    same_row = abs(center_a[1] - center_b[1]) <= avg_height * tolerance
    same_col = abs(center_a[0] - center_b[0]) <= avg_width * tolerance
    return same_row or same_col


def group_tokens_spatially(tokens: Sequence[OcrToken], threshold: float) -> List[TokenGroup]:
    groups: List[TokenGroup] = []
    for token in sorted(tokens, key=lambda t: (t.bbox[1], t.bbox[0])):
        assigned = False
        for group in groups:
            gap = bbox_gap(group.bbox, token.bbox)
            if gap <= threshold or _aligned(group.bbox, token.bbox):
                group.add(token)
                assigned = True
                break
        if not assigned:
            new_group = TokenGroup()
            new_group.add(token)
            groups.append(new_group)
    return groups


def group_tokens_semantically(
    tokens: Sequence[OcrToken],
    semantic_groups: List[List[int]],
) -> List[TokenGroup]:
    """
    Group tokens based on semantic groups from vision LLM.
    Each semantic group is a list of token indices that should be grouped together.
    """
    groups: List[TokenGroup] = []
    used_indices = set()
    
    # Create groups from semantic groupings
    for group_indices in semantic_groups:
        if not group_indices:
            continue
        
        # Filter to valid token indices
        valid_indices = [idx for idx in group_indices if 0 <= idx < len(tokens)]
        if not valid_indices:
            continue
        
        # Create a new group with these tokens
        new_group = TokenGroup()
        for idx in valid_indices:
            if idx not in used_indices:
                new_group.add(tokens[idx])
                used_indices.add(idx)
        
        if new_group.tokens:
            groups.append(new_group)
    
    # Add any remaining tokens that weren't in semantic groups as individual groups
    for idx, token in enumerate(tokens):
        if idx not in used_indices:
            new_group = TokenGroup()
            new_group.add(token)
            groups.append(new_group)
    
    return groups


def merge_semantic_with_spatial(
    tokens: Sequence[OcrToken],
    semantic_groups: Optional[List[List[int]]],
    spatial_threshold: float,
) -> List[TokenGroup]:
    """
    Merge semantic groups with spatial grouping for better results.
    If semantic groups are provided, use them; otherwise fall back to spatial-only.
    Within semantic groups, apply spatial grouping to handle multi-word labels.
    """
    if semantic_groups is None or not semantic_groups:
        # Fallback to spatial-only grouping
        return group_tokens_spatially(tokens, spatial_threshold)
    
    groups: List[TokenGroup] = []
    used_indices = set()
    
    # Process each semantic group
    for group_indices in semantic_groups:
        if not group_indices:
            continue
        
        # Filter to valid token indices
        valid_indices = [idx for idx in group_indices if 0 <= idx < len(tokens)]
        if not valid_indices:
            continue
        
        # Get tokens for this semantic group
        semantic_tokens = [tokens[idx] for idx in valid_indices]
        
        # Within this semantic group, apply spatial grouping to handle
        # cases where multiple words form a single label
        spatial_subgroups = group_tokens_spatially(semantic_tokens, spatial_threshold)
        
        # Merge spatial subgroups into a single semantic group
        # (or keep them separate if they're far apart - but for now merge)
        merged_group = TokenGroup()
        for subgroup in spatial_subgroups:
            for token in subgroup.tokens:
                # Find original index
                orig_idx = next(i for i, t in enumerate(tokens) if t is token)
                if orig_idx not in used_indices:
                    merged_group.add(token)
                    used_indices.add(orig_idx)
        
        if merged_group.tokens:
            groups.append(merged_group)
    
    # Add any remaining tokens that weren't in semantic groups
    for idx, token in enumerate(tokens):
        if idx not in used_indices:
            new_group = TokenGroup()
            new_group.add(token)
            groups.append(new_group)
    
    return groups


def calculate_bounding_boxes(
    groups: Sequence[TokenGroup],
    padding: float,
    image_width: int,
    image_height: int,
) -> List[Dict[str, Any]]:
    rectangles: List[Dict[str, Any]] = []
    for idx, group in enumerate(groups, start=1):
        left, top, width, height = group.bbox
        padded_left = max(0, left - padding)
        padded_top = max(0, top - padding)
        padded_right = min(image_width, left + width + padding)
        padded_bottom = min(image_height, top + height + padding)
        bbox_pixels = [
            int(round(padded_left)),
            int(round(padded_top)),
            int(round(padded_right - padded_left)),
            int(round(padded_bottom - padded_top)),
        ]
        
        # Normalize
        bbox_norm = {
            "left": round(bbox_pixels[0] / image_width, 4),
            "top": round(bbox_pixels[1] / image_height, 4),
            "width": round(bbox_pixels[2] / image_width, 4),
            "height": round(bbox_pixels[3] / image_height, 4),
        }
        
        rectangles.append(
            {
                "index": idx,
                "label": group.label,
                "tokens": [token.text for token in group.tokens],
                "bbox_pixels": bbox_pixels,
                "bbox_normalized": bbox_norm
            }
        )
    return rectangles


def create_occlusion_markup(rectangles: Sequence[Dict[str, Any]], occlude_all: bool = True) -> str:
    """
    Create Anki image occlusion markup.
    
    Args:
        rectangles: List of rectangle dictionaries with normalized coordinates
        occlude_all: If True (oi=1), hide all labels and guess one. If False (oi=0), hide one and guess one.
    """
    oi_value = 1 if occlude_all else 0
    parts = []
    for idx, rect in enumerate(rectangles, start=1):
        norm = rect["bbox_normalized"]
        parts.append(
            f"{{{{c{idx}::image-occlusion:rect:left={norm['left']}:top={norm['top']}:width={norm['width']}:height={norm['height']}:oi={oi_value}}}}}"
        )
    return "<br>".join(parts)


def generate_occlusion_card_data(
    ocr_data: Dict[str, Any],
    image_width: int,
    image_height: int,
    min_confidence: float = 75.0,
    proximity_threshold: float = 90.0,
    padding: float = 8.0,
    semantic_groups: Optional[List[List[int]]] = None,
    occlude_all: bool = True,
) -> Dict[str, Any]:
    """
    Generate occlusion rectangles and markup from OCR data.
    Optionally uses semantic groups from vision LLM to refine grouping.
    
    Args:
        ocr_data: OCR data with tokens
        image_width: Image width in pixels
        image_height: Image height in pixels
        min_confidence: Minimum confidence threshold for tokens
        proximity_threshold: Distance threshold for spatial grouping
        padding: Padding around bounding boxes
        semantic_groups: Optional list of token index groups from vision LLM
            (indices refer to original OCR tokens before confidence filtering)
        occlude_all: If True (oi=1), hide all labels and guess one. If False (oi=0), hide one and guess one.
    
    Returns:
        Dictionary with rectangles, markup, token_count, and group_count
    """
    # Parse tokens and create mapping from original OCR index to filtered token index
    tokens = []
    ocr_to_filtered_map: Dict[int, int] = {}  # Maps original OCR index -> filtered token index
    
    for orig_idx, raw in enumerate(ocr_data.get("tokens", [])):
        text = str(raw.get("text", "")).strip()
        if not text: continue
        
        conf = float(raw.get("confidence", 0) or 0)
        if conf < min_confidence: continue
        
        bbox = raw.get("bbox")
        if not bbox or len(bbox) != 4: continue
        
        filtered_idx = len(tokens)
        tokens.append(OcrToken(
            index=filtered_idx,
            text=text,
            bbox=tuple([int(float(v)) for v in bbox]), # type: ignore
            confidence=conf
        ))
        ocr_to_filtered_map[orig_idx] = filtered_idx
    
    # Map semantic groups from OCR indices to filtered token indices
    mapped_semantic_groups: Optional[List[List[int]]] = None
    if semantic_groups:
        mapped_semantic_groups = []
        for group in semantic_groups:
            mapped_group = [ocr_to_filtered_map[idx] for idx in group if idx in ocr_to_filtered_map]
            if mapped_group:
                mapped_semantic_groups.append(mapped_group)
        if not mapped_semantic_groups:
            mapped_semantic_groups = None
    
    # Group tokens using semantic groups if available, otherwise spatial-only
    if mapped_semantic_groups:
        groups = merge_semantic_with_spatial(tokens, mapped_semantic_groups, proximity_threshold)
        grouping_method = "semantic+spatial"
    else:
        groups = group_tokens_spatially(tokens, proximity_threshold)
        grouping_method = "spatial-only"
    
    # Calculate boxes
    rects = calculate_bounding_boxes(groups, padding, image_width, image_height)
    markup = create_occlusion_markup(rects, occlude_all=occlude_all)
    
    return {
        "rectangles": rects,
        "markup": markup,
        "token_count": len(tokens),
        "group_count": len(groups),
        "grouping_method": grouping_method
    }
