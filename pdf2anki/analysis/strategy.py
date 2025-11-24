"""
Analyze extracted text and OCR metadata to plan card generation.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ~= 4 chars)."""
    return len(text) // 4


def create_strategy_plan(
    metadata_path: Path,
    ocr_dir: Path,
    output_path: Path,
    model: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    tokens_per_card: int = None,
) -> None:
    """Analyze content and generate a card creation strategy."""
    
    # Default tokens_per_card from env or use 300
    if tokens_per_card is None:
        tokens_per_card = int(os.getenv("PDF2ANKI_TOKENS_PER_CARD", "300"))
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load all metadata
    pages = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pages.append(json.loads(line))
    
    # Group by page number to reconstruct document flow
    pages_by_num = {}
    for p in pages:
        p_num = p.get("page_number")
        if p_num:
            if p_num not in pages_by_num:
                pages_by_num[p_num] = {"text": "", "text_path": None, "images": []}
            
            # Add text if we haven't already (metadata is per-image, so text repeats)
            if not pages_by_num[p_num]["text"] and p.get("page_text"):
                pages_by_num[p_num]["text"] = p["page_text"]
            
            # Store text_path from any entry (prefer from entries with images, but any will do)
            if not pages_by_num[p_num]["text_path"] and p.get("page_text_path"):
                pages_by_num[p_num]["text_path"] = p["page_text_path"]
            
            pages_by_num[p_num]["images"].append(p)

    sorted_page_nums = sorted(pages_by_num.keys())
    
    # Create a high-level plan
    plan_entries = []
    
    # Process text in chunks (e.g., per page or multi-page)
    # For now, simple page-by-page analysis
    
    for p_num in sorted_page_nums:
        page_data = pages_by_num[p_num]
        text = page_data["text"]
        images = page_data["images"]
        
        # Strategy for Text
        # Heuristic: 1 card per tokens_per_card tokens (default 300, configurable)
        token_count = estimate_tokens(text)
        
        text_strategy = {
            "type": "text_qa",
            "page_number": p_num,
            "source_text_path": page_data.get("text_path") or (images[0].get("page_text_path") if images else None),
            "estimated_cards": max(1, math.ceil(token_count / tokens_per_card)) if token_count > 50 else 0,
            "context_tokens": token_count
        }
        if text_strategy["estimated_cards"] > 0:
            plan_entries.append(text_strategy)
            
        # Strategy for Images (skip text-only entries)
        for img in images:
            # Skip text-only entries (pages without images)
            if img.get("image_path") is None:
                continue
                
            img_id = img.get("id")
            ocr_path = ocr_dir / f"{img_id}.json"
            
            has_ocr = False
            ocr_tokens = 0
            if ocr_path.exists():
                try:
                    with open(ocr_path, "r", encoding="utf-8") as f:
                        ocr_data = json.load(f)
                        ocr_tokens = len(ocr_data.get("tokens", []))
                        has_ocr = True
                except Exception:
                    pass
            
            # Heuristic: If image has OCR tokens, it's likely a diagram suitable for occlusion
            # If few tokens, maybe just a visual aid (or handle manually)
            
            if has_ocr and ocr_tokens > 3:
                plan_entries.append({
                    "type": "image_occlusion",
                    "page_number": p_num,
                    "image_id": img_id,
                    "image_path": img.get("image_path"),
                    "ocr_source": str(ocr_path),
                    "ocr_token_count": ocr_tokens
                })
            else:
                plan_entries.append({
                    "type": "image_visual",
                    "page_number": p_num,
                    "image_id": img_id,
                    "image_path": img.get("image_path"),
                    "action": "describe_only" # Use for context in text cards, or skip
                })

    # Save Plan
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"plan": plan_entries}, f, indent=2, ensure_ascii=False)
        
    print(f"Generated strategy plan with {len(plan_entries)} items at {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze content and plan card generation.")
    parser.add_argument("--metadata", required=True, help="Path to assets_metadata.jsonl")
    parser.add_argument("--ocr-dir", required=True, help="Directory containing OCR JSON files")
    parser.add_argument("--output", required=True, help="Path to save plan.json")
    parser.add_argument(
        "--tokens-per-card",
        type=int,
        default=int(os.getenv("PDF2ANKI_TOKENS_PER_CARD", "300")),
        help="Target tokens per card (lower = denser cards, default: 300)",
    )
    
    args = parser.parse_args()
    create_strategy_plan(Path(args.metadata), Path(args.ocr_dir), Path(args.output), tokens_per_card=args.tokens_per_card)

