"""
Stage 0: Prepare pages with image descriptions for audiobook generation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from pdf2anki.analysis import vision

logger = logging.getLogger(__name__)

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token."""
    return len(text) // 4


def load_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
    """Load metadata.jsonl into a list."""
    records = []
    if not metadata_path.exists():
        logger.error(f"Metadata not found: {metadata_path}")
        return records
    
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid metadata line: {e}")
    
    return records


def group_pages_by_number(records: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Group metadata records by page number."""
    pages: Dict[int, Dict[str, Any]] = {}
    
    for record in records:
        page_num = record.get("page_number")
        if page_num is None:
            continue
        
        if page_num not in pages:
            pages[page_num] = {
                "page_number": page_num,
                "text": record.get("page_text", ""),
                "images": [],
            }
        
        # Add image if this record has one
        image_path = record.get("image_path")
        image_id = record.get("id")
        if image_path and image_id and not image_id.endswith("_text_only"):
            pages[page_num]["images"].append({
                "image_id": image_id,
                "image_path": image_path,
                "description": None,  # Will be filled later
            })
    
    return pages


def load_existing_descriptions(output_dir: Path) -> Dict[str, str]:
    """
    Load image descriptions from existing Anki card JSON files.
    Returns mapping of image_id -> description.
    """
    descriptions = {}
    
    # Try occlusion_cards.json (has back_extra field)
    occlusion_cards_path = output_dir / "occlusion_cards.json"
    if occlusion_cards_path.exists():
        try:
            with occlusion_cards_path.open("r", encoding="utf-8") as f:
                cards = json.load(f)
                for card in cards:
                    image_id = card.get("image_id")
                    back_extra = card.get("back_extra", "").strip()
                    if image_id and back_extra:
                        descriptions[image_id] = back_extra
                        logger.debug(f"Loaded description from occlusion_cards.json for {image_id}")
        except Exception as e:
            logger.warning(f"Failed to load occlusion_cards.json: {e}")
    
    # Try image_cards.json (has description field)
    image_cards_path = output_dir / "image_cards.json"
    if image_cards_path.exists():
        try:
            with image_cards_path.open("r", encoding="utf-8") as f:
                cards = json.load(f)
                for card in cards:
                    image_id = card.get("image_id")
                    description = card.get("description", "").strip()
                    if image_id and description:
                        # Only use if not already loaded from occlusion_cards.json
                        if image_id not in descriptions:
                            descriptions[image_id] = description
                            logger.debug(f"Loaded description from image_cards.json for {image_id}")
        except Exception as e:
            logger.warning(f"Failed to load image_cards.json: {e}")
    
    if descriptions:
        logger.info(f"Loaded {len(descriptions)} existing descriptions from card JSON files")
    
    return descriptions


def generate_image_descriptions(
    pages: Dict[int, Dict[str, Any]],
    output_dir: Path,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    language: str = "cs",
    skip_existing: bool = True,
) -> None:
    """Generate image descriptions for all pages that need them."""
    # Load existing descriptions from card JSON files
    existing_descriptions = load_existing_descriptions(output_dir)
    
    model = model or os.getenv("PDF2ANKI_VISION_MODEL", "gpt-4o")
    base_url = base_url or os.getenv("PDF2ANKI_BASE_URL")
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    total_images = sum(len(page["images"]) for page in pages.values())
    processed = 0
    reused = 0
    generated = 0
    
    for page_num, page_data in sorted(pages.items()):
        for img_info in page_data["images"]:
            processed += 1
            image_id = img_info["image_id"]
            image_path = output_dir / img_info["image_path"]
            
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path} (skipping)")
                continue
            
            # Check if description already exists in page data
            if skip_existing and img_info["description"]:
                logger.debug(f"Skipping {image_id} (description already in page data)")
                continue
            
            # Check if we have an existing description from card JSON files
            if image_id in existing_descriptions:
                img_info["description"] = existing_descriptions[image_id]
                reused += 1
                logger.info(f"[{processed}/{total_images}] ✓ Reused description for {image_id}")
                continue
            
            # Generate new description via LLM
            logger.info(f"[{processed}/{total_images}] Generating description for {image_id}...")
            generated += 1
            
            try:
                vision_result = vision.describe_image_only(
                    image_path,
                    page_text=page_data["text"],
                    language=language,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                )
                
                if "error" in vision_result:
                    logger.warning(f"Failed to describe {image_id}: {vision_result['error']}")
                    img_info["description"] = f"[Image description unavailable: {image_id}]"
                else:
                    description = vision_result.get("description", "").strip()
                    if description:
                        img_info["description"] = description
                        logger.info(f"Generated description for {image_id}: {description[:100]}...")
                    else:
                        img_info["description"] = f"[Image: {image_id}]"
                        logger.warning(f"Empty description for {image_id}")
            except Exception as e:
                logger.error(f"Error describing {image_id}: {e}")
                img_info["description"] = f"[Image description error: {image_id}]"
    
    logger.info(f"Description summary: {reused} reused, {generated} generated, {total_images} total")


def calculate_page_tokens(page_data: Dict[str, Any]) -> int:
    """Calculate total tokens for a page (text + image descriptions)."""
    tokens = estimate_tokens(page_data["text"])
    
    for img in page_data["images"]:
        if img.get("description"):
            tokens += estimate_tokens(img["description"])
    
    return tokens


def prepare_audiobook_pages(
    metadata_path: Path,
    output_dir: Path,
    output_path: Path | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    language: str = "cs",
    skip_existing_descriptions: bool = True,
) -> None:
    """Main function to prepare pages with image descriptions."""
    output_path = output_path or (output_dir / "audiobook" / "pages.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading metadata from {metadata_path}")
    records = load_metadata(metadata_path)
    logger.info(f"Loaded {len(records)} metadata records")
    
    logger.info("Grouping pages by page number...")
    pages = group_pages_by_number(records)
    logger.info(f"Found {len(pages)} pages")
    
    total_images = sum(len(page["images"]) for page in pages.values())
    logger.info(f"Found {total_images} images across all pages")
    
    if total_images > 0:
        logger.info("Generating image descriptions...")
        generate_image_descriptions(
            pages,
            output_dir,
            model=model,
            base_url=base_url,
            api_key=api_key,
            language=language,
            skip_existing=skip_existing_descriptions,
        )
    
    # Calculate tokens and prepare output
    logger.info("Calculating token counts...")
    enhanced_pages = []
    for page_num in sorted(pages.keys()):
        page_data = pages[page_num]
        page_data["token_count"] = calculate_page_tokens(page_data)
        page_data["has_diagrams"] = len(page_data["images"]) > 0
        enhanced_pages.append(page_data)
    
    # Save enhanced pages
    logger.info(f"Saving enhanced pages to {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        for page_data in enhanced_pages:
            f.write(json.dumps(page_data, ensure_ascii=False) + "\n")
    
    total_tokens = sum(p["token_count"] for p in enhanced_pages)
    logger.info(f"✓ Prepared {len(enhanced_pages)} pages ({total_tokens:,} total tokens)")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 0: Prepare pages with image descriptions for audiobook generation"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Path to metadata.jsonl (default: <output-dir>/metadata.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("PDF2ANKI_OUTPUT_DIR", "output")),
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for pages.jsonl (default: <output-dir>/audiobook/pages.jsonl)",
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default=os.getenv("PDF2ANKI_VISION_MODEL", "gpt-4o"),
        help="Vision model for image descriptions",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="API key",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("PDF2ANKI_BASE_URL"),
        help="API base URL",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="cs",
        help="Language for image descriptions (default: cs)",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate descriptions even if they exist",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    metadata_path = args.metadata or (args.output_dir / "metadata.jsonl")
    
    prepare_audiobook_pages(
        metadata_path=metadata_path,
        output_dir=args.output_dir,
        output_path=args.output,
        model=args.vision_model,
        base_url=args.base_url,
        api_key=args.api_key,
        language=args.language,
        skip_existing_descriptions=not args.force_regenerate,
    )


if __name__ == "__main__":
    main()

