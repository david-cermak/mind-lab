"""
Generate a single image occlusion card without rerunning the full pipeline.
Useful for testing occlusion card generation after OCR and strategy steps are done.
"""

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from pdf2anki.analysis import vision
from pdf2anki.generators import occlusion
from pdf2anki.utils.anki_db import DeckBuilder

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_single_occlusion_card(
    image_id: str,
    output_dir: Path,
    deck_name: str = "Test Deck",
    output_apkg: Path = Path("test_occlusion.apkg"),
    vision_model: str = None,
    base_url: str = None,
    api_key: str = None,
):
    """Generate a single occlusion card for the given image_id."""
    
    # Set defaults from env
    vision_model = vision_model or os.getenv("PDF2ANKI_VISION_MODEL", "gpt-4o")
    base_url = base_url or os.getenv("PDF2ANKI_BASE_URL")
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    
    # Load metadata to find image info
    metadata_path = output_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    image_record = None
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record.get("id") == image_id:
                    image_record = record
                    break
    
    if not image_record:
        raise ValueError(f"Image ID '{image_id}' not found in metadata")
    
    img_rel_path = image_record.get("image_path")
    page_text = image_record.get("page_text", "")
    page_num = image_record.get("page_number", "?")
    
    # Resolve paths
    img_path = output_dir / img_rel_path
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    # Find OCR file - try multiple possible locations
    ocr_paths = [
        output_dir / "ocr" / f"{image_id}.json",
        output_dir / "output" / "ocr" / f"{image_id}.json",
        Path(f"output/ocr/{image_id}.json"),
    ]
    
    ocr_path = None
    for path in ocr_paths:
        if path.exists():
            ocr_path = path
            break
    
    if not ocr_path:
        # Try to find it from plan.json
        plan_path = output_dir / "plan.json"
        if plan_path.exists():
            with open(plan_path, "r", encoding="utf-8") as f:
                plan_data = json.load(f)
                for item in plan_data.get("plan", []):
                    if item.get("image_id") == image_id and item.get("type") == "image_occlusion":
                        ocr_source = item.get("ocr_source")
                        if ocr_source:
                            ocr_path_candidate = Path(ocr_source)
                            if not ocr_path_candidate.is_absolute():
                                # Handle "output/ocr/..." pattern
                                if str(ocr_path_candidate).startswith("output/"):
                                    ocr_path_candidate = output_dir / str(ocr_path_candidate)[7:]
                                else:
                                    ocr_path_candidate = output_dir / ocr_path_candidate
                            if ocr_path_candidate.exists():
                                ocr_path = ocr_path_candidate
                                break
    
    if not ocr_path or not ocr_path.exists():
        raise FileNotFoundError(f"OCR file not found for {image_id}. Tried: {ocr_paths}")
    
    logging.info(f"Processing image: {image_id}")
    logging.info(f"  Image: {img_path}")
    logging.info(f"  OCR: {ocr_path}")
    logging.info(f"  Page: {page_num}")
    
    # Load OCR data
    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)
    
    # Get image dimensions
    with Image.open(img_path) as img:
        w, h = img.size
    
    # Call vision LLM
    semantic_groups = None
    vision_description = None
    try:
        logging.info("Calling vision LLM...")
        ocr_tokens = ocr_data.get("tokens", [])
        vision_result = vision.analyze_image_context(
            img_path,
            page_text,
            ocr_tokens,
            model=vision_model,
            base_url=base_url,
            api_key=api_key,
        )
        
        if "error" not in vision_result:
            semantic_groups = vision_result.get("groups")
            vision_description = vision_result.get("description")
            if semantic_groups:
                logging.info(f"Vision LLM provided {len(semantic_groups)} semantic groups")
            else:
                logging.warning("Vision LLM returned no groups, using spatial-only")
        else:
            logging.warning(f"Vision LLM error: {vision_result.get('error')}, using spatial-only")
    except Exception as e:
        logging.warning(f"Vision LLM call failed: {e}, using spatial-only grouping")
    
    # Generate occlusion card
    occlusion_data = occlusion.generate_occlusion_card_data(
        ocr_data, w, h, min_confidence=75.0, semantic_groups=semantic_groups
    )
    
    if not occlusion_data["rectangles"]:
        raise ValueError(f"No occlusion rectangles generated for {image_id}")
    
    method = occlusion_data.get("grouping_method", "unknown")
    logging.info(f"Generated {occlusion_data['group_count']} groups using {method}")
    
    # Create deck with single card
    deck_builder = DeckBuilder(deck_name, output_apkg)
    deck_builder.add_occlusion_note(
        markup=occlusion_data["markup"],
        image_path=img_path,
        header=f"Diagram (Page {page_num})",
        back_extra=vision_description if vision_description else None,
        tags=["pdf-import", "occlusion", "test"]
    )
    
    deck_builder.build()
    logging.info(f"Deck saved to {output_apkg}")
    logging.info(f"Card has {len(occlusion_data['rectangles'])} occlusion rectangles")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a single image occlusion card (assumes OCR and metadata already exist)"
    )
    parser.add_argument("image_id", help="Image ID (e.g., page3_img0_xref23)")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--deck-name", default="Test Occlusion Card", help="Deck name")
    parser.add_argument("--output-apkg", type=Path, default=Path("test_occlusion.apkg"), help="Output .apkg file")
    parser.add_argument("--vision-model", default=os.getenv("PDF2ANKI_VISION_MODEL", "gpt-4o"), help="Vision model")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="API key")
    parser.add_argument("--base-url", default=os.getenv("PDF2ANKI_BASE_URL"), help="API base URL")
    
    args = parser.parse_args()
    
    generate_single_occlusion_card(
        args.image_id,
        args.output_dir,
        args.deck_name,
        args.output_apkg,
        args.vision_model,
        args.base_url,
        args.api_key,
    )


if __name__ == "__main__":
    main()

