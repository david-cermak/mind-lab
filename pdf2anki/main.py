"""
Master orchestrator for PDF to Anki pipeline.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from pdf2anki.extractors import pdf, ocr
from pdf2anki.analysis import strategy, vision
from pdf2anki.generators import text, occlusion
from pdf2anki.utils.anki_db import DeckBuilder

# Load .env file from pdf2anki directory or current directory
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()  # Fallback to current directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="PDF to Anki Converter")
    parser.add_argument("pdf_path", type=Path, help="Input PDF file")
    parser.add_argument("--output-dir", type=Path, default=Path(os.getenv("PDF2ANKI_OUTPUT_DIR", "output")), help="Directory for intermediate files")
    parser.add_argument("--final-apkg", type=Path, default=Path(os.getenv("PDF2ANKI_FINAL_APKG", "result.apkg")), help="Path for final .apkg")
    parser.add_argument("--deck-name", type=str, default="PDF Deck", help="Name of the Anki deck")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR if already done")
    parser.add_argument("--max-text-cards", type=int, default=int(os.getenv("PDF2ANKI_MAX_TEXT_CARDS", "10")), help="Safety limit for text card generation")
    parser.add_argument("--tokens-per-card", type=int, default=int(os.getenv("PDF2ANKI_TOKENS_PER_CARD", "300")), help="Target tokens per card (lower = denser cards, default: 300)")
    
    # Occlusion config
    parser.add_argument("--occlude-all", type=lambda x: x.lower() == "true", default=os.getenv("PDF2ANKI_OCCLUDE_ALL", "true").lower() == "true", help="Occlusion mode: true (oi=1, hide all + guess one) or false (oi=0, hide one + guess one)")
    
    # LLM Config - use .env defaults if not provided via CLI
    parser.add_argument("--llm-model", default=os.getenv("PDF2ANKI_LLM_MODEL", "gpt-4o-mini"), help="Model to use")
    parser.add_argument("--vision-model", default=os.getenv("PDF2ANKI_VISION_MODEL", "gpt-4o"), help="Vision model to use")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API Key")
    parser.add_argument("--base-url", default=os.getenv("PDF2ANKI_BASE_URL"), help="Custom API Base URL")

    args = parser.parse_args()
    
    # Set environment variables for downstream modules
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["PDF2ANKI_BASE_URL"] = args.base_url
        
    # 1. Extraction
    logging.info("Step 1: Extracting assets from PDF...")
    pdf.extract_assets(args.pdf_path, args.output_dir)
    metadata_path = args.output_dir / "metadata.jsonl"
    
    # 2. OCR
    logging.info("Step 2: Running OCR on images...")
    ocr_dir = args.output_dir / "ocr"
    ocr.run_ocr(metadata_path, ocr_dir, skip_existing=args.skip_ocr)
    
    # 3. Analysis & Strategy
    logging.info("Step 3: Analyzing content and planning strategy...")
    plan_path = args.output_dir / "plan.json"
    strategy.create_strategy_plan(metadata_path, ocr_dir, plan_path, tokens_per_card=args.tokens_per_card)
    
    with open(plan_path, "r", encoding="utf-8") as f:
        plan_data = json.load(f)
        plan = plan_data.get("plan", [])

    deck_builder = DeckBuilder(args.deck_name, args.final_apkg)
    text_cards_generated = 0
    
    # Execute Plan
    logging.info(f"Step 4-6: Executing card generation plan ({len(plan)} items)...")
    
    for item in plan:
        item_type = item.get("type")
        page_num = item.get("page_number")
        
        if item_type == "text_qa":
            if text_cards_generated >= args.max_text_cards:
                logging.info(f"Skipping text card for page {page_num} (limit reached)")
                continue
                
            # Load text context
            # In a real run, we'd look up the exact text file from metadata
            # Here we can try to find it or use what's in metadata if we cached it
            # Simple approach: assume page text is available
            # Re-read metadata to find text path
            text_file = item.get("source_text_path")
            if not text_file:
                continue
                
            text_path = args.output_dir / text_file
            if text_path.exists():
                page_text = text_path.read_text(encoding="utf-8")
                
                logging.info(f"Generating {item.get('estimated_cards', 1)} text cards for page {page_num}")
                
                cards = text.generate_text_cards(
                    page_text, 
                    num_cards=item.get("estimated_cards", 1),
                    model=args.llm_model,
                    base_url=args.base_url,
                    api_key=args.api_key
                )
                
                for c in cards: 
                    if c.get("front") and c.get("back"):
                        deck_builder.add_text_note(c['front'], c['back'], tags=["pdf-import", "text"])
                        text_cards_generated += 1
            
            
        elif item_type == "image_occlusion":
            img_id = item.get("image_id")
            ocr_source = item.get("ocr_source")
            img_rel_path = item.get("image_path")
            
            if not img_rel_path or not ocr_source:
                continue
            
            # Resolve OCR path - handle various formats
            ocr_path = Path(ocr_source)
            
            if ocr_path.is_absolute():
                # Absolute path - use as-is if it exists
                if not ocr_path.exists():
                    # Try relative to output_dir if absolute path doesn't exist
                    ocr_path = args.output_dir / "ocr" / ocr_path.name
            else:
                # Relative path - check if it starts with "output/" and replace with actual output_dir
                ocr_str = str(ocr_path)
                if ocr_str.startswith("output/"):
                    # Replace "output/" prefix with actual output_dir
                    ocr_path = args.output_dir / ocr_str[7:]  # Remove "output/" prefix (7 chars)
                elif ocr_str.startswith("ocr/"):
                    # Just "ocr/..." - prepend output_dir
                    ocr_path = args.output_dir / ocr_path
                else:
                    # Other relative path - try relative to output_dir
                    ocr_path = args.output_dir / ocr_path
            
            if not ocr_path.exists():
                logging.warning(f"OCR file not found: {ocr_path} for {img_id} (source: {ocr_source})")
                continue
                
            img_path = args.output_dir / img_rel_path
            
            # Generate Occlusion
            try:
                with open(ocr_path, "r", encoding="utf-8") as f:
                    ocr_data = json.load(f)
                    
                with open(img_path, "rb") as f:
                    # We need dims, PIL does this
                    from PIL import Image
                    with Image.open(img_path) as img:
                        w, h = img.size
                
                # Get page text from metadata for vision LLM context
                page_text = ""
                metadata_path = args.output_dir / "metadata.jsonl"
                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                record = json.loads(line)
                                if record.get("id") == img_id:
                                    page_text = record.get("page_text", "")
                                    break
                
                # Call vision LLM to get semantic groups
                semantic_groups = None
                vision_description = None
                try:
                    logging.info(f"Calling vision LLM for {img_id}...")
                    ocr_tokens = ocr_data.get("tokens", [])
                    vision_result = vision.analyze_image_context(
                        img_path,
                        page_text,
                        ocr_tokens,
                        model=args.vision_model,
                        base_url=args.base_url,
                        api_key=args.api_key,
                    )
                    
                    if "error" not in vision_result:
                        semantic_groups = vision_result.get("groups")
                        vision_description = vision_result.get("description")
                        if semantic_groups:
                            logging.info(f"Vision LLM provided {len(semantic_groups)} semantic groups for {img_id}")
                        else:
                            logging.warning(f"Vision LLM returned no groups for {img_id}, using spatial-only")
                    else:
                        logging.warning(f"Vision LLM error for {img_id}: {vision_result.get('error')}, using spatial-only")
                except Exception as e:
                    logging.warning(f"Vision LLM call failed for {img_id}: {e}, using spatial-only grouping")
                
                # Generate occlusion with semantic groups if available
                occlusion_data = occlusion.generate_occlusion_card_data(
                    ocr_data, w, h, min_confidence=75.0, semantic_groups=semantic_groups, occlude_all=args.occlude_all
                )
                
                if occlusion_data["rectangles"]:
                    # Use vision description as back_extra if available
                    back_extra = vision_description if vision_description else None
                    
                    deck_builder.add_occlusion_note(
                        markup=occlusion_data["markup"],
                        image_path=img_path,
                        header=f"Diagram (Page {page_num})",
                        back_extra=back_extra,
                        tags=["pdf-import", "occlusion"]
                    )
                    method = occlusion_data.get("grouping_method", "unknown")
                    logging.info(f"Added occlusion card for {img_id} ({method}, {occlusion_data['group_count']} groups)")
            except Exception as e:
                logging.error(f"Failed to process occlusion for {img_id}: {e}")

    # 7. Build Deck
    logging.info("Step 7: Assembling final deck...")
    deck_builder.build()
    logging.info(f"Done! Deck saved to {args.final_apkg}")


if __name__ == "__main__":
    main()

