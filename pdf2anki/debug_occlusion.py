"""
Debug tool for image occlusion processing.
Processes a single image and prints prompts without ling LLM.
"""

import argparse
import json
import base64
import os
from pathlib import Path
from typing import Any, Dict, List

from pdf2anki.analysis import vision
from pdf2anki.generators import occlusion
from pdf2anki.ocr_ascii_canvas import ascii_canvas_from_tokens


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token."""
    return len(text) // 4


def estimate_image_tokens(image_base64_size: int) -> int:
    """
    Estimate tokens for image.
    For GPT-4o vision: ~85 tokens base + (base64_size / 4) for detail.
    This is approximate.
    """
    return 85 + (image_base64_size // 4)


def print_prompt_preview(
    image_path: Path,
    page_text: str,
    ocr_tokens: List[Dict[str, Any]],
    *,
    model: str,
    image_width: int,
    image_height: int,
    ascii_rows: int = 40,
    ascii_cols: int = 80,
    min_confidence: float = 70.0,
) -> None:
    """Print what would be sent to the vision LLM without actually calling it."""
    
    ascii_canvas, legend = ascii_canvas_from_tokens(
        ocr_tokens,
        image_width,
        image_height,
        rows=ascii_rows,
        cols=ascii_cols,
        min_confidence=min_confidence,
    )
    legend_text = "\n".join(legend) if legend else "(no OCR tokens after filtering)"
    
    system_msg = "You are a helpful assistant analyzing educational diagrams. Output valid JSON."
    
    ascii_section = (
        f"ASCII map of detected labels (rows={ascii_rows}, cols={ascii_cols}; indices are original OCR positions and may have gaps):\n"
        "```\n"
        f"{ascii_canvas}\n"
        "```\n"
        "Legend (indices map to the ASCII markers and approximate widths; numbering starts at 0):\n"
        f"{legend_text}\n"
    )
    
    prompt = (
        "Analyze this educational diagram from a textbook.\n\n"
        f"Context from page text:\n{page_text[:1000]}\n\n"
        f"{ascii_section}\n"
        "Tasks:\n"
        "1. Provide a concise description (2-3 sentences) of what this diagram shows.\n"
        "2. Group the text labels that belong together semantically (e.g., labels for the same structure, "
        "or labels that form a complete concept). Return groups as an array where each group contains "
        "the indices of tokens (referenced via the legend, 0-based original OCR indices; numbers may be missing).\n\n"
        "Return JSON with keys:\n"
        "- 'description': string\n"
        "- 'groups': array of arrays, where each inner array contains token indices (0-based original OCR indices) that belong together\n"
        "- 'group_labels': optional array of descriptive labels for each group"
    )
    
    # Encode image
    suffix = image_path.suffix.lower().lstrip(".") or "png"
    mime = f"image/{'jpeg' if suffix in {'jpg', 'jpeg'} else suffix}"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
    
    # Estimate tokens
    system_tokens = estimate_tokens(system_msg)
    prompt_tokens = estimate_tokens(prompt)
    image_tokens = estimate_image_tokens(len(encoded))
    total_estimated = system_tokens + prompt_tokens + image_tokens
    
    image_data_url = f"data:{mime};base64,{encoded[:100]}..."  # Truncate for display
    
    print("=" * 80)
    print("VISION LLM PROMPT PREVIEW")
    print("=" * 80)
    print(f"\nModel: {model}")
    print(f"\nImage: {image_path}")
    print(f"Image size: {len(image_bytes)} bytes raw, {len(encoded)} bytes base64")
    print(f"\nToken Estimates:")
    print(f"  System message: ~{system_tokens} tokens")
    print(f"  User prompt text: ~{prompt_tokens} tokens")
    print(f"  Image: ~{image_tokens} tokens")
    print(f"  TOTAL ESTIMATED: ~{total_estimated} tokens")
    print(f"\nSystem Message:")
    print(system_msg)
    print(f"\nUser Message:")
    print("-" * 80)
    print("Text content:")
    print(prompt)
    print("-" * 80)
    print(f"\nImage URL (truncated): {image_data_url}")
    print(f"\nOCR tokens sent ({len(ocr_tokens)} total):")
    for i, tok in enumerate(ocr_tokens[:10]):
        bbox = tok.get("bbox", [])
        print(f"  {i}. '{tok.get('text')}' @ ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}) (conf: {tok.get('confidence', 0):.1f}%)")
    if len(ocr_tokens) > 10:
        print(f"  ... and {len(ocr_tokens) - 10} more tokens (see JSON above)")
    print("=" * 80)


def debug_single_image(
    image_id: str,
    output_dir: Path,
    *,
    print_prompt: bool = True,
    skip_llm: bool = True,
    vision_model: str,
) -> None:
    """Debug a single image occlusion card."""
    
    # Load metadata to find image
    metadata_path = output_dir / "metadata.jsonl"
    ocr_dir = output_dir / "ocr"
    
    image_record = None
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record.get("id") == image_id:
                    image_record = record
                    break
    
    if not image_record:
        print(f"ERROR: Image ID '{image_id}' not found in metadata")
        return
    
    img_rel_path = image_record.get("image_path")
    page_text = image_record.get("page_text", "")
    page_num = image_record.get("page_number", "?")
    
    img_path = output_dir / img_rel_path
    ocr_path = ocr_dir / f"{image_id}.json"
    
    if not img_path.exists():
        print(f"ERROR: Image not found: {img_path}")
        return
    
    if not ocr_path.exists():
        print(f"ERROR: OCR data not found: {ocr_path}")
        return
    
    print(f"\nProcessing image: {image_id}")
    print(f"Page: {page_num}")
    print(f"Image: {img_path}")
    print(f"OCR: {ocr_path}\n")
    
    # Load OCR data
    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)
    
    ocr_tokens = ocr_data.get("tokens", [])
    print(f"Found {len(ocr_tokens)} OCR tokens")
    
    # Get image dimensions
    from PIL import Image
    with Image.open(img_path) as img:
        w, h = img.size
        print(f"Image dimensions: {w}x{h}\n")
    
    # Generate occlusion data (spatial only, no LLM)
    print("=" * 80)
    print("SPATIAL GROUPING RESULTS")
    print("=" * 80)
    occlusion_data = occlusion.generate_occlusion_card_data(
        ocr_data, w, h, min_confidence=75.0
    )
    
    print(f"\nTokens after confidence filter: {occlusion_data['token_count']}")
    print(f"Groups created: {occlusion_data['group_count']}")
    print(f"\nRectangles:")
    for rect in occlusion_data["rectangles"]:
        print(f"  {rect['index']}. '{rect['label']}'")
        print(f"     Tokens: {rect['tokens']}")
        print(f"     BBox (pixels): {rect['bbox_pixels']}")
        print(f"     BBox (normalized): {rect['bbox_normalized']}")
    
    print(f"\nAnki Markup:")
    print(occlusion_data["markup"])
    
    # Print vision prompt if requested
    if print_prompt:
        print_prompt_preview(
            img_path,
            page_text,
            ocr_tokens,
            model=vision_model,
            image_width=w,
            image_height=h,
        )
    
    # Optionally call vision LLM if not skipping
    if not skip_llm:
        print("\n" + "=" * 80)
        print("CALLING VISION LLM...")
        print("=" * 80)
        try:
            result = vision.analyze_image_context(
                img_path,
                page_text,
                ocr_tokens,
                model=vision_model,
                base_url=os.getenv("PDF2ANKI_BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            print("\nVision Analysis Result:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"ERROR calling vision LLM: {e}")


def main():
    parser = argparse.ArgumentParser(description="Debug single image occlusion card")
    parser.add_argument("image_id", help="Image ID (e.g., page3_img0_xref23)")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--no-prompt", action="store_true", help="Don't print prompt preview")
    parser.add_argument("--call-llm", action="store_true", help="Actually call the vision LLM (default: skip)")
    parser.add_argument(
        "--vision-model",
        type=str,
        default=None,
        help="Override vision model (default: PDF2ANKI_VISION_MODEL or gpt-4o)",
    )
    
    args = parser.parse_args()
    
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    vision_model = args.vision_model or os.getenv("PDF2ANKI_VISION_MODEL", "gpt-4o")
    
    debug_single_image(
        args.image_id,
        args.output_dir,
        print_prompt=not args.no_prompt,
        skip_llm=not args.call_llm,
        vision_model=vision_model,
    )


if __name__ == "__main__":
    main()

