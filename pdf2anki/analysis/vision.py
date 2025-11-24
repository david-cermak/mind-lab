"""
Vision analysis module to describe images and verify occlusion suitability.
wraps describe_image.py functionality.
"""

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _image_to_data_url(image_path: Path) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    suffix = image_path.suffix.lower().lstrip(".") or "png"
    mime = f"image/{'jpeg' if suffix in {'jpg', 'jpeg'} else suffix}"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def analyze_image_context(
    image_path: Path,
    page_text: str,
    ocr_tokens: List[Dict[str, Any]],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 2000,
) -> Dict[str, Any]:
    """
    Ask Vision LLM to describe the image and group text labels semantically.
    Returns description and groups of token indices that belong together.
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed")

    # Use provided model, or env var, or default
    if model is None:
        model = os.environ.get("PDF2ANKI_VISION_MODEL", "gpt-4o")

    client = OpenAI(
        base_url=base_url or os.environ.get("PDF2ANKI_BASE_URL"),
        api_key=api_key or os.environ.get("OPENAI_API_KEY")
    )
    
    # Format OCR tokens with positions for grouping
    ocr_formatted = []
    for tok in ocr_tokens:
        bbox = tok.get("bbox", [])
        ocr_formatted.append({
            "text": tok.get("text", ""),
            "bbox": {"left": bbox[0], "top": bbox[1], "width": bbox[2], "height": bbox[3]},
            "confidence": tok.get("confidence", 0)
        })
    
    ocr_json = json.dumps(ocr_formatted, ensure_ascii=False, indent=2)
    
    system_msg = "You are a helpful assistant analyzing educational diagrams. Output valid JSON."
    
    prompt = (
        "Analyze this educational diagram from a textbook.\n\n"
        f"Context from page text:\n{page_text[:1000]}\n\n"
        f"OCR detected text labels with their positions:\n{ocr_json}\n\n"
        "Tasks:\n"
        "1. Provide a concise description (2-3 sentences) of what this diagram shows.\n"
        "2. Group the text labels that belong together semantically (e.g., labels for the same structure, "
        "or labels that form a complete concept). Return groups as an array where each group contains "
        "the indices of tokens that should be grouped together.\n\n"
        "Return JSON with keys:\n"
        "- 'description': string\n"
        "- 'groups': array of arrays, where each inner array contains token indices (0-based) that belong together\n"
        "- 'group_labels': optional array of descriptive labels for each group"
    )

    image_url = _image_to_data_url(image_path)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]
                }
            ],
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if not content:
            return {"error": "Empty response"}
            
        return json.loads(content)
        
    except Exception as e:
        logging.error(f"Vision analysis failed: {e}")
        return {"error": str(e)}

