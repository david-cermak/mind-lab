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
    model: str = "gpt-4o",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 600,
) -> Dict[str, Any]:
    """
    Ask Vision LLM to describe the image in the context of the page
    and verify if OCR tokens match visual labels.
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed")

    client = OpenAI(
        base_url=base_url or os.environ.get("PDF2ANKI_BASE_URL"),
        api_key=api_key or os.environ.get("OPENAI_API_KEY")
    )
    
    # Summarize OCR for context
    ocr_summary = ", ".join([t.get("text", "") for t in ocr_tokens[:50]]) # Limit to avoid huge context
    
    prompt = (
        "Analyze this image from a textbook.\n"
        f"Context from page text:\n{page_text[:1000]}...\n\n"
        f"OCR detected text labels:\n{ocr_summary}\n\n"
        "1. Provide a concise description of what this diagram/image shows.\n"
        "2. Is this a diagram suitable for image occlusion (hiding labels to test memory)? (Yes/No)\n"
        "3. Are the OCR labels accurate representations of the visual labels? (Yes/No)\n"
        "Return JSON with keys: 'description', 'suitable_for_occlusion', 'ocr_accuracy_comment'."
    )

    image_url = _image_to_data_url(image_path)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant analysing educational images. Output valid JSON."},
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

