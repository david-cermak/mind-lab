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

from PIL import Image

from pdf2anki.ocr_ascii_canvas import ascii_canvas_from_tokens

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

    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception:
        width = height = 1

    ascii_canvas, legend = ascii_canvas_from_tokens(
        ocr_tokens,
        width,
        height,
        rows=int(os.environ.get("PDF2ANKI_ASCII_ROWS", "40")),
        cols=int(os.environ.get("PDF2ANKI_ASCII_COLS", "80")),
        min_confidence=float(os.environ.get("PDF2ANKI_ASCII_MIN_CONF", "70")),
    )
    legend_text = "\n".join(legend) if legend else "(no OCR tokens with sufficient confidence)"
    ascii_section = (
        "OCR label map (ASCII projection, indices are original OCR positions and may skip numbers):\n"
        "```\n"
        f"{ascii_canvas}\n"
        "```\n"
        "Legend:\n"
        f"{legend_text}\n"
    )

    system_msg = "You are a helpful assistant analyzing educational diagrams. Output valid JSON."

    prompt = (
        "Analyze this educational diagram from a textbook.\n\n"
        f"Context from page text:\n{page_text[:1000]}\n\n"
        f"{ascii_section}\n"
        "Tasks:\n"
        "1. Provide a concise description (2-3 sentences) of what this diagram shows.\n"
        "2. Group the text labels that belong together semantically (e.g., labels for the same structure, "
        "or labels that form a complete concept). Return groups as an array where each group contains "
        "the indices of tokens (0-based original OCR indices from the legend; some numbers may be missing) that belong together.\n\n"
        "Return JSON with keys:\n"
        "- 'description': string\n"
        "- 'groups': array of arrays, where each inner array contains token indices (0-based original OCR indices) that belong together\n"
        "- 'group_labels': optional array of descriptive labels for each group"
    )

    debug_dir = os.environ.get("PDF2ANKI_VISION_DEBUG_DIR")
    image_url = _image_to_data_url(image_path)
    if debug_dir:
        try:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            debug_input = {
                "image_path": str(image_path),
                "model": model,
                "system": system_msg,
                "prompt": prompt,
            }
            input_path = Path(debug_dir) / f"{image_path.stem}_vision_input.json"
            input_path.write_text(json.dumps(debug_input, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logging.warning(f"Failed to write vision debug input: {exc}")
    
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
        
        if debug_dir:
            try:
                output_path = Path(debug_dir) / f"{image_path.stem}_vision_output.json"
                output_path.write_text(content, encoding="utf-8")
            except Exception as exc:
                logging.warning(f"Failed to write vision debug output: {exc}")
            
        return json.loads(content)
        
    except Exception as e:
        logging.error(f"Vision analysis failed: {e}")
        return {"error": str(e)}


def describe_image_only(
    image_path: Path,
    page_text: str = "",
    *,
    language: str = "cs",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 800,
) -> Dict[str, Any]:
    """
    Ask a vision-capable LLM to describe the image in the requested language.
    Returns JSON with at least the key 'description'.
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed")

    if model is None:
        model = os.environ.get("PDF2ANKI_VISION_MODEL", "gpt-4o")

    client = OpenAI(
        base_url=base_url or os.environ.get("PDF2ANKI_BASE_URL"),
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
    )

    system_msg = (
        "You are a helpful assistant that writes short educational image descriptions "
        "and always returns valid JSON."
    )

    prompt = (
        "Analyze the provided educational illustration and describe what it shows.\n"
        f"- Language: {language} (use fluent, natural sentences).\n"
        "- Length: 2-3 sentences that mention the key structures or concepts.\n"
        "- Be factual; do not invent content that is not clearly visible.\n"
        "- If contextual text is provided, only include details that match the image.\n\n"
        f"Context from surrounding text (optional):\n{page_text[:1000]}\n\n"
        "Return JSON with:\n"
        '{\n  "description": "<paragraph in requested language>"\n}\n'
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
                    ],
                },
            ],
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            return {"error": "Empty response"}

        return json.loads(content)

    except Exception as e:
        logging.error(f"Vision description failed: {e}")
        return {"error": str(e)}

