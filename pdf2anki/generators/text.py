"""
Generate Q&A flashlcards from text using an LLM.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def generate_text_cards(
    text_segment: str,
    num_cards: int = 3,
    model: str = "gpt-4o-mini",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Generate conceptual Q&A cards from the provided text.
    Returns a list of dicts with 'Front' and 'Back' keys.
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed")

    client = OpenAI(
        base_url=base_url or os.environ.get("PDF2ANKI_BASE_URL"), 
        api_key=api_key or os.environ.get("OPENAI_API_KEY")
    )
    
    prompt = (
        f"Read the following educational text and generate {num_cards} high-quality flashcards "
        "to test understanding of the key concepts. \n"
        "Output a JSON object with a key 'cards' containing a list of objects, "
        "each with 'front' (question) and 'back' (answer) fields.\n"
        "Keep questions concise and answers comprehensive but to the point.\n\n"
        f"Text:\n{text_segment}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert educator creating Anki flashcards. Output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if not content:
            return []
            
        data = json.loads(content)
        return data.get("cards", [])
        
    except Exception as e:
        logging.error(f"Text card generation failed: {e}")
        return []

