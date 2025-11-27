"""
Stage 2: Process chunks into continuous narrative text.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


def load_chunks(chunks_path: Path) -> Dict[str, Any]:
    """Load chunks.json."""
    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        return {"chunks": []}
    
    with chunks_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_enhanced_pages(pages_path: Path) -> Dict[int, Dict[str, Any]]:
    """Load enhanced pages and index by page number."""
    pages = {}
    if not pages_path.exists():
        logger.error(f"Pages file not found: {pages_path}")
        return pages
    
    with pages_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                page_data = json.loads(line)
                page_num = page_data.get("page_number")
                if page_num is not None:
                    pages[page_num] = page_data
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid page line: {e}")
    
    return pages


def build_chunk_text(
    chunk: Dict[str, Any],
    pages: Dict[int, Dict[str, Any]],
) -> str:
    """Build the full text for a chunk including page text and image descriptions."""
    parts = []
    
    for page_num in sorted(chunk["pages"]):
        if page_num not in pages:
            logger.warning(f"Page {page_num} not found in pages data")
            continue
        
        page_data = pages[page_num]
        
        # Add page text
        page_text = page_data.get("text", "").strip()
        if page_text:
            parts.append(page_text)
        
        # Add image descriptions at the end of page
        # Note: We don't know exact position, so they appear at end
        images = page_data.get("images", [])
        if images:
            for img in images:
                description = img.get("description")
                if description:
                    # Mark as image description so LLM knows it's at end of page
                    parts.append(f"\n[Image: {description}]")
    
    return "\n\n".join(parts)


def process_chunk(
    chunk: Dict[str, Any],
    chunk_text: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    language: str,
    style: str,
) -> Dict[str, Any]:
    """Process a single chunk with LLM."""
    if OpenAI is None:
        raise RuntimeError("openai package not installed")
    
    client = OpenAI(
        base_url=base_url or os.getenv("PDF2ANKI_BASE_URL"),
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
    )
    
    system_msg = (
        "You are converting educational textbook content into a continuous narrative "
        "suitable for audiobook narration. Output plain text only, no markdown."
    )
    
    prompt = f"""You are converting educational textbook content into a continuous narrative suitable for audiobook narration.

Task:
- Convert structured text (lists, bullet points, itemization) into flowing prose
- Integrate image descriptions naturally into the narrative flow
- Maintain educational tone and accuracy
- Use natural transitions between topics
- Language: {language}
- Style: {style}

Input text:
{chunk_text}

Instructions:
- Convert "• Item 1\\n• Item 2" → "The first item is... Additionally, the second item..."
- Convert "1. First\\n2. Second" → "First, we have... Second, we consider..."
- IMPORTANT: Image descriptions appear at the end of each page's text. You don't know the exact position 
  of images in the original PDF, so integrate them naturally where they make contextual sense in the narrative.
  For example, if a page discusses "plant tissues" and ends with "[Image: Diagram showing plant tissue structure]",
  integrate it as: "As illustrated in the accompanying diagram, plant tissues consist of..."
- Preserve technical terms and definitions exactly
- Maintain paragraph structure for natural pauses
- Create smooth transitions between pages and topics
- Remove any "[Image: ...]" markers after integrating the descriptions

Output: Continuous narrative text ready for TTS. Use plain text only, no markdown formatting."""
    
    start_time = time.time()
    
    try:
        # Some models (like gpt-5) only support temperature=1, so we'll try 0.7 first, then fallback to 1
        temperature = float(os.getenv("PDF2ANKI_TEMPERATURE", "0.7"))
        
        logger.debug(f"Calling LLM with model={model}, temperature={temperature}")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
        except Exception as temp_error:
            # If temperature error and we're not already at 1, retry with temperature=1
            error_str = str(temp_error)
            if "temperature" in error_str.lower() and temperature != 1.0:
                logger.warning(f"Temperature {temperature} not supported, retrying with temperature=1.0")
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=1.0,
                )
            else:
                raise
        
        narrative = response.choices[0].message.content or ""
        processing_time = time.time() - start_time
        
        # Estimate tokens
        input_tokens = len(chunk_text) // 4  # Rough estimate
        output_tokens = len(narrative) // 4
        
        return {
            "narrative": narrative.strip(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "processing_time": processing_time,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Error processing chunk {chunk.get('chunk_id')}: {e}")
        return {
            "narrative": "",
            "error": str(e),
            "success": False,
        }


def process_audiobook_chunks(
    chunks_path: Path,
    pages_path: Path,
    output_dir: Path,
    model: str,
    base_url: str | None,
    api_key: str | None,
    language: str,
    style: str,
) -> None:
    """Main function to process chunks into narratives."""
    logger.info(f"Loading chunks from {chunks_path}")
    chunks_data = load_chunks(chunks_path)
    chunks = chunks_data.get("chunks", [])
    
    if not chunks:
        logger.error("No chunks found. Run split_audiobook_text first.")
        return
    
    logger.info(f"Loaded {len(chunks)} chunks")
    
    logger.info(f"Loading pages from {pages_path}")
    pages = load_enhanced_pages(pages_path)
    logger.info(f"Loaded {len(pages)} pages")
    
    # Log model configuration
    logger.info(f"Using LLM model: {model}")
    if base_url:
        logger.info(f"API base URL: {base_url}")
    
    narratives_dir = output_dir / "audiobook" / "narratives"
    narratives_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {"chunks": []}
    
    for i, chunk in enumerate(chunks, 1):
        chunk_id = chunk["chunk_id"]
        logger.info(f"[{i}/{len(chunks)}] Processing {chunk_id}...")
        
        # Build chunk text
        chunk_text = build_chunk_text(chunk, pages)
        
        # Process with LLM
        result = process_chunk(
            chunk,
            chunk_text,
            model=model,
            base_url=base_url,
            api_key=api_key,
            language=language,
            style=style,
        )
        
        if result["success"]:
            # Save narrative
            narrative_path = narratives_dir / f"{chunk_id}.txt"
            narrative_path.write_text(result["narrative"], encoding="utf-8")
            logger.info(
                f"  ✓ Saved {chunk_id}: {result['output_tokens']:,} tokens "
                f"({result['processing_time']:.1f}s)"
            )
            
            metadata["chunks"].append({
                "chunk_id": chunk_id,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "processing_time": result["processing_time"],
                "file": f"narratives/{chunk_id}.txt",
                "success": True,
            })
        else:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"  ✗ Failed {chunk_id}: {error_msg}")
            metadata["chunks"].append({
                "chunk_id": chunk_id,
                "error": str(error_msg),
                "success": False,
            })
    
    # Save metadata
    metadata_path = output_dir / "audiobook" / "narratives_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    successful = sum(1 for c in metadata["chunks"] if c.get("success", False))
    failed = len(chunks) - successful
    if successful == len(chunks):
        logger.info(f"✓ Processed {successful}/{len(chunks)} chunks successfully")
    else:
        logger.warning(f"⚠ Processed {successful}/{len(chunks)} chunks successfully ({failed} failed)")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Process chunks into continuous narrative text"
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        help="Path to chunks.json (default: <output-dir>/audiobook/chunks.json)",
    )
    parser.add_argument(
        "--pages",
        type=Path,
        help="Path to pages.jsonl (default: <output-dir>/audiobook/pages.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("PDF2ANKI_OUTPUT_DIR", "output")),
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.getenv("PDF2ANKI_LLM_MODEL", "gpt-4o-mini"),
        help="LLM model for narrative conversion",
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
        default=os.getenv("PDF2ANKI_AUDIOBOOK_LANGUAGE", "cs"),
        help="Language for narrative (default: cs)",
    )
    parser.add_argument(
        "--style",
        type=str,
        default=os.getenv("PDF2ANKI_AUDIOBOOK_STYLE", "educational"),
        choices=["formal", "conversational", "educational"],
        help="Narrative style (default: educational)",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    chunks_path = args.chunks or (args.output_dir / "audiobook" / "chunks.json")
    pages_path = args.pages or (args.output_dir / "audiobook" / "pages.jsonl")
    
    process_audiobook_chunks(
        chunks_path=chunks_path,
        pages_path=pages_path,
        output_dir=args.output_dir,
        model=args.llm_model,
        base_url=args.base_url,
        api_key=args.api_key,
        language=args.language,
        style=args.style,
    )


if __name__ == "__main__":
    main()

