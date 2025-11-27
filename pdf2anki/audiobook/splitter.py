"""
Stage 1: Split pages into chunks of approximately target token size.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


def load_enhanced_pages(pages_path: Path) -> List[Dict[str, Any]]:
    """Load enhanced pages from pages.jsonl."""
    pages = []
    if not pages_path.exists():
        logger.error(f"Pages file not found: {pages_path}")
        return pages
    
    with pages_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pages.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid page line: {e}")
    
    return pages


def split_into_chunks(
    pages: List[Dict[str, Any]],
    chunk_size: int = 50000,
    overlap: int = 0,
) -> List[Dict[str, Any]]:
    """
    Split pages into chunks of approximately chunk_size tokens.
    
    Args:
        pages: List of enhanced page dictionaries
        chunk_size: Target tokens per chunk
        overlap: Token overlap between chunks (not implemented yet)
    
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    current_chunk: Dict[str, Any] = {
        "pages": [],
        "images": [],
        "token_count": 0,
    }
    
    for page in pages:
        page_num = page["page_number"]
        page_tokens = page["token_count"]
        
        # If adding this page would exceed chunk_size and we have pages already,
        # save current chunk and start a new one
        if (
            current_chunk["token_count"] + page_tokens > chunk_size
            and current_chunk["pages"]
        ):
            # Finalize current chunk
            chunk_id = f"chunk_{len(chunks) + 1:03d}"
            current_chunk["chunk_id"] = chunk_id
            current_chunk["page_range"] = [
                min(current_chunk["pages"]),
                max(current_chunk["pages"]),
            ]
            # Get text preview (first 200 chars from first page)
            first_page_idx = current_chunk["pages"][0] - 1
            if 0 <= first_page_idx < len(pages):
                preview_text = pages[first_page_idx]["text"][:200]
                current_chunk["text_preview"] = preview_text.replace("\n", " ")
            
            chunks.append(current_chunk)
            
            # Start new chunk
            current_chunk = {
                "pages": [],
                "images": [],
                "token_count": 0,
            }
        
        # Add page to current chunk
        current_chunk["pages"].append(page_num)
        current_chunk["token_count"] += page_tokens
        
        # Add images from this page
        for img in page.get("images", []):
            current_chunk["images"].append(img["image_id"])
    
    # Add final chunk if it has pages
    if current_chunk["pages"]:
        chunk_id = f"chunk_{len(chunks) + 1:03d}"
        current_chunk["chunk_id"] = chunk_id
        current_chunk["page_range"] = [
            min(current_chunk["pages"]),
            max(current_chunk["pages"]),
        ]
        first_page_idx = current_chunk["pages"][0] - 1
        if 0 <= first_page_idx < len(pages):
            preview_text = pages[first_page_idx]["text"][:200]
            current_chunk["text_preview"] = preview_text.replace("\n", " ")
        chunks.append(current_chunk)
    
    return chunks


def split_audiobook_text(
    pages_path: Path,
    output_path: Path,
    chunk_size: int = 50000,
    overlap: int = 0,
) -> None:
    """Main function to split pages into chunks."""
    logger.info(f"Loading enhanced pages from {pages_path}")
    pages = load_enhanced_pages(pages_path)
    
    if not pages:
        logger.error("No pages loaded. Run prepare_audiobook_pages first.")
        return
    
    logger.info(f"Loaded {len(pages)} pages")
    total_tokens = sum(p["token_count"] for p in pages)
    logger.info(f"Total tokens: {total_tokens:,}")
    
    logger.info(f"Splitting into chunks of ~{chunk_size:,} tokens...")
    chunks = split_into_chunks(pages, chunk_size=chunk_size, overlap=overlap)
    
    logger.info(f"Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks, 1):
        logger.info(
            f"  Chunk {i}: pages {chunk['page_range'][0]}-{chunk['page_range'][1]}, "
            f"{chunk['token_count']:,} tokens, {len(chunk['images'])} images"
        )
    
    # Save chunks
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "chunks": chunks,
        "total_chunks": len(chunks),
        "total_tokens": total_tokens,
        "chunk_size_target": chunk_size,
        "overlap": overlap,
    }
    
    logger.info(f"Saving chunks to {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info("✓ Chunking complete")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Split pages into chunks for audiobook processing"
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
        "--output",
        type=Path,
        help="Output path for chunks.json (default: <output-dir>/audiobook/chunks.json)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("PDF2ANKI_AUDIOBOOK_CHUNK_SIZE", "50000")),
        help="Target tokens per chunk (default: %(default)s)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=int(os.getenv("PDF2ANKI_AUDIOBOOK_OVERLAP", "0")),
        help="Token overlap between chunks (default: %(default)s)",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    pages_path = args.pages or (args.output_dir / "audiobook" / "pages.jsonl")
    output_path = args.output or (args.output_dir / "audiobook" / "chunks.json")
    
    split_audiobook_text(
        pages_path=pages_path,
        output_path=output_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()

