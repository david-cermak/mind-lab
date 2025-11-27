"""
Stage 3: Combine processed narratives into final output files.
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


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """Load narratives_metadata.json."""
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return {"chunks": []}
    
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def combine_narratives(
    narratives_dir: Path,
    metadata: Dict[str, Any],
    output_format: str = "all",
    output_dir: Path | None = None,
) -> None:
    """
    Combine narratives based on output_format.
    
    Args:
        narratives_dir: Directory containing narrative .txt files
        metadata: Metadata from narratives_metadata.json
        output_format: "single" | "chapters" | "all"
        output_dir: Output directory (for single/chapters modes)
    """
    chunks = metadata.get("chunks", [])
    successful_chunks = [c for c in chunks if c.get("success", True)]
    
    if not successful_chunks:
        logger.error("No successful chunks found")
        return
    
    logger.info(f"Found {len(successful_chunks)} successful narratives")
    
    if output_format == "all":
        logger.info("Keeping individual files (output_format=all)")
        return
    
    narratives = []
    for chunk_info in sorted(successful_chunks, key=lambda x: x["chunk_id"]):
        chunk_id = chunk_info["chunk_id"]
        narrative_path = narratives_dir / f"{chunk_id}.txt"
        
        if not narrative_path.exists():
            logger.warning(f"Narrative not found: {narrative_path}")
            continue
        
        narrative_text = narrative_path.read_text(encoding="utf-8").strip()
        narratives.append({
            "chunk_id": chunk_id,
            "text": narrative_text,
        })
        logger.info(f"Loaded {chunk_id}")
    
    if output_format == "single":
        # Combine into one file
        output_path = (output_dir or narratives_dir.parent) / "full_narrative.txt"
        combined_text = "\n\n".join(n["text"] for n in narratives)
        output_path.write_text(combined_text, encoding="utf-8")
        logger.info(f"✓ Combined {len(narratives)} chunks into {output_path}")
        logger.info(f"  Total length: {len(combined_text):,} characters")
    
    elif output_format == "chapters":
        # Save as chapter_N.txt files
        output_dir = output_dir or narratives_dir.parent
        for i, narrative in enumerate(narratives, 1):
            chapter_path = output_dir / f"chapter_{i:03d}.txt"
            chapter_path.write_text(narrative["text"], encoding="utf-8")
            logger.info(f"  Saved {chapter_path.name}")
        logger.info(f"✓ Saved {len(narratives)} chapter files")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Combine processed narratives into final output"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("PDF2ANKI_OUTPUT_DIR", "output")),
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="all",
        choices=["single", "chapters", "all"],
        help="Output format: single (one file), chapters (chapter_N.txt), all (keep individual) (default: all)",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    narratives_dir = args.output_dir / "audiobook" / "narratives"
    metadata_path = args.output_dir / "audiobook" / "narratives_metadata.json"
    
    if not narratives_dir.exists():
        logger.error(f"Narratives directory not found: {narratives_dir}")
        logger.error("Run process_audiobook_chunks first.")
        return
    
    metadata = load_metadata(metadata_path)
    combine_narratives(
        narratives_dir=narratives_dir,
        metadata=metadata,
        output_format=args.output_format,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

