"""
CLI target that generates only image occlusion cards from an existing plan.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from pdf2anki.occlusion_card_pipeline import generate_occlusion_cards_from_plan
from pdf2anki.text_pipeline import ensure_plan
from pdf2anki.utils.anki_db import DeckBuilder

logger = logging.getLogger("pdf2anki.generate_image_occlusion_cards")
ENV_PATH = Path(__file__).parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an Anki deck that contains only image occlusion cards."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("PDF2ANKI_OUTPUT_DIR", "output")),
        help="Directory with extracted assets (default: %(default)s)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional explicit path to metadata.jsonl (defaults to <output-dir>/metadata.jsonl)",
    )
    parser.add_argument(
        "--ocr-dir",
        type=Path,
        help="Optional explicit OCR directory (defaults to <output-dir>/ocr)",
    )
    parser.add_argument(
        "--plan-path",
        type=Path,
        help="Optional plan.json path (defaults to <output-dir>/plan.json)",
    )
    parser.add_argument(
        "--final-apkg",
        type=Path,
        default=None,
        help="Where to store the generated Anki package (default: <output-dir>/occlusion_cards.apkg)",
    )
    parser.add_argument(
        "--deck-name",
        type=str,
        default="PDF Occlusion Cards",
        help="Deck name for generated occlusion cards (default: %(default)s)",
    )
    parser.add_argument(
        "--max-occlusion-cards",
        type=int,
        help="Optional limit on how many occlusion cards to generate",
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default=os.getenv("PDF2ANKI_VISION_MODEL", "gpt-4o"),
        help="Vision model for semantic grouping (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="Override OpenAI API key (falls back to env)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("PDF2ANKI_BASE_URL"),
        help="Override API base URL",
    )
    parser.add_argument(
        "--force-plan",
        action="store_true",
        help="Regenerate plan.json even if it already exists",
    )
    parser.add_argument(
        "--cards-json",
        type=Path,
        help="Optional path to dump generated cards as JSON (default: <output-dir>/occlusion_cards.json)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=75.0,
        help="Minimum OCR confidence for occlusion tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--disable-vision",
        action="store_true",
        help="Skip the vision LLM step and rely on spatial grouping only",
    )
    parser.add_argument(
        "--occlude-all",
        type=lambda x: x.lower() == "true",
        default=os.getenv("PDF2ANKI_OCCLUDE_ALL", "true").lower() == "true",
        help="Occlusion mode: true (oi=1, hide all + guess one) or false (oi=0, hide one + guess one)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    output_dir = args.output_dir.resolve()
    metadata_path = (args.metadata or (output_dir / "metadata.jsonl")).resolve()
    ocr_dir = (args.ocr_dir or (output_dir / "ocr")).resolve()
    plan_path = (args.plan_path or (output_dir / "plan.json")).resolve()

    final_apkg = args.final_apkg or (output_dir / "occlusion_cards.apkg")
    final_apkg = final_apkg.resolve()

    cards_json_path = args.cards_json or (output_dir / "occlusion_cards.json")
    cards_json_path = cards_json_path.resolve()

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["PDF2ANKI_BASE_URL"] = args.base_url

    logger.info("Loading or creating plan from %s", plan_path)
    plan = ensure_plan(
        metadata_path,
        ocr_dir,
        plan_path,
        force_rebuild=args.force_plan,
    )

    cards = generate_occlusion_cards_from_plan(
        plan,
        output_dir,
        metadata_path,
        ocr_dir,
        model=None if args.disable_vision else args.vision_model,
        occlude_all=args.occlude_all,
        base_url=args.base_url,
        api_key=args.api_key,
        min_confidence=args.min_confidence,
        max_cards=args.max_occlusion_cards,
        disable_vision=args.disable_vision,
    )

    if not cards:
        logger.warning("No occlusion cards were generated. Exiting without building a deck.")
        return

    logger.info("Generated %s occlusion cards. Writing deck to %s", len(cards), final_apkg)

    deck_builder = DeckBuilder(args.deck_name, final_apkg)
    for card in cards:
        image_path = Path(card["image_path"])
        if not image_path.is_absolute():
            image_path = output_dir / image_path
        deck_builder.add_occlusion_note(
            markup=card["markup"],
            image_path=image_path,
            header=card["header"],
            back_extra=card["back_extra"],
            tags=["pdf-import", "occlusion"],
        )
    deck_builder.build()

    cards_json_path.parent.mkdir(parents=True, exist_ok=True)
    cards_json_path.write_text(json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved occlusion card preview JSON to %s", cards_json_path)


if __name__ == "__main__":
    main()

