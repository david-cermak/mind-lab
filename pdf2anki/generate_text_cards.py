"""
CLI target that runs only the text-card portion of the pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from pdf2anki.text_pipeline import ensure_plan, generate_text_cards_from_plan
from pdf2anki.utils.anki_db import DeckBuilder

logger = logging.getLogger("pdf2anki.generate_text_cards")
ENV_PATH = Path(__file__).parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an Anki deck that contains only text cards."
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
        help="Where to store the generated Anki package (default: <output-dir>/text_cards.apkg)",
    )
    parser.add_argument(
        "--deck-name",
        type=str,
        default="PDF Text Cards",
        help="Deck name for generated text cards (default: %(default)s)",
    )
    parser.add_argument(
        "--max-text-cards",
        type=int,
        default=int(os.getenv("PDF2ANKI_MAX_TEXT_CARDS", "10")),
        help="Safety limit for total text cards (default: %(default)s)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.getenv("PDF2ANKI_LLM_MODEL", "gpt-4o-mini"),
        help="LLM model for text generation (default: %(default)s)",
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
        help="Optional path to dump generated cards as JSON (default: <output-dir>/text_cards.json)",
    )
    parser.add_argument(
        "--tokens-per-card",
        type=int,
        default=int(os.getenv("PDF2ANKI_TOKENS_PER_CARD", "300")),
        help="Target tokens per card (lower = denser cards, default: 300)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    output_dir = args.output_dir.resolve()
    metadata_path = (args.metadata or (output_dir / "metadata.jsonl")).resolve()
    ocr_dir = (args.ocr_dir or (output_dir / "ocr")).resolve()
    plan_path = (args.plan_path or (output_dir / "plan.json")).resolve()

    final_apkg = args.final_apkg or (output_dir / "text_cards.apkg")
    final_apkg = final_apkg.resolve()

    cards_json_path = args.cards_json or (output_dir / "text_cards.json")
    cards_json_path = cards_json_path.resolve()

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["PDF2ANKI_BASE_URL"] = args.base_url

    logger.info("Loading or creating plan from %s", plan_path)
    plan = ensure_plan(metadata_path, ocr_dir, plan_path, force_rebuild=args.force_plan, tokens_per_card=args.tokens_per_card)

    cards = generate_text_cards_from_plan(
        plan,
        output_dir,
        model=args.llm_model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_cards=args.max_text_cards,
    )

    if not cards:
        logger.warning("No text cards were generated. Exiting without building a deck.")
        return

    logger.info("Generated %s text cards. Writing deck to %s", len(cards), final_apkg)

    deck_builder = DeckBuilder(args.deck_name, final_apkg)
    for card in cards:
        deck_builder.add_text_note(card["front"], card["back"], tags=["pdf-import", "text"])
    deck_builder.build()

    cards_json_path.parent.mkdir(parents=True, exist_ok=True)
    cards_json_path.write_text(json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved text card preview JSON to %s", cards_json_path)


if __name__ == "__main__":
    main()

