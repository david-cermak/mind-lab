"""
Utility CLI to preview text cards without opening the generated .apkg file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from pdf2anki.text_pipeline import ensure_plan, generate_text_cards_from_plan

logger = logging.getLogger("pdf2anki.debug_text_cards")
ENV_PATH = Path(__file__).parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print generated text cards for quick inspection.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("PDF2ANKI_OUTPUT_DIR", "output")),
        help="Directory with extracted assets (default: %(default)s)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional explicit path to metadata.jsonl",
    )
    parser.add_argument(
        "--ocr-dir",
        type=Path,
        help="Optional explicit OCR directory",
    )
    parser.add_argument(
        "--plan-path",
        type=Path,
        help="Optional plan.json path",
    )
    parser.add_argument(
        "--max-text-cards",
        type=int,
        default=int(os.getenv("PDF2ANKI_MAX_TEXT_CARDS", "10")),
        help="Limit the number of cards to preview",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.getenv("PDF2ANKI_LLM_MODEL", "gpt-4o-mini"),
        help="LLM model for text generation",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="Override OpenAI API key",
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
        help="Use an existing JSON dump of cards (default: <output-dir>/text_cards.json)",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Ignore cached JSON and regenerate cards",
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
    cards_json = (args.cards_json or (output_dir / "text_cards.json")).resolve()

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["PDF2ANKI_BASE_URL"] = args.base_url

    cards = None
    if cards_json.exists() and not args.regenerate:
        logger.info("Loading cached cards from %s", cards_json)
        cards = json.loads(cards_json.read_text(encoding="utf-8"))
    else:
        plan = ensure_plan(metadata_path, ocr_dir, plan_path, force_rebuild=args.force_plan, tokens_per_card=args.tokens_per_card)
        cards = generate_text_cards_from_plan(
            plan,
            output_dir,
            model=args.llm_model,
            base_url=args.base_url,
            api_key=args.api_key,
            max_cards=args.max_text_cards,
        )
        if cards:
            cards_json.parent.mkdir(parents=True, exist_ok=True)
            cards_json.write_text(json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Stored preview JSON at %s", cards_json)

    if not cards:
        logger.warning("No text cards ready to preview.")
        return

    print("=" * 80)
    print(f"Previewing {len(cards)} text cards")
    print("=" * 80)
    for idx, card in enumerate(cards, start=1):
        header = f"[{idx}] Page {card.get('page_number')} ({card.get('text_source')})"
        print(header)
        print("-" * len(header))
        print(f"Q: {card['front']}")
        print(f"A: {card['back']}")
        print()


if __name__ == "__main__":
    main()

