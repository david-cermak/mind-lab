"""Send a single image to an OpenAI vision model and print a description."""

import argparse
import base64
from pathlib import Path
from typing import Optional

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe an image using the gpt-4o vision-capable model."
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to the image (PNG/JPEG/etc.) that should be described.",
    )
    parser.add_argument(
        "--prompt",
        default="Popiš stručně obsah obrázku a vypiš klíčové prvky.",
        help="Instruction text sent alongside the image (default: Czech summary prompt).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Vision-capable OpenAI model to use (default: %(default)s).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=400,
        help="Maximum tokens in the response (default: %(default)s).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional custom API base URL (e.g., LiteLLM proxy).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional override for OPENAI_API_KEY",
    )
    return parser.parse_args()


def _image_to_data_url(image_path: Path) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    suffix = image_path.suffix.lower().lstrip(".") or "png"
    mime = f"image/{'jpeg' if suffix in {'jpg', 'jpeg'} else suffix}"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def describe_image(
    image_path: Path,
    prompt: str,
    model: str,
    max_tokens: int,
    base_url: Optional[str],
    api_key: Optional[str],
) -> str:
    client = OpenAI(base_url=base_url, api_key=api_key)
    image_url = _image_to_data_url(image_path)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who describes botanical diagrams in Czech and English.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        max_tokens=max_tokens,
    )

    message = response.choices[0].message
    content = message.content
    if isinstance(content, str):
        return content.strip()

    parts = []
    for item in content or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "\n".join(filter(None, parts)).strip()


def main() -> None:
    args = parse_args()
    description = describe_image(
        image_path=args.image_path.expanduser().resolve(),
        prompt=args.prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    print(description)


if __name__ == "__main__":
    main()

