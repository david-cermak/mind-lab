#!/usr/bin/env python3
"""
Convert LinkedIn-styled Unicode "fancy fonts" (Mathematical Alphanumeric Symbols,
fullwidth, etc.) into plain ASCII and wrap contiguous runs in Markdown emphasis.

Examples:
  𝗟𝗶𝗴𝗵𝘁𝘄𝗲𝗶𝗴𝗵𝘁 𝗠𝗤𝗧𝗧 𝗯𝗿𝗼𝗸𝗲𝗿 𝗿𝘂𝗻𝗻𝗶𝗻𝗴 𝗼𝗻 𝗘𝗦𝗣𝟯𝟮
  -> *Lightweight MQTT broker running on ESP32*

Usage:
  - Read from stdin and write to stdout:
      python scripts/linkedin_unicode_to_markdown.py
  - Convert a file to stdout:
      python scripts/linkedin_unicode_to_markdown.py -i /path/to/input.md
  - Convert a file in-place:
      python scripts/linkedin_unicode_to_markdown.py -i /path/to/input.md --in-place
  - Choose emphasis style (italic|bold|none):
      python scripts/linkedin_unicode_to_markdown.py -i input.md -o output.md --style italic
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, Iterable, Tuple


def _build_linear_range_map(start_cp: int, count: int, ascii_start_char: str) -> Dict[str, str]:
    """Map a contiguous Unicode range to ASCII starting at ascii_start_char."""
    base = ord(ascii_start_char)
    return {chr(start_cp + i): chr(base + i) for i in range(count)}


def _merge_dicts(dicts: Iterable[Dict[str, str]]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for d in dicts:
        merged.update(d)
    return merged


def _build_style_mapping() -> Dict[str, str]:
    """
    Build a mapping of common styled alphabets/digits to ASCII.
    Covers contiguous blocks (avoids the sparse Script/Fraktur letter gaps).
    """
    ranges: Tuple[Dict[str, str], ...] = (
        # Mathematical Bold
        _build_linear_range_map(0x1D400, 26, "A"),
        _build_linear_range_map(0x1D41A, 26, "a"),
        _build_linear_range_map(0x1D7CE, 10, "0"),
        # Mathematical Italic
        _build_linear_range_map(0x1D434, 26, "A"),
        _build_linear_range_map(0x1D44E, 26, "a"),
        # Mathematical Bold Italic
        _build_linear_range_map(0x1D468, 26, "A"),
        _build_linear_range_map(0x1D482, 26, "a"),
        # Mathematical Sans-Serif
        _build_linear_range_map(0x1D5A0, 26, "A"),
        _build_linear_range_map(0x1D5BA, 26, "a"),
        _build_linear_range_map(0x1D7E2, 10, "0"),
        # Mathematical Sans-Serif Bold
        _build_linear_range_map(0x1D5D4, 26, "A"),
        _build_linear_range_map(0x1D5EE, 26, "a"),
        _build_linear_range_map(0x1D7EC, 10, "0"),
        # Mathematical Sans-Serif Italic
        _build_linear_range_map(0x1D608, 26, "A"),
        _build_linear_range_map(0x1D622, 26, "a"),
        # Mathematical Sans-Serif Bold Italic
        _build_linear_range_map(0x1D63C, 26, "A"),
        _build_linear_range_map(0x1D656, 26, "a"),
        # Mathematical Monospace
        _build_linear_range_map(0x1D670, 26, "A"),
        _build_linear_range_map(0x1D68A, 26, "a"),
        _build_linear_range_map(0x1D7F6, 10, "0"),
        # Mathematical Double-Struck digits
        _build_linear_range_map(0x1D7D8, 10, "0"),
        # Fullwidth ASCII letters/digits (often used for "wide" look)
        _build_linear_range_map(0xFF21, 26, "A"),
        _build_linear_range_map(0xFF41, 26, "a"),
        _build_linear_range_map(0xFF10, 10, "0"),
    )
    return _merge_dicts(ranges)


STYLED_TO_ASCII_MAP: Dict[str, str] = _build_style_mapping()


def is_styled_char(ch: str) -> bool:
    """True if the character is one we know how to de-style."""
    return ch in STYLED_TO_ASCII_MAP


def map_char(ch: str) -> str:
    """Map a styled character to ASCII if possible; else return original."""
    return STYLED_TO_ASCII_MAP.get(ch, ch)


def normalize_and_emphasize(text: str, style: str = "italic") -> str:
    """
    Convert styled characters to ASCII and wrap contiguous sequences with Markdown emphasis.
    - style: 'italic' -> *text*, 'bold' -> **text**, 'none' -> no wrapping (only normalize)
    Heuristics:
      - A "run" starts at the first styled character.
      - Runs remain open across spaces and simple punctuation until a non-styled
        alphanumeric appears or a newline boundary is reached.
    """
    if style not in {"italic", "bold", "none"}:
        raise ValueError("style must be one of: italic, bold, none")

    if style == "none":
        return "".join(map_char(c) for c in text)

    open_mark, close_mark = ("*", "*") if style == "italic" else ("**", "**")

    def is_boundary_char(ch: str) -> bool:
        # Characters that can appear inside a styled run without forcing closure
        # (spaces and lightweight punctuation commonly present in titles)
        return ch.isspace() or ch in "-–—,:;.!?'\"()[]/{}/\\"

    out: list[str] = []
    in_run = False

    for ch in text:
        if ch == "\n":
            if in_run:
                out.append(close_mark)
                in_run = False
            out.append(ch)
            continue

        if is_styled_char(ch):
            if not in_run:
                out.append(open_mark)
                in_run = True
            out.append(map_char(ch))
            continue

        if in_run:
            if is_boundary_char(ch):
                # Keep the run open across boundaries like spaces/punctuation
                out.append(ch)
                continue
            else:
                # Close run before non-styled alphanumeric or other symbols
                out.append(close_mark)
                in_run = False

        # Append character (either we weren't in a run, or we just closed it)
        out.append(ch)

    if in_run:
        out.append(close_mark)

    return "".join(out)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize LinkedIn-styled Unicode letters/digits to ASCII and wrap with Markdown emphasis."
    )
    parser.add_argument(
        "-i",
        "--input",
        metavar="PATH",
        default="-",
        help="Input file path (or '-' for stdin).",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default="-",
        help="Output file path (or '-' for stdout). Ignored when --in-place is used.",
    )
    parser.add_argument(
        "--style",
        choices=["italic", "bold", "none"],
        default="italic",
        help="Emphasis style: italic (default), bold, or none (only normalize).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write the converted content back to the input file.",
    )
    return parser.parse_args(list(argv))


def _read_text(path: str) -> str:
    if path == "-" or path is None:
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: str, content: str) -> None:
    if path == "-" or path is None:
        sys.stdout.write(content)
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.in_place and (args.input == "-" or args.input is None):
        print("Error: --in-place requires a real input file path.", file=sys.stderr)
        return 2

    original = _read_text(args.input)
    converted = normalize_and_emphasize(original, style=args.style)

    if args.in_place:
        _write_text(args.input, converted)
    else:
        _write_text(args.output, converted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


