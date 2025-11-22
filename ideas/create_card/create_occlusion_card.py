#!/usr/bin/env python3
"""Create an Anki image-occlusion card from OCR output."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


BBox = Tuple[int, int, int, int]


@dataclass
class OcrToken:
    index: int
    text: str
    bbox: BBox
    confidence: float = 0.0


@dataclass
class TokenGroup:
    tokens: List[OcrToken] = field(default_factory=list)
    bbox: BBox = (0, 0, 0, 0)

    def add(self, token: OcrToken) -> None:
        self.tokens.append(token)
        if len(self.tokens) == 1:
            self.bbox = token.bbox
        else:
            self.bbox = merge_bboxes(self.bbox, token.bbox)

    @property
    def label(self) -> str:
        return " ".join(tok.text for tok in self.tokens).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-json", required=True, help="Path to OCR JSON file.")
    parser.add_argument(
        "--image",
        help="Path to the source image. Falls back to path referenced in the OCR JSON.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination .apkg path for the generated card.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path to save intermediate occlusion metadata as JSON.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM refinement for text grouping.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="Vision-capable model name (default: %(default)s).",
    )
    parser.add_argument(
        "--llm-base-url",
        help="Optional custom OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--llm-api-key",
        help="Explicit API key (otherwise OPENAI_API_KEY is used).",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=800,
        help="Maximum tokens for the refinement response.",
    )
    parser.add_argument(
        "--proximity-threshold",
        type=float,
        default=90.0,
        help="Pixel distance threshold when grouping nearby tokens.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=8.0,
        help="Padding (in pixels) added around each occlusion box.",
    )
    parser.add_argument(
        "--deck-name",
        default="Image Occlusion Deck",
        help="Deck name to embed inside the .apkg.",
    )
    parser.add_argument(
        "--note-tag",
        default="image-occlusion",
        help="Tag to attach to the generated note.",
    )
    parser.add_argument(
        "--header-text",
        default="",
        help="Optional header text shown above the image.",
    )
    parser.add_argument(
        "--back-extra",
        default="",
        help="Optional text shown on the back of the card.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def load_ocr_tokens(
    path: Path, min_confidence: float
) -> Tuple[List[OcrToken], Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tokens: List[OcrToken] = []
    for raw in data.get("tokens", []):
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        bbox = raw.get("bbox") or raw.get("bounding_box") or raw.get("box")
        if not bbox or len(bbox) != 4:
            continue
        left, top, width, height = [int(float(v)) for v in bbox]
        confidence = float(raw.get("confidence", 0.0) or 0.0)
        if confidence < min_confidence:
            continue
        tokens.append(
            OcrToken(
                index=len(tokens),
                text=text,
                bbox=(left, top, width, height),
                confidence=confidence,
            )
        )
    if not tokens:
        raise ValueError(f"No usable tokens found in {path}")
    return tokens, data


def bbox_gap(a: BBox, b: BBox) -> float:
    left_a, top_a, width_a, height_a = a
    left_b, top_b, width_b, height_b = b
    right_a, bottom_a = left_a + width_a, top_a + height_a
    right_b, bottom_b = left_b + width_b, top_b + height_b

    if right_a < left_b:
        horizontal = left_b - right_a
    elif right_b < left_a:
        horizontal = left_a - right_b
    else:
        horizontal = 0

    if bottom_a < top_b:
        vertical = top_b - bottom_a
    elif bottom_b < top_a:
        vertical = top_a - bottom_b
    else:
        vertical = 0

    return math.hypot(horizontal, vertical)


def merge_bboxes(a: BBox, b: BBox) -> BBox:
    left_a, top_a, width_a, height_a = a
    left_b, top_b, width_b, height_b = b
    right_a, bottom_a = left_a + width_a, top_a + height_a
    right_b, bottom_b = left_b + width_b, top_b + height_b
    left = min(left_a, left_b)
    top = min(top_a, top_b)
    right = max(right_a, right_b)
    bottom = max(bottom_a, bottom_b)
    return (left, top, right - left, bottom - top)


def _aligned(a: BBox, b: BBox, tolerance: float = 0.6) -> bool:
    """Return True if boxes roughly share the same row or column."""
    left_a, top_a, width_a, height_a = a
    left_b, top_b, width_b, height_b = b
    center_a = (left_a + width_a / 2, top_a + height_a / 2)
    center_b = (left_b + width_b / 2, top_b + height_b / 2)
    avg_height = (height_a + height_b) / 2
    avg_width = (width_a + width_b) / 2
    same_row = abs(center_a[1] - center_b[1]) <= avg_height * tolerance
    same_col = abs(center_a[0] - center_b[0]) <= avg_width * tolerance
    return same_row or same_col


def group_tokens_spatially(tokens: Sequence[OcrToken], threshold: float) -> List[TokenGroup]:
    groups: List[TokenGroup] = []
    for token in sorted(tokens, key=lambda t: (t.bbox[1], t.bbox[0])):
        assigned = False
        for group in groups:
            gap = bbox_gap(group.bbox, token.bbox)
            if gap <= threshold or _aligned(group.bbox, token.bbox):
                group.add(token)
                assigned = True
                break
        if not assigned:
            new_group = TokenGroup()
            new_group.add(token)
            groups.append(new_group)
    logging.info("Grouped %d tokens into %d candidate regions", len(tokens), len(groups))
    return groups


def _image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower().lstrip(".") or "png"
    mime = f"image/{'jpeg' if suffix in {'jpg', 'jpeg'} else suffix}"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _extract_json_blob(text: str) -> str:
    text = text.strip()
    fenced = re.findall(r"```(?:json)?(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced[0].strip()
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    return match.group(0).strip() if match else text


def refine_groups_with_llm(
    groups: Sequence[TokenGroup],
    tokens: Sequence[OcrToken],
    image_path: Path,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    max_tokens: int,
) -> Sequence[TokenGroup]:
    if OpenAI is None:
        raise RuntimeError("openai package is not available. Install dependencies first.")

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set and --llm-api-key was not provided.")

    client = OpenAI(base_url=base_url, api_key=api_key)
    payload = [
        {
            "index": token.index,
            "text": token.text,
            "bbox": {
                "left": token.bbox[0],
                "top": token.bbox[1],
                "width": token.bbox[2],
                "height": token.bbox[3],
            },
        }
        for token in tokens
    ]
    instruction = (
        "You are grouping textual labels extracted from a botanical diagram. "
        "Tokens that describe the same structure should be in the same group. "
        "Return a JSON array where each entry contains `token_indices` (list of ints) "
        "and optional `label`. Only reference the provided token indices. "
        "Do not add explanations."
    )
    user_content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": instruction + "\n\nTokens:\n" + json.dumps(payload, ensure_ascii=False),
        }
    ]
    if image_path.exists():
        user_content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}})

    logging.info("Calling LLM (%s) for group refinement...", model)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a meticulous assistant who only outputs valid JSON.",
            },
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
    )
    message = response.choices[0].message
    content = message.content
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        content = "\n".join(text_parts)
    if not isinstance(content, str):
        logging.warning("Unexpected LLM response shape; using original groups.")
        return groups

    try:
        parsed = json.loads(_extract_json_blob(content))
    except json.JSONDecodeError as exc:
        logging.warning("Failed to parse LLM response (%s); falling back to spatial groups.", exc)
        return groups

    index_map = {token.index: token for token in tokens}
    llm_groups: List[TokenGroup] = []
    assigned: set[int] = set()
    for entry in parsed if isinstance(parsed, list) else []:
        indices = [idx for idx in entry.get("token_indices", []) if idx in index_map]
        if not indices:
            continue
        group = TokenGroup()
        for idx in indices:
            group.add(index_map[idx])
            assigned.add(idx)
        llm_groups.append(group)

    for token in tokens:
        if token.index not in assigned:
            solo = TokenGroup()
            solo.add(token)
            llm_groups.append(solo)

    if not llm_groups:
        logging.warning("LLM returned no groups; reverting to spatial grouping.")
        return groups

    logging.info("LLM refinement produced %d groups.", len(llm_groups))
    return llm_groups


def calculate_bounding_boxes(
    groups: Sequence[TokenGroup],
    padding: float,
    image_width: int,
    image_height: int,
) -> List[Dict[str, Any]]:
    rectangles: List[Dict[str, Any]] = []
    for idx, group in enumerate(groups, start=1):
        left, top, width, height = group.bbox
        padded_left = max(0, left - padding)
        padded_top = max(0, top - padding)
        padded_right = min(image_width, left + width + padding)
        padded_bottom = min(image_height, top + height + padding)
        bbox_pixels = [
            int(round(padded_left)),
            int(round(padded_top)),
            int(round(padded_right - padded_left)),
            int(round(padded_bottom - padded_top)),
        ]
        rectangles.append(
            {
                "index": idx,
                "label": group.label,
                "tokens": [token.text for token in group.tokens],
                "bbox_pixels": bbox_pixels,
            }
        )
    return rectangles


def normalize_coordinates(rectangles: Sequence[Dict[str, Any]], width: int, height: int) -> None:
    for rect in rectangles:
        left, top, box_width, box_height = rect["bbox_pixels"]
        rect["bbox_normalized"] = {
            "left": round(left / width, 4),
            "top": round(top / height, 4),
            "width": round(box_width / width, 4),
            "height": round(box_height / height, 4),
        }


def create_occlusion_markup(rectangles: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for idx, rect in enumerate(rectangles, start=1):
        norm = rect["bbox_normalized"]
        parts.append(
            f"{{{{c{idx}::image-occlusion:rect:left={norm['left']}:top={norm['top']}:width={norm['width']}:height={norm['height']}:oi=1}}}}"
        )
    return "<br>".join(parts)


def _checksum(field_text: str) -> int:
    import hashlib

    digest = hashlib.md5(field_text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _guid() -> str:
    return uuid.uuid4().hex[:10]


def _current_millis() -> int:
    return int(time.time() * 1000)


def create_image_occlusion_model(model_id: int) -> Dict[str, Any]:
    template = {
        "name": "Image Occlusion",
        "ord": 0,
        "qfmt": '{{#Header}}<div>{{Header}}</div>{{/Header}}\n<div style="display: none">{{cloze:Occlusion}}</div>\n<div id="err"></div>\n<div id="image-occlusion-container">\n    {{Image}}\n    <canvas id="image-occlusion-canvas"></canvas>\n</div>\n<script>\ntry {\n    anki.imageOcclusion.setup();\n} catch (exc) {\n    document.getElementById("err").innerHTML = `Error loading image occlusion. Is your Anki version up to date?<br><br>${exc}`;\n}\n</script>\n',
        "afmt": '{{#Header}}<div>{{Header}}</div>{{/Header}}\n<div style="display: none">{{cloze:Occlusion}}</div>\n<div id="err"></div>\n<div id="image-occlusion-container">\n    {{Image}}\n    <canvas id="image-occlusion-canvas"></canvas>\n</div>\n<script>\ntry {\n    anki.imageOcclusion.setup();\n} catch (exc) {\n    document.getElementById("err").innerHTML = `Error loading image occlusion. Is your Anki version up to date?<br><br>${exc}`;\n}\n</script>\n\n<div><button id="toggle">Toggle Masks</button></div>\n{{#Back Extra}}<div>{{Back Extra}}</div>{{/Back Extra}}\n',
        "did": None,
        "bqfmt": "",
        "bafmt": "",
        "bfont": "",
        "bsize": 0,
        "id": _current_millis(),
    }
    fields = [
        {"name": "Occlusion", "ord": 0, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": "", "plainText": False, "collapsed": False, "excludeFromSearch": False, "id": uuid.uuid4().int & 0xFFFFFFFFFFFFFFFF, "tag": 0, "preventDeletion": True},
        {"name": "Image", "ord": 1, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": "", "plainText": False, "collapsed": False, "excludeFromSearch": False, "id": uuid.uuid4().int & 0xFFFFFFFFFFFFFFFF, "tag": 1, "preventDeletion": True},
        {"name": "Header", "ord": 2, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": "", "plainText": False, "collapsed": False, "excludeFromSearch": False, "id": uuid.uuid4().int & 0xFFFFFFFFFFFFFFFF, "tag": 2, "preventDeletion": True},
        {"name": "Back Extra", "ord": 3, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": "", "plainText": False, "collapsed": False, "excludeFromSearch": False, "id": uuid.uuid4().int & 0xFFFFFFFFFFFFFFFF, "tag": 3, "preventDeletion": True},
        {"name": "Comments", "ord": 4, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": "", "plainText": False, "collapsed": False, "excludeFromSearch": False, "id": uuid.uuid4().int & 0xFFFFFFFFFFFFFFFF, "tag": 4, "preventDeletion": False},
    ]
    return {
        "id": model_id,
        "name": "Image Occlusion",
        "type": 1,
        "mod": int(time.time()),
        "usn": 0,
        "sortf": 0,
        "did": None,
        "tmpls": [template],
        "flds": fields,
        "css": "#image-occlusion-canvas {\n    --inactive-shape-color: #ffeba2;\n    --active-shape-color: #ff8e8e;\n    --inactive-shape-border: 1px #212121;\n    --active-shape-border: 1px #212121;\n    --highlight-shape-color: #ff8e8e00;\n    --highlight-shape-border: 1px #ff8e8e;\n}\n\n.card {\n    font-family: arial;\n    font-size: 20px;\n    text-align: center;\n    color: black;\n    background-color: white;\n}\n",
        "latexPre": "\\documentclass[12pt]{article}\n\\special{papersize=3in,5in}\n\\usepackage[utf8]{inputenc}\n\\usepackage{amssymb,amsmath}\n\\pagestyle{empty}\n\\setlength{\\parindent}{0in}\n\\begin{document}\n",
        "latexPost": "\\end{document}",
        "latexsvg": False,
        "req": [[0, "any", [0, 1, 2]]],
        "originalStockKind": 6,
    }


def create_apkg_file(
    occlusion_markup: str,
    image_path: Path,
    output_path: Path,
    image_filename: str,
    deck_name: str,
    header_text: str,
    back_extra: str,
    note_tag: str,
) -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="apkg_build_"))
    try:
        db_path = temp_dir / "collection.anki21"
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE col (
              id integer PRIMARY KEY,
              crt integer NOT NULL,
              mod integer NOT NULL,
              scm integer NOT NULL,
              ver integer NOT NULL,
              dty integer NOT NULL,
              usn integer NOT NULL,
              ls integer NOT NULL,
              conf text NOT NULL,
              models text NOT NULL,
              decks text NOT NULL,
              dconf text NOT NULL,
              tags text NOT NULL
            );
            CREATE TABLE notes (
              id integer PRIMARY KEY,
              guid text NOT NULL,
              mid integer NOT NULL,
              mod integer NOT NULL,
              usn integer NOT NULL,
              tags text NOT NULL,
              flds text NOT NULL,
              sfld integer NOT NULL,
              csum integer NOT NULL,
              flags integer NOT NULL,
              data text NOT NULL
            );
            CREATE TABLE cards (
              id integer PRIMARY KEY,
              nid integer NOT NULL,
              did integer NOT NULL,
              ord integer NOT NULL,
              mod integer NOT NULL,
              usn integer NOT NULL,
              type integer NOT NULL,
              queue integer NOT NULL,
              due integer NOT NULL,
              ivl integer NOT NULL,
              factor integer NOT NULL,
              reps integer NOT NULL,
              lapses integer NOT NULL,
              left integer NOT NULL,
              odue integer NOT NULL,
              odid integer NOT NULL,
              flags integer NOT NULL,
              data text NOT NULL
            );
            CREATE TABLE revlog (
              id integer PRIMARY KEY,
              cid integer NOT NULL,
              usn integer NOT NULL,
              ease integer NOT NULL,
              ivl integer NOT NULL,
              lastIvl integer NOT NULL,
              factor integer NOT NULL,
              time integer NOT NULL,
              type integer NOT NULL
            );
            CREATE TABLE graves (
              usn integer NOT NULL,
              oid integer NOT NULL,
              type integer NOT NULL
            );
            """
        )
        now = int(time.time())
        model_id = _current_millis()
        deck_id = 1
        conf = {
            "curDeck": deck_id,
            "dueCounts": True,
            "timeLim": 0,
            "newSpread": 0,
            "schedVer": 2,
            "nextPos": 1,
            "sched2021": True,
            "collapseTime": 1200,
            "creationOffset": 0,
            "estTimes": True,
            "addToCur": True,
            "curModel": model_id,
            "sortType": "noteFld",
            "dayLearnFirst": False,
            "activeDecks": [deck_id],
            "sortBackwards": False,
        }
        models = {str(model_id): create_image_occlusion_model(model_id)}
        decks = {
            str(deck_id): {
                "id": deck_id,
                "mod": now,
                "name": deck_name,
                "usn": 0,
                "lrnToday": [0, 0],
                "revToday": [0, 0],
                "newToday": [0, 0],
                "timeToday": [0, 0],
                "collapsed": False,
                "browserCollapsed": False,
                "desc": "",
                "dyn": 0,
                "conf": 1,
                "extendNew": 0,
                "extendRev": 0,
                "reviewLimit": None,
                "newLimit": None,
            }
        }
        dconf = {
            "1": {
                "id": 1,
                "mod": now,
                "name": "Default",
                "usn": 0,
                "maxTaken": 60,
                "autoplay": True,
                "timer": 0,
                "replayq": True,
                "new": {"bury": False, "delays": [1.0, 10.0], "initialFactor": 2500, "ints": [1, 4, 0], "order": 1, "perDay": 20},
                "rev": {"bury": False, "ease4": 1.3, "ivlFct": 1.0, "maxIvl": 36500, "perDay": 200, "hardFactor": 1.2},
                "lapse": {"delays": [10.0], "leechAction": 1, "leechFails": 8, "minInt": 1, "mult": 0.0},
                "dyn": False,
            }
        }
        tags = {}
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO col (id, crt, mod, scm, ver, dty, usn, ls, conf, models, decks, dconf, tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                now,
                _current_millis(),
                _current_millis(),
                11,
                0,
                0,
                0,
                json.dumps(conf),
                json.dumps(models),
                json.dumps(decks),
                json.dumps(dconf),
                json.dumps(tags),
            ),
        )
        note_id = _current_millis()
        card_id = note_id + 1
        occlusion_field = occlusion_markup
        image_field = f'<img src="{image_filename}">'
        fields = [
            occlusion_field,
            image_field,
            header_text or "",
            back_extra or "",
            "",
        ]
        field_blob = "\x1f".join(fields)
        cur.execute(
            "INSERT INTO notes (id, guid, mid, mod, usn, tags, flds, sfld, csum, flags, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                note_id,
                _guid(),
                model_id,
                now,
                0,
                f" {note_tag} " if note_tag else "",
                field_blob,
                occlusion_field,
                _checksum(occlusion_field),
                0,
                "",
            ),
        )
        cur.execute(
            "INSERT INTO cards (id, nid, did, ord, mod, usn, type, queue, due, ivl, factor, reps, lapses, left, odue, odid, flags, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                card_id,
                note_id,
                deck_id,
                0,
                now,
                0,
                0,
                0,
                now,
                0,
                2500,
                0,
                0,
                0,
                0,
                0,
                0,
                "",
            ),
        )
        conn.commit()
        conn.close()

        # Prepare media
        shutil.copy2(image_path, temp_dir / "0")
        (temp_dir / "media").write_text(json.dumps({"0": image_filename}, ensure_ascii=False), encoding="utf-8")
        (temp_dir / "meta").write_bytes(b"\x08\x02")

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as apkg:
            for name in ["collection.anki21", "media", "meta", "0"]:
                apkg.write(temp_dir / name, arcname=name)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def build_metadata_payload(
    image_path: Path,
    rectangles: Sequence[Dict[str, Any]],
    markup: str,
) -> Dict[str, Any]:
    with Image.open(image_path) as img:
        width, height = img.size
    return {
        "image_path": str(image_path),
        "image_dimensions": {"width": width, "height": height},
        "groups": rectangles,
        "anki_markup": markup,
    }


def resolve_image_path(cli_path: Optional[str], ocr_data: Dict[str, Any], ocr_json_path: Path) -> Path:
    candidates: List[Path] = []
    if cli_path:
        candidates.append(Path(cli_path))
    for key in ("image_path", "image", "imagePath"):
        value = ocr_data.get(key)
        if value:
            candidates.append(Path(value))
    for candidate in candidates:
        candidate = candidate.expanduser()
        if not candidate.is_absolute():
            candidate = (ocr_json_path.parent / candidate).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not resolve image path from CLI or OCR metadata.")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    ocr_path = Path(args.ocr_json).expanduser().resolve()
    tokens, ocr_data = load_ocr_tokens(ocr_path, min_confidence=75.0)
    image_path = resolve_image_path(args.image, ocr_data, ocr_path)
    with Image.open(image_path) as img:
        image_width, image_height = img.size

    groups = group_tokens_spatially(tokens, args.proximity_threshold)
    if args.use_llm:
        try:
            groups = list(
                refine_groups_with_llm(
                    groups,
                    tokens,
                    image_path,
                    args.llm_model,
                    args.llm_base_url,
                    args.llm_api_key,
                    args.llm_max_tokens,
                )
            )
        except Exception as exc:  # pragma: no cover
            logging.warning("LLM refinement failed (%s). Continuing with spatial groups.", exc)

    rectangles = calculate_bounding_boxes(groups, args.padding, image_width, image_height)
    normalize_coordinates(rectangles, image_width, image_height)
    occlusion_markup = create_occlusion_markup(rectangles)

    output_path = Path(args.output).expanduser().resolve()
    ensure_parent_dir(output_path)
    create_apkg_file(
        occlusion_markup=occlusion_markup,
        image_path=image_path,
        output_path=output_path,
        image_filename=image_path.name,
        deck_name=args.deck_name,
        header_text=args.header_text,
        back_extra=args.back_extra,
        note_tag=args.note_tag,
    )
    logging.info("Created .apkg at %s", output_path)

    if args.json_output:
        json_path = Path(args.json_output).expanduser().resolve()
        ensure_parent_dir(json_path)
        payload = build_metadata_payload(image_path, rectangles, occlusion_markup)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Wrote metadata JSON to %s", json_path)


if __name__ == "__main__":
    main()


