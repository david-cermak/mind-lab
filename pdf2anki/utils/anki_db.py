"""
Anki deck builder utility.
Combines text and image notes into a single .apkg file.
"""

import json
import shutil
import sqlite3
import time
import uuid
import zipfile
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Schema templates
BASIC_MODEL_ID = 1757653544726
OCCLUSION_MODEL_ID = 1757653544731


def _checksum(field_text: str) -> int:
    import hashlib
    digest = hashlib.md5(field_text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _current_millis() -> int:
    return int(time.time() * 1000)


def _guid() -> str:
    return uuid.uuid4().hex[:10]


def get_basic_model_template(model_id: int) -> Dict[str, Any]:
    return {
        "id": model_id,
        "name": "Basic",
        "type": 0,
        "mod": int(time.time()),
        "usn": 0,
        "sortf": 0,
        "did": None,
        "tmpls": [{
            "name": "Card 1",
            "ord": 0,
            "qfmt": "{{Front}}",
            "afmt": "{{FrontSide}}\n\n<hr id=answer>\n\n{{Back}}",
            "bqfmt": "", "bafmt": "", "did": None, "bfont": "", "bsize": 0, "id": _current_millis()
        }],
        "flds": [
            {"name": "Front", "ord": 0, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": "", "plainText": False, "collapsed": False, "excludeFromSearch": False, "id": _current_millis() + 1, "tag": None, "preventDeletion": False},
            {"name": "Back", "ord": 1, "sticky": False, "rtl": False, "font": "Arial", "size": 20, "description": "", "plainText": False, "collapsed": False, "excludeFromSearch": False, "id": _current_millis() + 2, "tag": None, "preventDeletion": False}
        ],
        "css": ".card {\n    font-family: arial;\n    font-size: 20px;\n    line-height: 1.5;\n    text-align: center;\n    color: black;\n    background-color: white;\n}\n",
        "latexPre": "\\documentclass[12pt]{article}\n\\special{papersize=3in,5in}\n\\usepackage[utf8]{inputenc}\n\\usepackage{amssymb,amsmath}\n\\pagestyle{empty}\n\\setlength{\\parindent}{0in}\n\\begin{document}\n",
        "latexPost": "\\end{document}",
        "latexsvg": False,
        "req": [[0, "any", [0]]],
        "originalStockKind": 1,
    }


def get_occlusion_model_template(model_id: int) -> Dict[str, Any]:
    # Same template as in occlusion.py, reproduced here for self-containment
    # (Ideally imported from shared utils, but keeping simple for now)
    template = {
        "name": "Image Occlusion",
        "ord": 0,
        "qfmt": '{{#Header}}<div>{{Header}}</div>{{/Header}}\n<div style="display: none">{{cloze:Occlusion}}</div>\n<div id="err"></div>\n<div id="image-occlusion-container">\n    {{Image}}\n    <canvas id="image-occlusion-canvas"></canvas>\n</div>\n<script>\ntry {\n    anki.imageOcclusion.setup();\n} catch (exc) {\n    document.getElementById("err").innerHTML = `Error loading image occlusion. Is your Anki version up to date?<br><br>${exc}`;\n}\n</script>\n',
        "afmt": '{{#Header}}<div>{{Header}}</div>{{/Header}}\n<div style="display: none">{{cloze:Occlusion}}</div>\n<div id="err"></div>\n<div id="image-occlusion-container">\n    {{Image}}\n    <canvas id="image-occlusion-canvas"></canvas>\n</div>\n<script>\ntry {\n    anki.imageOcclusion.setup();\n} catch (exc) {\n    document.getElementById("err").innerHTML = `Error loading image occlusion. Is your Anki version up to date?<br><br>${exc}`;\n}\n</script>\n\n<div><button id="toggle">Toggle Masks</button></div>\n{{#Back Extra}}<div>{{Back Extra}}</div>{{/Back Extra}}\n',
        "did": None,
        "bqfmt": "", "bafmt": "", "bfont": "", "bsize": 0, "id": _current_millis()
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


class DeckBuilder:
    def __init__(self, deck_name: str, output_path: Path):
        self.deck_name = deck_name
        self.output_path = output_path
        self.notes = []
        self.media_files = {} # map "filename" -> "source_path"
        
    def add_text_note(self, front: str, back: str, tags: Optional[List[str]] = None):
        self.notes.append({
            "type": "basic",
            "fields": [front, back],
            "tags": tags or []
        })
        
    def add_occlusion_note(self, 
                           markup: str, 
                           image_path: Path, 
                           header: str = "", 
                           back_extra: str = "", 
                           tags: Optional[List[str]] = None):
        
        # Ensure filename is unique in media dict, simplistic approach
        filename = image_path.name
        if filename in self.media_files and self.media_files[filename] != image_path:
            # Name collision with different content, uniqueify
            stem = image_path.stem
            suffix = image_path.suffix
            filename = f"{stem}_{uuid.uuid4().hex[:6]}{suffix}"
            
        self.media_files[filename] = image_path
        
        self.notes.append({
            "type": "occlusion",
            "fields": [
                markup,
                f'<img src="{filename}">',
                header,
                back_extra,
                "" # Comments
            ],
            "tags": tags or ["image-occlusion"]
        })

    def build(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="apkg_build_"))
        try:
            # 1. Create DB
            db_path = temp_dir / "collection.anki21"
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            
            # Create tables (schema from original script)
            cur.executescript("""
                CREATE TABLE col (id integer PRIMARY KEY, crt integer NOT NULL, mod integer NOT NULL, scm integer NOT NULL, ver integer NOT NULL, dty integer NOT NULL, usn integer NOT NULL, ls integer NOT NULL, conf text NOT NULL, models text NOT NULL, decks text NOT NULL, dconf text NOT NULL, tags text NOT NULL);
                CREATE TABLE notes (id integer PRIMARY KEY, guid text NOT NULL, mid integer NOT NULL, mod integer NOT NULL, usn integer NOT NULL, tags text NOT NULL, flds text NOT NULL, sfld integer NOT NULL, csum integer NOT NULL, flags integer NOT NULL, data text NOT NULL);
                CREATE TABLE cards (id integer PRIMARY KEY, nid integer NOT NULL, did integer NOT NULL, ord integer NOT NULL, mod integer NOT NULL, usn integer NOT NULL, type integer NOT NULL, queue integer NOT NULL, due integer NOT NULL, ivl integer NOT NULL, factor integer NOT NULL, reps integer NOT NULL, lapses integer NOT NULL, left integer NOT NULL, odue integer NOT NULL, odid integer NOT NULL, flags integer NOT NULL, data text NOT NULL);
                CREATE TABLE revlog (id integer PRIMARY KEY, cid integer NOT NULL, usn integer NOT NULL, ease integer NOT NULL, ivl integer NOT NULL, lastIvl integer NOT NULL, factor integer NOT NULL, time integer NOT NULL, type integer NOT NULL);
                CREATE TABLE graves (usn integer NOT NULL, oid integer NOT NULL, type integer NOT NULL);
            """)
            
            # 2. Setup collection config
            now = int(time.time())
            deck_id = 1
            
            # Initialize Models
            basic_model_id = _current_millis()
            occlusion_model_id = basic_model_id + 1000 # prevent collision
            
            models = {
                str(basic_model_id): get_basic_model_template(basic_model_id),
                str(occlusion_model_id): get_occlusion_model_template(occlusion_model_id)
            }
            
            decks = {
                str(deck_id): {
                    "id": deck_id, "mod": now, "name": self.deck_name, "usn": 0,
                    "lrnToday": [0, 0], "revToday": [0, 0], "newToday": [0, 0],
                    "timeToday": [0, 0], "collapsed": False, "browserCollapsed": False,
                    "desc": "", "dyn": 0, "conf": 1, "extendNew": 0, "extendRev": 0,
                    "reviewLimit": None, "newLimit": None,
                }
            }
            
            dconf = {"1": {"id": 1, "mod": now, "name": "Default", "usn": 0, "maxTaken": 60, "autoplay": True, "timer": 0, "replayq": True, "new": {"bury": False, "delays": [1.0, 10.0], "initialFactor": 2500, "ints": [1, 4, 0], "order": 1, "perDay": 20}, "rev": {"bury": False, "ease4": 1.3, "ivlFct": 1.0, "maxIvl": 36500, "perDay": 200, "hardFactor": 1.2}, "lapse": {"delays": [10.0], "leechAction": 1, "leechFails": 8, "minInt": 1, "mult": 0.0}, "dyn": False}}
            
            cur.execute(
                "INSERT INTO col VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (1, now, now, now, 11, 0, 0, 0, json.dumps({}), json.dumps(models), json.dumps(decks), json.dumps(dconf), "{}")
            )
            
            # 3. Insert Notes & Cards
            for i, note in enumerate(self.notes):
                note_id = _current_millis() + i * 10
                guid = _guid()
                
                if note["type"] == "basic":
                    mid = basic_model_id
                else:
                    mid = occlusion_model_id
                
                flds = "\x1f".join(note["fields"])
                tags_str = " " + " ".join(note["tags"]) + " "
                
                cur.execute(
                    "INSERT INTO notes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (note_id, guid, mid, now, 0, tags_str, flds, note["fields"][0], _checksum(note["fields"][0]), 0, "")
                )
                
                # One card per note (simplification - IO creates 1 note with masks baked in)
                cur.execute(
                    "INSERT INTO cards VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (note_id + 1, note_id, deck_id, 0, now, 0, 0, 0, now, 0, 2500, 0, 0, 0, 0, 0, 0, "")
                )
            
            conn.commit()
            conn.close()
            
            # 4. Handle Media
            media_map = {}
            for idx, (filename, src_path) in enumerate(self.media_files.items()):
                media_id = str(idx)
                media_map[media_id] = filename
                shutil.copy2(src_path, temp_dir / media_id)
            
            (temp_dir / "media").write_text(json.dumps(media_map), encoding="utf-8")
            (temp_dir / "meta").write_bytes(b"\x08\x02")
            
            # 5. Zip it up
            with zipfile.ZipFile(self.output_path, "w", compression=zipfile.ZIP_DEFLATED) as apkg:
                apkg.write(temp_dir / "collection.anki21", arcname="collection.anki21")
                apkg.write(temp_dir / "media", arcname="media")
                apkg.write(temp_dir / "meta", arcname="meta")
                for media_id in media_map:
                    apkg.write(temp_dir / media_id, arcname=media_id)
                    
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

