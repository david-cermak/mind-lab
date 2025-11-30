# Audiobook Generation Pipeline

## Overview

Converts PDF textbooks into continuous narrative text suitable for text-to-speech (TTS) conversion. Handles structured text (lists, itemization) and integrates image descriptions naturally into the narrative.

## Pipeline Stages

### Stage 0: Page + Image Artifacts Preparation
**Purpose**: Collect and enhance page data with image descriptions

**Inputs**:
- `metadata.jsonl` - Page text and image metadata
- `output/images/` - Extracted images
- `output/ocr/` - OCR data (optional, for image context)
- `output/plan.json` - Plan with image types (optional)

**Outputs**:
- `output/audiobook/pages.jsonl` - Enhanced page data with:
  ```json
  {
    "page_number": 1,
    "text": "Full page text...",
    "images": [
      {
        "image_id": "page1_img0_xref14",
        "image_path": "images/page1_img0_xref14.png",
        "description": "LLM-generated description",
        "position_hint": "top|middle|bottom"  // Where in page
      }
    ],
    "token_count": 1234,
    "has_diagrams": true
  }
  ```

**Process**:
1. Load metadata.jsonl
2. For each page with images:
   - If image description doesn't exist, call `vision.describe_image_only()`
   - Store description with image metadata
3. Estimate tokens per page (text + image descriptions)
4. Save enhanced pages

**CLI Target**: `python -m pdf2anki.prepare_audiobook_pages`

---

### Stage 1: Text Splitter
**Purpose**: Split pages into chunks of ~50K tokens each

**Inputs**:
- `output/audiobook/pages.jsonl` - Enhanced pages from Stage 0
- `--chunk-size` - Target tokens per chunk (default: 50000)
- `--overlap` - Token overlap between chunks (default: 0)

**Outputs**:
- `output/audiobook/chunks.json` - Chunk definitions:
  ```json
  {
    "chunks": [
      {
        "chunk_id": "chunk_001",
        "page_range": [1, 5],
        "image_ids": ["page1_img0_xref14", "page3_img0_xref23"],
        "token_count": 48750,
        "text_preview": "First 200 chars..."
      }
    ],
    "total_chunks": 3,
    "total_tokens": 145000
  }
  ```

**Process**:
1. Load enhanced pages
2. Group pages into chunks:
   - Start new chunk when adding next page would exceed chunk_size
   - Prefer splitting at page boundaries (don't break mid-page)
   - Include image descriptions in token count
   - Track which images belong to each chunk
3. Save chunk definitions

**CLI Target**: `python -m pdf2anki.split_audiobook_text`

**Algorithm**:
```python
chunks = []
current_chunk = {"pages": [], "tokens": 0, "images": []}

for page in pages:
    page_tokens = page["token_count"]
    
    if current_chunk["tokens"] + page_tokens > chunk_size and current_chunk["pages"]:
        # Save current chunk, start new one
        chunks.append(current_chunk)
        current_chunk = {"pages": [], "tokens": 0, "images": []}
    
    current_chunk["pages"].append(page["page_number"])
    current_chunk["tokens"] += page_tokens
    current_chunk["images"].extend(page["images"])

if current_chunk["pages"]:
    chunks.append(current_chunk)
```

---

### Stage 2: Chunk Processor
**Purpose**: Convert structured text chunks into continuous narrative

**Inputs**:
- `output/audiobook/chunks.json` - Chunk definitions
- `output/audiobook/pages.jsonl` - Enhanced pages
- `--language` - Target language (default: "cs")
- `--style` - Narrative style: "formal"|"conversational"|"educational" (default: "educational")

**Outputs**:
- `output/audiobook/narratives/chunk_001.txt` - Processed narrative text per chunk
- `output/audiobook/narratives_metadata.json` - Processing metadata:
  ```json
  {
    "chunks": [
      {
        "chunk_id": "chunk_001",
        "input_tokens": 48750,
        "output_tokens": 45230,
        "processing_time": 45.2,
        "file": "narratives/chunk_001.txt"
      }
    ]
  }
  ```

**Process**:
1. For each chunk:
   - Load pages in chunk
   - Combine text + image descriptions in order
   - Send to LLM with prompt to convert to narrative
   - Save processed narrative
   - Track metadata

**LLM Prompt Template**:
```
You are converting educational textbook content into a continuous narrative suitable for audiobook narration.

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
- Convert "• Item 1\n• Item 2" → "The first item is... Additionally, the second item..."
- Convert "1. First\n2. Second" → "First, we have... Second, we consider..."
- IMPORTANT: Image descriptions appear at the end of each page's text. You don't know the exact position 
  of images in the original PDF, so integrate them naturally where they make contextual sense in the narrative.
  For example, if a page discusses "plant tissues" and ends with "[Image: Diagram showing plant tissue structure]",
  integrate it as: "As illustrated in the accompanying diagram, plant tissues consist of..."
- Preserve technical terms and definitions exactly
- Maintain paragraph structure for natural pauses
- Create smooth transitions between pages and topics

Output: Continuous narrative text ready for TTS.
```

**CLI Target**: `python -m pdf2anki.process_audiobook_chunks`

---

### Stage 3: Combine Narratives (Optional)
**Purpose**: Combine all chunks into single file or multiple files

**Inputs**:
- `output/audiobook/narratives/` - All processed chunks
- `--output-format` - "single"|"chapters"|"all" (default: "all")

**Outputs**:
- `output/audiobook/full_narrative.txt` - Combined text (if single)
- `output/audiobook/chapter_N.txt` - Per-chunk files (if chapters)
- Or keep individual files (if all)

**CLI Target**: `python -m pdf2anki.combine_audiobook_narratives`

---

### Stage 4: Text-to-Speech
**Purpose**: Convert narrative text files to audio using Google Cloud TTS

**Inputs**:
- `output/audiobook/narratives/` - Processed narrative text files
- `output/audiobook/narratives_metadata.json` - Processing metadata
- `--language-code` - Language code (default: "cs-CZ")
- `--voice-name` - Specific voice name (optional)
- `--ssml-gender` - Voice gender: "NEUTRAL"|"MALE"|"FEMALE" (default: "NEUTRAL")
- `--audio-encoding` - Audio format: "MP3"|"LINEAR16"|"OGG_OPUS" (default: "MP3")

**Outputs**:
- `output/audiobook/audio/chunk_001.mp3` - Audio file per chunk
- `output/audiobook/audio_metadata.json` - Processing metadata:
  ```json
  {
    "chunks": [
      {
        "chunk_id": "chunk_001",
        "success": true,
        "file": "audiobook/audio/chunk_001.mp3",
        "file_size_bytes": 5242880,
        "file_size_mb": 5.0,
        "text_chars": 45230,
        "tts_chunks": 10
      }
    ]
  }
  ```

**Process**:
1. Load narrative files and metadata
2. For each narrative:
   - Split text into chunks based on byte length (Google TTS limit: 5000 bytes per request)
   - Uses conservative 4500 byte limit to account for UTF-8 encoding (Czech characters are multi-byte)
   - Synthesize each chunk using Google Cloud TTS
   - Combine audio chunks into single MP3 file
   - Save audio file and track metadata

**Prerequisites**:
- Google Cloud account with Text-to-Speech API enabled
- Service account credentials JSON file
- `GOOGLE_APPLICATION_CREDENTIALS` environment variable set

**CLI Target**: `python -m pdf2anki.audiobook.tts`

---

## Full Pipeline Command

```bash
# Stage 0: Prepare pages with image descriptions
python -m pdf2anki.prepare_audiobook_pages \
  --output-dir output \
  --vision-model gpt-4o

# Stage 1: Split into chunks
python -m pdf2anki.split_audiobook_text \
  --output-dir output \
  --chunk-size 50000

# Stage 2: Process chunks into narrative
python -m pdf2anki.process_audiobook_chunks \
  --output-dir output \
  --language cs \
  --style educational \
  --llm-model gpt-4o-mini

# Stage 3: Combine (optional)
python -m pdf2anki.combine_audiobook_narratives \
  --output-dir output \
  --output-format single

# Stage 4: Convert to audio
python -m pdf2anki.audiobook.tts \
  --output-dir output \
  --language-code cs-CZ \
  --ssml-gender NEUTRAL
```

## Integration with Existing Pipeline

The audiobook pipeline reuses:
- ✅ PDF extraction (`extractors/pdf.py`)
- ✅ OCR data (`extractors/ocr.py`)
- ✅ Vision LLM (`analysis/vision.py`)
- ✅ Token estimation (`analysis/strategy.py`)

New components needed:
- `audiobook/prepare_pages.py` - Stage 0
- `audiobook/splitter.py` - Stage 1
- `audiobook/processor.py` - Stage 2
- `audiobook/combiner.py` - Stage 3 (optional)
- `audiobook/tts.py` - Stage 4

## Configuration

Add to `.env`:
```bash
# Audiobook settings
PDF2ANKI_AUDIOBOOK_CHUNK_SIZE=50000
PDF2ANKI_AUDIOBOOK_LANGUAGE=cs
PDF2ANKI_AUDIOBOOK_STYLE=educational
PDF2ANKI_AUDIOBOOK_OVERLAP=0

# TTS settings
PDF2ANKI_TTS_LANGUAGE_CODE=cs-CZ
PDF2ANKI_TTS_VOICE_NAME=  # Optional
PDF2ANKI_TTS_GENDER=NEUTRAL
PDF2ANKI_TTS_ENCODING=MP3
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

## Output Format

### Narrative Text
Final narrative text should be:
- Plain text (no markdown)
- Paragraphs separated by double newlines
- Natural pauses indicated by sentence structure
- Image descriptions integrated seamlessly

### Audio Files
After Stage 4, audio files are:
- MP3 format (or LINEAR16/OGG_OPUS if specified)
- One file per chunk (chunk_001.mp3, chunk_002.mp3, ...)
- High quality Google Cloud TTS synthesis
- Ready for playback in any audio player

