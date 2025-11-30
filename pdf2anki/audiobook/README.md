# Audiobook Generation Pipeline

Convert PDF textbooks into continuous narrative text suitable for text-to-speech (TTS) conversion. This pipeline handles structured text (lists, bullet points) and integrates image descriptions naturally into the narrative flow.

## Overview

The audiobook pipeline consists of 5 stages:

1. **Stage 0: Prepare Pages** - Enhance page data with image descriptions
2. **Stage 1: Split Text** - Divide pages into manageable chunks (~50K tokens)
3. **Stage 2: Process Chunks** - Convert structured text to continuous narrative
4. **Stage 3: Combine** (Optional) - Merge narratives into final output files
5. **Stage 4: Text-to-Speech** - Convert narrative text to audio files

## Prerequisites

Before running the audiobook pipeline, you need:

1. **PDF extraction completed** - Run the main PDF extraction pipeline first:
   ```bash
   python -m pdf2anki.main pdfs/book.pdf --output-dir output
   ```

2. **Image descriptions available** (optional but recommended):
   - Run `generate_image_occlusion_cards` or `generate_image_only_cards` first
   - Descriptions will be automatically reused from `occlusion_cards.json` or `image_cards.json`
   - If not available, Stage 0 will generate them using vision LLM

## Quick Start

```bash
# Stage 0: Prepare pages with image descriptions
python -m pdf2anki.audiobook.prepare_pages --output-dir output

# Stage 1: Split into chunks
python -m pdf2anki.audiobook.splitter --output-dir output

# Stage 2: Process chunks into narrative
python -m pdf2anki.audiobook.processor --output-dir output --language cs --style educational

# Stage 3: Combine (optional)
python -m pdf2anki.audiobook.combiner --output-dir output --output-format single

# Stage 4: Convert to audio
python -m pdf2anki.audiobook.tts --output-dir output --language-code cs-CZ
```

## Stage-by-Stage Guide

### Stage 0: Prepare Pages (`prepare_pages.py`)

**Purpose**: Collect page text and generate image descriptions if needed.

**What it does**:
- Loads `metadata.jsonl` from PDF extraction
- Groups pages by page number
- Reuses image descriptions from existing card JSON files (`occlusion_cards.json`, `image_cards.json`)
- Generates new descriptions for images without existing ones (via vision LLM)
- Calculates token counts per page (text + image descriptions)
- Outputs `audiobook/pages.jsonl` with enhanced page data

**Usage**:
```bash
python -m pdf2anki.audiobook.prepare_pages \
  --output-dir output \
  --vision-model gpt-4o \
  --language cs \
  --force-regenerate  # Optional: regenerate all descriptions
```

**Output**: `output/audiobook/pages.jsonl`

**Example page entry**:
```json
{
  "page_number": 3,
  "text": "Full page text...",
  "images": [
    {
      "image_id": "page3_img0_xref23",
      "image_path": "images/page3_img0_xref23.png",
      "description": "Diagram shows the anatomical structure..."
    }
  ],
  "token_count": 1234,
  "has_diagrams": true
}
```

**Notes**:
- Descriptions are automatically reused from existing Anki card JSON files
- Only images without descriptions will trigger vision LLM calls
- Image descriptions appear at the end of each page (position unknown in PDF)

---

### Stage 1: Split Text (`splitter.py`)

**Purpose**: Divide pages into chunks of approximately target token size.

**What it does**:
- Loads enhanced pages from Stage 0
- Groups pages into chunks respecting page boundaries (no mid-page splits)
- Includes image descriptions in token count
- Tracks which pages and images belong to each chunk
- Outputs `audiobook/chunks.json` with chunk definitions

**Usage**:
```bash
python -m pdf2anki.audiobook.splitter \
  --output-dir output \
  --chunk-size 50000 \
  --overlap 0
```

**Parameters**:
- `--chunk-size`: Target tokens per chunk (default: 50000)
- `--overlap`: Token overlap between chunks (default: 0, not yet implemented)

**Output**: `output/audiobook/chunks.json`

**Example chunk entry**:
```json
{
  "chunk_id": "chunk_001",
  "page_range": [1, 5],
  "images": ["page1_img0_xref14", "page3_img0_xref23"],
  "token_count": 48750,
  "text_preview": "First 200 chars..."
}
```

**Notes**:
- Chunks always split at page boundaries (never mid-page)
- Chunk size is approximate - actual size may vary slightly
- Each chunk tracks which images it contains

---

### Stage 2: Process Chunks (`processor.py`)

**Purpose**: Convert structured text chunks into continuous narrative suitable for TTS.

**What it does**:
- Loads chunks and pages from previous stages
- Builds chunk text (page text + image descriptions at end of each page)
- Sends each chunk to LLM with instructions to:
  - Convert lists/bullets → flowing prose
  - Integrate image descriptions naturally
  - Maintain educational tone
  - Create smooth transitions
- Saves processed narratives as `audiobook/narratives/chunk_XXX.txt`
- Outputs metadata with processing stats

**Usage**:
```bash
python -m pdf2anki.audiobook.processor \
  --output-dir output \
  --llm-model gpt-4o-mini \
  --language cs \
  --style educational
```

**Parameters**:
- `--llm-model`: Model for narrative conversion (default: gpt-4o-mini)
- `--language`: Target language (default: cs)
- `--style`: Narrative style - `formal`, `conversational`, or `educational` (default: educational)

**Output**:
- `output/audiobook/narratives/chunk_001.txt` - Processed narrative per chunk
- `output/audiobook/narratives_metadata.json` - Processing metadata

**LLM Prompt Features**:
- Converts structured text (lists, bullets) into flowing prose
- Integrates image descriptions naturally (they appear at end of page)
- Maintains technical accuracy
- Creates natural transitions between topics
- Removes `[Image: ...]` markers after integration

**Notes**:
- Image descriptions are placed at the end of each page's text
- LLM integrates them where they make contextual sense
- Temperature automatically adjusts for models that require temperature=1 (e.g., gpt-5)

---

### Stage 3: Combine Narratives (`combiner.py`)

**Purpose**: (Optional) Combine processed chunks into single file or chapter files.

**What it does**:
- Loads all processed narratives
- Combines them based on output format:
  - `single`: One combined file
  - `chapters`: Separate chapter files (chapter_001.txt, chapter_002.txt, ...)
  - `all`: Keep individual chunk files (default)

**Usage**:
```bash
python -m pdf2anki.audiobook.combiner \
  --output-dir output \
  --output-format single
```

**Parameters**:
- `--output-format`: `single`, `chapters`, or `all` (default: all)

**Output**:
- `single`: `output/audiobook/full_narrative.txt`
- `chapters`: `output/audiobook/chapter_001.txt`, `chapter_002.txt`, ...
- `all`: Keep `output/audiobook/narratives/chunk_XXX.txt` files

---

### Stage 4: Text-to-Speech (`tts.py`)

**Purpose**: Convert narrative text files to audio using Google Cloud Text-to-Speech.

**What it does**:
- Loads processed narratives from Stage 2
- Splits long texts into chunks (Google TTS limit: 5000 chars per request)
- Synthesizes each chunk using Google Cloud TTS
- Combines audio chunks into complete MP3 files
- Saves audio files as `audiobook/audio/chunk_XXX.mp3`
- Outputs metadata with file sizes and processing info

**Prerequisites**:
- Google Cloud account with Text-to-Speech API enabled
- Service account credentials JSON file
- `google-cloud-texttospeech` package installed

**Setup**:
1. Enable Google Cloud Text-to-Speech API in your GCP project
2. Create a service account and download credentials JSON
3. Set environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
   ```
   Or pass via CLI: `--credentials /path/to/credentials.json`

**Usage**:
```bash
python -m pdf2anki.audiobook.tts \
  --output-dir output \
  --language-code cs-CZ \
  --ssml-gender NEUTRAL \
  --audio-encoding MP3
```

**Parameters**:
- `--language-code`: Language code (default: cs-CZ)
- `--voice-name`: Specific voice name (optional, e.g., "cs-CZ-Wavenet-A")
- `--ssml-gender`: Voice gender - `NEUTRAL`, `MALE`, or `FEMALE` (default: NEUTRAL)
- `--audio-encoding`: Audio format - `MP3`, `LINEAR16`, or `OGG_OPUS` (default: MP3)
- `--credentials`: Path to Google Cloud credentials JSON (optional if env var set)

**Output**:
- `output/audiobook/audio/chunk_001.mp3` - Audio file per chunk
- `output/audiobook/audio_metadata.json` - Processing metadata

**Example audio metadata entry**:
```json
{
  "chunk_id": "chunk_001",
  "success": true,
  "file": "audiobook/audio/chunk_001.mp3",
  "file_size_bytes": 5242880,
  "file_size_mb": 5.0,
  "text_chars": 45230,
  "tts_chunks": 10
}
```

**Notes**:
- Long texts are automatically split at sentence/paragraph boundaries
- Each chunk is synthesized separately and then combined
- Google TTS has a 5000 byte limit per request (not characters!)
- UTF-8 characters (especially Czech with diacritics) can be multi-byte, so the code uses byte length
- Uses a conservative 4500 byte limit per chunk to ensure we stay under the limit
- Audio files are saved as MP3 by default (can be changed to LINEAR16 or OGG_OPUS)

**Voice Selection**:
- If `--voice-name` is not specified, Google TTS will select a default voice for the language
- To list available voices for a language:
  ```python
  from google.cloud import texttospeech
  client = texttospeech.TextToSpeechClient()
  voices = client.list_voices(language_code="cs-CZ")
  for voice in voices.voices:
      print(f"{voice.name}: {voice.ssml_gender}")
  ```

---

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Audiobook Configuration
PDF2ANKI_AUDIOBOOK_CHUNK_SIZE=50000
PDF2ANKI_AUDIOBOOK_LANGUAGE=cs
PDF2ANKI_AUDIOBOOK_STYLE=educational
PDF2ANKI_AUDIOBOOK_OVERLAP=0
PDF2ANKI_TEMPERATURE=0.7  # LLM temperature (auto-adjusts for gpt-5)

# TTS Configuration
PDF2ANKI_TTS_LANGUAGE_CODE=cs-CZ
PDF2ANKI_TTS_VOICE_NAME=  # Optional: specific voice name
PDF2ANKI_TTS_GENDER=NEUTRAL  # NEUTRAL, MALE, or FEMALE
PDF2ANKI_TTS_ENCODING=MP3  # MP3, LINEAR16, or OGG_OPUS
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

### CLI Overrides

All environment variables can be overridden via CLI arguments. See `--help` for each command:

```bash
python -m pdf2anki.audiobook.prepare_pages --help
python -m pdf2anki.audiobook.splitter --help
python -m pdf2anki.audiobook.processor --help
python -m pdf2anki.audiobook.combiner --help
python -m pdf2anki.audiobook.tts --help
```

## Complete Example

```bash
# 1. Extract PDF (if not done already)
python -m pdf2anki.main pdfs/biology.pdf --output-dir output --skip-ocr

# 2. Generate image cards (to get descriptions)
python -m pdf2anki.generate_image_occlusion_cards --output-dir output
python -m pdf2anki.generate_image_only_cards --output-dir output

# 3. Stage 0: Prepare pages (reuses descriptions from cards)
python -m pdf2anki.audiobook.prepare_pages --output-dir output

# 4. Stage 1: Split into chunks
python -m pdf2anki.audiobook.splitter --output-dir output --chunk-size 50000

# 5. Stage 2: Process chunks
python -m pdf2anki.audiobook.processor \
  --output-dir output \
  --language cs \
  --style educational

# 6. Stage 3: Combine into single file (optional)
python -m pdf2anki.audiobook.combiner \
  --output-dir output \
  --output-format single

# 7. Stage 4: Convert to audio
python -m pdf2anki.audiobook.tts \
  --output-dir output \
  --language-code cs-CZ \
  --ssml-gender NEUTRAL

# Result: 
# - output/audiobook/full_narrative.txt (if Stage 3 run)
# - output/audiobook/audio/chunk_001.mp3, chunk_002.mp3, ...
```

## Output Format

### Narrative Text
The final narrative text is:
- **Plain text** (no markdown)
- **Paragraphs** separated by double newlines
- **Natural pauses** indicated by sentence structure
- **Image descriptions** integrated seamlessly

### Audio Files
After Stage 4, audio files are:
- **MP3 format** (or LINEAR16/OGG_OPUS if specified)
- **One file per chunk** (chunk_001.mp3, chunk_002.mp3, ...)
- **High quality** Google Cloud TTS synthesis
- **Ready for playback** in any audio player

## Troubleshooting

### Image Descriptions Not Found

If Stage 0 tries to generate descriptions but fails:
- Check that `OPENAI_API_KEY` and `PDF2ANKI_BASE_URL` are set correctly
- Verify vision model is available: `PDF2ANKI_VISION_MODEL=gpt-4o`
- Run image card generation first to create descriptions:
  ```bash
  python -m pdf2anki.generate_image_occlusion_cards --output-dir output
  ```

### Temperature Errors with gpt-5

If you see temperature errors:
- The code automatically retries with `temperature=1.0`
- Or set `PDF2ANKI_TEMPERATURE=1` in `.env`
- Or use a different model: `--llm-model gpt-4o-mini`

### Chunks Too Large/Small

Adjust chunk size:
```bash
python -m pdf2anki.audiobook.splitter --chunk-size 30000  # Smaller chunks
python -m pdf2anki.audiobook.splitter --chunk-size 80000  # Larger chunks
```

### Processing Fails

Check:
- LLM API key and base URL are correct
- Model name is valid for your API provider
- Chunks file exists: `output/audiobook/chunks.json`
- Pages file exists: `output/audiobook/pages.jsonl`

### TTS Synthesis Fails

If Stage 4 fails:
- Verify `GOOGLE_APPLICATION_CREDENTIALS` is set correctly
- Check that Google Cloud Text-to-Speech API is enabled in your project
- Ensure service account has Text-to-Speech permissions
- Verify language code is supported (e.g., "cs-CZ" for Czech)
- Check that `google-cloud-texttospeech` is installed:
  ```bash
  pip install google-cloud-texttospeech
  ```

## Integration with Existing Pipeline

The audiobook pipeline reuses:
- ✅ PDF extraction (`extractors/pdf.py`)
- ✅ OCR data (`extractors/ocr.py`) - for image context
- ✅ Vision LLM (`analysis/vision.py`) - for image descriptions
- ✅ Token estimation (`analysis/strategy.py`)
- ✅ Existing card JSON files - for description reuse

## File Structure

```
output/
├── metadata.jsonl              # From PDF extraction
├── occlusion_cards.json        # Reused for descriptions
├── image_cards.json            # Reused for descriptions
└── audiobook/
    ├── pages.jsonl             # Stage 0 output
    ├── chunks.json             # Stage 1 output
    ├── narratives/             # Stage 2 output
    │   ├── chunk_001.txt
    │   ├── chunk_002.txt
    │   └── ...
    ├── narratives_metadata.json
    ├── full_narrative.txt      # Stage 3 output (if single format)
    ├── audio/                   # Stage 4 output
    │   ├── chunk_001.mp3
    │   ├── chunk_002.mp3
    │   └── ...
    └── audio_metadata.json      # Stage 4 metadata
```

## Next Steps

After generating audio files, you can:

1. **Combine audio files** into a single audiobook using audio editing tools (ffmpeg, Audacity, etc.)

2. **Review and edit** the narrative text before TTS conversion (re-run Stage 2 if needed)

3. **Split into chapters** for better organization:
   ```bash
   python -m pdf2anki.audiobook.combiner --output-format chapters
   ```

4. **Use different TTS voices** by specifying `--voice-name` with a specific voice identifier

5. **Adjust audio quality** by changing `--audio-encoding` (MP3 for compatibility, LINEAR16 for highest quality)

