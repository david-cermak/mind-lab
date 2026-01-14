#!/bin/bash
# Automated PDF to Audiobook Pipeline
# Processes all PDFs in a directory through the complete audiobook generation workflow

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Exit if an uninitialized variable is used
set -o pipefail  # Exit if any command in a pipeline fails

# Default configuration
PDFS_DIR="${PDFS_DIR:-pdfs}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-output}"
CHUNK_SIZE="${CHUNK_SIZE:-50000}"
LANGUAGE="${LANGUAGE:-cs}"
STYLE="${STYLE:-educational}"
LANGUAGE_CODE="${LANGUAGE_CODE:-cs-CZ}"
SSML_GENDER="${SSML_GENDER:-NEUTRAL}"
SKIP_OCR="${SKIP_OCR:-true}"
COMBINE_FORMAT="${COMBINE_FORMAT:-single}"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pdfs-dir)
            PDFS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        --style)
            STYLE="$2"
            shift 2
            ;;
        --language-code)
            LANGUAGE_CODE="$2"
            shift 2
            ;;
        --ssml-gender)
            SSML_GENDER="$2"
            shift 2
            ;;
        --no-skip-ocr)
            SKIP_OCR="false"
            shift
            ;;
        --combine-format)
            COMBINE_FORMAT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Processes all PDFs in a directory through the complete audiobook generation workflow."
            echo ""
            echo "Options:"
            echo "  --pdfs-dir DIR          Directory containing PDF files (default: pdfs)"
            echo "  --output-dir DIR        Base output directory (default: output)"
            echo "  --chunk-size SIZE      Target tokens per chunk (default: 50000)"
            echo "  --language LANG         Target language (default: cs)"
            echo "  --style STYLE           Narrative style: formal, conversational, or educational (default: educational)"
            echo "  --language-code CODE    TTS language code (default: cs-CZ)"
            echo "  --ssml-gender GENDER    Voice gender: NEUTRAL, MALE, or FEMALE (default: NEUTRAL)"
            echo "  --no-skip-ocr           Don't skip OCR (default: skip OCR)"
            echo "  --combine-format FMT    Combine format: single, chapters, or all (default: single)"
            echo "  --help                  Show this help message"
            echo ""
            echo "Environment variables (can override defaults):"
            echo "  PDFS_DIR, OUTPUT_BASE_DIR, CHUNK_SIZE, LANGUAGE, STYLE"
            echo "  LANGUAGE_CODE, SSML_GENDER, SKIP_OCR, COMBINE_FORMAT"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if PDFs directory exists
if [[ ! -d "$PDFS_DIR" ]]; then
    echo "Error: PDFs directory '$PDFS_DIR' does not exist"
    exit 1
fi

# Find all PDF files
PDF_FILES=("$PDFS_DIR"/*.pdf)
if [[ ! -e "${PDF_FILES[0]}" ]]; then
    echo "Error: No PDF files found in '$PDFS_DIR'"
    exit 1
fi

echo "=========================================="
echo "PDF to Audiobook Batch Processor"
echo "=========================================="
echo "PDFs directory: $PDFS_DIR"
echo "Output base directory: $OUTPUT_BASE_DIR"
echo "Found ${#PDF_FILES[@]} PDF file(s)"
echo ""

# Process each PDF
for PDF_FILE in "${PDF_FILES[@]}"; do
    PDF_NAME=$(basename "$PDF_FILE" .pdf)
    OUTPUT_DIR="$OUTPUT_BASE_DIR/$PDF_NAME"
    
    echo "=========================================="
    echo "Processing: $PDF_NAME"
    echo "Output directory: $OUTPUT_DIR"
    echo "=========================================="
    
    # Step 1: Extract PDF
    echo "[1/8] Extracting PDF..."
    SKIP_OCR_FLAG=""
    if [[ "$SKIP_OCR" == "true" ]]; then
        SKIP_OCR_FLAG="--skip-ocr"
    fi
    python -m pdf2anki.main "$PDF_FILE" --output-dir "$OUTPUT_DIR" $SKIP_OCR_FLAG || {
        echo "Error: PDF extraction failed for $PDF_NAME"
        exit 1
    }
    
    # Step 2: Generate image cards (to get descriptions)
    echo "[2/8] Generating image occlusion cards..."
    python -m pdf2anki.generate_image_occlusion_cards --output-dir "$OUTPUT_DIR" || {
        echo "Error: Image occlusion card generation failed for $PDF_NAME"
        exit 1
    }
    
    echo "[2/8] Generating image-only cards..."
    python -m pdf2anki.generate_image_only_cards --output-dir "$OUTPUT_DIR" || {
        echo "Error: Image-only card generation failed for $PDF_NAME"
        exit 1
    }
    
    # Step 3: Stage 0 - Prepare pages (reuses descriptions from cards)
    echo "[3/8] Preparing pages..."
    python -m pdf2anki.audiobook.prepare_pages --output-dir "$OUTPUT_DIR" || {
        echo "Error: Page preparation failed for $PDF_NAME"
        exit 1
    }
    
    # Step 4: Stage 1 - Split into chunks
    echo "[4/8] Splitting into chunks..."
    python -m pdf2anki.audiobook.splitter --output-dir "$OUTPUT_DIR" --chunk-size "$CHUNK_SIZE" || {
        echo "Error: Chunk splitting failed for $PDF_NAME"
        exit 1
    }
    
    # Step 5: Stage 2 - Process chunks
    echo "[5/8] Processing chunks..."
    python -m pdf2anki.audiobook.processor \
        --output-dir "$OUTPUT_DIR" \
        --language "$LANGUAGE" \
        --style "$STYLE" || {
        echo "Error: Chunk processing failed for $PDF_NAME"
        exit 1
    }
    
    # Step 6: Stage 3 - Combine into single file (optional)
    if [[ "$COMBINE_FORMAT" != "none" ]]; then
        echo "[6/8] Combining narratives..."
        python -m pdf2anki.audiobook.combiner \
            --output-dir "$OUTPUT_DIR" \
            --output-format "$COMBINE_FORMAT" || {
            echo "Error: Narrative combination failed for $PDF_NAME"
            exit 1
        }
    else
        echo "[6/8] Skipping combination (format: none)"
    fi
    
    # Step 7: Stage 4 - Convert to audio
    echo "[7/8] Converting to audio..."
    python -m pdf2anki.audiobook.tts \
        --output-dir "$OUTPUT_DIR" \
        --language-code "$LANGUAGE_CODE" \
        --ssml-gender "$SSML_GENDER" || {
        echo "Error: TTS conversion failed for $PDF_NAME"
        exit 1
    }
    
    # Step 8: Rename MP3 files to include PDF name prefix
    echo "[8/8] Renaming audio files with PDF prefix..."
    AUDIO_DIR="$OUTPUT_DIR/audiobook/audio"
    if [[ -d "$AUDIO_DIR" ]]; then
        RENAMED_COUNT=0
        # Use nullglob to handle case where no files match
        shopt -s nullglob
        for MP3_FILE in "$AUDIO_DIR"/chunk_*.mp3; do
            BASENAME=$(basename "$MP3_FILE")
            # Skip if already renamed (contains PDF name prefix)
            if [[ "$BASENAME" == "${PDF_NAME}_"* ]]; then
                continue
            fi
            NEW_NAME="${PDF_NAME}_${BASENAME}"
            NEW_PATH="$AUDIO_DIR/$NEW_NAME"
            mv "$MP3_FILE" "$NEW_PATH" || {
                echo "Warning: Failed to rename $MP3_FILE"
            }
            RENAMED_COUNT=$((RENAMED_COUNT + 1))
        done
        shopt -u nullglob
        if [[ $RENAMED_COUNT -gt 0 ]]; then
            echo "  Renamed $RENAMED_COUNT audio file(s) with prefix '$PDF_NAME'"
        else
            echo "  No audio files found to rename"
        fi
    else
        echo "  Warning: Audio directory not found: $AUDIO_DIR"
    fi
    
    echo ""
    echo "✓ Successfully processed $PDF_NAME"
    echo "  Output: $OUTPUT_DIR/audiobook/"
    echo ""
done

echo "=========================================="
echo "All PDFs processed successfully!"
echo "=========================================="
