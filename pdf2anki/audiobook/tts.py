"""
Stage 4: Convert narrative text to speech using Google Cloud TTS.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

try:
    from google.cloud import texttospeech
except ImportError:
    texttospeech = None

logger = logging.getLogger(__name__)

# Load .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# Google TTS has a limit of 5000 bytes per request (not characters!)
# Using 4500 bytes as a safe limit to account for encoding overhead
MAX_BYTES_PER_REQUEST = 4500

# Neural2 voices have a much lower character limit (~500-600 chars)
# Using 500 characters as a conservative limit for Neural2 voices
MAX_CHARS_PER_REQUEST = 500


def get_byte_length(text: str) -> int:
    """Get UTF-8 byte length of text."""
    return len(text.encode("utf-8"))


def load_metadata(metadata_path: Path) -> Dict[str, Any]:
    """Load narratives_metadata.json."""
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return {"chunks": []}
    
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_text_for_tts(
    text: str, 
    max_bytes: int = MAX_BYTES_PER_REQUEST,
    max_chars: int = MAX_CHARS_PER_REQUEST
) -> List[str]:
    """
    Split text into chunks suitable for TTS based on byte length and character length.
    Tries to split at sentence boundaries to avoid cutting mid-sentence.
    
    Note: Google TTS limit is 5000 bytes, but Neural2 voices have a lower character
    limit (~500-600 chars). UTF-8 characters (especially Czech with diacritics) can be multi-byte.
    """
    text_bytes = get_byte_length(text)
    text_chars = len(text)
    if text_bytes <= max_bytes and text_chars <= max_chars:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_bytes = get_byte_length(para)
        para_chars = len(para)
        current_bytes = get_byte_length(current_chunk) if current_chunk else 0
        current_chars = len(current_chunk) if current_chunk else 0
        separator_bytes = get_byte_length("\n\n")
        separator_chars = len("\n\n")
        
        # If adding this paragraph would exceed either limit
        if current_chunk and (
            current_bytes + separator_bytes + para_bytes > max_bytes or
            current_chars + separator_chars + para_chars > max_chars
        ):
            # Save current chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        
        # If single paragraph exceeds limit, split by sentences
        while get_byte_length(current_chunk) > max_bytes or len(current_chunk) > max_chars:
            sentences = current_chunk.split(". ")
            temp_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Add period if it was removed by split
                if not sentence.endswith(".") and not sentence.endswith("!") and not sentence.endswith("?"):
                    sentence += "."
                
                sentence_bytes = get_byte_length(sentence)
                sentence_chars = len(sentence)
                temp_bytes = get_byte_length(temp_chunk) if temp_chunk else 0
                temp_chars = len(temp_chunk) if temp_chunk else 0
                space_bytes = get_byte_length(" ")
                space_chars = len(" ")
                
                # If single sentence exceeds limit, split by commas
                if sentence_chars > max_chars or sentence_bytes > max_bytes:
                    # Split this long sentence by commas
                    parts = sentence.split(", ")
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        # Add comma if it was removed (except for last part)
                        if part != parts[-1] and not part.endswith(","):
                            part += ","
                        
                        part_bytes = get_byte_length(part)
                        part_chars = len(part)
                        temp_bytes = get_byte_length(temp_chunk) if temp_chunk else 0
                        temp_chars = len(temp_chunk) if temp_chunk else 0
                        space_bytes = get_byte_length(" ")
                        space_chars = len(" ")
                        
                        if temp_chunk and (
                            temp_bytes + space_bytes + part_bytes > max_bytes or
                            temp_chars + space_chars + part_chars > max_chars
                        ):
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = part
                        else:
                            if temp_chunk:
                                temp_chunk += " " + part
                            else:
                                temp_chunk = part
                    
                    # If still too long after comma split, force split at character limit
                    if temp_chunk and (len(temp_chunk) > max_chars or get_byte_length(temp_chunk) > max_bytes):
                        while len(temp_chunk) > max_chars or get_byte_length(temp_chunk) > max_bytes:
                            # Find a good split point (space or punctuation near the limit)
                            split_pos = max_chars
                            for i in range(max_chars - 50, max_chars):
                                if i < len(temp_chunk) and temp_chunk[i] in [' ', ',', '.', ';']:
                                    split_pos = i + 1
                                    break
                            
                            chunks.append(temp_chunk[:split_pos].strip())
                            temp_chunk = temp_chunk[split_pos:].strip()
                    continue
                
                if temp_chunk and (
                    temp_bytes + space_bytes + sentence_bytes > max_bytes or
                    temp_chars + space_chars + sentence_chars > max_chars
                ):
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
                    temp_chunk = sentence
                else:
                    if temp_chunk:
                        temp_chunk += " " + sentence
                    else:
                        temp_chunk = sentence
            
            current_chunk = temp_chunk
            # If still too long after sentence splitting, force split to avoid infinite loop
            if get_byte_length(current_chunk) > max_bytes or len(current_chunk) > max_chars:
                # Force split at character limit
                while len(current_chunk) > max_chars or get_byte_length(current_chunk) > max_bytes:
                    # Find a good split point (space or punctuation near the limit)
                    split_pos = max_chars
                    for i in range(max(max_chars - 100, 0), max_chars):
                        if i < len(current_chunk) and current_chunk[i] in [' ', ',', '.', ';', '\n']:
                            split_pos = i + 1
                            break
                    
                    chunks.append(current_chunk[:split_pos].strip())
                    current_chunk = current_chunk[split_pos:].strip()
                break
            else:
                break
    
    # Add remaining chunk, but ensure it doesn't exceed limits
    if current_chunk:
        # Final safety check - if still too long, force split
        while len(current_chunk) > max_chars or get_byte_length(current_chunk) > max_bytes:
            split_pos = max_chars
            # Try to find a good split point
            for i in range(max_chars - 50, max_chars):
                if i < len(current_chunk) and current_chunk[i] in [' ', ',', '.', ';', '\n']:
                    split_pos = i + 1
                    break
            
            chunks.append(current_chunk[:split_pos].strip())
            current_chunk = current_chunk[split_pos:].strip()
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    # Final validation: ensure no chunk exceeds limits (safety check)
    validated_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        
        while len(chunk) > max_chars or get_byte_length(chunk) > max_bytes:
            # Force split if somehow a chunk still exceeds limits
            split_pos = max_chars
            for i in range(max(max_chars - 100, 0), max_chars):
                if i < len(chunk) and chunk[i] in [' ', ',', '.', ';', '\n']:
                    split_pos = i + 1
                    break
            
            validated_chunks.append(chunk[:split_pos].strip())
            chunk = chunk[split_pos:].strip()
        
        if chunk:
            validated_chunks.append(chunk)
    
    return validated_chunks


def synthesize_chunk(
    text: str,
    client: Any,
    language_code: str,
    voice_name: str | None,
    ssml_gender: str = "NEUTRAL",
    audio_encoding: str = "MP3",
) -> bytes:
    """
    Synthesize a single text chunk to audio.
    
    Args:
        text: Text to synthesize
        client: Google Cloud TTS client
        language_code: Language code (e.g., "cs-CZ")
        voice_name: Specific voice name (optional)
        ssml_gender: Voice gender (NEUTRAL, MALE, FEMALE)
        audio_encoding: Audio encoding (MP3, LINEAR16, etc.)
    
    Returns:
        Audio content as bytes
    """
    if texttospeech is None:
        raise RuntimeError("google-cloud-texttospeech package not installed")
    
    # Set the text input
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    # Build voice selection
    voice_params = texttospeech.VoiceSelectionParams(
        language_code=language_code,
    )
    
    if voice_name:
        voice_params.name = voice_name
    
    # Map string gender to enum
    gender_map = {
        "NEUTRAL": texttospeech.SsmlVoiceGender.NEUTRAL,
        "MALE": texttospeech.SsmlVoiceGender.MALE,
        "FEMALE": texttospeech.SsmlVoiceGender.FEMALE,
    }
    voice_params.ssml_gender = gender_map.get(ssml_gender.upper(), texttospeech.SsmlVoiceGender.NEUTRAL)
    
    # Audio configuration
    audio_config = texttospeech.AudioConfig(
        audio_encoding=getattr(texttospeech.AudioEncoding, audio_encoding.upper())
    )
    
    # Perform synthesis
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice_params,
        audio_config=audio_config,
    )
    
    return response.audio_content


def synthesize_narrative(
    narrative_path: Path,
    output_path: Path,
    client: Any,
    language_code: str,
    voice_name: str | None,
    ssml_gender: str,
    audio_encoding: str,
) -> Dict[str, Any]:
    """
    Synthesize a complete narrative file to audio.
    
    Returns:
        Metadata dict with processing info
    """
    logger.info(f"  Reading narrative: {narrative_path.name}")
    text = narrative_path.read_text(encoding="utf-8").strip()
    
    if not text:
        logger.warning(f"  Empty narrative file: {narrative_path}")
        return {"success": False, "error": "Empty text"}
    
    # Split text into chunks
    text_chunks = split_text_for_tts(text)
    logger.info(f"  Split into {len(text_chunks)} TTS chunks")
    
    # Synthesize each chunk
    audio_chunks = []
    total_chars = 0
    total_bytes = 0
    
    for i, chunk_text in enumerate(text_chunks, 1):
        chunk_chars = len(chunk_text)
        chunk_bytes = get_byte_length(chunk_text)
        total_chars += chunk_chars
        total_bytes += chunk_bytes
        logger.info(f"    Synthesizing chunk {i}/{len(text_chunks)} ({chunk_chars:,} chars, {chunk_bytes:,} bytes)...")
        
        try:
            audio_data = synthesize_chunk(
                text=chunk_text,
                client=client,
                language_code=language_code,
                voice_name=voice_name,
                ssml_gender=ssml_gender,
                audio_encoding=audio_encoding,
            )
            audio_chunks.append(audio_data)
        except Exception as e:
            chunk_chars = len(chunk_text)
            chunk_bytes = get_byte_length(chunk_text)
            logger.error(f"    Error synthesizing chunk {i}: {e}")
            logger.error(f"    Chunk {i} details: {chunk_chars:,} chars, {chunk_bytes:,} bytes")
            logger.error(f"    First 200 chars of failed chunk: {chunk_text[:200]!r}")
            return {
                "success": False,
                "error": str(e),
                "chunks_processed": i - 1,
                "failed_chunk": i,
                "failed_chunk_chars": chunk_chars,
                "failed_chunk_bytes": chunk_bytes,
            }
    
    # Combine audio chunks
    logger.info(f"  Combining {len(audio_chunks)} audio chunks...")
    combined_audio = b"".join(audio_chunks)
    
    # Save audio file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(combined_audio)
    
    file_size_mb = len(combined_audio) / (1024 * 1024)
    logger.info(f"  ✓ Saved audio: {output_path.name} ({file_size_mb:.2f} MB)")
    
    return {
        "success": True,
        "file": str(output_path.relative_to(output_path.parent.parent.parent)),
        "file_size_bytes": len(combined_audio),
        "file_size_mb": round(file_size_mb, 2),
        "text_chars": total_chars,
        "text_bytes": total_bytes,
        "tts_chunks": len(text_chunks),
    }


def synthesize_audiobook(
    narratives_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    language_code: str,
    voice_name: str | None,
    ssml_gender: str,
    audio_encoding: str,
    credentials_path: str | None,
) -> None:
    """
    Main function to synthesize all narratives to audio.
    """
    if texttospeech is None:
        logger.error("google-cloud-texttospeech package not installed")
        logger.error("Install it with: pip install google-cloud-texttospeech")
        return
    
    # Initialize TTS client
    try:
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        client = texttospeech.TextToSpeechClient()
    except Exception as e:
        logger.error(f"Failed to initialize Google Cloud TTS client: {e}")
        logger.error("Make sure GOOGLE_APPLICATION_CREDENTIALS is set or credentials are configured")
        return
    
    # Load metadata
    logger.info(f"Loading metadata from {metadata_path}")
    metadata = load_metadata(metadata_path)
    chunks = metadata.get("chunks", [])
    successful_chunks = [c for c in chunks if c.get("success", True)]
    
    if not successful_chunks:
        logger.error("No successful chunks found in metadata")
        return
    
    logger.info(f"Found {len(successful_chunks)} narratives to synthesize")
    logger.info(f"Language: {language_code}, Voice gender: {ssml_gender}")
    if voice_name:
        logger.info(f"Voice name: {voice_name}")
    
    # Create audio output directory
    audio_dir = output_dir / "audiobook" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    audio_metadata = {"chunks": []}
    
    for i, chunk_info in enumerate(successful_chunks, 1):
        chunk_id = chunk_info["chunk_id"]
        logger.info(f"[{i}/{len(successful_chunks)}] Processing {chunk_id}...")
        
        narrative_path = narratives_dir / f"{chunk_id}.txt"
        if not narrative_path.exists():
            logger.warning(f"  Narrative not found: {narrative_path}")
            continue
        
        audio_path = audio_dir / f"{chunk_id}.mp3"
        
        result = synthesize_narrative(
            narrative_path=narrative_path,
            output_path=audio_path,
            client=client,
            language_code=language_code,
            voice_name=voice_name,
            ssml_gender=ssml_gender,
            audio_encoding=audio_encoding,
        )
        
        if result["success"]:
            audio_metadata["chunks"].append({
                "chunk_id": chunk_id,
                **result,
            })
        else:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"  ✗ Failed {chunk_id}: {error_msg}")
            audio_metadata["chunks"].append({
                "chunk_id": chunk_id,
                "success": False,
                "error": error_msg,
            })
    
    # Save audio metadata
    audio_metadata_path = output_dir / "audiobook" / "audio_metadata.json"
    with audio_metadata_path.open("w", encoding="utf-8") as f:
        json.dump(audio_metadata, f, indent=2, ensure_ascii=False)
    
    successful = sum(1 for c in audio_metadata["chunks"] if c.get("success", False))
    failed = len(successful_chunks) - successful
    
    if successful == len(successful_chunks):
        logger.info(f"✓ Synthesized {successful}/{len(successful_chunks)} narratives successfully")
    else:
        logger.warning(f"⚠ Synthesized {successful}/{len(successful_chunks)} narratives successfully ({failed} failed)")
    
    total_size_mb = sum(c.get("file_size_mb", 0) for c in audio_metadata["chunks"] if c.get("success", False))
    logger.info(f"Total audio size: {total_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Convert narrative text to speech using Google Cloud TTS"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.getenv("PDF2ANKI_OUTPUT_DIR", "output")),
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--language-code",
        type=str,
        default=os.getenv("PDF2ANKI_TTS_LANGUAGE_CODE", "cs-CZ"),
        help="Language code for TTS (default: cs-CZ)",
    )
    parser.add_argument(
        "--voice-name",
        type=str,
        default=os.getenv("PDF2ANKI_TTS_VOICE_NAME"),
        help="Specific voice name (optional, e.g., 'cs-CZ-Wavenet-A')",
    )
    parser.add_argument(
        "--ssml-gender",
        type=str,
        default=os.getenv("PDF2ANKI_TTS_GENDER", "NEUTRAL"),
        choices=["NEUTRAL", "MALE", "FEMALE"],
        help="Voice gender (default: NEUTRAL)",
    )
    parser.add_argument(
        "--audio-encoding",
        type=str,
        default=os.getenv("PDF2ANKI_TTS_ENCODING", "MP3"),
        choices=["MP3", "LINEAR16", "OGG_OPUS"],
        help="Audio encoding format (default: MP3)",
    )
    parser.add_argument(
        "--credentials",
        type=str,
        default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        help="Path to Google Cloud credentials JSON file",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    narratives_dir = args.output_dir / "audiobook" / "narratives"
    metadata_path = args.output_dir / "audiobook" / "narratives_metadata.json"
    
    if not narratives_dir.exists():
        logger.error(f"Narratives directory not found: {narratives_dir}")
        logger.error("Run process_audiobook_chunks first.")
        return
    
    synthesize_audiobook(
        narratives_dir=narratives_dir,
        metadata_path=metadata_path,
        output_dir=args.output_dir,
        language_code=args.language_code,
        voice_name=args.voice_name,
        ssml_gender=args.ssml_gender,
        audio_encoding=args.audio_encoding,
        credentials_path=args.credentials,
    )


if __name__ == "__main__":
    main()

