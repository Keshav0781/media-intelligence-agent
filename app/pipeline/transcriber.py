import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import whisper
import os
from app.logger import get_logger

logger = get_logger(__name__)


def transcribe_audio(audio_path: str, language: str = None) -> dict:
    """
    Transcribe audio file using Whisper.
    
    Args:
        audio_path: Path to audio file (WAV format)
        language: Optional language code (e.g. 'en', 'de'). 
                 If None, Whisper auto-detects language.
    
    Returns:
        dict containing:
            - text: Full transcript as string
            - segments: List of segments with timestamps
            - language: Detected/specified language
    """
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(f"Loading Whisper model...")
    model = whisper.load_model("base")

    logger.info(f"Transcribing audio: {audio_path}")

    options = {}
    if language:
        options["language"] = language
        logger.debug(f"Language hint provided: {language}")

    result = model.transcribe(audio_path, **options)

    transcript = {
        "text": result["text"].strip(),
        "segments": [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            }
            for segment in result["segments"]
        ],
        "language": result["language"]
    }

    logger.info(f"Transcription complete. Language detected: {transcript['language']}")
    logger.info(f"Total segments: {len(transcript['segments'])}")
    logger.debug(f"Transcript length: {len(transcript['text'])} characters")

    return transcript