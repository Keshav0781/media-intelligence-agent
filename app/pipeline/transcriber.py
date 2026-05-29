import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import whisper
import os

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
    # Check audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    print(f"Loading Whisper model...")
    # Using 'base' model - good balance of speed and accuracy for testing
    # In production we would use 'medium' or 'large' for better accuracy
    model = whisper.load_model("base")
    
    print(f"Transcribing audio: {audio_path}")
    
    # Transcribe with or without language hint
    options = {}
    if language:
        options["language"] = language
    
    result = model.transcribe(audio_path, **options)
    
    # Structure the output clearly
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
    
    print(f"Transcription complete. Language detected: {transcript['language']}")
    print(f"Total segments: {len(transcript['segments'])}")
    
    return transcript