import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.pipeline.transcriber import transcribe_audio

def test_transcription():
    audio_path = "data/test_audio.wav"
    
    # Test 1 - Check audio file exists
    assert os.path.exists(audio_path), f"Audio file not found: {audio_path}"
    print(" Test 1 passed: Audio file exists")
    
    # Test 2 - Transcribe audio
    result = transcribe_audio(audio_path)
    print(" Test 2 passed: Transcription completed")
    
    # Test 3 - Check transcript has text
    assert len(result["text"]) > 0, "Transcript text is empty"
    print(f" Test 3 passed: Transcript has {len(result['text'])} characters")
    
    # Test 4 - Check segments exist
    assert len(result["segments"]) > 0, "No segments found"
    print(f" Test 4 passed: Found {len(result['segments'])} segments")
    
    # Test 5 - Check language was detected
    assert result["language"] is not None, "Language not detected"
    print(f" Test 5 passed: Language detected as '{result['language']}'")
    
    # Test 6 - Check segments have correct structure
    first_segment = result["segments"][0]
    assert "start" in first_segment, "Segment missing start time"
    assert "end" in first_segment, "Segment missing end time"
    assert "text" in first_segment, "Segment missing text"
    print(" Test 6 passed: Segments have correct structure")
    
    # Print sample output
    print("\n--- Sample transcript (first 3 segments) ---")
    for segment in result["segments"][:3]:
        print(f"[{segment['start']:.1f}s - {segment['end']:.1f}s]: {segment['text']}")
    
    print("\n All tests passed!")

if __name__ == "__main__":
    test_transcription()