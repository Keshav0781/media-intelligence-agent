import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.pipeline.frame_extractor import extract_keyframes
from app.pipeline.transcriber import transcribe_audio

def test_frame_extraction():
    video_path = "data/test_video.mp4"
    output_dir = "data/keyframes"
    audio_path = "data/test_audio.wav"

    # Test 1 - Check video file exists
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    print(" Test 1 passed: Video file exists")

    # Test 2 - Load transcript segments from Whisper
    # We use existing transcript to test multimodal detection
    print("Loading transcript for multimodal detection...")
    transcript = transcribe_audio(audio_path)
    assert len(transcript["segments"]) > 0, "No segments in transcript"
    print(f" Test 2 passed: Loaded {len(transcript['segments'])} transcript segments")

    # Test 3 - Extract keyframes with multimodal detection
    keyframes = extract_keyframes(
        video_path=video_path,
        output_dir=output_dir,
        transcript_segments=transcript["segments"]
    )
    print(" Test 3 passed: Keyframe extraction completed")

    # Test 4 - Check keyframes were extracted
    assert len(keyframes) > 0, "No keyframes extracted"
    print(f" Test 4 passed: {len(keyframes)} keyframes extracted")

    # Test 5 - Check keyframe files exist on disk
    for kf in keyframes:
        assert os.path.exists(kf["frame_path"]), f"Frame file missing: {kf['frame_path']}"
    print(" Test 5 passed: All keyframe files exist on disk")

    # Test 6 - Check keyframe structure is correct
    first_kf = keyframes[0]
    assert "frame_path" in first_kf, "Missing frame_path"
    assert "timestamp" in first_kf, "Missing timestamp"
    assert "frame_number" in first_kf, "Missing frame_number"
    assert "base64" in first_kf, "Missing base64"
    assert "detection_type" in first_kf, "Missing detection_type"
    print(" Test 6 passed: Keyframe structure is correct")

    # Test 7 - Check base64 is not empty
    assert len(first_kf["base64"]) > 0, "base64 is empty"
    print(" Test 7 passed: base64 encoding is present")

    # Test 8 - Check detection types are valid
    valid_types = {"visual", "audio", "visual+audio"}
    for kf in keyframes:
        assert kf["detection_type"] in valid_types, \
            f"Invalid detection type: {kf['detection_type']}"
    print(" Test 8 passed: All detection types are valid")

    # Print sample output
    print("\n--- Sample keyframes ---")
    for kf in keyframes[:5]:
        print(f"[{kf['timestamp']}s] Frame {kf['frame_number']} — {kf['detection_type']}")

    print("\n All tests passed!")

if __name__ == "__main__":
    test_frame_extraction()