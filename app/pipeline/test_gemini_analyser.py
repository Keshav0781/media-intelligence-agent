import os
import sys
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.pipeline.gemini_analyser import analyse_video
from app.pipeline.transcriber import transcribe_audio
from app.pipeline.frame_extractor import extract_keyframes

def test_gemini_analyser():
    video_path = "data/test_video.mp4"
    audio_path = "data/test_audio.wav"
    keyframes_dir = "data/keyframes"

    # Test 1 - Check all required files exist
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    assert os.path.exists(audio_path), f"Audio file not found: {audio_path}"
    assert os.path.exists(keyframes_dir), f"Keyframes directory not found: {keyframes_dir}"
    print(" Test 1 passed: All required files exist")

    # Test 2 - Load transcript
    print("Loading transcript...")
    transcript = transcribe_audio(audio_path)
    assert len(transcript["segments"]) > 0, "No segments in transcript"
    print(f" Test 2 passed: Transcript loaded - {len(transcript['segments'])} segments")

    # Test 3 - Load keyframes
    print("Loading keyframes...")
    keyframes = extract_keyframes(
        video_path=video_path,
        output_dir=keyframes_dir,
        transcript_segments=transcript["segments"]
    )
    assert len(keyframes) > 0, "No keyframes extracted"
    print(f" Test 3 passed: {len(keyframes)} keyframes loaded")

    # Test 4 - Run full video analysis
    print("Running Gemini analysis...")
    result = analyse_video(
        video_path=video_path,
        transcript=transcript,
        keyframes=keyframes
    )
    print(" Test 4 passed: Gemini analysis completed")

    # Test 5 - Check result has required fields
    assert "overall_summary" in result, "Missing overall_summary"
    assert "main_topics" in result, "Missing main_topics"
    assert "speakers" in result, "Missing speakers"
    assert "broadcast_type" in result, "Missing broadcast_type"
    assert "language" in result, "Missing language"
    assert "key_stories" in result, "Missing key_stories"
    print(" Test 5 passed: Result has all required fields")

    # Test 6 - Check result content is not empty
    assert len(result["overall_summary"]) > 0, "Overall summary is empty"
    assert len(result["main_topics"]) > 0, "No main topics found"
    print(" Test 6 passed: Result has meaningful content")

    # Test 7 - Check cache was created
    from app.pipeline.gemini_analyser import get_video_hash, CACHE_DIR
    video_hash = get_video_hash(video_path)
    cache_path = os.path.join(CACHE_DIR, f"{video_hash}.json")
    assert os.path.exists(cache_path), "Cache file was not created"
    print(" Test 7 passed: Cache file created")

    # Test 8 - Run again and verify cache is used
    print("Running analysis again to test cache...")
    import time
    start = time.time()
    result_cached = analyse_video(
        video_path=video_path,
        transcript=transcript,
        keyframes=keyframes
    )
    elapsed = time.time() - start
    assert elapsed < 1.0, f"Cache lookup took too long: {elapsed:.2f}s"
    assert result_cached["overall_summary"] == result["overall_summary"], "Cached result differs"
    print(f" Test 8 passed: Cache returned in {elapsed:.3f}s")

    # Print results
    print("\n--- Analysis Results ---")
    print(f"Language: {result['language']}")
    print(f"Broadcast type: {result['broadcast_type']}")
    print(f"Main topics: {result['main_topics']}")
    print(f"Speakers: {result['speakers']}")
    print(f"Summary: {result['overall_summary']}")
    print("\nKey stories:")
    for story in result.get("key_stories", []):
        print(f"  [{story.get('timestamp', '')}] {story.get('story', '')}")

    print("\n All tests passed!")

if __name__ == "__main__":
    test_gemini_analyser()