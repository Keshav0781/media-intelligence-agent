import os
import sys
import json
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.rag.indexer import index_video, search, get_qdrant_client, get_collection_name
from app.pipeline.gemini_analyser import get_video_hash
from app.pipeline.transcriber import transcribe_audio


def test_indexer():
    video_path = "data/test_video.mp4"
    audio_path = "data/test_audio.wav"
    cache_path = "data/cache/03b26d23864237a264d0e262ccbb655c.json"

    # Test 1 - Check all required files exist
    assert os.path.exists(video_path), "Video file not found"
    assert os.path.exists(audio_path), "Audio file not found"
    assert os.path.exists(cache_path), "Cached analysis not found"
    print(" Test 1 passed: All required files exist")

    # Test 2 - Load transcript
    print("Loading transcript...")
    transcript = transcribe_audio(audio_path)
    assert len(transcript["segments"]) > 0, "No segments in transcript"
    print(f" Test 2 passed: Transcript loaded - {len(transcript['segments'])} segments")

    # Test 3 - Load cached Gemini analysis
    with open(cache_path, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    assert len(analysis["main_topics"]) > 0, "No topics in analysis"
    assert len(analysis["key_stories"]) > 0, "No stories in analysis"
    print(f" Test 3 passed: Analysis loaded - {len(analysis['main_topics'])} topics, {len(analysis['key_stories'])} stories")

    # Test 4 - Get video hash
    video_hash = get_video_hash(video_path)
    assert len(video_hash) > 0, "Video hash is empty"
    print(f" Test 4 passed: Video hash generated - {video_hash}")

    # Test 5 - Index video content
    print("Indexing video content...")
    total_indexed = index_video(
        video_hash=video_hash,
        video_path=video_path,
        transcript=transcript,
        analysis=analysis
    )
    expected_total = len(transcript["segments"]) + len(analysis["key_stories"]) + 1 + len(analysis["main_topics"])
    assert total_indexed == expected_total, f"Expected {expected_total} documents, got {total_indexed}"
    print(f" Test 5 passed: {total_indexed} documents indexed")

    # Test 6 - Verify collection exists in Qdrant
    client = get_qdrant_client()
    collection_name = get_collection_name(video_hash)
    assert client.collection_exists(collection_name), "Collection not found in Qdrant"
    collection_info = client.get_collection(collection_name)
    assert collection_info.points_count == total_indexed, "Points count mismatch"
    print(f" Test 6 passed: Collection exists with {collection_info.points_count} points")

    # Test 7 - Search in German — verify relevant content returned
    print("Testing German search...")
    results = search(video_hash, "Scheuer Gericht PKW Maut")
    assert len(results) > 0, "No results for German query"
    top_text = results[0]["text"].lower() + results[0]["context"].lower()
    assert any(word in top_text for word in ["scheuer", "gericht", "pkw", "maut"]), \
        f"Top result not relevant: {results[0]['text']}"
    print(f" Test 7 passed: German search returned relevant result - score:{results[0]['score']:.3f}")
    print(f"   Top result: {results[0]['text'][:80]}")

    # Test 8 - Search in English for German content — verify multilingual works
    print("Testing English search for German content...")
    results_en = search(video_hash, "minister court case toll road")
    assert len(results_en) > 0, "No results for English query"
    top_text_en = results_en[0]["text"].lower() + results_en[0]["context"].lower()
    assert any(word in top_text_en for word in ["scheuer", "gericht", "pkw", "maut", "minister", "court"]), \
        f"Top result not relevant: {results_en[0]['text']}"
    print(f" Test 8 passed: English search returned relevant result - score:{results_en[0]['score']:.3f}")
    print(f"   Top result: {results_en[0]['text'][:80]}")

    # Test 9 - Content type filtering
    print("Testing content type filtering...")
    story_results = search(video_hash, "Hungary EU funds", content_type="story")
    assert len(story_results) > 0, "No story results"
    assert all(r["type"] == "story" for r in story_results), "Non-story results returned"
    print(f" Test 9 passed: Content type filter works - {len(story_results)} stories returned")

    # Test 10 - Re-index same video does not duplicate
    print("Testing re-indexing same video...")
    index_video(
        video_hash=video_hash,
        video_path=video_path,
        transcript=transcript,
        analysis=analysis
    )
    collection_info = client.get_collection(collection_name)
    assert collection_info.points_count == total_indexed, "Re-indexing created duplicates"
    print(f" Test 10 passed: Re-indexing did not create duplicates")

    # Print sample search results
    print("\n--- Sample search results for 'drone Romania Russia' ---")
    drone_results = search(video_hash, "drone Romania Russia", limit=3)
    for r in drone_results:
        print(f"  [{r['type']}] score:{r['score']:.3f} timestamp:{r['timestamp_start']} — {r['text'][:60]}")

    print("\n All tests passed!")


if __name__ == "__main__":
    test_indexer()