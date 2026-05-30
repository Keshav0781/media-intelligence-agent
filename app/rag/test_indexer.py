import os
import pytest


def test_index_video(clean_qdrant, video_hash, test_video_path, transcript, analysis):
    """Test that video content is indexed correctly."""
    from app.rag.indexer import index_video
    total = index_video(
        video_hash=video_hash,
        video_path=test_video_path,
        transcript=transcript,
        analysis=analysis
    )
    expected = len(transcript["segments"]) + len(analysis["key_stories"]) + 1 + len(analysis["main_topics"])
    assert total == expected


def test_collection_exists(clean_qdrant, video_hash, test_video_path, transcript, analysis):
    """Test that Qdrant collection was created."""
    from app.rag.indexer import index_video, get_qdrant_client, get_collection_name
    index_video(video_hash=video_hash, video_path=test_video_path, transcript=transcript, analysis=analysis)
    client = get_qdrant_client()
    collection_name = get_collection_name(video_hash)
    assert client.collection_exists(collection_name)


def test_collection_point_count(clean_qdrant, video_hash, test_video_path, transcript, analysis):
    """Test that correct number of points stored in collection."""
    from app.rag.indexer import index_video, get_qdrant_client, get_collection_name
    total = index_video(video_hash=video_hash, video_path=test_video_path, transcript=transcript, analysis=analysis)
    client = get_qdrant_client()
    collection_name = get_collection_name(video_hash)
    info = client.get_collection(collection_name)
    assert info.points_count == total


def test_german_keyword_search(clean_qdrant, video_hash, test_video_path, transcript, analysis):
    """Test German keyword search returns relevant content."""
    from app.rag.indexer import index_video, search
    index_video(video_hash=video_hash, video_path=test_video_path, transcript=transcript, analysis=analysis)
    results = search(video_hash, "Scheuer Gericht PKW Maut")
    assert len(results) > 0
    top_text = results[0]["text"].lower() + results[0]["context"].lower()
    assert any(word in top_text for word in ["scheuer", "gericht", "pkw", "maut"])


def test_english_search_german_content(clean_qdrant, video_hash, test_video_path, transcript, analysis):
    """Test English query finds German content — multilingual capability."""
    from app.rag.indexer import index_video, search
    index_video(video_hash=video_hash, video_path=test_video_path, transcript=transcript, analysis=analysis)
    results = search(video_hash, "minister court case toll road")
    assert len(results) > 0
    top_text = results[0]["text"].lower() + results[0]["context"].lower()
    assert any(word in top_text for word in ["scheuer", "gericht", "pkw", "maut", "minister", "court"])


def test_content_type_filtering(clean_qdrant, video_hash, test_video_path, transcript, analysis):
    """Test that content type filtering works correctly."""
    from app.rag.indexer import index_video, search
    index_video(video_hash=video_hash, video_path=test_video_path, transcript=transcript, analysis=analysis)
    story_results = search(video_hash, "Hungary EU funds", content_type="story")
    assert len(story_results) > 0
    assert all(r["type"] == "story" for r in story_results)


def test_reindexing_no_duplicates(clean_qdrant, video_hash, test_video_path, transcript, analysis):
    """Test that re-indexing same video does not create duplicates."""
    from app.rag.indexer import index_video, get_qdrant_client, get_collection_name
    total = index_video(video_hash=video_hash, video_path=test_video_path, transcript=transcript, analysis=analysis)
    index_video(video_hash=video_hash, video_path=test_video_path, transcript=transcript, analysis=analysis)
    client = get_qdrant_client()
    collection_name = get_collection_name(video_hash)
    info = client.get_collection(collection_name)
    assert info.points_count == total


def test_drone_search_returns_relevant(clean_qdrant, video_hash, test_video_path, transcript, analysis):
    """Test that drone story search returns relevant content."""
    from app.rag.indexer import index_video, search
    index_video(video_hash=video_hash, video_path=test_video_path, transcript=transcript, analysis=analysis)
    results = search(video_hash, "drone Romania Russia", limit=3)
    assert len(results) > 0
    top_text = results[0]["text"].lower() + results[0]["context"].lower()
    assert any(word in top_text for word in ["drohne", "rumänien", "drone", "romania", "russian"])