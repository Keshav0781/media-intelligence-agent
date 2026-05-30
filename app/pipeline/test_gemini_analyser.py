import os
import time
import pytest


def test_analysis_has_required_fields(analysis):
    """Test that analysis has all required fields."""
    assert "overall_summary" in analysis
    assert "main_topics" in analysis
    assert "speakers" in analysis
    assert "broadcast_type" in analysis
    assert "language" in analysis
    assert "key_stories" in analysis


def test_analysis_language_is_german(analysis):
    """Test that German language was correctly identified."""
    assert analysis["language"] == "de"


def test_analysis_broadcast_type(analysis):
    """Test that broadcast type was identified."""
    assert len(analysis["broadcast_type"]) > 0


def test_analysis_has_topics(analysis):
    """Test that main topics were extracted."""
    assert len(analysis["main_topics"]) > 0


def test_analysis_topic_count(analysis):
    """Test expected number of topics."""
    assert len(analysis["main_topics"]) == 6


def test_analysis_has_stories(analysis):
    """Test that key stories were extracted."""
    assert len(analysis["key_stories"]) > 0


def test_analysis_story_count(analysis):
    """Test expected number of key stories."""
    assert len(analysis["key_stories"]) == 6


def test_analysis_story_structure(analysis):
    """Test that stories have correct structure."""
    for story in analysis["key_stories"]:
        assert "timestamp" in story
        assert "story" in story
        assert len(story["story"]) > 0


def test_analysis_has_speakers(analysis):
    """Test that speakers were identified."""
    assert len(analysis["speakers"]) > 0


def test_analysis_summary_not_empty(analysis):
    """Test that summary has meaningful content."""
    assert len(analysis["overall_summary"]) > 100


def test_cache_returns_quickly(test_video_path, transcript, keyframes):
    """Test that cache returns result in under 1 second."""
    from app.pipeline.gemini_analyser import analyse_video
    start = time.time()
    result = analyse_video(
        video_path=test_video_path,
        transcript=transcript,
        keyframes=keyframes
    )
    elapsed = time.time() - start
    assert elapsed < 1.0
    assert "overall_summary" in result