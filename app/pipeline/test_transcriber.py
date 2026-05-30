import pytest


def test_transcript_has_segments(transcript):
    """Test that transcription produced segments."""
    assert len(transcript["segments"]) > 0


def test_transcript_language_detected(transcript):
    """Test that language was detected."""
    assert transcript["language"] is not None
    assert len(transcript["language"]) > 0


def test_transcript_language_is_german(transcript):
    """Test that German language was correctly detected."""
    assert transcript["language"] == "de"


def test_transcript_has_text(transcript):
    """Test that transcript has text content."""
    assert len(transcript["text"]) > 0


def test_transcript_segment_count(transcript):
    """Test expected number of segments."""
    assert len(transcript["segments"]) == 43


def test_transcript_segment_structure(transcript):
    """Test that segments have correct structure."""
    first_segment = transcript["segments"][0]
    assert "start" in first_segment
    assert "end" in first_segment
    assert "text" in first_segment


def test_transcript_segment_timestamps(transcript):
    """Test that segment timestamps are valid."""
    for segment in transcript["segments"]:
        assert segment["start"] >= 0
        assert segment["end"] > segment["start"]