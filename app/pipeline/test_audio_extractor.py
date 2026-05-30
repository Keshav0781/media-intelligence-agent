import os
import pytest


def test_video_file_exists(test_video_path):
    """Test that video file exists."""
    assert os.path.exists(test_video_path)


def test_audio_extraction(test_video_path):
    """Test audio extraction from video."""
    from app.pipeline.audio_extractor import extract_audio
    output_path = "data/test_audio.wav"
    result = extract_audio(test_video_path, output_path)
    assert result == output_path


def test_audio_file_created():
    """Test that audio file was created."""
    assert os.path.exists("data/test_audio.wav")


def test_audio_file_not_empty():
    """Test that audio file has content."""
    size = os.path.getsize("data/test_audio.wav")
    assert size > 0
    size_mb = size / (1024 * 1024)
    assert size_mb > 1.0