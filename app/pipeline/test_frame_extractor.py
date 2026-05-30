import os
import pytest
from PIL import Image


def test_keyframes_extracted(keyframes):
    """Test that keyframes were extracted."""
    assert len(keyframes) > 0


def test_keyframe_count(keyframes):
    """Test expected number of keyframes."""
    assert len(keyframes) == 38


def test_keyframe_files_exist(keyframes):
    """Test that all keyframe files exist on disk."""
    for kf in keyframes:
        assert os.path.exists(kf["frame_path"]), f"Frame file missing: {kf['frame_path']}"


def test_keyframe_structure(keyframes):
    """Test that keyframes have correct structure — no base64, lazy loading."""
    first_kf = keyframes[0]
    assert "frame_path" in first_kf
    assert "timestamp" in first_kf
    assert "frame_number" in first_kf
    assert "detection_type" in first_kf
    assert "base64" not in first_kf


def test_keyframe_pil_loadable(keyframes):
    """Test that keyframes can be loaded as PIL Images."""
    img = Image.open(keyframes[0]["frame_path"])
    assert img is not None
    assert img.size == (640, 360)


def test_keyframe_detection_types_valid(keyframes):
    """Test that detection types are valid."""
    valid_types = {"visual", "audio", "visual+audio"}
    for kf in keyframes:
        assert kf["detection_type"] in valid_types


def test_keyframe_timestamps_valid(keyframes):
    """Test that timestamps are valid positive numbers."""
    for kf in keyframes:
        assert kf["timestamp"] >= 0
        assert kf["frame_number"] >= 0


def test_keyframe_visual_count(keyframes):
    """Test visual scene change count."""
    visual = [k for k in keyframes if k["detection_type"] == "visual"]
    assert len(visual) == 6


def test_keyframe_audio_count(keyframes):
    """Test audio boundary count."""
    audio = [k for k in keyframes if k["detection_type"] == "audio"]
    assert len(audio) == 32