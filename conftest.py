import pytest
import os
import shutil
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


@pytest.fixture(scope="session")
def test_video_path():
    """Path to test video file."""
    path = "data/test_video.mp4"
    assert os.path.exists(path), f"Test video not found: {path}"
    return path


@pytest.fixture(scope="session")
def test_audio_path():
    """Path to test audio file."""
    return "data/test_audio.wav"


@pytest.fixture(scope="session")
def keyframes_dir():
    """Path to keyframes directory."""
    return "data/keyframes"


@pytest.fixture(scope="session")
def cache_path():
    """Path to cached Gemini analysis."""
    path = "data/cache/03b26d23864237a264d0e262ccbb655c.json"
    assert os.path.exists(path), f"Cache file not found: {path}"
    return path


@pytest.fixture(scope="session")
def transcript(test_audio_path):
    """Load transcript once for entire test session."""
    from app.pipeline.transcriber import transcribe_audio
    return transcribe_audio(test_audio_path)


@pytest.fixture(scope="session")
def analysis(cache_path):
    """Load cached Gemini analysis once for entire test session."""
    import json
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def keyframes(test_video_path, keyframes_dir, transcript):
    """Extract keyframes once for entire test session."""
    from app.pipeline.frame_extractor import extract_keyframes
    return extract_keyframes(
        video_path=test_video_path,
        output_dir=keyframes_dir,
        transcript_segments=transcript["segments"]
    )


@pytest.fixture(scope="session")
def video_hash(test_video_path):
    """Get video hash once for entire test session."""
    from app.pipeline.gemini_analyser import get_video_hash
    return get_video_hash(test_video_path)


@pytest.fixture(scope="function")
def clean_qdrant():
    """
    Provide clean Qdrant database for each test function that needs it.
    Creates fresh database before test, cleans up after.
    No manual rm -rf needed.
    """
    # Clean before test
    if os.path.exists("data/qdrant"):
        shutil.rmtree("data/qdrant")
    if os.path.exists("data/bm25"):
        shutil.rmtree("data/bm25")

    # Reset singleton so fresh client is created
    import app.rag.indexer as indexer
    indexer._qdrant_client = None

    yield

    # Clean after test
    if os.path.exists("data/qdrant"):
        shutil.rmtree("data/qdrant")
    if os.path.exists("data/bm25"):
        shutil.rmtree("data/bm25")

    # Reset singleton again
    indexer._qdrant_client = None