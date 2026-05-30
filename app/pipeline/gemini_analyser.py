import os
import json
import hashlib
import time
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import google.genai as genai
from dotenv import load_dotenv
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from app.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

# Cache directory
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Maximum parallel workers
MAX_WORKERS = 5

# Chunk size in seconds
CHUNK_SIZE_SECONDS = 30.0

# Threshold for adaptive processing
DIRECT_ANALYSIS_THRESHOLD_SECONDS = 600.0


def get_video_hash(video_path: str) -> str:
    """Generate unique MD5 hash for video file. Used as cache key."""
    hasher = hashlib.md5()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_cache(cache_path: str) -> dict:
    """Load cached analysis results if they exist."""
    if os.path.exists(cache_path):
        logger.debug(f"Loading cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(cache_path: str, data: dict) -> None:
    """Save analysis results to cache."""
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.debug(f"Cache saved: {cache_path}")


def create_chunks(
    transcript: dict,
    keyframes: List[Dict],
    chunk_size: float = CHUNK_SIZE_SECONDS
) -> List[Dict]:
    """Split transcript segments and keyframes into time-based chunks."""
    if not transcript["segments"]:
        return []

    total_duration = transcript["segments"][-1]["end"]
    chunks = []

    chunk_start = 0.0
    while chunk_start < total_duration:
        chunk_end = chunk_start + chunk_size

        chunk_segments = [
            seg for seg in transcript["segments"]
            if seg["start"] >= chunk_start and seg["start"] < chunk_end
        ]

        chunk_frames = [
            kf for kf in keyframes
            if kf["timestamp"] >= chunk_start and kf["timestamp"] < chunk_end
        ]

        if chunk_segments or chunk_frames:
            chunks.append({
                "chunk_index": len(chunks),
                "start_time": chunk_start,
                "end_time": chunk_end,
                "segments": chunk_segments,
                "frames": chunk_frames
            })

        chunk_start = chunk_end

    logger.info(f"Created {len(chunks)} chunks of {chunk_size}s each")
    return chunks


def analyse_chunk(chunk: Dict, language: str) -> Dict:
    """Analyse a single chunk using Gemini."""
    transcript_text = " ".join([seg["text"] for seg in chunk["segments"]])

    contents = []

    for frame in chunk["frames"]:
        try:
            img = Image.open(frame["frame_path"])
            contents.append(img)
        except Exception as e:
            logger.warning(f"Could not load frame {frame['frame_path']}: {e}")

    prompt = f"""
You are an AI analyst for a professional broadcast media archive system.

Analyse this {chunk['end_time'] - chunk['start_time']:.0f} second news segment 
from timestamp {chunk['start_time']:.1f}s to {chunk['end_time']:.1f}s.

Transcript ({language}):
{transcript_text}

The images show keyframes extracted from this segment.

Provide analysis in this exact JSON format:
{{
    "topics": ["topic1", "topic2"],
    "speakers": ["speaker description 1"],
    "summary": "2-3 sentence summary of this segment",
    "key_moment": "most important moment in this segment"
}}

Respond with valid JSON only. No other text.
"""
    contents.append(prompt)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents
            )

            response_text = response.text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            result = json.loads(response_text)
            result["chunk_index"] = chunk["chunk_index"]
            result["start_time"] = chunk["start_time"]
            result["end_time"] = chunk["end_time"]
            logger.info(f"Chunk {chunk['chunk_index']} analysed successfully")
            return result

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Chunk {chunk['chunk_index']} attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2)
            else:
                logger.error(f"Chunk {chunk['chunk_index']} failed after {max_retries} attempts: {e}")
                return {
                    "chunk_index": chunk["chunk_index"],
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "topics": [],
                    "speakers": [],
                    "summary": f"Analysis failed: {str(e)}",
                    "key_moment": ""
                }


def analyse_all_chunks_parallel(chunks: List[Dict], language: str) -> List[Dict]:
    """Analyse all chunks in parallel using ThreadPoolExecutor."""
    logger.info(f"Analysing {len(chunks)} chunks in parallel (max {MAX_WORKERS} workers)")

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_chunk = {
            executor.submit(analyse_chunk, chunk, language): chunk
            for chunk in chunks
        }

        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Chunk {chunk['chunk_index']} failed: {e}")

    results.sort(key=lambda x: x["chunk_index"])
    return results


def generate_final_summary(chunk_analyses: List[Dict], language: str) -> Dict:
    """Hierarchical summarisation — combine all chunk summaries."""
    logger.info("Generating final hierarchical summary...")

    chunk_summaries = "\n".join([
        f"[{ca['start_time']:.1f}s - {ca['end_time']:.1f}s]: {ca['summary']}"
        for ca in chunk_analyses
    ])

    all_topics = list(set([
        topic
        for ca in chunk_analyses
        for topic in ca.get("topics", [])
    ]))

    all_speakers = list(set([
        speaker
        for ca in chunk_analyses
        for speaker in ca.get("speakers", [])
    ]))

    prompt = f"""
You are an AI analyst for a professional broadcast media archive system.

Based on these segment summaries from a {language} news broadcast,
provide a complete video analysis.

Segment summaries:
{chunk_summaries}

Provide analysis in this exact JSON format:
{{
    "overall_summary": "3-4 sentence complete summary of entire broadcast",
    "main_topics": ["topic1", "topic2", "topic3"],
    "speakers": ["speaker1", "speaker2"],
    "broadcast_type": "type of broadcast (e.g. news bulletin, interview, documentary)",
    "language": "{language}",
    "key_stories": [
        {{"timestamp": "0:00", "story": "story description"}},
        {{"timestamp": "0:30", "story": "story description"}}
    ]
}}

Respond with valid JSON only. No other text.
"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt]
            )

            response_text = response.text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            result = json.loads(response_text)
            result["chunk_analyses"] = chunk_analyses
            logger.info("Final summary generated successfully")
            return result

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Final summary attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2)
            else:
                logger.error(f"Final summary failed after {max_retries} attempts: {e}")
                return {
                    "overall_summary": f"Final summary generation failed: {str(e)}",
                    "main_topics": all_topics,
                    "speakers": all_speakers,
                    "broadcast_type": "unknown",
                    "language": language,
                    "key_stories": [],
                    "chunk_analyses": chunk_analyses
                }


def analyse_video_direct(transcript: dict, keyframes: List[Dict]) -> Dict:
    """
    Direct analysis — send everything to Gemini in one call.
    Used for short videos under DIRECT_ANALYSIS_THRESHOLD_SECONDS.
    """
    logger.info("Short video detected — using direct analysis (single Gemini call)")

    full_transcript = " ".join([seg["text"] for seg in transcript["segments"]])
    contents = []

    for frame in keyframes:
        try:
            img = Image.open(frame["frame_path"])
            contents.append(img)
        except Exception as e:
            logger.warning(f"Could not load frame {frame['frame_path']}: {e}")

    prompt = f"""
You are an AI analyst for a professional broadcast media archive system.

Analyse this complete news broadcast.

Full transcript ({transcript['language']}):
{full_transcript}

The images show keyframes extracted throughout the broadcast.

Provide analysis in this exact JSON format:
{{
    "overall_summary": "3-4 sentence complete summary of entire broadcast",
    "main_topics": ["topic1", "topic2", "topic3"],
    "speakers": ["speaker1", "speaker2"],
    "broadcast_type": "type of broadcast (e.g. news bulletin, interview, documentary)",
    "language": "{transcript['language']}",
    "key_stories": [
        {{"timestamp": "0:00", "story": "story description"}},
        {{"timestamp": "0:30", "story": "story description"}}
    ]
}}

Respond with valid JSON only. No other text.
"""
    contents.append(prompt)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents
            )

            response_text = response.text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            result = json.loads(response_text)
            result["chunk_analyses"] = []
            logger.info("Direct analysis completed successfully")
            return result

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Direct analysis attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2)
            else:
                logger.error(f"Direct analysis failed after {max_retries} attempts: {e}")
                return {
                    "overall_summary": f"Analysis failed: {str(e)}",
                    "main_topics": [],
                    "speakers": [],
                    "broadcast_type": "unknown",
                    "language": transcript["language"],
                    "key_stories": [],
                    "chunk_analyses": []
                }


def analyse_video(
    video_path: str,
    transcript: dict,
    keyframes: List[Dict]
) -> Dict:
    """
    Main function — complete video analysis using Gemini.
    Implements adaptive processing, caching, parallel chunks.
    """
    video_hash = get_video_hash(video_path)
    cache_path = os.path.join(CACHE_DIR, f"{video_hash}.json")

    cached_result = load_cache(cache_path)
    if cached_result:
        logger.info(f"Cache hit — returning cached analysis for {video_path}")
        return cached_result

    logger.info(f"Cache miss — analysing video: {video_path}")

    if transcript["segments"]:
        video_duration = transcript["segments"][-1]["end"]
    else:
        video_duration = 0

    logger.info(f"Video duration: {video_duration:.1f}s — threshold: {DIRECT_ANALYSIS_THRESHOLD_SECONDS}s")

    if video_duration < DIRECT_ANALYSIS_THRESHOLD_SECONDS:
        final_analysis = analyse_video_direct(transcript, keyframes)
    else:
        chunks = create_chunks(transcript, keyframes)
        chunk_analyses = analyse_all_chunks_parallel(chunks, transcript["language"])
        final_analysis = generate_final_summary(chunk_analyses, transcript["language"])

    save_cache(cache_path, final_analysis)
    logger.info(f"Analysis cached: {cache_path}")

    return final_analysis