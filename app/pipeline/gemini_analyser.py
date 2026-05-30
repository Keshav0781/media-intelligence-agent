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

load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

# Cache directory
CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Maximum parallel workers - stays within Gemini rate limits
MAX_WORKERS = 5

# Chunk size in seconds
CHUNK_SIZE_SECONDS = 30.0

# Threshold for adaptive processing
# Videos shorter than this are sent to Gemini in one call
# Videos longer than this use chunking + parallel processing
DIRECT_ANALYSIS_THRESHOLD_SECONDS = 600.0  # 10 minutes


def get_video_hash(video_path: str) -> str:
    """
    Generate unique hash for video file.
    Used as cache key — same video always gets same hash.
    """
    hasher = hashlib.md5()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_cache(cache_path: str) -> dict:
    """Load cached analysis results if they exist."""
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cache(cache_path: str, data: dict) -> None:
    """Save analysis results to cache."""
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_chunks(
    transcript: dict,
    keyframes: List[Dict],
    chunk_size: float = CHUNK_SIZE_SECONDS
) -> List[Dict]:
    """
    Split transcript segments and keyframes into time-based chunks.
    Each chunk covers chunk_size seconds of video.
    Used for videos longer than DIRECT_ANALYSIS_THRESHOLD_SECONDS.
    """
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

    print(f"Created {len(chunks)} chunks of {chunk_size}s each")
    return chunks


def analyse_chunk(chunk: Dict, language: str) -> Dict:
    """
    Analyse a single chunk using Gemini.
    Used for long videos processed in chunks.
    """
    transcript_text = " ".join([seg["text"] for seg in chunk["segments"]])

    contents = []

    for frame in chunk["frames"]:
        try:
            img = Image.open(frame["frame_path"])
            contents.append(img)
        except Exception as e:
            print(f"Warning: Could not load frame {frame['frame_path']}: {e}")

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
            return result

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Chunk {chunk['chunk_index']} attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2)
            else:
                print(f"Chunk {chunk['chunk_index']} failed after {max_retries} attempts: {e}")
                return {
                    "chunk_index": chunk["chunk_index"],
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "topics": [],
                    "speakers": [],
                    "summary": f"Analysis failed for this segment: {str(e)}",
                    "key_moment": ""
                }


def analyse_all_chunks_parallel(chunks: List[Dict], language: str) -> List[Dict]:
    """
    Analyse all chunks in parallel using ThreadPoolExecutor.
    Max 5 workers to stay within Gemini rate limits.
    Used for long videos.
    """
    print(f"Analysing {len(chunks)} chunks in parallel (max {MAX_WORKERS} workers)...")

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
                print(f" Chunk {chunk['chunk_index']} analysed: {result.get('summary', '')[:50]}...")
            except Exception as e:
                print(f" Chunk {chunk['chunk_index']} failed: {e}")

    results.sort(key=lambda x: x["chunk_index"])
    return results


def generate_final_summary(chunk_analyses: List[Dict], language: str) -> Dict:
    """
    Hierarchical summarisation — combine all chunk summaries
    into one final complete video analysis.
    Used for long videos after all chunks are analysed.
    """
    print("Generating final hierarchical summary...")

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
            return result

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Final summary attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2)
            else:
                return {
                    "overall_summary": f"Final summary generation failed: {str(e)}",
                    "main_topics": all_topics,
                    "speakers": all_speakers,
                    "broadcast_type": "unknown",
                    "language": language,
                    "key_stories": [],
                    "chunk_analyses": chunk_analyses
                }


def analyse_video_direct(
    transcript: dict,
    keyframes: List[Dict]
) -> Dict:
    """
    Direct analysis — send everything to Gemini in one call.
    Used for short videos under DIRECT_ANALYSIS_THRESHOLD_SECONDS.
    Faster and cheaper than chunking for short videos.
    """
    print("Short video detected — using direct analysis (single Gemini call)...")

    # Build full transcript text
    full_transcript = " ".join([seg["text"] for seg in transcript["segments"]])

    # Build contents — all frames + transcript + prompt
    contents = []

    # Load all keyframes as PIL Images
    for frame in keyframes:
        try:
            img = Image.open(frame["frame_path"])
            contents.append(img)
        except Exception as e:
            print(f"Warning: Could not load frame {frame['frame_path']}: {e}")

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
            return result

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Direct analysis attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2)
            else:
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
    Implements adaptive processing:
    - Short videos (< 10 min): direct single Gemini call
    - Long videos (>= 10 min): chunking + parallel + hierarchical summary
    Also implements caching and retry logic.
    """
    # Check cache first
    video_hash = get_video_hash(video_path)
    cache_path = os.path.join(CACHE_DIR, f"{video_hash}.json")

    cached_result = load_cache(cache_path)
    if cached_result:
        print(f" Cache hit — returning cached analysis for {video_path}")
        return cached_result

    print(f"Cache miss — analysing video: {video_path}")

    # Calculate video duration from transcript
    if transcript["segments"]:
        video_duration = transcript["segments"][-1]["end"]
    else:
        video_duration = 0

    print(f"Video duration: {video_duration:.1f}s")
    print(f"Threshold for direct analysis: {DIRECT_ANALYSIS_THRESHOLD_SECONDS}s")

    # Adaptive processing decision
    if video_duration < DIRECT_ANALYSIS_THRESHOLD_SECONDS:
        # Short video — send everything at once
        final_analysis = analyse_video_direct(transcript, keyframes)
    else:
        # Long video — chunk + parallel + hierarchical summary
        chunks = create_chunks(transcript, keyframes)
        chunk_analyses = analyse_all_chunks_parallel(chunks, transcript["language"])
        final_analysis = generate_final_summary(chunk_analyses, transcript["language"])

    # Save to cache
    save_cache(cache_path, final_analysis)
    print(f" Analysis cached at {cache_path}")

    return final_analysis