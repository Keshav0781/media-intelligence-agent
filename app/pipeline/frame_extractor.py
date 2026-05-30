import cv2
import os
from typing import List, Dict
from app.logger import get_logger

logger = get_logger(__name__)


def extract_keyframes(
    video_path: str,
    output_dir: str,
    transcript_segments: List[Dict] = None,
    threshold: float = None,
    min_interval_seconds: float = None
) -> List[Dict]:
    # Load defaults from config if not provided
    from app.config import get_pipeline_config
    cfg = get_pipeline_config()["frames"]
    if threshold is None:
        threshold = cfg["histogram_threshold"]
    if min_interval_seconds is None:
        min_interval_seconds = cfg["min_interval_seconds"]
    """
    Extract keyframes using multimodal scene detection.
    Combines visual histogram comparison with audio transcript timestamps.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        transcript_segments: Whisper segments with timestamps (optional)
        threshold: Visual scene change sensitivity 0-1 (lower = more sensitive)
        min_interval_seconds: Minimum seconds between extracted frames

    Returns:
        List of dicts containing frame_path, timestamp, frame_number, detection_type
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    min_interval_frames = int(min_interval_seconds * fps)

    logger.info(f"Video properties: FPS={fps:.1f}, frames={total_frames}, duration={duration:.1f}s")

    # Build set of frame numbers where speech boundaries occur
    speech_boundary_frames = set()
    if transcript_segments:
        for segment in transcript_segments:
            boundary_frame = int(segment["start"] * fps)
            speech_boundary_frames.add(boundary_frame)
        logger.info(f"Speech boundaries detected: {len(speech_boundary_frames)}")

    keyframes = []
    prev_hist = None
    frame_number = 0
    last_extracted_frame = -min_interval_frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [50, 60],
            [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        frames_since_last = frame_number - last_extracted_frame
        should_extract = False
        detection_type = None

        if prev_hist is not None and frames_since_last >= min_interval_frames:

            # Signal 1 — Visual scene change detection
            similarity = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)
            if similarity < (1 - threshold):
                should_extract = True
                detection_type = "visual"

            # Signal 2 — Audio speech boundary detection
            if frame_number in speech_boundary_frames:
                if should_extract:
                    detection_type = "visual+audio"
                else:
                    should_extract = True
                    detection_type = "audio"

        if should_extract:
            timestamp = frame_number / fps
            frame_filename = f"frame_{frame_number:06d}_{timestamp:.2f}s.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)

            keyframes.append({
                "frame_path": frame_path,
                "timestamp": round(timestamp, 2),
                "frame_number": frame_number,
                "detection_type": detection_type
            })

            last_extracted_frame = frame_number
            logger.debug(f"Keyframe extracted: {timestamp:.2f}s [{detection_type}]")

        prev_hist = hist
        frame_number += 1

    cap.release()

    visual_count = sum(1 for k in keyframes if k["detection_type"] == "visual")
    audio_count = sum(1 for k in keyframes if k["detection_type"] == "audio")
    combined_count = sum(1 for k in keyframes if k["detection_type"] == "visual+audio")

    logger.info(f"Keyframes extracted: {len(keyframes)} (visual={visual_count}, audio={audio_count}, combined={combined_count})")

    return keyframes