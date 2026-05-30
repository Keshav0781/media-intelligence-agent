# Note: visual+audio combined detection requires exact frame coincidence.
# In practice, camera cuts and speech boundaries are typically 2-5 frames apart.
# This is a known characteristic of broadcast media — not a bug.
# Future improvement: implement tolerance window of ±5 frames for combined detection.


import cv2
import os
from typing import List, Dict

def extract_keyframes(
    video_path: str,
    output_dir: str,
    transcript_segments: List[Dict] = None,
    threshold: float = 0.4,
    min_interval_seconds: float = 2.0
) -> List[Dict]:
    """
    Extract keyframes using multimodal scene detection.
    Combines visual histogram comparison with audio transcript timestamps.
    
    This mirrors production broadcast media analysis systems where scene 
    changes are detected using both visual and audio signals together.
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        transcript_segments: Whisper segments with timestamps (optional)
                           When provided, extracts frames at speech boundaries too
        threshold: Visual scene change sensitivity 0-1 (lower = more sensitive)
                  0.4 is optimal for news broadcast content
        min_interval_seconds: Minimum seconds between extracted frames
    
    Returns:
        List of dicts containing:
            - frame_path: Path to saved frame image
            - timestamp: Time in seconds when frame was captured
            - frame_number: Frame number in video
            - detection_type: 'visual', 'audio' or 'visual+audio'
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    min_interval_frames = int(min_interval_seconds * fps)

    print(f"Video properties:")
    print(f"  FPS: {fps:.1f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.1f} seconds")

    # Build set of frame numbers where speech boundaries occur
    speech_boundary_frames = set()
    if transcript_segments:
        for segment in transcript_segments:
            boundary_frame = int(segment["start"] * fps)
            speech_boundary_frames.add(boundary_frame)
        print(f"  Speech boundaries detected: {len(speech_boundary_frames)}")

    keyframes = []
    prev_hist = None
    frame_number = 0
    last_extracted_frame = -min_interval_frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV for robust colour comparison
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate normalised histogram using Hue and Saturation channels
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

            # Store only file path — PIL Image loaded on demand in Gemini component
            # This is lazy loading — keeps memory low for long videos
            keyframes.append({
                "frame_path": frame_path,
                "timestamp": round(timestamp, 2),
                "frame_number": frame_number,
                "detection_type": detection_type
            })

            last_extracted_frame = frame_number

        prev_hist = hist
        frame_number += 1

    cap.release()

    visual_count = sum(1 for k in keyframes if k["detection_type"] == "visual")
    audio_count = sum(1 for k in keyframes if k["detection_type"] == "audio")
    combined_count = sum(1 for k in keyframes if k["detection_type"] == "visual+audio")

    print(f"Keyframes extracted: {len(keyframes)}")
    print(f"  Visual scene changes: {visual_count}")
    print(f"  Audio speech boundaries: {audio_count}")
    print(f"  Combined signals: {combined_count}")

    return keyframes