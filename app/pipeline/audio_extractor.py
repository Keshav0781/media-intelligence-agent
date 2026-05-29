import subprocess
import os

def extract_audio(video_path: str, output_path: str) -> str:
    """
    Extract audio from video file and save as WAV.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save extracted audio
    
    Returns:
        Path to extracted audio file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # FFmpeg command to extract audio
    # -i: input file
    # -vn: no video
    # -acodec pcm_s16le: uncompressed WAV format
    # -ar 16000: 16kHz sample rate (optimal for Whisper)
    # -ac 1: mono audio (Whisper works best with mono)
    # -y: overwrite output if exists
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        output_path
    ]
    
    # Run FFmpeg command
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Check if extraction was successful
    if result.returncode != 0:
        raise RuntimeError(
            f"Audio extraction failed: {result.stderr.decode()}"
        )
    
    print(f"Audio extracted successfully: {output_path}")
    return output_path