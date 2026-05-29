import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from app.pipeline.audio_extractor import extract_audio

def test_audio_extraction():
    video_path = "data/test_video.mp4"
    output_path = "data/test_audio.wav"
    
    # Test 1 - Check video file exists
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    print(" Test 1 passed: Video file exists")
    
    # Test 2 - Extract audio
    result = extract_audio(video_path, output_path)
    print(" Test 2 passed: Audio extraction completed")
    
    # Test 3 - Check output file exists
    assert os.path.exists(output_path), "Audio file was not created"
    print(" Test 3 passed: Audio file created")
    
    # Test 4 - Check output file is not empty
    file_size = os.path.getsize(output_path)
    assert file_size > 0, "Audio file is empty"
    print(f" Test 4 passed: Audio file size is {file_size/1024/1024:.2f} MB")
    
    print("\n All tests passed!")

if __name__ == "__main__":
    test_audio_extraction()