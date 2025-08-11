#!/usr/bin/env python3
"""
Test script to verify the audio extraction workaround.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.youtube_dl_service import YouTubeDLService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_audio_workaround():
    """Test the audio extraction workaround."""
    service = YouTubeDLService(download_path="./temp/test_videos")
    
    # Test URL - using a simple, reliable YouTube video
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll - should be available
    
    print("Testing Audio Extraction Workaround")
    print("=" * 40)
    print(f"Test URL: {test_url}")
    
    try:
        # Test the workaround method
        print("\n1. Testing audio extraction workaround...")
        audio_info = await service.extract_audio_workaround(test_url)
        print(f"✓ Audio extracted successfully using workaround:")
        print(f"  Audio path: {audio_info.audio_path}")
        print(f"  Format: {audio_info.format}")
        print(f"  Duration: {audio_info.duration} seconds")
        print(f"  Bitrate: {audio_info.bitrate} kbps")
        
        # Verify the audio file exists
        audio_file = Path(audio_info.audio_path)
        if audio_file.exists():
            file_size = audio_file.stat().st_size
            print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
            print("✅ Audio file created successfully!")
        else:
            print("❌ Audio file not found!")
            return False
        
        # Cleanup test files
        print("\n2. Cleaning up test files...")
        await service.cleanup_files([audio_info.audio_path])
        print("✅ Cleanup completed!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

async def test_direct_ffmpeg_extraction():
    """Test direct ffmpeg extraction from a video file."""
    service = YouTubeDLService(download_path="./temp/test_videos")
    
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    print("\nTesting Direct FFmpeg Extraction")
    print("=" * 40)
    
    try:
        # First download a video
        print("1. Downloading video...")
        video_info = await service.download_video(test_url)
        print(f"✓ Video downloaded: {video_info.video_path}")
        
        # Then extract audio using ffmpeg
        print("2. Extracting audio using ffmpeg...")
        audio_info = await service.extract_audio_from_video(video_info.video_path)
        print(f"✓ Audio extracted: {audio_info.audio_path}")
        print(f"  Format: {audio_info.format}")
        print(f"  Duration: {audio_info.duration} seconds")
        
        # Cleanup
        print("3. Cleaning up...")
        await service.cleanup_files([video_info.video_path, audio_info.audio_path])
        print("✅ Cleanup completed!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

async def main():
    """Main test function."""
    print("Audio Extraction Workaround Test Suite")
    print("=" * 50)
    
    # Test 1: Workaround method
    success1 = await test_audio_workaround()
    
    # Test 2: Direct ffmpeg extraction
    success2 = await test_direct_ffmpeg_extraction()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The audio extraction workaround is working correctly.")
        return 0
    else:
        print("\n💥 Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
