#!/usr/bin/env python3
"""
Test script to verify YouTube download fixes.
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

async def test_youtube_download():
    """Test YouTube download functionality."""
    service = YouTubeDLService(download_path="./temp/test_videos")
    
    # Test URL - using a simple, reliable YouTube video
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll - should be available
    
    print("Testing YouTube download service...")
    print(f"Test URL: {test_url}")
    
    try:
        # Test metadata extraction first
        print("\n1. Testing metadata extraction...")
        metadata = await service.get_metadata(test_url)
        print(f"✓ Metadata extracted successfully:")
        print(f"  Title: {metadata.title}")
        print(f"  Duration: {metadata.duration} seconds")
        print(f"  Platform: {metadata.platform}")
        
        # Test video download
        print("\n2. Testing video download...")
        video_info = await service.download_video(test_url)
        print(f"✓ Video downloaded successfully:")
        print(f"  Title: {video_info.title}")
        print(f"  Video path: {video_info.video_path}")
        print(f"  Duration: {video_info.duration} seconds")
        
        # Test audio extraction
        print("\n3. Testing audio extraction...")
        audio_info = await service.extract_audio(test_url)
        print(f"✓ Audio extracted successfully:")
        print(f"  Audio path: {audio_info.audio_path}")
        print(f"  Format: {audio_info.format}")
        print(f"  Duration: {audio_info.duration} seconds")
        
        print("\n✅ All tests passed! YouTube download service is working correctly.")
        
        # Cleanup test files
        if video_info.video_path:
            await service.cleanup_files([video_info.video_path])
        if audio_info.audio_path:
            await service.cleanup_files([audio_info.audio_path])
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
    
    return True

async def test_error_handling():
    """Test error handling with invalid URLs."""
    service = YouTubeDLService(download_path="./temp/test_videos")
    
    print("\nTesting error handling...")
    
    # Test with invalid URL
    invalid_url = "https://www.youtube.com/watch?v=INVALID_VIDEO_ID"
    
    try:
        await service.get_metadata(invalid_url)
        print("❌ Should have failed with invalid URL")
        return False
    except Exception as e:
        print(f"✓ Correctly handled invalid URL: {type(e).__name__}")
    
    # Test with non-YouTube URL
    non_youtube_url = "https://www.google.com"
    
    try:
        await service.get_metadata(non_youtube_url)
        print("❌ Should have failed with non-YouTube URL")
        return False
    except Exception as e:
        print(f"✓ Correctly handled non-YouTube URL: {type(e).__name__}")
    
    print("✅ Error handling tests passed!")
    return True

async def main():
    """Main test function."""
    print("YouTube Download Service Test Suite")
    print("=" * 40)
    
    # Test 1: Basic functionality
    success1 = await test_youtube_download()
    
    # Test 2: Error handling
    success2 = await test_error_handling()
    
    if success1 and success2:
        print("\n🎉 All tests passed! The YouTube download fixes are working correctly.")
        return 0
    else:
        print("\n💥 Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
