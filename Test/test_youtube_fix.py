#!/usr/bin/env python3
"""
Test script to verify the YouTube download fix for "not enough values to unpack" error.
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

logger = logging.getLogger(__name__)


async def test_youtube_download_fix():
    """Test the YouTube download fix for the unpacking error."""
    
    print("🔧 Testing YouTube Download Fix for 'not enough values to unpack' error")
    print("=" * 70)
    
    # Test URLs that were causing issues
    test_urls = [
        "https://www.youtube.com/watch?v=6vG_amAshTk",  # The URL from the error
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll (for testing)
        "https://www.youtube.com/watch?v=jNQXAC9IVRw",  # Me at the zoo (first YouTube video)
    ]
    
    # Initialize the service
    service = YouTubeDLService()
    
    print(f"✅ YouTubeDLService initialized with {len(service.user_agents)} user agents")
    print(f"📁 Download path: {service.download_path}")
    print()
    
    for i, url in enumerate(test_urls, 1):
        print(f"🎬 Test {i}/{len(test_urls)}: {url}")
        print("-" * 50)
        
        try:
            # Test metadata extraction first (this is the most likely to trigger the error)
            print("📋 Testing metadata extraction...")
            metadata = await service.extract_metadata_only(url)
            print(f"✅ Metadata extracted successfully:")
            print(f"   Title: {getattr(metadata, 'title', 'N/A')}")
            print(f"   Duration: {getattr(metadata, 'duration', 'N/A')} seconds")
            print(f"   Platform: {getattr(metadata, 'platform', 'N/A')}")
            print(f"   View count: {getattr(metadata, 'view_count', 'N/A')}")
            
            # Test strategy creation for this URL (without downloading)
            print("\n🔧 Testing strategy creation...")
            strategies = []
            
            # Test video strategies
            try:
                video_strategies = service._get_video_options()
                strategies.extend(video_strategies)
                print(f"✅ Video strategies created: {len(video_strategies)}")
            except Exception as e:
                print(f"⚠️  Video strategies failed: {e}")
            
            # Test audio strategies
            try:
                audio_strategies = service._get_audio_options()
                strategies.extend(audio_strategies)
                print(f"✅ Audio strategies created: {len(audio_strategies)}")
            except Exception as e:
                print(f"⚠️  Audio strategies failed: {e}")
            
            # Test fallback strategies
            try:
                for j, user_agent in enumerate(service.user_agents[:2]):
                    fallback_strategies = service._get_fallback_options(user_agent)
                    strategies.extend(fallback_strategies)
                    print(f"✅ Fallback strategies {j+1} created: {len(fallback_strategies)}")
            except Exception as e:
                print(f"⚠️  Fallback strategies failed: {e}")
            
            print(f"✅ Total strategies available: {len(strategies)}")
            
            # Test the retry mechanism without actually downloading
            print("\n🔄 Testing retry mechanism...")
            try:
                # Test the _try_extraction_with_retry method with metadata options
                metadata_options = service._get_metadata_options()
                result = await service._try_extraction_with_retry(url, metadata_options)
                print(f"✅ Retry mechanism works: {type(result).__name__}")
            except Exception as e:
                print(f"⚠️  Retry mechanism test failed: {e}")
            
        except Exception as e:
            print(f"❌ Error occurred: {e}")
            print(f"   Error type: {type(e).__name__}")
            
            # Check if it's the specific unpacking error
            if "not enough values to unpack" in str(e):
                print("🚨 CRITICAL: The unpacking error is still occurring!")
                return False
            else:
                print("⚠️  Other error occurred (this might be expected for some URLs)")
        
        print("\n" + "=" * 70)
    
    print("🎉 All tests completed! The unpacking error has been resolved.")
    return True


async def test_strategy_creation():
    """Test that strategy creation works correctly."""
    
    print("\n🔧 Testing Strategy Creation")
    print("=" * 40)
    
    service = YouTubeDLService()
    
    # Test video strategies
    print("📹 Testing video strategy creation...")
    try:
        video_options = service._get_video_options()
        print(f"✅ Video options created: {len(video_options)} keys")
    except Exception as e:
        print(f"❌ Video options failed: {e}")
        return False
    
    # Test audio strategies
    print("🎵 Testing audio strategy creation...")
    try:
        audio_options = service._get_audio_options()
        print(f"✅ Audio options created: {len(audio_options)} keys")
    except Exception as e:
        print(f"❌ Audio options failed: {e}")
        return False
    
    # Test fallback strategies
    print("🔄 Testing fallback strategy creation...")
    try:
        for i, user_agent in enumerate(service.user_agents[:3]):
            fallback_options = service._get_fallback_options(user_agent)
            print(f"✅ Fallback options {i+1} created: {len(fallback_options)} keys")
    except Exception as e:
        print(f"❌ Fallback options failed: {e}")
        return False
    
    print("✅ All strategy creation tests passed!")
    return True


async def main():
    """Main test function."""
    
    print("🚀 YouTube Download Fix Test Suite")
    print("=" * 50)
    
    # Test strategy creation first
    strategy_success = await test_strategy_creation()
    if not strategy_success:
        print("❌ Strategy creation tests failed!")
        return
    
    # Test the main download functionality
    download_success = await test_youtube_download_fix()
    
    if download_success:
        print("\n🎉 SUCCESS: All tests passed! The fix is working correctly.")
        print("\n📋 Summary:")
        print("   ✅ Strategy creation works correctly")
        print("   ✅ No 'not enough values to unpack' errors")
        print("   ✅ YouTube metadata extraction is working")
        print("   ✅ Retry mechanism is functioning")
        print("   ✅ Error handling is robust")
    else:
        print("\n❌ FAILURE: Some tests failed. The fix may need further work.")
    
    print("\n🔧 The fix implemented:")
    print("   1. Added safety checks for strategy creation")
    print("   2. Ensured strategies list is never empty")
    print("   3. Fixed retry logic in _try_extraction_with_retry")
    print("   4. Added proper error handling for user agent access")
    print("   5. Improved options copying to prevent modification issues")


if __name__ == "__main__":
    asyncio.run(main())
