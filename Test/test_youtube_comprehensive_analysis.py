#!/usr/bin/env python3
"""
Test script for comprehensive YouTube video analysis.
Demonstrates the full audio/visual sentiment analysis pipeline.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.youtube_comprehensive_analyzer import YouTubeComprehensiveAnalyzer
from loguru import logger


async def test_comprehensive_analysis():
    """Test the comprehensive YouTube analysis pipeline."""
    
    # Test URLs
    test_urls = [
        "https://www.youtube.com/watch?v=zcikIautNR4",  # The URL from the user's request
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll (for testing)
    ]
    
    analyzer = YouTubeComprehensiveAnalyzer()
    
    print("🎬 Testing Comprehensive YouTube Analysis Pipeline")
    print("=" * 60)
    
    for i, url in enumerate(test_urls, 1):
        print(f"\n📹 Test {i}: Analyzing {url}")
        print("-" * 40)
        
        try:
            # Test with different configurations
            print("🔧 Configuration: Full analysis (audio + visual)")
            result = await analyzer.analyze_youtube_video(
                url,
                extract_audio=True,
                extract_frames=True,
                num_frames=5  # Reduced for faster testing
            )
            
            # Display results
            print(f"✅ Analysis completed!")
            print(f"   Video: {result.video_metadata.get('title', 'Unknown')}")
            print(f"   Duration: {result.video_metadata.get('duration', 0)}s")
            print(f"   Audio Sentiment: {result.audio_sentiment.label} ({result.audio_sentiment.confidence:.2f})")
            print(f"   Visual Sentiment: {result.visual_sentiment.label} ({result.visual_sentiment.confidence:.2f})")
            print(f"   Combined: {result.combined_sentiment.label} ({result.combined_sentiment.confidence:.2f})")
            print(f"   Processing Time: {result.processing_time:.2f}s")
            print(f"   Frames Analyzed: {len(result.extracted_frames)}")
            
            # Test audio-only analysis
            print(f"\n🔧 Configuration: Audio-only analysis")
            audio_result = await analyzer.analyze_youtube_video(
                url,
                extract_audio=True,
                extract_frames=False
            )
            print(f"   Audio Sentiment: {audio_result.audio_sentiment.label} ({audio_result.audio_sentiment.confidence:.2f})")
            
            # Test visual-only analysis
            print(f"\n🔧 Configuration: Visual-only analysis")
            visual_result = await analyzer.analyze_youtube_video(
                url,
                extract_audio=False,
                extract_frames=True,
                num_frames=3
            )
            print(f"   Visual Sentiment: {visual_result.visual_sentiment.label} ({visual_result.visual_sentiment.confidence:.2f})")
            print(f"   Frames Analyzed: {len(visual_result.extracted_frames)}")
            
        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
            logger.error(f"Test {i} failed: {e}")


async def test_batch_analysis():
    """Test batch analysis of multiple videos."""
    
    test_urls = [
        "https://www.youtube.com/watch?v=zcikIautNR4",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    ]
    
    analyzer = YouTubeComprehensiveAnalyzer()
    
    print(f"\n🎬 Testing Batch Analysis ({len(test_urls)} videos)")
    print("=" * 60)
    
    try:
        results = await analyzer.analyze_youtube_urls_batch(
            test_urls,
            extract_audio=True,
            extract_frames=True,
            num_frames=3
        )
        
        print(f"✅ Batch analysis completed!")
        
        for i, result in enumerate(results, 1):
            print(f"\n📹 Video {i}:")
            print(f"   Title: {result.video_metadata.get('title', 'Unknown')}")
            print(f"   Combined Sentiment: {result.combined_sentiment.label} ({result.combined_sentiment.confidence:.2f})")
            print(f"   Processing Time: {result.processing_time:.2f}s")
        
        # Calculate summary
        successful = sum(1 for r in results if r.combined_sentiment.confidence > 0)
        total_time = sum(r.processing_time for r in results)
        avg_time = total_time / len(results) if results else 0
        
        print(f"\n📊 Batch Summary:")
        print(f"   Total Videos: {len(test_urls)}")
        print(f"   Successful: {successful}")
        print(f"   Average Time: {avg_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Batch test failed: {e}")
        logger.error(f"Batch test failed: {e}")


async def test_error_handling():
    """Test error handling with invalid URLs."""
    
    invalid_urls = [
        "https://www.youtube.com/watch?v=invalid_video_id",
        "https://invalid-domain.com/video",
        "not_a_url",
    ]
    
    analyzer = YouTubeComprehensiveAnalyzer()
    
    print(f"\n🎬 Testing Error Handling")
    print("=" * 60)
    
    for url in invalid_urls:
        print(f"\n🔧 Testing invalid URL: {url}")
        
        try:
            result = await analyzer.analyze_youtube_video(url)
            
            if result.video_metadata.get("error"):
                print(f"   ✅ Error properly handled: {result.video_metadata['error']}")
            else:
                print(f"   ⚠️  Unexpected success for invalid URL")
                
        except Exception as e:
            print(f"   ✅ Exception properly caught: {e}")


async def main():
    """Run all tests."""
    print("🚀 Starting YouTube Comprehensive Analysis Tests")
    print("=" * 60)
    
    try:
        # Test 1: Comprehensive analysis
        await test_comprehensive_analysis()
        
        # Test 2: Batch analysis
        await test_batch_analysis()
        
        # Test 3: Error handling
        await test_error_handling()
        
        print(f"\n✅ All tests completed!")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        logger.error(f"Test suite failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
