#!/usr/bin/env python3
"""
Test script to demonstrate improved YouTube video handling with user feedback.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.web_agent import WebAgent
from core.models import AnalysisRequest, DataType


async def test_youtube_video_handling():
    """Test YouTube video URL handling with improved user feedback."""
    
    print("🧪 Testing YouTube Video Handling with User Feedback")
    print("=" * 60)
    
    # Test URLs
    test_urls = [
        "https://www.youtube.com/watch?v=UrT72MNbSI8",
        "https://youtu.be/UrT72MNbSI8",
        "https://vimeo.com/123456789",
        "https://www.tiktok.com/@user/video/123456789",
        "https://www.instagram.com/p/ABC123/",
        "https://www.foxnews.com/travel/meteorite-fragment-slammed-through-homeowners-roof-billions-years-old-predates-earth-professor"
    ]
    
    web_agent = WebAgent()
    
    for url in test_urls:
        print(f"\n🔗 Testing URL: {url}")
        print("-" * 40)
        
        try:
            # Create analysis request
            request = AnalysisRequest(
                data_type=DataType.WEBPAGE,
                content=url
            )
            
            # Process the request
            result = await web_agent.process(request)
            
            # Display results
            print(f"✅ Status: {result.status}")
            print(f"📊 Sentiment: {result.sentiment.label}")
            print(f"🎯 Confidence: {result.sentiment.confidence:.2f}")
            
            # Check for video platform warnings
            if result.metadata and "warning" in result.metadata:
                print(f"⚠️  Warning: {result.metadata['warning']}")
                print(f"📺 Platform: {result.metadata.get('platform', 'Unknown')}")
                print(f"🚫 Limitation: {result.metadata.get('limitation', 'None')}")
                
                suggestions = result.metadata.get('suggestions', [])
                if suggestions:
                    print("💡 Suggestions:")
                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"   {i}. {suggestion}")
                
                available_content = result.metadata.get('available_content', '')
                if available_content:
                    print(f"📄 Available Content Preview: {available_content[:100]}...")
            
            elif result.metadata and "error" in result.metadata:
                print(f"❌ Error: {result.metadata['error']}")
            
            else:
                print("✅ Regular webpage analysis completed successfully")
                if result.extracted_text:
                    print(f"📄 Content Preview: {result.extracted_text[:100]}...")
            
        except Exception as e:
            print(f"❌ Error processing URL: {e}")
        
        print()


async def test_video_platform_detection():
    """Test the video platform detection functionality."""
    
    print("\n🔍 Testing Video Platform Detection")
    print("=" * 40)
    
    web_agent = WebAgent()
    
    test_urls = [
        "https://www.youtube.com/watch?v=UrT72MNbSI8",
        "https://youtu.be/UrT72MNbSI8", 
        "https://vimeo.com/123456789",
        "https://www.tiktok.com/@user/video/123456789",
        "https://www.instagram.com/p/ABC123/",
        "https://www.foxnews.com/article",
        "https://www.google.com"
    ]
    
    for url in test_urls:
        video_info = web_agent._is_video_platform(url)
        print(f"\n🔗 {url}")
        print(f"   Is Video: {video_info['is_video']}")
        if video_info['is_video']:
            print(f"   Platform: {video_info['platform']}")
            print(f"   Limitation: {video_info['limitation']}")
            print(f"   Suggestions: {len(video_info['suggestions'])} available")


def main():
    """Main test function."""
    print("🚀 YouTube Video Handling Test Suite")
    print("=" * 50)
    
    # Run tests
    asyncio.run(test_video_platform_detection())
    asyncio.run(test_youtube_video_handling())
    
    print("\n✅ Test completed!")
    print("\n📋 Summary of Improvements:")
    print("1. ✅ Video platform detection (YouTube, Vimeo, TikTok, Instagram)")
    print("2. ✅ Clear user warnings about video content limitations")
    print("3. ✅ Helpful suggestions for alternative analysis methods")
    print("4. ✅ Better error handling and user feedback")
    print("5. ✅ Content preview of available metadata")


if __name__ == "__main__":
    main()
