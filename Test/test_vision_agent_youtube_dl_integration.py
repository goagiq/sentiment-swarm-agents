#!/usr/bin/env python3
"""
Test script for Vision Agent YouTube-DL integration.
Tests the enhanced vision agent with YouTube-DL capabilities.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.vision_agent import VisionAgent
from core.models import AnalysisRequest, DataType


class VisionAgentYouTubeDLTest:
    """Test class for Vision Agent YouTube-DL integration."""
    
    def __init__(self):
        self.vision_agent = VisionAgent()
        self.test_videos = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll
            "https://www.youtube.com/watch?v=UrT72MNbSI8",  # Another test video
        ]
        self.test_images = [
            "https://example.com/test_image.jpg"  # Placeholder
        ]
    
    async def test_video_metadata_extraction(self):
        """Test video metadata extraction."""
        print("\n=== Testing Video Metadata Extraction ===")
        
        for video_url in self.test_videos:
            try:
                print(f"\nTesting metadata for: {video_url}")
                result = await self.vision_agent.get_video_metadata(video_url)
                
                if result["status"] == "success":
                    metadata = result["content"][0]["json"]
                    print(f"✓ Title: {metadata['title']}")
                    print(f"✓ Duration: {metadata['duration']} seconds")
                    print(f"✓ Platform: {metadata['platform']}")
                    print(f"✓ Views: {metadata['view_count']}")
                else:
                    print(f"✗ Failed: {result['content'][0]['text']}")
                    
            except Exception as e:
                print(f"✗ Error: {e}")
    
    async def test_video_frame_extraction(self):
        """Test video frame extraction."""
        print("\n=== Testing Video Frame Extraction ===")
        
        for video_url in self.test_videos:
            try:
                print(f"\nTesting frame extraction for: {video_url}")
                result = await self.vision_agent.download_video_frames(
                    video_url, num_frames=5
                )
                
                if result["status"] == "success":
                    frame_data = result["content"][0]["json"]
                    print(f"✓ Title: {frame_data['title']}")
                    print(f"✓ Frames extracted: {frame_data['frames_extracted']}")
                    print(f"✓ Frame paths: {len(frame_data['frame_paths'])} files")
                else:
                    print(f"✗ Failed: {result['content'][0]['text']}")
                    
            except Exception as e:
                print(f"✗ Error: {e}")
    
    async def test_video_sentiment_analysis(self):
        """Test video sentiment analysis."""
        print("\n=== Testing Video Sentiment Analysis ===")
        
        for video_url in self.test_videos:
            try:
                print(f"\nTesting sentiment analysis for: {video_url}")
                result = await self.vision_agent.analyze_video_sentiment(video_url)
                
                if result["status"] == "success":
                    analysis_data = result["content"][0]["json"]
                    sentiment = analysis_data["sentiment"]
                    print(f"✓ Title: {analysis_data['title']}")
                    print(f"✓ Frames analyzed: {analysis_data['frames_analyzed']}")
                    print(f"✓ Sentiment: {sentiment['label']}")
                    print(f"✓ Confidence: {sentiment['confidence']:.2f}")
                    print(f"✓ Reasoning: {sentiment['reasoning'][:100]}...")
                else:
                    print(f"✗ Failed: {result['content'][0]['text']}")
                    
            except Exception as e:
                print(f"✗ Error: {e}")
    
    async def test_enhanced_vision_processing(self):
        """Test enhanced vision processing with YouTube-DL."""
        print("\n=== Testing Enhanced Vision Processing ===")
        
        # Test with a video URL
        video_url = self.test_videos[0]
        
        try:
            print(f"\nTesting enhanced processing for: {video_url}")
            
            # Create analysis request
            request = AnalysisRequest(
                id="test_video_001",
                content=video_url,
                data_type=DataType.VIDEO,
                metadata={"source": "youtube"}
            )
            
            # Process with enhanced vision agent
            result = await self.vision_agent.process(request)
            
            print(f"✓ Processing time: {result.processing_time:.2f}s")
            print(f"✓ Status: {result.status}")
            print(f"✓ Sentiment: {result.sentiment.label}")
            print(f"✓ Confidence: {result.sentiment.confidence:.2f}")
            print(f"✓ Extracted text length: {len(result.extracted_text)}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    async def test_error_handling(self):
        """Test error handling for unsupported URLs."""
        print("\n=== Testing Error Handling ===")
        
        unsupported_urls = [
            "https://example.com/video.mp4",
            "https://unsupported-platform.com/video",
            "invalid-url"
        ]
        
        for url in unsupported_urls:
            try:
                print(f"\nTesting unsupported URL: {url}")
                result = await self.vision_agent.get_video_metadata(url)
                
                if result["status"] == "error":
                    print(f"✓ Correctly handled: {result['content'][0]['text']}")
                else:
                    print(f"✗ Unexpected success for unsupported URL")
                    
            except Exception as e:
                print(f"✓ Exception handled: {e}")
    
    async def run_all_tests(self):
        """Run all tests."""
        print("Starting Vision Agent YouTube-DL Integration Tests")
        print("=" * 50)
        
        try:
            # Initialize the agent
            await self.vision_agent._initialize_models()
            print("✓ Vision agent initialized successfully")
            
            # Run tests
            await self.test_video_metadata_extraction()
            await self.test_video_frame_extraction()
            await self.test_video_sentiment_analysis()
            await self.test_enhanced_vision_processing()
            await self.test_error_handling()
            
            print("\n" + "=" * 50)
            print("All tests completed!")
            
        except Exception as e:
            print(f"✗ Test suite failed: {e}")
        finally:
            # Cleanup
            await self.vision_agent.cleanup()


async def main():
    """Main test function."""
    test_suite = VisionAgentYouTubeDLTest()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
