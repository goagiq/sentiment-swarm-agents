#!/usr/bin/env python3
"""
Advanced video analysis script to extract actual content from the Innovation Workshop video.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from loguru import logger
    from src.agents.unified_vision_agent import UnifiedVisionAgent
    from src.core.models import AnalysisRequest, DataType
    from src.core.youtube_comprehensive_analyzer import YouTubeComprehensiveAnalyzer
    from src.core.large_file_processor import LargeFileProcessor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("This script requires the full sentiment analysis system dependencies.")
    sys.exit(1)


async def analyze_video_content():
    """Analyze the Innovation Workshop video content using advanced tools."""
    
    video_path = "data/Innovation Workshop-20250731_151128-Meeting Recording.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    print("="*80)
    print("ADVANCED VIDEO CONTENT ANALYSIS")
    print("="*80)
    print(f"Analyzing: {video_path}")
    
    try:
        # Method 1: Try UnifiedVisionAgent
        print("\n🔍 Method 1: Using UnifiedVisionAgent...")
        try:
            agent = UnifiedVisionAgent()
            request = AnalysisRequest(
                data_type=DataType.VIDEO,
                content=video_path,
                language="en"
            )
            
            result = await agent.process(request)
            
            print("✅ UnifiedVisionAgent Analysis Results:")
            print(f"   Sentiment: {result.sentiment.label} (confidence: {result.sentiment.confidence:.2f})")
            print(f"   Processing Time: {result.processing_time:.2f} seconds")
            
            if result.metadata:
                print("\n📋 Extracted Content:")
                for key, value in result.metadata.items():
                    if key not in ['agent_id', 'error']:
                        print(f"   {key}: {value}")
            
            await agent.cleanup()
            
        except Exception as e:
            print(f"❌ UnifiedVisionAgent failed: {e}")
        
        # Method 2: Try YouTubeComprehensiveAnalyzer
        print("\n🔍 Method 2: Using YouTubeComprehensiveAnalyzer...")
        try:
            analyzer = YouTubeComprehensiveAnalyzer()
            result = await analyzer.analyze_local_video(video_path)
            
            print("✅ YouTubeComprehensiveAnalyzer Results:")
            print(f"   Analysis completed: {result}")
            
        except Exception as e:
            print(f"❌ YouTubeComprehensiveAnalyzer failed: {e}")
        
        # Method 3: Try LargeFileProcessor
        print("\n🔍 Method 3: Using LargeFileProcessor...")
        try:
            processor = LargeFileProcessor(
                chunk_duration=300,  # 5 minutes
                max_workers=2,
                cache_dir="./cache/video",
                temp_dir="./temp/video"
            )
            
            # Process video in chunks
            result = await processor.progressive_video_analysis(
                video_path, 
                lambda chunk_path: {"chunk_processed": chunk_path}
            )
            
            print("✅ LargeFileProcessor Results:")
            print(f"   Chunks processed: {len(result) if isinstance(result, list) else 'Unknown'}")
            
        except Exception as e:
            print(f"❌ LargeFileProcessor failed: {e}")
        
    except Exception as e:
        print(f"❌ Overall analysis failed: {e}")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


def main():
    """Main function."""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Run the analysis
    asyncio.run(analyze_video_content())


if __name__ == "__main__":
    main()
