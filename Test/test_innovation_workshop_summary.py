#!/usr/bin/env python3
"""
Test script to generate a comprehensive summary of the Innovation Workshop video.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from loguru import logger
from src.agents.video_summarization_agent import VideoSummarizationAgent
from src.core.models import AnalysisRequest, DataType


async def analyze_innovation_workshop():
    """Analyze the Innovation Workshop video and generate a comprehensive summary."""
    
    # Video file path
    video_path = "data/Innovation Workshop-20250731_151128-Meeting Recording.mp4"
    
    # Check if file exists
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return
    
    logger.info(f"Starting analysis of: {video_path}")
    
    try:
        # Initialize the video summarization agent
        agent = VideoSummarizationAgent()
        
        # Create analysis request
        request = AnalysisRequest(
            data_type=DataType.VIDEO,
            content=video_path,
            language="en"
        )
        
        logger.info("Processing video with VideoSummarizationAgent...")
        
        # Process the video
        result = await agent.process(request)
        
        # Display results
        print("\n" + "="*80)
        print("INNOVATION WORKSHOP VIDEO SUMMARY")
        print("="*80)
        
        print("\n📊 Analysis Results:")
        print(f"   Processing Time: {result.processing_time:.2f} seconds")
        print(f"   Overall Sentiment: {result.sentiment.label} "
              f"(confidence: {result.sentiment.confidence:.2f})")
        
        # Display metadata
        if result.metadata:
            print("\n📋 Analysis Details:")
            for key, value in result.metadata.items():
                if key not in ['agent_id', 'error']:
                    print(f"   {key.replace('_', ' ').title()}: {value}")
        
        # Display summary content
        if 'summary' in result.metadata:
            print("\n📝 Summary:")
            print(f"   {result.metadata['summary']}")
        
        # Display key scenes
        if 'key_scenes' in result.metadata and result.metadata['key_scenes']:
            print("\n🎬 Key Scenes:")
            for i, scene in enumerate(result.metadata['key_scenes'][:5], 1):
                print(f"   {i}. {scene}")
        
        # Display key moments
        if 'key_moments' in result.metadata and result.metadata['key_moments']:
            print("\n⏰ Key Moments:")
            for i, moment in enumerate(result.metadata['key_moments'][:5], 1):
                print(f"   {i}. {moment}")
        
        # Display topics
        if 'topics' in result.metadata and result.metadata['topics']:
            print("\n🏷️  Topics Identified:")
            for i, topic in enumerate(result.metadata['topics'][:5], 1):
                print(f"   {i}. {topic}")
        
        # Display executive summary
        if 'executive_summary' in result.metadata:
            print("\n👔 Executive Summary:")
            print(f"   {result.metadata['executive_summary']}")
        
        print("\n" + "="*80)
        print("Analysis Complete!")
        print("="*80)
        
        # Cleanup
        await agent.cleanup()
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n❌ Error during analysis: {e}")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Run the analysis
    asyncio.run(analyze_innovation_workshop())
