#!/usr/bin/env python3
"""
Test script to verify large file processing integration.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from src.agents.audio_summarization_agent import AudioSummarizationAgent
from src.agents.video_summarization_agent import VideoSummarizationAgent
from src.core.models import AnalysisRequest, DataType

async def test_large_file_integration():
    """Test that large file processing is properly integrated."""
    
    print(f"\n{'='*60}")
    print(f"Testing Large File Processing Integration")
    print(f"{'='*60}")
    
    try:
        # Test Audio Summarization Agent
        print(f"\n🔍 Testing Audio Summarization Agent...")
        audio_agent = AudioSummarizationAgent()
        
        # Check capabilities
        capabilities = audio_agent.metadata.get('capabilities', [])
        print(f"✅ Audio Agent Capabilities: {capabilities}")
        
        # Verify large file processing capability
        if 'large_file_processing' in capabilities:
            print(f"✅ Large file processing capability found")
        else:
            print(f"❌ Large file processing capability missing")
            return False
        
        # Test Video Summarization Agent
        print(f"\n🔍 Testing Video Summarization Agent...")
        video_agent = VideoSummarizationAgent()
        
        # Check capabilities
        capabilities = video_agent.metadata.get('capabilities', [])
        print(f"✅ Video Agent Capabilities: {capabilities}")
        
        # Verify large file processing capability
        if 'large_file_processing' in capabilities:
            print(f"✅ Large file processing capability found")
        else:
            print(f"❌ Large file processing capability missing")
            return False
        
        # Test LargeFileProcessor import
        print(f"\n🔍 Testing LargeFileProcessor import...")
        try:
            from src.core.large_file_processor import LargeFileProcessor
            processor = LargeFileProcessor()
            print(f"✅ LargeFileProcessor imported and initialized successfully")
        except Exception as e:
            print(f"❌ LargeFileProcessor import failed: {e}")
            return False
        
        # Test agent initialization with large file processor
        print(f"\n🔍 Testing agent initialization...")
        if hasattr(audio_agent, 'large_file_processor'):
            print(f"✅ Audio agent has large_file_processor attribute")
        else:
            print(f"❌ Audio agent missing large_file_processor attribute")
            return False
        
        if hasattr(video_agent, 'large_file_processor'):
            print(f"✅ Video agent has large_file_processor attribute")
        else:
            print(f"❌ Video agent missing large_file_processor attribute")
            return False
        
        # Test file size detection methods
        print(f"\n🔍 Testing file size detection...")
        if hasattr(audio_agent, '_get_file_size'):
            print(f"✅ Audio agent has _get_file_size method")
        else:
            print(f"❌ Audio agent missing _get_file_size method")
            return False
        
        if hasattr(video_agent, '_get_file_size'):
            print(f"✅ Video agent has _get_file_size method")
        else:
            print(f"❌ Video agent missing _get_file_size method")
            return False
        
        # Test large file processing methods
        print(f"\n🔍 Testing large file processing methods...")
        if hasattr(audio_agent, '_process_large_audio_file'):
            print(f"✅ Audio agent has _process_large_audio_file method")
        else:
            print(f"❌ Audio agent missing _process_large_audio_file method")
            return False
        
        if hasattr(video_agent, '_process_large_video_file'):
            print(f"✅ Video agent has _process_large_video_file method")
        else:
            print(f"❌ Video agent missing _process_large_video_file method")
            return False
        
        print(f"\n🎉 All integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        logger.exception("Detailed error information:")
        return False
        
    finally:
        # Cleanup
        try:
            await audio_agent.cleanup()
            await video_agent.cleanup()
            print(f"🧹 Cleanup completed")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")

if __name__ == "__main__":
    print("Starting large file processing integration test...")
    success = asyncio.run(test_large_file_integration())
    if success:
        print("\n✅ Integration test completed successfully!")
    else:
        print("\n❌ Integration test failed!")
        sys.exit(1)
