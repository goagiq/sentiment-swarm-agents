#!/usr/bin/env python3
"""
Test script for audio summarization agent with the OCR audio file.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agents.audio_summarization_agent import AudioSummarizationAgent
from src.core.models import AnalysisRequest, DataType


async def test_audio_summarization():
    """Test audio summarization with the OCR audio file."""
    
    # Audio file path
    audio_file = "data/OCR App with Ollama and Llama Vision - Install Locally.mp3"
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return
    
    print(f"🎵 Testing audio summarization for: {audio_file}")
    print("=" * 60)
    
    try:
        # Initialize the audio summarization agent
        agent = AudioSummarizationAgent()
        
        # Create analysis request
        request = AnalysisRequest(
            data_type=DataType.AUDIO,
            content=audio_file,
            language="en"
        )
        
        print("🔍 Processing audio file...")
        
        # Process the request
        result = await agent.process(request)
        
        print("✅ Audio summarization completed!")
        print("\n📋 Results:")
        print(f"   - Sentiment: {result.sentiment.label}")
        print(f"   - Confidence: {result.sentiment.confidence}")
        print(f"   - Processing Time: {result.processing_time:.2f}s")
        
        if result.extracted_text:
            print(f"\n📝 Extracted Text Preview:")
            preview = result.extracted_text[:500] + "..." if len(result.extracted_text) > 500 else result.extracted_text
            print(f"   {preview}")
        
        if result.metadata:
            print(f"\n🔧 Metadata:")
            for key, value in result.metadata.items():
                print(f"   - {key}: {value}")
        
        # Try to get a summary using the generate_audio_summary tool
        print("\n📄 Generating detailed summary...")
        summary_result = await agent.generate_audio_summary(audio_file)
        
        if summary_result.get("status") == "success":
            print("✅ Summary generated successfully!")
            content = summary_result.get("content", [])
            if content:
                summary = content[0].get("summary", "No summary available")
                print(f"\n📋 Summary:")
                print(summary)
        else:
            print(f"❌ Summary generation failed: {summary_result}")
        
    except Exception as e:
        print(f"❌ Error during audio summarization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_audio_summarization())
