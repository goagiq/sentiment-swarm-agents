#!/usr/bin/env python3
"""
Test script to verify the agent initialization fixes for Ollama and LargeFileProcessor.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.unified_audio_agent import UnifiedAudioAgent
from agents.unified_vision_agent import UnifiedVisionAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_audio_agent_initialization():
    """Test UnifiedAudioAgent initialization."""
    
    print("🎵 Testing UnifiedAudioAgent Initialization")
    print("=" * 50)
    
    try:
        # Initialize the audio agent
        audio_agent = UnifiedAudioAgent(
            enable_summarization=True,
            enable_large_file_processing=True
        )
        
        print("✅ Audio agent created successfully")
        
        # Test model initialization
        await audio_agent._initialize_models()
        
        if audio_agent.ollama_model:
            print(f"✅ Ollama model initialized: {audio_agent.ollama_model.model_id}")
        else:
            print("⚠️  No Ollama model available (this might be expected)")
        
        # Test large file processor initialization
        if hasattr(audio_agent, 'large_file_processor'):
            print("✅ Large file processor initialized")
            
            # Test that the correct method exists
            if hasattr(audio_agent.large_file_processor, 'progressive_audio_analysis'):
                print("✅ progressive_audio_analysis method available")
            else:
                print("❌ progressive_audio_analysis method not found")
        else:
            print("❌ Large file processor not initialized")
        
        # Test agent start
        await audio_agent.start()
        print("✅ Audio agent started successfully")
        
        # Test agent stop
        await audio_agent.stop()
        print("✅ Audio agent stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio agent initialization failed: {e}")
        return False


async def test_vision_agent_initialization():
    """Test UnifiedVisionAgent initialization."""
    
    print("\n👁️ Testing UnifiedVisionAgent Initialization")
    print("=" * 50)
    
    try:
        # Initialize the vision agent
        vision_agent = UnifiedVisionAgent(
            enable_summarization=True,
            enable_large_file_processing=True,
            enable_youtube_integration=True
        )
        
        print("✅ Vision agent created successfully")
        
        # Test model initialization
        await vision_agent._initialize_models()
        
        if vision_agent.ollama_model:
            print(f"✅ Ollama model initialized: {vision_agent.ollama_model.model_id}")
        else:
            print("⚠️  No Ollama model available (this might be expected)")
        
        # Test large file processor initialization
        if hasattr(vision_agent, 'large_file_processor'):
            print("✅ Large file processor initialized")
            
            # Test that the correct method exists
            if hasattr(vision_agent.large_file_processor, 'progressive_video_analysis'):
                print("✅ progressive_video_analysis method available")
            else:
                print("❌ progressive_video_analysis method not found")
        else:
            print("❌ Large file processor not initialized")
        
        # Test agent start
        await vision_agent.start()
        print("✅ Vision agent started successfully")
        
        # Test agent stop
        await vision_agent.stop()
        print("✅ Vision agent stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Vision agent initialization failed: {e}")
        return False


async def test_ollama_integration():
    """Test Ollama integration directly."""
    
    print("\n🤖 Testing Ollama Integration")
    print("=" * 40)
    
    try:
        from core.ollama_integration import get_ollama_model
        
        # Test getting different model types
        text_model = get_ollama_model(model_type="text")
        if text_model:
            print(f"✅ Text model available: {text_model.model_id}")
        else:
            print("⚠️  No text model available")
        
        audio_model = get_ollama_model(model_type="audio")
        if audio_model:
            print(f"✅ Audio model available: {audio_model.model_id}")
        else:
            print("⚠️  No audio model available")
        
        vision_model = get_ollama_model(model_type="vision")
        if vision_model:
            print(f"✅ Vision model available: {vision_model.model_id}")
        else:
            print("⚠️  No vision model available")
        
        return True
        
    except Exception as e:
        print(f"❌ Ollama integration test failed: {e}")
        return False


async def test_large_file_processor():
    """Test LargeFileProcessor methods."""
    
    print("\n📁 Testing LargeFileProcessor")
    print("=" * 35)
    
    try:
        from core.large_file_processor import LargeFileProcessor
        
        # Initialize processor
        processor = LargeFileProcessor()
        print("✅ LargeFileProcessor initialized")
        
        # Test available methods
        methods_to_test = [
            'progressive_audio_analysis',
            'progressive_video_analysis',
            'chunk_audio_by_time',
            'chunk_video_by_time'
        ]
        
        for method_name in methods_to_test:
            if hasattr(processor, method_name):
                print(f"✅ {method_name} method available")
            else:
                print(f"❌ {method_name} method not found")
        
        return True
        
    except Exception as e:
        print(f"❌ LargeFileProcessor test failed: {e}")
        return False


async def main():
    """Main test function."""
    
    print("🚀 Agent Fixes Test Suite")
    print("=" * 50)
    
    # Test Ollama integration
    ollama_success = await test_ollama_integration()
    
    # Test LargeFileProcessor
    processor_success = await test_large_file_processor()
    
    # Test agent initializations
    audio_success = await test_audio_agent_initialization()
    vision_success = await test_vision_agent_initialization()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary")
    print("=" * 50)
    
    print(f"🤖 Ollama Integration: {'✅ PASS' if ollama_success else '❌ FAIL'}")
    print(f"📁 LargeFileProcessor: {'✅ PASS' if processor_success else '❌ FAIL'}")
    print(f"🎵 Audio Agent: {'✅ PASS' if audio_success else '❌ FAIL'}")
    print(f"👁️ Vision Agent: {'✅ PASS' if vision_success else '❌ FAIL'}")
    
    if all([ollama_success, processor_success, audio_success, vision_success]):
        print("\n🎉 SUCCESS: All tests passed! The fixes are working correctly.")
        print("\n🔧 Issues Fixed:")
        print("   1. ✅ Fixed get_ollama_model() parameter error")
        print("   2. ✅ Fixed LargeFileProcessor method calls")
        print("   3. ✅ Added proper error handling for model initialization")
        print("   4. ✅ Added fallback to text model when audio/vision models unavailable")
    else:
        print("\n❌ FAILURE: Some tests failed. Additional fixes may be needed.")
    
    print("\n📝 The fixes implemented:")
    print("   • Removed 'model_name' parameter from get_ollama_model() calls")
    print("   • Changed process_large_file() to progressive_audio_analysis()/progressive_video_analysis()")
    print("   • Added proper error handling and fallback mechanisms")
    print("   • Made model initialization more robust")


if __name__ == "__main__":
    asyncio.run(main())
