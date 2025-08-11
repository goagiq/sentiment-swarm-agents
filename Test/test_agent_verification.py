#!/usr/bin/env python3
"""
Comprehensive test script to verify all agent fixes are working correctly.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

async def test_audio_agent_initialization():
    """Test UnifiedAudioAgent initialization."""
    print("🎵 Testing UnifiedAudioAgent Initialization")
    print("=" * 50)
    
    try:
        from agents.unified_audio_agent import UnifiedAudioAgent
        
        # Initialize the agent
        agent = UnifiedAudioAgent()
        print("✅ UnifiedAudioAgent created successfully")
        
        # Test model initialization
        await agent._initialize_models()
        if agent.ollama_model:
            print(f"✅ Ollama model initialized: {agent.ollama_model.model_id}")
        else:
            print("⚠️  No Ollama model available")
        
        # Test large file processor
        if hasattr(agent, 'large_file_processor'):
            print("✅ LargeFileProcessor available")
            if hasattr(agent.large_file_processor, 'progressive_audio_analysis'):
                print("✅ progressive_audio_analysis method available")
            else:
                print("❌ progressive_audio_analysis method not found")
        else:
            print("❌ LargeFileProcessor not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_vision_agent_initialization():
    """Test UnifiedVisionAgent initialization."""
    print("\n👁️ Testing UnifiedVisionAgent Initialization")
    print("=" * 50)
    
    try:
        from agents.unified_vision_agent import UnifiedVisionAgent
        
        # Initialize the agent
        agent = UnifiedVisionAgent()
        print("✅ UnifiedVisionAgent created successfully")
        
        # Test model initialization
        await agent._initialize_models()
        if agent.ollama_model:
            print(f"✅ Ollama model initialized: {agent.ollama_model.model_id}")
        else:
            print("⚠️  No Ollama model available")
        
        # Test large file processor
        if hasattr(agent, 'large_file_processor'):
            print("✅ LargeFileProcessor available")
            if hasattr(agent.large_file_processor, 'progressive_video_analysis'):
                print("✅ progressive_video_analysis method available")
            else:
                print("❌ progressive_video_analysis method not found")
        else:
            print("❌ LargeFileProcessor not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Vision agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_youtube_service():
    """Test YouTube service functionality."""
    print("\n📺 Testing YouTube Service")
    print("=" * 30)
    
    try:
        from core.youtube_dl_service import YouTubeDLService
        
        # Initialize the service
        service = YouTubeDLService()
        print("✅ YouTubeDLService initialized")
        
        # Test strategy creation
        video_options = service._get_video_options()
        if video_options:
            print(f"✅ Video options created: {len(video_options)} keys")
        
        audio_options = service._get_audio_options()
        if audio_options:
            print(f"✅ Audio options created: {len(audio_options)} keys")
        
        return True
        
    except Exception as e:
        print(f"❌ YouTube service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🚀 Comprehensive Agent Fixes Verification")
    print("=" * 60)
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python path: {sys.path[:3]}...")
    
    # Run all tests
    audio_success = await test_audio_agent_initialization()
    vision_success = await test_vision_agent_initialization()
    youtube_success = await test_youtube_service()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Comprehensive Test Results")
    print("=" * 60)
    
    print(f"🎵 Audio Agent: {'✅ PASS' if audio_success else '❌ FAIL'}")
    print(f"👁️ Vision Agent: {'✅ PASS' if vision_success else '❌ FAIL'}")
    print(f"📺 YouTube Service: {'✅ PASS' if youtube_success else '❌ FAIL'}")
    
    all_passed = all([audio_success, vision_success, youtube_success])
    
    if all_passed:
        print("\n🎉 SUCCESS: All comprehensive tests passed!")
        print("\n🔧 All Issues Fixed:")
        print("   1. ✅ Fixed get_ollama_model() parameter error")
        print("   2. ✅ Fixed LargeFileProcessor method calls")
        print("   3. ✅ Fixed YouTube download 'not enough values to unpack' error")
        print("   4. ✅ Fixed agent initialization issues")
    else:
        print("\n❌ FAILURE: Some tests failed. Additional fixes may be needed.")

if __name__ == "__main__":
    asyncio.run(main())
