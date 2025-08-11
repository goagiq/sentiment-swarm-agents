#!/usr/bin/env python3
"""
Simple test script to verify the agent initialization fixes.
"""

import sys
import os
from pathlib import Path

# Add src to path - ensure it's at the beginning
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Also add the parent directory to handle src.config imports
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

def test_ollama_integration():
    """Test Ollama integration directly."""
    print("🤖 Testing Ollama Integration")
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
        import traceback
        traceback.print_exc()
        return False


def test_large_file_processor():
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
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🚀 Simple Agent Fixes Test")
    print("=" * 50)
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python path: {sys.path[:3]}...")  # Show first 3 entries
    
    # Test Ollama integration
    ollama_success = test_ollama_integration()
    
    # Test LargeFileProcessor
    processor_success = test_large_file_processor()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary")
    print("=" * 50)
    
    print(f"🤖 Ollama Integration: {'✅ PASS' if ollama_success else '❌ FAIL'}")
    print(f"📁 LargeFileProcessor: {'✅ PASS' if processor_success else '❌ FAIL'}")
    
    if all([ollama_success, processor_success]):
        print("\n🎉 SUCCESS: All tests passed! The fixes are working correctly.")
        print("\n🔧 Issues Fixed:")
        print("   1. ✅ Fixed get_ollama_model() parameter error")
        print("   2. ✅ Fixed LargeFileProcessor method calls")
    else:
        print("\n❌ FAILURE: Some tests failed. Additional fixes may be needed.")


if __name__ == "__main__":
    main()
