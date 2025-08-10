#!/usr/bin/env python3
"""
Test script to verify configurable models work correctly.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.model_config import model_config


def test_model_configuration():
    """Test that model configuration works correctly."""
    print("🧪 Testing Configurable Models...")
    
    try:
        # Test text model configuration
        print("\n📋 Testing Text Model Configuration:")
        text_config = model_config.get_text_model_config()
        print(f"  - Model ID: {text_config['model_id']}")
        print(f"  - Fallback Model: {text_config['fallback_model']}")
        print(f"  - Host: {text_config['host']}")
        print(f"  - Temperature: {text_config['temperature']}")
        print(f"  - Max Tokens: {text_config['max_tokens']}")
        
        # Test vision model configuration
        print("\n📋 Testing Vision Model Configuration:")
        vision_config = model_config.get_vision_model_config()
        print(f"  - Model ID: {vision_config['model_id']}")
        print(f"  - Fallback Model: {vision_config['fallback_model']}")
        print(f"  - Host: {vision_config['host']}")
        print(f"  - Temperature: {vision_config['temperature']}")
        print(f"  - Max Tokens: {vision_config['max_tokens']}")
        
        # Test model type routing
        print("\n📋 Testing Model Type Routing:")
        text_model = model_config.get_model_config("text")
        vision_model = model_config.get_model_config("vision")
        audio_model = model_config.get_model_config("audio")
        video_model = model_config.get_model_config("video")
        
        print(f"  - Text model: {text_model['model_id']}")
        print(f"  - Vision model: {vision_model['model_id']}")
        print(f"  - Audio model: {audio_model['model_id']}")
        print(f"  - Video model: {video_model['model_id']}")
        
        # Verify audio and video use vision model
        assert audio_model['model_id'] == vision_model['model_id'], "Audio should use vision model"
        assert video_model['model_id'] == vision_model['model_id'], "Video should use vision model"
        
        # Test Ollama configuration
        print("\n📋 Testing Ollama Configuration:")
        host = model_config.get_ollama_host()
        timeout = model_config.get_ollama_timeout()
        print(f"  - Host: {host}")
        print(f"  - Timeout: {timeout}")
        
        print("\n✅ SUCCESS: All model configurations working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILURE: Model configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_variables():
    """Test that environment variables can override defaults."""
    print("\n🧪 Testing Environment Variable Override...")
    
    try:
        # Set test environment variables
        os.environ["TEXT_MODEL"] = "ollama:test-text-model:latest"
        os.environ["VISION_MODEL"] = "ollama:test-vision-model:latest"
        os.environ["OLLAMA_HOST"] = "http://test-host:11434"
        
        # Create new config instance to pick up environment variables
        from config.model_config import ModelConfig
        test_config = ModelConfig()
        
        # Test that environment variables are used
        text_config = test_config.get_text_model_config()
        vision_config = test_config.get_vision_model_config()
        host = test_config.get_ollama_host()
        
        print(f"  - Text Model (from env): {text_config['model_id']}")
        print(f"  - Vision Model (from env): {vision_config['model_id']}")
        print(f"  - Host (from env): {host}")
        
        # Verify environment variables were used
        assert text_config['model_id'] == "ollama:test-text-model:latest"
        assert vision_config['model_id'] == "ollama:test-vision-model:latest"
        assert host == "http://test-host:11434"
        
        # Clean up environment variables
        del os.environ["TEXT_MODEL"]
        del os.environ["VISION_MODEL"]
        del os.environ["OLLAMA_HOST"]
        
        print("✅ SUCCESS: Environment variable override working")
        return True
        
    except Exception as e:
        print(f"❌ FAILURE: Environment variable test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Configurable Models Tests\n")
    
    # Run tests
    tests = [
        test_model_configuration,
        test_environment_variables
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*50)
    print("📋 CONFIGURABLE MODELS TEST SUMMARY")
    print("="*50)
    
    test_names = [
        "Model Configuration",
        "Environment Variable Override"
    ]
    
    for i, (test_name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {test_name}: {status}")
    
    all_passed = all(results)
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Configurable models are working correctly!")
        print("   - Models can be configured via environment variables ✅")
        print("   - Text and vision models properly separated ✅")
        print("   - Audio/video use vision model ✅")
        print("   - Ollama host is configurable ✅")
    
    return all_passed


if __name__ == "__main__":
    # Run the tests
    success = main()
    sys.exit(0 if success else 1)
