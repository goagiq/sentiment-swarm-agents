#!/usr/bin/env python3
"""
Test script to verify Strands configuration system and model refactoring.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.config import config
from loguru import logger


def test_strands_config():
    """Test that Strands configuration is properly set up."""
    
    print("Testing Strands Configuration...")
    
    # Test that all required config fields exist
    assert hasattr(config.model, 'strands_ollama_host')
    assert hasattr(config.model, 'strands_default_model')
    assert hasattr(config.model, 'strands_text_model')
    assert hasattr(config.model, 'strands_vision_model')
    assert hasattr(config.model, 'strands_translation_fast_model')
    
    print("✓ All Strands config fields exist")
    
    # Test that config values are not empty
    assert config.model.strands_ollama_host
    assert config.model.strands_default_model
    assert config.model.strands_text_model
    assert config.model.strands_vision_model
    assert config.model.strands_translation_fast_model
    
    print("✓ All Strands config values are set")
    
    # Test that the default model is not phi
    assert "phi" not in config.model.strands_default_model.lower()
    assert "phi" not in config.model.strands_text_model.lower()
    assert "phi" not in config.model.strands_translation_fast_model.lower()
    
    print("✓ No hardcoded phi models in configuration")
    
    return True


def test_get_strands_model_config():
    """Test the get_strands_model_config method."""
    
    print("Testing get_strands_model_config method...")
    
    # Test different agent types
    agent_types = ["text", "simple_text", "vision", "translation_fast", "default"]
    
    for agent_type in agent_types:
        model_config = config.get_strands_model_config(agent_type)
        
        # Verify required fields exist
        assert "model_id" in model_config
        assert "host" in model_config
        assert "temperature" in model_config
        assert "max_tokens" in model_config
        assert "fallback_model" in model_config
        
        # Verify values are not empty
        assert model_config["model_id"]
        assert model_config["host"]
        assert model_config["fallback_model"]
        
        # Verify no phi models
        assert "phi" not in model_config["model_id"].lower()
        assert "phi" not in model_config["fallback_model"].lower()
        
        print(f"✓ {agent_type} agent config is valid")
    
    return True


def test_agent_refactoring():
    """Test that agents have been refactored to use config instead of hardcoded phi."""
    
    print("Testing agent refactoring...")
    
    # Test text agent
    try:
        from agents.text_agent import TextAgent
        agent = TextAgent()
        print("✓ TextAgent imports successfully")
    except Exception as e:
        print(f"✗ TextAgent import failed: {e}")
        return False
    
    # Test simple text agent
    try:
        from agents.text_agent_simple import SimpleTextAgent
        agent = SimpleTextAgent()
        print("✓ SimpleTextAgent imports successfully")
    except Exception as e:
        print(f"✗ SimpleTextAgent import failed: {e}")
        return False
    
    # Test translation agent
    try:
        from agents.translation_agent import TranslationAgent
        agent = TranslationAgent()
        print("✓ TranslationAgent imports successfully")
    except Exception as e:
        print(f"✗ TranslationAgent import failed: {e}")
        return False
    
    return True


def test_config_consistency():
    """Test that configuration is consistent across different parts of the system."""
    
    print("Testing configuration consistency...")
    
    # Test that ollama_config.py doesn't have hardcoded phi
    try:
        from config.ollama_config import ollama_config
        
        # Check that translation_fast model is not phi
        translation_fast_config = ollama_config.models.get("translation_fast")
        if translation_fast_config:
            assert "phi" not in translation_fast_config.model_id.lower()
            print("✓ ollama_config.py translation_fast model is not phi")
        
    except Exception as e:
        print(f"✗ ollama_config.py check failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    
    logger.info("Starting Strands configuration tests")
    
    tests = [
        test_strands_config,
        test_get_strands_model_config,
        test_agent_refactoring,
        test_config_consistency
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Strands configuration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the configuration.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
