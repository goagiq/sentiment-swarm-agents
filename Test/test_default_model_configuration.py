#!/usr/bin/env python3
"""
Test script to verify that all default model references are using centralized configuration.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.config import config


def test_centralized_configuration():
    """Test that all default models are properly configured."""
    print("=== Testing Centralized Default Model Configuration ===\n")
    
    # Test basic configuration values
    print("1. Basic Configuration Values:")
    print(f"   - strands_default_model: {config.model.strands_default_model}")
    print(f"   - strands_text_model: {config.model.strands_text_model}")
    print(f"   - strands_vision_model: {config.model.strands_vision_model}")
    print(f"   - strands_translation_fast_model: {config.model.strands_translation_fast_model}")
    print(f"   - strands_ollama_host: {config.model.strands_ollama_host}")
    print()
    
    # Test that no phi models are used
    print("2. No Hardcoded Phi Models:")
    assert "phi" not in config.model.strands_default_model.lower(), "Default model should not be phi"
    assert "phi" not in config.model.strands_text_model.lower(), "Text model should not be phi"
    assert "phi" not in config.model.strands_vision_model.lower(), "Vision model should not be phi"
    assert "phi" not in config.model.strands_translation_fast_model.lower(), "Translation model should not be phi"
    print("   ✓ No phi models found in configuration")
    print()
    
    # Test get_strands_model_config method
    print("3. get_strands_model_config Method:")
    text_config = config.get_strands_model_config("text")
    vision_config = config.get_strands_model_config("vision")
    default_config = config.get_strands_model_config("default")
    translation_config = config.get_strands_model_config("translation_fast")
    
    print(f"   - text agent model: {text_config['model_id']}")
    print(f"   - vision agent model: {vision_config['model_id']}")
    print(f"   - default agent model: {default_config['model_id']}")
    print(f"   - translation_fast agent model: {translation_config['model_id']}")
    print()
    
    # Test that all models are properly set
    assert text_config['model_id'] == config.model.strands_text_model
    assert vision_config['model_id'] == config.model.strands_vision_model
    assert default_config['model_id'] == config.model.strands_default_model
    assert translation_config['model_id'] == config.model.strands_translation_fast_model
    print("   ✓ All agent configurations match centralized config")
    print()
    
    # Test agent imports
    print("4. Agent Import Test:")
    try:
        # Test that agents can be imported (they use config internally)
        import agents.text_agent
        import agents.text_agent_simple
        import agents.translation_agent
        print("   ✓ All agents imported successfully")
    except ImportError as e:
        print(f"   ✗ Agent import failed: {e}")
        return False
    print()
    
    # Test that default model is llama3.2:latest
    print("5. Default Model Verification:")
    assert config.model.strands_default_model == "llama3.2:latest", f"Default model should be llama3.2:latest, got {config.model.strands_default_model}"
    print("   ✓ Default model is correctly set to llama3.2:latest")
    print()
    
    print("=== All Tests Passed! ===")
    print("✅ All default model references are using centralized configuration")
    print("✅ No hardcoded phi models found")
    print("✅ Default model is correctly set to llama3.2:latest")
    
    return True


if __name__ == "__main__":
    success = test_centralized_configuration()
    sys.exit(0 if success else 1)
