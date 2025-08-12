#!/usr/bin/env python3
"""
Example demonstrating how to use Strands with Ollama using the centralized configuration system.

This example shows how to:
1. Use the centralized config system for Strands Ollama integration
2. Create agents with different model configurations
3. Handle model fallbacks and configuration management
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.config import config
from loguru import logger


def demonstrate_strands_config():
    """Demonstrate the Strands configuration system."""
    
    print("=== Strands Ollama Configuration Demo ===\n")
    
    # Show the current configuration
    print("Current Strands Configuration:")
    print(f"  Host: {config.model.strands_ollama_host}")
    print(f"  Default Model: {config.model.strands_default_model}")
    print(f"  Text Model: {config.model.strands_text_model}")
    print(f"  Vision Model: {config.model.strands_vision_model}")
    print(f"  Translation Fast Model: {config.model.strands_translation_fast_model}")
    print()
    
    # Demonstrate getting model configs for different agent types
    agent_types = ["text", "simple_text", "vision", "translation_fast", "default"]
    
    for agent_type in agent_types:
        model_config = config.get_strands_model_config(agent_type)
        print(f"Model config for '{agent_type}' agent:")
        print(f"  Model ID: {model_config['model_id']}")
        print(f"  Host: {model_config['host']}")
        print(f"  Temperature: {model_config['temperature']}")
        print(f"  Max Tokens: {model_config['max_tokens']}")
        print(f"  Fallback: {model_config['fallback_model']}")
        print()


def demonstrate_strands_usage():
    """Demonstrate how to use Strands with the configuration."""
    
    print("=== Strands Usage Example ===\n")
    
    try:
        # Import Strands (using mock implementation)
        from src.core.strands_mock import Agent
        from src.core.strands_mock import OllamaModel
        
        # Get configuration for text agent
        text_config = config.get_strands_model_config("text")
        
        # Create an Ollama model instance using config
        ollama_model = OllamaModel(
            host=text_config["host"],
            model_id=text_config["model_id"]
        )
        
        # Create an agent using the Ollama model
        agent = Agent(model=ollama_model)
        
        print(f"Created Strands agent with:")
        print(f"  Host: {text_config['host']}")
        print(f"  Model: {text_config['model_id']}")
        print()
        
        # Example usage (commented out to avoid actual API calls)
        # result = agent("Tell me about sentiment analysis.")
        # print("Agent response:", result)
        
        print("Agent created successfully! (API call commented out)")
        
    except ImportError:
        print("Strands not installed. Install with: pip install strands")
        print("This is just a demonstration of the configuration system.")
    except Exception as e:
        print(f"Error creating Strands agent: {e}")


def demonstrate_fallback_config():
    """Demonstrate fallback configuration handling."""
    
    print("=== Fallback Configuration Demo ===\n")
    
    # Show how fallbacks work
    text_config = config.get_strands_model_config("text")
    vision_config = config.get_strands_model_config("vision")
    
    print("Text Agent Fallback Chain:")
    print(f"  Primary: {text_config['model_id']}")
    print(f"  Fallback: {text_config['fallback_model']}")
    print()
    
    print("Vision Agent Fallback Chain:")
    print(f"  Primary: {vision_config['model_id']}")
    print(f"  Fallback: {vision_config['fallback_model']}")
    print()


def demonstrate_environment_override():
    """Demonstrate how environment variables can override config."""
    
    print("=== Environment Override Demo ===\n")
    
    print("You can override these settings with environment variables:")
    print("  STRANDS_OLLAMA_HOST=http://localhost:11434")
    print("  STRANDS_DEFAULT_MODEL=llama3")
    print("  STRANDS_TEXT_MODEL=mistral-small3.1:latest")
    print("  STRANDS_VISION_MODEL=llava:latest")
    print("  STRANDS_TRANSLATION_FAST_MODEL=llama3.2:latest")
    print()
    
    print("Example usage:")
    print("  export STRANDS_TEXT_MODEL=llama3.2:latest")
    print("  python examples/strands_ollama_config_demo.py")
    print()


async def main():
    """Main demonstration function."""
    
    logger.info("Starting Strands Ollama Configuration Demo")
    
    try:
        # Demonstrate the configuration system
        demonstrate_strands_config()
        
        # Demonstrate usage
        demonstrate_strands_usage()
        
        # Demonstrate fallback configuration
        demonstrate_fallback_config()
        
        # Demonstrate environment override
        demonstrate_environment_override()
        
        print("=== Demo Complete ===")
        print("The configuration system is now ready for Strands integration!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
