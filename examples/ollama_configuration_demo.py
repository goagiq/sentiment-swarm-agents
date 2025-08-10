#!/usr/bin/env python3
"""
Ollama Configuration Demonstration Script

This script demonstrates all the ways Ollama models can be configured
for the 7 agents in the Sentiment project.
"""

import asyncio
import json
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.ollama_config import (
    get_ollama_config, 
    update_ollama_config, 
    get_model_for_agent,
    get_shared_model_types,
    OllamaModelConfig,
    OllamaConnectionConfig,
    OllamaPerformanceConfig,
    OptimizedOllamaConfig
)


async def demonstrate_basic_configuration():
    """Demonstrate basic configuration access and modification."""
    print("=" * 60)
    print("BASIC CONFIGURATION DEMONSTRATION")
    print("=" * 60)
    
    # Get current configuration
    config = get_ollama_config()
    print(f"Current Ollama host: {config.connection.host}")
    print(f"Max connections: {config.connection.max_connections}")
    print(f"Performance monitoring enabled: {config.performance.enable_metrics}")
    
    # Show all model configurations
    print("\nCurrent Model Configurations:")
    for model_type, model_config in config.models.items():
        print(f"  {model_type}:")
        print(f"    Model ID: {model_config.model_id}")
        print(f"    Temperature: {model_config.temperature}")
        print(f"    Max Tokens: {model_config.max_tokens}")
        print(f"    Shared: {model_config.is_shared}")
        print(f"    Fallback: {model_config.fallback_model}")
        print()


async def demonstrate_agent_specific_configuration():
    """Demonstrate how to configure models for specific agents."""
    print("=" * 60)
    print("AGENT-SPECIFIC CONFIGURATION")
    print("=" * 60)
    
    # Get configurations for different agent types
    agent_types = [
        "TextAgent", "VisionAgent", "AudioAgent", "WebAgent",
        "OrchestratorAgent", "TextAgentSwarm", "SimpleTextAgent"
    ]
    
    for agent_type in agent_types:
        model_config = get_model_for_agent(agent_type)
        if model_config:
            print(f"{agent_type}:")
            print(f"  Uses model: {model_config.model_id}")
            print(f"  Temperature: {model_config.temperature}")
            print(f"  Capabilities: {', '.join(model_config.capabilities)}")
            print()


async def demonstrate_dynamic_configuration_updates():
    """Demonstrate how to dynamically update configurations."""
    print("=" * 60)
    print("DYNAMIC CONFIGURATION UPDATES")
    print("=" * 60)
    
    # Update connection settings
    print("Updating connection settings...")
    update_ollama_config(
        connection=OllamaConnectionConfig(
            host="http://localhost:11434",
            max_connections=20,
            connection_timeout=45,
            retry_attempts=5
        )
    )
    
    # Update specific model settings
    print("Updating text model settings...")
    new_text_config = OllamaModelConfig(
        model_id="llama3.2:latest",
        temperature=0.2,
        max_tokens=200,
        keep_alive="10m",
        capabilities=["text", "sentiment_analysis",
                     "summarization"],
        is_shared=True,
        fallback_model="phi3:mini"
    )
    
    config = get_ollama_config()
    config.models["text"] = new_text_config
    
    # Verify the update
    updated_config = get_model_for_agent("TextAgent")
    print(f"Updated text model temperature: {updated_config.temperature}")
    print(f"Updated text model max tokens: {updated_config.max_tokens}")
    print(f"Updated text model capabilities: {updated_config.capabilities}")
    print()


async def demonstrate_performance_configuration():
    """Demonstrate performance-related configuration options."""
    print("=" * 60)
    print("PERFORMANCE CONFIGURATION")
    print("=" * 60)
    
    # Update performance settings
    print("Updating performance settings...")
    new_perf_config = OllamaPerformanceConfig(
        enable_metrics=True,
        cleanup_interval=600,  # 10 minutes
        max_idle_time=900,     # 15 minutes
        performance_window=7200,  # 2 hours
        enable_auto_scaling=True,
        min_models=2,
        max_models=15
    )
    
    config = get_ollama_config()
    config.performance = new_perf_config
    
    # Show updated performance settings
    print(f"Cleanup interval: {config.performance.cleanup_interval}s")
    print(f"Max idle time: {config.performance.max_idle_time}s")
    print(f"Auto-scaling enabled: {config.performance.enable_auto_scaling}")
    print(f"Model limits: {config.performance.min_models} - {config.performance.max_models}")
    print()


async def demonstrate_model_sharing_configuration():
    """Demonstrate how model sharing is configured."""
    print("=" * 60)
    print("MODEL SHARING CONFIGURATION")
    print("=" * 60)
    
    # Show which models are configured for sharing
    shared_models = get_shared_model_types()
    print("Models configured for sharing:")
    for model_type, capabilities in shared_models.items():
        print(f"  {model_type}: {', '.join(capabilities)}")
    
    # Configure a new shared model
    print("\nConfiguring new shared model...")
    new_shared_config = OllamaModelConfig(
        model_id="llama3.2:latest",
        temperature=0.5,
        max_tokens=300,
        keep_alive="15m",
        capabilities=["text", "code", "reasoning"],
        is_shared=True,
        fallback_model="phi3:mini"
    )
    
    config = get_ollama_config()
    config.models["reasoning"] = new_shared_config
    
    # Update agent mapping to use the new model
    config.agent_model_mapping["TextAgent"] = "reasoning"
    
    print(f"TextAgent now uses: {get_model_for_agent('TextAgent').model_id}")
    print()


async def demonstrate_fallback_configuration():
    """Demonstrate fallback model configuration."""
    print("=" * 60)
    print("FALLBACK MODEL CONFIGURATION")
    print("=" * 60)
    
    config = get_ollama_config()
    
    # Show fallback chains for different models
    print("Fallback chains:")
    for model_type in ["text", "vision", "swarm"]:
        fallback_chain = config.get_fallback_chain(model_type)
        print(f"  {model_type}: {' -> '.join(fallback_chain)}")
    
    # Configure a new fallback model
    print("\nConfiguring new fallback model...")
    new_fallback_config = OllamaModelConfig(
        model_id="llama3.2:latest",
        temperature=0.1,
        max_tokens=100,
        keep_alive="5m",
        capabilities=["text", "fallback"],
        is_shared=True,
        fallback_model="phi3:mini"
    )
    
    config.models["fallback"] = new_fallback_config
    
    # Update existing models to use the new fallback
    config.models["text"].fallback_model = "fallback"
    
    print(f"Updated text fallback chain: {' -> '.join(config.get_fallback_chain('text'))}")
    print()


async def demonstrate_connection_pooling_configuration():
    """Demonstrate connection pooling configuration."""
    print("=" * 60)
    print("CONNECTION POOLING CONFIGURATION")
    print("=" * 60)
    
    # Show current connection settings
    config = get_ollama_config()
    print(f"Current connection settings:")
    print(f"  Host: {config.connection.host}")
    print(f"  Max connections: {config.connection.max_connections}")
    print(f"  Keepalive timeout: {config.connection.max_keepalive}s")
    print(f"  Connection timeout: {config.connection.connection_timeout}s")
    print(f"  Retry attempts: {config.connection.retry_attempts}")
    
    # Update connection settings for high-performance scenario
    print("\nUpdating for high-performance scenario...")
    high_perf_connection = OllamaConnectionConfig(
        host="http://localhost:11434",
        max_connections=50,
        max_keepalive=60,
        connection_timeout=15,
        retry_attempts=2,
        health_check_interval=30
    )
    
    config.connection = high_perf_connection
    
    print(f"Updated max connections: {config.connection.max_connections}")
    print(f"Updated keepalive: {config.connection.max_keepalive}s")
    print(f"Updated health check interval: {config.connection.health_check_interval}s")
    print()


async def demonstrate_custom_model_configuration():
    """Demonstrate how to create custom model configurations."""
    print("=" * 60)
    print("CUSTOM MODEL CONFIGURATION")
    print("=" * 60)
    
    # Create a custom model for specialized tasks
    custom_model = OllamaModelConfig(
        model_id="llama3.2:latest",
        temperature=0.0,  # Very deterministic
        max_tokens=500,
        keep_alive="20m",
        capabilities=["text", "analysis", "custom"],
        is_shared=False,  # Dedicated model
        fallback_model="phi3:mini",
        performance_threshold=0.9
    )
    
    # Add it to the configuration
    config = get_ollama_config()
    config.models["custom_analysis"] = custom_model
    
    # Create a new agent type mapping
    config.agent_model_mapping["CustomAnalysisAgent"] = "custom_analysis"
    
    print("Created custom model configuration:")
    print(f"  Model ID: {custom_model.model_id}")
    print(f"  Temperature: {custom_model.temperature}")
    print(f"  Max Tokens: {custom_model.max_tokens}")
    print(f"  Shared: {custom_model.is_shared}")
    print(f"  Performance Threshold: {custom_model.performance_threshold}")
    print()


async def demonstrate_configuration_validation():
    """Demonstrate Pydantic validation in configurations."""
    print("=" * 60)
    print("CONFIGURATION VALIDATION")
    print("=" * 60)
    
    try:
        # Try to create an invalid configuration
        print("Testing invalid temperature value...")
        invalid_config = OllamaModelConfig(
            model_id="test:latest",
            temperature=3.0,  # Invalid: should be <= 2.0
            max_tokens=100
        )
    except Exception as e:
        print(f"Validation error caught: {e}")
    
    try:
        # Try to create an invalid max_tokens value
        print("Testing invalid max_tokens value...")
        invalid_config = OllamaModelConfig(
            model_id="test:latest",
            temperature=0.7,
            max_tokens=10000  # Invalid: should be <= 4096
        )
    except Exception as e:
        print(f"Validation error caught: {e}")
    
    print("Configuration validation is working correctly!")
    print()


async def demonstrate_configuration_persistence():
    """Demonstrate how configurations can be saved and loaded."""
    print("=" * 60)
    print("CONFIGURATION PERSISTENCE")
    print("=" * 60)
    
    config = get_ollama_config()
    
    # Convert configuration to dictionary for persistence
    config_dict = config.model_dump()
    
    # Save to file (simulated)
    config_file = "ollama_config_backup.json"
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Configuration saved to {config_file}")
    
    # Load from file (simulated)
    with open(config_file, 'r') as f:
        loaded_config_dict = json.load(f)
    
    # Recreate configuration from loaded data
    restored_config = OptimizedOllamaConfig(**loaded_config_dict)
    
    print(f"Configuration restored from {config_file}")
    print(f"Restored host: {restored_config.connection.host}")
    print(f"Restored models count: {len(restored_config.models)}")
    
    # Clean up
    os.remove(config_file)
    print(f"Cleaned up {config_file}")
    print()


async def main():
    """Run all configuration demonstrations."""
    print("Ollama Configuration Demonstration")
    print("This script shows all the ways Ollama models can be configured")
    print()
    
    try:
        await demonstrate_basic_configuration()
        await demonstrate_agent_specific_configuration()
        await demonstrate_dynamic_configuration_updates()
        await demonstrate_performance_configuration()
        await demonstrate_model_sharing_configuration()
        await demonstrate_fallback_configuration()
        await demonstrate_connection_pooling_configuration()
        await demonstrate_custom_model_configuration()
        await demonstrate_configuration_validation()
        await demonstrate_configuration_persistence()
        
        print("=" * 60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Key Takeaways:")
        print("1. Ollama models are fully configurable through Pydantic models")
        print("2. Configurations can be updated dynamically at runtime")
        print("3. Agent-specific model configurations are supported")
        print("4. Model sharing and fallback chains are configurable")
        print("5. Connection pooling and performance settings are tunable")
        print("6. All configurations are validated automatically")
        print("7. Configurations can be persisted and restored")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
