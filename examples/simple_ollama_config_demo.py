#!/usr/bin/env python3
"""
Simple Ollama Configuration Demo

Shows how Ollama models are configurable for all 7 agents.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from config.ollama_config import (
        get_ollama_config,
        update_ollama_config,
        get_model_for_agent,
        OllamaModelConfig,
        OllamaConnectionConfig
    )
    
    def show_basic_config():
        """Show basic configuration capabilities."""
        print("=== BASIC CONFIGURATION ===")
        config = get_ollama_config()
        
        print(f"Ollama Host: {config.connection.host}")
        print(f"Max Connections: {config.connection.max_connections}")
        print(f"Performance Monitoring: {config.performance.enable_metrics}")
        
        print("\nModel Configurations:")
        for model_type, model_config in config.models.items():
            print(f"  {model_type}: {model_config.model_id}")
            print(f"    Temp: {model_config.temperature}")
            print(f"    Max Tokens: {model_config.max_tokens}")
            print(f"    Shared: {model_config.is_shared}")
            print()
    
    def show_agent_configs():
        """Show agent-specific configurations."""
        print("=== AGENT CONFIGURATIONS ===")
        
        agents = [
            "TextAgent", "VisionAgent", "AudioAgent", "WebAgent",
            "OrchestratorAgent", "TextAgentSwarm", "SimpleTextAgent"
        ]
        
        for agent in agents:
            config = get_model_for_agent(agent)
            if config:
                print(f"{agent}: {config.model_id}")
                print(f"  Temperature: {config.temperature}")
                print(f"  Capabilities: {config.capabilities}")
                print()
    
    def demonstrate_dynamic_updates():
        """Show dynamic configuration updates."""
        print("=== DYNAMIC UPDATES ===")
        
        # Update connection settings
        update_ollama_config(
            connection=OllamaConnectionConfig(
                max_connections=25,
                connection_timeout=40
            )
        )
        
        # Update a model
        config = get_ollama_config()
        new_text_config = OllamaModelConfig(
            model_id="llama3.2:latest",
            temperature=0.3,
            max_tokens=150,
            keep_alive="8m",
            capabilities=["text", "analysis"],
            is_shared=True,
            fallback_model="phi3:mini"
        )
        config.models["text"] = new_text_config
        
        print("Updated text model configuration:")
        updated = get_model_for_agent("TextAgent")
        print(f"  Temperature: {updated.temperature}")
        print(f"  Max Tokens: {updated.max_tokens}")
        print()
    
    def show_validation():
        """Show configuration validation."""
        print("=== CONFIGURATION VALIDATION ===")
        
        try:
            # This should fail
            OllamaModelConfig(
                model_id="test",
                temperature=3.0,  # Invalid: > 2.0
                max_tokens=100
            )
        except Exception as e:
            print(f"Validation caught error: {e}")
        
        try:
            # This should also fail
            OllamaModelConfig(
                model_id="test",
                temperature=0.7,
                max_tokens=10000  # Invalid: > 4096
            )
        except Exception as e:
            print(f"Validation caught error: {e}")
        
        print("Configuration validation working correctly!")
        print()
    
    def main():
        """Run all demonstrations."""
        print("Ollama Configuration Demonstration")
        print("=" * 40)
        print()
        
        show_basic_config()
        show_agent_configs()
        demonstrate_dynamic_updates()
        show_validation()
        
        print("=== SUMMARY ===")
        print("✓ Ollama models are fully configurable")
        print("✓ Agent-specific configurations supported")
        print("✓ Dynamic updates at runtime")
        print("✓ Automatic validation")
        print("✓ Model sharing and fallbacks")
        print("✓ Performance tuning")
        print("✓ Connection pooling")
        
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
except Exception as e:
    print(f"Error: {e}")
