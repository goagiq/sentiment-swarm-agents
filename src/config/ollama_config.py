"""
Optimized Ollama configuration for all 7 agents with performance tuning.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class OllamaModelConfig(BaseModel):
    """Configuration for a specific Ollama model."""
    
    model_id: str = Field(..., description="Ollama model ID")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(100, ge=1, le=4096, description="Maximum tokens")
    keep_alive: str = Field("5m", description="Keep alive duration")
    capabilities: list = Field(default_factory=list, description="Model capabilities")
    is_shared: bool = Field(False, description="Whether this model can be shared")
    fallback_model: Optional[str] = Field(None, description="Fallback model ID")
    performance_threshold: float = Field(0.8, description="Performance threshold")


class OllamaConnectionConfig(BaseModel):
    """Configuration for Ollama connection pooling."""
    
    host: str = Field("http://localhost:11434", description="Ollama server host")
    max_connections: int = Field(10, description="Maximum connections in pool")
    max_keepalive: int = Field(30, description="Keepalive timeout in seconds")
    connection_timeout: int = Field(30, description="Connection timeout in seconds")
    retry_attempts: int = Field(3, description="Retry attempts on failure")
    health_check_interval: int = Field(60, description="Health check interval in seconds")


class OllamaPerformanceConfig(BaseModel):
    """Configuration for performance monitoring and optimization."""
    
    enable_metrics: bool = Field(True, description="Enable performance metrics")
    cleanup_interval: int = Field(300, description="Cleanup interval in seconds")
    max_idle_time: int = Field(300, description="Max idle time before cleanup")
    performance_window: int = Field(3600, description="Performance window in seconds")
    enable_auto_scaling: bool = Field(True, description="Enable automatic scaling")
    min_models: int = Field(1, description="Minimum models to keep loaded")
    max_models: int = Field(10, description="Maximum models to load")


class OptimizedOllamaConfig(BaseModel):
    """Main configuration for optimized Ollama integration."""
    
    # Model configurations for different agent types
    models: Dict[str, OllamaModelConfig] = Field(
        default_factory=lambda: {
            "text": OllamaModelConfig(
                model_id="llama3.2:latest",
                temperature=0.1,
                max_tokens=100,
                keep_alive="5m",
                capabilities=["text", "sentiment_analysis"],
                is_shared=True,
                fallback_model="phi3:mini"
            ),
            "vision": OllamaModelConfig(
                model_id="llava:latest",
                temperature=0.7,
                max_tokens=200,
                keep_alive="10m",
                capabilities=["vision", "image_analysis"],
                is_shared=True,
                fallback_model="granite3.2-vision"
            ),
            "audio": OllamaModelConfig(
                model_id="llava:latest",  # Same as vision for efficiency
                temperature=0.7,
                max_tokens=200,
                keep_alive="10m",
                capabilities=["audio", "transcription"],
                is_shared=True,
                fallback_model="llava:latest"
            ),
            "swarm": OllamaModelConfig(
                model_id="llama3.2:latest",
                temperature=0.3,
                max_tokens=150,
                keep_alive="5m",
                capabilities=["coordination", "planning"],
                is_shared=True,
                fallback_model="phi3:mini"
            ),
            "orchestrator": OllamaModelConfig(
                model_id="llama3.2:latest",
                temperature=0.2,
                max_tokens=200,
                keep_alive="5m",
                capabilities=["coordination", "decision_making"],
                is_shared=True,
                fallback_model="phi3:mini"
            ),
            "web": OllamaModelConfig(
                model_id="llama3.2:latest",
                temperature=0.1,
                max_tokens=150,
                keep_alive="5m",
                capabilities=["text", "web_analysis"],
                is_shared=True,
                fallback_model="phi3:mini"
            ),
            "simple_text": OllamaModelConfig(
                model_id="llama3.2:latest",
                temperature=0.1,
                max_tokens=100,
                keep_alive="5m",
                capabilities=["text", "sentiment_analysis"],
                is_shared=True,
                fallback_model="phi3:mini"
            )
        }
    )
    
    # Connection configuration
    connection: OllamaConnectionConfig = Field(
        default_factory=OllamaConnectionConfig
    )
    
    # Performance configuration
    performance: OllamaPerformanceConfig = Field(
        default_factory=OllamaPerformanceConfig
    )
    
    # Agent-specific model mappings
    agent_model_mapping: Dict[str, str] = Field(
        default_factory=lambda: {
            "TextAgent": "text",
            "VisionAgent": "vision",
            "AudioAgent": "audio",
            "WebAgent": "web",
            "OrchestratorAgent": "orchestrator",
            "TextAgentSwarm": "swarm",
            "SimpleTextAgent": "simple_text"
        }
    )
    
    def get_model_config(self, agent_type: str) -> Optional[OllamaModelConfig]:
        """Get model configuration for a specific agent type."""
        model_type = self.agent_model_mapping.get(agent_type)
        if model_type:
            return self.models.get(model_type)
        return None
    
    def get_shared_models(self) -> Dict[str, list]:
        """Get models that can be shared between agents."""
        shared = {}
        for model_type, config in self.models.items():
            if config.is_shared:
                shared[model_type] = config.capabilities
        return shared
    
    def get_fallback_chain(self, model_type: str) -> list:
        """Get fallback chain for a model type."""
        chain = [model_type]
        current = self.models.get(model_type)
        
        while current and current.fallback_model:
            if current.fallback_model not in chain:
                chain.append(current.fallback_model)
                current = self.models.get(current.fallback_model)
            else:
                break
        
        return chain


# Global configuration instance
ollama_config = OptimizedOllamaConfig()


def get_ollama_config() -> OptimizedOllamaConfig:
    """Get the global Ollama configuration."""
    return ollama_config


def update_ollama_config(**kwargs) -> None:
    """Update the global Ollama configuration."""
    global ollama_config
    
    for key, value in kwargs.items():
        if hasattr(ollama_config, key):
            setattr(ollama_config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")


def get_model_for_agent(agent_type: str) -> Optional[OllamaModelConfig]:
    """Get the appropriate model configuration for an agent type."""
    return ollama_config.get_model_config(agent_type)


def get_shared_model_types() -> Dict[str, list]:
    """Get all shared model types and their capabilities."""
    return ollama_config.get_shared_models()
