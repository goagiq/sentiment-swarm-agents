"""
Model manager for handling different model types and capabilities.
"""

from typing import List
from loguru import logger

from src.core.models import (
    ModelType, ModelCapability, ModelConfig, ModelRegistry,
    AnalysisRequest
)
from src.core.ollama_integration import ollama_integration


class ModelManager:
    """Main model manager that coordinates different model types."""
    
    def __init__(self):
        self.registry = ModelRegistry()
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default models in the registry."""
        # Default Ollama text model (fast and lightweight)
        text_model = ModelConfig(
            model_id="llama3.2:latest",  # Use the actual available model
            model_type=ModelType.OLLAMA,
            capabilities=[ModelCapability.TEXT],
            host="http://localhost:11434",
            temperature=0.1,
            max_tokens=100,
            is_default=True
        )
        
        # Default Ollama vision model
        vision_model = ModelConfig(
            model_id="llava:latest",
            model_type=ModelType.OLLAMA,
            capabilities=[ModelCapability.VISION, ModelCapability.TOOL_CALLING],
            host="http://localhost:11434",
            vision_temperature=0.7,
            vision_max_tokens=200,
            is_default=True
        )
        
        # Default Ollama audio model
        audio_model = ModelConfig(
            model_id="llava:latest",  # Can handle audio too
            model_type=ModelType.OLLAMA,
            capabilities=[ModelCapability.AUDIO, ModelCapability.TOOL_CALLING],
            host="http://localhost:11434",
            temperature=0.7,
            max_tokens=200,
            is_default=True
        )
        
        # Register models
        self.registry.register_model(text_model)
        self.registry.register_model(vision_model)
        self.registry.register_model(audio_model)
    
    async def get_model_for_request(
        self, request: AnalysisRequest
    ) -> ModelConfig:
        """Get the appropriate model for a given request."""
        # Check if user specified a model preference
        if request.model_preference:
            model = self.registry.get_model(request.model_preference)
            if model:
                return model
        
        # Get default model for data type
        model = self.registry.get_default_model(request.data_type)
        if model:
            return model
        
        # Fallback to text model for any data type
        return self.registry.get_model("llama3.2:latest")
    
    async def initialize_ollama(self):
        """Initialize Ollama connection."""
        # The ollama_integration is already initialized
        logger.info("Ollama integration initialized")
    
    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Model manager cleanup completed")
    
    def get_available_models(self) -> List[ModelConfig]:
        """Get list of available models."""
        return list(self.registry.models.values())
    
    def register_custom_model(self, model_config: ModelConfig):
        """Register a custom model."""
        self.registry.register_model(model_config)
        logger.info(f"Registered custom model: {model_config.model_id}")
    
    def get_ollama_model(self, model_type: str = "text"):
        """Get an Ollama model by type."""
        if model_type == "text":
            return ollama_integration.get_text_model()
        elif model_type == "vision":
            return ollama_integration.get_vision_model()
        else:
            return ollama_integration.get_audio_model()
    
    def check_model_availability(self, model_id: str) -> bool:
        """Check if a specific model is available on the Ollama server."""
        return ollama_integration.check_model_availability(model_id)
