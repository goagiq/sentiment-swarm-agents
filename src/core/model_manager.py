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
        from src.config.config import config
        
        # Default Ollama text model (fast and lightweight)
        text_model = ModelConfig(
            model_id=config.model.strands_text_model,
            model_type=ModelType.OLLAMA,
            capabilities=[ModelCapability.TEXT],
            host=config.model.strands_ollama_host,
            temperature=config.model.text_temperature,
            max_tokens=config.model.text_max_tokens,
            is_default=True
        )
        
        # Default Ollama vision model
        vision_model = ModelConfig(
            model_id=config.model.strands_vision_model,
            model_type=ModelType.OLLAMA,
            capabilities=[ModelCapability.VISION, ModelCapability.TOOL_CALLING],
            host=config.model.strands_ollama_host,
            vision_temperature=config.model.vision_temperature,
            vision_max_tokens=config.model.vision_max_tokens,
            is_default=True
        )
        
        # Default Ollama audio model
        audio_model = ModelConfig(
            model_id=config.model.strands_vision_model,  # Same as vision
            model_type=ModelType.OLLAMA,
            capabilities=[ModelCapability.AUDIO, ModelCapability.TOOL_CALLING],
            host=config.model.strands_ollama_host,
            temperature=config.model.vision_temperature,
            max_tokens=config.model.vision_max_tokens,
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
        from src.config.config import config
        return self.registry.get_model(config.model.strands_text_model)
    
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
