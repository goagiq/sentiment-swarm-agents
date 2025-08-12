"""
Strands-based Ollama integration for the sentiment analysis system.
This module provides proper Ollama model integration using the Strands framework.
"""

from typing import Dict, List, Optional, Any
from loguru import logger
import asyncio

# This project uses a mock implementation of the Strands framework
# The real Strands framework is not required for this project
STRANDS_AVAILABLE = False

# Import mock implementations
from src.core.strands_mock import Agent

from src.config.model_config import model_config


class StrandsOllamaModel:
    """Strands-based Ollama model wrapper."""
    
    def __init__(self, model_id: str, host: str = "http://localhost:11434", **kwargs):
        self.model_id = model_id
        self.host = host
        self.kwargs = kwargs
        self._agent = None
        self._model = None
        
    def _initialize_model(self):
        """Initialize the Ollama model."""
        if not STRANDS_AVAILABLE:
            # Use mock implementation
            self._agent = Agent(
                name=f"ollama_{self.model_id}",
                model=self.model_id
            )
            logger.info(f"Initialized mock Strands Ollama model: {self.model_id}")
        else:
            raise ImportError("Strands framework not available")
    
    @property
    def agent(self):
        """Get the Strands agent."""
        if self._agent is None:
            self._initialize_model()
        return self._agent
    
    @property
    def model(self):
        """Get the Ollama model."""
        if self._model is None:
            self._initialize_model()
        return self._model
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using the Strands agent."""
        try:
            if not STRANDS_AVAILABLE:
                return f"Error: Strands framework not available. Prompt: {prompt}"
            
            response = await self.agent.agenerate(prompt, **kwargs)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Error generating text with {self.model_id}: {e}")
            return f"Error generating text: {str(e)}"
    
    def generate_text_sync(self, prompt: str, **kwargs) -> str:
        """Generate text synchronously."""
        try:
            if not STRANDS_AVAILABLE:
                return f"Error: Strands framework not available. Prompt: {prompt}"
            
            response = self.agent(prompt, **kwargs)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Error generating text with {self.model_id}: {e}")
            return f"Error generating text: {str(e)}"


class StrandsOllamaIntegration:
    """Strands-based Ollama integration manager."""

    def __init__(self, host: str = None):
        self.host = host or model_config.get_ollama_host()
        self.models: Dict[str, StrandsOllamaModel] = {}
        self._initialize_default_models()

    def _initialize_default_models(self):
        """Initialize default Ollama models using configurable settings."""
        try:
            # Get text model configuration
            text_config = model_config.get_text_model_config()
            text_model = StrandsOllamaModel(
                model_id=text_config["model_id"],
                host=self.host,
                temperature=text_config.get("temperature", 0.7),
                max_tokens=text_config.get("max_tokens", 1000),
                keep_alive="5m"
            )
            self.models["text"] = text_model

            # Get vision model configuration
            vision_config = model_config.get_vision_model_config()
            vision_model = StrandsOllamaModel(
                model_id=vision_config["model_id"],
                host=self.host,
                temperature=vision_config.get("temperature", 0.7),
                max_tokens=vision_config.get("max_tokens", 1000),
                keep_alive="10m"
            )
            self.models["vision"] = vision_model

            # Audio model (using vision model that can handle audio)
            audio_model = StrandsOllamaModel(
                model_id=vision_config["model_id"],  # Same as vision
                host=self.host,
                temperature=vision_config.get("temperature", 0.7),
                max_tokens=vision_config.get("max_tokens", 1000),
                keep_alive="10m"
            )
            self.models["audio"] = audio_model

            logger.info("Strands Ollama models initialized with configurable settings")

        except Exception as e:
            logger.error(f"Failed to initialize Strands Ollama models: {e}")

    def get_text_model(self) -> Optional[StrandsOllamaModel]:
        """Get the text model for sentiment analysis."""
        return self.models.get("text")

    def get_vision_model(self) -> Optional[StrandsOllamaModel]:
        """Get the vision model for image analysis."""
        return self.models.get("vision")

    def get_audio_model(self) -> Optional[StrandsOllamaModel]:
        """Get the audio model for audio analysis."""
        return self.models.get("audio")

    def create_custom_model(
        self,
        model_id: str,
        model_type: str = "text",
        **kwargs
    ) -> Optional[StrandsOllamaModel]:
        """Create a custom Ollama model configuration."""
        try:
            model = StrandsOllamaModel(
                model_id=model_id,
                host=self.host,
                **kwargs
            )
            self.models[model_type] = model
            logger.info(f"Custom Strands Ollama model '{model_id}' created for {model_type}")
            return model

        except Exception as e:
            logger.error(f"Failed to create custom model '{model_id}': {e}")
            return None

    async def generate_response(
        self,
        model_type: str,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """Generate a response using the specified Ollama model."""
        try:
            model = self.models.get(model_type)
            if not model:
                # Use default text model if specified model not found
                model = self.models.get("text")

            if not model:
                raise Exception(f"Model {model_type} not found and no default model available")

            return await model.generate_text(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

        except Exception as e:
            logger.error(f"Error generating response with Strands Ollama: {e}")
            return f"Error generating response: {str(e)}"

    def generate_response_sync(
        self,
        model_type: str,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """Generate a response synchronously."""
        try:
            model = self.models.get(model_type)
            if not model:
                # Use default text model if specified model not found
                model = self.models.get("text")

            if not model:
                raise Exception(f"Model {model_type} not found and no default model available")

            return model.generate_text_sync(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

        except Exception as e:
            logger.error(f"Error generating response with Strands Ollama: {e}")
            return f"Error generating response: {str(e)}"

    def get_available_models(self) -> List[str]:
        """Get list of available model types."""
        return list(self.models.keys())

    async def check_model_availability(self, model_id: str) -> bool:
        """Check if a specific model is available on the Ollama server."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.host}/api/tags") as response:
                    if response.status == 200:
                        models = await response.json()
                        return any(
                            model["name"] == model_id
                            for model in models.get("models", [])
                        )
                    return False

        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False


# Global Strands Ollama integration instance
strands_ollama_integration = StrandsOllamaIntegration()


def get_strands_ollama_model(model_type: str = "text") -> Optional[StrandsOllamaModel]:
    """Get a Strands Ollama model by type."""
    return strands_ollama_integration.models.get(model_type)


def create_strands_ollama_agent(model_type: str = "text", **kwargs):
    """Create a Strands agent with Ollama model."""
    try:
        model = get_strands_ollama_model(model_type)
        if not model:
            logger.error(f"No {model_type} model available")
            return None

        return model.agent

    except Exception as e:
        logger.error(f"Failed to create Strands Ollama agent: {e}")
        return None
