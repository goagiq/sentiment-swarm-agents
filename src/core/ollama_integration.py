"""
Ollama integration for the sentiment analysis system.
This module provides Ollama model integration with proper fallback handling.
"""

from typing import Dict, List, Optional
from loguru import logger
from src.config.model_config import model_config


class OllamaModel:
    """Simple Ollama model representation."""
    def __init__(self, **kwargs):
        self.host = kwargs.get('host', 'http://localhost:11434')
        self.model_id = kwargs.get('model_id', 'unknown')
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 100)
        self.keep_alive = kwargs.get('keep_alive', '5m')
    
    def __str__(self):
        return f"OllamaModel({self.model_id})"
    
    def __repr__(self):
        return self.__str__()


class OllamaIntegration:
    """Ollama integration manager."""
    
    def __init__(self, host: str = None):
        # Use configurable host or default
        self.host = host or model_config.get_ollama_host()
        self.models: Dict[str, OllamaModel] = {}
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default Ollama models using configurable settings."""
        try:
            # Get text model configuration
            text_config = model_config.get_text_model_config()
            text_model = OllamaModel(
                host=self.host,
                model_id=text_config["model_id"],
                temperature=text_config["temperature"],
                max_tokens=text_config["max_tokens"],
                keep_alive="5m"
            )
            self.models["text"] = text_model
            
            # Get vision model configuration (for audio, video, image)
            vision_config = model_config.get_vision_model_config()
            vision_model = OllamaModel(
                host=self.host,
                model_id=vision_config["model_id"],
                temperature=vision_config["temperature"],
                max_tokens=vision_config["max_tokens"],
                keep_alive="10m"
            )
            self.models["vision"] = vision_model
            
            # Audio model (using vision model that can handle audio)
            audio_model = OllamaModel(
                host=self.host,
                model_id=vision_config["model_id"],  # Same as vision
                temperature=vision_config["temperature"],
                max_tokens=vision_config["max_tokens"],
                keep_alive="10m"
            )
            self.models["audio"] = audio_model
            
            logger.info("Ollama models initialized with configurable settings")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama models: {e}")
    
    def get_text_model(self) -> Optional[OllamaModel]:
        """Get the text model for sentiment analysis."""
        return self.models.get("text")
    
    def get_vision_model(self) -> Optional[OllamaModel]:
        """Get the vision model for image analysis."""
        return self.models.get("vision")
    
    def get_audio_model(self) -> Optional[OllamaModel]:
        """Get the audio model for audio analysis."""
        return self.models.get("audio")
    
    def create_custom_model(
        self, 
        model_id: str, 
        model_type: str = "text",
        **kwargs
    ) -> Optional[OllamaModel]:
        """Create a custom Ollama model configuration."""
        try:
            model = OllamaModel(
                host=self.host,
                model_id=model_id,
                **kwargs
            )
            self.models[model_type] = model
            logger.info(
                f"Custom Ollama model '{model_id}' created for {model_type}"
            )
            return model
            
        except Exception as e:
            logger.error(f"Failed to create custom model '{model_id}': {e}")
            return None
    
    def update_model_config(self, model_type: str, **kwargs):
        """Update configuration for a specific model type."""
        model = self.models.get(model_type)
        if model:
            try:
                # Update attributes directly
                for key, value in kwargs.items():
                    if hasattr(model, key):
                        setattr(model, key, value)
                logger.info(f"Updated configuration for {model_type} model")
            except Exception as e:
                logger.error(f"Failed to update {model_type} model config: {e}")
        else:
            logger.warning(f"Cannot update {model_type} model: not available")
    
    def get_available_models(self) -> List[str]:
        """Get list of available model types."""
        return list(self.models.keys())
    
    def check_model_availability(self, model_id: str) -> bool:
        """Check if a specific model is available on the Ollama server."""
        try:
            import aiohttp
            import asyncio
            
            async def check():
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.host}/api/tags") as response:
                        if response.status == 200:
                            models = await response.json()
                            return any(
                                model["name"] == model_id 
                                for model in models.get("models", [])
                            )
                        return False
            
            # Run the async check
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a task
                task = asyncio.create_task(check())
                return asyncio.run_coroutine_threadsafe(task, loop).result()
            else:
                return asyncio.run(check())
                
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
    
    async def generate_response(
        self, 
        model: str, 
        prompt: str, 
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """Generate a response using the specified Ollama model."""
        try:
            import aiohttp
            import json
            
            # Get model configuration
            model_config = self.models.get(model)
            if not model_config:
                # Use default text model if specified model not found
                model_config = self.models.get("text", self.models.get("llama3.2:latest"))
            
            if not model_config:
                raise Exception(f"Model {model} not found and no default model available")
            
            # Prepare the request payload
            payload = {
                "model": model_config.model_id,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            # Make the request to Ollama
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response", "")
                    else:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"Error generating response with Ollama: {e}")
            return f"Error generating response: {str(e)}"


# Global Ollama integration instance
ollama_integration = OllamaIntegration()


def get_ollama_model(model_type: str = "text") -> Optional[OllamaModel]:
    """Get an Ollama model by type."""
    return ollama_integration.models.get(model_type)


def create_ollama_agent(model_type: str = "text", **kwargs):
    """Create an agent with Ollama model."""
    try:
        from core.strands_mock import Agent
        
        model = get_ollama_model(model_type)
        if not model:
            logger.error(f"No {model_type} model available")
            return None
        
        agent = Agent(model=model, **kwargs)
        logger.info(
            f"Created agent with {model_type} Ollama model"
        )
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create Ollama agent: {e}")
        return None
