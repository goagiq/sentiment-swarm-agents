"""
Model configuration with environment variable support.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ModelSettings(BaseSettings):
    """Model configuration with environment variable support."""
    
    # Primary models
    TEXT_MODEL: str = Field(
        default="mistral-small3.1:latest",
        description="Primary text model for sentiment analysis and entity extraction"
    )
    VISION_MODEL: str = Field(
        default="llava:latest",
        description="Primary vision model for audio, video, and image processing"
    )
    
    # Fallback models
    FALLBACK_TEXT_MODEL: str = Field(
        default="llama3.2:latest",
        description="Fallback text model when primary text model fails"
    )
    FALLBACK_VISION_MODEL: str = Field(
        default="granite3.2-vision",
        description="Fallback vision model when primary vision model fails"
    )
    
    # Ollama configuration
    OLLAMA_HOST: str = Field(
        default="http://localhost:11434",
        description="Ollama server host URL"
    )
    OLLAMA_TIMEOUT: int = Field(
        default=30,
        description="Ollama request timeout in seconds"
    )
    
    # Model parameters
    TEXT_TEMPERATURE: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for text models"
    )
    TEXT_MAX_TOKENS: int = Field(
        default=200,
        ge=1,
        le=4096,
        description="Maximum tokens for text models"
    )
    VISION_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for vision models"
    )
    VISION_MAX_TOKENS: int = Field(
        default=200,
        ge=1,
        le=4096,
        description="Maximum tokens for vision models"
    )
    
    # Tool calling
    ENABLE_TOOL_CALLING: bool = Field(
        default=True,
        description="Enable tool calling for models"
    )
    MAX_TOOL_CALLS: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum tool calls per request"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class ModelConfig:
    """Model configuration manager."""
    
    def __init__(self):
        self.settings = ModelSettings()
    
    def get_text_model_config(self) -> Dict[str, Any]:
        """Get text model configuration."""
        return {
            "model_id": self.settings.TEXT_MODEL,
            "fallback_model": self.settings.FALLBACK_TEXT_MODEL,
            "host": self.settings.OLLAMA_HOST,
            "temperature": self.settings.TEXT_TEMPERATURE,
            "max_tokens": self.settings.TEXT_MAX_TOKENS,
            "timeout": self.settings.OLLAMA_TIMEOUT,
            "enable_tool_calling": self.settings.ENABLE_TOOL_CALLING,
            "max_tool_calls": self.settings.MAX_TOOL_CALLS,
        }
    
    def get_vision_model_config(self) -> Dict[str, Any]:
        """Get vision model configuration (for audio, video, image)."""
        return {
            "model_id": self.settings.VISION_MODEL,
            "fallback_model": self.settings.FALLBACK_VISION_MODEL,
            "host": self.settings.OLLAMA_HOST,
            "temperature": self.settings.VISION_TEMPERATURE,
            "max_tokens": self.settings.VISION_MAX_TOKENS,
            "timeout": self.settings.OLLAMA_TIMEOUT,
            "enable_tool_calling": self.settings.ENABLE_TOOL_CALLING,
            "max_tool_calls": self.settings.MAX_TOOL_CALLS,
        }
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific model type."""
        if model_type in ["vision", "audio", "video", "image"]:
            return self.get_vision_model_config()
        else:
            return self.get_text_model_config()
    
    def get_ollama_host(self) -> str:
        """Get Ollama host URL."""
        return self.settings.OLLAMA_HOST
    
    def get_ollama_timeout(self) -> int:
        """Get Ollama timeout."""
        return self.settings.OLLAMA_TIMEOUT


# Global model configuration instance
model_config = ModelConfig()
