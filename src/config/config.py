"""
Configuration management for the sentiment analysis system.
"""

from pathlib import Path
from typing import Dict, Any

from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class ModelConfig(BaseSettings):
    """Configuration for models."""
    
    # Default models - using Ollama models
    default_text_model: str = "ollama:llama3.2:latest"  # Ollama text model
    default_vision_model: str = "ollama:llava:latest"   # Ollama vision model
    default_audio_model: str = "ollama:llava:latest"    # Ollama audio model
    
    # Ollama configuration
    ollama_host: str = "http://localhost:11434"
    ollama_timeout: int = 30
    
    # Model parameters
    vision_temperature: float = 0.7
    vision_max_tokens: int = 200
    audio_temperature: float = 0.7
    audio_max_tokens: int = 200
    
    # Tool calling
    enable_tool_calling: bool = True
    max_tool_calls: int = 5


class AgentConfig(BaseSettings):
    """Configuration for agents."""
    
    # Agent capacities
    text_agent_capacity: int = 10
    vision_agent_capacity: int = 5
    audio_agent_capacity: int = 5
    web_agent_capacity: int = 3
    
    # Processing limits
    max_image_size: int = 1024
    max_video_duration: int = 30  # seconds
    max_audio_duration: int = 300  # seconds
    
    # Reflection settings
    enable_reflection: bool = True
    max_reflection_iterations: int = 3
    confidence_threshold: float = 0.8


class YouTubeDLConfig(BaseSettings):
    """Configuration for YouTube-DL integration."""
    
    # Download settings
    download_path: str = "./temp/videos"
    max_video_duration: int = 600  # 10 minutes
    max_audio_duration: int = 1800  # 30 minutes
    
    # Video processing
    max_video_resolution: str = "720p"
    preferred_video_format: str = "mp4"
    frame_extraction_count: int = 10
    
    # Audio processing
    preferred_audio_format: str = "mp3"
    audio_quality: str = "192k"
    
    # Performance settings
    enable_caching: bool = True
    cache_duration: int = 3600  # 1 hour
    max_concurrent_downloads: int = 3
    
    # Error handling
    retry_attempts: int = 3
    retry_delay: int = 5  # seconds
    timeout: int = 60  # seconds


class APIConfig(BaseSettings):
    """Configuration for the API."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1
    
    # Rate limiting
    rate_limit_per_minute: int = 100
    rate_limit_per_hour: int = 1000
    
    # CORS
    cors_origins: list = ["*"]
    cors_methods: list = ["*"]
    cors_headers: list = ["*"]


class SentimentConfig(BaseSettings):
    """Main configuration class."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
    # Environment
    environment: str = "development"
    
    # Model configuration
    model: ModelConfig = Field(default_factory=ModelConfig)
    
    # Agent configuration
    agent: AgentConfig = Field(default_factory=AgentConfig)
    
    # YouTube-DL configuration
    youtube_dl: YouTubeDLConfig = Field(default_factory=YouTubeDLConfig)
    
    # API configuration
    api: APIConfig = Field(default_factory=APIConfig)
    
    # Directories
    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )
    test_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "Test"
    )
    results_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "Results"
    )
    
    # Logging
    log_level: str = "INFO"
    log_format: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | "
        "{name}:{function}:{line} | {message}"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        self.test_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific model type."""
        if model_type == "vision":
            return {
                "model_id": self.model.default_vision_model,
                "host": self.model.ollama_host,
                "temperature": self.model.vision_temperature,
                "max_tokens": self.model.vision_max_tokens,
            }
        elif model_type == "audio":
            return {
                "model_id": self.model.default_audio_model,
                "host": self.model.ollama_host,
                "temperature": self.model.audio_temperature,
                "max_tokens": self.model.audio_max_tokens,
            }
        else:
            return {
                "model_id": self.model.default_text_model,
                "host": None,
                "temperature": 0.5,
                "max_tokens": 100,
            }


# Global configuration instance
config = SentimentConfig()

