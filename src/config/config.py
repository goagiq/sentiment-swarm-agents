"""
Configuration management for the sentiment analysis system.
"""

from pathlib import Path
from typing import Dict, Any, List

from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class ModelConfig(BaseSettings):
    """Configuration for models."""
    
    # Default models - configurable through environment variables
    default_text_model: str = Field(
        default="mistral-small3.1:latest",
        description="Primary text model for sentiment analysis and entity "
                   "extraction"
    )
    default_vision_model: str = Field(
        default="llava:latest",
        description="Primary vision model for audio, video, and image "
                   "processing"
    )
    default_audio_model: str = Field(
        default="llava:latest",
        description="Primary audio model for audio processing and "
                   "transcription"
    )
    
    # Fallback models - configurable through environment variables
    fallback_text_model: str = Field(
        default="llama3.2:latest",
        description="Fallback text model when primary text model fails"
    )
    fallback_vision_model: str = Field(
        default="granite3.2-vision",
        description="Fallback vision model when primary vision model fails"
    )
    
    # Ollama configuration - configurable through environment variables
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server host URL"
    )
    ollama_timeout: int = Field(
        default=30,
        description="Ollama request timeout in seconds"
    )
    
    # Strands-specific Ollama configuration
    strands_ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server address for Strands integration"
    )
    strands_default_model: str = Field(
        default="llama3.2:latest",
        description="Default model ID for Strands agents"
    )
    strands_text_model: str = Field(
        default="mistral-small3.1:latest",
        description="Text model ID for Strands text agents"
    )
    strands_vision_model: str = Field(
        default="llava:latest",
        description="Vision model ID for Strands vision agents"
    )
    strands_translation_fast_model: str = Field(
        default="llama3.2:latest",
        description="Fast translation model ID for Strands translation agents"
    )
    
    # Model parameters
    text_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for text models"
    )
    text_max_tokens: int = Field(
        default=200,
        ge=1,
        le=4096,
        description="Maximum tokens for text models"
    )
    vision_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for vision models"
    )
    vision_max_tokens: int = Field(
        default=200,
        ge=1,
        le=4096,
        description="Maximum tokens for vision models"
    )
    
    # Tool calling
    enable_tool_calling: bool = Field(
        default=True,
        description="Enable tool calling for models"
    )
    max_tool_calls: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum tool calls per request"
    )


class AgentConfig(BaseSettings):
    """Configuration for agents."""
    
    # Agent capacities
    text_agent_capacity: int = 10
    vision_agent_capacity: int = 5
    audio_agent_capacity: int = 5
    web_agent_capacity: int = 3
    knowledge_graph_agent_capacity: int = 5
    
    # Processing limits
    max_image_size: int = 1024
    max_video_duration: int = 30  # seconds
    max_audio_duration: int = 300  # seconds
    
    # Content storage preferences
    store_full_content: bool = Field(
        default=True,
        description="Store full transcriptions/translations instead of summaries in vector database"
    )
    store_summaries_separately: bool = Field(
        default=True,
        description="Store summaries in metadata for quick access while keeping full content in extracted_text"
    )
    content_storage_priority: str = Field(
        default="full_content",
        description="Priority order for content storage: 'full_content', 'summary', 'both'"
    )
    min_content_length: int = Field(
        default=50,
        description="Minimum expected length for full content (used to detect summaries)"
    )
    
    # Knowledge Graph settings
    graph_storage_path: str = "./Results/knowledge_graphs"
    enable_graph_visualization: bool = True
    max_graph_nodes: int = 10000
    max_graph_edges: int = 50000
    
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
    port: int = 8003
    debug: bool = False
    workers: int = 1
    
    # Rate limiting
    rate_limit_per_minute: int = 100
    rate_limit_per_hour: int = 1000
    
    # CORS
    cors_origins: list = ["*"]
    cors_methods: list = ["*"]
    cors_headers: list = ["*"]


class LanguageConfig(BaseSettings):
    """Configuration for supported languages."""
    
    # Supported languages with their ISO codes and display names
    supported_languages: Dict[str, str] = Field(
        default={
            "en": "English",
            "es": "Spanish", 
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "th": "Thai",
            "ar": "Arabic",
            "hi": "Hindi",
            "ru": "Russian"
        },
        description="Supported languages with ISO codes and display names"
    )
    
    # Default language
    default_language: str = Field(
        default="en",
        description="Default language for analysis"
    )
    
    # Language detection settings
    enable_auto_detection: bool = Field(
        default=True,
        description="Enable automatic language detection"
    )
    
    def get_language_list(self) -> List[str]:
        """Get list of supported language codes."""
        return list(self.supported_languages.keys())
    
    def get_language_name(self, code: str) -> str:
        """Get display name for a language code."""
        return self.supported_languages.get(code, code)


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
    
    # Language configuration
    language: LanguageConfig = Field(default_factory=LanguageConfig)
    
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
        if model_type in ["vision", "audio", "video"]:
            return {
                "model_id": self.model.default_vision_model,
                "fallback_model": self.model.fallback_vision_model,
                "host": self.model.ollama_host,
                "temperature": self.model.vision_temperature,
                "max_tokens": self.model.vision_max_tokens,
            }
        else:
            return {
                "model_id": self.model.default_text_model,
                "fallback_model": self.model.fallback_text_model,
                "host": self.model.ollama_host,
                "temperature": self.model.text_temperature,
                "max_tokens": self.model.text_max_tokens,
            }
    
    def get_strands_model_config(
        self, agent_type: str
    ) -> Dict[str, Any]:
        """Get Strands-specific model configuration for different agent types."""
        base_config = {
            "host": self.model.strands_ollama_host,
            "temperature": self.model.text_temperature,
            "max_tokens": self.model.text_max_tokens,
        }
        
        if agent_type in ["text", "simple_text", "sentiment"]:
            return {
                **base_config,
                "model_id": self.model.strands_text_model,
                "fallback_model": self.model.fallback_text_model,
            }
        elif agent_type in ["vision", "audio", "video"]:
            return {
                **base_config,
                "model_id": self.model.strands_vision_model,
                "fallback_model": self.model.fallback_vision_model,
                "temperature": self.model.vision_temperature,
                "max_tokens": self.model.vision_max_tokens,
            }
        elif agent_type == "translation_fast":
            return {
                **base_config,
                "model_id": self.model.strands_translation_fast_model,
                "fallback_model": self.model.fallback_text_model,
                "temperature": 0.2,
                "max_tokens": 300,
            }
        else:
            return {
                **base_config,
                "model_id": self.model.strands_default_model,
                "fallback_model": self.model.fallback_text_model,
            }


# Global configuration instance
config = SentimentConfig()

