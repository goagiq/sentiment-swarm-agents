"""
Configuration settings for the sentiment analysis system.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True
    )
    
    # Project paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"
    RESULTS_DIR: Path = BASE_DIR / "results"
    
    # Database settings
    CHROMA_HOST: str = Field(
        default="localhost", 
        json_schema_extra={"env": "CHROMA_HOST"}
    )
    CHROMA_PORT: int = Field(
        default=8000, 
        json_schema_extra={"env": "CHROMA_PORT"}
    )
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "chroma_db"
    
    # Model settings
    SENTIMENT_MODEL: str = Field(
        default=(
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        ),
        json_schema_extra={"env": "SENTIMENT_MODEL"}
    )
    AUDIO_MODEL: str = Field(
        default="facebook/wav2vec2-base-960h",
        json_schema_extra={"env": "AUDIO_MODEL"}
    )
    VISION_MODEL: str = Field(
        default="microsoft/DialoGPT-medium",
        json_schema_extra={"env": "VISION_MODEL"}
    )
    MODEL_CACHE_DIR: Path = Field(
        default=MODELS_DIR, 
        json_schema_extra={"env": "MODEL_CACHE_DIR"}
    )
    
    # Processing settings
    MAX_BATCH_SIZE: int = Field(
        default=100, 
        json_schema_extra={"env": "MAX_BATCH_SIZE"}
    )
    ENABLE_GPU: bool = Field(
        default=False, 
        json_schema_extra={"env": "ENABLE_GPU"}
    )
    LOG_LEVEL: str = Field(
        default="INFO", 
        json_schema_extra={"env": "LOG_LEVEL"}
    )
    
    # API settings
    API_HOST: str = Field(
        default="0.0.0.0", 
        json_schema_extra={"env": "API_HOST"}
    )
    API_PORT: int = Field(
        default=8000, 
        json_schema_extra={"env": "API_PORT"}
    )
    API_RELOAD: bool = Field(
        default=True, 
        json_schema_extra={"env": "API_RELOAD"}
    )
    
    # Streamlit settings
    STREAMLIT_PORT: int = Field(
        default=8501, 
        json_schema_extra={"env": "STREAMLIT_PORT"}
    )
    
    # Agent settings
    MAX_WORKERS: int = Field(
        default=4, 
        json_schema_extra={"env": "MAX_WORKERS"}
    )
    AGENT_TIMEOUT: int = Field(
        default=300, 
        json_schema_extra={"env": "AGENT_TIMEOUT"}
    )
    
    # Language settings
    DEFAULT_LANGUAGE: str = Field(
        default="en", 
        json_schema_extra={"env": "DEFAULT_LANGUAGE"}
    )
    SUPPORTED_LANGUAGES: list = Field(
        default=["en"], 
        json_schema_extra={"env": "SUPPORTED_LANGUAGES"}
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)
        self.RESULTS_DIR.mkdir(exist_ok=True)
        self.CHROMA_PERSIST_DIR.mkdir(exist_ok=True)


# Global settings instance
settings = Settings()
