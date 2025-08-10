"""
Configuration settings for the sentiment analysis system.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Project paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"
    RESULTS_DIR: Path = BASE_DIR / "results"
    
    # Database settings
    CHROMA_HOST: str = Field(default="localhost", env="CHROMA_HOST")
    CHROMA_PORT: int = Field(default=8000, env="CHROMA_PORT")
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "chroma_db"
    
    # Model settings
    SENTIMENT_MODEL: str = Field(
        default=(
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        ),
        env="SENTIMENT_MODEL"
    )
    AUDIO_MODEL: str = Field(
        default="facebook/wav2vec2-base-960h",
        env="AUDIO_MODEL"
    )
    VISION_MODEL: str = Field(
        default="microsoft/DialoGPT-medium",
        env="VISION_MODEL"
    )
    MODEL_CACHE_DIR: Path = Field(default=MODELS_DIR, env="MODEL_CACHE_DIR")
    
    # Processing settings
    MAX_BATCH_SIZE: int = Field(default=100, env="MAX_BATCH_SIZE")
    ENABLE_GPU: bool = Field(default=False, env="ENABLE_GPU")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # API settings
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_RELOAD: bool = Field(default=True, env="API_RELOAD")
    
    # Streamlit settings
    STREAMLIT_PORT: int = Field(default=8501, env="STREAMLIT_PORT")
    
    # Agent settings
    MAX_WORKERS: int = Field(default=4, env="MAX_WORKERS")
    AGENT_TIMEOUT: int = Field(default=300, env="AGENT_TIMEOUT")
    
    # Language settings
    DEFAULT_LANGUAGE: str = Field(default="en", env="DEFAULT_LANGUAGE")
    SUPPORTED_LANGUAGES: list = Field(default=["en"], env="SUPPORTED_LANGUAGES")
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.DATA_DIR.mkdir(exist_ok=True)
        self.RESULTS_DIR.mkdir(exist_ok=True)
        self.CHROMA_PERSIST_DIR.mkdir(exist_ok=True)


# Global settings instance
settings = Settings()
