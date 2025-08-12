"""
External Data Integration Configuration for Phase 2.
"""

from typing import List, Dict, Any

from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class SocialMediaConfig(BaseSettings):
    """Configuration for social media platform integrations."""
    
    # Supported platforms
    supported_platforms: List[str] = Field(
        default=["twitter", "linkedin", "facebook", "instagram"],
        description="Supported social media platforms"
    )
    
    # API settings
    api_timeout: int = Field(
        default=30,
        ge=10,
        le=120,
        description="API timeout in seconds"
    )
    rate_limit_per_minute: int = Field(
        default=60,
        ge=10,
        le=1000,
        description="Rate limit per minute"
    )
    
    # Data collection settings
    default_time_range: str = Field(
        default="7d",
        description="Default time range for data collection"
    )
    max_posts_per_platform: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum posts to collect per platform"
    )
    
    # Analysis settings
    enable_sentiment_analysis: bool = Field(
        default=True,
        description="Enable sentiment analysis for social media content"
    )
    enable_topic_extraction: bool = Field(
        default=True,
        description="Enable topic extraction from social media content"
    )
    enable_engagement_metrics: bool = Field(
        default=True,
        description="Enable engagement metrics calculation"
    )


class DatabaseConfig(BaseSettings):
    """Configuration for database integrations."""
    
    # Supported database types
    supported_databases: List[str] = Field(
        default=["mongodb", "postgresql", "mysql", "elasticsearch"],
        description="Supported database types"
    )
    
    # Connection settings
    connection_timeout: int = Field(
        default=30,
        ge=10,
        le=120,
        description="Database connection timeout in seconds"
    )
    max_connections: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum database connections"
    )
    
    # Query settings
    max_query_time: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Maximum query execution time in seconds"
    )
    enable_query_caching: bool = Field(
        default=True,
        description="Enable query result caching"
    )
    cache_duration: int = Field(
        default=1800,
        ge=300,
        le=7200,
        description="Cache duration in seconds"
    )


class APIConfig(BaseSettings):
    """Configuration for external API integrations."""
    
    # Supported API types
    supported_api_types: List[str] = Field(
        default=["rest", "graphql", "soap"],
        description="Supported API types"
    )
    
    # Request settings
    request_timeout: int = Field(
        default=30,
        ge=10,
        le=120,
        description="API request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts"
    )
    retry_delay: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Delay between retries in seconds"
    )
    
    # Caching settings
    enable_response_caching: bool = Field(
        default=True,
        description="Enable API response caching"
    )
    cache_duration: int = Field(
        default=1800,
        ge=300,
        le=7200,
        description="Cache duration in seconds"
    )
    
    # Rate limiting
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable rate limiting for API requests"
    )
    requests_per_minute: int = Field(
        default=60,
        ge=10,
        le=1000,
        description="Requests per minute limit"
    )


class MarketDataConfig(BaseSettings):
    """Configuration for market data integrations."""
    
    # Supported data sources
    supported_data_sources: List[str] = Field(
        default=["yahoo_finance", "alpha_vantage", "quandl"],
        description="Supported market data sources"
    )
    
    # Data collection settings
    default_time_range: str = Field(
        default="30d",
        description="Default time range for market data"
    )
    max_data_points: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Maximum data points to collect"
    )
    
    # Analysis settings
    enable_sentiment_analysis: bool = Field(
        default=True,
        description="Enable sentiment analysis for market data"
    )
    enable_trend_analysis: bool = Field(
        default=True,
        description="Enable trend analysis"
    )
    enable_competitor_analysis: bool = Field(
        default=True,
        description="Enable competitor analysis"
    )


class NewsSourceConfig(BaseSettings):
    """Configuration for news source integrations."""
    
    # Supported news sources
    supported_sources: List[str] = Field(
        default=["reuters", "bloomberg", "cnn", "bbc"],
        description="Supported news sources"
    )
    
    # Data collection settings
    max_articles_per_source: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum articles to collect per source"
    )
    article_timeout: int = Field(
        default=30,
        ge=10,
        le=120,
        description="Article fetch timeout in seconds"
    )
    
    # Analysis settings
    enable_sentiment_analysis: bool = Field(
        default=True,
        description="Enable sentiment analysis for news articles"
    )
    enable_topic_extraction: bool = Field(
        default=True,
        description="Enable topic extraction from news articles"
    )
    enable_entity_extraction: bool = Field(
        default=True,
        description="Enable entity extraction from news articles"
    )


class ExternalDataConfig(BaseSettings):
    """Main external data integration configuration."""
    
    model_config = ConfigDict(env_prefix="EXTERNAL_DATA_")
    
    # Social media configuration
    social_media: SocialMediaConfig = Field(
        default_factory=SocialMediaConfig,
        description="Social media integration configuration"
    )
    
    # Database configuration
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="Database integration configuration"
    )
    
    # API configuration
    api: APIConfig = Field(
        default_factory=APIConfig,
        description="External API integration configuration"
    )
    
    # Market data configuration
    market_data: MarketDataConfig = Field(
        default_factory=MarketDataConfig,
        description="Market data integration configuration"
    )
    
    # News source configuration
    news_sources: NewsSourceConfig = Field(
        default_factory=NewsSourceConfig,
        description="News source integration configuration"
    )
    
    # General settings
    enable_external_integrations: bool = Field(
        default=True,
        description="Enable external data integrations"
    )
    enable_error_handling: bool = Field(
        default=True,
        description="Enable comprehensive error handling"
    )
    enable_logging: bool = Field(
        default=True,
        description="Enable detailed logging"
    )
    
    # Performance settings
    max_concurrent_requests: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum concurrent external requests"
    )
    request_timeout: int = Field(
        default=30,
        ge=10,
        le=120,
        description="Default request timeout in seconds"
    )
    
    # Storage settings
    data_storage_path: str = Field(
        default="./Results/external_data",
        description="Path for storing external data"
    )
    enable_data_persistence: bool = Field(
        default=True,
        description="Enable persistence of external data"
    )
    data_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Data retention period in days"
    )


# Global configuration instance
external_data_config = ExternalDataConfig()
