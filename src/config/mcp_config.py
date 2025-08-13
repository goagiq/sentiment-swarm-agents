"""
Configuration for MCP (Model Context Protocol) server.
Updated for consolidated server architecture.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

# Import existing configurations
try:
    from .language_config import ChineseConfig, RussianConfig, EnglishConfig
    CONFIG_IMPORTS_AVAILABLE = True
except ImportError:
    CONFIG_IMPORTS_AVAILABLE = False
    print("⚠️  Some config imports not available")


class ProcessingCategory(str, Enum):
    """Processing categories for consolidated MCP servers."""
    PDF = "pdf"
    AUDIO = "audio"
    VIDEO = "video"
    WEBSITE = "website"


class ConsolidatedServerConfig(BaseModel):
    """Configuration for individual consolidated processing servers."""
    
    # Server enablement
    enabled: bool = Field(default=True, description="Enable this processing server")
    
    # Model configuration
    primary_model: str = Field(default="", description="Primary model for this category")
    fallback_model: str = Field(default="", description="Fallback model for this category")
    
    # Processing settings
    max_file_size: int = Field(
        default=100 * 1024 * 1024, 
        description="Maximum file size in bytes"
    )  # 100MB
    timeout: int = Field(
        default=300, 
        description="Processing timeout in seconds"
    )
    enable_caching: bool = Field(
        default=True, 
        description="Enable result caching"
    )
    
    # Language support
    supported_languages: List[str] = Field(
        default=["en", "zh", "ru"],
        description="Supported languages for this server"
    )
    
    # Vector database settings
    vector_db_enabled: bool = Field(
        default=True, 
        description="Enable vector database storage"
    )
    vector_db_collection: str = Field(
        default="", 
        description="Vector database collection name"
    )
    
    # Knowledge graph settings
    knowledge_graph_enabled: bool = Field(
        default=True, 
        description="Enable knowledge graph creation"
    )
    knowledge_graph_name: str = Field(
        default="", 
        description="Knowledge graph name"
    )


class ConsolidatedMCPServerConfig(BaseModel):
    """Configuration for the unified MCP server."""
    
    # Server settings
    host: str = Field(default="localhost", description="Server host")
    port: int = Field(default=8003, description="Server port (integrated with FastAPI)")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Transport settings
    transport: str = Field(
        default="http", 
        description="Transport protocol (HTTP for FastAPI integration)"
    )
    
    # Unified server settings
    server_name: str = Field(
        default="unified_sentiment_mcp_server", 
        description="Unified MCP server name"
    )
    server_version: str = Field(
        default="1.0.0", 
        description="Unified MCP server version"
    )
    
    # FastAPI integration settings
    fastapi_mount_path: str = Field(
        default="/mcp", 
        description="FastAPI mount path for MCP server"
    )
    fastapi_port: int = Field(
        default=8003, 
        description="FastAPI server port"
    )
    
    # Consolidated server configurations
    pdf_server: ConsolidatedServerConfig = Field(
        default=ConsolidatedServerConfig(
            primary_model="llava:latest",
            fallback_model="granite3.2-vision",
            vector_db_collection="pdf_documents",
            knowledge_graph_name="pdf_knowledge_graph"
        ),
        description="PDF processing server configuration"
    )
    
    audio_server: ConsolidatedServerConfig = Field(
        default=ConsolidatedServerConfig(
            primary_model="llava:latest",
            fallback_model="granite3.2-vision",
            vector_db_collection="audio_transcripts",
            knowledge_graph_name="audio_knowledge_graph"
        ),
        description="Audio processing server configuration"
    )
    
    video_server: ConsolidatedServerConfig = Field(
        default=ConsolidatedServerConfig(
            primary_model="llava:latest",
            fallback_model="granite3.2-vision",
            vector_db_collection="video_analysis",
            knowledge_graph_name="video_knowledge_graph"
        ),
        description="Video processing server configuration"
    )
    
    website_server: ConsolidatedServerConfig = Field(
        default=ConsolidatedServerConfig(
            primary_model="mistral-small3.1:latest",
            fallback_model="llama3.2:latest",
            vector_db_collection="web_content",
            knowledge_graph_name="website_knowledge_graph"
        ),
        description="Website processing server configuration"
    )
    
    # Language-specific configurations
    language_configs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Language-specific configurations"
    )
    
    # Agent settings
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent requests")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    
    # Logging settings
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(
        default="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Log format"
    )
    
    # Security settings
    enable_cors: bool = Field(default=True, description="Enable CORS")
    allowed_origins: List[str] = Field(
        default=["*"], 
        description="Allowed CORS origins"
    )
    
    # Performance settings
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    max_cache_size: int = Field(default=1000, description="Maximum cache size")
    
    # Storage paths
    storage_base_path: str = Field(
        default="./data/consolidated_mcp",
        description="Base path for consolidated MCP storage"
    )
    temp_path: str = Field(
        default="./temp/consolidated_mcp",
        description="Temporary file storage path"
    )
    
    # Vector database settings
    vector_db_path: str = Field(
        default="./chroma_db/consolidated_mcp",
        description="Vector database storage path"
    )
    
    # Knowledge graph settings
    knowledge_graph_path: str = Field(
        default="./data/knowledge_graphs/consolidated_mcp",
        description="Knowledge graph storage path"
    )


class MCPServerConfig(BaseModel):
    """Legacy configuration for the MCP server (backward compatibility)."""
    
    # Server settings
    host: str = Field(default="localhost", description="Server host")
    port: int = Field(default=8001, description="Server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Transport settings
    transport: str = Field(default="streamable-http", description="Transport protocol")
    
    # Tool settings
    enable_text_analysis: bool = Field(default=True, description="Enable text sentiment analysis")
    enable_image_analysis: bool = Field(default=True, description="Enable image sentiment analysis")
    enable_audio_analysis: bool = Field(default=True, description="Enable audio sentiment analysis")
    enable_batch_processing: bool = Field(default=True, description="Enable batch processing")
    
    # Agent settings
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent requests")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    
    # Logging settings
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(
        default="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Log format"
    )
    
    # Security settings
    enable_cors: bool = Field(default=True, description="Enable CORS")
    allowed_origins: List[str] = Field(
        default=["*"], 
        description="Allowed CORS origins"
    )
    
    # Performance settings
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    max_cache_size: int = Field(default=1000, description="Maximum cache size")


class MCPClientConfig(BaseModel):
    """Configuration for MCP clients."""
    
    # Connection settings
    server_url: str = Field(
        default="http://localhost:8001/mcp/",
        description="MCP server URL"
    )
    
    # Client settings
    connection_timeout: int = Field(default=10, description="Connection timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")
    
    # Agent settings
    enable_agent_integration: bool = Field(default=True, description="Enable Strands agent integration")
    agent_model: str = Field(default="qwen2.5-coder:1.5b-base", description="Agent model")
    
    # Tool settings
    enable_tool_discovery: bool = Field(default=True, description="Enable automatic tool discovery")
    tool_cache_ttl: int = Field(default=300, description="Tool cache TTL in seconds")


# Default configurations
default_consolidated_mcp_config = ConsolidatedMCPServerConfig()
default_mcp_server_config = MCPServerConfig()
default_mcp_client_config = MCPClientConfig()


def get_consolidated_mcp_config() -> ConsolidatedMCPServerConfig:
    """Get consolidated MCP server configuration."""
    return default_consolidated_mcp_config


def get_mcp_server_config() -> MCPServerConfig:
    """Get legacy MCP server configuration."""
    return default_mcp_server_config


def get_mcp_client_config() -> MCPClientConfig:
    """Get MCP client configuration."""
    return default_mcp_client_config


def update_consolidated_mcp_config(**kwargs) -> ConsolidatedMCPServerConfig:
    """Update consolidated MCP server configuration."""
    global default_consolidated_mcp_config
    default_consolidated_mcp_config = default_consolidated_mcp_config.model_copy(update=kwargs)
    return default_consolidated_mcp_config


def update_mcp_server_config(**kwargs) -> MCPServerConfig:
    """Update legacy MCP server configuration."""
    global default_mcp_server_config
    default_mcp_server_config = default_mcp_server_config.model_copy(update=kwargs)
    return default_mcp_server_config


def update_mcp_client_config(**kwargs) -> MCPClientConfig:
    """Update MCP client configuration."""
    global default_mcp_client_config
    default_mcp_client_config = default_mcp_client_config.model_copy(update=kwargs)
    return default_mcp_client_config


def get_language_config(language: str) -> Optional[Any]:
    """Get language-specific configuration."""
    if not CONFIG_IMPORTS_AVAILABLE:
        return None
    
    language_map = {
        "zh": ChineseConfig,
        "ru": RussianConfig,
        "en": EnglishConfig
    }
    
    return language_map.get(language)


def get_server_config_for_category(category: ProcessingCategory) -> ConsolidatedServerConfig:
    """Get configuration for a specific processing category."""
    config = get_consolidated_mcp_config()
    
    category_map = {
        ProcessingCategory.PDF: config.pdf_server,
        ProcessingCategory.AUDIO: config.audio_server,
        ProcessingCategory.VIDEO: config.video_server,
        ProcessingCategory.WEBSITE: config.website_server
    }
    
    return category_map.get(category, ConsolidatedServerConfig())
