"""
Configuration for MCP (Model Context Protocol) server.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class MCPServerConfig(BaseModel):
    """Configuration for the MCP server."""
    
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
default_mcp_server_config = MCPServerConfig()
default_mcp_client_config = MCPClientConfig()


def get_mcp_server_config() -> MCPServerConfig:
    """Get MCP server configuration."""
    return default_mcp_server_config


def get_mcp_client_config() -> MCPClientConfig:
    """Get MCP client configuration."""
    return default_mcp_client_config


def update_mcp_server_config(**kwargs) -> MCPServerConfig:
    """Update MCP server configuration."""
    global default_mcp_server_config
    default_mcp_server_config = default_mcp_server_config.model_copy(update=kwargs)
    return default_mcp_server_config


def update_mcp_client_config(**kwargs) -> MCPClientConfig:
    """Update MCP client configuration."""
    global default_mcp_client_config
    default_mcp_client_config = default_mcp_client_config.model_copy(update=kwargs)
    return default_mcp_client_config
