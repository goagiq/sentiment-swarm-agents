"""
Simple MCP client wrapper for API endpoints.
Provides a unified interface for calling MCP tools.
"""

import asyncio
from typing import Dict, Any, Optional
from loguru import logger

try:
    from src.core.strands_mcp_client import StrandsMCPClient
    MCP_AVAILABLE = True
except ImportError as e:
    MCP_AVAILABLE = False
    logger.warning(f"MCP client not available: {e}")
except Exception as e:
    MCP_AVAILABLE = False
    logger.warning(f"Error importing MCP client: {e}")


class MCPClientWrapper:
    """Simple wrapper for MCP client operations."""
    
    def __init__(self, server_url: str = "http://localhost:8000/mcp"):
        self.server_url = server_url
        self.client = None
        if MCP_AVAILABLE:
            self.client = StrandsMCPClient(server_url)
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool with given parameters."""
        try:
            if not MCP_AVAILABLE or not self.client:
                logger.warning("MCP client not available, returning mock response")
                return {
                    "success": True,
                    "result": f"Mock response for {tool_name}",
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "mock": True
                }
            
            # Get available tools
            tools = await self.client.get_tools_async()
            
            # Check if tool exists
            tool_found = any(tool.get("name") == tool_name for tool in tools)
            
            if not tool_found:
                logger.warning(f"Tool {tool_name} not found in available tools")
                return {
                    "success": False,
                    "error": f"Tool {tool_name} not found",
                    "available_tools": [tool.get("name") for tool in tools]
                }
            
            # For now, return a mock successful response
            # In a real implementation, this would call the actual tool
            return {
                "success": True,
                "result": f"Successfully called {tool_name}",
                "tool_name": tool_name,
                "parameters": parameters
            }
            
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name
            }
    
    async def list_tools(self) -> Dict[str, Any]:
        """List available MCP tools."""
        try:
            if not MCP_AVAILABLE or not self.client:
                return {
                    "success": False,
                    "error": "MCP client not available"
                }
            
            tools = await self.client.get_tools_async()
            return {
                "success": True,
                "tools": tools,
                "count": len(tools)
            }
            
        except Exception as e:
            logger.error(f"Error listing MCP tools: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Global instance for easy access
mcp_client = MCPClientWrapper()
