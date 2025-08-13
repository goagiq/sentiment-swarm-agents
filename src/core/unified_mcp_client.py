"""
Unified MCP Client for interacting with the unified MCP server.
"""

from typing import Dict, Any
from loguru import logger

try:
    from fastmcp import FastMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("FastMCP not available - using mock client")


class UnifiedMCPClient:
    """Unified MCP client for interacting with the unified MCP server."""
    
    def __init__(self, server_url: str = "http://localhost:8003/mcp/"):
        self.server_url = server_url
        self.mcp_client = None
        self._connected = False
    
    async def connect(self) -> bool:
        """Connect to the unified MCP server."""
        if not MCP_AVAILABLE:
            logger.warning("FastMCP not available - using mock client")
            return False
        
        try:
            self.mcp_client = FastMCPClient(self.server_url)
            await self.mcp_client.connect()
            self._connected = True
            logger.info(
                f"✅ Connected to unified MCP server at {self.server_url}"
            )
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to MCP server: {e}")
            return False
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the unified MCP server."""
        if not self._connected:
            await self.connect()
        
        if not self._connected:
            return {"success": False, "error": "Not connected to MCP server"}
        
        try:
            result = await self.mcp_client.call_tool(tool_name, parameters)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def list_tools(self) -> Dict[str, Any]:
        """List available tools on the unified MCP server."""
        if not self._connected:
            await self.connect()
        
        if not self._connected:
            return {"success": False, "error": "Not connected to MCP server"}
        
        try:
            tools = await self.mcp_client.list_tools()
            return {"success": True, "tools": tools}
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return {"success": False, "error": str(e)}
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.mcp_client and self._connected:
            try:
                await self.mcp_client.disconnect()
                self._connected = False
                logger.info("Disconnected from MCP server")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")


# Global unified MCP client instance
unified_mcp_client = UnifiedMCPClient()


async def get_unified_mcp_client() -> UnifiedMCPClient:
    """Get the global unified MCP client instance."""
    return unified_mcp_client


async def call_unified_mcp_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Call a tool on the unified MCP server using the global client."""
    client = await get_unified_mcp_client()
    return await client.call_tool(tool_name, parameters)
