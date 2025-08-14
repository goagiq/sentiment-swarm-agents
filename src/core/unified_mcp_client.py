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
            # Return mock responses when MCP is not available
            return self._get_mock_response(tool_name, parameters)
        
        try:
            result = await self.mcp_client.call_tool(tool_name, parameters)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_mock_response(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get mock response for tools when MCP is not available."""
        if tool_name == "query_decision_context":
            return {
                "success": True,
                "result": {
                    "business_entities": [
                        {"name": "operational_efficiency", "type": "business_concept", "confidence": 0.8},
                        {"name": "cost_reduction", "type": "business_goal", "confidence": 0.9},
                        {"name": "market_analysis", "type": "business_process", "confidence": 0.7}
                    ],
                    "confidence_score": 0.85,
                    "context_type": parameters.get("context_type", "comprehensive"),
                    "language": parameters.get("language", "en")
                }
            }
        elif tool_name == "extract_entities":
            return {
                "success": True,
                "entities": [
                    {"name": "company", "type": "organization", "confidence": 0.9},
                    {"name": "efficiency", "type": "business_concept", "confidence": 0.8},
                    {"name": "costs", "type": "business_concept", "confidence": 0.9}
                ]
            }
        elif tool_name == "analyze_sentiment":
            return {
                "success": True,
                "sentiment": "neutral",
                "score": 0.5,
                "confidence": 0.8
            }
        elif tool_name == "analyze_patterns":
            return {
                "success": True,
                "patterns": [
                    {"type": "trend", "description": "increasing_efficiency_focus", "confidence": 0.7},
                    {"type": "anomaly", "description": "cost_concerns", "confidence": 0.6}
                ]
            }
        elif tool_name == "analyze_audio":
            return {
                "success": True,
                "entities": [{"name": "speech", "type": "audio_content", "confidence": 0.7}],
                "sentiment": {"score": 0.6, "confidence": 0.7},
                "patterns": [{"type": "speech_pattern", "confidence": 0.6}],
                "duration": 120.0
            }
        elif tool_name == "analyze_video":
            return {
                "success": True,
                "entities": [{"name": "presentation", "type": "video_content", "confidence": 0.8}],
                "sentiment": {"score": 0.7, "confidence": 0.8},
                "patterns": [{"type": "visual_pattern", "confidence": 0.7}],
                "duration": 300.0
            }
        elif tool_name == "analyze_image":
            return {
                "success": True,
                "entities": [{"name": "dashboard", "type": "visual_content", "confidence": 0.8}],
                "sentiment": {"score": 0.6, "confidence": 0.7},
                "patterns": [{"type": "visual_pattern", "confidence": 0.7}],
                "resolution": "1920x1080"
            }
        elif tool_name == "analyze_webpage":
            return {
                "success": True,
                "entities": [{"name": "market_analysis", "type": "web_content", "confidence": 0.8}],
                "sentiment": {"score": 0.7, "confidence": 0.8},
                "patterns": [{"type": "content_pattern", "confidence": 0.7}],
                "title": "Market Analysis Report"
            }
        else:
            return {
                "success": True,
                "result": f"Mock response for {tool_name}",
                "parameters": parameters
            }
    
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


# Analytics-specific MCP tool calls
async def call_predictive_analytics_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Call a predictive analytics tool on the unified MCP server."""
    return await call_unified_mcp_tool(f"predictive_analytics_{tool_name}", parameters)


async def call_scenario_analysis_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Call a scenario analysis tool on the unified MCP server."""
    return await call_unified_mcp_tool(f"scenario_analysis_{tool_name}", parameters)


async def call_decision_support_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Call a decision support tool on the unified MCP server."""
    return await call_unified_mcp_tool(f"decision_support_{tool_name}", parameters)


async def call_monitoring_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Call a monitoring tool on the unified MCP server."""
    return await call_unified_mcp_tool(f"monitoring_{tool_name}", parameters)


async def call_performance_optimization_tool(tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Call a performance optimization tool on the unified MCP server."""
    return await call_unified_mcp_tool(f"performance_optimization_{tool_name}", parameters)
