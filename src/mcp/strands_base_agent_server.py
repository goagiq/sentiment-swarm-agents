"""
MCP server for StrandsBaseAgent - provides base agent functionality using Strands framework.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import asyncio
from loguru import logger

# Import the correct models and agents
from core.models import (
    AnalysisRequest, 
    DataType,
    ProcessingStatus
)
from agents.base_agent import StrandsBaseAgent


class StrandsBaseAgentRequest(BaseModel):
    """Request model for base agent operations."""
    agent_id: Optional[str] = Field(None, description="Specific agent ID to target")
    operation: str = Field(..., description="Operation to perform: status, start, stop, health")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional data for the operation")


class StrandsBaseAgentResponse(BaseModel):
    """Response model for base agent operations."""
    success: bool = Field(..., description="Whether the operation was successful")
    agent_id: str = Field(..., description="Agent ID that performed the operation")
    agent_type: str = Field(..., description="Type of the agent")
    status: str = Field(..., description="Current agent status")
    message: str = Field(..., description="Result message")
    data: Optional[Dict[str, Any]] = Field(None, description="Additional response data")
    timestamp: float = Field(..., description="Operation timestamp")


class StrandsBaseAgentMCPServer:
    """MCP server providing base agent tools from StrandsBaseAgent."""
    
    def __init__(self, model_name: Optional[str] = None):
        # Initialize the base agent strands
        self.base_agent = StrandsBaseAgent(model_name=model_name)
        
        # Note: FastMCP will be imported when available
        # For now, we'll create a mock structure
        self.mcp = None
        self._initialize_mcp()
        
        # Register tools
        self._register_tools()
        
        model_name = self.base_agent.model_name
        logger.info(f"StrandsBaseAgent MCP Server initialized with model: {model_name}")
    
    def _initialize_mcp(self):
        """Initialize the MCP server."""
        try:
            from fastmcp import FastMCP
            self.mcp = FastMCP("StrandsBaseAgent Server")
            logger.info("FastMCP initialized successfully")
        except ImportError:
            logger.warning("FastMCP not available, using mock MCP server")
            # Create a mock MCP server for development
            self.mcp = MockMCPServer("StrandsBaseAgent Server")
    
    def _register_tools(self):
        """Register all base agent tools from StrandsBaseAgent."""
        
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        @self.mcp.tool(
            description="Get the status of a StrandsBaseAgent"
        )
        async def get_agent_status_strands(
            agent_id: Optional[str] = Field(None, description="Specific agent ID to check")
        ) -> Dict[str, Any]:
            """Get the current status of a StrandsBaseAgent."""
            try:
                # If no specific agent ID, use the current one
                target_agent = self.base_agent if agent_id is None or agent_id == self.base_agent.agent_id else None
                
                if target_agent is None:
                    return {
                        "success": False,
                        "error": f"Agent with ID {agent_id} not found",
                        "available_agents": [self.base_agent.agent_id]
                    }
                
                status = target_agent.get_status()
                
                return {
                    "success": True,
                    "agent_id": status["agent_id"],
                    "agent_type": status["agent_type"],
                    "status": status["status"],
                    "current_load": status["current_load"],
                    "max_capacity": status["max_capacity"],
                    "last_heartbeat": str(status["last_heartbeat"]),
                    "metadata": status["metadata"],
                    "tools_count": status["tools_count"]
                }
                
            except Exception as e:
                logger.error(f"Error getting agent status: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.mcp.tool(
            description="Start a StrandsBaseAgent"
        )
        async def start_agent_strands(
            agent_id: Optional[str] = Field(None, description="Specific agent ID to start")
        ) -> Dict[str, Any]:
            """Start a StrandsBaseAgent."""
            try:
                # If no specific agent ID, use the current one
                target_agent = self.base_agent if agent_id is None or agent_id == self.base_agent.agent_id else None
                
                if target_agent is None:
                    return {
                        "success": False,
                        "error": f"Agent with ID {agent_id} not found",
                        "available_agents": [self.base_agent.agent_id]
                    }
                
                await target_agent.start()
                
                return {
                    "success": True,
                    "agent_id": target_agent.agent_id,
                    "message": f"Agent {target_agent.agent_id} started successfully",
                    "status": target_agent.status
                }
                
            except Exception as e:
                logger.error(f"Error starting agent: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.mcp.tool(
            description="Stop a StrandsBaseAgent"
        )
        async def stop_agent_strands(
            agent_id: Optional[str] = Field(None, description="Specific agent ID to stop")
        ) -> Dict[str, Any]:
            """Stop a StrandsBaseAgent."""
            try:
                # If no specific agent ID, use the current one
                target_agent = self.base_agent if agent_id is None or agent_id == self.base_agent.agent_id else None
                
                if target_agent is None:
                    return {
                        "success": False,
                        "error": f"Agent with ID {agent_id} not found",
                        "available_agents": [self.base_agent.agent_id]
                    }
                
                await target_agent.stop()
                
                return {
                    "success": True,
                    "agent_id": target_agent.agent_id,
                    "message": f"Agent {target_agent.agent_id} stopped successfully",
                    "status": target_agent.status
                }
                
            except Exception as e:
                logger.error(f"Error stopping agent: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.mcp.tool(
            description="Check if a StrandsBaseAgent can process a specific request"
        )
        async def can_process_request_strands(
            data_type: str = Field(..., description="Type of data to process"),
            content: str = Field(..., description="Content to process"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Check if a StrandsBaseAgent can process a specific request."""
            try:
                # Create a mock request to test can_process
                from core.models import AnalysisRequest
                request = AnalysisRequest(
                    data_type=DataType(data_type),
                    content=content,
                    language=language
                )
                
                can_process = await self.base_agent.can_process(request)
                
                return {
                    "success": True,
                    "can_process": can_process,
                    "agent_id": self.base_agent.agent_id,
                    "data_type": data_type,
                    "reason": "Agent can process this request" if can_process else "Agent cannot process this request"
                }
                
            except Exception as e:
                logger.error(f"Error checking if agent can process request: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.mcp.tool(
            description="Get StrandsBaseAgent capabilities and configuration"
        )
        def get_base_agent_capabilities() -> Dict[str, Any]:
            """Get StrandsBaseAgent capabilities and configuration."""
            try:
                status = self.base_agent.get_status()
                
                return {
                    "success": True,
                    "agent_id": status["agent_id"],
                    "agent_type": status["agent_type"],
                    "capabilities": {
                        "max_capacity": status["max_capacity"],
                        "current_load": status["current_load"],
                        "tools_count": status["tools_count"],
                        "status": status["status"]
                    },
                    "metadata": status["metadata"],
                    "model": self.base_agent.model_name,
                    "available_operations": [
                        "get_status",
                        "start",
                        "stop", 
                        "can_process",
                        "get_capabilities"
                    ]
                }
                
            except Exception as e:
                logger.error(f"Error getting agent capabilities: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.mcp.tool(
            description="Get agent heartbeat and health information"
        )
        async def get_agent_health_strands(
            agent_id: Optional[str] = Field(None, description="Specific agent ID to check")
        ) -> Dict[str, Any]:
            """Get agent heartbeat and health information."""
            try:
                # If no specific agent ID, use the current one
                target_agent = self.base_agent if agent_id is None or agent_id == self.base_agent.agent_id else None
                
                if target_agent is None:
                    return {
                        "success": False,
                        "error": f"Agent with ID {agent_id} not found",
                        "available_agents": [self.base_agent.agent_id]
                    }
                
                status = target_agent.get_status()
                
                # Calculate time since last heartbeat
                import time
                current_time = time.time()
                last_heartbeat_time = status["last_heartbeat"].timestamp()
                time_since_heartbeat = current_time - last_heartbeat_time
                
                # Determine health status
                if time_since_heartbeat < 60:  # Less than 1 minute
                    health_status = "excellent"
                elif time_since_heartbeat < 300:  # Less than 5 minutes
                    health_status = "good"
                elif time_since_heartbeat < 900:  # Less than 15 minutes
                    health_status = "fair"
                else:
                    health_status = "poor"
                
                return {
                    "success": True,
                    "agent_id": status["agent_id"],
                    "health_status": health_status,
                    "last_heartbeat": str(status["last_heartbeat"]),
                    "time_since_heartbeat_seconds": time_since_heartbeat,
                    "current_status": status["status"],
                    "load_percentage": (status["current_load"] / status["max_capacity"]) * 100
                }
                
            except Exception as e:
                logger.error(f"Error getting agent health: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
    
    def run(self, host: str = "localhost", port: int = 8007, debug: bool = False):
        """Run the MCP server."""
        if self.mcp:
            self.mcp.run(host=host, port=port, debug=debug)
        else:
            logger.error("MCP server not initialized")


class MockMCPServer:
    """Mock MCP server for development when FastMCP is not available."""
    
    def __init__(self, name: str):
        self.name = name
        self.tools = {}
    
    def tool(self, description: str):
        """Mock tool decorator."""
        def decorator(func):
            self.tools[func.__name__] = {
                "function": func,
                "description": description
            }
            return func
        return decorator
    
    def run(self, host: str = "localhost", port: int = 8007, debug: bool = False):
        """Mock run method."""
        logger.info(f"Mock MCP server {self.name} would run on {host}:{port}")


def create_strands_base_agent_mcp_server(model_name: Optional[str] = None) -> StrandsBaseAgentMCPServer:
    """Create and return a StrandsBaseAgent MCP server instance."""
    return StrandsBaseAgentMCPServer(model_name=model_name)


if __name__ == "__main__":
    # Create and run the server
    server = create_strands_base_agent_mcp_server()
    server.run(debug=True)
