"""
Comprehensive MCP server for all agent swarms - provides unified access to all agent tools.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import asyncio
from loguru import logger

# Import all MCP servers
from .strands_base_agent_server import create_strands_base_agent_mcp_server
from .text_agent_server import create_text_agent_mcp_server
from .simple_text_agent_server import create_simple_text_agent_mcp_server
from .text_agent_strands_server import create_text_agent_strands_mcp_server
from .text_agent_swarm_server import create_text_agent_swarm_mcp_server
from .audio_agent_server import create_audio_agent_mcp_server
from .vision_agent_server import create_vision_agent_mcp_server
from .web_agent_server import create_web_agent_mcp_server
from .orchestrator_agent_server import create_orchestrator_agent_mcp_server

# Import models
from core.models import (
    AnalysisRequest, 
    DataType,
    ProcessingStatus
)


class AgentSwarmRequest(BaseModel):
    """Request model for agent swarm operations."""
    agent_type: str = Field(..., description="Type of agent to use")
    operation: str = Field(..., description="Operation to perform")
    data: Dict[str, Any] = Field(..., description="Data for the operation")
    model_preference: Optional[str] = Field(None, description="Preferred model")


class AgentSwarmResponse(BaseModel):
    """Response model for agent swarm operations."""
    success: bool = Field(..., description="Whether the operation was successful")
    agent_type: str = Field(..., description="Type of agent used")
    operation: str = Field(..., description="Operation performed")
    result: Dict[str, Any] = Field(..., description="Operation result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentsSwarmMCPServer:
    """Comprehensive MCP server providing access to all agent swarms."""
    
    def __init__(self):
        # Initialize all MCP servers
        self.agents = {}
        self._initialize_agents()
        
        # Initialize the main MCP server
        self.mcp = None
        self._initialize_mcp()
        
        # Register all tools
        self._register_tools()
        
        logger.info("AgentsSwarm MCP Server initialized with all agent types")
    
    def _initialize_agents(self):
        """Initialize all agent MCP servers."""
        try:
            # Initialize each agent type
            self.agents["strands_base"] = create_strands_base_agent_mcp_server()
            self.agents["text"] = create_text_agent_mcp_server()
            self.agents["text_simple"] = create_simple_text_agent_mcp_server()
            self.agents["text_strands"] = create_text_agent_strands_mcp_server()
            self.agents["text_swarm"] = create_text_agent_swarm_mcp_server()
            self.agents["audio"] = create_audio_agent_mcp_server()
            self.agents["vision"] = create_vision_agent_mcp_server()
            self.agents["web"] = create_web_agent_mcp_server()
            self.agents["orchestrator"] = create_orchestrator_agent_mcp_server()
            
            logger.info(f"Initialized {len(self.agents)} agent types")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            # Continue with available agents
            pass
    
    def _initialize_mcp(self):
        """Initialize the MCP server."""
        try:
            from fastmcp import FastMCP
            self.mcp = FastMCP("Agents Swarm Server")
            logger.info("FastMCP initialized successfully")
        except ImportError:
            logger.warning("FastMCP not available, using mock MCP server")
            # Create a mock MCP server for development
            self.mcp = MockMCPServer("Agents Swarm Server")
    
    def _register_tools(self):
        """Register all tools from all agent swarms."""
        
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        # Register unified agent access tools
        @self.mcp.tool(
            description="Get information about all available agent types"
        )
        def get_all_agent_types() -> Dict[str, Any]:
            """Get information about all available agent types."""
            try:
                agent_info = {}
                for agent_type, agent_server in self.agents.items():
                    if hasattr(agent_server, 'mcp') and agent_server.mcp:
                        if hasattr(agent_server.mcp, 'tools'):
                            tool_count = len(agent_server.mcp.tools)
                        else:
                            tool_count = 0
                        
                        agent_info[agent_type] = {
                            "available": True,
                            "tool_count": tool_count,
                            "port": getattr(agent_server, 'port', 'unknown')
                        }
                    else:
                        agent_info[agent_type] = {
                            "available": False,
                            "error": "MCP server not initialized"
                        }
                
                return {
                    "success": True,
                    "total_agents": len(self.agents),
                    "agents": agent_info
                }
                
            except Exception as e:
                logger.error(f"Error getting agent types: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.mcp.tool(
            description="Execute a tool from a specific agent type"
        )
        async def execute_agent_tool(
            agent_type: str = Field(..., description="Type of agent to use"),
            tool_name: str = Field(..., description="Name of the tool to execute"),
            **kwargs
        ) -> Dict[str, Any]:
            """Execute a specific tool from a specific agent type."""
            try:
                if agent_type not in self.agents:
                    return {
                        "success": False,
                        "error": f"Agent type '{agent_type}' not found",
                        "available_types": list(self.agents.keys())
                    }
                
                agent_server = self.agents[agent_type]
                
                if not hasattr(agent_server, 'mcp') or not agent_server.mcp:
                    return {
                        "success": False,
                        "error": f"Agent type '{agent_type}' MCP server not initialized"
                    }
                
                if not hasattr(agent_server.mcp, 'tools'):
                    return {
                        "success": False,
                        "error": f"Agent type '{agent_type}' has no tools available"
                    }
                
                if tool_name not in agent_server.mcp.tools:
                    return {
                        "success": False,
                        "error": f"Tool '{tool_name}' not found in agent type '{agent_type}'",
                        "available_tools": list(agent_server.mcp.tools.keys())
                    }
                
                # Execute the tool
                tool_info = agent_server.mcp.tools[tool_name]
                if hasattr(tool_info, 'function'):
                    tool_func = tool_info['function']
                elif isinstance(tool_info, dict) and 'function' in tool_info:
                    tool_func = tool_info['function']
                else:
                    return {
                        "success": False,
                        "error": f"Tool '{tool_name}' function not accessible"
                    }
                
                # Call the tool function
                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(**kwargs)
                else:
                    result = tool_func(**kwargs)
                
                return {
                    "success": True,
                    "agent_type": agent_type,
                    "tool_name": tool_name,
                    "result": result
                }
                
            except Exception as e:
                logger.error(f"Error executing agent tool: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "agent_type": agent_type,
                    "tool_name": tool_name
                }
        
        @self.mcp.tool(
            description="Get all available tools from all agent types"
        )
        def get_all_available_tools() -> Dict[str, Any]:
            """Get all available tools from all agent types."""
            try:
                all_tools = {}
                
                for agent_type, agent_server in self.agents.items():
                    if hasattr(agent_server, 'mcp') and agent_server.mcp:
                        if hasattr(agent_server.mcp, 'tools'):
                            tools = {}
                            for tool_name, tool_info in agent_server.mcp.tools.items():
                                if isinstance(tool_info, dict) and 'description' in tool_info:
                                    tools[tool_name] = tool_info['description']
                                else:
                                    tools[tool_name] = "No description available"
                            
                            all_tools[agent_type] = tools
                        else:
                            all_tools[agent_type] = {}
                    else:
                        all_tools[agent_type] = {}
                
                return {
                    "success": True,
                    "total_agent_types": len(all_tools),
                    "tools_by_agent": all_tools
                }
                
            except Exception as e:
                logger.error(f"Error getting all tools: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.mcp.tool(
            description="Get health status of all agent types"
        )
        async def get_all_agent_health() -> Dict[str, Any]:
            """Get health status of all agent types."""
            try:
                health_status = {}
                
                for agent_type, agent_server in self.agents.items():
                    try:
                        # Try to get health status from each agent
                        if hasattr(agent_server, 'mcp') and agent_server.mcp:
                            if hasattr(agent_server.mcp, 'tools'):
                                # Look for health-related tools
                                health_tools = [
                                    tool_name for tool_name in agent_server.mcp.tools.keys()
                                    if 'health' in tool_name.lower() or 'status' in tool_name.lower()
                                ]
                                
                                if health_tools:
                                    # Use the first available health tool
                                    health_tool_name = health_tools[0]
                                    tool_info = agent_server.mcp.tools[health_tool_name]
                                    
                                    if hasattr(tool_info, 'function'):
                                        tool_func = tool_info['function']
                                    elif isinstance(tool_info, dict) and 'function' in tool_info:
                                        tool_func = tool_info['function']
                                    else:
                                        health_status[agent_type] = {
                                            "status": "unknown",
                                            "error": "Health tool function not accessible"
                                        }
                                        continue
                                    
                                    # Call the health tool
                                    try:
                                        if asyncio.iscoroutinefunction(tool_func):
                                            result = await tool_func()
                                        else:
                                            result = tool_func()
                                        
                                        health_status[agent_type] = {
                                            "status": "healthy" if result.get("success", False) else "unhealthy",
                                            "details": result
                                        }
                                    except Exception as e:
                                        health_status[agent_type] = {
                                            "status": "error",
                                            "error": str(e)
                                        }
                                else:
                                    health_status[agent_type] = {
                                        "status": "unknown",
                                        "error": "No health tools available"
                                    }
                            else:
                                health_status[agent_type] = {
                                    "status": "unknown",
                                    "error": "No tools available"
                                }
                        else:
                            health_status[agent_type] = {
                                "status": "unknown",
                                "error": "MCP server not initialized"
                            }
                    except Exception as e:
                        health_status[agent_type] = {
                            "status": "error",
                            "error": str(e)
                        }
                
                return {
                    "success": True,
                    "total_agents": len(health_status),
                    "health_status": health_status
                }
                
            except Exception as e:
                logger.error(f"Error getting all agent health: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.mcp.tool(
            description="Start all agent types"
        )
        async def start_all_agents() -> Dict[str, Any]:
            """Start all agent types."""
            try:
                start_results = {}
                
                for agent_type, agent_server in self.agents.items():
                    try:
                        # Try to start each agent
                        if hasattr(agent_server, 'start'):
                            await agent_server.start()
                            start_results[agent_type] = {
                                "success": True,
                                "message": f"Agent {agent_type} started successfully"
                            }
                        else:
                            start_results[agent_type] = {
                                "success": False,
                                "error": "No start method available"
                            }
                    except Exception as e:
                        start_results[agent_type] = {
                            "success": False,
                            "error": str(e)
                        }
                
                return {
                    "success": True,
                    "total_agents": len(start_results),
                    "start_results": start_results
                }
                
            except Exception as e:
                logger.error(f"Error starting all agents: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.mcp.tool(
            description="Stop all agent types"
        )
        async def stop_all_agents() -> Dict[str, Any]:
            """Stop all agent types."""
            try:
                stop_results = {}
                
                for agent_type, agent_server in self.agents.items():
                    try:
                        # Try to stop each agent
                        if hasattr(agent_server, 'stop'):
                            await agent_server.stop()
                            stop_results[agent_type] = {
                                "success": True,
                                "message": f"Agent {agent_type} stopped successfully"
                            }
                        else:
                            stop_results[agent_type] = {
                                "success": False,
                                "error": "No stop method available"
                            }
                    except Exception as e:
                        stop_results[agent_type] = {
                            "success": False,
                            "error": str(e)
                        }
                
                return {
                    "success": True,
                    "total_agents": len(stop_results),
                    "stop_results": stop_results
                }
                
            except Exception as e:
                logger.error(f"Error stopping all agents: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
    
    def run(self, host: str = "localhost", port: int = 8000, debug: bool = False):
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
    
    def run(self, host: str = "localhost", port: int = 8000, debug: bool = False):
        """Mock run method."""
        logger.info(f"Mock MCP server {self.name} would run on {host}:{port}")


def create_agents_swarm_mcp_server() -> AgentsSwarmMCPServer:
    """Create and return an AgentsSwarm MCP server instance."""
    return AgentsSwarmMCPServer()


if __name__ == "__main__":
    # Create and run the server
    server = create_agents_swarm_mcp_server()
    server.run(debug=True)
