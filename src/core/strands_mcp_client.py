"""
Strands MCP Client implementation for proper tool integration.
This follows the official Strands documentation for MCP tool setup.
"""

import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from mcp.client.streamable_http import streamablehttp_client
    from strands import Agent
    from strands.tools.mcp.mcp_client import MCPClient
    from strands.multiagent import Swarm
    from strands.types.content import ContentBlock
    STRANDS_AVAILABLE = True
except ImportError:
    # Use mock implementation
    from src.core.strands_mock import Agent, Swarm
    STRANDS_AVAILABLE = False
    logger.info("ℹ️ Using mock Strands implementation for MCP client")


class StrandsMCPClient:
    """Proper Strands MCP client implementation."""
    
    def __init__(self, mcp_server_url: str = "http://localhost:8000/mcp"):
        self.mcp_server_url = mcp_server_url
        self.streamable_http_mcp_client = None
        self.tools = []
        self.agents = {}
        self.swarms = {}
        
        logger.info(f"Initializing Strands MCP client for {mcp_server_url}")
    
    def _create_mcp_client(self):
        """Create the MCP client with streamable HTTP."""
        try:
            self.streamable_http_mcp_client = MCPClient(
                lambda: streamablehttp_client(self.mcp_server_url)
            )
            logger.success("MCP client created successfully")
        except Exception as e:
            logger.error(f"Failed to create MCP client: {e}")
            raise
    
    def get_tools_sync(self) -> List[Dict[str, Any]]:
        """Get tools from MCP server synchronously."""
        try:
            if not self.streamable_http_mcp_client:
                self._create_mcp_client()
            
            with self.streamable_http_mcp_client:
                tools = self.streamable_http_mcp_client.list_tools_sync()
                logger.info(f"Retrieved {len(tools)} tools from MCP server")
                return tools
        except Exception as e:
            logger.error(f"Failed to get tools from MCP server: {e}")
            return []
    
    async def get_tools_async(self) -> List[Dict[str, Any]]:
        """Get tools from MCP server asynchronously."""
        try:
            if not self.streamable_http_mcp_client:
                self._create_mcp_client()
            
            async with self.streamable_http_mcp_client:
                tools = await self.streamable_http_mcp_client.list_tools_async()
                logger.info(f"Retrieved {len(tools)} tools from MCP server")
                return tools
        except Exception as e:
            logger.error(f"Failed to get tools from MCP server: {e}")
            return []
    
    def create_agent_with_mcp_tools(self, name: str, system_prompt: str = None) -> Agent:
        """Create a Strands agent with MCP tools."""
        try:
            # Get tools from MCP server
            tools = self.get_tools_sync()
            
            if not tools:
                logger.warning("No tools available from MCP server")
                # Create agent without tools
                agent = Agent(
                    name=name,
                    system_prompt=system_prompt
                )
            else:
                # Create agent with MCP tools
                agent = Agent(
                    name=name,
                    system_prompt=system_prompt,
                    tools=tools
                )
                logger.info(f"Created agent '{name}' with {len(tools)} MCP tools")
            
            self.agents[name] = agent
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent with MCP tools: {e}")
            raise
    
    def create_swarm_with_mcp_tools(self, agents: List[Agent], name: str = "swarm") -> Swarm:
        """Create a Strands swarm with MCP-enabled agents."""
        try:
            swarm = Swarm(agents)
            self.swarms[name] = swarm
            logger.info(f"Created swarm '{name}' with {len(agents)} agents")
            return swarm
        except Exception as e:
            logger.error(f"Failed to create swarm: {e}")
            raise
    
    async def run_agent_async(self, agent_name: str, prompt: str) -> str:
        """Run an agent asynchronously."""
        try:
            agent = self.agents.get(agent_name)
            if not agent:
                raise ValueError(f"Agent '{agent_name}' not found")
            
            result = await agent.invoke_async(prompt)
            logger.info(f"Agent '{agent_name}' completed execution")
            return result
            
        except Exception as e:
            logger.error(f"Failed to run agent '{agent_name}': {e}")
            raise
    
    async def run_swarm_async(self, swarm_name: str, content_blocks: List[ContentBlock]) -> str:
        """Run a swarm asynchronously with multi-modal content."""
        try:
            swarm = self.swarms.get(swarm_name)
            if not swarm:
                raise ValueError(f"Swarm '{swarm_name}' not found")
            
            result = await swarm.invoke_async(content_blocks)
            logger.info(f"Swarm '{swarm_name}' completed execution")
            return result
            
        except Exception as e:
            logger.error(f"Failed to run swarm '{swarm_name}': {e}")
            raise
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        return self.get_tools_sync()
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        return {
            "agents": {name: {"name": agent.name, "tools_count": len(agent.tools)} 
                      for name, agent in self.agents.items()},
            "swarms": {name: {"name": name, "agents_count": len(swarm.agents)} 
                      for name, swarm in self.swarms.items()},
            "total_tools": len(self.get_tools_sync())
        }


# Global instance
strands_mcp_client = StrandsMCPClient()


def create_mcp_agent(name: str, system_prompt: str = None) -> Agent:
    """Helper function to create an agent with MCP tools."""
    return strands_mcp_client.create_agent_with_mcp_tools(name, system_prompt)


def create_mcp_swarm(agents: List[Agent], name: str = "swarm") -> Swarm:
    """Helper function to create a swarm with MCP-enabled agents."""
    return strands_mcp_client.create_swarm_with_mcp_tools(agents, name)


async def run_mcp_agent(agent_name: str, prompt: str) -> str:
    """Helper function to run an agent asynchronously."""
    return await strands_mcp_client.run_agent_async(agent_name, prompt)


async def run_mcp_swarm(swarm_name: str, content_blocks: List[ContentBlock]) -> str:
    """Helper function to run a swarm asynchronously."""
    return await strands_mcp_client.run_swarm_async(swarm_name, content_blocks)
