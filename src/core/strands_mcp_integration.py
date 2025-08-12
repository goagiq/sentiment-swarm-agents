"""
Strands MCP Integration - Proper MCP tool integration following official documentation.
This provides the correct pattern for MCP tools even with the mock implementation.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Callable
from loguru import logger

# Import the mock Strands implementation
from src.core.strands_mock import Agent, Swarm, tool, strands


class MCPToolManager:
    """Manages MCP tools integration with Strands."""
    
    def __init__(self, mcp_server_url: str = "http://localhost:8000/mcp"):
        self.mcp_server_url = mcp_server_url
        self.tools = []
        self.agents = {}
        self.swarms = {}
        
        logger.info(f"Initializing MCP Tool Manager for {mcp_server_url}")
    
    def register_mcp_tool(self, name: str, description: str, func: Callable):
        """Register an MCP tool following the proper pattern."""
        try:
            # Create tool using the mock decorator
            tool_decorator = tool(name=name, description=description)
            decorated_func = tool_decorator(func)
            
            # Store tool information
            tool_info = {
                "name": name,
                "description": description,
                "function": decorated_func,
                "type": "mcp"
            }
            
            self.tools.append(tool_info)
            logger.info(f"Registered MCP tool: {name}")
            
            return decorated_func
            
        except Exception as e:
            logger.error(f"Failed to register MCP tool {name}: {e}")
            raise
    
    def create_agent_with_mcp_tools(self, name: str, system_prompt: str = None) -> Agent:
        """Create a Strands agent with MCP tools following the proper pattern."""
        try:
            # Get tool functions
            tool_functions = [tool_info["function"] for tool_info in self.tools]
            
            # Create agent with tools
            agent = Agent(
                name=name,
                system_prompt=system_prompt,
                tools=tool_functions
            )
            
            self.agents[name] = agent
            logger.info(f"Created agent '{name}' with {len(tool_functions)} MCP tools")
            
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
            
            result = await agent.run(prompt)
            logger.info(f"Agent '{agent_name}' completed execution")
            return result
            
        except Exception as e:
            logger.error(f"Failed to run agent '{agent_name}': {e}")
            raise
    
    async def run_swarm_async(self, swarm_name: str, prompt: str) -> str:
        """Run a swarm asynchronously."""
        try:
            swarm = self.swarms.get(swarm_name)
            if not swarm:
                raise ValueError(f"Swarm '{swarm_name}' not found")
            
            result = await swarm.run(prompt)
            logger.info(f"Swarm '{swarm_name}' completed execution")
            return result
            
        except Exception as e:
            logger.error(f"Failed to run swarm '{swarm_name}': {e}")
            raise
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        return [
            {
                "name": tool_info["name"],
                "description": tool_info["description"],
                "type": tool_info["type"]
            }
            for tool_info in self.tools
        ]
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        return {
            "agents": {name: {"name": agent.name, "tools_count": len(agent.tools)} 
                      for name, agent in self.agents.items()},
            "swarms": {name: {"name": name, "agents_count": len(swarm.agents)} 
                      for name, swarm in self.swarms.items()},
            "total_tools": len(self.tools)
        }


# Global MCP tool manager instance
mcp_tool_manager = MCPToolManager()


# Example MCP tool registrations following the proper pattern
@tool(name="analyze_sentiment", description="Analyze sentiment of text content")
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment of the given text."""
    try:
        # Mock sentiment analysis
        if any(word in text.lower() for word in ["love", "great", "amazing", "wonderful"]):
            sentiment = "positive"
            confidence = 0.9
        elif any(word in text.lower() for word in ["hate", "terrible", "awful", "bad"]):
            sentiment = "negative"
            confidence = 0.9
        else:
            sentiment = "neutral"
            confidence = 0.7
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "text": text
        }
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        return {"error": str(e)}


@tool(name="extract_entities", description="Extract entities from text content")
def extract_entities(text: str) -> Dict[str, Any]:
    """Extract entities from the given text."""
    try:
        # Mock entity extraction
        entities = []
        
        # Simple entity detection
        import re
        
        # Person names
        person_patterns = [r'\b[A-Z][a-z]+ [A-Z][a-z]+\b']
        for pattern in person_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append({
                    "text": match,
                    "type": "PERSON",
                    "confidence": 0.8
                })
        
        # Organizations
        org_patterns = [r'\b[A-Z][A-Z]+\b']  # Simple pattern for orgs
        for pattern in org_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 2:  # Filter out short matches
                    entities.append({
                        "text": match,
                        "type": "ORGANIZATION",
                        "confidence": 0.7
                    })
        
        return {
            "entities": entities,
            "text": text
        }
    except Exception as e:
        logger.error(f"Error in entity extraction: {e}")
        return {"error": str(e)}


@tool(name="process_image", description="Process and analyze image content")
def process_image(image_path: str) -> Dict[str, Any]:
    """Process and analyze the given image."""
    try:
        # Mock image processing
        return {
            "image_path": image_path,
            "objects_detected": ["person", "car", "building"],
            "confidence": 0.85,
            "processing_time": 1.2
        }
    except Exception as e:
        logger.error(f"Error in image processing: {e}")
        return {"error": str(e)}


# Helper functions following the official documentation pattern
def create_mcp_agent(name: str, system_prompt: str = None) -> Agent:
    """Create an agent with MCP tools following the official pattern."""
    return mcp_tool_manager.create_agent_with_mcp_tools(name, system_prompt)


def create_mcp_swarm(agents: List[Agent], name: str = "swarm") -> Swarm:
    """Create a swarm with MCP-enabled agents following the official pattern."""
    return mcp_tool_manager.create_swarm_with_mcp_tools(agents, name)


async def run_mcp_agent(agent_name: str, prompt: str) -> str:
    """Run an agent asynchronously following the official pattern."""
    return await mcp_tool_manager.run_agent_async(agent_name, prompt)


async def run_mcp_swarm(swarm_name: str, prompt: str) -> str:
    """Run a swarm asynchronously following the official pattern."""
    return await mcp_tool_manager.run_swarm_async(swarm_name, prompt)
