"""
Mock Strands framework implementation for testing purposes.
This provides the core functionality needed to test the agents without 
external dependencies.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class Tool:
    """Mock tool implementation for Strands."""
    name: str
    description: str
    func: Callable
    parameters: Dict[str, Any]


class Agent:
    """Mock Strands Agent implementation."""
    
    def __init__(
        self, 
        name: str, 
        model: str = "llama3.2:latest",
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None
    ):
        self.name = name
        self.model = model
        self.tools = tools or []
        self.conversation_history: List[Dict[str, Any]] = []
        self._system_prompt = system_prompt
        
        logger.info(
            f"Mock Strands Agent '{name}' initialized with model '{model}'"
        )
    
    @property
    def system_prompt(self) -> Optional[str]:
        """Get the system prompt."""
        return self._system_prompt
    
    @system_prompt.setter
    def system_prompt(self, value: str):
        """Set the system prompt."""
        self._system_prompt = value
        logger.info(f"Agent '{self.name}' system prompt updated")
    
    async def run(self, prompt: str, **kwargs) -> str:
        """Mock run method that simulates agent execution."""
        logger.info(f"Agent '{self.name}' processing prompt: {prompt[:100]}...")
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Mock response based on prompt content
        if "sentiment" in prompt.lower():
            if "positive" in prompt.lower() or "happy" in prompt.lower():
                response = "The sentiment analysis indicates a positive emotional tone."
            elif "negative" in prompt.lower() or "sad" in prompt.lower():
                response = "The sentiment analysis indicates a negative emotional tone."
            else:
                response = "The sentiment analysis indicates a neutral emotional tone."
        else:
            response = f"Mock response from agent '{self.name}' for: {prompt[:50]}..."
        
        # Store in conversation history
        self.conversation_history.append({
            "prompt": prompt,
            "response": response,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        return response
    
    async def invoke_async(self, prompt: str) -> str:
        """Mock async invocation method."""
        return await self.run(prompt)
    
    async def start(self):
        """Mock start method."""
        logger.info(f"Mock Strands Agent '{self.name}' started.")
    
    async def stop(self):
        """Mock stop method."""
        logger.info(f"Mock Strands Agent '{self.name}' stopped.")
    
    def add_tool(self, tool: Tool):
        """Add a tool to the agent."""
        self.tools.append(tool)
        logger.info(f"Added tool '{tool.name}' to agent '{self.name}'")
    
    def get_tools(self) -> List[Tool]:
        """Get all tools available to this agent."""
        return self.tools


class StrandsFramework:
    """Mock Strands framework for testing."""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tools: Dict[str, Tool] = {}
        
    def create_agent(self, name: str, model: str = "llama3.2:latest") -> Agent:
        """Create a new agent."""
        agent = Agent(name, model)
        self.agents[name] = agent
        return agent
    
    def register_tool(self, tool: Tool):
        """Register a tool in the framework."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool '{tool.name}' in Strands framework")
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name."""
        return self.agents.get(name)


class Swarm:
    """Mock Swarm class for multi-agent coordination."""
    
    def __init__(self, agents: List[Agent], max_handoffs: int = 20, max_iterations: int = 20, 
                 execution_timeout: float = 900.0, node_timeout: float = 300.0,
                 repetitive_handoff_detection_window: int = 8, 
                 repetitive_handoff_min_unique_agents: int = 3):
        self.agents = agents
        self.coordinator = Agent("swarm_coordinator", "llama3.2:latest")
        self.max_handoffs = max_handoffs
        self.max_iterations = max_iterations
        self.execution_timeout = execution_timeout
        self.node_timeout = node_timeout
        self.repetitive_handoff_detection_window = repetitive_handoff_detection_window
        self.repetitive_handoff_min_unique_agents = repetitive_handoff_min_unique_agents
        logger.info(f"Mock Swarm initialized with {len(agents)} agents")
    
    async def run(self, prompt: str, **kwargs) -> str:
        """Mock swarm execution."""
        logger.info(f"Mock Swarm processing prompt: {prompt[:100]}...")
        
        # Simulate coordinated processing
        responses = []
        for agent in self.agents:
            response = await agent.run(prompt)
            responses.append(response)
        
        # Combine responses
        combined_response = f"Swarm analysis complete. {len(responses)} agents processed the request."
        
        return combined_response
    
    async def invoke_async(self, prompt: str) -> str:
        """Mock async invocation method for swarm."""
        return await self.run(prompt)
    
    def add_agent(self, agent: Agent):
        """Add an agent to the swarm."""
        self.agents.append(agent)
        logger.info(f"Added agent '{agent.name}' to swarm")
    
    def get_agents(self) -> List[Agent]:
        """Get all agents in the swarm."""
        return self.agents


# Global instance
strands = StrandsFramework()


def tool(name: str, description: str = ""):
    """Decorator to create tools for agents."""
    def decorator(func: Callable) -> Callable:
        # Create tool metadata
        tool_obj = Tool(
            name=name,
            description=description,
            func=func,
            parameters={}  # Simplified for mock
        )
        
        # Register in framework
        strands.register_tool(tool_obj)
        
        return func
    return decorator


def agent(name: str, model: str = "llama3.2:latest"):
    """Decorator to create agents."""
    def decorator(cls):
        # Create agent instance
        agent_instance = strands.create_agent(name, model)
        
        # Add tools from class methods
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if hasattr(attr, '_is_tool'):
                agent_instance.add_tool(attr._tool_obj)
        
        return cls
    return decorator
