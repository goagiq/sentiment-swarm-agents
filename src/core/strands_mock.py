"""
Mock Strands framework implementation for testing purposes.
This provides the core functionality needed to test the agents without 
external dependencies.
"""

import asyncio
import json
import re
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
        elif "请从以下中文文本中精确提取实体" in prompt:
            # Enhanced Chinese entity extraction response
            response = self._generate_chinese_entity_response(prompt)
        elif "extract entities" in prompt.lower():
            # English entity extraction response
            response = self._generate_english_entity_response(prompt)
        else:
            response = f"Mock response from agent '{self.name}' for: {prompt[:50]}..."
        
        # Store in conversation history
        self.conversation_history.append({
            "prompt": prompt,
            "response": response,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        return response

    def _generate_chinese_entity_response(self, prompt: str) -> str:
        """Generate structured Chinese entity extraction response."""
        # Extract text from prompt
        text_match = re.search(r'文本：(.+?)\n', prompt)
        if not text_match:
            return '{"entities": []}'
        
        text = text_match.group(1)
        entities = []
        
        # Extract person names
        person_patterns = [
            r'习近平', r'李克强', r'马云', r'马化腾', r'任正非', r'李彦宏'
        ]
        for pattern in person_patterns:
            if re.search(pattern, text):
                entities.append({
                    "text": pattern,
                    "type": "PERSON",
                    "confidence": 0.9
                })
        
        # Extract organizations
        org_patterns = [
            r'华为', r'阿里巴巴', r'腾讯', r'百度', r'清华大学', r'北京大学'
        ]
        for pattern in org_patterns:
            if re.search(pattern, text):
                entities.append({
                    "text": pattern,
                    "type": "ORGANIZATION",
                    "confidence": 0.9
                })
        
        # Extract locations
        loc_patterns = [
            r'北京', r'上海', r'深圳', r'广州', r'杭州', r'南京'
        ]
        for pattern in loc_patterns:
            if re.search(pattern, text):
                entities.append({
                    "text": pattern,
                    "type": "LOCATION",
                    "confidence": 0.9
                })
        
        # Extract technical terms
        tech_patterns = [
            r'人工智能', r'机器学习', r'深度学习', r'神经网络'
        ]
        for pattern in tech_patterns:
            if re.search(pattern, text):
                entities.append({
                    "text": pattern,
                    "type": "CONCEPT",
                    "confidence": 0.9
                })
        
        return json.dumps({"entities": entities}, ensure_ascii=False)

    def _generate_english_entity_response(self, prompt: str) -> str:
        """Generate structured English entity extraction response."""
        # Extract text from prompt
        text_match = re.search(r'Text: (.+?)\n', prompt)
        if not text_match:
            return '{"entities": []}'
        
        text = text_match.group(1)
        entities = []
        
        # Extract person names
        person_patterns = [
            r'Joe Biden', r'Elon Musk', r'Donald Trump', r'Barack Obama'
        ]
        for pattern in person_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                entities.append({
                    "name": pattern,
                    "type": "person",
                    "importance": "high",
                    "description": f"Known person: {pattern}"
                })
        
        # Extract organizations
        org_patterns = [
            r'Microsoft', r'Apple', r'Google', r'Amazon'
        ]
        for pattern in org_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                entities.append({
                    "name": pattern,
                    "type": "organization",
                    "importance": "high",
                    "description": f"Known organization: {pattern}"
                })
        
        # Extract technical terms
        tech_patterns = [
            r'AI', r'Artificial Intelligence', r'Machine Learning'
        ]
        for pattern in tech_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                entities.append({
                    "name": pattern,
                    "type": "concept",
                    "importance": "medium",
                    "description": f"Technical term: {pattern}"
                })
        
        return json.dumps({"entities": entities})
    
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
