# MCP Implementation Guide: FastMCP with Swarm Agents and Orchestrator

## 📋 **Overview**

This guide provides step-by-step instructions for implementing Model Context Protocol (MCP) servers with FastMCP, including swarm agents and orchestrator patterns. Based on the [sentiment-swarm-agents](https://raw.githubusercontent.com/goagiq/sentiment-swarm-agents/refs/heads/master/main.py) implementation.

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
pip install fastmcp mcp
```

### **2. Basic MCP Server Structure**
```python
from fastmcp import FastMCP
import asyncio
from typing import List, Dict, Any

class OptimizedMCPServer:
    def __init__(self):
        self.mcp = FastMCP("Your Server Name")
        self.agents = {}
        self._initialize_agents()
        self._register_tools()
    
    def _initialize_agents(self):
        # Initialize your agents here
        pass
    
    def _register_tools(self):
        # Register MCP tools here
        pass
    
    def run(self, host="localhost", port=8000):
        return self.mcp.run(transport="streamable-http")
```

## 🔧 **Step-by-Step Implementation**

### **Step 1: Set Up Project Structure**
```
your_project/
├── src/
│   ├── core/
│   │   ├── mcp_server.py      # Main MCP server
│   │   ├── agents/            # Agent implementations
│   │   └── orchestrator.py    # Orchestrator logic
│   ├── config/
│   │   ├── settings.py        # Configuration settings
│   │   └── config.py          # Model configurations
│   └── tools/                 # MCP tool implementations
├── main.py                    # Entry point
└── requirements.txt
```

### **Step 2: Create Agent Base Classes**
```python
# src/core/agents/base_agent.py
class BaseAgent:
    def __init__(self, name: str, system_prompt: str, tools: List = None):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.status = "idle"
    
    async def start(self):
        self.status = "active"
    
    async def stop(self):
        self.status = "stopped"
    
    async def run(self, prompt: str):
        # Implement agent logic here
        return f"Agent {self.name} processed: {prompt}"
    
    def get_status(self):
        return {
            "name": self.name,
            "status": self.status,
            "tools_count": len(self.tools)
        }
```

### **Step 3: Implement Specialized Agents**
```python
# src/core/agents/sentiment_agent.py
from .base_agent import BaseAgent

class SentimentAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="sentiment_analyzer",
            system_prompt="You are a sentiment analysis expert.",
            tools=[]
        )
    
    async def analyze_sentiment(self, text: str):
        # Implement sentiment analysis logic
        return {"sentiment": "positive", "confidence": 0.8}

# src/core/agents/vision_agent.py
class VisionAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="vision_analyzer",
            system_prompt="You are a computer vision expert.",
            tools=[]
        )
    
    async def analyze_image(self, image_path: str):
        # Implement image analysis logic
        return {"objects": ["person", "car"], "confidence": 0.9}
```

### **Step 4: Create Orchestrator**
```python
# src/core/orchestrator.py
class Orchestrator:
    def __init__(self, agents: Dict[str, BaseAgent]):
        self.agents = agents
        self.swarms = {}
    
    def create_swarm(self, name: str, agent_names: List[str]):
        """Create a swarm with specified agents."""
        swarm_agents = [self.agents[name] for name in agent_names if name in self.agents]
        self.swarms[name] = swarm_agents
        return swarm_agents
    
    async def run_swarm_analysis(self, swarm_name: str, content: str):
        """Run analysis using a swarm of agents."""
        if swarm_name not in self.swarms:
            raise ValueError(f"Swarm '{swarm_name}' not found")
        
        results = {}
        for agent in self.swarms[swarm_name]:
            result = await agent.run(content)
            results[agent.name] = result
        
        return {
            "swarm_name": swarm_name,
            "results": results,
            "agents_used": len(self.swarms[swarm_name])
        }
    
    async def coordinate_agents(self, query: str):
        """Coordinate multiple agents for complex queries."""
        # Implement coordination logic
        return await self.run_swarm_analysis("default", query)
```

### **Step 5: Implement MCP Server**
```python
# src/core/mcp_server.py
import warnings
from typing import List, Dict, Any
from fastmcp import FastMCP
from .agents.sentiment_agent import SentimentAgent
from .agents.vision_agent import VisionAgent
from .orchestrator import Orchestrator

# Suppress websockets warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets")

class OptimizedMCPServer:
    def __init__(self):
        self.mcp = FastMCP("Your Analysis Server")
        self.agents = {}
        self.orchestrator = None
        self._initialize_agents()
        self._initialize_orchestrator()
        self._register_tools()
    
    def _initialize_agents(self):
        """Initialize all agents."""
        self.agents["sentiment"] = SentimentAgent()
        self.agents["vision"] = VisionAgent()
        # Add more agents as needed
    
    def _initialize_orchestrator(self):
        """Initialize the orchestrator."""
        self.orchestrator = Orchestrator(self.agents)
        # Create default swarms
        self.orchestrator.create_swarm("analysis", ["sentiment", "vision"])
    
    def _register_tools(self):
        """Register MCP tools."""
        
        # Core Management Tools
        @self.mcp.tool(description="Get status of all available agents")
        async def get_all_agents_status():
            status = {}
            for name, agent in self.agents.items():
                status[name] = agent.get_status()
            return {
                "success": True,
                "total_agents": len(self.agents),
                "agents": status
            }
        
        @self.mcp.tool(description="Start all agents")
        async def start_all_agents():
            results = {}
            for name, agent in self.agents.items():
                try:
                    await agent.start()
                    results[name] = {"success": True, "message": "Started"}
                except Exception as e:
                    results[name] = {"success": False, "error": str(e)}
            return {"success": True, "results": results}
        
        @self.mcp.tool(description="Stop all agents")
        async def stop_all_agents():
            results = {}
            for name, agent in self.agents.items():
                try:
                    await agent.stop()
                    results[name] = {"success": True, "message": "Stopped"}
                except Exception as e:
                    results[name] = {"success": False, "error": str(e)}
            return {"success": True, "results": results}
        
        # Analysis Tools
        @self.mcp.tool(description="Analyze text sentiment")
        async def analyze_text_sentiment(text: str, language: str = "en"):
            try:
                agent = self.agents["sentiment"]
                result = await agent.analyze_sentiment(text)
                return {
                    "success": True,
                    "agent_used": "sentiment_analyzer",
                    "text": text,
                    "language": language,
                    "result": result
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.mcp.tool(description="Analyze image content")
        async def analyze_image_sentiment(image_path: str):
            try:
                agent = self.agents["vision"]
                result = await agent.analyze_image(image_path)
                return {
                    "success": True,
                    "agent_used": "vision_analyzer",
                    "image_path": image_path,
                    "result": result
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        # Orchestrator Tools
        @self.mcp.tool(description="Process query using orchestrator")
        async def process_query_orchestrator(query: str):
            try:
                result = await self.orchestrator.coordinate_agents(query)
                return {
                    "success": True,
                    "agent_type": "orchestrator",
                    "query": query,
                    "result": result
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.mcp.tool(description="Run swarm analysis")
        async def run_swarm_analysis(swarm_name: str, content: str):
            try:
                result = await self.orchestrator.run_swarm_analysis(swarm_name, content)
                return {
                    "success": True,
                    "swarm_name": swarm_name,
                    "content": content,
                    "result": result
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
    
    def run(self, host: str = "localhost", port: int = 8000):
        """Run the MCP server."""
        return self.mcp.run(transport="streamable-http")
```

### **Step 6: Create Main Entry Point**
```python
# main.py
#!/usr/bin/env python3
"""
Main entry point for the MCP server.
"""

import warnings
import threading
import uvicorn

# Suppress websockets warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets")

from src.core.mcp_server import OptimizedMCPServer

def start_mcp_server():
    """Start the MCP server."""
    try:
        mcp_server = OptimizedMCPServer()
        
        def run_mcp_server():
            try:
                mcp_server.run(host="localhost", port=8000)
            except Exception as e:
                print(f"❌ Error starting MCP server: {e}")
        
        mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
        mcp_thread.start()
        
        print("✅ MCP server started successfully")
        print(" - MCP Server: http://localhost:8000/mcp")
        
        return mcp_server
        
    except Exception as e:
        print(f"⚠️ Warning: Could not start MCP server: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Starting MCP Server")
    print("=" * 50)
    
    # Start MCP server
    mcp_server = start_mcp_server()
    
    # Keep the server running
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down MCP server...")
```

## 🧪 **Testing Your Implementation**

### **1. Test MCP Client**
```python
# test_mcp_client.py
import asyncio
from mcp.client.streamable_http import streamablehttp_client

async def test_mcp_server():
    client = streamablehttp_client("http://localhost:8000/mcp")
    
    async with client:
        # List tools
        tools = await client.list_tools()
        print(f"Available tools: {len(tools)}")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        # Test tool call
        result = await client.call_tool("get_all_agents_status", {})
        print(f"Agent status: {result}")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())
```

### **2. Test Individual Tools**
```python
# test_tools.py
import asyncio
from mcp.client.streamable_http import streamablehttp_client

async def test_tools():
    client = streamablehttp_client("http://localhost:8000/mcp")
    
    async with client:
        # Test sentiment analysis
        result = await client.call_tool("analyze_text_sentiment", {
            "text": "I love this product!",
            "language": "en"
        })
        print(f"Sentiment result: {result}")
        
        # Test orchestrator
        result = await client.call_tool("process_query_orchestrator", {
            "query": "Analyze this content comprehensively"
        })
        print(f"Orchestrator result: {result}")

if __name__ == "__main__":
    asyncio.run(test_tools())
```

## 📋 **Best Practices**

### **1. Error Handling**
```python
@self.mcp.tool(description="Your tool description")
async def your_tool(param: str):
    try:
        # Your tool logic here
        result = await some_async_operation(param)
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

### **2. Configuration Management**
```python
# src/config/settings.py
from pydantic import BaseModel
from pathlib import Path

class Settings(BaseModel):
    mcp_host: str = "localhost"
    mcp_port: int = 8000
    default_language: str = "en"
    max_agents: int = 10

settings = Settings()
```

### **3. Logging**
```python
from loguru import logger

logger.info("✅ MCP server initialized")
logger.error(f"❌ Error: {e}")
logger.success("🎉 Operation completed successfully")
```

### **4. Agent Lifecycle Management**
```python
class AgentManager:
    def __init__(self):
        self.agents = {}
        self.active_agents = set()
    
    async def start_agent(self, agent_name: str):
        if agent_name in self.agents:
            await self.agents[agent_name].start()
            self.active_agents.add(agent_name)
    
    async def stop_agent(self, agent_name: str):
        if agent_name in self.agents:
            await self.agents[agent_name].stop()
            self.active_agents.discard(agent_name)
```

## 🔧 **Advanced Features**

### **1. Dynamic Tool Registration**
```python
def register_dynamic_tool(self, tool_name: str, tool_func, description: str):
    """Register a tool dynamically."""
    @self.mcp.tool(description=description)
    async def dynamic_tool(*args, **kwargs):
        return await tool_func(*args, **kwargs)
```

### **2. Swarm Coordination Patterns**
```python
class SwarmCoordinator:
    async def sequential_processing(self, agents: List, content: str):
        """Process content sequentially through agents."""
        results = []
        for agent in agents:
            result = await agent.run(content)
            results.append(result)
        return results
    
    async def parallel_processing(self, agents: List, content: str):
        """Process content in parallel through agents."""
        tasks = [agent.run(content) for agent in agents]
        results = await asyncio.gather(*tasks)
        return results
```

### **3. Tool Chaining**
```python
async def chain_tools(self, tool_sequence: List[str], initial_data: Any):
    """Chain multiple tools together."""
    data = initial_data
    for tool_name in tool_sequence:
        result = await self.call_tool(tool_name, data)
        data = result
    return data
```

## 🎯 **Common Patterns**

### **1. Agent Factory Pattern**
```python
class AgentFactory:
    @staticmethod
    def create_agent(agent_type: str, **kwargs):
        if agent_type == "sentiment":
            return SentimentAgent(**kwargs)
        elif agent_type == "vision":
            return VisionAgent(**kwargs)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
```

### **2. Tool Registry Pattern**
```python
class ToolRegistry:
    def __init__(self):
        self.tools = {}
    
    def register(self, name: str, func, description: str):
        self.tools[name] = {
            "func": func,
            "description": description
        }
    
    def get_tool(self, name: str):
        return self.tools.get(name)
```

## 🚀 **Deployment**

### **1. Production Configuration**
```python
# production_config.py
class ProductionConfig:
    MCP_HOST = "0.0.0.0"  # Allow external connections
    MCP_PORT = 8000
    LOG_LEVEL = "INFO"
    MAX_CONNECTIONS = 100
    TIMEOUT = 30
```

### **2. Docker Deployment**
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

## 📚 **Resources**

- [FastMCP Documentation](https://github.com/fastmcp/fastmcp)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [Sentiment Swarm Agents Reference](https://raw.githubusercontent.com/goagiq/sentiment-swarm-agents/refs/heads/master/main.py)

## 🎉 **Summary**

This guide provides a complete framework for implementing MCP servers with FastMCP, including:

1. ✅ **Agent Management** - Create and manage specialized agents
2. ✅ **Swarm Coordination** - Coordinate multiple agents for complex tasks
3. ✅ **Orchestrator Pattern** - Centralized coordination and workflow management
4. ✅ **Tool Registration** - Register and expose tools via MCP protocol
5. ✅ **Error Handling** - Robust error handling and logging
6. ✅ **Testing** - Comprehensive testing strategies
7. ✅ **Best Practices** - Production-ready patterns and practices

Follow this guide to create scalable, maintainable MCP servers with swarm agent capabilities!
