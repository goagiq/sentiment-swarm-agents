"""
MCP server for OrchestratorAgent - provides orchestration and coordination of multiple sentiment analysis agents.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import asyncio
from loguru import logger

# Import the correct models and agents
from core.models import (
    AnalysisRequest, 
    DataType
)
from agents.orchestrator_agent import OrchestratorAgent


class OrchestratorAnalysisRequest(BaseModel):
    """Request model for orchestrator analysis."""
    content: str = Field(..., description="Content to analyze (text, image path, audio path, URL)")
    data_type: str = Field(
        default="auto", 
        description="Type of content: auto, text, image, audio, webpage"
    )
    language: str = Field(default="en", description="Language code")
    confidence_threshold: float = Field(
        default=0.8, 
        description="Minimum confidence threshold"
    )


class OrchestratorAnalysisResponse(BaseModel):
    """Response model for orchestrator analysis."""
    content: str = Field(..., description="Analyzed content")
    data_type: str = Field(..., description="Type of content analyzed")
    sentiment: str = Field(..., description="Detected sentiment: positive, negative, neutral")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    method: str = Field(..., description="Method used for analysis")
    agent_id: str = Field(..., description="ID of the agent that performed analysis")
    analysis_time: float = Field(..., description="Analysis time in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata"
    )


class OrchestratorAgentMCPServer:
    """MCP server providing orchestration tools from OrchestratorAgent."""
    
    def __init__(self, model_name: Optional[str] = None):
        # Initialize the orchestrator agent
        self.orchestrator_agent = OrchestratorAgent(model_name=model_name)
        
        # Note: FastMCP will be imported when available
        # For now, we'll create a mock structure
        self.mcp = None
        self._initialize_mcp()
        
        # Register tools
        self._register_tools()
        
        model_name = self.orchestrator_agent.model_name
        logger.info(f"OrchestratorAgent MCP Server initialized with model: {model_name}")
    
    def _initialize_mcp(self):
        """Initialize the MCP server."""
        try:
            from fastmcp import FastMCP
            self.mcp = FastMCP("OrchestratorAgent Server")
            logger.info("FastMCP initialized successfully")
        except ImportError:
            logger.warning("FastMCP not available, using mock MCP server")
            # Create a mock MCP server for development
            self.mcp = MockMCPServer("OrchestratorAgent Server")
    
    def _register_tools(self):
        """Register all orchestration tools from OrchestratorAgent."""
        
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        @self.mcp.tool(
            description="Handle text-based sentiment analysis queries"
        )
        async def text_sentiment_analysis(
            query: str = Field(..., description="Text content to analyze for sentiment")
        ) -> Dict[str, Any]:
            """Handle text-based sentiment analysis queries."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Create specialized text agent
                from agents.text_agent import TextAgent
                text_agent = TextAgent()
                
                # Create analysis request
                request = AnalysisRequest(
                    data_type=DataType.TEXT,
                    content=query,
                    language="en"
                )
                
                # Process the request
                result = await text_agent.process(request)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                return {
                    "status": "success",
                    "content": query,
                    "data_type": "text",
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "method": "text_agent",
                    "agent_id": text_agent.agent_id,
                    "analysis_time": analysis_time,
                    "metadata": {
                        "method": "text_agent",
                        "agent_id": text_agent.agent_id,
                        "framework": "orchestrator"
                    }
                }
                
            except Exception as e:
                logger.error(f"Text sentiment analysis failed: {e}")
                return {
                    "status": "error",
                    "content": query,
                    "data_type": "text",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "method": "text_agent",
                    "agent_id": "unknown",
                    "analysis_time": 0.0,
                    "metadata": {
                        "error": str(e),
                        "method": "text_agent",
                        "framework": "orchestrator"
                    }
                }
        
        @self.mcp.tool(
            description="Handle image and video sentiment analysis queries"
        )
        async def vision_sentiment_analysis(
            image_path: str = Field(..., description="Path to image or video file")
        ) -> Dict[str, Any]:
            """Handle image and video sentiment analysis queries."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Create specialized vision agent
                from agents.vision_agent_enhanced import EnhancedVisionAgent
                vision_agent = EnhancedVisionAgent()
                
                # Create analysis request
                request = AnalysisRequest(
                    data_type=DataType.IMAGE,
                    content=image_path
                )
                
                # Process the request
                result = await vision_agent.process(request)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                return {
                    "status": "success",
                    "content": image_path,
                    "data_type": "image",
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "method": "vision_agent",
                    "agent_id": vision_agent.agent_id,
                    "analysis_time": analysis_time,
                    "metadata": {
                        "method": "vision_agent",
                        "agent_id": vision_agent.agent_id,
                        "framework": "orchestrator"
                    }
                }
                
            except Exception as e:
                logger.error(f"Vision sentiment analysis failed: {e}")
                return {
                    "status": "error",
                    "content": image_path,
                    "data_type": "image",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "method": "vision_agent",
                    "agent_id": "unknown",
                    "analysis_time": 0.0,
                    "metadata": {
                        "error": str(e),
                        "method": "vision_agent",
                        "framework": "orchestrator"
                    }
                }
        
        @self.mcp.tool(
            description="Handle audio sentiment analysis queries"
        )
        async def audio_sentiment_analysis(
            audio_path: str = Field(..., description="Path to audio file")
        ) -> Dict[str, Any]:
            """Handle audio sentiment analysis queries."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Create specialized audio agent
                from agents.audio_agent_enhanced import EnhancedAudioAgent
                audio_agent = EnhancedAudioAgent()
                
                # Create analysis request
                request = AnalysisRequest(
                    data_type=DataType.AUDIO,
                    content=audio_path
                )
                
                # Process the request
                result = await audio_agent.process(request)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                return {
                    "status": "success",
                    "content": audio_path,
                    "data_type": "audio",
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "method": "audio_agent",
                    "agent_id": audio_agent.agent_id,
                    "analysis_time": analysis_time,
                    "metadata": {
                        "method": "audio_agent",
                        "agent_id": audio_agent.agent_id,
                        "framework": "orchestrator"
                    }
                }
                
            except Exception as e:
                logger.error(f"Audio sentiment analysis failed: {e}")
                return {
                    "status": "error",
                    "content": audio_path,
                    "data_type": "audio",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "method": "audio_agent",
                    "agent_id": "unknown",
                    "analysis_time": 0.0,
                    "metadata": {
                        "error": str(e),
                        "method": "audio_agent",
                        "framework": "orchestrator"
                    }
                }
        
        @self.mcp.tool(
            description="Handle webpage sentiment analysis queries"
        )
        async def web_sentiment_analysis(
            url: str = Field(..., description="URL of webpage to analyze")
        ) -> Dict[str, Any]:
            """Handle webpage sentiment analysis queries."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Create specialized web agent
                from agents.web_agent_enhanced import EnhancedWebAgent
                web_agent = EnhancedWebAgent()
                
                # Create analysis request
                request = AnalysisRequest(
                    data_type=DataType.WEBPAGE,
                    content=url
                )
                
                # Process the request
                result = await web_agent.process(request)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                return {
                    "status": "success",
                    "content": url,
                    "data_type": "webpage",
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "method": "web_agent",
                    "agent_id": web_agent.agent_id,
                    "analysis_time": analysis_time,
                    "metadata": {
                        "method": "web_agent",
                        "agent_id": web_agent.agent_id,
                        "framework": "orchestrator"
                    }
                }
                
            except Exception as e:
                logger.error(f"Web sentiment analysis failed: {e}")
                return {
                    "status": "error",
                    "content": url,
                    "data_type": "webpage",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "method": "web_agent",
                    "agent_id": "unknown",
                    "analysis_time": 0.0,
                    "metadata": {
                        "error": str(e),
                        "method": "web_agent",
                        "framework": "orchestrator"
                    }
                }
        
        @self.mcp.tool(
            description="Handle complex text analysis using coordinated swarm of agents"
        )
        async def swarm_text_analysis(
            text: str = Field(..., description="Text content to analyze with swarm coordination")
        ) -> Dict[str, Any]:
            """Handle complex text analysis using coordinated swarm of agents."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Create specialized swarm agent
                from agents.text_agent_swarm import TextAgentSwarm
                swarm_agent = TextAgentSwarm(agent_count=3)
                
                # Create analysis request
                request = AnalysisRequest(
                    data_type=DataType.TEXT,
                    content=text,
                    language="en"
                )
                
                # Process the request
                result = await swarm_agent.process(request)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                return {
                    "status": "success",
                    "content": text,
                    "data_type": "text",
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "method": "swarm_coordination",
                    "agent_id": swarm_agent.agent_id,
                    "analysis_time": analysis_time,
                    "metadata": {
                        "method": "swarm_coordination",
                        "agent_id": swarm_agent.agent_id,
                        "swarm_size": swarm_agent.agent_count,
                        "framework": "orchestrator"
                    }
                }
                
            except Exception as e:
                logger.error(f"Swarm text analysis failed: {e}")
                return {
                    "status": "error",
                    "content": text,
                    "data_type": "text",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "method": "swarm_coordination",
                    "agent_id": "unknown",
                    "analysis_time": 0.0,
                    "metadata": {
                        "error": str(e),
                        "method": "swarm_coordination",
                        "framework": "orchestrator"
                    }
                }
        
        @self.mcp.tool(
            description="Process a natural language query and route to appropriate tools"
        )
        async def process_query(
            query: str = Field(..., description="Natural language query to process and route")
        ) -> Dict[str, Any]:
            """Process a natural language query and route to appropriate tools."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the orchestrator's query processing logic
                result = await self.orchestrator_agent.process_query(query)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                # Parse the result to extract sentiment information
                if isinstance(result, dict) and result.get("status") == "success":
                    content = result.get("content", [])
                    if content and "json" in content[0]:
                        json_data = content[0]["json"]
                        return {
                            "status": "success",
                            "content": query,
                            "data_type": "auto",
                            "sentiment": json_data.get("sentiment", "neutral"),
                            "confidence": json_data.get("confidence", 0.5),
                            "method": json_data.get("method", "orchestrator"),
                            "agent_id": json_data.get("agent_id", "unknown"),
                            "analysis_time": analysis_time,
                            "metadata": {
                                "method": json_data.get("method", "orchestrator"),
                                "agent_id": json_data.get("agent_id", "unknown"),
                                "framework": "orchestrator"
                            }
                        }
                
                # Fallback for text-based queries
                return await text_sentiment_analysis(query)
                
            except Exception as e:
                logger.error(f"Query processing failed: {e}")
                return {
                    "status": "error",
                    "content": query,
                    "data_type": "auto",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "method": "orchestrator",
                    "agent_id": "unknown",
                    "analysis_time": 0.0,
                    "metadata": {
                        "error": str(e),
                        "method": "orchestrator",
                        "framework": "orchestrator"
                    }
                }
        
        @self.mcp.tool(
            description="Get list of available tools with metadata"
        )
        async def get_available_tools() -> Dict[str, Any]:
            """Get list of available tools with metadata."""
            try:
                tools = await self.orchestrator_agent.get_available_tools()
                return {
                    "status": "success",
                    "tools_count": len(tools),
                    "tools": [
                        {
                            "name": tool["name"],
                            "description": tool["description"]
                        }
                        for tool in tools
                    ],
                    "metadata": {
                        "framework": "orchestrator",
                        "agent_id": self.orchestrator_agent.agent_id
                    }
                }
            except Exception as e:
                logger.error(f"Error getting available tools: {e}")
                return {
                    "status": "error",
                    "tools_count": 0,
                    "tools": [],
                    "metadata": {
                        "error": str(e),
                        "framework": "orchestrator"
                    }
                }
        
        @self.mcp.tool(
            description="Get current status of the orchestrator"
        )
        def get_orchestrator_status() -> Dict[str, Any]:
            """Get current status of the orchestrator."""
            try:
                status = self.orchestrator_agent.get_status()
                return {
                    "status": "success",
                    "orchestrator_status": status,
                    "metadata": {
                        "framework": "orchestrator"
                    }
                }
            except Exception as e:
                logger.error(f"Error getting orchestrator status: {e}")
                return {
                    "status": "error",
                    "orchestrator_status": {},
                    "metadata": {
                        "error": str(e),
                        "framework": "orchestrator"
                    }
                }
        
        @self.mcp.tool(
            description="Start the orchestrator agent"
        )
        async def start_orchestrator() -> Dict[str, Any]:
            """Start the orchestrator agent."""
            try:
                result = await self.orchestrator_agent.start()
                return {
                    "status": "success",
                    "started": result,
                    "message": "Orchestrator agent started successfully",
                    "metadata": {
                        "framework": "orchestrator",
                        "agent_id": self.orchestrator_agent.agent_id
                    }
                }
            except Exception as e:
                logger.error(f"Error starting orchestrator: {e}")
                return {
                    "status": "error",
                    "started": False,
                    "message": f"Failed to start orchestrator: {str(e)}",
                    "metadata": {
                        "error": str(e),
                        "framework": "orchestrator"
                    }
                }
        
        @self.mcp.tool(
            description="Stop the orchestrator agent"
        )
        async def stop_orchestrator() -> Dict[str, Any]:
            """Stop the orchestrator agent."""
            try:
                result = await self.orchestrator_agent.stop()
                return {
                    "status": "success",
                    "stopped": result,
                    "message": "Orchestrator agent stopped successfully",
                    "metadata": {
                        "framework": "orchestrator",
                        "agent_id": self.orchestrator_agent.agent_id
                    }
                }
            except Exception as e:
                logger.error(f"Error stopping orchestrator: {e}")
                return {
                    "status": "error",
                    "stopped": False,
                    "message": f"Failed to stop orchestrator: {str(e)}",
                    "metadata": {
                        "error": str(e),
                        "framework": "orchestrator"
                    }
                }
        
        @self.mcp.tool(
            description="Get OrchestratorAgent capabilities and configuration"
        )
        def get_orchestrator_capabilities() -> Dict[str, Any]:
            """Get information about OrchestratorAgent capabilities and configuration."""
            return {
                "agent_id": self.orchestrator_agent.agent_id,
                "model": self.orchestrator_agent.model_name,
                "supported_data_types": ["text", "image", "audio", "webpage"],
                "capabilities": [
                    "text_sentiment_analysis",
                    "vision_sentiment_analysis", 
                    "audio_sentiment_analysis",
                    "web_sentiment_analysis",
                    "swarm_text_analysis",
                    "query_routing",
                    "agent_coordination"
                ],
                "available_tools": [
                    "text_sentiment_analysis",
                    "vision_sentiment_analysis",
                    "audio_sentiment_analysis", 
                    "web_sentiment_analysis",
                    "swarm_text_analysis",
                    "process_query",
                    "get_available_tools",
                    "get_orchestrator_status",
                    "start_orchestrator",
                    "stop_orchestrator"
                ],
                "features": [
                    "multi-modal sentiment analysis",
                    "intelligent query routing",
                    "agent coordination",
                    "swarm intelligence",
                    "unified interface",
                    "Ollama integration"
                ],
                "framework": "orchestrator"
            }
    
    def run(self, host: str = "localhost", port: int = 8007, debug: bool = False):
        """Run the OrchestratorAgent MCP server."""
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        logger.info(f"Starting OrchestratorAgent MCP server on {host}:{port}")
        
        if hasattr(self.mcp, 'run'):
            # Real FastMCP server
            self.mcp.run(
                transport="streamable-http",
                host=host,
                port=port,
                debug=debug
            )
        else:
            # Mock server
            logger.info("Running mock MCP server - install FastMCP for full functionality")
            self.mcp.run(host=host, port=port, debug=debug)


class MockMCPServer:
    """Mock MCP server for development when FastMCP is not available."""
    
    def __init__(self, name: str):
        self.name = name
        self.tools = {}
    
    def tool(self, description: str):
        """Decorator to register tools."""
        def decorator(func):
            self.tools[func.__name__] = {
                "function": func,
                "description": description
            }
            return func
        return decorator
    
    def run(self, host: str = "localhost", port: int = 8007, debug: bool = False):
        """Mock run method."""
        logger.info(f"Mock MCP server '{self.name}' would run on {host}:{port}")
        logger.info(f"Available tools: {list(self.tools.keys())}")


def create_orchestrator_agent_mcp_server(model_name: Optional[str] = None) -> OrchestratorAgentMCPServer:
    """Factory function to create an OrchestratorAgent MCP server."""
    return OrchestratorAgentMCPServer(model_name=model_name)


if __name__ == "__main__":
    # Run the server directly
    server = create_orchestrator_agent_mcp_server()
    server.run(host="0.0.0.0", port=8007, debug=True)
