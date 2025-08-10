"""
MCP server for TextAgentSwarm - provides coordinated text sentiment analysis using swarm.
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
from agents.text_agent_swarm import TextAgentSwarm


class TextSwarmAnalysisRequest(BaseModel):
    """Request model for text swarm analysis."""
    text: str = Field(..., description="Text content to analyze")
    content_type: str = Field(
        default="text", 
        description="Type of content: text, social_media"
    )
    language: str = Field(default="en", description="Language code")
    confidence_threshold: float = Field(
        default=0.8, 
        description="Minimum confidence threshold"
    )


class TextSwarmAnalysisResponse(BaseModel):
    """Response model for text swarm analysis."""
    text: str = Field(..., description="Analyzed text content")
    content_type: str = Field(..., description="Type of content analyzed")
    sentiment: str = Field(..., description="Detected sentiment: positive, negative, neutral")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    scores: Dict[str, float] = Field(..., description="Detailed sentiment scores")
    features: Optional[Dict[str, Any]] = Field(None, description="Extracted text features")
    analysis_time: float = Field(..., description="Analysis time in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata"
    )


class TextAgentSwarmMCPServer:
    """MCP server providing text swarm analysis tools from TextAgentSwarm."""
    
    def __init__(self, agent_count: int = 3, model_name: Optional[str] = None):
        # Initialize the text agent swarm
        self.text_agent_swarm = TextAgentSwarm(
            agent_count=agent_count,
            model_name=model_name
        )
        
        # Note: FastMCP will be imported when available
        # For now, we'll create a mock structure
        self.mcp = None
        self._initialize_mcp()
        
        # Register tools
        self._register_tools()
        
        model_name = self.text_agent_swarm.metadata.get('model', 'default')
        logger.info(f"TextAgentSwarm MCP Server initialized with model: {model_name}")
    
    def _initialize_mcp(self):
        """Initialize the MCP server."""
        try:
            from fastmcp import FastMCP
            self.mcp = FastMCP("TextAgentSwarm Server")
            logger.info("FastMCP initialized successfully")
        except ImportError:
            logger.warning("FastMCP not available, using mock MCP server")
            # Create a mock MCP server for development
            self.mcp = MockMCPServer("TextAgentSwarm Server")
    
    def _register_tools(self):
        """Register all text swarm analysis tools from TextAgentSwarm."""
        
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        @self.mcp.tool(
            description="Analyze text sentiment using TextAgentSwarm"
        )
        async def analyze_text_sentiment_swarm(
            text: str = Field(..., description="Text content to analyze"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Analyze the sentiment of text content using TextAgentSwarm."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the swarm's coordinate method for sentiment analysis
                sentiment_result = await self.text_agent_swarm.coordinate_sentiment_analysis(text)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if sentiment_result.get("status") == "success":
                    best_result = sentiment_result.get("content", [{}])[0].get("json", {})
                    return {
                        "text": text,
                        "content_type": "text",
                        "sentiment": best_result.get("sentiment", "neutral"),
                        "confidence": best_result.get("confidence", 0.0),
                        "scores": best_result.get("scores", {}),
                        "features": best_result.get("features"),
                        "analysis_time": analysis_time,
                        "metadata": {
                            "method": "swarm_coordination",
                            "agents_used": best_result.get("agents_used", 0),
                            "total_agents": best_result.get("total_agents", 0),
                            "coordination_success": best_result.get("coordination_success", False)
                        }
                    }
                else:
                    return {
                        "text": text,
                        "content_type": "text",
                        "sentiment": "neutral",
                        "confidence": 0.0,
                        "scores": {"neutral": 1.0},
                        "features": None,
                        "analysis_time": analysis_time,
                        "metadata": {
                            "error": sentiment_result.get("content", [{}])[0].get("text", "Unknown error"),
                            "method": "swarm_coordination"
                        }
                    }
                    
            except Exception as e:
                logger.error(f"TextAgentSwarm sentiment analysis failed: {e}")
                return {
                    "text": text,
                    "content_type": "text",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {"neutral": 1.0},
                    "features": None,
                    "analysis_time": 0.0,
                    "metadata": {
                        "error": str(e),
                        "method": "swarm_coordination"
                    }
                }
        
        @self.mcp.tool(
            description="Coordinate sentiment analysis across swarm agents"
        )
        async def coordinate_sentiment_analysis_swarm(
            text: str = Field(..., description="Text content to analyze"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Coordinate sentiment analysis across multiple text agents in the swarm."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the swarm's coordinate method
                result = await self.text_agent_swarm.coordinate_sentiment_analysis(text)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                return {
                    "status": "success",
                    "result": result,
                    "analysis_time": analysis_time,
                    "method": "swarm_coordination"
                }
                
            except Exception as e:
                logger.error(f"Swarm coordination failed: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "method": "swarm_coordination"
                }
        
        @self.mcp.tool(
            description="Analyze text using the entire swarm"
        )
        async def analyze_text_with_swarm_tool(
            text: str = Field(..., description="Text content to analyze"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Analyze text using the entire swarm of agents."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the swarm's analyze method
                result = await self.text_agent_swarm.analyze_text_with_swarm(text)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                return {
                    "status": "success",
                    "result": result,
                    "analysis_time": analysis_time,
                    "method": "swarm_analysis"
                }
                
            except Exception as e:
                logger.error(f"Swarm analysis failed: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "method": "swarm_analysis"
                }
        
        @self.mcp.tool(
            description="Get status of all agents in the swarm"
        )
        async def get_swarm_status_tool() -> Dict[str, Any]:
            """Get the current status of all agents in the swarm."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the swarm's status method
                status_result = await self.text_agent_swarm.get_swarm_status()
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                return {
                    "status": "success",
                    "result": status_result,
                    "analysis_time": analysis_time,
                    "method": "swarm_status"
                }
                
            except Exception as e:
                logger.error(f"Failed to get swarm status: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "method": "swarm_status"
                }
        
        @self.mcp.tool(
            description="Distribute multiple texts across the swarm"
        )
        async def distribute_workload_swarm(
            texts: List[str] = Field(..., description="List of text items to analyze"),
            language: str = Field(default="en", description="Language code")
        ) -> List[Dict[str, Any]]:
            """Distribute multiple text items across the swarm for parallel processing."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the swarm's workload distribution method
                distribution_result = await self.text_agent_swarm.distribute_workload(texts)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if distribution_result.get("status") == "success":
                    results = distribution_result.get("content", [{}])[0].get("json", {}).get("results", [])
                    return [
                        {
                            "text_index": result.get("text_index"),
                            "agent_id": result.get("agent_id"),
                            "result": result.get("result"),
                            "error": result.get("error"),
                            "analysis_time": analysis_time,
                            "method": "workload_distribution"
                        }
                        for result in results
                    ]
                else:
                    return [{
                        "error": distribution_result.get("content", [{}])[0].get("text", "Distribution failed"),
                        "analysis_time": analysis_time,
                        "method": "workload_distribution"
                    }]
                    
            except Exception as e:
                logger.error(f"Workload distribution failed: {e}")
                return [{
                    "error": str(e),
                    "analysis_time": 0.0,
                    "method": "workload_distribution"
                }]
        
        @self.mcp.tool(
            description="Comprehensive text analysis using swarm"
        )
        async def comprehensive_text_analysis_swarm(
            text: str = Field(..., description="Text content to analyze"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Perform comprehensive text analysis using the entire swarm."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the swarm's coordinate method for comprehensive analysis
                sentiment_result = await self.text_agent_swarm.coordinate_sentiment_analysis(text)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if sentiment_result.get("status") == "success":
                    best_result = sentiment_result.get("content", [{}])[0].get("json", {})
                    
                    # Check confidence threshold
                    confidence = best_result.get("confidence", 0.0)
                    if confidence < confidence_threshold:
                        logger.warning(f"Confidence {confidence} below threshold {confidence_threshold}")
                    
                    return {
                        "text": text,
                        "content_type": "text",
                        "sentiment": best_result.get("sentiment", "neutral"),
                        "confidence": confidence,
                        "scores": best_result.get("scores", {}),
                        "features": best_result.get("features"),
                        "analysis_time": analysis_time,
                        "confidence_threshold_met": confidence >= confidence_threshold,
                        "metadata": {
                            "method": "swarm_comprehensive",
                            "agents_used": best_result.get("agents_used", 0),
                            "total_agents": best_result.get("total_agents", 0),
                            "coordination_success": best_result.get("coordination_success", False),
                            "confidence_threshold": confidence_threshold
                        }
                    }
                else:
                    return {
                        "text": text,
                        "content_type": "text",
                        "sentiment": "neutral",
                        "confidence": 0.0,
                        "scores": {"neutral": 1.0},
                        "features": None,
                        "analysis_time": analysis_time,
                        "confidence_threshold_met": False,
                        "metadata": {
                            "error": sentiment_result.get("content", [{}])[0].get("text", "Unknown error"),
                            "method": "swarm_comprehensive",
                            "confidence_threshold": confidence_threshold
                        }
                    }
                    
            except Exception as e:
                logger.error(f"Comprehensive swarm analysis failed: {e}")
                return {
                    "text": text,
                    "content_type": "text",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {"neutral": 1.0},
                    "features": None,
                    "analysis_time": 0.0,
                    "confidence_threshold_met": False,
                    "metadata": {
                        "error": str(e),
                        "method": "swarm_comprehensive",
                        "confidence_threshold": confidence_threshold
                    }
                }
        
        @self.mcp.tool(
            description="Check if TextAgentSwarm can process specific content"
        )
        async def can_process_text_swarm(
            text: str = Field(..., description="Text content to check"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Check if the TextAgentSwarm can process the given text content."""
            try:
                # Create a mock request to test processing capability
                mock_request = AnalysisRequest(
                    content=text,
                    data_type=DataType.TEXT,
                    language=language
                )
                
                can_process = await self.text_agent_swarm.can_process(mock_request)
                
                return {
                    "text": text,
                    "language": language,
                    "can_process": can_process,
                    "method": "swarm_capability_check",
                    "metadata": {
                        "swarm_size": self.text_agent_swarm.agent_count,
                        "total_agents": len(self.text_agent_swarm.text_agents)
                    }
                }
                
            except Exception as e:
                logger.error(f"Capability check failed: {e}")
                return {
                    "text": text,
                    "language": language,
                    "can_process": False,
                    "error": str(e),
                    "method": "swarm_capability_check"
                }
        
        @self.mcp.tool(
            description="Get TextAgentSwarm capabilities and configuration"
        )
        def get_text_agent_swarm_capabilities() -> Dict[str, Any]:
            """Get the capabilities and configuration of the TextAgentSwarm."""
            try:
                swarm_status = self.text_agent_swarm.get_status()
                
                return {
                    "agent_type": "TextAgentSwarm",
                    "agent_id": swarm_status.get("agent_id"),
                    "status": swarm_status.get("status"),
                    "capabilities": {
                        "text_analysis": True,
                        "sentiment_analysis": True,
                        "swarm_coordination": True,
                        "workload_distribution": True,
                        "parallel_processing": True
                    },
                    "configuration": {
                        "agent_count": self.text_agent_swarm.agent_count,
                        "model_name": self.text_agent_swarm.metadata.get("model", "default"),
                        "strands_swarm_active": True
                    },
                    "metadata": swarm_status.get("metadata", {}),
                    "method": "swarm_capabilities"
                }
                
            except Exception as e:
                logger.error(f"Failed to get capabilities: {e}")
                return {
                    "agent_type": "TextAgentSwarm",
                    "error": str(e),
                    "method": "swarm_capabilities"
                }
    
    def run(self, host: str = "localhost", port: int = 8007, debug: bool = False):
        """Run the MCP server."""
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        try:
            logger.info(f"Starting TextAgentSwarm MCP Server on {host}:{port}")
            self.mcp.run(host=host, port=port, debug=debug)
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")


class MockMCPServer:
    """Mock MCP server for development when FastMCP is not available."""
    
    def __init__(self, name: str):
        self.name = name
        self.tools = []
    
    def tool(self, description: str):
        def decorator(func):
            self.tools.append({"func": func, "description": description})
            return func
        return decorator
    
    def run(self, host: str = "localhost", port: int = 8007, debug: bool = False):
        logger.info(f"Mock MCP Server '{self.name}' would run on {host}:{port}")
        logger.info(f"Registered {len(self.tools)} tools")


def create_text_agent_swarm_mcp_server(agent_count: int = 3, model_name: Optional[str] = None) -> TextAgentSwarmMCPServer:
    """Factory function to create a TextAgentSwarm MCP server."""
    return TextAgentSwarmMCPServer(agent_count=agent_count, model_name=model_name)


if __name__ == "__main__":
    # Create and run the server
    server = create_text_agent_swarm_mcp_server()
    server.run()
