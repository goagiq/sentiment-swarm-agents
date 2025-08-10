"""
MCP server for SimpleTextAgent - provides simplified text sentiment analysis tools.
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
from agents.text_agent_simple import SimpleTextAgent


class SimpleTextAnalysisRequest(BaseModel):
    """Request model for simple text analysis."""
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


class SimpleTextAnalysisResponse(BaseModel):
    """Response model for simple text analysis."""
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


class SimpleTextAgentMCPServer:
    """MCP server providing simple text analysis tools from SimpleTextAgent."""
    
    def __init__(self, model_name: Optional[str] = None):
        # Initialize the simple text agent
        self.simple_text_agent = SimpleTextAgent(model_name=model_name)
        
        # Note: FastMCP will be imported when available
        # For now, we'll create a mock structure
        self.mcp = None
        self._initialize_mcp()
        
        # Register tools
        self._register_tools()
        
        model_name = self.simple_text_agent.metadata.get('model', 'default')
        logger.info(f"SimpleTextAgent MCP Server initialized with model: {model_name}")
    
    def _initialize_mcp(self):
        """Initialize the MCP server."""
        try:
            from fastmcp import FastMCP
            self.mcp = FastMCP("SimpleTextAgent Server")
            logger.info("FastMCP initialized successfully")
        except ImportError:
            logger.warning("FastMCP not available, using mock MCP server")
            # Create a mock MCP server for development
            self.mcp = MockMCPServer("SimpleTextAgent Server")
    
    def _register_tools(self):
        """Register all simple text analysis tools from SimpleTextAgent."""
        
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        @self.mcp.tool(
            description="Analyze text sentiment using SimpleTextAgent"
        )
        async def analyze_text_sentiment(
            text: str = Field(..., description="Text content to analyze"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Analyze the sentiment of text content using SimpleTextAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the simple text agent's sentiment analysis tool directly
                sentiment_result = await self.simple_text_agent.analyze_text_sentiment(text)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if sentiment_result.get("status") == "success":
                    content = sentiment_result.get("content", [{}])[0]
                    sentiment_data = content.get("json", {})
                    
                    return {
                        "text": text,
                        "content_type": "text",
                        "sentiment": sentiment_data.get("sentiment", "neutral"),
                        "confidence": sentiment_data.get("confidence", 0.0),
                        "scores": sentiment_data.get("scores", {"neutral": 1.0}),
                        "analysis_time": analysis_time,
                        "metadata": {
                            "method": sentiment_data.get("method", "simple_text_agent"),
                            "language": language,
                            "raw_response": sentiment_data.get("raw_response", "")
                        }
                    }
                else:
                    return {
                        "error": "Sentiment analysis failed",
                        "text": text,
                        "content_type": "text",
                        "sentiment": "neutral",
                        "confidence": 0.0,
                        "scores": {"neutral": 1.0},
                        "analysis_time": analysis_time
                    }
                    
            except Exception as e:
                logger.error(f"Error analyzing text sentiment: {e}")
                return {
                    "error": str(e),
                    "text": text,
                    "content_type": "text",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {"neutral": 1.0},
                    "analysis_time": 0.0
                }
        
        @self.mcp.tool(
            description="Extract text features using SimpleTextAgent"
        )
        async def extract_text_features(
            text: str = Field(..., description="Text content to analyze"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Extract features from text content using SimpleTextAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the simple text agent's feature extraction tool directly
                features_result = await self.simple_text_agent.extract_text_features(text)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if features_result.get("status") == "success":
                    content = features_result.get("content", [{}])[0]
                    features = content.get("json", {})
                    
                    return {
                        "text": text,
                        "content_type": "text",
                        "features": features,
                        "analysis_time": analysis_time,
                        "status": "success",
                        "metadata": {
                            "method": "simple_text_agent_feature_extraction",
                            "language": language
                        }
                    }
                else:
                    return {
                        "error": "Feature extraction failed",
                        "text": text,
                        "content_type": "text",
                        "features": {},
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error extracting text features: {e}")
                return {
                    "error": str(e),
                    "text": text,
                    "content_type": "text",
                    "features": {},
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Fallback text sentiment analysis using SimpleTextAgent"
        )
        async def fallback_text_sentiment_analysis(
            text: str = Field(..., description="Text content to analyze"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Use fallback text sentiment analysis when primary method fails."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the simple text agent's fallback tool directly
                fallback_result = await self.simple_text_agent.fallback_sentiment_analysis(text)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if fallback_result.get("status") == "success":
                    content = fallback_result.get("content", [{}])[0]
                    sentiment_data = content.get("json", {})
                    
                    return {
                        "text": text,
                        "content_type": "text",
                        "sentiment": sentiment_data.get("sentiment", "neutral"),
                        "confidence": sentiment_data.get("confidence", 0.0),
                        "scores": sentiment_data.get("scores", {"neutral": 1.0}),
                        "analysis_time": analysis_time,
                        "method": "fallback_text_analysis",
                        "metadata": {
                            "method": "simple_text_agent_fallback_analysis",
                            "language": language,
                            "fallback_method": sentiment_data.get("method", "rule_based"),
                            "positive_words_found": sentiment_data.get("positive_words_found", 0),
                            "negative_words_found": sentiment_data.get("negative_words_found", 0)
                        }
                    }
                else:
                    return {
                        "error": "Fallback analysis failed",
                        "text": text,
                        "content_type": "text",
                        "sentiment": "neutral",
                        "confidence": 0.0,
                        "scores": {"neutral": 1.0},
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error in fallback text analysis: {e}")
                return {
                    "error": str(e),
                    "text": text,
                    "content_type": "text",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {"neutral": 1.0},
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Comprehensive text analysis using SimpleTextAgent"
        )
        async def comprehensive_text_analysis(
            text: str = Field(..., description="Text content to analyze"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Perform comprehensive text analysis including sentiment and features."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Create analysis request
                request = AnalysisRequest(
                    content=text,
                    data_type=DataType.TEXT,
                    language=language,
                    confidence_threshold=confidence_threshold
                )
                
                # Process with simple text agent for comprehensive analysis
                result = await self.simple_text_agent.process(request)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                return {
                    "text": text,
                    "content_type": "text",
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "scores": result.sentiment.scores,
                    "features": result.metadata.get("features", {}),
                    "analysis_time": analysis_time,
                    "metadata": {
                        "agent_id": result.agent_id,
                        "processing_time": result.processing_time,
                        "status": result.status or "completed",
                        "method": "simple_text_agent_comprehensive_analysis",
                        "language": language,
                        "tools_used": result.metadata.get("tools_used", [])
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in comprehensive text analysis: {e}")
                return {
                    "error": str(e),
                    "text": text,
                    "content_type": "text",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {"neutral": 1.0},
                    "features": {},
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Batch analyze multiple text items using SimpleTextAgent"
        )
        async def batch_analyze_texts(
            texts: List[str] = Field(..., description="List of text items to analyze"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> List[Dict[str, Any]]:
            """Analyze sentiment for multiple text items in batch using SimpleTextAgent."""
            try:
                results = []
                for text in texts:
                    # Use comprehensive analysis for each text
                    result = await comprehensive_text_analysis(text, language, confidence_threshold)
                    results.append(result)
                return results
            except Exception as e:
                logger.error(f"Error in batch text analysis: {e}")
                return [{"error": str(e), "text": text} for text in texts]
        
        @self.mcp.tool(
            description="Get SimpleTextAgent capabilities and configuration"
        )
        def get_simple_text_agent_capabilities() -> Dict[str, Any]:
            """Get information about SimpleTextAgent capabilities and configuration."""
            return {
                "agent_id": self.simple_text_agent.agent_id,
                "model": self.simple_text_agent.metadata.get("model", "default"),
                "supported_languages": self.simple_text_agent.metadata.get("supported_languages", ["en"]),
                "capabilities": self.simple_text_agent.metadata.get("capabilities", ["text", "sentiment_analysis"]),
                "available_tools": [
                    "analyze_text_sentiment",
                    "extract_text_features",
                    "fallback_text_sentiment_analysis",
                    "comprehensive_text_analysis",
                    "batch_analyze_texts"
                ],
                "features": [
                    "text sentiment analysis",
                    "text feature extraction",
                    "fallback analysis",
                    "comprehensive analysis",
                    "batch processing",
                    "rule-based fallback",
                    "Ollama integration"
                ]
            }
    
    def run(self, host: str = "localhost", port: int = 8005, debug: bool = False):
        """Run the SimpleTextAgent MCP server."""
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        logger.info(f"Starting SimpleTextAgent MCP server on {host}:{port}")
        
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
    
    def run(self, host: str = "localhost", port: int = 8005, debug: bool = False):
        """Mock run method."""
        logger.info(f"Mock MCP server '{self.name}' would run on {host}:{port}")
        logger.info(f"Available tools: {list(self.tools.keys())}")


def create_simple_text_agent_mcp_server(model_name: Optional[str] = None) -> SimpleTextAgentMCPServer:
    """Factory function to create a SimpleTextAgent MCP server."""
    return SimpleTextAgentMCPServer(model_name=model_name)


if __name__ == "__main__":
    # Run the server directly
    server = create_simple_text_agent_mcp_server()
    server.run(host="0.0.0.0", port=8005, debug=True)
