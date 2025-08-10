"""
FastMCP server for sentiment analysis tools.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field
import asyncio
from loguru import logger

# Import the correct models and agents
from ..core.models import (
    AnalysisRequest, 
    DataType
)
from ..agents.text_agent import TextAgent
from ..agents.vision_agent_enhanced import EnhancedVisionAgent
from ..agents.audio_agent_enhanced import EnhancedAudioAgent


class SentimentRequest(BaseModel):
    """Request model for sentiment analysis."""
    content: str = Field(..., description="Content to analyze")
    content_type: str = Field(
        default="text", 
        description="Type of content: text, image, audio, video"
    )
    language: str = Field(default="en", description="Language code")
    confidence_threshold: float = Field(
        default=0.8, 
        description="Minimum confidence threshold"
    )


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""
    sentiment: str = Field(..., description="Detected sentiment: positive, negative, neutral")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    score: float = Field(..., description="Sentiment score (-1.0 to 1.0)")
    language: str = Field(..., description="Detected language")
    analysis_time: float = Field(..., description="Analysis time in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata"
    )


class SentimentMCPServer:
    """FastMCP server providing sentiment analysis tools."""
    
    def __init__(self):
        # Initialize agents
        self.text_agent = TextAgent()
        self.vision_agent = EnhancedVisionAgent()
        self.audio_agent = EnhancedAudioAgent()
        
        # Note: FastMCP will be imported when available
        # For now, we'll create a mock structure
        self.mcp = None
        self._initialize_mcp()
        
        # Register tools
        self._register_tools()
    
    def _initialize_mcp(self):
        """Initialize the MCP server."""
        try:
            from fastmcp import FastMCP
            self.mcp = FastMCP("Sentiment Analysis Server")
            logger.info("FastMCP initialized successfully")
        except ImportError:
            logger.warning("FastMCP not available, using mock MCP server")
            # Create a mock MCP server for development
            self.mcp = MockMCPServer("Sentiment Analysis Server")
    
    def _register_tools(self):
        """Register all sentiment analysis tools."""
        
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        @self.mcp.tool(description="Analyze text sentiment")
        async def analyze_text_sentiment(
            text: str = Field(..., description="Text content to analyze"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Analyze the sentiment of text content."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Create analysis request
                request = AnalysisRequest(
                    content=text,
                    data_type=DataType.TEXT,
                    language=language,
                    confidence_threshold=confidence_threshold
                )
                
                # Process with text agent
                result = await self.text_agent.process(request)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                return {
                    "sentiment": result.sentiment.label.value,
                    "confidence": result.sentiment.confidence,
                    "score": result.sentiment.scores.get("overall", 0.0),
                    "language": result.language,
                    "analysis_time": analysis_time,
                    "metadata": {
                        "agent_id": result.agent_id,
                        "processing_time": result.processing_time,
                        "status": result.status or "completed"
                    }
                }
            except Exception as e:
                logger.error(f"Error analyzing text sentiment: {e}")
                return {
                    "error": str(e),
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "score": 0.0,
                    "language": language,
                    "analysis_time": 0.0
                }
        
        @self.mcp.tool(description="Analyze image sentiment")
        async def analyze_image_sentiment(
            image_url: str = Field(..., description="URL or path to image"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Analyze the sentiment of image content."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Create analysis request
                request = AnalysisRequest(
                    content=image_url,
                    data_type=DataType.IMAGE,
                    language=language,
                    confidence_threshold=confidence_threshold
                )
                
                # Process with vision agent
                result = await self.vision_agent.process(request)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                return {
                    "sentiment": result.sentiment.label.value,
                    "confidence": result.sentiment.confidence,
                    "score": result.sentiment.scores.get("overall", 0.0),
                    "language": result.language,
                    "analysis_time": analysis_time,
                    "metadata": {
                        "agent_id": result.agent_id,
                        "processing_time": result.processing_time,
                        "status": result.status or "completed",
                        "image_url": image_url
                    }
                }
            except Exception as e:
                logger.error(f"Error analyzing image sentiment: {e}")
                return {
                    "error": str(e),
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "score": 0.0,
                    "language": language,
                    "analysis_time": 0.0
                }
        
        @self.mcp.tool(description="Analyze audio sentiment")
        async def analyze_audio_sentiment(
            audio_url: str = Field(..., description="URL or path to audio file"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Analyze the sentiment of audio content."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Create analysis request
                request = AnalysisRequest(
                    content=audio_url,
                    data_type=DataType.AUDIO,
                    language=language,
                    confidence_threshold=confidence_threshold
                )
                
                # Process with audio agent
                result = await self.audio_agent.process(request)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                return {
                    "sentiment": result.sentiment.label.value,
                    "confidence": result.sentiment.confidence,
                    "score": result.sentiment.scores.get("overall", 0.0),
                    "language": result.language,
                    "analysis_time": analysis_time,
                    "metadata": {
                        "agent_id": result.agent_id,
                        "processing_time": result.processing_time,
                        "status": result.status or "completed",
                        "audio_url": audio_url
                    }
                }
            except Exception as e:
                logger.error(f"Error analyzing audio sentiment: {e}")
                return {
                    "error": str(e),
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "score": 0.0,
                    "language": language,
                    "analysis_time": 0.0
                }
        
        @self.mcp.tool(description="Get available sentiment analysis capabilities")
        def get_capabilities() -> Dict[str, Any]:
            """Get information about available sentiment analysis capabilities."""
            return {
                "supported_types": ["text", "image", "audio", "video"],
                "supported_languages": ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                "models": {
                    "text": "Ollama text models",
                    "vision": "LLaVA vision models",
                    "audio": "Audio processing models"
                },
                "features": [
                    "sentiment classification",
                    "confidence scoring",
                    "multi-language support",
                    "real-time processing",
                    "batch processing"
                ]
            }
        
        @self.mcp.tool(description="Batch analyze multiple text items")
        async def batch_analyze_text(
            texts: List[str] = Field(..., description="List of text items to analyze"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> List[Dict[str, Any]]:
            """Analyze sentiment for multiple text items in batch."""
            try:
                results = []
                for text in texts:
                    result = await analyze_text_sentiment(text, language, confidence_threshold)
                    results.append(result)
                return results
            except Exception as e:
                logger.error(f"Error in batch text analysis: {e}")
                return [{"error": str(e)} for _ in texts]
    
    def run(self, host: str = "localhost", port: int = 8001, debug: bool = False):
        """Run the FastMCP server."""
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        logger.info(f"Starting Sentiment Analysis MCP server on {host}:{port}")
        
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
    
    def run(self, host: str = "localhost", port: int = 8001, debug: bool = False):
        """Mock run method."""
        logger.info(f"Mock MCP server '{self.name}' would run on {host}:{port}")
        logger.info(f"Available tools: {list(self.tools.keys())}")


def create_sentiment_mcp_server() -> SentimentMCPServer:
    """Factory function to create a sentiment MCP server."""
    return SentimentMCPServer()


if __name__ == "__main__":
    # Run the server directly
    server = create_sentiment_mcp_server()
    server.run(host="0.0.0.0", port=8001, debug=True)
