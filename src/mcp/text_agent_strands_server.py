"""
MCP server for TextAgentStrands - provides text sentiment analysis using Strands framework.
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
from agents.text_agent_strands import TextAgentStrands


class TextStrandsAnalysisRequest(BaseModel):
    """Request model for text strands analysis."""
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


class TextStrandsAnalysisResponse(BaseModel):
    """Response model for text strands analysis."""
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


class TextAgentStrandsMCPServer:
    """MCP server providing text analysis tools from TextAgentStrands."""
    
    def __init__(self, model_name: Optional[str] = None):
        # Initialize the text agent strands
        self.text_agent_strands = TextAgentStrands(model_name=model_name)
        
        # Note: FastMCP will be imported when available
        # For now, we'll create a mock structure
        self.mcp = None
        self._initialize_mcp()
        
        # Register tools
        self._register_tools()
        
        model_name = self.text_agent_strands.model_name
        logger.info(f"TextAgentStrands MCP Server initialized with model: {model_name}")
    
    def _initialize_mcp(self):
        """Initialize the MCP server."""
        try:
            from fastmcp import FastMCP
            self.mcp = FastMCP("TextAgentStrands Server")
            logger.info("FastMCP initialized successfully")
        except ImportError:
            logger.warning("FastMCP not available, using mock MCP server")
            # Create a mock MCP server for development
            self.mcp = MockMCPServer("TextAgentStrands Server")
    
    def _register_tools(self):
        """Register all text analysis tools from TextAgentStrands."""
        
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        @self.mcp.tool(
            description="Analyze text sentiment using TextAgentStrands"
        )
        async def analyze_text_sentiment_strands(
            text: str = Field(..., description="Text content to analyze"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Analyze the sentiment of text content using TextAgentStrands."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the text agent strands' sentiment analysis directly
                sentiment_result = await self.text_agent_strands.analyze_text_sentiment(text)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                # Parse the sentiment result
                if "SENTIMENT:" in sentiment_result and "CONFIDENCE:" in sentiment_result:
                    # Parse structured response
                    parts = sentiment_result.split(",")
                    sentiment_part = parts[0].strip()
                    confidence_part = parts[1].strip()
                    
                    # Extract sentiment label
                    if "POSITIVE" in sentiment_part:
                        sentiment_label = "positive"
                    elif "NEGATIVE" in sentiment_part:
                        sentiment_label = "negative"
                    else:
                        sentiment_label = "neutral"
                    
                    # Extract confidence
                    try:
                        confidence_str = confidence_part.split(":")[1].strip()
                        confidence = float(confidence_str)
                    except (IndexError, ValueError):
                        confidence = 0.6
                else:
                    # Fallback parsing
                    if "POSITIVE" in sentiment_result.upper():
                        sentiment_label = "positive"
                        confidence = 0.8
                    elif "NEGATIVE" in sentiment_result.upper():
                        sentiment_label = "negative"
                        confidence = 0.8
                    else:
                        sentiment_label = "neutral"
                        confidence = 0.6
                
                return {
                    "text": text,
                    "content_type": "text",
                    "sentiment": sentiment_label,
                    "confidence": confidence,
                    "scores": {"strands_response": confidence},
                    "analysis_time": analysis_time,
                    "metadata": {
                        "method": "text_agent_strands",
                        "language": language,
                        "raw_response": sentiment_result,
                        "framework": "strands"
                    }
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
            description="Fallback text sentiment analysis using TextAgentStrands"
        )
        async def fallback_text_sentiment_analysis_strands(
            text: str = Field(..., description="Text content to analyze"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Use fallback text sentiment analysis when primary method fails."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the text agent strands' fallback tool directly
                fallback_result = await self.text_agent_strands.fallback_sentiment_analysis(text)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                # Parse the fallback result
                if "SENTIMENT:" in fallback_result and "CONFIDENCE:" in fallback_result:
                    # Parse structured response
                    parts = fallback_result.split(",")
                    sentiment_part = parts[0].strip()
                    confidence_part = parts[1].strip()
                    
                    # Extract sentiment label
                    if "POSITIVE" in sentiment_part:
                        sentiment_label = "positive"
                    elif "NEGATIVE" in sentiment_part:
                        sentiment_label = "negative"
                    else:
                        sentiment_label = "neutral"
                    
                    # Extract confidence
                    try:
                        confidence_str = confidence_part.split(":")[1].strip()
                        confidence = float(confidence_str)
                    except (IndexError, ValueError):
                        confidence = 0.6
                else:
                    # Fallback parsing
                    if "POSITIVE" in fallback_result.upper():
                        sentiment_label = "positive"
                        confidence = 0.8
                    elif "NEGATIVE" in fallback_result.upper():
                        sentiment_label = "negative"
                        confidence = 0.8
                    else:
                        sentiment_label = "neutral"
                        confidence = 0.6
                
                return {
                    "text": text,
                    "content_type": "text",
                    "sentiment": sentiment_label,
                    "confidence": confidence,
                    "scores": {"fallback_response": confidence},
                    "analysis_time": analysis_time,
                    "method": "fallback_text_analysis",
                    "metadata": {
                        "method": "text_agent_strands_fallback",
                        "language": language,
                        "fallback_method": "rule_based",
                        "framework": "strands"
                    }
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
            description="Comprehensive text analysis using TextAgentStrands"
        )
        async def comprehensive_text_analysis_strands(
            text: str = Field(..., description="Text content to analyze"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Perform comprehensive text analysis including sentiment using TextAgentStrands."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Create analysis request
                request = AnalysisRequest(
                    content=text,
                    data_type=DataType.TEXT,
                    language=language,
                    confidence_threshold=confidence_threshold
                )
                
                # Process with text agent strands for comprehensive analysis
                result = await self.text_agent_strands.process(request)
                
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
                        "method": "text_agent_strands_comprehensive",
                        "language": language,
                        "framework": "strands",
                        "extracted_text": result.extracted_text,
                        "raw_content": result.raw_content
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
            description="Batch analyze multiple text items using TextAgentStrands"
        )
        async def batch_analyze_texts_strands(
            texts: List[str] = Field(..., description="List of text items to analyze"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> List[Dict[str, Any]]:
            """Analyze sentiment for multiple text items in batch using TextAgentStrands."""
            try:
                results = []
                for text in texts:
                    # Use comprehensive analysis for each text
                    result = await comprehensive_text_analysis_strands(
                        text, language, confidence_threshold
                    )
                    results.append(result)
                return results
            except Exception as e:
                logger.error(f"Error in batch text analysis: {e}")
                return [{"error": str(e), "text": text} for text in texts]
        
        @self.mcp.tool(
            description="Check if TextAgentStrands can process specific content"
        )
        async def can_process_text_strands(
            text: str = Field(..., description="Text content to check"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Check if TextAgentStrands can process the given text content."""
            try:
                # Create analysis request
                request = AnalysisRequest(
                    content=text,
                    data_type=DataType.TEXT,
                    language=language
                )
                
                # Check if can process
                can_process = await self.text_agent_strands.can_process(request)
                
                return {
                    "text": text,
                    "can_process": can_process,
                    "language": language,
                    "data_type": "text",
                    "agent_id": self.text_agent_strands.agent_id,
                    "framework": "strands"
                }
                
            except Exception as e:
                logger.error(f"Error checking if can process: {e}")
                return {
                    "error": str(e),
                    "text": text,
                    "can_process": False,
                    "language": language,
                    "data_type": "text"
                }
        
        @self.mcp.tool(
            description="Get TextAgentStrands capabilities and configuration"
        )
        def get_text_agent_strands_capabilities() -> Dict[str, Any]:
            """Get information about TextAgentStrands capabilities and configuration."""
            return {
                "agent_id": self.text_agent_strands.agent_id,
                "model": self.text_agent_strands.model_name,
                "supported_languages": ["en"],
                "capabilities": ["text", "sentiment_analysis"],
                "available_tools": [
                    "analyze_text_sentiment_strands",
                    "fallback_text_sentiment_analysis_strands",
                    "comprehensive_text_analysis_strands",
                    "batch_analyze_texts_strands",
                    "can_process_text_strands"
                ],
                "features": [
                    "text sentiment analysis",
                    "strands framework integration",
                    "fallback analysis",
                    "comprehensive analysis",
                    "batch processing",
                    "Ollama integration",
                    "rule-based fallback"
                ],
                "framework": "strands"
            }
    
    def run(self, host: str = "localhost", port: int = 8006, debug: bool = False):
        """Run the TextAgentStrands MCP server."""
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        logger.info(f"Starting TextAgentStrands MCP server on {host}:{port}")
        
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
    
    def run(self, host: str = "localhost", port: int = 8006, debug: bool = False):
        """Mock run method."""
        logger.info(f"Mock MCP server '{self.name}' would run on {host}:{port}")
        logger.info(f"Available tools: {list(self.tools.keys())}")


def create_text_agent_strands_mcp_server(model_name: Optional[str] = None) -> TextAgentStrandsMCPServer:
    """Factory function to create a TextAgentStrands MCP server."""
    return TextAgentStrandsMCPServer(model_name=model_name)


if __name__ == "__main__":
    # Run the server directly
    server = create_text_agent_strands_mcp_server()
    server.run(host="0.0.0.0", port=8006, debug=True)
