"""
MCP server for TextAgent - provides text sentiment analysis tools.
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
from agents.text_agent import TextAgent


class TextAnalysisRequest(BaseModel):
    """Request model for text analysis."""
    text: str = Field(..., description="Text content to analyze")
    language: str = Field(default="en", description="Language code")
    confidence_threshold: float = Field(
        default=0.8, 
        description="Minimum confidence threshold"
    )
    analysis_type: str = Field(
        default="sentiment", 
        description="Type of analysis: sentiment, features, or both"
    )


class TextAnalysisResponse(BaseModel):
    """Response model for text analysis."""
    text: str = Field(..., description="Analyzed text")
    sentiment: str = Field(..., description="Detected sentiment: positive, negative, neutral")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    scores: Dict[str, float] = Field(..., description="Detailed sentiment scores")
    features: Optional[Dict[str, Any]] = Field(None, description="Extracted text features")
    analysis_time: float = Field(..., description="Analysis time in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata"
    )


class TextAgentMCPServer:
    """MCP server providing text analysis tools from TextAgent."""
    
    def __init__(self, model_name: Optional[str] = None):
        # Initialize the text agent
        self.text_agent = TextAgent(model_name=model_name)
        
        # Note: FastMCP will be imported when available
        # For now, we'll create a mock structure
        self.mcp = None
        self._initialize_mcp()
        
        # Register tools
        self._register_tools()
        
        model_name = self.text_agent.metadata.get('model', 'default')
        logger.info(f"TextAgent MCP Server initialized with model: {model_name}")
    
    def _initialize_mcp(self):
        """Initialize the MCP server."""
        try:
            from fastmcp import FastMCP
            self.mcp = FastMCP("TextAgent Server")
            logger.info("FastMCP initialized successfully")
        except ImportError:
            logger.warning("FastMCP not available, using mock MCP server")
            # Create a mock MCP server for development
            self.mcp = MockMCPServer("TextAgent Server")
    
    def _register_tools(self):
        """Register all text analysis tools from TextAgent."""
        
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        @self.mcp.tool(
            description="Analyze text sentiment using TextAgent"
        )
        async def analyze_text_sentiment(
            text: str = Field(..., description="Text content to analyze"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Analyze the sentiment of text content using TextAgent."""
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
                    "text": text,
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "scores": result.sentiment.scores,
                    "analysis_time": analysis_time,
                    "metadata": {
                        "agent_id": result.agent_id,
                        "processing_time": result.processing_time,
                        "status": result.status or "completed",
                        "method": "text_agent_sentiment_analysis"
                    }
                }
            except Exception as e:
                logger.error(f"Error analyzing text sentiment: {e}")
                return {
                    "error": str(e),
                    "text": text,
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {"neutral": 1.0},
                    "analysis_time": 0.0
                }
        
        @self.mcp.tool(
            description="Extract text features using TextAgent"
        )
        async def extract_text_features(
            text: str = Field(..., description="Text content to analyze"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Extract text features for analysis using TextAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the text agent's feature extraction tool directly
                features_result = await self.text_agent.extract_text_features(text)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if features_result.get("status") == "success":
                    features = features_result["content"][0].get("json", {})
                    return {
                        "text": text,
                        "features": features,
                        "analysis_time": analysis_time,
                        "status": "success",
                        "metadata": {
                            "method": "text_agent_feature_extraction",
                            "language": language
                        }
                    }
                else:
                    return {
                        "error": "Feature extraction failed",
                        "text": text,
                        "features": {},
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error extracting text features: {e}")
                return {
                    "error": str(e),
                    "text": text,
                    "features": {},
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(description="Comprehensive text analysis using TextAgent")
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
                
                # Process with text agent for sentiment
                sentiment_result = await self.text_agent.process(request)
                
                # Extract features
                features_result = await self.text_agent.extract_text_features(text)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                # Combine results
                features = {}
                if features_result.get("status") == "success":
                    features = features_result["content"][0].get("json", {})
                
                return {
                    "text": text,
                    "sentiment": sentiment_result.sentiment.label,
                    "confidence": sentiment_result.sentiment.confidence,
                    "scores": sentiment_result.sentiment.scores,
                    "features": features,
                    "analysis_time": analysis_time,
                    "metadata": {
                        "agent_id": sentiment_result.agent_id,
                        "processing_time": sentiment_result.processing_time,
                        "status": sentiment_result.status or "completed",
                        "method": "text_agent_comprehensive_analysis",
                        "language": language
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in comprehensive text analysis: {e}")
                return {
                    "error": str(e),
                    "text": text,
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {"neutral": 1.0},
                    "features": {},
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(description="Fallback sentiment analysis using TextAgent")
        async def fallback_sentiment_analysis(
            text: str = Field(..., description="Text content to analyze"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Use fallback rule-based sentiment analysis when primary method fails."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the text agent's fallback tool directly
                fallback_result = await self.text_agent.fallback_sentiment_analysis(text)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if fallback_result.get("status") == "success":
                    sentiment_data = fallback_result["content"][0].get("json", {})
                    return {
                        "text": text,
                        "sentiment": sentiment_data.get("sentiment", "neutral"),
                        "confidence": sentiment_data.get("confidence", 0.0),
                        "scores": sentiment_data.get("scores", {"neutral": 1.0}),
                        "analysis_time": analysis_time,
                        "method": "fallback_rule_based",
                        "metadata": {
                            "method": "text_agent_fallback_analysis",
                            "language": language,
                            "positive_words_found": sentiment_data.get("positive_words_found", 0),
                            "negative_words_found": sentiment_data.get("negative_words_found", 0)
                        }
                    }
                else:
                    return {
                        "error": "Fallback analysis failed",
                        "text": text,
                        "sentiment": "neutral",
                        "confidence": 0.0,
                        "scores": {"neutral": 1.0},
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error in fallback sentiment analysis: {e}")
                return {
                    "error": str(e),
                    "text": text,
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {"neutral": 1.0},
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(description="Batch analyze multiple text items using TextAgent")
        async def batch_analyze_texts(
            texts: List[str] = Field(..., description="List of text items to analyze"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> List[Dict[str, Any]]:
            """Analyze sentiment for multiple text items in batch using TextAgent."""
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
        
        @self.mcp.tool(description="Get TextAgent capabilities and configuration")
        def get_text_agent_capabilities() -> Dict[str, Any]:
            """Get information about TextAgent capabilities and configuration."""
            return {
                "agent_id": self.text_agent.agent_id,
                "model": self.text_agent.metadata.get("model", "default"),
                "supported_languages": self.text_agent.metadata.get("supported_languages", ["en"]),
                "capabilities": self.text_agent.metadata.get("capabilities", ["text", "sentiment_analysis"]),
                "available_tools": [
                    "analyze_text_sentiment",
                    "extract_text_features", 
                    "comprehensive_text_analysis",
                    "fallback_sentiment_analysis",
                    "batch_analyze_texts"
                ],
                "features": [
                    "sentiment classification",
                    "confidence scoring",
                    "text feature extraction",
                    "rule-based fallback analysis",
                    "batch processing",
                    "multi-language support"
                ]
            }
    
    def run(self, host: str = "localhost", port: int = 8002, debug: bool = False):
        """Run the TextAgent MCP server."""
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        logger.info(f"Starting TextAgent MCP server on {host}:{port}")
        
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
    
    def run(self, host: str = "localhost", port: int = 8002, debug: bool = False):
        """Mock run method."""
        logger.info(f"Mock MCP server '{self.name}' would run on {host}:{port}")
        logger.info(f"Available tools: {list(self.tools.keys())}")


def create_text_agent_mcp_server(model_name: Optional[str] = None) -> TextAgentMCPServer:
    """Factory function to create a TextAgent MCP server."""
    return TextAgentMCPServer(model_name=model_name)


if __name__ == "__main__":
    # Run the server directly
    server = create_text_agent_mcp_server()
    server.run(host="0.0.0.0", port=8002, debug=True)
