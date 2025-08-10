"""
MCP server for WebAgent - provides web scraping and webpage sentiment analysis tools.
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
from agents.web_agent import WebAgent


class WebAnalysisRequest(BaseModel):
    """Request model for web analysis."""
    url: str = Field(..., description="URL of webpage to analyze")
    content_type: str = Field(
        default="webpage", 
        description="Type of content: webpage, article, blog"
    )
    language: str = Field(default="en", description="Language code")
    confidence_threshold: float = Field(
        default=0.8, 
        description="Minimum confidence threshold"
    )
    analysis_type: str = Field(
        default="sentiment", 
        description="Type of analysis: sentiment, features, or both"
    )


class WebAnalysisResponse(BaseModel):
    """Response model for web analysis."""
    url: str = Field(..., description="Analyzed webpage URL")
    content_type: str = Field(..., description="Type of content analyzed")
    sentiment: str = Field(..., description="Detected sentiment: positive, negative, neutral")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    scores: Dict[str, float] = Field(..., description="Detailed sentiment scores")
    features: Optional[Dict[str, Any]] = Field(None, description="Extracted webpage features")
    extracted_text: Optional[str] = Field(None, description="Extracted webpage text content")
    title: Optional[str] = Field(None, description="Webpage title")
    analysis_time: float = Field(..., description="Analysis time in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata"
    )


class WebAgentMCPServer:
    """MCP server providing web analysis tools from WebAgent."""
    
    def __init__(self, model_name: Optional[str] = None):
        # Initialize the web agent
        self.web_agent = WebAgent(model_name=model_name)
        
        # Note: FastMCP will be imported when available
        # For now, we'll create a mock structure
        self.mcp = None
        self._initialize_mcp()
        
        # Register tools
        self._register_tools()
        
        model_name = self.web_agent.metadata.get('model', 'default')
        logger.info(f"WebAgent MCP Server initialized with model: {model_name}")
    
    def _initialize_mcp(self):
        """Initialize the MCP server."""
        try:
            from fastmcp import FastMCP
            self.mcp = FastMCP("WebAgent Server")
            logger.info("FastMCP initialized successfully")
        except ImportError:
            logger.warning("FastMCP not available, using mock MCP server")
            # Create a mock MCP server for development
            self.mcp = MockMCPServer("WebAgent Server")
    
    def _register_tools(self):
        """Register all web analysis tools from WebAgent."""
        
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        @self.mcp.tool(
            description="Scrape and extract content from a webpage using WebAgent"
        )
        async def scrape_webpage(
            url: str = Field(..., description="URL of webpage to scrape"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Scrape and extract content from a webpage using WebAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the web agent's scraping tool directly
                webpage_content = await self.web_agent.scrape_webpage(url)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if "error" not in webpage_content:
                    return {
                        "url": url,
                        "content_type": "webpage",
                        "status": "success",
                        "title": webpage_content.get("title", ""),
                        "text": webpage_content.get("text", ""),
                        "html": webpage_content.get("html", ""),
                        "status_code": webpage_content.get("status_code", 0),
                        "analysis_time": analysis_time,
                        "metadata": {
                            "method": "web_agent_scraping",
                            "language": language,
                            "content_length": len(webpage_content.get("text", ""))
                        }
                    }
                else:
                    return {
                        "error": webpage_content["error"],
                        "url": url,
                        "content_type": "webpage",
                        "status": "failed",
                        "analysis_time": analysis_time
                    }
                    
            except Exception as e:
                logger.error(f"Error scraping webpage: {e}")
                return {
                    "error": str(e),
                    "url": url,
                    "content_type": "webpage",
                    "status": "failed",
                    "analysis_time": 0.0
                }
        
        @self.mcp.tool(
            description="Analyze webpage sentiment using WebAgent"
        )
        async def analyze_webpage_sentiment(
            url: str = Field(..., description="URL of webpage to analyze"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Analyze the sentiment of webpage content using WebAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Create analysis request
                request = AnalysisRequest(
                    content=url,
                    data_type=DataType.WEBPAGE,
                    language=language,
                    confidence_threshold=confidence_threshold
                )
                
                # Process with web agent
                result = await self.web_agent.process(request)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                return {
                    "url": url,
                    "content_type": "webpage",
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "scores": result.sentiment.scores,
                    "extracted_text": result.extracted_text,
                    "title": result.metadata.get("title", ""),
                    "analysis_time": analysis_time,
                    "metadata": {
                        "agent_id": result.agent_id,
                        "processing_time": result.processing_time,
                        "status": result.status or "completed",
                        "method": "web_agent_sentiment_analysis",
                        "language": language
                    }
                }
            except Exception as e:
                logger.error(f"Error analyzing webpage sentiment: {e}")
                return {
                    "error": str(e),
                    "url": url,
                    "content_type": "webpage",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {"neutral": 1.0},
                    "extracted_text": "",
                    "title": "",
                    "analysis_time": 0.0
                }
        
        @self.mcp.tool(
            description="Extract webpage features using WebAgent"
        )
        async def extract_webpage_features(
            url: str = Field(..., description="URL of webpage to analyze"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Extract features from webpage content using WebAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # First scrape the webpage to get content
                webpage_content = await self.web_agent.scrape_webpage(url)
                
                if "error" in webpage_content:
                    return {
                        "error": webpage_content["error"],
                        "url": url,
                        "content_type": "webpage",
                        "features": {},
                        "analysis_time": 0.0,
                        "status": "failed"
                    }
                
                # Use the web agent's feature extraction tool
                features_result = await self.web_agent.extract_webpage_features(webpage_content)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if "error" not in features_result:
                    return {
                        "url": url,
                        "content_type": "webpage",
                        "features": features_result,
                        "analysis_time": analysis_time,
                        "status": "success",
                        "metadata": {
                            "method": "web_agent_feature_extraction",
                            "language": language
                        }
                    }
                else:
                    return {
                        "error": features_result["error"],
                        "url": url,
                        "content_type": "webpage",
                        "features": {},
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error extracting webpage features: {e}")
                return {
                    "error": str(e),
                    "url": url,
                    "content_type": "webpage",
                    "features": {},
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Comprehensive webpage analysis using WebAgent"
        )
        async def comprehensive_webpage_analysis(
            url: str = Field(..., description="URL of webpage to analyze"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Perform comprehensive webpage analysis including sentiment and features."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Create analysis request
                request = AnalysisRequest(
                    content=url,
                    data_type=DataType.WEBPAGE,
                    language=language,
                    confidence_threshold=confidence_threshold
                )
                
                # Process with web agent for sentiment
                sentiment_result = await self.web_agent.process(request)
                
                # Extract features
                webpage_content = await self.web_agent.scrape_webpage(url)
                features_result = await self.web_agent.extract_webpage_features(webpage_content)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                # Combine results
                features = {}
                if "error" not in features_result:
                    features = features_result
                
                return {
                    "url": url,
                    "content_type": "webpage",
                    "sentiment": sentiment_result.sentiment.label,
                    "confidence": sentiment_result.sentiment.confidence,
                    "scores": sentiment_result.sentiment.scores,
                    "features": features,
                    "extracted_text": sentiment_result.extracted_text,
                    "title": sentiment_result.metadata.get("title", ""),
                    "analysis_time": analysis_time,
                    "metadata": {
                        "agent_id": sentiment_result.agent_id,
                        "processing_time": sentiment_result.processing_time,
                        "status": sentiment_result.status or "completed",
                        "method": "web_agent_comprehensive_analysis",
                        "language": language
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in comprehensive webpage analysis: {e}")
                return {
                    "error": str(e),
                    "url": url,
                    "content_type": "webpage",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {"neutral": 1.0},
                    "features": {},
                    "extracted_text": "",
                    "title": "",
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Fallback webpage analysis using WebAgent"
        )
        async def fallback_webpage_analysis(
            url: str = Field(..., description="URL of webpage to analyze"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Use fallback webpage analysis when primary method fails."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # First scrape the webpage to get content
                webpage_content = await self.web_agent.scrape_webpage(url)
                
                if "error" in webpage_content:
                    return {
                        "error": webpage_content["error"],
                        "url": url,
                        "content_type": "webpage",
                        "sentiment": "neutral",
                        "confidence": 0.0,
                        "scores": {"neutral": 1.0},
                        "extracted_text": "",
                        "title": "",
                        "analysis_time": 0.0,
                        "status": "failed"
                    }
                
                # Use the web agent's fallback tool
                fallback_result = await self.web_agent.fallback_webpage_analysis(webpage_content)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if "error" not in fallback_result:
                    return {
                        "url": url,
                        "content_type": "webpage",
                        "sentiment": fallback_result["sentiment"],
                        "confidence": fallback_result["confidence"],
                        "scores": {"fallback": fallback_result["confidence"]},
                        "extracted_text": webpage_content.get("text", ""),
                        "title": webpage_content.get("title", ""),
                        "analysis_time": analysis_time,
                        "method": "fallback_webpage_analysis",
                        "metadata": {
                            "method": "web_agent_fallback_analysis",
                            "language": language,
                            "fallback_method": fallback_result.get("method", "rule_based")
                        }
                    }
                else:
                    return {
                        "error": fallback_result["error"],
                        "url": url,
                        "content_type": "webpage",
                        "sentiment": "neutral",
                        "confidence": 0.0,
                        "scores": {"neutral": 1.0},
                        "extracted_text": "",
                        "title": "",
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error in fallback webpage analysis: {e}")
                return {
                    "error": str(e),
                    "url": url,
                    "content_type": "webpage",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {"neutral": 1.0},
                    "extracted_text": "",
                    "title": "",
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Batch analyze multiple webpages using WebAgent"
        )
        async def batch_analyze_webpages(
            urls: List[str] = Field(..., description="List of webpage URLs to analyze"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> List[Dict[str, Any]]:
            """Analyze sentiment for multiple webpages in batch using WebAgent."""
            try:
                results = []
                for url in urls:
                    # Use comprehensive analysis for each webpage
                    result = await comprehensive_webpage_analysis(url, language, confidence_threshold)
                    results.append(result)
                return results
            except Exception as e:
                logger.error(f"Error in batch webpage analysis: {e}")
                return [{"error": str(e), "url": url} for url in urls]
        
        @self.mcp.tool(
            description="Validate webpage URL format"
        )
        def validate_webpage_url(
            url: str = Field(..., description="URL to validate")
        ) -> Dict[str, Any]:
            """Validate if a URL has proper format for webpage analysis."""
            try:
                from urllib.parse import urlparse
                result = urlparse(url)
                is_valid = all([result.scheme, result.netloc])
                
                return {
                    "url": url,
                    "is_valid": is_valid,
                    "scheme": result.scheme,
                    "netloc": result.netloc,
                    "path": result.path,
                    "query": result.query,
                    "fragment": result.fragment
                }
            except Exception as e:
                return {
                    "url": url,
                    "is_valid": False,
                    "error": str(e)
                }
        
        @self.mcp.tool(
            description="Get WebAgent capabilities and configuration"
        )
        def get_web_agent_capabilities() -> Dict[str, Any]:
            """Get information about WebAgent capabilities and configuration."""
            return {
                "agent_id": self.web_agent.agent_id,
                "model": self.web_agent.metadata.get("model", "default"),
                "max_content_length": self.web_agent.metadata.get("max_content_length", 10000),
                "timeout": self.web_agent.metadata.get("timeout", 30),
                "user_agent": self.web_agent.metadata.get("user_agent", ""),
                "capabilities": self.web_agent.metadata.get("capabilities", ["web", "scraping", "sentiment_analysis"]),
                "available_tools": [
                    "scrape_webpage",
                    "analyze_webpage_sentiment",
                    "extract_webpage_features",
                    "comprehensive_webpage_analysis",
                    "fallback_webpage_analysis",
                    "batch_analyze_webpages",
                    "validate_webpage_url"
                ],
                "features": [
                    "webpage scraping",
                    "content extraction",
                    "sentiment analysis",
                    "feature extraction",
                    "comprehensive analysis",
                    "fallback analysis",
                    "batch processing",
                    "URL validation"
                ]
            }
    
    def run(self, host: str = "localhost", port: int = 8004, debug: bool = False):
        """Run the WebAgent MCP server."""
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        logger.info(f"Starting WebAgent MCP server on {host}:{port}")
        
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
    
    def run(self, host: str = "localhost", port: int = 8004, debug: bool = False):
        """Mock run method."""
        logger.info(f"Mock MCP server '{self.name}' would run on {host}:{port}")
        logger.info(f"Available tools: {list(self.tools.keys())}")


def create_web_agent_mcp_server(model_name: Optional[str] = None) -> WebAgentMCPServer:
    """Factory function to create a WebAgent MCP server."""
    return WebAgentMCPServer(model_name=model_name)


if __name__ == "__main__":
    # Run the server directly
    server = create_web_agent_mcp_server()
    server.run(host="0.0.0.0", port=8004, debug=True)
