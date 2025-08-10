"""
MCP server for VisionAgent - provides vision sentiment analysis tools.
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
from agents.vision_agent import VisionAgent


class VisionAnalysisRequest(BaseModel):
    """Request model for vision analysis."""
    image_path: str = Field(..., description="Path or URL to image/video file")
    content_type: str = Field(
        default="image", 
        description="Type of content: image, video"
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


class VisionAnalysisResponse(BaseModel):
    """Response model for vision analysis."""
    image_path: str = Field(..., description="Analyzed image/video path")
    content_type: str = Field(..., description="Type of content analyzed")
    sentiment: str = Field(..., description="Detected sentiment: positive, negative, neutral")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    scores: Dict[str, float] = Field(..., description="Detailed sentiment scores")
    features: Optional[Dict[str, Any]] = Field(None, description="Extracted vision features")
    description: Optional[str] = Field(None, description="Vision content description")
    analysis_time: float = Field(..., description="Analysis time in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata"
    )


class VisionAgentMCPServer:
    """MCP server providing vision analysis tools from VisionAgent."""
    
    def __init__(self, model_name: Optional[str] = None):
        # Initialize the vision agent
        self.vision_agent = VisionAgent(model_name=model_name)
        
        # Note: FastMCP will be imported when available
        # For now, we'll create a mock structure
        self.mcp = None
        self._initialize_mcp()
        
        # Register tools
        self._register_tools()
        
        model_name = self.vision_agent.metadata.get('model', 'default')
        logger.info(f"VisionAgent MCP Server initialized with model: {model_name}")
    
    def _initialize_mcp(self):
        """Initialize the MCP server."""
        try:
            from fastmcp import FastMCP
            self.mcp = FastMCP("VisionAgent Server")
            logger.info("FastMCP initialized successfully")
        except ImportError:
            logger.warning("FastMCP not available, using mock MCP server")
            # Create a mock MCP server for development
            self.mcp = MockMCPServer("VisionAgent Server")
    
    def _register_tools(self):
        """Register all vision analysis tools from VisionAgent."""
        
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        @self.mcp.tool(
            description="Analyze image sentiment using VisionAgent"
        )
        async def analyze_image_sentiment(
            image_path: str = Field(..., description="Path or URL to image file"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Analyze the sentiment of image content using VisionAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Create analysis request
                request = AnalysisRequest(
                    content=image_path,
                    data_type=DataType.IMAGE,
                    language=language,
                    confidence_threshold=confidence_threshold
                )
                
                # Process with vision agent
                result = await self.vision_agent.process(request)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                return {
                    "image_path": image_path,
                    "content_type": "image",
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "scores": result.sentiment.scores,
                    "description": result.extracted_text,
                    "analysis_time": analysis_time,
                    "metadata": {
                        "agent_id": result.agent_id,
                        "processing_time": result.processing_time,
                        "status": result.status or "completed",
                        "method": "vision_agent_sentiment_analysis"
                    }
                }
            except Exception as e:
                logger.error(f"Error analyzing image sentiment: {e}")
                return {
                    "error": str(e),
                    "image_path": image_path,
                    "content_type": "image",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {"neutral": 1.0},
                    "description": "",
                    "analysis_time": 0.0
                }
        
        @self.mcp.tool(
            description="Process video frame for sentiment analysis using VisionAgent"
        )
        async def process_video_frame(
            video_path: str = Field(..., description="Path or URL to video file"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Process video frame for sentiment analysis using VisionAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the vision agent's video frame processing tool directly
                video_result = await self.vision_agent.process_video_frame(video_path)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if video_result.get("status") == "success":
                    description = video_result["content"][0].get("text", "")
                    return {
                        "video_path": video_path,
                        "content_type": "video",
                        "sentiment": "neutral",  # Will be determined by sentiment analysis
                        "confidence": 0.0,
                        "scores": {"neutral": 1.0},
                        "description": description,
                        "analysis_time": analysis_time,
                        "status": "success",
                        "metadata": {
                            "method": "vision_agent_video_frame_processing",
                            "language": language
                        }
                    }
                else:
                    return {
                        "error": "Video frame processing failed",
                        "video_path": video_path,
                        "content_type": "video",
                        "sentiment": "neutral",
                        "confidence": 0.0,
                        "scores": {"neutral": 1.0},
                        "description": "",
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error processing video frame: {e}")
                return {
                    "error": str(e),
                    "video_path": video_path,
                    "content_type": "video",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {"neutral": 1.0},
                    "description": "",
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Extract vision features using VisionAgent"
        )
        async def extract_vision_features(
            image_path: str = Field(..., description="Path or URL to image file"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Extract vision features for analysis using VisionAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the vision agent's feature extraction tool directly
                features_result = await self.vision_agent.extract_vision_features(image_path)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if features_result.get("status") == "success":
                    features = features_result["content"][0].get("json", {})
                    return {
                        "image_path": image_path,
                        "content_type": "image",
                        "features": features,
                        "analysis_time": analysis_time,
                        "status": "success",
                        "metadata": {
                            "method": "vision_agent_feature_extraction",
                            "language": language
                        }
                    }
                else:
                    return {
                        "error": "Feature extraction failed",
                        "image_path": image_path,
                        "content_type": "image",
                        "features": {},
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error extracting vision features: {e}")
                return {
                    "error": str(e),
                    "image_path": image_path,
                    "content_type": "image",
                    "features": {},
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Comprehensive vision analysis using VisionAgent"
        )
        async def comprehensive_vision_analysis(
            image_path: str = Field(..., description="Path or URL to image file"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Perform comprehensive vision analysis including sentiment and features."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Create analysis request
                request = AnalysisRequest(
                    content=image_path,
                    data_type=DataType.IMAGE,
                    language=language,
                    confidence_threshold=confidence_threshold
                )
                
                # Process with vision agent for sentiment
                sentiment_result = await self.vision_agent.process(request)
                
                # Extract features
                features_result = await self.vision_agent.extract_vision_features(image_path)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                # Combine results
                features = {}
                if features_result.get("status") == "success":
                    features = features_result["content"][0].get("json", {})
                
                return {
                    "image_path": image_path,
                    "content_type": "image",
                    "sentiment": sentiment_result.sentiment.label,
                    "confidence": sentiment_result.sentiment.confidence,
                    "scores": sentiment_result.sentiment.scores,
                    "features": features,
                    "description": sentiment_result.extracted_text,
                    "analysis_time": analysis_time,
                    "metadata": {
                        "agent_id": sentiment_result.agent_id,
                        "processing_time": sentiment_result.processing_time,
                        "status": sentiment_result.status or "completed",
                        "method": "vision_agent_comprehensive_analysis",
                        "language": language
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in comprehensive vision analysis: {e}")
                return {
                    "error": str(e),
                    "image_path": image_path,
                    "content_type": "image",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {"neutral": 1.0},
                    "features": {},
                    "description": "",
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Fallback vision analysis using VisionAgent"
        )
        async def fallback_vision_analysis(
            image_path: str = Field(..., description="Path or URL to image file"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Use fallback vision analysis when primary method fails."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the vision agent's fallback tool directly
                fallback_result = await self.vision_agent.fallback_vision_analysis(image_path)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if fallback_result.get("status") == "success":
                    description = fallback_result["content"][0].get("text", "")
                    return {
                        "image_path": image_path,
                        "content_type": "image",
                        "sentiment": "neutral",
                        "confidence": 0.5,
                        "scores": {"neutral": 1.0},
                        "description": description,
                        "analysis_time": analysis_time,
                        "method": "fallback_vision_analysis",
                        "metadata": {
                            "method": "vision_agent_fallback_analysis",
                            "language": language
                        }
                    }
                else:
                    return {
                        "error": "Fallback analysis failed",
                        "image_path": image_path,
                        "content_type": "image",
                        "sentiment": "neutral",
                        "confidence": 0.0,
                        "scores": {"neutral": 1.0},
                        "description": "",
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error in fallback vision analysis: {e}")
                return {
                    "error": str(e),
                    "image_path": image_path,
                    "content_type": "image",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {"neutral": 1.0},
                    "description": "",
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Analyze video sentiment using VisionAgent"
        )
        async def analyze_video_sentiment(
            video_path: str = Field(..., description="Path or URL to video file"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Analyze the sentiment of video content using VisionAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Create analysis request
                request = AnalysisRequest(
                    content=video_path,
                    data_type=DataType.VIDEO,
                    language=language,
                    confidence_threshold=confidence_threshold
                )
                
                # Process with vision agent
                result = await self.vision_agent.process(request)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                return {
                    "video_path": video_path,
                    "content_type": "video",
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "scores": result.sentiment.scores,
                    "description": result.extracted_text,
                    "analysis_time": analysis_time,
                    "metadata": {
                        "agent_id": result.agent_id,
                        "processing_time": result.processing_time,
                        "status": result.status or "completed",
                        "method": "vision_agent_video_sentiment_analysis"
                    }
                }
            except Exception as e:
                logger.error(f"Error analyzing video sentiment: {e}")
                return {
                    "error": str(e),
                    "video_path": video_path,
                    "content_type": "video",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {"neutral": 1.0},
                    "description": "",
                    "analysis_time": 0.0
                }
        
        @self.mcp.tool(
            description="Batch analyze multiple images using VisionAgent"
        )
        async def batch_analyze_images(
            image_paths: List[str] = Field(..., description="List of image paths to analyze"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> List[Dict[str, Any]]:
            """Analyze sentiment for multiple images in batch using VisionAgent."""
            try:
                results = []
                for image_path in image_paths:
                    # Use comprehensive analysis for each image
                    result = await comprehensive_vision_analysis(image_path, language, confidence_threshold)
                    results.append(result)
                return results
            except Exception as e:
                logger.error(f"Error in batch image analysis: {e}")
                return [{"error": str(e), "image_path": image_path} for image_path in image_paths]
        
        @self.mcp.tool(
            description="Get VisionAgent capabilities and configuration"
        )
        def get_vision_agent_capabilities() -> Dict[str, Any]:
            """Get information about VisionAgent capabilities and configuration."""
            return {
                "agent_id": self.vision_agent.agent_id,
                "model": self.vision_agent.metadata.get("model", "default"),
                "supported_formats": self.vision_agent.metadata.get("supported_formats", []),
                "max_image_size": self.vision_agent.metadata.get("max_image_size", 1024),
                "max_video_duration": self.vision_agent.metadata.get("max_video_duration", 30),
                "capabilities": self.vision_agent.metadata.get("capabilities", ["vision", "tool_calling"]),
                "available_tools": [
                    "analyze_image_sentiment",
                    "process_video_frame",
                    "extract_vision_features",
                    "comprehensive_vision_analysis",
                    "fallback_vision_analysis",
                    "analyze_video_sentiment",
                    "batch_analyze_images"
                ],
                "features": [
                    "image sentiment analysis",
                    "video frame processing",
                    "vision feature extraction",
                    "comprehensive vision analysis",
                    "fallback analysis",
                    "batch processing",
                    "multi-format support"
                ]
            }
    
    def run(self, host: str = "localhost", port: int = 8003, debug: bool = False):
        """Run the VisionAgent MCP server."""
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        logger.info(f"Starting VisionAgent MCP server on {host}:{port}")
        
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
    
    def run(self, host: str = "localhost", port: int = 8003, debug: bool = False):
        """Mock run method."""
        logger.info(f"Mock MCP server '{self.name}' would run on {host}:{port}")
        logger.info(f"Available tools: {list(self.tools.keys())}")


def create_vision_agent_mcp_server(model_name: Optional[str] = None) -> VisionAgentMCPServer:
    """Factory function to create a VisionAgent MCP server."""
    return VisionAgentMCPServer(model_name=model_name)


if __name__ == "__main__":
    # Run the server directly
    server = create_vision_agent_mcp_server()
    server.run(host="0.0.0.0", port=8003, debug=True)
