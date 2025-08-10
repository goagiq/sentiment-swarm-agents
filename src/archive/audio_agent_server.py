"""
MCP server for AudioAgent - provides audio sentiment analysis tools.
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
from agents.audio_agent import AudioAgent


class AudioAnalysisRequest(BaseModel):
    """Request model for audio analysis."""
    audio_path: str = Field(..., description="Path or URL to audio file")
    content_type: str = Field(
        default="audio", 
        description="Type of content: audio, voice, music"
    )
    language: str = Field(default="en", description="Language code")
    confidence_threshold: float = Field(
        default=0.8, 
        description="Minimum confidence threshold"
    )
    analysis_type: str = Field(
        default="sentiment", 
        description="Type of analysis: sentiment, transcription, features, or all"
    )


class AudioAnalysisResponse(BaseModel):
    """Response model for audio analysis."""
    audio_path: str = Field(..., description="Analyzed audio file path")
    content_type: str = Field(..., description="Type of content analyzed")
    sentiment: str = Field(..., description="Detected sentiment: positive, negative, neutral")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    scores: Dict[str, float] = Field(..., description="Detailed sentiment scores")
    transcription: Optional[str] = Field(None, description="Audio transcription text")
    features: Optional[Dict[str, Any]] = Field(None, description="Extracted audio features")
    analysis_time: float = Field(..., description="Analysis time in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata"
    )


class AudioAgentMCPServer:
    """MCP server providing audio analysis tools from AudioAgent."""
    
    def __init__(self, model_name: Optional[str] = None):
        # Initialize the audio agent
        self.audio_agent = AudioAgent(model_name=model_name)
        
        # Note: FastMCP will be imported when available
        # For now, we'll create a mock structure
        self.mcp = None
        self._initialize_mcp()
        
        # Register tools
        self._register_tools()
        
        model_name = self.audio_agent.metadata.get('model', 'default')
        logger.info(f"AudioAgent MCP Server initialized with model: {model_name}")
    
    def _initialize_mcp(self):
        """Initialize the MCP server."""
        try:
            from fastmcp import FastMCP
            self.mcp = FastMCP("AudioAgent Server")
            logger.info("FastMCP initialized successfully")
        except ImportError:
            logger.warning("FastMCP not available, using mock MCP server")
            # Create a mock MCP server for development
            self.mcp = MockMCPServer("AudioAgent Server")
    
    def _register_tools(self):
        """Register all audio analysis tools from AudioAgent."""
        
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        @self.mcp.tool(
            description="Transcribe audio to text using AudioAgent"
        )
        async def transcribe_audio(
            audio_path: str = Field(..., description="Path or URL to audio file"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Transcribe audio content to text using AudioAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the audio agent's transcription tool directly
                transcription_result = await self.audio_agent.transcribe_audio(audio_path)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if transcription_result.get("status") == "success":
                    transcription = transcription_result["content"][0].get("text", "")
                    return {
                        "audio_path": audio_path,
                        "transcription": transcription,
                        "analysis_time": analysis_time,
                        "status": "success",
                        "metadata": {
                            "method": "audio_agent_transcription",
                            "language": language,
                            "agent_id": self.audio_agent.agent_id
                        }
                    }
                else:
                    return {
                        "error": "Transcription failed",
                        "audio_path": audio_path,
                        "transcription": "",
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error transcribing audio: {e}")
                return {
                    "error": str(e),
                    "audio_path": audio_path,
                    "transcription": "",
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Analyze audio sentiment using AudioAgent"
        )
        async def analyze_audio_sentiment(
            audio_path: str = Field(..., description="Path or URL to audio file"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Analyze the sentiment of audio content using AudioAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the audio agent's sentiment analysis tool directly
                sentiment_result = await self.audio_agent.analyze_audio_sentiment(audio_path)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if sentiment_result.get("status") == "success":
                    sentiment_data = sentiment_result["content"][0].get("json", {})
                    return {
                        "audio_path": audio_path,
                        "sentiment": sentiment_data.get("sentiment", "neutral"),
                        "confidence": sentiment_data.get("confidence", 0.0),
                        "transcription": sentiment_data.get("transcription", ""),
                        "analysis_time": analysis_time,
                        "status": "success",
                        "metadata": {
                            "method": "audio_agent_sentiment_analysis",
                            "language": language,
                            "agent_id": self.audio_agent.agent_id,
                            "raw_response": sentiment_data.get("raw_response", "")
                        }
                    }
                else:
                    return {
                        "error": "Sentiment analysis failed",
                        "audio_path": audio_path,
                        "sentiment": "neutral",
                        "confidence": 0.0,
                        "transcription": "",
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error analyzing audio sentiment: {e}")
                return {
                    "error": str(e),
                    "audio_path": audio_path,
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "transcription": "",
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Extract audio features using AudioAgent"
        )
        async def extract_audio_features(
            audio_path: str = Field(..., description="Path or URL to audio file"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Extract audio features for analysis using AudioAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the audio agent's feature extraction tool directly
                features_result = await self.audio_agent.extract_audio_features(audio_path)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if features_result.get("status") == "success":
                    features = features_result["content"][0].get("json", {})
                    return {
                        "audio_path": audio_path,
                        "features": features,
                        "analysis_time": analysis_time,
                        "status": "success",
                        "metadata": {
                            "method": "audio_agent_feature_extraction",
                            "language": language
                        }
                    }
                else:
                    return {
                        "error": "Feature extraction failed",
                        "audio_path": audio_path,
                        "features": {},
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error extracting audio features: {e}")
                return {
                    "error": str(e),
                    "audio_path": audio_path,
                    "features": {},
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Comprehensive audio analysis using AudioAgent"
        )
        async def comprehensive_audio_analysis(
            audio_path: str = Field(..., description="Path or URL to audio file"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Perform comprehensive audio analysis including transcription, sentiment, and features."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Create analysis request
                request = AnalysisRequest(
                    content=audio_path,
                    data_type=DataType.AUDIO,
                    language=language,
                    confidence_threshold=confidence_threshold
                )
                
                # Process with audio agent for comprehensive analysis
                result = await self.audio_agent.process(request)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                return {
                    "audio_path": audio_path,
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "scores": result.sentiment.scores,
                    "transcription": result.extracted_text,
                    "analysis_time": analysis_time,
                    "status": "success",
                    "metadata": {
                        "agent_id": result.agent_id,
                        "processing_time": result.processing_time,
                        "status": result.status or "completed",
                        "method": "audio_agent_comprehensive_analysis",
                        "language": language,
                        "tools_used": result.metadata.get("tools_used", [])
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in comprehensive audio analysis: {e}")
                return {
                    "error": str(e),
                    "audio_path": audio_path,
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {"neutral": 1.0},
                    "transcription": "",
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Fallback audio analysis using AudioAgent"
        )
        async def fallback_audio_analysis(
            audio_path: str = Field(..., description="Path or URL to audio file"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Use fallback audio analysis when primary method fails."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the audio agent's fallback tool directly
                fallback_result = await self.audio_agent.fallback_audio_analysis(audio_path)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if fallback_result.get("status") == "success":
                    sentiment_data = fallback_result["content"][0].get("json", {})
                    return {
                        "audio_path": audio_path,
                        "sentiment": sentiment_data.get("sentiment", "neutral"),
                        "confidence": sentiment_data.get("confidence", 0.0),
                        "analysis_time": analysis_time,
                        "method": "fallback",
                        "status": "success",
                        "metadata": {
                            "method": "audio_agent_fallback_analysis",
                            "language": language,
                            "analysis": sentiment_data.get("analysis", "")
                        }
                    }
                else:
                    return {
                        "error": "Fallback analysis failed",
                        "audio_path": audio_path,
                        "sentiment": "neutral",
                        "confidence": 0.0,
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error in fallback audio analysis: {e}")
                return {
                    "error": str(e),
                    "audio_path": audio_path,
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Batch analyze multiple audio files using AudioAgent"
        )
        async def batch_analyze_audio(
            audio_paths: List[str] = Field(..., description="List of audio file paths to analyze"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> List[Dict[str, Any]]:
            """Analyze sentiment for multiple audio files in batch using AudioAgent."""
            try:
                results = []
                for audio_path in audio_paths:
                    # Use comprehensive analysis for each audio file
                    result = await comprehensive_audio_analysis(audio_path, language, confidence_threshold)
                    results.append(result)
                return results
            except Exception as e:
                logger.error(f"Error in batch audio analysis: {e}")
                return [{"error": str(e), "audio_path": audio_path} for audio_path in audio_paths]
        
        @self.mcp.tool(
            description="Get AudioAgent capabilities and configuration"
        )
        def get_audio_agent_capabilities() -> Dict[str, Any]:
            """Get information about AudioAgent capabilities and configuration."""
            return {
                "agent_id": self.audio_agent.agent_id,
                "model": self.audio_agent.metadata.get("model", "default"),
                "supported_formats": self.audio_agent.metadata.get("supported_formats", []),
                "max_audio_duration": self.audio_agent.metadata.get("max_audio_duration", 300),
                "capabilities": self.audio_agent.metadata.get("capabilities", ["audio", "transcription", "sentiment_analysis"]),
                "available_tools": [
                    "transcribe_audio",
                    "analyze_audio_sentiment",
                    "extract_audio_features",
                    "comprehensive_audio_analysis",
                    "fallback_audio_analysis",
                    "batch_analyze_audio"
                ],
                "features": [
                    "audio transcription",
                    "sentiment classification",
                    "confidence scoring",
                    "audio feature extraction",
                    "fallback analysis",
                    "batch processing",
                    "multi-format support"
                ]
            }
    
    def run(self, host: str = "localhost", port: int = 8007, debug: bool = False):
        """Run the AudioAgent MCP server."""
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        logger.info(f"Starting AudioAgent MCP server on {host}:{port}")
        
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


def create_audio_agent_mcp_server(model_name: Optional[str] = None) -> AudioAgentMCPServer:
    """Factory function to create an AudioAgent MCP server."""
    return AudioAgentMCPServer(model_name=model_name)


if __name__ == "__main__":
    # Run the server directly
    server = create_audio_agent_mcp_server()
    server.run(host="0.0.0.0", port=8007, debug=True)
