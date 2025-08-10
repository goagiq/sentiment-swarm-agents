#!/usr/bin/env python3
"""
Enhanced MCP server for AudioAgent - provides comprehensive audio analysis tools
including transcription, sentiment analysis, feature extraction, quality assessment,
and emotion analysis.
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
from agents.audio_agent_enhanced import EnhancedAudioAgent


class EnhancedAudioAnalysisRequest(BaseModel):
    """Enhanced request model for audio analysis."""
    audio_path: str = Field(..., description="Path or URL to audio file")
    content_type: str = Field(
        default="audio", 
        description="Type of content: audio, voice, music, podcast"
    )
    language: str = Field(default="en", description="Language code")
    confidence_threshold: float = Field(
        default=0.8, 
        description="Minimum confidence threshold"
    )
    analysis_type: str = Field(
        default="comprehensive", 
        description="Type of analysis: transcription, sentiment, features, quality, emotion, or comprehensive"
    )
    enhanced_features: bool = Field(
        default=True,
        description="Enable enhanced audio analysis features"
    )


class EnhancedAudioAnalysisResponse(BaseModel):
    """Enhanced response model for audio analysis."""
    audio_path: str = Field(..., description="Analyzed audio file path")
    content_type: str = Field(..., description="Type of content analyzed")
    sentiment: str = Field(..., description="Detected sentiment: positive, negative, neutral")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    scores: Dict[str, float] = Field(..., description="Detailed sentiment scores")
    transcription: Optional[str] = Field(None, description="Audio transcription text")
    features: Optional[Dict[str, Any]] = Field(None, description="Extracted audio features")
    quality_analysis: Optional[Dict[str, Any]] = Field(None, description="Audio quality assessment")
    emotion_analysis: Optional[Dict[str, Any]] = Field(None, description="Emotional content analysis")
    analysis_time: float = Field(..., description="Analysis time in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata"
    )


class EnhancedAudioAgentMCPServer:
    """Enhanced MCP server providing comprehensive audio analysis tools from EnhancedAudioAgent."""
    
    def __init__(self, model_name: Optional[str] = None):
        # Initialize the enhanced audio agent
        self.audio_agent = EnhancedAudioAgent(model_name=model_name)
        
        # Note: FastMCP will be imported when available
        # For now, we'll create a mock structure
        self.mcp = None
        self._initialize_mcp()
        
        # Register enhanced tools
        self._register_enhanced_tools()
        
        model_name = self.audio_agent.metadata.get('model', 'default')
        logger.info(f"Enhanced AudioAgent MCP Server initialized with model: {model_name}")
    
    def _initialize_mcp(self):
        """Initialize the MCP server."""
        try:
            from fastmcp import FastMCP
            self.mcp = FastMCP("Enhanced AudioAgent Server")
            logger.info("FastMCP initialized successfully")
        except ImportError:
            logger.warning("FastMCP not available, using mock MCP server")
            # Create a mock MCP server for development
            self.mcp = MockMCPServer("Enhanced AudioAgent Server")
    
    def _register_enhanced_tools(self):
        """Register all enhanced audio analysis tools from EnhancedAudioAgent."""
        
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        @self.mcp.tool(
            description="Enhanced audio transcription using EnhancedAudioAgent"
        )
        async def transcribe_audio_enhanced(
            audio_path: str = Field(..., description="Path or URL to audio file"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Enhanced audio transcription with improved accuracy and error handling."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the enhanced audio agent's transcription tool
                transcription_result = await self.audio_agent.transcribe_audio_enhanced(audio_path)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if transcription_result.get("status") == "success":
                    transcription = transcription_result["content"][0].get("text", "")
                    return {
                        "audio_path": audio_path,
                        "transcription": transcription,
                        "analysis_time": analysis_time,
                        "status": "success",
                        "method": "enhanced_transcription",
                        "metadata": {
                            "method": "enhanced_audio_transcription",
                            "language": language,
                            "agent_id": self.audio_agent.agent_id,
                            "enhanced_features": True
                        }
                    }
                else:
                    return {
                        "error": "Enhanced transcription failed",
                        "audio_path": audio_path,
                        "transcription": "",
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error in enhanced audio transcription: {e}")
                return {
                    "error": str(e),
                    "audio_path": audio_path,
                    "transcription": "",
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Enhanced audio sentiment analysis using EnhancedAudioAgent"
        )
        async def analyze_audio_sentiment_enhanced(
            audio_path: str = Field(..., description="Path or URL to audio file"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Enhanced audio sentiment analysis with improved accuracy and emotional insights."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the enhanced audio agent's sentiment analysis tool
                sentiment_result = await self.audio_agent.analyze_audio_sentiment_enhanced(audio_path)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if sentiment_result.get("status") == "success":
                    sentiment_data = sentiment_result["content"][0].get("json", {})
                    return {
                        "audio_path": audio_path,
                        "sentiment": sentiment_data.get("sentiment", "neutral"),
                        "confidence": sentiment_data.get("confidence", 0.0),
                        "transcription": sentiment_data.get("transcription", ""),
                        "scores": sentiment_data.get("scores", {}),
                        "emotions": sentiment_data.get("emotions", {}),
                        "analysis_time": analysis_time,
                        "status": "success",
                        "method": "enhanced_sentiment_analysis",
                        "metadata": {
                            "method": "enhanced_audio_sentiment_analysis",
                            "language": language,
                            "agent_id": self.audio_agent.agent_id,
                            "raw_response": sentiment_data.get("raw_response", ""),
                            "enhanced_features": True
                        }
                    }
                else:
                    return {
                        "error": "Enhanced sentiment analysis failed",
                        "audio_path": audio_path,
                        "sentiment": "neutral",
                        "confidence": 0.0,
                        "transcription": "",
                        "scores": {"neutral": 1.0},
                        "emotions": {},
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error in enhanced audio sentiment analysis: {e}")
                return {
                    "error": str(e),
                    "audio_path": audio_path,
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "transcription": "",
                    "scores": {"neutral": 1.0},
                    "emotions": {},
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Enhanced audio feature extraction using EnhancedAudioAgent"
        )
        async def extract_audio_features_enhanced(
            audio_path: str = Field(..., description="Path or URL to audio file"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Extract comprehensive audio features for analysis using EnhancedAudioAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the enhanced audio agent's feature extraction tool
                features_result = await self.audio_agent.extract_audio_features_enhanced(audio_path)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if features_result.get("status") == "success":
                    features = features_result["content"][0].get("json", {})
                    return {
                        "audio_path": audio_path,
                        "features": features,
                        "analysis_time": analysis_time,
                        "status": "success",
                        "method": "enhanced_feature_extraction",
                        "metadata": {
                            "method": "enhanced_audio_feature_extraction",
                            "language": language,
                            "enhanced_features": True
                        }
                    }
                else:
                    return {
                        "error": "Enhanced feature extraction failed",
                        "audio_path": audio_path,
                        "features": {},
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error in enhanced audio feature extraction: {e}")
                return {
                    "error": str(e),
                    "audio_path": audio_path,
                    "features": {},
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Audio quality assessment using EnhancedAudioAgent"
        )
        async def analyze_audio_quality(
            audio_path: str = Field(..., description="Path or URL to audio file"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Assess audio quality and characteristics using EnhancedAudioAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the enhanced audio agent's quality analysis tool
                quality_result = await self.audio_agent.analyze_audio_quality(audio_path)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if quality_result.get("status") == "success":
                    quality_data = quality_result["content"][0].get("json", {})
                    return {
                        "audio_path": audio_path,
                        "quality_analysis": quality_data,
                        "analysis_time": analysis_time,
                        "status": "success",
                        "method": "enhanced_quality_assessment",
                        "metadata": {
                            "method": "enhanced_audio_quality_assessment",
                            "language": language,
                            "enhanced_features": True
                        }
                    }
                else:
                    return {
                        "error": "Audio quality analysis failed",
                        "audio_path": audio_path,
                        "quality_analysis": {},
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error in audio quality analysis: {e}")
                return {
                    "error": str(e),
                    "audio_path": audio_path,
                    "quality_analysis": {},
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Audio emotion analysis using EnhancedAudioAgent"
        )
        async def analyze_audio_emotion(
            audio_path: str = Field(..., description="Path or URL to audio file"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Analyze emotional content in audio using EnhancedAudioAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the enhanced audio agent's emotion analysis tool
                emotion_result = await self.audio_agent.analyze_audio_emotion(audio_path)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if emotion_result.get("status") == "success":
                    emotion_data = emotion_result["content"][0].get("json", {})
                    return {
                        "audio_path": audio_path,
                        "emotion_analysis": emotion_data,
                        "analysis_time": analysis_time,
                        "status": "success",
                        "method": "enhanced_emotion_analysis",
                        "metadata": {
                            "method": "enhanced_audio_emotion_analysis",
                            "language": language,
                            "enhanced_features": True
                        }
                    }
                else:
                    return {
                        "error": "Audio emotion analysis failed",
                        "audio_path": audio_path,
                        "emotion_analysis": {},
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error in audio emotion analysis: {e}")
                return {
                    "error": str(e),
                    "audio_path": audio_path,
                    "emotion_analysis": {},
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Comprehensive enhanced audio analysis using EnhancedAudioAgent"
        )
        async def comprehensive_enhanced_audio_analysis(
            audio_path: str = Field(..., description="Path or URL to audio file"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> Dict[str, Any]:
            """Perform comprehensive enhanced audio analysis including transcription, sentiment, features, quality, and emotion."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Create analysis request
                request = AnalysisRequest(
                    content=audio_path,
                    data_type=DataType.AUDIO,
                    language=language,
                    confidence_threshold=confidence_threshold
                )
                
                # Process with enhanced audio agent for comprehensive analysis
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
                    "method": "enhanced_comprehensive_analysis",
                    "metadata": {
                        "agent_id": result.agent_id,
                        "processing_time": result.processing_time,
                        "status": result.status or "completed",
                        "method": "enhanced_audio_comprehensive_analysis",
                        "language": language,
                        "tools_used": result.metadata.get("tools_used", []),
                        "enhanced_features": True
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in comprehensive enhanced audio analysis: {e}")
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
            description="Enhanced fallback audio analysis using EnhancedAudioAgent"
        )
        async def fallback_audio_analysis_enhanced(
            audio_path: str = Field(..., description="Path or URL to audio file"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Use enhanced fallback audio analysis when primary methods fail."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the enhanced audio agent's fallback tool
                fallback_result = await self.audio_agent.fallback_audio_analysis_enhanced(audio_path)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if fallback_result.get("status") == "success":
                    sentiment_data = fallback_result["content"][0].get("json", {})
                    return {
                        "audio_path": audio_path,
                        "sentiment": sentiment_data.get("sentiment", "neutral"),
                        "confidence": sentiment_data.get("confidence", 0.0),
                        "analysis_time": analysis_time,
                        "method": "enhanced_fallback",
                        "status": "success",
                        "metadata": {
                            "method": "enhanced_audio_fallback_analysis",
                            "language": language,
                            "analysis": sentiment_data.get("analysis", ""),
                            "enhanced_features": True
                        }
                    }
                else:
                    return {
                        "error": "Enhanced fallback analysis failed",
                        "audio_path": audio_path,
                        "sentiment": "neutral",
                        "confidence": 0.0,
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error in enhanced fallback audio analysis: {e}")
                return {
                    "error": str(e),
                    "audio_path": audio_path,
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Enhanced batch analyze multiple audio files using EnhancedAudioAgent"
        )
        async def batch_analyze_audio_enhanced(
            audio_paths: List[str] = Field(..., description="List of audio file paths to analyze"),
            language: str = Field(default="en", description="Language code"),
            confidence_threshold: float = Field(
                default=0.8, 
                description="Minimum confidence threshold"
            )
        ) -> List[Dict[str, Any]]:
            """Analyze sentiment for multiple audio files in batch using EnhancedAudioAgent."""
            try:
                results = []
                for audio_path in audio_paths:
                    # Use comprehensive enhanced analysis for each audio file
                    result = await comprehensive_enhanced_audio_analysis(
                        audio_path, language, confidence_threshold
                    )
                    results.append(result)
                return results
            except Exception as e:
                logger.error(f"Error in enhanced batch audio analysis: {e}")
                return [{"error": str(e), "audio_path": audio_path} for audio_path in audio_paths]
        
        @self.mcp.tool(
            description="Validate audio format using EnhancedAudioAgent"
        )
        async def validate_audio_format(
            audio_path: str = Field(..., description="Path or URL to audio file"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Validate audio format and compatibility using EnhancedAudioAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the enhanced audio agent's validation tool
                validation_result = await self.audio_agent.validate_audio_format(audio_path)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if validation_result.get("status") == "success":
                    validation_data = validation_result["content"][0].get("json", {})
                    return {
                        "audio_path": audio_path,
                        "validation": validation_data,
                        "analysis_time": analysis_time,
                        "status": "success",
                        "method": "enhanced_format_validation",
                        "metadata": {
                            "method": "enhanced_audio_format_validation",
                            "language": language,
                            "enhanced_features": True
                        }
                    }
                else:
                    return {
                        "error": "Audio format validation failed",
                        "audio_path": audio_path,
                        "validation": {},
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error in audio format validation: {e}")
                return {
                    "error": str(e),
                    "audio_path": audio_path,
                    "validation": {},
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Get audio metadata using EnhancedAudioAgent"
        )
        async def get_audio_metadata(
            audio_path: str = Field(..., description="Path or URL to audio file"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Extract comprehensive audio metadata using EnhancedAudioAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the enhanced audio agent's metadata tool
                metadata_result = await self.audio_agent.get_audio_metadata(audio_path)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if metadata_result.get("status") == "success":
                    metadata_data = metadata_result["content"][0].get("json", {})
                    return {
                        "audio_path": audio_path,
                        "metadata": metadata_data,
                        "analysis_time": analysis_time,
                        "status": "success",
                        "method": "enhanced_metadata_extraction",
                        "metadata": {
                            "method": "enhanced_audio_metadata_extraction",
                            "language": language,
                            "enhanced_features": True
                        }
                    }
                else:
                    return {
                        "error": "Audio metadata extraction failed",
                        "audio_path": audio_path,
                        "metadata": {},
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error in audio metadata extraction: {e}")
                return {
                    "error": str(e),
                    "audio_path": audio_path,
                    "metadata": {},
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Process audio stream using EnhancedAudioAgent"
        )
        async def process_audio_stream(
            audio_url: str = Field(..., description="URL of streaming audio content"),
            language: str = Field(default="en", description="Language code")
        ) -> Dict[str, Any]:
            """Process streaming audio content using EnhancedAudioAgent."""
            try:
                start_time = asyncio.get_event_loop().time()
                
                # Use the enhanced audio agent's stream processing tool
                stream_result = await self.audio_agent.process_audio_stream(audio_url)
                
                analysis_time = asyncio.get_event_loop().time() - start_time
                
                if stream_result.get("status") == "success":
                    stream_data = stream_result["content"][0].get("json", {})
                    return {
                        "audio_url": audio_url,
                        "stream_analysis": stream_data,
                        "analysis_time": analysis_time,
                        "status": "success",
                        "method": "enhanced_stream_processing",
                        "metadata": {
                            "method": "enhanced_audio_stream_processing",
                            "language": language,
                            "enhanced_features": True
                        }
                    }
                else:
                    return {
                        "error": "Audio stream processing failed",
                        "audio_url": audio_url,
                        "stream_analysis": {},
                        "analysis_time": analysis_time,
                        "status": "failed"
                    }
                    
            except Exception as e:
                logger.error(f"Error in audio stream processing: {e}")
                return {
                    "error": str(e),
                    "audio_url": audio_url,
                    "stream_analysis": {},
                    "analysis_time": 0.0,
                    "status": "failed"
                }
        
        @self.mcp.tool(
            description="Get EnhancedAudioAgent capabilities and configuration"
        )
        def get_enhanced_audio_agent_capabilities() -> Dict[str, Any]:
            """Get information about EnhancedAudioAgent capabilities and configuration."""
            return {
                "agent_id": self.audio_agent.agent_id,
                "model": self.audio_agent.metadata.get("model", "default"),
                "supported_formats": self.audio_agent.metadata.get("supported_formats", []),
                "max_audio_duration": self.audio_agent.metadata.get("max_audio_duration", 300),
                "capabilities": self.audio_agent.metadata.get("capabilities", []),
                "available_tools": [
                    "transcribe_audio_enhanced",
                    "analyze_audio_sentiment_enhanced",
                    "extract_audio_features_enhanced",
                    "analyze_audio_quality",
                    "analyze_audio_emotion",
                    "comprehensive_enhanced_audio_analysis",
                    "fallback_audio_analysis_enhanced",
                    "batch_analyze_audio_enhanced",
                    "validate_audio_format",
                    "get_audio_metadata",
                    "process_audio_stream"
                ],
                "enhanced_features": [
                    "enhanced audio transcription",
                    "enhanced sentiment classification",
                    "comprehensive feature extraction",
                    "audio quality assessment",
                    "emotion analysis",
                    "format validation",
                    "metadata extraction",
                    "stream processing",
                    "batch processing",
                    "multi-format support",
                    "enhanced error handling"
                ],
                "enhanced_capabilities": True
            }
    
    def run(self, host: str = "localhost", port: int = 8008, debug: bool = False):
        """Run the Enhanced AudioAgent MCP server."""
        if self.mcp is None:
            logger.error("MCP server not initialized")
            return
        
        logger.info(f"Starting Enhanced AudioAgent MCP server on {host}:{port}")
        
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
    
    def run(self, host: str = "localhost", port: int = 8008, debug: bool = False):
        """Mock run method."""
        logger.info(f"Mock MCP server '{self.name}' would run on {host}:{port}")
        logger.info(f"Available tools: {list(self.tools.keys())}")


def create_enhanced_audio_agent_mcp_server(model_name: Optional[str] = None) -> EnhancedAudioAgentMCPServer:
    """Factory function to create an Enhanced AudioAgent MCP server."""
    return EnhancedAudioAgentMCPServer(model_name=model_name)


if __name__ == "__main__":
    # Run the server directly
    server = create_enhanced_audio_agent_mcp_server()
    server.run(host="0.0.0.0", port=8008, debug=True)
