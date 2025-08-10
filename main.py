#!/usr/bin/env python3
"""
Main entry point for the Sentiment Analysis Swarm system.
Provides both MCP server and FastAPI server functionality.
"""

# Suppress websockets deprecation warnings BEFORE any other imports
import warnings
import sys

# Set warnings filter to ignore all websockets-related deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets.legacy")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets.server")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="uvicorn.protocols.websockets")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="uvicorn")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*websockets.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*WebSocketServerProtocol.*")

# Custom warning filter function
def ignore_websockets_warnings(message, category, filename, lineno, file=None, line=None):
    """Custom warning filter to ignore websockets-related deprecation warnings."""
    if category == DeprecationWarning:
        if any(keyword in str(message).lower() for keyword in ['websockets', 'websocket']):
            return True
    return False

# Add custom filter
warnings.showwarning = ignore_websockets_warnings

import os
import threading
import uvicorn
from typing import List, Dict, Any

# Import MCP server before adding src to path to avoid conflicts
try:
    from fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP server not available")

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import after path modification
# flake8: noqa: E402
from api.main import app
from core.error_handler import with_error_handling
from agents.text_agent import TextAgent
from agents.text_agent_simple import SimpleTextAgent
from agents.text_agent_strands import TextAgentStrands
from agents.text_agent_swarm import TextAgentSwarm
from agents.audio_agent_enhanced import EnhancedAudioAgent
from agents.vision_agent_enhanced import EnhancedVisionAgent
from agents.web_agent_enhanced import EnhancedWebAgent
from agents.audio_summarization_agent import AudioSummarizationAgent
from agents.video_summarization_agent import VideoSummarizationAgent
from agents.ocr_agent import OCRAgent
from agents.orchestrator_agent import OrchestratorAgent, unified_video_analysis
from agents.translation_agent import TranslationAgent
from core.youtube_comprehensive_analyzer import YouTubeComprehensiveAnalyzer
from core.models import (
    AnalysisRequest, 
    DataType,
    ProcessingStatus,
    SentimentResult
)


class UnifiedMCPServer:
    """Unified MCP server providing access to all 7 agents as tools with streamable HTTP support."""
    
    def __init__(self):
        # Initialize the MCP server with proper streamable HTTP support
        self.mcp = None
        self._initialize_mcp()
        
        # Initialize all 7 agents
        self.agents = {}
        self._initialize_agents()
        
        # Register all tools
        self._register_tools()
        
        print("✅ Unified MCP Server initialized with all 7 agents as tools")
    
    def _initialize_mcp(self):
        """Initialize the MCP server using FastMCP with streamable HTTP support."""
        if MCP_AVAILABLE:
            self.mcp = FastMCP("Sentiment Analysis Agents Server")
            print("✅ FastMCP Server with streamable HTTP support initialized successfully")
        else:
            print("⚠️  FastMCP not available - skipping MCP server initialization")
            self.mcp = None
    
    def _initialize_agents(self):
        """Initialize all 7 agents."""
        try:
            # Initialize each agent type
            self.agents["text"] = TextAgent()
            self.agents["text_simple"] = SimpleTextAgent()
            self.agents["text_strands"] = TextAgentStrands()
            self.agents["text_swarm"] = TextAgentSwarm()
            self.agents["audio"] = EnhancedAudioAgent()
            self.agents["vision"] = EnhancedVisionAgent()
            self.agents["web"] = EnhancedWebAgent()
            self.agents["audio_summary"] = AudioSummarizationAgent()
            self.agents["video_summary"] = VideoSummarizationAgent()
            self.agents["ocr"] = OCRAgent()
            self.agents["orchestrator"] = OrchestratorAgent()
            self.agents["translation"] = TranslationAgent()
            self.agents["youtube"] = YouTubeComprehensiveAnalyzer()
            
            print(f"✅ Initialized {len(self.agents)} agents including translation")
            
        except Exception as e:
            print(f"⚠️  Error initializing agents: {e}")
    
    def _register_tools(self):
        """Register all tools from all agents using FastMCP pattern."""
        
        if self.mcp is None:
            print("❌ MCP server not initialized")
            return
        
        try:
            # Register agent status and management tools
            @self.mcp.tool(description="Get status of all available agents")
            def get_all_agents_status():
                """Get status of all available agents."""
                try:
                    status = {}
                    for agent_name, agent in self.agents.items():
                        if hasattr(agent, 'get_status'):
                            status[agent_name] = agent.get_status()
                        else:
                            status[agent_name] = {
                                "agent_id": getattr(agent, 'agent_id', f"{agent_name}_agent"),
                                "status": "active",
                                "type": agent.__class__.__name__
                            }
                    
                    return {
                        "success": True,
                        "total_agents": len(self.agents),
                        "agents": status
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            @self.mcp.tool(description="Start all agents")
            async def start_all_agents():
                """Start all agents."""
                try:
                    results = {}
                    for agent_name, agent in self.agents.items():
                        try:
                            if hasattr(agent, 'start'):
                                await agent.start()
                                results[agent_name] = {"success": True, "message": "Started"}
                            else:
                                results[agent_name] = {"success": True, "message": "No start method needed"}
                        except Exception as e:
                            results[agent_name] = {"success": False, "error": str(e)}
                    
                    return {
                        "success": True,
                        "results": results
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            @self.mcp.tool(description="Stop all agents")
            async def stop_all_agents():
                """Stop all agents."""
                try:
                    results = {}
                    for agent_name, agent in self.agents.items():
                        try:
                            if hasattr(agent, 'stop'):
                                await agent.stop()
                                results[agent_name] = {"success": True, "message": "Stopped"}
                            else:
                                results[agent_name] = {"success": True, "message": "No stop method needed"}
                        except Exception as e:
                            results[agent_name] = {"success": False, "error": str(e)}
                    
                    return {
                        "success": True,
                        "results": results
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            # Register text analysis tools
            @self.mcp.tool(description="Analyze text sentiment using TextAgent")
            @with_error_handling("text_sentiment_analysis")
            async def analyze_text_sentiment(text: str, language: str = "en"):
                """Analyze text sentiment using TextAgent."""
                analysis_request = AnalysisRequest(
                    data_type=DataType.TEXT,
                    content=text,
                    language=language
                )
                
                result = await self.agents["text"].process(analysis_request)
                
                return {
                    "success": True,
                    "agent": "text",
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "processing_time": result.processing_time
                }
            
            @self.mcp.tool(description="Analyze text sentiment using SimpleTextAgent")
            async def analyze_text_simple(text: str, language: str = "en"):
                """Analyze text sentiment using SimpleTextAgent."""
                try:
                    analysis_request = AnalysisRequest(
                        data_type=DataType.TEXT,
                        content=text,
                        language=language
                    )
                    
                    result = await self.agents["text_simple"].process(analysis_request)
                    
                    return {
                        "success": True,
                        "agent": "text_simple",
                        "sentiment": result.sentiment.label,
                        "confidence": result.sentiment.confidence,
                        "processing_time": result.processing_time
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            @self.mcp.tool(description="Analyze text sentiment using TextAgentStrands")
            async def analyze_text_strands(text: str, language: str = "en"):
                """Analyze text sentiment using TextAgentStrands."""
                try:
                    analysis_request = AnalysisRequest(
                        data_type=DataType.TEXT,
                        content=text,
                        language=language
                    )
                    
                    result = await self.agents["text_strands"].process(analysis_request)
                    
                    return {
                        "success": True,
                        "agent": "text_strands",
                        "sentiment": result.sentiment.label,
                        "confidence": result.sentiment.confidence,
                        "processing_time": result.processing_time
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            @self.mcp.tool(description="Analyze text sentiment using TextAgentSwarm")
            async def analyze_text_swarm(text: str, language: str = "en"):
                """Analyze text sentiment using TextAgentSwarm."""
                try:
                    analysis_request = AnalysisRequest(
                        data_type=DataType.TEXT,
                        content=text,
                        language=language
                    )
                    
                    result = await self.agents["text_swarm"].process(analysis_request)
                    
                    return {
                        "success": True,
                        "agent": "text_swarm",
                        "sentiment": result.sentiment.label,
                        "confidence": result.sentiment.confidence,
                        "processing_time": result.processing_time
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            # Register audio analysis tools
            @self.mcp.tool(description="Analyze audio sentiment using EnhancedAudioAgent")
            async def analyze_audio_sentiment(audio_path: str, language: str = "en"):
                """Analyze audio sentiment using EnhancedAudioAgent."""
                try:
                    # Validate file existence
                    if not os.path.exists(audio_path):
                        return {
                            "success": False,
                            "error": f"Audio file not found: {audio_path}",
                            "suggestion": "Please check the file path and ensure the file exists"
                        }
                    
                    analysis_request = AnalysisRequest(
                        data_type=DataType.AUDIO,
                        content=audio_path,
                        language=language
                    )
                    
                    result = await self.agents["audio"].process(analysis_request)
                    
                    # Check if analysis was successful
                    if result.status == "completed" or result.status is None:
                        return {
                            "success": True,
                            "agent": "enhanced_audio",
                            "sentiment": result.sentiment.label,
                            "confidence": result.sentiment.confidence,
                            "processing_time": result.processing_time,
                            "metadata": result.metadata if result.metadata else {},
                            "extracted_text": result.extracted_text if result.extracted_text else None
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Analysis failed with status: {result.status}",
                            "metadata": result.metadata if result.metadata else {}
                        }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "suggestion": "Check if audio file is in supported format (MP3, WAV, FLAC, etc.)"
                    }
            
            # Register vision analysis tools
            @self.mcp.tool(description="Analyze image sentiment using EnhancedVisionAgent")
            async def analyze_image_sentiment(image_path: str, language: str = "en"):
                """Analyze image sentiment using EnhancedVisionAgent."""
                try:
                    # Validate file existence
                    if not os.path.exists(image_path):
                        return {
                            "success": False,
                            "error": f"Image file not found: {image_path}",
                            "suggestion": "Please check the file path and ensure the file exists"
                        }
                    
                    analysis_request = AnalysisRequest(
                        data_type=DataType.IMAGE,
                        content=image_path,
                        language=language
                    )
                    
                    result = await self.agents["vision"].process(analysis_request)
                    
                    # Check if analysis was successful
                    if result.status == "completed" or result.status is None:
                        return {
                            "success": True,
                            "agent": "enhanced_vision",
                            "sentiment": result.sentiment.label,
                            "confidence": result.sentiment.confidence,
                            "processing_time": result.processing_time,
                            "metadata": result.metadata if result.metadata else {},
                            "extracted_text": result.extracted_text if result.extracted_text else None
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Analysis failed with status: {result.status}",
                            "metadata": result.metadata if result.metadata else {}
                        }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "suggestion": "Check if image file is in supported format (JPG, PNG, etc.)"
                    }
            
            # Register web analysis tools
            @self.mcp.tool(description="Analyze webpage sentiment using EnhancedWebAgent")
            async def analyze_webpage_sentiment(url: str, language: str = "en"):
                """Analyze webpage sentiment using EnhancedWebAgent."""
                try:
                    analysis_request = AnalysisRequest(
                        data_type=DataType.WEBPAGE,
                        content=url,
                        language=language
                    )
                    
                    result = await self.agents["web"].process(analysis_request)
                    
                    return {
                        "success": True,
                        "agent": "enhanced_web",
                        "sentiment": result.sentiment.label,
                        "confidence": result.sentiment.confidence,
                        "processing_time": result.processing_time
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            # Register orchestrator tools
            @self.mcp.tool(description="Process query using OrchestratorAgent")
            async def process_query_orchestrator(query: str):
                """Process query using OrchestratorAgent."""
                try:
                    result = await self.agents["orchestrator"].process_query(query)
                    
                    return {
                        "success": True,
                        "agent": "orchestrator",
                        "result": result
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            @self.mcp.tool(description="Get available tools from OrchestratorAgent")
            async def get_orchestrator_tools():
                """Get available tools from OrchestratorAgent."""
                try:
                    tools = await self.agents["orchestrator"].get_available_tools()
                    
                    return {
                        "success": True,
                        "agent": "orchestrator",
                        "tools": tools
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            # Register YouTube comprehensive analysis tool
            @self.mcp.tool(description="Analyze YouTube video comprehensively with audio and visual sentiment")
            async def analyze_youtube_comprehensive(youtube_url: str):
                """Analyze a YouTube video comprehensively with audio and visual sentiment."""
                try:
                    result = await self.agents["youtube"].analyze_youtube_video(
                        youtube_url,
                        extract_audio=True,
                        extract_frames=True,
                        num_frames=5
                    )
                    return {
                        "success": True,
                        "agent": "youtube_comprehensive",
                        "result": {
                            "video_title": result.video_metadata.get("title", "Unknown"),
                            "duration": result.video_metadata.get("duration", 0),
                            "views": result.video_metadata.get("view_count", 0),
                            "likes": result.video_metadata.get("like_count", 0),
                            "combined_sentiment": result.combined_sentiment.label,
                            "confidence": result.combined_sentiment.confidence,
                            "audio_sentiment": result.audio_sentiment.label,
                            "visual_sentiment": result.visual_sentiment.label,
                            "frames_analyzed": len(result.extracted_frames),
                            "processing_time": result.processing_time,
                            "uploader": result.video_metadata.get("uploader", "Unknown")
                        }
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "agent": "youtube_comprehensive",
                        "result": {
                            "status": "error",
                            "content": [{"text": f"YouTube analysis error: {str(e)}"}]
                        }
                    }
            
            # Register enhanced audio summarization tool
            @self.mcp.tool(description="Generate comprehensive audio summary with key points, action items, sentiment analysis, and detailed metadata")
            async def analyze_audio_summarization(audio_path: str, language: str = "en"):
                """Generate comprehensive audio summary with key points, action items, sentiment analysis, and detailed metadata."""
                try:
                    # Validate file existence
                    if not os.path.exists(audio_path):
                        return {
                            "success": False,
                            "error": f"Audio file not found: {audio_path}",
                            "suggestion": "Please check the file path and ensure the file exists"
                        }
                    
                    print(f"🎵 Starting enhanced audio analysis: {audio_path}")
                    
                    analysis_request = AnalysisRequest(
                        data_type=DataType.AUDIO,
                        content=audio_path,
                        language=language
                    )
                    
                    result = await self.agents["audio_summary"].process(analysis_request)
                    
                    # Check if analysis was successful
                    if result.status == "completed" or result.status is None:
                        # Enhanced response with detailed analysis information
                        response_data = {
                            "success": True,
                            "agent": "audio_summarization",
                            "sentiment": {
                                "label": result.sentiment.label,
                                "confidence": result.sentiment.confidence
                            },
                            "processing_time": result.processing_time,
                            "analysis_details": {
                                "key_points_count": result.metadata.get("key_points_count", 0) if result.metadata else 0,
                                "action_items_count": result.metadata.get("action_items_count", 0) if result.metadata else 0,
                                "topics_identified": result.metadata.get("topics_identified", 0) if result.metadata else 0,
                                "summary_length": result.metadata.get("summary_length", 0) if result.metadata else 0,
                                "model_used": result.metadata.get("model_used", "unknown") if result.metadata else "unknown",
                                "audio_duration": result.metadata.get("audio_duration", "unknown") if result.metadata else "unknown",
                                "speakers_detected": result.metadata.get("speakers_detected", 0) if result.metadata else 0,
                                "language_detected": result.metadata.get("language_detected", language) if result.metadata else language
                            },
                            "content_summary": result.metadata.get("summary", "") if result.metadata else "",
                            "key_points": result.metadata.get("key_points", []) if result.metadata else [],
                            "action_items": result.metadata.get("action_items", []) if result.metadata else [],
                            "topics": result.metadata.get("topics", []) if result.metadata else [],
                            "executive_summary": result.metadata.get("executive_summary", "") if result.metadata else "",
                            "meeting_minutes": result.metadata.get("meeting_minutes", "") if result.metadata else "",
                            "quotes": result.metadata.get("quotes", []) if result.metadata else [],
                            "emotions": result.metadata.get("emotions", []) if result.metadata else [],
                            "timeline": result.metadata.get("timeline", "") if result.metadata else "",
                            "transcript": result.metadata.get("transcript", "") if result.metadata else "",
                            "speaker_analysis": result.metadata.get("speaker_analysis", {}) if result.metadata else {},
                            "audio_quality_metrics": result.metadata.get("audio_quality_metrics", {}) if result.metadata else {},
                            "content_classification": result.metadata.get("content_classification", {}) if result.metadata else {},
                            "recommendations": result.metadata.get("recommendations", []) if result.metadata else []
                        }
                        
                        print(f"✅ Audio analysis completed in {result.processing_time:.2f} seconds")
                        print(f"📊 Sentiment: {result.sentiment.label} (confidence: {result.sentiment.confidence:.2f})")
                        
                        return response_data
                    else:
                        return {
                            "success": False,
                            "error": f"Analysis failed with status: {result.status}",
                            "metadata": result.metadata if result.metadata else {},
                            "suggestion": "Audio analysis did not complete successfully"
                        }
                    
                except Exception as e:
                    print(f"❌ Audio summarization failed: {e}")
                    return {
                        "success": False,
                        "error": str(e),
                        "message": "Audio summarization failed",
                        "suggestion": "Check if audio file exists and is in supported format (MP3, WAV, FLAC, etc.)"
                    }
            
            # Register unified video analysis tool
            @self.mcp.tool(description="Unified video analysis for YouTube, local videos, and other platforms")
            async def analyze_video_unified(video_input: str, language: str = "en"):
                """Analyze video content using unified approach that detects platform type."""
                try:
                    # Use the orchestrator's unified video analysis tool
                    result = await unified_video_analysis(video_input)
                    return result
                except Exception as e:
                    return {
                        "status": "error",
                        "error": str(e)
                    }

            # Register video summarization tool
            @self.mcp.tool(description="Generate comprehensive video summary with key scenes, visual analysis, and sentiment analysis")
            async def analyze_video_summarization(video_path: str, language: str = "en"):
                """Generate comprehensive video summary with key scenes, visual analysis, and sentiment analysis."""
                try:
                    # Validate file existence for local files
                    if not video_path.startswith(('http://', 'https://', 'youtube.com', 'youtu.be')) and not os.path.exists(video_path):
                        return {
                            "success": False,
                            "error": f"Video file not found: {video_path}",
                            "suggestion": "Please check the file path and ensure the file exists"
                        }
                    
                    print(f"🎬 Starting video analysis: {video_path}")
                    
                    # Create video summarization agent directly for better control
                    video_agent = VideoSummarizationAgent()
                    
                    # Determine data type based on content
                    data_type = DataType.VIDEO
                    if isinstance(video_path, str) and ("youtube.com" in video_path or "youtu.be" in video_path):
                        data_type = DataType.WEBPAGE
                    
                    analysis_request = AnalysisRequest(
                        data_type=data_type,
                        content=video_path,
                        language=language
                    )
                    
                    print("📊 Processing video content with advanced AI analysis...")
                    result = await video_agent.process(analysis_request)
                    
                    # Enhanced response with detailed analysis information
                    response_data = {
                        "success": True,
                        "agent": "video_summarization",
                        "sentiment": {
                            "label": result.sentiment.label,
                            "confidence": result.sentiment.confidence
                        },
                        "processing_time": result.processing_time,
                        "analysis_details": {
                            "key_scenes_count": result.metadata.get("key_scenes_count", 0),
                            "key_moments_count": result.metadata.get("key_moments_count", 0),
                            "topics_identified": result.metadata.get("topics_identified", 0),
                            "summary_length": result.metadata.get("summary_length", 0),
                            "model_used": result.metadata.get("model_used", "unknown")
                        },
                        "content_summary": result.metadata.get("summary", ""),
                        "key_scenes": result.metadata.get("key_scenes", []),
                        "key_moments": result.metadata.get("key_moments", []),
                        "topics": result.metadata.get("topics", []),
                        "executive_summary": result.metadata.get("executive_summary", ""),
                        "transcript": result.metadata.get("transcript", ""),
                        "visual_analysis": result.metadata.get("visual_analysis", ""),
                        "timeline": result.metadata.get("timeline", "")
                    }
                    
                    print(f"✅ Video analysis completed in {result.processing_time:.2f} seconds")
                    print(f"📊 Sentiment: {result.sentiment.label} (confidence: {result.sentiment.confidence:.2f})")
                    
                    # Cleanup
                    await video_agent.cleanup()
                    
                    return response_data
                    
                except Exception as e:
                    print(f"❌ Video summarization failed: {e}")
                    return {
                        "success": False,
                        "error": str(e),
                        "message": "Video summarization failed",
                        "suggestion": "Check if video file exists and is in supported format (MP4, AVI, MOV, etc.)"
                    }
            
            # Register OCR tools
            @self.mcp.tool(description="Extract text from images using OCR with Ollama and Llama Vision")
            async def analyze_ocr_text_extraction(image_path: str):
                """Extract text from images using OCR with Ollama and Llama Vision"""
                try:
                    # Validate file existence
                    if not os.path.exists(image_path):
                        return {
                            "status": "error",
                            "error": f"Image file not found: {image_path}",
                            "suggestion": "Please check the file path and ensure the file exists"
                        }
                    
                    result = await self.agents["ocr"].extract_text(image_path)
                    return result
                except Exception as e:
                    return {
                        "status": "error",
                        "error": str(e),
                        "suggestion": "Check if image file is in supported format (JPG, PNG, etc.)"
                    }

            @self.mcp.tool(description="Analyze document structure and extract key information using OCR")
            async def analyze_ocr_document(image_path: str):
                """Analyze document structure and extract key information using OCR"""
                try:
                    # Validate file existence
                    if not os.path.exists(image_path):
                        return {
                            "status": "error",
                            "error": f"Image file not found: {image_path}",
                            "suggestion": "Please check the file path and ensure the file exists"
                        }
                    
                    result = await self.agents["ocr"].analyze_document(image_path)
                    return result
                except Exception as e:
                    return {
                        "status": "error",
                        "error": str(e),
                        "suggestion": "Check if image file is in supported format (JPG, PNG, etc.)"
                    }

            @self.mcp.tool(description="Process multiple images for OCR in batch")
            async def analyze_ocr_batch(image_paths: List[str]):
                """Process multiple images for OCR in batch"""
                try:
                    # Validate all files exist
                    for image_path in image_paths:
                        if not os.path.exists(image_path):
                            return {
                                "status": "error",
                                "error": f"Image file not found: {image_path}",
                                "suggestion": "Please check the file paths and ensure all files exist"
                            }
                    
                    result = await self.agents["ocr"].batch_extract_text(image_paths)
                    return result
                except Exception as e:
                    return {
                        "status": "error",
                        "error": str(e),
                        "suggestion": "Check if image files are in supported format (JPG, PNG, etc.)"
                    }

            @self.mcp.tool(description="Generate comprehensive OCR report for an image")
            async def analyze_ocr_report(image_path: str):
                """Generate comprehensive OCR report for an image"""
                try:
                    # Validate file existence
                    if not os.path.exists(image_path):
                        return {
                            "status": "error",
                            "error": f"Image file not found: {image_path}",
                            "suggestion": "Please check the file path and ensure the file exists"
                        }
                    
                    result = await self.agents["ocr"].generate_ocr_report(image_path)
                    return result
                except Exception as e:
                    return {
                        "status": "error",
                        "error": str(e),
                        "suggestion": "Check if image file is in supported format (JPG, PNG, etc.)"
                    }

            @self.mcp.tool(description="Optimize image specifically for OCR processing")
            async def analyze_ocr_optimize(image_path: str, optimization_type: str = "auto"):
                """Optimize image specifically for OCR processing"""
                try:
                    # Validate file existence
                    if not os.path.exists(image_path):
                        return {
                            "status": "error",
                            "error": f"Image file not found: {image_path}",
                            "suggestion": "Please check the file path and ensure the file exists"
                        }
                    
                    result = await self.agents["ocr"].optimize_for_ocr(image_path, optimization_type)
                    return result
                except Exception as e:
                    return {
                        "status": "error",
                        "error": str(e),
                        "suggestion": "Check if image file is in supported format (JPG, PNG, etc.)"
                    }

            # Register translation tools
            @self.mcp.tool(description="Translate text content to English with automatic language detection")
            async def translate_text(text: str, language: str = "en"):
                """Translate text content to English with automatic language detection."""
                try:
                    analysis_request = AnalysisRequest(
                        data_type=DataType.TEXT,
                        content=text,
                        language=language
                    )
                    
                    result = await self.agents["translation"].process(analysis_request)
                    
                    return {
                        "success": True,
                        "agent": "translation",
                        "original_text": result.metadata.get("translation", {}).get("original_text", text),
                        "translated_text": result.extracted_text,
                        "source_language": result.metadata.get("original_language", "unknown"),
                        "translation_memory_hit": result.metadata.get("translation_memory_hit", False),
                        "model_used": result.metadata.get("model_used", "unknown"),
                        "processing_time": result.processing_time
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }

            @self.mcp.tool(description="Translate webpage content to English")
            async def translate_webpage(url: str):
                """Translate webpage content to English."""
                try:
                    analysis_request = AnalysisRequest(
                        data_type=DataType.WEBPAGE,
                        content=url,
                        language="auto"
                    )
                    
                    result = await self.agents["translation"].process(analysis_request)
                    
                    return {
                        "success": True,
                        "agent": "translation",
                        "url": url,
                        "original_text": result.metadata.get("translation", {}).get("original_text", ""),
                        "translated_text": result.extracted_text,
                        "source_language": result.metadata.get("original_language", "unknown"),
                        "translation_memory_hit": result.metadata.get("translation_memory_hit", False),
                        "model_used": result.metadata.get("model_used", "unknown"),
                        "processing_time": result.processing_time
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }

            @self.mcp.tool(description="Translate audio content to English (transcribe and translate)")
            async def translate_audio(audio_path: str):
                """Translate audio content to English (transcribe and translate)."""
                try:
                    # Validate file existence
                    if not os.path.exists(audio_path):
                        return {
                            "success": False,
                            "error": f"Audio file not found: {audio_path}",
                            "suggestion": "Please check the file path and ensure the file exists"
                        }
                    
                    analysis_request = AnalysisRequest(
                        data_type=DataType.AUDIO,
                        content=audio_path,
                        language="auto"
                    )
                    
                    result = await self.agents["translation"].process(analysis_request)
                    
                    return {
                        "success": True,
                        "agent": "translation",
                        "audio_path": audio_path,
                        "original_text": result.metadata.get("translation", {}).get("original_text", ""),
                        "translated_text": result.extracted_text,
                        "source_language": result.metadata.get("original_language", "unknown"),
                        "translation_memory_hit": result.metadata.get("translation_memory_hit", False),
                        "model_used": result.metadata.get("model_used", "unknown"),
                        "processing_time": result.processing_time
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "suggestion": "Check if audio file is in supported format (MP3, WAV, FLAC, etc.)"
                    }

            @self.mcp.tool(description="Translate video content to English (extract audio/visual and translate)")
            async def translate_video(video_path: str):
                """Translate video content to English (extract audio/visual and translate)."""
                try:
                    # Validate file existence for local files
                    if not video_path.startswith(('http://', 'https://', 'youtube.com', 'youtu.be')) and not os.path.exists(video_path):
                        return {
                            "success": False,
                            "error": f"Video file not found: {video_path}",
                            "suggestion": "Please check the file path and ensure the file exists"
                        }
                    
                    analysis_request = AnalysisRequest(
                        data_type=DataType.VIDEO,
                        content=video_path,
                        language="auto"
                    )
                    
                    result = await self.agents["translation"].process(analysis_request)
                    
                    return {
                        "success": True,
                        "agent": "translation",
                        "video_path": video_path,
                        "original_text": result.metadata.get("translation", {}).get("original_text", ""),
                        "translated_text": result.extracted_text,
                        "source_language": result.metadata.get("original_language", "unknown"),
                        "translation_memory_hit": result.metadata.get("translation_memory_hit", False),
                        "model_used": result.metadata.get("model_used", "unknown"),
                        "processing_time": result.processing_time
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "suggestion": "Check if video file is in supported format (MP4, AVI, MOV, etc.)"
                    }

            @self.mcp.tool(description="Translate PDF content to English (extract text and translate)")
            async def translate_pdf(pdf_path: str):
                """Translate PDF content to English (extract text and translate)."""
                try:
                    # Validate file existence
                    if not os.path.exists(pdf_path):
                        return {
                            "success": False,
                            "error": f"PDF file not found: {pdf_path}",
                            "suggestion": "Please check the file path and ensure the file exists"
                        }
                    
                    analysis_request = AnalysisRequest(
                        data_type=DataType.PDF,
                        content=pdf_path,
                        language="auto"
                    )
                    
                    result = await self.agents["translation"].process(analysis_request)
                    
                    return {
                        "success": True,
                        "agent": "translation",
                        "pdf_path": pdf_path,
                        "original_text": result.metadata.get("translation", {}).get("original_text", ""),
                        "translated_text": result.extracted_text,
                        "source_language": result.metadata.get("original_language", "unknown"),
                        "translation_memory_hit": result.metadata.get("translation_memory_hit", False),
                        "model_used": result.metadata.get("model_used", "unknown"),
                        "processing_time": result.processing_time
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "suggestion": "Check if PDF file is valid and readable"
                    }

            @self.mcp.tool(description="Batch translate multiple content items to English")
            async def batch_translate(requests: List[Dict[str, Any]]):
                """Batch translate multiple content items to English."""
                try:
                    # Convert requests to AnalysisRequest objects
                    analysis_requests = []
                    for req in requests:
                        analysis_requests.append(AnalysisRequest(
                            data_type=DataType(req.get("data_type", "text")),
                            content=req.get("content", ""),
                            language=req.get("language", "auto")
                        ))
                    
                    # Process batch translation
                    results = await self.agents["translation"].batch_translate(analysis_requests)
                    
                    # Format results
                    formatted_results = []
                    for result in results:
                        formatted_results.append({
                            "success": result.status == "completed",
                            "request_id": result.request_id,
                            "original_text": result.metadata.get("translation", {}).get("original_text", ""),
                            "translated_text": result.extracted_text,
                            "source_language": result.metadata.get("original_language", "unknown"),
                            "translation_memory_hit": result.metadata.get("translation_memory_hit", False),
                            "model_used": result.metadata.get("model_used", "unknown"),
                            "processing_time": result.processing_time,
                            "error": result.metadata.get("error") if result.status == "failed" else None
                        })
                    
                    return {
                        "success": True,
                        "agent": "translation",
                        "total_requests": len(analysis_requests),
                        "completed": len([r for r in formatted_results if r["success"]]),
                        "failed": len([r for r in formatted_results if not r["success"]]),
                        "results": formatted_results
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }

            @self.mcp.tool(description="Translate text with comprehensive analysis")
            async def translate_text_comprehensive(text: str, language: str = "en"):
                """Translate text content to English with automatic language detection and comprehensive analysis including summary and sentiment analysis."""
                try:
                    # Use the comprehensive translation method from the translation agent
                    result = await self.agents["translation"].comprehensive_translate_and_analyze(text, include_analysis=True)
                    
                    return {
                        "success": True,
                        "agent": "translation_comprehensive",
                        "translation": result["translation"],
                        "sentiment_analysis": result.get("sentiment_analysis", {}),
                        "summary_analysis": result.get("summary_analysis", {}),
                        "processing_time": result["translation"]["processing_time"]
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "agent": "translation_comprehensive"
                    }

            @self.mcp.tool(description="Analyze Chinese news with comprehensive translation, summary, and sentiment analysis")
            async def analyze_chinese_news_comprehensive(text: str, language: str = "en"):
                """Analyze Chinese news content with comprehensive translation, sentiment analysis, and summarization."""
                try:
                    # Use the dynamic news analysis method from the translation agent
                    result = await self.agents["translation"].analyze_chinese_news_dynamic(text, include_timestamp=True)
                    
                    return {
                        "success": True,
                        "agent": "chinese_news_analysis",
                        "original_text": text,
                        "translation": result["translation"],
                        "sentiment_analysis": result.get("sentiment_analysis", {}),
                        "summary_analysis": result.get("summary_analysis", {}),
                        "key_themes": result.get("key_themes", []),
                        "news_analysis": result.get("news_analysis", {}),
                        "processing_time": result["translation"]["processing_time"],
                        "analysis_timestamp": result.get("analysis_timestamp", "")
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "agent": "chinese_news_analysis"
                    }

            print("✅ Registered 34 tools with streamable HTTP support")
            
        except Exception as e:
            print(f"❌ Error registering tools: {e}")
    
    def run(self, host: str = "localhost", port: int = 8000, debug: bool = False):
        """Run the MCP server with streamable HTTP support."""
        if self.mcp:
            print(f"🚀 Starting MCP server with streamable HTTP on {host}:{port}")
            return self.mcp.run(transport="streamable-http")
        else:
            print("❌ MCP server not initialized")


def start_mcp_server():
    """Start the unified MCP server with streamable HTTP support."""
    try:
        # Create the unified MCP server
        mcp_server = UnifiedMCPServer()
        
        if mcp_server.mcp is None:
            print("⚠️  MCP server not available - skipping MCP server startup")
            return None
        
        # Start the server in a separate thread
        def run_mcp_server():
            try:
                mcp_server.run(host="localhost", port=8000, debug=False)
            except Exception as e:
                print(f"❌ Error starting MCP server: {e}")
        
        mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
        mcp_thread.start()
        
        print("✅ Unified MCP server with streamable HTTP started successfully")
        print("   - MCP Server: http://localhost:8000/mcp")
        print("   - Available agents: text, text_simple, text_strands, text_swarm, audio, vision, web, audio_summary, video_summary, orchestrator, translation, youtube")
        
        return mcp_server
        
    except Exception as e:
        print(f"⚠️  Warning: Could not start MCP server: {e}")
        print("   The application will run without MCP server integration")
        return None


def get_mcp_tools_info():
    """Get information about available MCP tools."""
    try:
        mcp_server = UnifiedMCPServer()
        
        if mcp_server.mcp is None:
            print("⚠️  MCP server not available")
            return []
        
        # Get available tools from the server
        tools = []
        if hasattr(mcp_server.mcp, 'tools'):
            tools = list(mcp_server.mcp.tools.keys())
            print(f"🔧 Available MCP tools: {tools}")
        else:
            print("⚠️  MCP server has no tools available")
        
        return tools
            
    except Exception as e:
        print(f"⚠️  Could not get MCP tools info: {e}")
        return []


if __name__ == "__main__":
    print("🚀 Starting Sentiment Analysis Swarm with Streamable HTTP MCP Integration")
    print("=" * 70)
    
    # Start MCP server
    mcp_server = start_mcp_server()
    
    # Show available tools
    if mcp_server:
        print("\n🔧 MCP Tools Available:")
        tools = get_mcp_tools_info()
        if tools:
            for tool in tools:
                print(f"   - {tool}")
    
    print("\n🌐 Starting FastAPI server...")
    print("   - API Endpoints: http://0.0.0.0:8001")
    print("   - Health Check: http://0.0.0.0:8001/health")
    print("   - API Docs: http://0.0.0.0:8001/docs")
    if mcp_server:
        print("   - MCP Server: http://localhost:8000/mcp")
    
    # Start the FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
