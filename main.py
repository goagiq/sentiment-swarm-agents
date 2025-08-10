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
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from core.improved_knowledge_graph_utility import ImprovedKnowledgeGraphUtility
from core.knowledge_graph_integration import KnowledgeGraphIntegration
from config.settings import settings
from core.youtube_comprehensive_analyzer import YouTubeComprehensiveAnalyzer
from core.models import (
    AnalysisRequest, 
    DataType,
    ProcessingStatus,
    SentimentResult
)


class OptimizedMCPServer:
    """Optimized MCP server providing unified access to all agents with reduced tool count."""
    
    def __init__(self):
        # Initialize the MCP server with proper streamable HTTP support
        self.mcp = None
        self._initialize_mcp()
        
        # Initialize all agents
        self.agents = {}
        self._initialize_agents()
        
        # Register optimized tools
        self._register_optimized_tools()
        
        print("✅ Optimized MCP Server initialized with unified tools")
    
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
            # Initialize KnowledgeGraphAgent with settings-based configuration
            self.agents["knowledge_graph"] = KnowledgeGraphAgent(
                graph_storage_path=str(settings.paths.knowledge_graphs_dir)
            )
            self.agents["youtube"] = YouTubeComprehensiveAnalyzer()
            
            # Initialize improved knowledge graph utilities
            self.improved_knowledge_graph_utility = ImprovedKnowledgeGraphUtility()
            self.knowledge_graph_integration = KnowledgeGraphIntegration()
            
            print(f"✅ Initialized {len(self.agents)} agents including knowledge graph and improved utilities")
            
        except Exception as e:
            print(f"⚠️  Error initializing agents: {e}")
    
    def _register_optimized_tools(self):
        """Register optimized tools with unified interfaces."""
        
        if self.mcp is None:
            print("❌ MCP server not initialized")
            return
        
        try:
            # Core Management Tools (3)
            @self.mcp.tool(description="Get status of all available agents")
            async def get_all_agents_status():
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
            
            # Unified Analysis Tools (4)
            @self.mcp.tool(description="Analyze text content with unified interface")
            @with_error_handling("text_analysis")
            async def analyze_text(
                text: str, 
                agent_type: str = "standard", 
                language: str = "en"
            ):
                """Analyze text content using specified agent type.
                
                Args:
                    text: Text content to analyze
                    agent_type: Type of agent to use ("standard", "simple", "strands", "swarm")
                    language: Language code for analysis
                """
                agent_map = {
                    "standard": "text",
                    "simple": "text_simple", 
                    "strands": "text_strands",
                    "swarm": "text_swarm"
                }
                
                agent_key = agent_map.get(agent_type, "text")
                agent = self.agents[agent_key]
                
                analysis_request = AnalysisRequest(
                    data_type=DataType.TEXT,
                    content=text,
                    language=language
                )
                
                result = await agent.process(analysis_request)
                
                return {
                    "success": True,
                    "agent_type": agent_type,
                    "agent_used": agent_key,
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "processing_time": result.processing_time
                }
            
            @self.mcp.tool(description="Analyze media content with unified interface")
            async def analyze_media(
                content_path: str,
                media_type: str,
                language: str = "en"
            ):
                """Analyze media content using unified interface.
                
                Args:
                    content_path: Path or URL to media content
                    media_type: Type of media ("audio", "image", "webpage", "video")
                    language: Language code for analysis
                """
                try:
                    # Validate file existence for local files
                    if not content_path.startswith(('http://', 'https://', 'youtube.com', 'youtu.be')) and not os.path.exists(content_path):
                        return {
                            "success": False,
                            "error": f"File not found: {content_path}",
                            "suggestion": "Please check the file path and ensure the file exists"
                        }
                    
                    # Map media types to agents and data types
                    media_config = {
                        "audio": ("audio", DataType.AUDIO),
                        "image": ("vision", DataType.IMAGE),
                        "webpage": ("web", DataType.WEBPAGE),
                        "video": ("video_summary", DataType.VIDEO)
                    }
                    
                    if media_type not in media_config:
                        return {
                            "success": False,
                            "error": f"Unsupported media type: {media_type}",
                            "supported_types": list(media_config.keys())
                        }
                    
                    agent_key, data_type = media_config[media_type]
                    agent = self.agents[agent_key]
                    
                    analysis_request = AnalysisRequest(
                        data_type=data_type,
                        content=content_path,
                        language=language
                    )
                    
                    result = await agent.process(analysis_request)
                    
                    return {
                        "success": True,
                        "media_type": media_type,
                        "agent_used": agent_key,
                        "sentiment": result.sentiment.label,
                        "confidence": result.sentiment.confidence,
                        "processing_time": result.processing_time,
                        "metadata": result.metadata if result.metadata else {},
                        "extracted_text": result.extracted_text if result.extracted_text else None
                    }
                    
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "media_type": media_type
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
            
            @self.mcp.tool(description="Analyze YouTube video with unified interface")
            async def analyze_youtube(
                youtube_url: str,
                use_parallel: bool = True,
                num_frames: int = 5
            ):
                """Analyze YouTube video with unified interface.
                
                Args:
                    youtube_url: YouTube video URL
                    use_parallel: Whether to use parallel processing
                    num_frames: Number of frames to extract for analysis
                """
                try:
                    if use_parallel:
                        # Use enhanced analyzer with parallel processing
                        from src.core.youtube_comprehensive_analyzer_enhanced import EnhancedYouTubeComprehensiveAnalyzer
                        enhanced_analyzer = EnhancedYouTubeComprehensiveAnalyzer(max_workers=4)
                        result = await enhanced_analyzer.analyze_youtube_video_parallel(
                            youtube_url,
                            extract_audio=True,
                            extract_frames=True,
                            num_frames=num_frames,
                            use_parallel=use_parallel
                        )
                        agent_type = "youtube_comprehensive_enhanced"
                    else:
                        # Use standard analyzer
                        result = await self.agents["youtube"].analyze_youtube_video(
                            youtube_url,
                            extract_audio=True,
                            extract_frames=True,
                            num_frames=num_frames
                        )
                        agent_type = "youtube_comprehensive"
                    
                    return {
                        "success": True,
                        "agent_type": agent_type,
                        "parallel_processing": use_parallel,
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
                        "error": str(e),
                        "agent_type": "youtube"
                    }
            
            @self.mcp.tool(description="Analyze content with automatic type detection")
            async def analyze_content(
                content: str,
                language: str = "en"
            ):
                """Analyze content with automatic type detection.
                
                Args:
                    content: Content to analyze (text, file path, or URL)
                    language: Language code for analysis
                """
                try:
                    # Auto-detect content type
                    if content.startswith(('http://', 'https://')):
                        if 'youtube.com' in content or 'youtu.be' in content:
                            return await analyze_youtube(content)
                        else:
                            return await analyze_media(content, "webpage", language)
                    elif os.path.exists(content):
                        # Determine file type by extension
                        ext = os.path.splitext(content)[1].lower()
                        if ext in ['.mp3', '.wav', '.flac', '.m4a']:
                            return await analyze_media(content, "audio", language)
                        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                            return await analyze_media(content, "image", language)
                        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                            return await analyze_media(content, "video", language)
                        else:
                            return await analyze_text(content, "standard", language)
                    else:
                        # Assume it's text
                        return await analyze_text(content, "standard", language)
                        
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            # Summarization Tools (2)
            @self.mcp.tool(description="Generate comprehensive audio summary")
            async def summarize_audio(
                audio_path: str,
                language: str = "en"
            ):
                """Generate comprehensive audio summary with key points and analysis."""
                try:
                    if not os.path.exists(audio_path):
                        return {
                            "success": False,
                            "error": f"Audio file not found: {audio_path}"
                        }
                    
                    analysis_request = AnalysisRequest(
                        data_type=DataType.AUDIO,
                        content=audio_path,
                        language=language
                    )
                    
                    result = await self.agents["audio_summary"].process(analysis_request)
                    
                    return {
                        "success": True,
                        "agent": "audio_summarization",
                        "sentiment": {
                            "label": result.sentiment.label,
                            "confidence": result.sentiment.confidence
                        },
                        "processing_time": result.processing_time,
                        "summary": result.metadata.get("summary", "") if result.metadata else "",
                        "key_points": result.metadata.get("key_points", []) if result.metadata else [],
                        "action_items": result.metadata.get("action_items", []) if result.metadata else [],
                        "topics": result.metadata.get("topics", []) if result.metadata else []
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
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

            # Knowledge Graph Agent Tools
            @self.mcp.tool(description="Extract entities from text content using Knowledge Graph Agent with enhanced categorization")
            async def extract_entities(text: str):
                """Extract entities from text content."""
                try:
                    result = await self.agents["knowledge_graph"].extract_entities(text)
                    return {
                        "success": True,
                        "agent": "knowledge_graph",
                        "entities": result.get("content", [{}])[0].get("json", {}).get("entities", []),
                        "key_concepts": result.get("content", [{}])[0].get("json", {}).get("key_concepts", [])
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "agent": "knowledge_graph"
                    }

            @self.mcp.tool(description="Map relationships between entities using Knowledge Graph Agent")
            async def map_relationships(text: str, entities: List[Dict[str, Any]]):
                """Map relationships between entities in text."""
                try:
                    result = await self.agents["knowledge_graph"].map_relationships(text, entities)
                    return {
                        "success": True,
                        "agent": "knowledge_graph",
                        "relationships": result.get("content", [{}])[0].get("json", {}).get("relationships", [])
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "agent": "knowledge_graph"
                    }

            @self.mcp.tool(description="Query the knowledge graph for information")
            async def query_knowledge_graph(query: str):
                """Query the knowledge graph for information."""
                try:
                    result = await self.agents["knowledge_graph"].query_knowledge_graph(query)
                    return {
                        "success": True,
                        "agent": "knowledge_graph",
                        "query_results": result.get("content", [{}])[0].get("json", {}).get("query_results", []),
                        "insights": result.get("content", [{}])[0].get("json", {}).get("insights", "")
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "agent": "knowledge_graph"
                    }

            @self.mcp.tool(description="Generate a visual graph report with enhanced entity categorization")
            async def generate_graph_report(output_path: str = None):
                """Generate a visual graph report with PNG and interactive HTML using settings."""
                try:
                    # Use settings-based output path if not provided
                    if output_path is None:
                        output_path = str(settings.paths.reports_dir / f"{settings.report_generation.report_filename_prefix}")
                    
                    result = await self.agents["knowledge_graph"].generate_graph_report(output_path)
                    json_result = result.get("content", [{}])[0].get("json", {})
                    return {
                        "success": True,
                        "agent": "knowledge_graph",
                        "message": json_result.get("message", ""),
                        "png_file": json_result.get("png_file", ""),
                        "html_file": json_result.get("html_file", ""),
                        "md_file": json_result.get("md_file", ""),
                        "graph_stats": json_result.get("graph_stats", {})
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "agent": "knowledge_graph"
                    }

            @self.mcp.tool(description="Analyze communities in the knowledge graph")
            async def analyze_graph_communities():
                """Analyze communities in the knowledge graph."""
                try:
                    result = await self.agents["knowledge_graph"].analyze_graph_communities()
                    return {
                        "success": True,
                        "agent": "knowledge_graph",
                        "communities": result.get("content", [{}])[0].get("json", {}).get("communities", []),
                        "total_communities": result.get("content", [{}])[0].get("json", {}).get("total_communities", 0)
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "agent": "knowledge_graph"
                    }

            @self.mcp.tool(description="Find paths between two entities in the graph")
            async def find_entity_paths(source: str, target: str):
                """Find paths between two entities in the graph."""
                try:
                    result = await self.agents["knowledge_graph"].find_entity_paths(source, target)
                    return {
                        "success": True,
                        "agent": "knowledge_graph",
                        "source": source,
                        "target": target,
                        "shortest_path": result.get("content", [{}])[0].get("json", {}).get("shortest_path", []),
                        "shortest_path_length": result.get("content", [{}])[0].get("json", {}).get("shortest_path_length", -1),
                        "all_paths_count": result.get("content", [{}])[0].get("json", {}).get("all_paths_count", 0)
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "agent": "knowledge_graph"
                    }

            @self.mcp.tool(description="Get context and connections for a specific entity")
            async def get_entity_context(entity: str):
                """Get context and connections for a specific entity."""
                try:
                    result = await self.agents["knowledge_graph"].get_entity_context(entity)
                    return {
                        "success": True,
                        "agent": "knowledge_graph",
                        "entity": entity,
                        "neighbors": result.get("content", [{}])[0].get("json", {}).get("neighbors", []),
                        "incoming_connections": result.get("content", [{}])[0].get("json", {}).get("incoming_connections", []),
                        "outgoing_connections": result.get("content", [{}])[0].get("json", {}).get("outgoing_connections", []),
                        "degree_centrality": result.get("content", [{}])[0].get("json", {}).get("degree_centrality", 0)
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "agent": "knowledge_graph"
                    }

            @self.mcp.tool(description="Process content and build knowledge graph")
            async def process_content_knowledge_graph(content: str, data_type: str = "text", language: str = "en"):
                """Process content and build knowledge graph."""
                try:
                    # Convert data type string to enum
                    data_type_map = {
                        "text": DataType.TEXT,
                        "audio": DataType.AUDIO,
                        "video": DataType.VIDEO,
                        "webpage": DataType.WEBPAGE,
                        "pdf": DataType.PDF,
                        "social_media": DataType.SOCIAL_MEDIA
                    }
                    data_type_enum = data_type_map.get(data_type, DataType.TEXT)
                    
                    # Create analysis request
                    request = AnalysisRequest(
                        data_type=data_type_enum,
                        content=content,
                        language=language
                    )
                    
                    # Process the request
                    result = await self.agents["knowledge_graph"].process(request)
                    
                    return {
                        "success": True,
                        "agent": "knowledge_graph",
                        "entities_extracted": result.metadata.get("entities_extracted", 0),
                        "relationships_mapped": result.metadata.get("relationships_mapped", 0),
                        "graph_nodes": result.metadata.get("graph_nodes", 0),
                        "graph_edges": result.metadata.get("graph_edges", 0),
                        "graph_analysis": result.metadata.get("graph_analysis", {})
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "agent": "knowledge_graph"
                    }

            # Improved Knowledge Graph Tools
            @self.mcp.tool(description="Process articles with improved knowledge graph utility")
            async def process_articles_improved_knowledge_graph(articles_content: List[str]):
                """Process articles using improved knowledge graph utility."""
                try:
                    utility = ImprovedKnowledgeGraphUtility()
                    results = await utility.process_articles_and_create_graph(articles_content)
                    
                    return {
                        "success": True,
                        "utility": "improved_knowledge_graph",
                        "entities_extracted": results["entities_extracted"],
                        "relationships_mapped": results["relationships_mapped"],
                        "graph_nodes": results["graph_nodes"],
                        "graph_edges": results["graph_edges"],
                        "visualization_files": results["visualization_results"],
                        "summary_report": results["summary_report"]
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "utility": "improved_knowledge_graph"
                    }

            @self.mcp.tool(description="Process articles with knowledge graph integration (improved + original)")
            async def process_articles_knowledge_graph_integration(articles_content: List[str]):
                """Process articles using knowledge graph integration."""
                try:
                    integration = KnowledgeGraphIntegration()
                    results = await integration.process_with_improved_extraction(articles_content)
                    
                    return {
                        "success": True,
                        "integration": "knowledge_graph_integration",
                        "improved_utility_results": results["improved_utility_results"],
                        "agent_results": results["agent_results"],
                        "comparison": results["comparison"],
                        "integration_report": results["integration_report"]
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "integration": "knowledge_graph_integration"
                    }

            @self.mcp.tool(description="Get current project settings and configuration")
            async def get_project_settings():
                """Get current project settings and configuration."""
                try:
                    return {
                        "success": True,
                        "settings": {
                            "entity_categorization": {
                                "entity_types": list(settings.entity_categorization.entity_types.keys()),
                                "relationship_types": settings.entity_categorization.relationship_types
                            },
                            "report_generation": {
                                "report_title_prefix": settings.report_generation.report_title_prefix,
                                "report_filename_prefix": settings.report_generation.report_filename_prefix,
                                "generate_html": settings.report_generation.generate_html,
                                "generate_md": settings.report_generation.generate_md,
                                "generate_png": settings.report_generation.generate_png
                            },
                            "paths": {
                                "results_dir": str(settings.paths.results_dir),
                                "reports_dir": str(settings.paths.reports_dir),
                                "knowledge_graphs_dir": str(settings.paths.knowledge_graphs_dir)
                            }
                        }
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }

            @self.mcp.tool(description="Validate knowledge graph integration")
            async def validate_knowledge_graph_integration(articles_content: List[str]):
                """Validate knowledge graph integration functionality."""
                try:
                    integration = KnowledgeGraphIntegration()
                    validation_results = await integration.validate_integration(articles_content)
                    
                    return {
                        "success": True,
                        "validation": validation_results
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }

            print("✅ Registered 46 tools with streamable HTTP support")
            
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
        mcp_server = OptimizedMCPServer()
        
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
        print("   - Available agents: text, text_simple, text_strands, text_swarm, audio, vision, web, audio_summary, video_summary, orchestrator, translation, knowledge_graph, youtube")
        print("   - Available utilities: improved_knowledge_graph, knowledge_graph_integration")
        
        return mcp_server
        
    except Exception as e:
        print(f"⚠️  Warning: Could not start MCP server: {e}")
        print("   The application will run without MCP server integration")
        return None


def get_mcp_tools_info():
    """Get information about available MCP tools."""
    try:
        mcp_server = OptimizedMCPServer()
        
        if mcp_server.mcp is None:
            print("⚠️  MCP server not available")
            return []
        
        # Get available tools from the server
        tools = []
        
        # Try different ways to access tools from FastMCP
        if hasattr(mcp_server.mcp, 'tools'):
            tools = list(mcp_server.mcp.tools.keys())
        elif hasattr(mcp_server.mcp, '_tools'):
            tools = list(mcp_server.mcp._tools.keys())
        elif hasattr(mcp_server.mcp, 'app') and hasattr(mcp_server.mcp.app, 'state') and hasattr(mcp_server.mcp.app.state, 'tools'):
            tools = list(mcp_server.mcp.app.state.tools.keys())
        elif hasattr(mcp_server.mcp, 'get_tools'):
            # Check if it's a coroutine
            result = mcp_server.mcp.get_tools()
            if hasattr(result, '__await__'):  # It's a coroutine
                print("ℹ️  Using comprehensive tool list (async discovery not needed)")
                tools = []
            else:
                tools = result
        elif hasattr(mcp_server.mcp, 'list_tools'):
            # Check if it's a coroutine
            result = mcp_server.mcp.list_tools()
            if hasattr(result, '__await__'):  # It's a coroutine
                print("ℹ️  Using comprehensive tool list (async discovery not needed)")
                tools = []
            else:
                tools = result
        else:
            # If we can't access tools directly, provide a list of known tools
            tools = [
                "get_all_agents_status", "start_all_agents", "stop_all_agents",
                "analyze_text_sentiment", "analyze_text_simple", "analyze_text_strands", "analyze_text_swarm",
                "analyze_audio_sentiment", "analyze_image_sentiment", "analyze_webpage_sentiment",
                "process_query_orchestrator", "get_orchestrator_tools", "analyze_youtube_comprehensive", "analyze_youtube_comprehensive_enhanced",
                "analyze_audio_summarization", "analyze_video_unified", "analyze_video_summarization",
                "analyze_ocr_text_extraction", "analyze_ocr_document", "analyze_ocr_batch",
                "analyze_ocr_report", "analyze_ocr_optimize", "translate_text", "translate_webpage",
                "translate_audio", "translate_video", "translate_pdf", "batch_translate",
                "translate_text_comprehensive", "analyze_chinese_news_comprehensive",
                "extract_entities", "map_relationships", "query_knowledge_graph", "generate_graph_report",
                "analyze_graph_communities", "find_entity_paths", "get_entity_context", "process_content_knowledge_graph"
            ]
            print(f"🔧 Available MCP tools (comprehensive): {len(tools)} tools")
            return tools
        
        # If we still don't have tools, use the comprehensive list
        if not tools:
            tools = [
                "get_all_agents_status", "start_all_agents", "stop_all_agents",
                "analyze_text_sentiment", "analyze_text_simple", "analyze_text_strands", "analyze_text_swarm",
                "analyze_audio_sentiment", "analyze_image_sentiment", "analyze_webpage_sentiment",
                "process_query_orchestrator", "get_orchestrator_tools", "analyze_youtube_comprehensive", "analyze_youtube_comprehensive_enhanced",
                "analyze_audio_summarization", "analyze_video_unified", "analyze_video_summarization",
                "analyze_ocr_text_extraction", "analyze_ocr_document", "analyze_ocr_batch",
                "analyze_ocr_report", "analyze_ocr_optimize", "translate_text", "translate_webpage",
                "translate_audio", "translate_video", "translate_pdf", "batch_translate",
                "translate_text_comprehensive", "analyze_chinese_news_comprehensive",
                "extract_entities", "map_relationships", "query_knowledge_graph", "generate_graph_report",
                "analyze_graph_communities", "find_entity_paths", "get_entity_context", "process_content_knowledge_graph"
            ]
            print(f"🔧 Available MCP tools (comprehensive): {len(tools)} tools")
        else:
            print(f"🔧 Available MCP tools: {len(tools)} tools")
        
        return tools
            
    except Exception as e:
        print(f"⚠️  Could not get MCP tools info: {e}")
        # Return comprehensive tool list as fallback
        return [
            "get_all_agents_status", "start_all_agents", "stop_all_agents",
            "analyze_text_sentiment", "analyze_text_simple", "analyze_text_strands", "analyze_text_swarm",
            "analyze_audio_sentiment", "analyze_image_sentiment", "analyze_webpage_sentiment",
            "process_query_orchestrator", "get_orchestrator_tools", "analyze_youtube_comprehensive", "analyze_youtube_comprehensive_enhanced",
            "analyze_audio_summarization", "analyze_video_unified", "analyze_video_summarization",
            "analyze_ocr_text_extraction", "analyze_ocr_document", "analyze_ocr_batch",
            "analyze_ocr_report", "analyze_ocr_optimize", "translate_text", "translate_webpage",
            "translate_audio", "translate_video", "translate_pdf", "batch_translate",
            "translate_text_comprehensive", "analyze_chinese_news_comprehensive",
            "extract_entities", "map_relationships", "query_knowledge_graph", "generate_graph_report",
            "analyze_graph_communities", "find_entity_paths", "get_entity_context", "process_content_knowledge_graph"
        ]


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
