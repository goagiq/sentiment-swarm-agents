"""
Orchestrator agent that implements the "Agents as Tools" pattern using 
Strands framework. This agent acts as the primary coordinator and has 
access to all specialized agents as tools.
"""

from typing import Any, Dict, List

from loguru import logger
from src.core.strands_mock import tool

from src.core.models import (
    AnalysisRequest, DataType, AnalysisResult, SentimentResult
)
from src.config.config import config
from src.agents.text_agent import TextAgent
from src.agents.vision_agent_enhanced import EnhancedVisionAgent
from src.agents.audio_agent_enhanced import EnhancedAudioAgent
from src.agents.web_agent_enhanced import EnhancedWebAgent
from src.agents.text_agent_swarm import TextAgentSwarm
from src.agents.audio_summarization_agent import AudioSummarizationAgent
from src.agents.video_summarization_agent import VideoSummarizationAgent
from src.agents.ocr_agent import OCRAgent
from src.agents.base_agent import StrandsBaseAgent
from src.core.youtube_comprehensive_analyzer import YouTubeComprehensiveAnalyzer


# Define specialized agent tools using @tool decorator
@tool("text_sentiment_analysis", "Handle text-based sentiment analysis queries")
async def text_sentiment_analysis(query: str) -> dict:
    """
    Handle text-based sentiment analysis queries.
    
    Args:
        query: Text content to analyze for sentiment
        
    Returns:
        Detailed sentiment analysis result
    """
    try:
        # Create specialized text agent
        text_agent = TextAgent()
        
        # Create analysis request
        request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=query,
            language="en"
        )
        
        # Process the request
        result = await text_agent.process(request)
        
        return {
            "status": "success",
            "content": [{
                "json": {
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "method": "text_agent",
                    "agent_id": text_agent.agent_id
                }
            }]
        }
        
    except Exception as e:
        logger.error(f"Text sentiment analysis failed: {e}")
        return {
            "status": "error",
            "content": [{"text": f"Text analysis error: {str(e)}"}]
        }


@tool("vision_sentiment_analysis", "Handle comprehensive image, video, and YouTube analysis with yt-dlp integration")
async def vision_sentiment_analysis(image_path: str) -> dict:
    """
    Handle comprehensive image, video, and YouTube analysis with yt-dlp integration.
    
    Args:
        image_path: Path to image/video file or YouTube URL
        
    Returns:
        Enhanced vision sentiment analysis result with metadata and visual insights
    """
    try:
        # Check if this is a YouTube URL
        if "youtube.com" in image_path or "youtu.be" in image_path:
            # Use YouTube Comprehensive Analyzer for YouTube URLs
            youtube_analyzer = YouTubeComprehensiveAnalyzer()
            result = await youtube_analyzer.analyze_youtube_video(
                image_path,
                extract_audio=True,
                extract_frames=True,
                num_frames=5
            )
            
            return {
                "status": "success",
                "content": [{
                    "json": {
                        "sentiment": result.combined_sentiment.label,
                        "confidence": result.combined_sentiment.confidence,
                        "method": "youtube_comprehensive_analyzer",
                        "video_title": result.video_metadata.get("title", "Unknown"),
                        "duration": result.video_metadata.get("duration", 0),
                        "audio_sentiment": result.audio_sentiment.label,
                        "visual_sentiment": result.visual_sentiment.label,
                        "frames_analyzed": len(result.extracted_frames),
                        "processing_time": result.processing_time
                    }
                }]
            }
        else:
            # Use enhanced vision agent for regular images/videos
            vision_agent = EnhancedVisionAgent()
            
            # Create analysis request
            request = AnalysisRequest(
                data_type=DataType.IMAGE,
                content=image_path
            )
            
            # Process the request
            result = await vision_agent.process(request)
            
            return {
                "status": "success",
                "content": [{
                    "json": {
                        "sentiment": result.sentiment.label,
                        "confidence": result.sentiment.confidence,
                        "method": "enhanced_vision_agent",
                    "agent_id": vision_agent.agent_id
                }
            }]
        }
        
    except Exception as e:
        logger.error(f"Vision sentiment analysis failed: {e}")
        return {
            "status": "error",
            "content": [{"text": f"Vision analysis error: {str(e)}"}]
        }


@tool("youtube_comprehensive_analysis", "Handle comprehensive YouTube video analysis with audio and visual sentiment")
async def youtube_comprehensive_analysis(youtube_url: str) -> dict:
    """
    Handle comprehensive YouTube video analysis with audio and visual sentiment.
    
    Args:
        youtube_url: YouTube video URL to analyze
        
    Returns:
        Comprehensive YouTube analysis result with audio and visual sentiment
    """
    try:
        # Use YouTube Comprehensive Analyzer
        youtube_analyzer = YouTubeComprehensiveAnalyzer()
        result = await youtube_analyzer.analyze_youtube_video(
            youtube_url,
            extract_audio=True,
            extract_frames=True,
            num_frames=5
        )
        
        return {
            "status": "success",
            "content": [{
                "json": {
                    "sentiment": result.combined_sentiment.label,
                    "confidence": result.combined_sentiment.confidence,
                    "method": "youtube_comprehensive_analyzer",
                    "video_title": result.video_metadata.get("title", "Unknown"),
                    "duration": result.video_metadata.get("duration", 0),
                    "views": result.video_metadata.get("view_count", 0),
                    "likes": result.video_metadata.get("like_count", 0),
                    "audio_sentiment": result.audio_sentiment.label,
                    "visual_sentiment": result.visual_sentiment.label,
                    "frames_analyzed": len(result.extracted_frames),
                    "processing_time": result.processing_time,
                    "uploader": result.video_metadata.get("uploader", "Unknown"),
                    "upload_date": result.video_metadata.get("upload_date", "Unknown")
                }
            }]
        }
        
    except Exception as e:
        logger.error(f"YouTube comprehensive analysis failed: {e}")
        return {
            "status": "error",
            "content": [{"text": f"YouTube analysis error: {str(e)}"}]
        }


@tool("enhanced_audio_sentiment_analysis", "Handle enhanced audio sentiment analysis queries with comprehensive features")
async def enhanced_audio_sentiment_analysis(audio_path: str) -> dict:
    """
    Handle enhanced audio sentiment analysis queries with comprehensive features.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Enhanced audio sentiment analysis result with transcription, features, and quality assessment
    """
    try:
        # Create specialized enhanced audio agent
        audio_agent = EnhancedAudioAgent()
        
        # Create analysis request
        request = AnalysisRequest(
            data_type=DataType.AUDIO,
            content=audio_path
        )
        
        # Process the request with enhanced capabilities
        result = await audio_agent.process(request)
        
        return {
            "status": "success",
            "content": [{
                "json": {
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "method": "enhanced_audio_agent",
                    "agent_id": audio_agent.agent_id,
                    "enhanced_features": True,
                    "capabilities": audio_agent.metadata.get("capabilities", [])
                }
            }]
        }
        
    except Exception as e:
        logger.error(f"Enhanced audio sentiment analysis failed: {e}")
        return {
            "status": "error",
            "content": [{"text": f"Enhanced audio analysis error: {str(e)}"}]
        }


@tool("web_sentiment_analysis", "Handle webpage sentiment analysis queries")
async def web_sentiment_analysis(url: str) -> dict:
    """
    Handle webpage sentiment analysis queries.
    
    Args:
        url: URL of webpage to analyze
        
    Returns:
        Detailed webpage sentiment analysis result
    """
    try:
        # Create specialized web agent
        web_agent = WebAgent()
        
        # Create analysis request
        request = AnalysisRequest(
            data_type=DataType.WEBPAGE,
            content=url
        )
        
        # Process the request
        result = await web_agent.process(request)
        
        return {
            "status": "success",
            "content": [{
                "json": {
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "method": "web_agent",
                    "agent_id": web_agent.agent_id
                }
            }]
        }
        
    except Exception as e:
        logger.error(f"Web sentiment analysis failed: {e}")
        return {
            "status": "error",
            "content": [{"text": f"Web analysis error: {str(e)}"}]
        }


@tool("swarm_text_analysis", "Handle complex text analysis using coordinated swarm of agents")
async def swarm_text_analysis(text: str) -> dict:
    """
    Handle complex text analysis using coordinated swarm of agents.
    
    Args:
        text: Text content to analyze with swarm coordination
        
    Returns:
        Coordinated sentiment analysis result from multiple agents
    """
    try:
        # Create specialized swarm agent
        swarm_agent = TextAgentSwarm(agent_count=3)
        
        # Create analysis request
        request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=text,
            language="en"
        )
        
        # Process the request
        result = await swarm_agent.process(request)
        
        return {
            "status": "success",
            "content": [{
                "json": {
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "method": "swarm_coordination",
                    "agent_id": swarm_agent.agent_id,
                    "swarm_size": swarm_agent.agent_count
                }
            }]
        }
        
    except Exception as e:
        logger.error(f"Swarm text analysis failed: {e}")
        return {
            "status": "error",
            "content": [{"text": f"Swarm analysis error: {str(e)}"}]
        }


@tool("audio_summarization_analysis", "Handle comprehensive audio summarization with key points, action items, sentiment analysis, and detailed metadata")
async def audio_summarization_analysis(audio_path: str) -> dict:
    """
    Handle comprehensive audio summarization with key points, action items, sentiment analysis, and detailed metadata.
    
    Args:
        audio_path: Path to audio file or URL
        
    Returns:
        Comprehensive audio summarization result with enhanced metadata
    """
    try:
        logger.info(f"🎵 Starting enhanced audio summarization analysis: {audio_path}")
        
        # Create specialized audio summarization agent
        audio_summary_agent = AudioSummarizationAgent()
        
        # Create analysis request
        request = AnalysisRequest(
            data_type=DataType.AUDIO,
            content=audio_path,
            language="en"
        )
        
        # Process the request with enhanced capabilities
        result = await audio_summary_agent.process(request)
        
        # Enhanced response with detailed analysis information
        response_data = {
            "status": "success",
            "content": [{
                "json": {
                    "sentiment": {
                        "label": result.sentiment.label,
                        "confidence": result.sentiment.confidence
                    },
                    "processing_time": result.processing_time,
                    "analysis_details": {
                        "key_points_count": result.metadata.get("key_points_count", 0),
                        "action_items_count": result.metadata.get("action_items_count", 0),
                        "topics_identified": result.metadata.get("topics_identified", 0),
                        "summary_length": result.metadata.get("summary_length", 0),
                        "model_used": result.metadata.get("model_used", "unknown"),
                        "audio_duration": result.metadata.get("audio_duration", "unknown"),
                        "speakers_detected": result.metadata.get("speakers_detected", 0),
                        "language_detected": result.metadata.get("language_detected", "en"),
                        "file_size": result.metadata.get("file_size", "unknown"),
                        "audio_format": result.metadata.get("audio_format", "unknown"),
                        "quality_score": result.metadata.get("quality_score", 0.0)
                    },
                    "method": "audio_summarization_agent",
                    "agent_id": audio_summary_agent.agent_id,
                    "content_summary": result.metadata.get("summary", ""),
                    "key_points": result.metadata.get("key_points", []),
                    "action_items": result.metadata.get("action_items", []),
                    "topics": result.metadata.get("topics", []),
                    "executive_summary": result.metadata.get("executive_summary", ""),
                    "meeting_minutes": result.metadata.get("meeting_minutes", ""),
                    "quotes": result.metadata.get("quotes", []),
                    "emotions": result.metadata.get("emotions", []),
                    "timeline": result.metadata.get("timeline", ""),
                    "transcript": result.metadata.get("transcript", ""),
                    "speaker_analysis": result.metadata.get("speaker_analysis", {}),
                    "audio_quality_metrics": result.metadata.get("audio_quality_metrics", {}),
                    "content_classification": result.metadata.get("content_classification", {}),
                    "recommendations": result.metadata.get("recommendations", []),
                    "enhanced_features": {
                        "speaker_diarization": result.metadata.get("speaker_diarization", False),
                        "emotion_tracking": result.metadata.get("emotion_tracking", False),
                        "topic_modeling": result.metadata.get("topic_modeling", False),
                        "sentiment_timeline": result.metadata.get("sentiment_timeline", False),
                        "action_item_extraction": result.metadata.get("action_item_extraction", False)
                    }
                }
            }]
        }
        
        logger.info(f"✅ Audio analysis completed in {result.processing_time:.2f} seconds")
        logger.info(f"📊 Sentiment: {result.sentiment.label} (confidence: {result.sentiment.confidence:.2f})")
        
        # Cleanup
        await audio_summary_agent.cleanup()
        
        return response_data
        
    except Exception as e:
        logger.error(f"Audio summarization analysis failed: {e}")
        return {
            "status": "error",
            "content": [{
                "text": f"Audio summarization error: {str(e)}",
                "suggestion": "Check if audio file exists and is in supported format (MP3, WAV, FLAC, etc.)"
            }]
        }


@tool("unified_video_analysis", "Handle comprehensive video analysis for YouTube, local videos, and other platforms")
async def unified_video_analysis(video_input: str) -> dict:
    """
    Handle comprehensive video analysis for YouTube, local videos, and other platforms.
    Automatically detects the video type and routes to appropriate analysis method.
    
    Args:
        video_input: Video file path, YouTube URL, or other video platform URL
        
    Returns:
        Comprehensive video analysis result
    """
    try:
        # Detect video type
        video_type = _detect_video_type(video_input)
        logger.info(f"Detected video type: {video_type} for input: {video_input}")
        
        if video_type == "youtube":
            # Use YouTube comprehensive analyzer
            youtube_analyzer = YouTubeComprehensiveAnalyzer()
            result = await youtube_analyzer.analyze_youtube_video(video_input)
            
            return {
                "status": "success",
                "content": [{
                    "json": {
                        "video_type": "youtube",
                        "video_url": result.video_url,
                        "sentiment": result.combined_sentiment.label,
                        "confidence": result.combined_sentiment.confidence,
                        "method": "youtube_comprehensive_analyzer",
                        "audio_sentiment": result.audio_sentiment.label if result.audio_sentiment else "neutral",
                        "visual_sentiment": result.visual_sentiment.label if result.visual_sentiment else "neutral",
                        "video_metadata": result.video_metadata,
                        "processing_time": result.processing_time,
                        "extracted_frames": len(result.extracted_frames),
                        "audio_analysis": result.audio_analysis,
                        "visual_analysis": result.visual_analysis
                    }
                }]
            }
            
        elif video_type == "local_video":
            # Use video summarization agent for local videos
            video_agent = VideoSummarizationAgent()
            
            # Create analysis request
            request = AnalysisRequest(
                data_type=DataType.VIDEO,
                content=video_input
            )
            
            # Process the request
            result = await video_agent.process(request)
            
            return {
                "status": "success",
                "content": [{
                    "json": {
                        "video_type": "local_video",
                        "video_path": video_input,
                        "sentiment": result.sentiment.label,
                        "confidence": result.sentiment.confidence,
                        "method": "video_summarization_agent",
                        "agent_id": video_agent.agent_id,
                        "summary": result.metadata.get("summary", ""),
                        "key_scenes": result.metadata.get("key_scenes", []),
                        "key_moments": result.metadata.get("key_moments", []),
                        "topics": result.metadata.get("topics", []),
                        "executive_summary": result.metadata.get("executive_summary", ""),
                        "transcript": result.metadata.get("transcript", ""),
                        "visual_analysis": result.metadata.get("visual_analysis", ""),
                        "timeline": result.metadata.get("timeline", "")
                    }
                }]
            }
            
        elif video_type == "other_platform":
            # Use web agent for other video platforms
            web_agent = EnhancedWebAgent()
            
            # Create analysis request
            request = AnalysisRequest(
                data_type=DataType.WEBPAGE,
                content=video_input
            )
            
            # Process the request
            result = await web_agent.process(request)
            
            return {
                "status": "success",
                "content": [{
                    "json": {
                        "video_type": "other_platform",
                        "video_url": video_input,
                        "sentiment": result.sentiment.label,
                        "confidence": result.sentiment.confidence,
                        "method": "web_agent",
                        "agent_id": web_agent.agent_id,
                        "webpage_analysis": result.metadata.get("webpage_analysis", {})
                    }
                }]
            }
        
        else:
            return {
                "status": "error",
                "content": [{"text": f"Unsupported video type: {video_type}"}]
            }
        
    except Exception as e:
        logger.error(f"Unified video analysis failed: {e}")
        return {
            "status": "error",
            "content": [{"text": f"Video analysis error: {str(e)}"}]
        }


def _detect_video_type(video_input: str) -> str:
    """
    Detect the type of video input (YouTube, local video, or other platform).
    
    Args:
        video_input: Video file path or URL
        
    Returns:
        Video type: "youtube", "local_video", or "other_platform"
    """
    video_input_lower = video_input.lower()
    
    # YouTube platforms
    youtube_platforms = [
        "youtube.com", "youtu.be", "youtube-nocookie.com",
        "m.youtube.com", "www.youtube.com"
    ]
    
    # Other video platforms
    other_video_platforms = [
        "vimeo.com", "tiktok.com", "instagram.com", "facebook.com",
        "twitter.com", "twitch.tv", "dailymotion.com", "vimeo.com",
        "bilibili.com", "rutube.ru", "ok.ru", "vk.com"
    ]
    
    # Check if it's a YouTube URL
    if any(platform in video_input_lower for platform in youtube_platforms):
        return "youtube"
    
    # Check if it's another video platform
    elif any(platform in video_input_lower for platform in other_video_platforms):
        return "other_platform"
    
    # Check if it's a local file path
    elif any(ext in video_input_lower for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"]):
        return "local_video"
    
    # Check if it's a file path without extension (assume local video)
    elif "/" in video_input or "\\" in video_input:
        return "local_video"
    
    # Default to other platform for unknown URLs
    elif video_input.startswith(("http://", "https://")):
        return "other_platform"
    
    # Default to local video for unknown inputs
    else:
        return "local_video"


@tool("video_summarization_analysis", "Handle comprehensive video summarization with key scenes, visual analysis, and sentiment analysis")
async def video_summarization_analysis(video_path: str) -> dict:
    """
    Handle comprehensive video summarization with key scenes, visual analysis, and sentiment analysis.
    
    Args:
        video_path: Path to video file, URL, or YouTube URL
        
    Returns:
        Comprehensive video summarization result
    """
    try:
        logger.info(f"🎬 Starting video summarization analysis: {video_path}")
        
        # Create specialized video summarization agent
        video_summary_agent = VideoSummarizationAgent()
        
        # Determine data type based on content
        data_type = DataType.VIDEO
        if isinstance(video_path, str) and ("youtube.com" in video_path or "youtu.be" in video_path):
            data_type = DataType.WEBPAGE
        
        # Create analysis request
        request = AnalysisRequest(
            data_type=data_type,
            content=video_path,
            language="en"
        )
        
        # Process the request with progress tracking
        logger.info("📊 Processing video content with advanced AI analysis...")
        result = await video_summary_agent.process(request)
        
        # Enhanced response with detailed analysis information
        response_data = {
            "status": "success",
            "content": [{
                "json": {
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
                    "timeline": result.metadata.get("timeline", ""),
                    "method": "video_summarization_agent",
                    "agent_id": video_summary_agent.agent_id
                }
            }]
        }
        
        logger.info(f"✅ Video analysis completed in {result.processing_time:.2f} seconds")
        logger.info(f"📊 Sentiment: {result.sentiment.label} (confidence: {result.sentiment.confidence:.2f})")
        
        # Cleanup
        await video_summary_agent.cleanup()
        
        return response_data
        
    except Exception as e:
        logger.error(f"Video summarization analysis failed: {e}")
        return {
            "status": "error",
            "content": [{
                "text": f"Video summarization error: {str(e)}",
                "suggestion": "Check if video file exists and is in supported format (MP4, AVI, MOV, etc.)"
            }]
        }


@tool("ocr_analysis", "Handle optical character recognition using Ollama and Llama Vision")
async def ocr_analysis(image_path: str) -> dict:
    """
    Handle optical character recognition using Ollama and Llama Vision.
    
    Args:
        image_path: Path to image file for OCR processing
        
    Returns:
        OCR analysis result with extracted text and document analysis
    """
    try:
        # Create specialized OCR agent
        ocr_agent = OCRAgent()
        
        # Create analysis request
        request = AnalysisRequest(
            data_type=DataType.IMAGE,
            content=image_path,
            language="en"
        )
        
        # Process the request
        result = await ocr_agent.process(request)
        
        return {
            "status": "success",
            "content": [{
                "json": {
                    "extracted_text": result.metadata.get("extracted_text", ""),
                    "document_analysis": result.metadata.get("document_analysis", {}),
                    "ocr_confidence": result.metadata.get("ocr_confidence", 0.0),
                    "image_metadata": result.metadata.get("image_metadata", {}),
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "method": "ocr_agent",
                    "agent_id": ocr_agent.agent_id,
                    "processing_time": result.processing_time
                }
            }]
        }
        
    except Exception as e:
        logger.error(f"OCR analysis failed: {e}")
        return {
            "status": "error",
            "content": [{"text": f"OCR analysis error: {str(e)}"}]
        }


@tool("ocr_text_extraction", "Extract text from images using OCR")
async def ocr_text_extraction(image_path: str) -> dict:
    """
    Extract text from images using OCR.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Extracted text with confidence score
    """
    try:
        # Create specialized OCR agent
        ocr_agent = OCRAgent()
        
        # Extract text
        result = await ocr_agent.extract_text(image_path)
        
        return {
            "status": "success",
            "content": [{
                "json": {
                    "text": result.get("text", ""),
                    "confidence": result.get("confidence", 0.0),
                    "method": "llama_vision_ocr",
                    "agent_id": ocr_agent.agent_id
                }
            }]
        }
        
    except Exception as e:
        logger.error(f"OCR text extraction failed: {e}")
        return {
            "status": "error",
            "content": [{"text": f"OCR text extraction error: {str(e)}"}]
        }


@tool("ocr_document_analysis", "Analyze document structure and extract key information")
async def ocr_document_analysis(image_path: str) -> dict:
    """
    Analyze document structure and extract key information.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Document analysis with structure and key information
    """
    try:
        # Create specialized OCR agent
        ocr_agent = OCRAgent()
        
        # Analyze document
        result = await ocr_agent.analyze_document(image_path)
        
        return {
            "status": "success",
            "content": [{
                "json": {
                    "document_analysis": result.get("document_analysis", ""),
                    "extracted_text": result.get("extracted_text", ""),
                    "method": "ocr_agent",
                    "agent_id": ocr_agent.agent_id
                }
            }]
        }
        
    except Exception as e:
        logger.error(f"OCR document analysis failed: {e}")
        return {
            "status": "error",
            "content": [{"text": f"OCR document analysis error: {str(e)}"}]
        }


@tool("ocr_batch_processing", "Process multiple images for OCR in batch")
async def ocr_batch_processing(image_paths: List[str]) -> dict:
    """
    Process multiple images for OCR in batch.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        Batch processing results with statistics
    """
    try:
        # Create specialized OCR agent
        ocr_agent = OCRAgent()
        
        # Process batch
        result = await ocr_agent.batch_extract_text(image_paths)
        
        return {
            "status": "success",
            "content": [{
                "json": {
                    "total_images": result.get("total_images", 0),
                    "successful": result.get("successful", 0),
                    "failed": result.get("failed", 0),
                    "success_rate": result.get("success_rate", 0.0),
                    "results": result.get("results", []),
                    "method": "ocr_batch_processing",
                    "agent_id": ocr_agent.agent_id
                }
            }]
        }
        
    except Exception as e:
        logger.error(f"OCR batch processing failed: {e}")
        return {
            "status": "error",
            "content": [{"text": f"OCR batch processing error: {str(e)}"}]
        }


class OrchestratorAgent(StrandsBaseAgent):
    """
    Orchestrator agent that implements the "Agents as Tools" pattern.
    This agent coordinates all specialized agents and provides them as tools.
    """
    
    def __init__(self, model_name: str = None):
        # Use config system instead of hardcoded values
        super().__init__(
            agent_id="orchestrator_agent",
            model_name=model_name or config.model.default_text_model
        )
        
        # Set system prompt after initialization
        self.system_prompt = (
            "You are an orchestrator agent that coordinates sentiment "
            "analysis across multiple modalities."
        )
        
        # Initialize specialized agents
        self.text_agent = TextAgent()
        self.vision_agent = EnhancedVisionAgent()
        self.audio_agent = EnhancedAudioAgent()
        self.web_agent = EnhancedWebAgent()
        self.swarm = TextAgentSwarm()
        
        # Collect all available tools
        self.tools = [
            text_sentiment_analysis,
            vision_sentiment_analysis,
            enhanced_audio_sentiment_analysis,
            web_sentiment_analysis,
            swarm_text_analysis,
            audio_summarization_analysis,
            video_summarization_analysis,
            unified_video_analysis,
            youtube_comprehensive_analysis,
            ocr_analysis,
            ocr_text_extraction,
            ocr_document_analysis,
            ocr_batch_processing
        ]
        
        # Initialize conversation history
        self.conversation_history = []
        
        logger.info(
            f"Orchestrator agent initialized with {len(self.tools)} tools"
        )
    
    async def process_query(self, query: str) -> str:
        """Process a natural language query and route to appropriate tools."""
        try:
            # Simple routing logic - can be enhanced with LLM-based routing
            query_lower = query.lower()
            
            # Check for video content first (URLs or file paths)
            if self._is_video_content(query):
                return await unified_video_analysis(query)
            elif any(word in query_lower for word in [
                "image", "picture", "photo"
            ]):
                return await vision_sentiment_analysis(query)
            elif any(word in query_lower for word in [
                "audio", "sound", "voice", "music"
            ]):
                return await enhanced_audio_sentiment_analysis(query)
            elif any(word in query_lower for word in [
                "website", "webpage", "url", "link"
            ]):
                return await web_sentiment_analysis(query)
            elif any(word in query_lower for word in [
                "swarm", "multiple", "ensemble"
            ]):
                return await swarm_text_analysis(query)
            else:
                # Default to text analysis
                return await text_sentiment_analysis(query)
                
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return f"Error processing query: {str(e)}"


    @staticmethod
    def _is_video_content(content: str) -> bool:
        """
        Check if the content is video-related (URL or file path).
        
        Args:
            content: Content to check
            
        Returns:
            True if content is video-related, False otherwise
        """
        content_lower = content.lower()
        
        # Video file extensions
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"]
        
        # Video platforms
        video_platforms = [
            "youtube.com", "youtu.be", "vimeo.com", "tiktok.com", 
            "instagram.com", "facebook.com", "twitter.com", "twitch.tv",
            "dailymotion.com", "bilibili.com", "rutube.ru", "ok.ru", "vk.com"
        ]
        
        # Check for video file extensions
        if any(ext in content_lower for ext in video_extensions):
            return True
        
        # Check for video platform URLs
        if any(platform in content_lower for platform in video_platforms):
            return True
        
        # Check for file paths (assume video if it contains path separators)
        if "/" in content or "\\" in content:
            return True
        
        return False
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools with metadata."""
        return [
            {
                "name": tool.__name__,
                "description": tool.__doc__ or "No description available",
                "function": tool
            }
            for tool in self.tools
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the orchestrator."""
        return {
            "agent_id": self.agent_id,
            "status": "active",
            "tools_available": len(self.tools),
            "conversation_history_length": len(self.conversation_history),
            "model": self.model_name
        }
    
    async def start(self):
        """Start the orchestrator agent."""
        logger.info("Orchestrator agent started")
        return True
    
    async def stop(self):
        """Stop the orchestrator agent."""
        logger.info("Orchestrator agent stopped")
        return True
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        return True  # Orchestrator can route any request
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process the analysis request by routing to appropriate specialist."""
        try:
            # Route to appropriate specialist based on data type
            if request.data_type == DataType.TEXT:
                result = await text_sentiment_analysis(str(request.content))
            elif request.data_type == DataType.IMAGE:
                result = await vision_sentiment_analysis(str(request.content))
            elif request.data_type == DataType.AUDIO:
                result = await enhanced_audio_sentiment_analysis(str(request.content))
            elif request.data_type == DataType.WEBPAGE:
                result = await web_sentiment_analysis(str(request.content))
            else:
                # Default to text analysis
                result = await text_sentiment_analysis(str(request.content))
            
            # Convert tool result to AnalysisResult
            if result.get("status") == "success":
                content = result.get("content", [])
                if content and "json" in content[0]:
                    json_data = content[0]["json"]
                    sentiment = SentimentResult(
                        label=json_data.get("sentiment", "neutral"),
                        confidence=json_data.get("confidence", 0.5)
                    )
                    return AnalysisResult(
                        content=request.content,
                        data_type=request.data_type,
                        sentiment=sentiment,
                        metadata={
                            "method": json_data.get("method", "orchestrator"),
                            "agent_id": json_data.get(
                                "agent_id", self.agent_id
                            )
                        }
                    )
            
            # Fallback result
            return AnalysisResult(
                content=request.content,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral", confidence=0.0
                ),
                metadata={
                    "method": "orchestrator", 
                    "agent_id": self.agent_id
                }
            )
            
        except Exception as e:
            logger.error(f"Orchestrator processing failed: {e}")
            return AnalysisResult(
                content=request.content,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="uncertain", confidence=0.0
                ),
                metadata={
                    "method": "orchestrator", 
                    "agent_id": self.agent_id, 
                    "error": str(e)
                }
            )
