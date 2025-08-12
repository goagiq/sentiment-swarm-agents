"""
Video Processing Server for Consolidated MCP System.

This module provides a unified video processing server that consolidates
all video-related functionality into a single server with 6 core functions.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# Import base class
from .consolidated_mcp_server import BaseProcessingServer, ConsolidatedMCPServerConfig

# Import agents
from agents.unified_vision_agent import UnifiedVisionAgent
from agents.unified_audio_agent import UnifiedAudioAgent
from agents.unified_text_agent import UnifiedTextAgent
from agents.enhanced_knowledge_graph_agent import EnhancedKnowledgeGraphAgent

# Import core services
from core.models import AnalysisRequest, DataType
from core.vector_db import VectorDBManager
from core.improved_knowledge_graph_utility import ImprovedKnowledgeGraphUtility
from core.translation_service import TranslationService


class VideoProcessingServer(BaseProcessingServer):
    """Video processing server with unified interface for all video operations."""
    
    def __init__(self, config: ConsolidatedMCPServerConfig):
        """Initialize the video processing server."""
        super().__init__(config)
        
        # Initialize video-specific agents
        self.vision_agent = UnifiedVisionAgent()
        self.audio_agent = UnifiedAudioAgent()
        self.text_agent = UnifiedTextAgent()
        self.knowledge_graph_agent = EnhancedKnowledgeGraphAgent(
            graph_storage_path=self.config.knowledge_graph_path
        )
        
        logger.info("✅ Video Processing Server initialized")
    
    async def extract_text(self, content: str, language: str = "en") -> Dict[str, Any]:
        """Extract text from video content (subtitles/transcription).
        
        Args:
            content: Path to video file or video URL
            language: Language code for transcription
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            logger.info(f"🎬 Extracting text from video: {content}")
            
            # Check if content is a URL or file path
            if content.startswith(('http://', 'https://')):
                # Handle YouTube or other video URLs
                return await self._extract_text_from_url(content, language)
            elif os.path.exists(content):
                # Handle local video files
                return await self._extract_text_from_file(content, language)
            else:
                return {
                    "success": False,
                    "error": f"Video content not found: {content}",
                    "suggestion": "Please provide a valid video file path or URL"
                }
                
        except Exception as e:
            logger.error(f"❌ Error extracting text from video: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content
            }
    
    async def convert_content(self, content: str, target_format: str) -> Dict[str, Any]:
        """Convert video content to alternative formats when needed.
        
        Args:
            content: Path to video file or video URL
            target_format: Target format ("text", "summary", "frames", "audio")
            
        Returns:
            Dictionary with converted content
        """
        try:
            logger.info(f"🔄 Converting video to {target_format}: {content}")
            
            if target_format == "text":
                # Extract text (reuse extract_text method)
                return await self.extract_text(content)
            
            elif target_format == "summary":
                # Generate summary
                return await self.summarize_content(content)
            
            elif target_format == "frames":
                # Extract video frames
                frames_result = await self._extract_video_frames(content)
                return {
                    "success": True,
                    "format": "frames",
                    "frames": frames_result,
                    "source": content
                }
            
            elif target_format == "audio":
                # Extract audio from video
                audio_result = await self._extract_audio_from_video(content)
                return {
                    "success": True,
                    "format": "audio",
                    "audio": audio_result,
                    "source": content
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported target format: {target_format}",
                    "supported_formats": ["text", "summary", "frames", "audio"]
                }
                
        except Exception as e:
            logger.error(f"❌ Error converting video content: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content,
                "target_format": target_format
            }
    
    async def summarize_content(self, content: str, language: str = "en") -> Dict[str, Any]:
        """Generate comprehensive summaries of video content.
        
        Args:
            content: Path to video file or video URL
            language: Language code for summarization
            
        Returns:
            Dictionary with summary and key points
        """
        try:
            logger.info(f"📝 Summarizing video content: {content}")
            
            # Create analysis request for video summarization
            analysis_request = AnalysisRequest(
                data_type=DataType.VIDEO,
                content=content,
                language=language
            )
            
            # Use vision agent for video summarization
            result = await self.vision_agent.process(analysis_request)
            
            return {
                "success": True,
                "summary": result.metadata.get("summary", "") if result.metadata else "",
                "key_points": result.metadata.get("key_points", []) if result.metadata else [],
                "topics": result.metadata.get("topics", []) if result.metadata else [],
                "sentiment": {
                    "label": result.sentiment.label,
                    "confidence": result.sentiment.confidence
                },
                "processing_time": result.processing_time,
                "source": content,
                "language": language,
                "video_metadata": result.metadata.get("video_metadata", {}) if result.metadata else {}
            }
            
        except Exception as e:
            logger.error(f"❌ Error summarizing video content: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content
            }
    
    async def translate_content(self, content: str, source_language: str, target_language: str = "en") -> Dict[str, Any]:
        """Translate video content to English.
        
        Args:
            content: Path to video file or video URL
            source_language: Source language code
            target_language: Target language code (default: "en")
            
        Returns:
            Dictionary with translated content
        """
        try:
            logger.info(f"🌐 Translating video content from {source_language} to {target_language}")
            
            # Extract text first
            text_result = await self.extract_text(content, source_language)
            if not text_result["success"]:
                return text_result
            
            text_content = text_result["text"]
            
            if not text_content.strip():
                return {
                    "success": False,
                    "error": "No text content available for translation"
                }
            
            # Use translation service
            translation_result = await self.translation_service.translate_text(
                text_content, source_language, target_language
            )
            
            return {
                "success": True,
                "original_text": text_content,
                "translated_text": translation_result.get("translated_text", ""),
                "source_language": source_language,
                "target_language": target_language,
                "confidence": translation_result.get("confidence", 0.0),
                "processing_time": translation_result.get("processing_time", 0.0),
                "source": content
            }
            
        except Exception as e:
            logger.error(f"❌ Error translating video content: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content,
                "source_language": source_language,
                "target_language": target_language
            }
    
    async def store_in_vector_db(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Store processed video content in vector database.
        
        Args:
            content: Path to video file or video URL
            metadata: Additional metadata for storage
            
        Returns:
            Dictionary with storage result
        """
        try:
            logger.info(f"💾 Storing video content in vector database: {content}")
            
            # Extract text first
            text_result = await self.extract_text(content)
            if not text_result["success"]:
                return text_result
            
            text_content = text_result["text"]
            source_type = "url" if content.startswith(('http://', 'https://')) else "file"
            
            if not text_content.strip():
                return {
                    "success": False,
                    "error": "No text content available for storage"
                }
            
            # Prepare metadata
            storage_metadata = {
                "source_type": source_type,
                "content_type": "video",
                "timestamp": asyncio.get_event_loop().time(),
                **metadata
            }
            
            # Store in vector database
            storage_result = await self.vector_store.store_text(
                text_content, storage_metadata
            )
            
            return {
                "success": True,
                "vector_id": storage_result.get("vector_id", ""),
                "metadata": storage_metadata,
                "text_length": len(text_content),
                "source": content,
                "storage_time": storage_result.get("storage_time", 0.0)
            }
            
        except Exception as e:
            logger.error(f"❌ Error storing video content in vector database: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content
            }
    
    async def create_knowledge_graph(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create knowledge graph from video content.
        
        Args:
            content: Path to video file or video URL
            metadata: Additional metadata for knowledge graph
            
        Returns:
            Dictionary with knowledge graph result
        """
        try:
            logger.info(f"🧠 Creating knowledge graph from video content: {content}")
            
            # Extract text first
            text_result = await self.extract_text(content)
            if not text_result["success"]:
                return text_result
            
            text_content = text_result["text"]
            source_type = "url" if content.startswith(('http://', 'https://')) else "file"
            
            if not text_content.strip():
                return {
                    "success": False,
                    "error": "No text content available for knowledge graph creation"
                }
            
            # Prepare metadata
            kg_metadata = {
                "source_type": source_type,
                "content_type": "video",
                "timestamp": asyncio.get_event_loop().time(),
                **metadata
            }
            
            # Create knowledge graph
            kg_result = await self.knowledge_graph_agent.create_knowledge_graph(
                text_content, kg_metadata
            )
            
            return {
                "success": True,
                "knowledge_graph_id": kg_result.get("graph_id", ""),
                "entities": kg_result.get("entities", []),
                "relationships": kg_result.get("relationships", []),
                "metadata": kg_metadata,
                "source": content,
                "processing_time": kg_result.get("processing_time", 0.0)
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating knowledge graph from video content: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content
            }
    
    async def _extract_text_from_url(self, url: str, language: str) -> Dict[str, Any]:
        """Extract text from video URL (YouTube, etc.)."""
        try:
            # Create analysis request for URL video
            analysis_request = AnalysisRequest(
                data_type=DataType.WEBPAGE,  # Use WEBPAGE for URLs
                content=url,
                language=language
            )
            
            # Use vision agent for video analysis
            result = await self.vision_agent.process(analysis_request)
            
            extracted_text = result.extracted_text or ""
            
            return {
                "success": True,
                "text": extracted_text,
                "source": url,
                "language": language,
                "extraction_method": "video_url_analysis",
                "text_length": len(extracted_text),
                "has_text": bool(extracted_text.strip()),
                "confidence": result.sentiment.confidence,
                "processing_time": result.processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting text from video URL: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": url
            }
    
    async def _extract_text_from_file(self, file_path: str, language: str) -> Dict[str, Any]:
        """Extract text from local video file."""
        try:
            # Create analysis request for local video file
            analysis_request = AnalysisRequest(
                data_type=DataType.VIDEO,
                content=file_path,
                language=language
            )
            
            # Use vision agent for video analysis
            result = await self.vision_agent.process(analysis_request)
            
            extracted_text = result.extracted_text or ""
            
            return {
                "success": True,
                "text": extracted_text,
                "source": file_path,
                "language": language,
                "extraction_method": "video_file_analysis",
                "text_length": len(extracted_text),
                "has_text": bool(extracted_text.strip()),
                "confidence": result.sentiment.confidence,
                "processing_time": result.processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting text from video file: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": file_path
            }
    
    async def _extract_video_frames(self, video_path: str) -> Dict[str, Any]:
        """Extract frames from video for analysis."""
        try:
            # This would integrate with existing video frame extraction logic
            # For now, return a placeholder
            return {
                "frames_extracted": 0,
                "frame_analysis": [],
                "processing_time": 0.0
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting video frames: {e}")
            return {
                "frames_extracted": 0,
                "frame_analysis": [],
                "processing_time": 0.0,
                "error": str(e)
            }
    
    async def _extract_audio_from_video(self, video_path: str) -> Dict[str, Any]:
        """Extract audio from video for analysis."""
        try:
            # This would integrate with existing audio extraction logic
            # For now, return a placeholder
            return {
                "audio_extracted": False,
                "audio_path": "",
                "processing_time": 0.0
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting audio from video: {e}")
            return {
                "audio_extracted": False,
                "audio_path": "",
                "processing_time": 0.0,
                "error": str(e)
            }
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Cleanup agents
            if hasattr(self.vision_agent, 'cleanup'):
                await self.vision_agent.cleanup()
            if hasattr(self.audio_agent, 'cleanup'):
                await self.audio_agent.cleanup()
            if hasattr(self.text_agent, 'cleanup'):
                await self.text_agent.cleanup()
            if hasattr(self.knowledge_graph_agent, 'cleanup'):
                await self.knowledge_graph_agent.cleanup()
            
            logger.info("✅ Video Processing Server cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during Video Processing Server cleanup: {e}")


# Export for use in other modules
__all__ = ["VideoProcessingServer"]
