"""
Audio Processing Server for Consolidated MCP System.

This module provides a unified audio processing server that consolidates
all audio-related functionality into a single server with 6 core functions.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# Import base class
from .consolidated_mcp_server import BaseProcessingServer, ConsolidatedMCPServerConfig

# Import agents
from agents.unified_audio_agent import UnifiedAudioAgent
from agents.unified_text_agent import UnifiedTextAgent
from agents.enhanced_knowledge_graph_agent import EnhancedKnowledgeGraphAgent

# Import core services
from core.models import AnalysisRequest, DataType
from core.vector_db import VectorDBManager
from core.improved_knowledge_graph_utility import ImprovedKnowledgeGraphUtility
from core.translation_service import TranslationService


class AudioProcessingServer(BaseProcessingServer):
    """Audio processing server with unified interface for all audio operations."""
    
    def __init__(self, config: ConsolidatedMCPServerConfig):
        """Initialize the audio processing server."""
        super().__init__(config)
        
        # Initialize audio-specific agents
        self.audio_agent = UnifiedAudioAgent()
        self.text_agent = UnifiedTextAgent()
        self.knowledge_graph_agent = EnhancedKnowledgeGraphAgent(
            graph_storage_path=self.config.knowledge_graph_path
        )
        
        logger.info("✅ Audio Processing Server initialized")
    
    async def extract_text(self, content: str, language: str = "en") -> Dict[str, Any]:
        """Extract text from audio content (transcription).
        
        Args:
            content: Path to audio file or audio content
            language: Language code for transcription
            
        Returns:
            Dictionary with transcribed text and metadata
        """
        try:
            logger.info(f"🎵 Extracting text from audio: {content}")
            
            # Check if content is a file path
            if os.path.exists(content):
                # Create analysis request for audio transcription
                analysis_request = AnalysisRequest(
                    data_type=DataType.AUDIO,
                    content=content,
                    language=language
                )
                
                # Use audio agent for transcription
                result = await self.audio_agent.process(analysis_request)
                
                transcribed_text = result.extracted_text or ""
                
                return {
                    "success": True,
                    "text": transcribed_text,
                    "source": content,
                    "language": language,
                    "extraction_method": "audio_transcription",
                    "text_length": len(transcribed_text),
                    "has_text": bool(transcribed_text.strip()),
                    "confidence": result.sentiment.confidence,
                    "processing_time": result.processing_time
                }
            else:
                return {
                    "success": False,
                    "error": f"Audio file not found: {content}",
                    "suggestion": "Please provide a valid audio file path"
                }
                
        except Exception as e:
            logger.error(f"❌ Error extracting text from audio: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content
            }
    
    async def convert_content(self, content: str, target_format: str) -> Dict[str, Any]:
        """Convert audio content to alternative formats when needed.
        
        Args:
            content: Path to audio file
            target_format: Target format ("text", "summary", "transcript")
            
        Returns:
            Dictionary with converted content
        """
        try:
            logger.info(f"🔄 Converting audio to {target_format}: {content}")
            
            if not os.path.exists(content):
                return {
                    "success": False,
                    "error": f"Audio file not found: {content}"
                }
            
            if target_format == "text":
                # Extract text (reuse extract_text method)
                return await self.extract_text(content)
            
            elif target_format == "summary":
                # Generate summary
                return await self.summarize_content(content)
            
            elif target_format == "transcript":
                # Generate detailed transcript
                transcript_result = await self._generate_transcript(content)
                return {
                    "success": True,
                    "format": "transcript",
                    "transcript": transcript_result,
                    "source": content
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported target format: {target_format}",
                    "supported_formats": ["text", "summary", "transcript"]
                }
                
        except Exception as e:
            logger.error(f"❌ Error converting audio content: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content,
                "target_format": target_format
            }
    
    async def summarize_content(self, content: str, language: str = "en") -> Dict[str, Any]:
        """Generate comprehensive summaries of audio content.
        
        Args:
            content: Path to audio file or transcribed text
            language: Language code for summarization
            
        Returns:
            Dictionary with summary and key points
        """
        try:
            logger.info(f"📝 Summarizing audio content: {content}")
            
            # Extract text first if content is a file path
            if os.path.exists(content):
                text_result = await self.extract_text(content, language)
                if not text_result["success"]:
                    return text_result
                text_content = text_result["text"]
            else:
                text_content = content
            
            if not text_content.strip():
                return {
                    "success": False,
                    "error": "No text content available for summarization"
                }
            
            # Create analysis request for summarization
            analysis_request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=text_content,
                language=language
            )
            
            # Use text agent for summarization
            result = await self.text_agent.process(analysis_request)
            
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
                "language": language
            }
            
        except Exception as e:
            logger.error(f"❌ Error summarizing audio content: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content
            }
    
    async def translate_content(self, content: str, source_language: str, target_language: str = "en") -> Dict[str, Any]:
        """Translate audio content to English.
        
        Args:
            content: Path to audio file or transcribed text
            source_language: Source language code
            target_language: Target language code (default: "en")
            
        Returns:
            Dictionary with translated content
        """
        try:
            logger.info(f"🌐 Translating audio content from {source_language} to {target_language}")
            
            # Extract text first if content is a file path
            if os.path.exists(content):
                text_result = await self.extract_text(content, source_language)
                if not text_result["success"]:
                    return text_result
                text_content = text_result["text"]
            else:
                text_content = content
            
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
            logger.error(f"❌ Error translating audio content: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content,
                "source_language": source_language,
                "target_language": target_language
            }
    
    async def store_in_vector_db(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Store processed audio content in vector database.
        
        Args:
            content: Path to audio file or transcribed text
            metadata: Additional metadata for storage
            
        Returns:
            Dictionary with storage result
        """
        try:
            logger.info(f"💾 Storing audio content in vector database: {content}")
            
            # Extract text first if content is a file path
            if os.path.exists(content):
                text_result = await self.extract_text(content)
                if not text_result["success"]:
                    return text_result
                text_content = text_result["text"]
                source_type = "file"
            else:
                text_content = content
                source_type = "text"
            
            if not text_content.strip():
                return {
                    "success": False,
                    "error": "No text content available for storage"
                }
            
            # Prepare metadata
            storage_metadata = {
                "source_type": source_type,
                "content_type": "audio",
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
            logger.error(f"❌ Error storing audio content in vector database: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content
            }
    
    async def create_knowledge_graph(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create knowledge graph from audio content.
        
        Args:
            content: Path to audio file or transcribed text
            metadata: Additional metadata for knowledge graph
            
        Returns:
            Dictionary with knowledge graph result
        """
        try:
            logger.info(f"🧠 Creating knowledge graph from audio content: {content}")
            
            # Extract text first if content is a file path
            if os.path.exists(content):
                text_result = await self.extract_text(content)
                if not text_result["success"]:
                    return text_result
                text_content = text_result["text"]
                source_type = "file"
            else:
                text_content = content
                source_type = "text"
            
            if not text_content.strip():
                return {
                    "success": False,
                    "error": "No text content available for knowledge graph creation"
                }
            
            # Prepare metadata
            kg_metadata = {
                "source_type": source_type,
                "content_type": "audio",
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
            logger.error(f"❌ Error creating knowledge graph from audio content: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content
            }
    
    async def _generate_transcript(self, audio_path: str) -> Dict[str, Any]:
        """Generate detailed transcript with timestamps and speaker identification."""
        try:
            # Create analysis request for detailed transcription
            analysis_request = AnalysisRequest(
                data_type=DataType.AUDIO,
                content=audio_path,
                language="en"
            )
            
            # Use audio agent for detailed transcription
            result = await self.audio_agent.process(analysis_request)
            
            return {
                "transcript": result.extracted_text or "",
                "metadata": result.metadata or {},
                "processing_time": result.processing_time,
                "confidence": result.sentiment.confidence
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating transcript: {e}")
            return {
                "transcript": "",
                "metadata": {},
                "processing_time": 0.0,
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Cleanup agents
            if hasattr(self.audio_agent, 'cleanup'):
                await self.audio_agent.cleanup()
            if hasattr(self.text_agent, 'cleanup'):
                await self.text_agent.cleanup()
            if hasattr(self.knowledge_graph_agent, 'cleanup'):
                await self.knowledge_graph_agent.cleanup()
            
            logger.info("✅ Audio Processing Server cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during Audio Processing Server cleanup: {e}")


# Export for use in other modules
__all__ = ["AudioProcessingServer"]
