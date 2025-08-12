"""
Website Processing Server for Consolidated MCP System.

This module provides a unified website processing server that consolidates
all website-related functionality into a single server with 6 core functions.
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
from agents.web_agent_enhanced import EnhancedWebAgent
from agents.unified_text_agent import UnifiedTextAgent
from agents.enhanced_knowledge_graph_agent import EnhancedKnowledgeGraphAgent

# Import core services
from core.models import AnalysisRequest, DataType
from core.vector_db import VectorDBManager
from core.improved_knowledge_graph_utility import ImprovedKnowledgeGraphUtility
from core.translation_service import TranslationService


class WebsiteProcessingServer(BaseProcessingServer):
    """Website processing server with unified interface for all website operations."""
    
    def __init__(self, config: ConsolidatedMCPServerConfig):
        """Initialize the website processing server."""
        super().__init__(config)
        
        # Initialize website-specific agents
        self.web_agent = EnhancedWebAgent()
        self.text_agent = UnifiedTextAgent()
        self.knowledge_graph_agent = EnhancedKnowledgeGraphAgent(
            graph_storage_path=self.config.knowledge_graph_path
        )
        
        logger.info("✅ Website Processing Server initialized")
    
    async def extract_text(self, content: str, language: str = "en") -> Dict[str, Any]:
        """Extract text from website content.
        
        Args:
            content: Website URL or HTML content
            language: Language code for processing
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            logger.info(f"🌐 Extracting text from website: {content}")
            
            # Check if content is a URL
            if content.startswith(('http://', 'https://')):
                # Create analysis request for website
                analysis_request = AnalysisRequest(
                    data_type=DataType.WEBPAGE,
                    content=content,
                    language=language
                )
                
                # Use web agent for website analysis
                result = await self.web_agent.process(analysis_request)
                
                extracted_text = result.extracted_text or ""
                
                return {
                    "success": True,
                    "text": extracted_text,
                    "source": content,
                    "language": language,
                    "extraction_method": "website_scraping",
                    "text_length": len(extracted_text),
                    "has_text": bool(extracted_text.strip()),
                    "confidence": result.sentiment.confidence,
                    "processing_time": result.processing_time,
                    "metadata": result.metadata or {}
                }
            else:
                # Assume it's HTML content
                return await self._extract_text_from_html(content, language)
                
        except Exception as e:
            logger.error(f"❌ Error extracting text from website: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content
            }
    
    async def convert_content(self, content: str, target_format: str) -> Dict[str, Any]:
        """Convert website content to alternative formats when needed.
        
        Args:
            content: Website URL or HTML content
            target_format: Target format ("text", "summary", "structured", "json")
            
        Returns:
            Dictionary with converted content
        """
        try:
            logger.info(f"🔄 Converting website to {target_format}: {content}")
            
            if target_format == "text":
                # Extract text (reuse extract_text method)
                return await self.extract_text(content)
            
            elif target_format == "summary":
                # Generate summary
                return await self.summarize_content(content)
            
            elif target_format == "structured":
                # Convert to structured data
                structured_result = await self._convert_to_structured_data(content)
                return {
                    "success": True,
                    "format": "structured",
                    "structured_data": structured_result,
                    "source": content
                }
            
            elif target_format == "json":
                # Convert to JSON format
                json_result = await self._convert_to_json(content)
                return {
                    "success": True,
                    "format": "json",
                    "json_data": json_result,
                    "source": content
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported target format: {target_format}",
                    "supported_formats": ["text", "summary", "structured", "json"]
                }
                
        except Exception as e:
            logger.error(f"❌ Error converting website content: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content,
                "target_format": target_format
            }
    
    async def summarize_content(self, content: str, language: str = "en") -> Dict[str, Any]:
        """Generate comprehensive summaries of website content.
        
        Args:
            content: Website URL or HTML content
            language: Language code for summarization
            
        Returns:
            Dictionary with summary and key points
        """
        try:
            logger.info(f"📝 Summarizing website content: {content}")
            
            # Extract text first
            text_result = await self.extract_text(content, language)
            if not text_result["success"]:
                return text_result
            
            text_content = text_result["text"]
            
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
                "language": language,
                "website_metadata": text_result.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"❌ Error summarizing website content: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content
            }
    
    async def translate_content(self, content: str, source_language: str, target_language: str = "en") -> Dict[str, Any]:
        """Translate website content to English.
        
        Args:
            content: Website URL or HTML content
            source_language: Source language code
            target_language: Target language code (default: "en")
            
        Returns:
            Dictionary with translated content
        """
        try:
            logger.info(f"🌐 Translating website content from {source_language} to {target_language}")
            
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
            logger.error(f"❌ Error translating website content: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content,
                "source_language": source_language,
                "target_language": target_language
            }
    
    async def store_in_vector_db(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Store processed website content in vector database.
        
        Args:
            content: Website URL or HTML content
            metadata: Additional metadata for storage
            
        Returns:
            Dictionary with storage result
        """
        try:
            logger.info(f"💾 Storing website content in vector database: {content}")
            
            # Extract text first
            text_result = await self.extract_text(content)
            if not text_result["success"]:
                return text_result
            
            text_content = text_result["text"]
            source_type = "url" if content.startswith(('http://', 'https://')) else "html"
            
            if not text_content.strip():
                return {
                    "success": False,
                    "error": "No text content available for storage"
                }
            
            # Prepare metadata
            storage_metadata = {
                "source_type": source_type,
                "content_type": "website",
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
            logger.error(f"❌ Error storing website content in vector database: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content
            }
    
    async def create_knowledge_graph(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create knowledge graph from website content.
        
        Args:
            content: Website URL or HTML content
            metadata: Additional metadata for knowledge graph
            
        Returns:
            Dictionary with knowledge graph result
        """
        try:
            logger.info(f"🧠 Creating knowledge graph from website content: {content}")
            
            # Extract text first
            text_result = await self.extract_text(content)
            if not text_result["success"]:
                return text_result
            
            text_content = text_result["text"]
            source_type = "url" if content.startswith(('http://', 'https://')) else "html"
            
            if not text_content.strip():
                return {
                    "success": False,
                    "error": "No text content available for knowledge graph creation"
                }
            
            # Prepare metadata
            kg_metadata = {
                "source_type": source_type,
                "content_type": "website",
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
            logger.error(f"❌ Error creating knowledge graph from website content: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content
            }
    
    async def _extract_text_from_html(self, html_content: str, language: str) -> Dict[str, Any]:
        """Extract text from HTML content."""
        try:
            # Create analysis request for HTML content
            analysis_request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=html_content,
                language=language
            )
            
            # Use text agent for HTML processing
            result = await self.text_agent.process(analysis_request)
            
            extracted_text = result.extracted_text or html_content
            
            return {
                "success": True,
                "text": extracted_text,
                "source": "html_content",
                "language": language,
                "extraction_method": "html_processing",
                "text_length": len(extracted_text),
                "has_text": bool(extracted_text.strip()),
                "confidence": result.sentiment.confidence,
                "processing_time": result.processing_time
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting text from HTML: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": "html_content"
            }
    
    async def _convert_to_structured_data(self, content: str) -> Dict[str, Any]:
        """Convert website content to structured data."""
        try:
            # This would integrate with existing structured data extraction logic
            # For now, return a placeholder
            return {
                "structured_data": {},
                "processing_time": 0.0
            }
            
        except Exception as e:
            logger.error(f"❌ Error converting to structured data: {e}")
            return {
                "structured_data": {},
                "processing_time": 0.0,
                "error": str(e)
            }
    
    async def _convert_to_json(self, content: str) -> Dict[str, Any]:
        """Convert website content to JSON format."""
        try:
            # Extract text first
            text_result = await self.extract_text(content)
            if not text_result["success"]:
                return text_result
            
            # Convert to JSON format
            json_data = {
                "content": text_result["text"],
                "metadata": text_result.get("metadata", {}),
                "source": content,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            return {
                "json_data": json_data,
                "processing_time": 0.0
            }
            
        except Exception as e:
            logger.error(f"❌ Error converting to JSON: {e}")
            return {
                "json_data": {},
                "processing_time": 0.0,
                "error": str(e)
            }
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Cleanup agents
            if hasattr(self.web_agent, 'cleanup'):
                await self.web_agent.cleanup()
            if hasattr(self.text_agent, 'cleanup'):
                await self.text_agent.cleanup()
            if hasattr(self.knowledge_graph_agent, 'cleanup'):
                await self.knowledge_graph_agent.cleanup()
            
            logger.info("✅ Website Processing Server cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during Website Processing Server cleanup: {e}")


# Export for use in other modules
__all__ = ["WebsiteProcessingServer"]
