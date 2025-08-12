"""
PDF Processing Server for Consolidated MCP System.

This module provides a unified PDF processing server that consolidates
all PDF-related functionality into a single server with 6 core functions.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import fitz  # PyMuPDF
from PIL import Image
import io

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# Import base class
from .consolidated_mcp_server import BaseProcessingServer, ConsolidatedMCPServerConfig

# Import agents
from agents.file_extraction_agent import FileExtractionAgent
from agents.extract_pdf_text import extract_pdf_text
from agents.ocr_agent import OCRAgent
from agents.unified_text_agent import UnifiedTextAgent
from agents.enhanced_knowledge_graph_agent import EnhancedKnowledgeGraphAgent

# Import core services
from core.models import AnalysisRequest, DataType
from core.vector_db import VectorDBManager
from core.improved_knowledge_graph_utility import ImprovedKnowledgeGraphUtility
from core.translation_service import TranslationService


class PDFProcessingServer(BaseProcessingServer):
    """PDF processing server with unified interface for all PDF operations."""
    
    def __init__(self, config: ConsolidatedMCPServerConfig):
        """Initialize the PDF processing server."""
        super().__init__(config)
        
        # Initialize PDF-specific agents
        self.file_extraction_agent = FileExtractionAgent()
        self.ocr_agent = OCRAgent()
        self.text_agent = UnifiedTextAgent()
        self.knowledge_graph_agent = EnhancedKnowledgeGraphAgent(
            graph_storage_path=self.config.knowledge_graph_path
        )
        
        logger.info("✅ PDF Processing Server initialized")
    
    async def extract_text(self, content: str, language: str = "en") -> Dict[str, Any]:
        """Extract text from PDF content.
        
        Args:
            content: Path to PDF file or PDF content
            language: Language code for processing
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            logger.info(f"📄 Extracting text from PDF: {content}")
            
            # Check if content is a file path
            if os.path.exists(content):
                # Extract text using existing PDF extraction function
                extracted_text = extract_pdf_text(content)
                
                if not extracted_text or extracted_text.strip() == "":
                    logger.warning("⚠️  No text extracted from PDF, attempting OCR")
                    # Fallback to OCR if no text extracted
                    ocr_result = await self.ocr_agent.extract_text(content)
                    extracted_text = ocr_result.get("text", "")
                
                return {
                    "success": True,
                    "text": extracted_text,
                    "source": content,
                    "language": language,
                    "extraction_method": "pdf_text_extraction",
                    "text_length": len(extracted_text),
                    "has_text": bool(extracted_text.strip())
                }
            else:
                return {
                    "success": False,
                    "error": f"PDF file not found: {content}",
                    "suggestion": "Please provide a valid PDF file path"
                }
                
        except Exception as e:
            logger.error(f"❌ Error extracting text from PDF: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content
            }
    
    async def convert_content(self, content: str, target_format: str) -> Dict[str, Any]:
        """Convert PDF content to alternative formats when needed.
        
        Args:
            content: Path to PDF file
            target_format: Target format ("image", "text", "html")
            
        Returns:
            Dictionary with converted content
        """
        try:
            logger.info(f"🔄 Converting PDF to {target_format}: {content}")
            
            if not os.path.exists(content):
                return {
                    "success": False,
                    "error": f"PDF file not found: {content}"
                }
            
            if target_format == "image":
                # Convert PDF pages to images
                images = await self._pdf_to_images(content)
                return {
                    "success": True,
                    "format": "image",
                    "images": images,
                    "page_count": len(images),
                    "source": content
                }
            
            elif target_format == "text":
                # Extract text (reuse extract_text method)
                return await self.extract_text(content)
            
            elif target_format == "html":
                # Convert to HTML format
                html_content = await self._pdf_to_html(content)
                return {
                    "success": True,
                    "format": "html",
                    "html": html_content,
                    "source": content
                }
            
            else:
                return {
                    "success": False,
                    "error": f"Unsupported target format: {target_format}",
                    "supported_formats": ["image", "text", "html"]
                }
                
        except Exception as e:
            logger.error(f"❌ Error converting PDF content: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content,
                "target_format": target_format
            }
    
    async def summarize_content(self, content: str, language: str = "en") -> Dict[str, Any]:
        """Generate comprehensive summaries of PDF content.
        
        Args:
            content: Path to PDF file or extracted text
            language: Language code for summarization
            
        Returns:
            Dictionary with summary and key points
        """
        try:
            logger.info(f"📝 Summarizing PDF content: {content}")
            
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
            logger.error(f"❌ Error summarizing PDF content: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content
            }
    
    async def translate_content(self, content: str, source_language: str, target_language: str = "en") -> Dict[str, Any]:
        """Translate PDF content to English.
        
        Args:
            content: Path to PDF file or extracted text
            source_language: Source language code
            target_language: Target language code (default: "en")
            
        Returns:
            Dictionary with translated content
        """
        try:
            logger.info(f"🌐 Translating PDF content from {source_language} to {target_language}")
            
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
            logger.error(f"❌ Error translating PDF content: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content,
                "source_language": source_language,
                "target_language": target_language
            }
    
    async def store_in_vector_db(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Store processed PDF content in vector database.
        
        Args:
            content: Path to PDF file or extracted text
            metadata: Additional metadata for storage
            
        Returns:
            Dictionary with storage result
        """
        try:
            logger.info(f"💾 Storing PDF content in vector database: {content}")
            
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
                "content_type": "pdf",
                "timestamp": asyncio.get_event_loop().time(),
                **metadata
            }
            
            # Store in vector database
            storage_result = await self.vector_store.add_text(
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
            logger.error(f"❌ Error storing PDF content in vector database: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content
            }
    
    async def create_knowledge_graph(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create knowledge graph from PDF content.
        
        Args:
            content: Path to PDF file or extracted text
            metadata: Additional metadata for knowledge graph
            
        Returns:
            Dictionary with knowledge graph result
        """
        try:
            logger.info(f"🧠 Creating knowledge graph from PDF content: {content}")
            
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
                "content_type": "pdf",
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
            logger.error(f"❌ Error creating knowledge graph from PDF content: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": content
            }
    
    async def _pdf_to_images(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Convert PDF pages to images."""
        try:
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                images.append({
                    "page": page_num + 1,
                    "image": img,
                    "width": pix.width,
                    "height": pix.height
                })
            
            doc.close()
            return images
            
        except Exception as e:
            logger.error(f"❌ Error converting PDF to images: {e}")
            return []
    
    async def _pdf_to_html(self, pdf_path: str) -> str:
        """Convert PDF to HTML format."""
        try:
            doc = fitz.open(pdf_path)
            html_content = "<html><body>"
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                html_content += f"<h2>Page {page_num + 1}</h2>"
                html_content += page.get_text("html")
            
            html_content += "</body></html>"
            doc.close()
            
            return html_content
            
        except Exception as e:
            logger.error(f"❌ Error converting PDF to HTML: {e}")
            return ""
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            # Cleanup agents
            if hasattr(self.file_extraction_agent, 'cleanup'):
                await self.file_extraction_agent.cleanup()
            if hasattr(self.ocr_agent, 'cleanup'):
                await self.ocr_agent.cleanup()
            if hasattr(self.text_agent, 'cleanup'):
                await self.text_agent.cleanup()
            if hasattr(self.knowledge_graph_agent, 'cleanup'):
                await self.knowledge_graph_agent.cleanup()
            
            logger.info("✅ PDF Processing Server cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during PDF Processing Server cleanup: {e}")


# Export for use in other modules
__all__ = ["PDFProcessingServer"]
