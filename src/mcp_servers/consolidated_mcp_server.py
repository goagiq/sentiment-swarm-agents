"""
Consolidated MCP Server for Sentiment Analysis System.

This module provides a unified MCP server that consolidates functionality from
44 individual MCP servers into 4 main categories with 6 core functions each.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from pydantic import BaseModel, Field

# Import core services
from core.model_manager import ModelManager
from core.vector_db import VectorDBManager
from core.improved_knowledge_graph_utility import ImprovedKnowledgeGraphUtility
from core.translation_service import TranslationService

# Import configuration
from config.mcp_config import ConsolidatedMCPServerConfig, ProcessingCategory

# Try to import FastMCP for MCP server functionality
try:
    from fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("FastMCP not available - using mock MCP server")


class BaseProcessingServer(ABC):
    """Base class for all processing servers with unified interface."""
    
    def __init__(self, config: ConsolidatedMCPServerConfig):
        self.config = config
        self.model_manager = ModelManager()
        self.vector_store = VectorDBManager()
        self.knowledge_graph = ImprovedKnowledgeGraphUtility()
        self.translation_service = TranslationService()
        
    @abstractmethod
    async def extract_text(self, content: str, language: str = "en") -> Dict[str, Any]:
        """Extract text from source content."""
        pass
    
    @abstractmethod
    async def convert_content(self, content: str, target_format: str) -> Dict[str, Any]:
        """Convert content to alternative formats when needed."""
        pass
    
    @abstractmethod
    async def summarize_content(self, content: str, language: str = "en") -> Dict[str, Any]:
        """Generate comprehensive summaries."""
        pass
    
    @abstractmethod
    async def translate_content(self, content: str, source_language: str, target_language: str = "en") -> Dict[str, Any]:
        """Translate foreign language content to English."""
        pass
    
    @abstractmethod
    async def store_in_vector_db(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Store processed content in vector database."""
        pass
    
    @abstractmethod
    async def create_knowledge_graph(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create and manage knowledge graphs."""
        pass


class ConsolidatedMCPServer:
    """Consolidated MCP server providing unified access to all processing categories."""
    
    def __init__(self, config: Optional[ConsolidatedMCPServerConfig] = None):
        """Initialize the consolidated MCP server."""
        self.config = config or ConsolidatedMCPServerConfig()
        self.mcp = None
        self.processing_servers = {}
        
        # Initialize MCP server
        self._initialize_mcp()
        
        # Initialize processing servers
        self._initialize_processing_servers()
        
        # Register tools
        self._register_tools()
        
        logger.info("✅ Consolidated MCP Server initialized successfully")
    
    def _initialize_mcp(self):
        """Initialize the MCP server using FastMCP."""
        if MCP_AVAILABLE:
            self.mcp = FastMCP("Consolidated Sentiment Analysis Server")
            logger.info("✅ FastMCP Server initialized successfully")
        else:
            logger.warning("⚠️  FastMCP not available - using mock MCP server")
            self.mcp = None
    
    def _initialize_processing_servers(self):
        """Initialize all processing servers."""
        try:
            if self.config.pdf_server.enabled:
                from .pdf_processing_server import PDFProcessingServer
                self.processing_servers["pdf"] = PDFProcessingServer(self.config)
                logger.info("✅ PDF Processing Server initialized")
            
            if self.config.audio_server.enabled:
                from .audio_processing_server import AudioProcessingServer
                self.processing_servers["audio"] = AudioProcessingServer(self.config)
                logger.info("✅ Audio Processing Server initialized")
            
            if self.config.video_server.enabled:
                from .video_processing_server import VideoProcessingServer
                self.processing_servers["video"] = VideoProcessingServer(self.config)
                logger.info("✅ Video Processing Server initialized")
            
            if self.config.website_server.enabled:
                from .website_processing_server import WebsiteProcessingServer
                self.processing_servers["website"] = WebsiteProcessingServer(self.config)
                logger.info("✅ Website Processing Server initialized")
            
            logger.info(f"✅ Initialized {len(self.processing_servers)} processing servers")
            
        except Exception as e:
            logger.error(f"❌ Error initializing processing servers: {e}")
    
    def _register_tools(self):
        """Register all MCP tools."""
        if self.mcp is None:
            logger.warning("⚠️  MCP server not available - skipping tool registration")
            return
        
        try:
            # Server management tools
            @self.mcp.tool(description="Get status of all processing servers")
            async def get_servers_status():
                """Get status of all processing servers."""
                try:
                    status = {}
                    for server_name, server in self.processing_servers.items():
                        status[server_name] = {
                            "name": server_name,
                            "status": "active",
                            "type": server.__class__.__name__,
                            "capabilities": ["extract_text", "convert_content", "summarize_content", 
                                           "translate_content", "store_in_vector_db", "create_knowledge_graph"]
                        }
                    
                    return {
                        "success": True,
                        "total_servers": len(self.processing_servers),
                        "servers": status
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            # Unified processing tools for each category
            for category, server in self.processing_servers.items():
                
                @self.mcp.tool(description=f"Extract text from {category} content")
                async def extract_text(category: str, content: str, language: str = "en"):
                    """Extract text from content."""
                    try:
                        if category not in self.processing_servers:
                            return {
                                "success": False,
                                "error": f"Category '{category}' not supported",
                                "supported_categories": list(self.processing_servers.keys())
                            }
                        
                        result = await self.processing_servers[category].extract_text(content, language)
                        return {
                            "success": True,
                            "category": category,
                            "result": result
                        }
                    except Exception as e:
                        return {
                            "success": False,
                            "error": str(e),
                            "category": category
                        }
                
                @self.mcp.tool(description=f"Convert {category} content to alternative format")
                async def convert_content(category: str, content: str, target_format: str):
                    """Convert content to alternative format."""
                    try:
                        if category not in self.processing_servers:
                            return {
                                "success": False,
                                "error": f"Category '{category}' not supported",
                                "supported_categories": list(self.processing_servers.keys())
                            }
                        
                        result = await self.processing_servers[category].convert_content(content, target_format)
                        return {
                            "success": True,
                            "category": category,
                            "result": result
                        }
                    except Exception as e:
                        return {
                            "success": False,
                            "error": str(e),
                            "category": category
                        }
                
                @self.mcp.tool(description=f"Summarize {category} content")
                async def summarize_content(category: str, content: str, language: str = "en"):
                    """Summarize content."""
                    try:
                        if category not in self.processing_servers:
                            return {
                                "success": False,
                                "error": f"Category '{category}' not supported",
                                "supported_categories": list(self.processing_servers.keys())
                            }
                        
                        result = await self.processing_servers[category].summarize_content(content, language)
                        return {
                            "success": True,
                            "category": category,
                            "result": result
                        }
                    except Exception as e:
                        return {
                            "success": False,
                            "error": str(e),
                            "category": category
                        }
                
                @self.mcp.tool(description=f"Translate {category} content to English")
                async def translate_content(category: str, content: str, source_language: str, target_language: str = "en"):
                    """Translate content to English."""
                    try:
                        if category not in self.processing_servers:
                            return {
                                "success": False,
                                "error": f"Category '{category}' not supported",
                                "supported_categories": list(self.processing_servers.keys())
                            }
                        
                        result = await self.processing_servers[category].translate_content(content, source_language, target_language)
                        return {
                            "success": True,
                            "category": category,
                            "result": result
                        }
                    except Exception as e:
                        return {
                            "success": False,
                            "error": str(e),
                            "category": category
                        }
                
                @self.mcp.tool(description=f"Store {category} content in vector database")
                async def store_in_vector_db(category: str, content: str, metadata: Dict[str, Any]):
                    """Store content in vector database."""
                    try:
                        if category not in self.processing_servers:
                            return {
                                "success": False,
                                "error": f"Category '{category}' not supported",
                                "supported_categories": list(self.processing_servers.keys())
                            }
                        
                        result = await self.processing_servers[category].store_in_vector_db(content, metadata)
                        return {
                            "success": True,
                            "category": category,
                            "result": result
                        }
                    except Exception as e:
                        return {
                            "success": False,
                            "error": str(e),
                            "category": category
                        }
                
                @self.mcp.tool(description=f"Create knowledge graph from {category} content")
                async def create_knowledge_graph(category: str, content: str, metadata: Dict[str, Any]):
                    """Create knowledge graph from content."""
                    try:
                        if category not in self.processing_servers:
                            return {
                                "success": False,
                                "error": f"Category '{category}' not supported",
                                "supported_categories": list(self.processing_servers.keys())
                            }
                        
                        result = await self.processing_servers[category].create_knowledge_graph(content, metadata)
                        return {
                            "success": True,
                            "category": category,
                            "result": result
                        }
                    except Exception as e:
                        return {
                            "success": False,
                            "error": str(e),
                            "category": category
                        }
            
            # Universal processing tool
            @self.mcp.tool(description="Process content with automatic category detection")
            async def process_content_universal(content: str, language: str = "en"):
                """Process content with automatic category detection."""
                try:
                    # Auto-detect category based on content type
                    category = self._detect_content_category(content)
                    
                    if category not in self.processing_servers:
                        return {
                            "success": False,
                            "error": f"Category '{category}' not supported",
                            "supported_categories": list(self.processing_servers.keys())
                        }
                    
                    # Extract text first
                    text_result = await self.processing_servers[category].extract_text(content, language)
                    
                    # Summarize content
                    summary_result = await self.processing_servers[category].summarize_content(content, language)
                    
                    # Store in vector database
                    metadata = {
                        "category": category,
                        "language": language,
                        "summary": summary_result.get("summary", ""),
                        "timestamp": asyncio.get_event_loop().time()
                    }
                    vector_result = await self.processing_servers[category].store_in_vector_db(content, metadata)
                    
                    # Create knowledge graph
                    kg_result = await self.processing_servers[category].create_knowledge_graph(content, metadata)
                    
                    return {
                        "success": True,
                        "category": category,
                        "language": language,
                        "results": {
                            "text_extraction": text_result,
                            "summarization": summary_result,
                            "vector_storage": vector_result,
                            "knowledge_graph": kg_result
                        }
                    }
                    
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            logger.info(f"✅ Registered {len(self.processing_servers) * 6 + 2} MCP tools")
            
        except Exception as e:
            logger.error(f"❌ Error registering tools: {e}")
    
    def _detect_content_category(self, content: str) -> str:
        """Detect content category based on content type."""
        if content.startswith(('http://', 'https://')):
            if 'youtube.com' in content or 'youtu.be' in content:
                return "video"
            else:
                return "website"
        elif os.path.exists(content):
            ext = os.path.splitext(content)[1].lower()
            if ext in ['.pdf']:
                return "pdf"
            elif ext in ['.mp3', '.wav', '.flac', '.m4a']:
                return "audio"
            elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                return "video"
            else:
                return "pdf"  # Default to PDF for unknown file types
        else:
            return "website"  # Default to website for text content
    
    async def run(self, host: str = "localhost", port: int = 8000, debug: bool = False):
        """Run the consolidated MCP server."""
        if self.mcp is None:
            logger.error("❌ MCP server not initialized")
            return
        
        try:
            logger.info(f"🚀 Starting Consolidated MCP Server on {host}:{port}")
            await self.mcp.run(host=host, port=port, debug=debug)
        except Exception as e:
            logger.error(f"❌ Error starting MCP server: {e}")
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            for server in self.processing_servers.values():
                if hasattr(server, 'cleanup'):
                    await server.cleanup()
            logger.info("✅ Consolidated MCP Server cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


def create_consolidated_mcp_server(config: Optional[ConsolidatedMCPServerConfig] = None) -> ConsolidatedMCPServer:
    """Create a consolidated MCP server instance."""
    return ConsolidatedMCPServer(config)


# Export for use in other modules
__all__ = [
    "ConsolidatedMCPServer",
    "ConsolidatedMCPServerConfig", 
    "BaseProcessingServer",
    "create_consolidated_mcp_server"
]
