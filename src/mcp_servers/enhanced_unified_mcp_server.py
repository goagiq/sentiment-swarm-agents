#!/usr/bin/env python3
"""
Enhanced Unified MCP Server with Open Library integration.
Extends the existing unified MCP server with Open Library download capabilities.
"""

import asyncio
import re
from typing import Any, Dict, Optional
from pathlib import Path

from loguru import logger

from src.mcp_servers.unified_mcp_server import UnifiedMCPServer
from src.core.vector_db import VectorDBManager
from src.core.improved_knowledge_graph_utility import ImprovedKnowledgeGraphUtility
from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.agents.web_agent_enhanced import EnhancedWebAgent
from src.core.translation_service import TranslationService


class EnhancedUnifiedMCPServer(UnifiedMCPServer):
    """
    Enhanced Unified MCP Server with Open Library integration.
    
    Extends the base UnifiedMCPServer with:
    - Open Library URL detection and processing
    - Enhanced content type detection
    - Integrated vector database storage
    - Knowledge graph generation
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize additional services for Open Library processing
        self.vector_db = VectorDBManager()
        self.kg_utility = ImprovedKnowledgeGraphUtility()
        self.kg_agent = KnowledgeGraphAgent()
        self.web_agent = EnhancedWebAgent()
        self.translation_service = TranslationService()
        
        logger.info("Enhanced Unified MCP Server initialized with Open Library support")
    
    def _detect_content_type(self, content: str) -> str:
        """Enhanced content type detection with Open Library support."""
        content_lower = content.lower()
        
        # Check for Open Library URLs first
        if 'openlibrary.org' in content_lower:
            return "open_library"
        
        # Check for other URLs
        if content.startswith(('http://', 'https://')):
            if any(ext in content_lower for ext in ['.pdf', '.doc', '.docx']):
                return "document"
            elif any(ext in content_lower for ext in ['.mp3', '.wav', '.m4a']):
                return "audio"
            elif any(ext in content_lower for ext in ['.mp4', '.avi', '.mov']):
                return "video"
            elif any(ext in content_lower for ext in ['.jpg', '.png', '.gif']):
                return "image"
            else:
                return "webpage"
        
        # Check for file paths
        if Path(content).exists():
            ext = Path(content).suffix.lower()
            if ext in ['.pdf', '.doc', '.docx', '.txt']:
                return "document"
            elif ext in ['.mp3', '.wav', '.m4a']:
                return "audio"
            elif ext in ['.mp4', '.avi', '.mov']:
                return "video"
            elif ext in ['.jpg', '.png', '.gif']:
                return "image"
        
        # Default to text
        return "text"
    
    async def _download_openlibrary_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Download content from Open Library URL."""
        try:
            # Use the web agent's _fetch_webpage method directly
            webpage_data = await self.web_agent._fetch_webpage(url)
            
            # Process the webpage data
            cleaned_text = self.web_agent._clean_webpage_text(webpage_data["html"])
            
            webpage_content = {
                "url": url,
                "title": webpage_data["title"],
                "text": cleaned_text,
                "html": webpage_data["html"],
                "status_code": webpage_data["status_code"]
            }
            
            logger.info(f"✅ Successfully downloaded Open Library content: {len(cleaned_text)} characters")
            return webpage_content
            
        except Exception as e:
            logger.error(f"❌ Error downloading Open Library content: {e}")
            return None
    
    def _extract_metadata_from_content(self, content: str, title: str, url: str = "") -> Dict[str, Any]:
        """Extract metadata from content text."""
        content_lower = content.lower()
        
        # Try to extract author
        author = "Unknown"
        author_patterns = ["by ", "author:", "written by", "author is"]
        for pattern in author_patterns:
            if pattern in content_lower:
                start_idx = content_lower.find(pattern) + len(pattern)
                end_idx = content.find("\n", start_idx)
                if end_idx == -1:
                    end_idx = start_idx + 100
                author = content[start_idx:end_idx].strip()
                break
        
        # Try to extract publication year
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, content)
        publication_year = years[0] if years else "Unknown"
        
        # Determine genre
        genre_keywords = {
            "fiction": ["novel", "story", "tale", "fiction"],
            "non-fiction": ["history", "biography", "memoir", "essay"],
            "poetry": ["poem", "poetry", "verse"],
            "drama": ["play", "drama", "theater", "theatre"],
            "science": ["science", "physics", "chemistry", "biology"],
            "philosophy": ["philosophy", "philosophical", "ethics"],
            "religion": ["religion", "religious", "spiritual", "theology"]
        }
        
        detected_genre = "Literature"
        for genre, keywords in genre_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_genre = genre.title()
                break
        
        # Extract subjects
        subjects = []
        subject_keywords = [
            "history", "war", "peace", "love", "family", "politics", 
            "society", "culture", "art", "music", "science", "philosophy",
            "religion", "nature", "travel", "adventure", "mystery"
        ]
        
        for subject in subject_keywords:
            if subject in content_lower:
                subjects.append(subject.title())
        
        return {
            "title": title,
            "author": author,
            "publication_year": publication_year,
            "genre": detected_genre,
            "category": "Classic Literature" if "classic" in content_lower else detected_genre,
            "subjects": subjects[:10],
            "source": "Open Library" if url else "Direct Input",
            "source_url": url,
            "content_type": "book_description",
            "language": "en"
        }
    
    def _generate_summary(self, content: str, title: str, author: Optional[str] = None) -> str:
        """Generate a summary of the content."""
        content_lower = content.lower()
        
        # Extract key themes
        themes = []
        theme_keywords = {
            "war": ["war", "battle", "conflict", "military"],
            "love": ["love", "romance", "relationship", "marriage"],
            "family": ["family", "parent", "child", "sibling"],
            "society": ["society", "social", "class", "aristocracy"],
            "philosophy": ["philosophy", "meaning", "purpose", "existence"],
            "history": ["history", "historical", "past", "era"]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                themes.append(theme)
        
        # Create summary
        author_text = f" by {author}" if author else ""
        themes_text = ", ".join(themes[:3]) if themes else "various themes"
        
        summary = f"{title}{author_text} is a literary work that explores {themes_text}. "
        summary += f"The book contains {len(content.split())} words and covers topics related to {', '.join(themes[:5]) if themes else 'literature and human experience'}."
        
        return summary
    
    def _register_tools(self):
        """Register enhanced tools with Open Library support."""
        if not self.mcp:
            logger.warning("MCP server not available - skipping tool registration")
            return
        
        # Enhanced process_content tool with Open Library support
        @self.mcp.tool(description="Enhanced content processing with Open Library support")
        async def process_content(
            content: str,
            content_type: str = "auto",
            language: str = "en",
            options: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Process any type of content with unified interface and Open Library support."""
            try:
                # Auto-detect content type if not specified
                if content_type == "auto":
                    content_type = self._detect_content_type(content)
                
                # Handle Open Library URLs specially
                if content_type == "open_library":
                    return await self._process_openlibrary_content(content, language, options)
                
                # Route to appropriate agent based on content type
                if content_type in ["text", "pdf"]:
                    result = await self.text_agent.process_content(
                        content, language, options
                    )
                elif content_type in ["audio", "video"]:
                    result = await self.audio_agent.process_content(
                        content, language, options
                    )
                elif content_type in ["image", "vision"]:
                    result = await self.vision_agent.process_content(
                        content, language, options
                    )
                else:
                    result = await self.text_agent.process_content(
                        content, language, options
                    )
                
                return {"success": True, "content_type": content_type, "result": result}
            except Exception as e:
                logger.error(f"Error processing content: {e}")
                return {"success": False, "error": str(e)}
        
        # Enhanced extract_text_from_content tool
        @self.mcp.tool(description="Extract text from any content type including Open Library")
        async def extract_text_from_content(
            content: str,
            content_type: str = "auto",
            language: str = "en"
        ) -> Dict[str, Any]:
            """Extract text from any content type with Open Library support."""
            try:
                if content_type == "auto":
                    content_type = self._detect_content_type(content)
                
                # Handle Open Library URLs
                if content_type == "open_library":
                    webpage_content = await self._download_openlibrary_content(content)
                    if webpage_content:
                        return {
                            "success": True,
                            "text": webpage_content.get("text", ""),
                            "title": webpage_content.get("title", ""),
                            "language": language,
                            "content_type": "open_library"
                        }
                    else:
                        return {"success": False, "error": "Failed to download Open Library content"}
                
                # Handle other content types
                if content_type == "pdf":
                    result = await self.file_agent.extract_text(content)
                elif content_type in ["audio", "video"]:
                    result = await self.audio_agent.extract_text(content)
                elif content_type in ["image", "vision"]:
                    result = await self.vision_agent.extract_text(content)
                else:
                    result = {"text": content, "language": language}
                
                return {"success": True, "result": result, "content_type": content_type}
            except Exception as e:
                logger.error(f"Error extracting text: {e}")
                return {"success": False, "error": str(e)}
        
        # New tool for Open Library specific processing
        @self.mcp.tool(description="Download and process Open Library content")
        async def download_openlibrary_content(url: str) -> Dict[str, Any]:
            """Download and process content from Open Library URL."""
            try:
                webpage_content = await self._download_openlibrary_content(url)
                
                if not webpage_content:
                    return {"success": False, "error": "Failed to download content"}
                
                # Extract metadata
                metadata = self._extract_metadata_from_content(
                    webpage_content.get("text", ""),
                    webpage_content.get("title", "Unknown"),
                    url
                )
                
                # Store in vector database
                vector_id = await self.vector_db.store_content(
                    webpage_content.get("text", ""), 
                    metadata
                )
                
                # Extract entities and create knowledge graph
                entities_result = await self.kg_agent.extract_entities(
                    webpage_content.get("text", ""), "en"
                )
                entities = entities_result.get("content", [{}])[0].get("json", {}).get("entities", [])
                
                relationships_result = await self.kg_agent.map_relationships(
                    webpage_content.get("text", ""), entities
                )
                relationships = relationships_result.get("content", [{}])[0].get("json", {}).get("relationships", [])
                
                # Create knowledge graph
                transformed_entities = [
                    {
                        "name": entity.get("text", ""),
                        "type": entity.get("type", "CONCEPT"),
                        "confidence": entity.get("confidence", 0.0),
                        "source": webpage_content.get("title", "Unknown")
                    }
                    for entity in entities
                ]
                
                transformed_relationships = [
                    {
                        "source": rel.get("source", ""),
                        "target": rel.get("target", ""),
                        "relationship_type": rel.get("type", "RELATED_TO"),
                        "confidence": rel.get("confidence", 0.0),
                        "source_type": webpage_content.get("title", "Unknown")
                    }
                    for rel in relationships
                ]
                
                kg_result = await self.kg_utility.create_knowledge_graph(transformed_entities, transformed_relationships)
                
                # Generate summary
                summary = self._generate_summary(
                    webpage_content.get("text", ""),
                    webpage_content.get("title", "Unknown"),
                    metadata.get("author")
                )
                
                # Store summary
                summary_metadata = metadata.copy()
                summary_metadata.update({
                    "content_type": "summary",
                    "parent_id": vector_id,
                    "summary_type": "book_summary"
                })
                summary_id = await self.vector_db.store_content(summary, summary_metadata)
                
                return {
                    "success": True,
                    "title": webpage_content.get("title"),
                    "content_length": len(webpage_content.get("text", "")),
                    "vector_id": vector_id,
                    "summary_id": summary_id,
                    "metadata": metadata,
                    "entities_count": len(entities),
                    "relationships_count": len(relationships),
                    "knowledge_graph_nodes": kg_result.number_of_nodes(),
                    "knowledge_graph_edges": kg_result.number_of_edges(),
                    "summary": summary[:200] + "..." if len(summary) > 200 else summary
                }
                
            except Exception as e:
                logger.error(f"Error downloading Open Library content: {e}")
                return {"success": False, "error": str(e)}
        
        # Enhanced summarize_content tool
        @self.mcp.tool(description="Generate summary of content including Open Library books")
        async def summarize_content(
            content: str,
            summary_length: str = "medium"
        ) -> Dict[str, Any]:
            """Generate summary of content with Open Library support."""
            try:
                # Check if content is Open Library URL
                if 'openlibrary.org' in content.lower():
                    webpage_content = await self._download_openlibrary_content(content)
                    if webpage_content:
                        content_text = webpage_content.get("text", "")
                        title = webpage_content.get("title", "Unknown")
                    else:
                        return {"success": False, "error": "Failed to download content"}
                else:
                    content_text = content
                    title = "Content"
                
                summary = self._generate_summary(content_text, title)
                
                return {
                    "success": True,
                    "summary": summary,
                    "title": title,
                    "content_type": "open_library" if 'openlibrary.org' in content.lower() else "text"
                }
                
            except Exception as e:
                logger.error(f"Error summarizing content: {e}")
                return {"success": False, "error": str(e)}
        
        # Enhanced translate_content tool
        @self.mcp.tool(description="Translate content to target language with Open Library support")
        async def translate_content(
            content: str,
            target_language: str,
            source_language: str = "auto"
        ) -> Dict[str, Any]:
            """Translate content to target language with Open Library support."""
            try:
                # Extract text first
                if 'openlibrary.org' in content.lower():
                    webpage_content = await self._download_openlibrary_content(content)
                    if webpage_content:
                        content_text = webpage_content.get("text", "")
                    else:
                        return {"success": False, "error": "Failed to download content"}
                else:
                    content_text = content
                
                # Translate using translation service
                translated_text = await self.translation_service.translate(
                    content_text, target_language, source_language
                )
                
                return {
                    "success": True,
                    "translated_text": translated_text,
                    "target_language": target_language,
                    "source_language": source_language,
                    "content_type": "open_library" if 'openlibrary.org' in content.lower() else "text"
                }
                
            except Exception as e:
                logger.error(f"Error translating content: {e}")
                return {"success": False, "error": str(e)}
        
        # New tool for content type detection
        @self.mcp.tool(description="Detect content type including Open Library URLs")
        async def detect_content_type(content: str) -> Dict[str, Any]:
            """Detect the type of content."""
            try:
                content_type = self._detect_content_type(content)
                is_openlibrary = 'openlibrary.org' in content.lower()
                
                return {
                    "success": True,
                    "content_type": content_type,
                    "is_openlibrary": is_openlibrary
                }
                
            except Exception as e:
                logger.error(f"Error detecting content type: {e}")
                return {"success": False, "error": str(e)}
        
        # Register the base tools from parent class
        super()._register_tools()
        
        logger.info("✅ Registered enhanced MCP tools with Open Library support")
    
    async def _process_openlibrary_content(self, url: str, language: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process Open Library content with full pipeline."""
        try:
            # Download content from Open Library
            webpage_content = await self._download_openlibrary_content(url)
            
            if not webpage_content:
                return {"success": False, "error": "Failed to download content from Open Library"}
            
            # Extract text content
            content_text = webpage_content.get("text", "")
            title = webpage_content.get("title", "Unknown Book")
            
            # Extract metadata
            metadata = self._extract_metadata_from_content(content_text, title, url)
            
            # Store in vector database
            vector_id = await self.vector_db.store_content(content_text, metadata)
            
            # Extract entities and create knowledge graph
            entities_result = await self.kg_agent.extract_entities(content_text, language)
            entities = entities_result.get("content", [{}])[0].get("json", {}).get("entities", [])
            
            relationships_result = await self.kg_agent.map_relationships(content_text, entities)
            relationships = relationships_result.get("content", [{}])[0].get("json", {}).get("relationships", [])
            
            # Create knowledge graph
            transformed_entities = [
                {
                    "name": entity.get("text", ""),
                    "type": entity.get("type", "CONCEPT"),
                    "confidence": entity.get("confidence", 0.0),
                    "source": title
                }
                for entity in entities
            ]
            
            transformed_relationships = [
                {
                    "source": rel.get("source", ""),
                    "target": rel.get("target", ""),
                    "relationship_type": rel.get("type", "RELATED_TO"),
                    "confidence": rel.get("confidence", 0.0),
                    "source_type": title
                }
                for rel in relationships
            ]
            
            kg_result = await self.kg_utility.create_knowledge_graph(transformed_entities, transformed_relationships)
            
            # Generate summary
            summary = self._generate_summary(content_text, title, metadata.get("author"))
            
            # Store summary
            summary_metadata = metadata.copy()
            summary_metadata.update({
                "content_type": "summary",
                "parent_id": vector_id,
                "summary_type": "book_summary"
            })
            summary_id = await self.vector_db.store_content(summary, summary_metadata)
            
            return {
                "success": True,
                "content_type": "open_library",
                "title": title,
                "vector_id": vector_id,
                "summary_id": summary_id,
                "entities_count": len(entities),
                "relationships_count": len(relationships),
                "knowledge_graph_nodes": kg_result.number_of_nodes(),
                "knowledge_graph_edges": kg_result.number_of_edges(),
                "summary": summary[:200] + "..." if len(summary) > 200 else summary,
                "metadata": metadata,
                "content_length": len(content_text)
            }
            
        except Exception as e:
            logger.error(f"Error processing Open Library content: {e}")
            return {"success": False, "error": str(e)}
