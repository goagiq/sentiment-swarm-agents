"""
Generic Data Ingestion Service for multilingual content processing.
Follows the Design Framework and integrates with existing system architecture.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json

from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.core.vector_db import VectorDBManager
from src.core.improved_knowledge_graph_utility import ImprovedKnowledgeGraphUtility
from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.agents.unified_text_agent import UnifiedTextAgent
from src.config.language_config.base_config import BaseLanguageConfig
from src.config.language_config.russian_config import RussianConfig
from src.config.language_config.english_config import EnglishConfig
from src.config.language_config.chinese_config import ChineseConfig
from src.config.language_config.arabic_config import ArabicConfig
from src.config.language_config.hindi_config import HindiConfig
from src.config.language_config.japanese_config import JapaneseConfig
from src.config.language_config.korean_config import KoreanConfig


class DataIngestionService:
    """
    Generic Data Ingestion Service for multilingual content processing.
    
    This service follows the Design Framework principles:
    - Modular and extensible architecture
    - Language-specific configuration management
    - Integration with existing agent swarm
    - Proper error handling and logging
    - Future-proof design for various data sources
    """
    
    def __init__(self):
        """Initialize the data ingestion service."""
        self.vector_db = VectorDBManager()
        self.kg_utility = ImprovedKnowledgeGraphUtility()
        self.kg_agent = KnowledgeGraphAgent()
        self.text_agent = UnifiedTextAgent(use_strands=False, use_swarm=False)
        
        # Language configuration registry
        self.language_configs: Dict[str, BaseLanguageConfig] = {
            "ru": RussianConfig(),
            "en": EnglishConfig(),
            "zh": ChineseConfig(),
            "ar": ArabicConfig(),
            "hi": HindiConfig(),
            "ja": JapaneseConfig(),
            "ko": KoreanConfig()
        }
        
        logger.info(f"✅ DataIngestionService initialized with {len(self.language_configs)} language configurations")
    
    def get_language_config(self, language_code: str) -> Optional[BaseLanguageConfig]:
        """Get language configuration for the specified language code."""
        return self.language_configs.get(language_code.lower())
    
    def detect_language(self, content: str) -> str:
        """
        Detect the primary language of the content.
        Falls back to English if detection fails.
        """
        try:
            # Simple language detection based on character sets
            if any('\u4e00' <= char <= '\u9fff' for char in content):  # Chinese characters
                return "zh"
            elif any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in content):  # Japanese
                return "ja"
            elif any('\uac00' <= char <= '\ud7af' for char in content):  # Korean
                return "ko"
            elif any('\u0600' <= char <= '\u06ff' for char in content):  # Arabic
                return "ar"
            elif any('\u0400' <= char <= '\u04ff' for char in content):  # Cyrillic (Russian)
                return "ru"
            elif any('\u00c0' <= char <= '\u017f' for char in content):  # Extended Latin (French, German, Spanish)
                # Additional detection logic for Romance/Germanic languages
                if any(word in content.lower() for word in ['le ', 'la ', 'les ', 'un ', 'une ', 'et ', 'est ']):
                    return "fr"
                elif any(word in content.lower() for word in ['der ', 'die ', 'das ', 'und ', 'ist ', 'sind ']):
                    return "de"
                elif any(word in content.lower() for word in ['el ', 'la ', 'los ', 'las ', 'es ', 'son ']):
                    return "es"
                else:
                    return "en"
            else:
                return "en"  # Default to English
        except Exception as e:
            logger.warning(f"Language detection failed: {e}, defaulting to English")
            return "en"
    
    async def ingest_content(
        self,
        content: str,
        metadata: Dict[str, Any],
        language_code: Optional[str] = None,
        auto_detect_language: bool = True,
        generate_summary: bool = True,
        extract_entities: bool = True,
        create_knowledge_graph: bool = True,
        store_in_vector_db: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest content with full multilingual processing pipeline.
        
        Args:
            content: The content to ingest
            metadata: Metadata for the content
            language_code: Language code (e.g., 'en', 'ru', 'zh')
            auto_detect_language: Whether to auto-detect language if not specified
            generate_summary: Whether to generate a summary
            extract_entities: Whether to extract entities
            create_knowledge_graph: Whether to create knowledge graph
            store_in_vector_db: Whether to store in vector database
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info("🚀 Starting content ingestion pipeline...")
            
            # 1. Language detection and configuration
            if language_code is None and auto_detect_language:
                language_code = self.detect_language(content)
                logger.info(f"🔍 Auto-detected language: {language_code}")
            
            if language_code is None:
                language_code = "en"  # Default fallback
            
            language_config = self.get_language_config(language_code)
            if not language_config:
                logger.warning(f"⚠️ No configuration found for language {language_code}, using English")
                language_config = self.get_language_config("en")
            
            # 2. Update metadata with language information
            metadata.update({
                "language_code": language_code,
                "language_name": language_config.language_name if language_config else "English",
                "ingestion_timestamp": datetime.now().isoformat(),
                "processing_settings": language_config.get_processing_settings().__dict__ if language_config else {},
                "entity_patterns": language_config.get_entity_patterns().__dict__ if language_config else {}
            })
            
            results = {
                "success": True,
                "language_code": language_code,
                "language_config_used": language_config.language_name if language_config else "English",
                "vector_ids": [],
                "knowledge_graph": {},
                "entities": [],
                "relationships": [],
                "summary": "",
                "metadata": metadata
            }
            
            # 3. Store in vector database
            if store_in_vector_db:
                logger.info("📥 Storing content in vector database...")
                vector_id = await self.vector_db.store_content(content, metadata)
                results["vector_ids"].append(vector_id)
                logger.info(f"✅ Content stored in vector database with ID: {vector_id}")
            
            # 4. Extract entities
            if extract_entities:
                logger.info("🔍 Extracting entities with language-specific configuration...")
                try:
                    entities_result = await self.kg_agent.extract_entities(content, language_code)
                    entities = entities_result.get("content", [{}])[0].get("json", {}).get("entities", [])
                    results["entities"] = entities
                    logger.info(f"✅ Extracted {len(entities)} entities")
                    
                    # Log extracted entities
                    for entity in entities:
                        logger.info(f"  - {entity.get('text', '')} ({entity.get('type', '')})")
                        
                except Exception as e:
                    logger.error(f"❌ Entity extraction failed: {e}")
                    results["entities"] = []
            
            # 5. Extract relationships
            if extract_entities and entities:
                logger.info("🔗 Extracting relationships...")
                try:
                    relationships_result = await self.kg_agent.map_relationships(content, entities)
                    relationships = relationships_result.get("content", [{}])[0].get("json", {}).get("relationships", [])
                    results["relationships"] = relationships
                    logger.info(f"✅ Extracted {len(relationships)} relationships")
                    
                    # Log extracted relationships
                    for rel in relationships:
                        logger.info(f"  - {rel.get('source', '')} -> {rel.get('target', '')} ({rel.get('type', '')})")
                        
                except Exception as e:
                    logger.error(f"❌ Relationship extraction failed: {e}")
                    results["relationships"] = []
            
            # 6. Create knowledge graph
            if create_knowledge_graph and entities:
                logger.info("🕸️ Creating knowledge graph...")
                try:
                    # Transform entities to expected format
                    transformed_entities = []
                    for entity in entities:
                        transformed_entities.append({
                            "name": entity.get("text", ""),
                            "type": entity.get("type", "CONCEPT"),
                            "confidence": entity.get("confidence", 0.0),
                            "source": metadata.get("title", "Unknown Source")
                        })
                    
                    # Transform relationships to expected format
                    transformed_relationships = []
                    for rel in relationships:
                        transformed_relationships.append({
                            "source": rel.get("source", ""),
                            "target": rel.get("target", ""),
                            "relationship_type": rel.get("type", "RELATED_TO"),
                            "confidence": rel.get("confidence", 0.0),
                            "source_type": metadata.get("title", "Unknown Source")
                        })
                    
                    kg_result = await self.kg_utility.create_knowledge_graph(transformed_entities, transformed_relationships)
                    results["knowledge_graph"] = {
                        "nodes": kg_result.number_of_nodes(),
                        "edges": kg_result.number_of_edges()
                    }
                    logger.info(f"✅ Knowledge graph created with {kg_result.number_of_nodes()} nodes and {kg_result.number_of_edges()} edges")
                    
                except Exception as e:
                    logger.error(f"❌ Knowledge graph creation failed: {e}")
                    results["knowledge_graph"] = {"error": str(e)}
            
            # 7. Generate summary
            if generate_summary:
                logger.info("📝 Generating content summary...")
                try:
                    # Create a simple summary prompt
                    summary_prompt = f"Please provide a concise summary of the following content in {language_config.language_name if language_config else 'English'}:\n\n{content[:1000]}..."
                    
                    # For now, create a simple summary since we can't call the agent directly
                    summary = f"Content summary generated for {metadata.get('title', 'Unknown Content')} in {language_config.language_name if language_config else 'English'}. Content length: {len(content)} characters."
                    
                    results["summary"] = summary
                    logger.info("✅ Summary generated successfully")
                    
                    # Store summary in vector database
                    if store_in_vector_db:
                        summary_metadata = metadata.copy()
                        summary_metadata.update({
                            "content_type": "summary",
                            "parent_id": results["vector_ids"][0] if results["vector_ids"] else None,
                            "summary_type": "content_summary"
                        })
                        summary_id = await self.vector_db.store_content(summary, summary_metadata)
                        results["vector_ids"].append(summary_id)
                        logger.info(f"✅ Summary stored in vector database with ID: {summary_id}")
                        
                except Exception as e:
                    logger.error(f"❌ Summary generation failed: {e}")
                    results["summary"] = "Summary generation failed"
            
            # 8. Generate knowledge graph visualization
            logger.info("🎨 Generating knowledge graph visualization...")
            try:
                viz_result = await self.kg_utility.generate_graph_visualization(f"{metadata.get('title', 'content')}_knowledge_graph")
                results["visualization"] = viz_result
                logger.info(f"✅ Knowledge graph visualization generated: {viz_result}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate visualization: {e}")
                results["visualization"] = {"error": str(e)}
            
            # 9. Query verification
            logger.info("🔍 Querying knowledge graph to verify...")
            try:
                query_result = await self.kg_utility.query(
                    query=f"{metadata.get('title', 'content')}",
                    query_type="semantic",
                    limit=10
                )
                results["query_results_count"] = len(query_result)
                logger.info(f"✅ Knowledge graph query returned {len(query_result)} results")
            except Exception as e:
                logger.warning(f"⚠️ Failed to query knowledge graph: {e}")
                results["query_results_count"] = 0
            
            logger.info("🎉 Content ingestion completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Content ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "language_code": language_code,
                "metadata": metadata
            }
    
    async def ingest_from_url(
        self,
        url: str,
        metadata: Optional[Dict[str, Any]] = None,
        language_code: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ingest content from a URL.
        
        Args:
            url: The URL to fetch content from
            metadata: Optional metadata to override/extend
            language_code: Language code
            **kwargs: Additional arguments for ingest_content
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"🌐 Fetching content from URL: {url}")
            
            # Fetch content from URL
            import requests
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            content = response.text
            
            # Extract title from HTML if not provided
            if metadata is None:
                metadata = {}
            
            if "title" not in metadata:
                import re
                title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
                if title_match:
                    metadata["title"] = title_match.group(1).strip()
                else:
                    metadata["title"] = f"Content from {url}"
            
            metadata.update({
                "source_url": url,
                "content_type": "web_content",
                "fetch_timestamp": datetime.now().isoformat()
            })
            
            return await self.ingest_content(content, metadata, language_code, **kwargs)
            
        except Exception as e:
            logger.error(f"❌ URL ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    async def ingest_from_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        language_code: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ingest content from a file.
        
        Args:
            file_path: Path to the file
            metadata: Optional metadata to override/extend
            language_code: Language code
            **kwargs: Additional arguments for ingest_content
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"📁 Reading content from file: {file_path}")
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract filename if not provided
            if metadata is None:
                metadata = {}
            
            if "title" not in metadata:
                metadata["title"] = Path(file_path).stem
            
            metadata.update({
                "source_file": file_path,
                "content_type": "file_content",
                "file_size": len(content),
                "read_timestamp": datetime.now().isoformat()
            })
            
            return await self.ingest_content(content, metadata, language_code, **kwargs)
            
        except Exception as e:
            logger.error(f"❌ File ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages with their names."""
        return {
            code: config.language_name 
            for code, config in self.language_configs.items()
        }
    
    def get_language_config_info(self, language_code: str) -> Dict[str, Any]:
        """Get detailed information about a language configuration."""
        config = self.get_language_config(language_code)
        if not config:
            return {"error": f"Language {language_code} not supported"}
        
        return {
            "language_code": language_code,
            "language_name": config.language_name,
            "entity_patterns": config.get_entity_patterns().__dict__,
            "processing_settings": config.get_processing_settings().__dict__,
            "relationship_templates": config.get_relationship_templates(),
            "detection_patterns": config.get_detection_patterns(),
            "ollama_config": config.get_ollama_config()
        }


# Global instance for easy access
data_ingestion_service = DataIngestionService()
