"""
Semantic Search Service

This module provides high-level semantic search functionality that integrates
with the VectorDBManager and provides unified search capabilities across
different content types and languages.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from loguru import logger

from src.core.vector_db import VectorDBManager
from src.core.translation_service import TranslationService
from src.config.semantic_search_config import semantic_search_config, SearchType


class SemanticSearchService:
    """High-level semantic search service with multilingual support."""

    def __init__(self):
        self.vector_db = VectorDBManager()
        self.translation_service = TranslationService()
        self.config = semantic_search_config
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour

    async def search(
        self,
        query: str,
        search_type: SearchType = SearchType.SEMANTIC,
        language: str = "en",
        content_types: Optional[List[str]] = None,
        n_results: int = 10,
        similarity_threshold: float = 0.7,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Perform semantic search with unified interface.
        
        Args:
            query: Search query
            search_type: Type of search to perform
            language: Language for search
            content_types: Content types to search
            n_results: Number of results
            similarity_threshold: Similarity threshold
            include_metadata: Whether to include metadata
            
        Returns:
            Search results with metadata
        """
        try:
            start_time = datetime.now()
            
            # Validate parameters
            validated_params = self.config.validate_search_parameters(
                n_results, similarity_threshold, language
            )
            
            # Detect query language if not specified
            if language == "auto":
                detected_lang = await self.translation_service.detect_language(query)
                language = detected_lang
                logger.info(f"Detected language: {language}")
            
            # Translate query if needed
            original_query = query
            if language != "en":
                translation_result = await self.translation_service.translate_text(
                    query, target_language="en"
                )
                query = translation_result.translated_text
                logger.info(f"Translated query: '{original_query}' -> '{query}'")
            
            # Perform search based on type
            if search_type == SearchType.SEMANTIC:
                results = await self._semantic_search(
                    query, language, content_types, 
                    validated_params["n_results"], 
                    validated_params["similarity_threshold"],
                    include_metadata
                )
            elif search_type == SearchType.CONCEPTUAL:
                results = await self._conceptual_search(
                    query, validated_params["n_results"],
                    validated_params["similarity_threshold"]
                )
            elif search_type == SearchType.MULTILINGUAL:
                results = await self._multilingual_search(
                    query, validated_params["n_results"],
                    validated_params["similarity_threshold"]
                )
            elif search_type == SearchType.CROSS_CONTENT:
                results = await self._cross_content_search(
                    query, content_types or validated_params["content_types"],
                    validated_params["n_results"],
                    validated_params["similarity_threshold"]
                )
            else:
                results = await self._semantic_search(
                    query, language, content_types,
                    validated_params["n_results"],
                    validated_params["similarity_threshold"],
                    include_metadata
                )
            
            # Translate results back if needed
            if language != "en" and isinstance(results, list):
                results = await self._translate_results(results, language)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "query": original_query,
                "search_type": search_type.value,
                "language": language,
                "results": results,
                "total_results": len(results) if isinstance(results, list) else 0,
                "processing_time": processing_time,
                "parameters": validated_params
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "search_type": search_type.value,
                "language": language,
                "results": [],
                "total_results": 0,
                "processing_time": 0
            }

    async def _semantic_search(
        self,
        query: str,
        language: str,
        content_types: Optional[List[str]],
        n_results: int,
        similarity_threshold: float,
        include_metadata: bool
    ) -> List[Dict[str, Any]]:
        """Perform semantic search."""
        return await self.vector_db.semantic_search(
            query=query,
            language=language,
            content_types=content_types,
            n_results=n_results,
            similarity_threshold=similarity_threshold,
            include_metadata=include_metadata
        )

    async def _conceptual_search(
        self,
        query: str,
        n_results: int,
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Perform conceptual search."""
        return await self.vector_db.search_by_concept(
            concept=query,
            n_results=n_results,
            similarity_threshold=similarity_threshold
        )

    async def _multilingual_search(
        self,
        query: str,
        n_results: int,
        similarity_threshold: float
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Perform multilingual search."""
        return await self.vector_db.multi_language_semantic_search(
            query=query,
            n_results=n_results,
            similarity_threshold=similarity_threshold
        )

    async def _cross_content_search(
        self,
        query: str,
        content_types: List[str],
        n_results: int,
        similarity_threshold: float
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Perform cross-content-type search."""
        return await self.vector_db.search_across_content_types(
            query=query,
            content_types=content_types,
            n_results=n_results,
            similarity_threshold=similarity_threshold
        )

    async def _translate_results(
        self,
        results: List[Dict[str, Any]],
        target_language: str
    ) -> List[Dict[str, Any]]:
        """Translate search results to target language."""
        try:
            translated_results = []
            
            for result in results:
                translated_result = result.copy()
                
                # Translate content if present
                if "content" in result and result["content"]:
                    translation = await self.translation_service.translate_text(
                        result["content"], target_language=target_language
                    )
                    translated_result["content"] = translation.translated_text
                    translated_result["original_content"] = result["content"]
                
                # Translate metadata fields if present
                if "metadata" in result and result["metadata"]:
                    metadata = result["metadata"].copy()
                    if "sentiment" in metadata:
                        sentiment_translation = await self.translation_service.translate_text(
                            metadata["sentiment"], target_language=target_language
                        )
                        metadata["sentiment"] = sentiment_translation.translated_text
                    translated_result["metadata"] = metadata
                
                translated_results.append(translated_result)
            
            return translated_results
            
        except Exception as e:
            logger.error(f"Failed to translate results: {e}")
            return results

    async def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics and index information."""
        try:
            stats = await self.vector_db.get_search_statistics()
            
            # Add configuration information
            stats["supported_languages"] = self.config.get_supported_languages()
            stats["supported_content_types"] = self.config.get_supported_content_types()
            stats["search_strategies"] = list(self.config.search_strategies.keys())
            
            return {
                "success": True,
                "statistics": stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get search statistics: {e}")
            return {
                "success": False,
                "error": str(e),
                "statistics": {},
                "timestamp": datetime.now().isoformat()
            }

    async def search_with_knowledge_graph(
        self,
        query: str,
        language: str = "en",
        n_results: int = 10,
        similarity_threshold: float = 0.7,
        include_kg_results: bool = True
    ) -> Dict[str, Any]:
        """
        Perform combined semantic search and knowledge graph search.
        
        Args:
            query: Search query
            language: Language for search
            n_results: Number of results
            similarity_threshold: Similarity threshold
            include_kg_results: Whether to include knowledge graph results
            
        Returns:
            Combined search results
        """
        try:
            start_time = datetime.now()
            
            # Perform semantic search
            semantic_results = await self.search(
                query=query,
                search_type=SearchType.SEMANTIC,
                language=language,
                n_results=n_results,
                similarity_threshold=similarity_threshold
            )
            
            # Perform knowledge graph search if requested
            kg_results = None
            if include_kg_results:
                # This would integrate with the knowledge graph agent
                # For now, we'll return the semantic search results
                kg_results = {
                    "success": True,
                    "results": [],
                    "message": "Knowledge graph search not yet integrated"
                }
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "query": query,
                "language": language,
                "semantic_search": semantic_results,
                "knowledge_graph_search": kg_results,
                "processing_time": processing_time,
                "total_results": semantic_results.get("total_results", 0)
            }
            
        except Exception as e:
            logger.error(f"Combined search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "language": language,
                "semantic_search": None,
                "knowledge_graph_search": None,
                "processing_time": 0,
                "total_results": 0
            }

    async def batch_search(
        self,
        queries: List[str],
        search_type: SearchType = SearchType.SEMANTIC,
        language: str = "en",
        n_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of search queries
            search_type: Type of search to perform
            language: Language for search
            n_results: Number of results per query
            similarity_threshold: Similarity threshold
            
        Returns:
            List of search results for each query
        """
        try:
            tasks = []
            for query in queries:
                task = self.search(
                    query=query,
                    search_type=search_type,
                    language=language,
                    n_results=n_results,
                    similarity_threshold=similarity_threshold
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "success": False,
                        "error": str(result),
                        "query": queries[i],
                        "results": [],
                        "total_results": 0
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Batch search failed: {e}")
            return [{
                "success": False,
                "error": str(e),
                "query": query,
                "results": [],
                "total_results": 0
            } for query in queries]


# Global service instance
semantic_search_service = SemanticSearchService()
