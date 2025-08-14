"""
Vector database manager using ChromaDB for sentiment analysis results.
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import uuid

import chromadb
from chromadb.config import Settings
from loguru import logger

from src.core.models import AnalysisResult
from src.config.settings import settings


class VectorDBManager:
    """Manages ChromaDB vector database for sentiment analysis results."""

    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = (
            persist_directory or
            str(settings.paths.cache_dir / "chroma_db")
        )

        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Initialize collections
        self._init_collections()
        logger.info(f"VectorDB initialized at {self.persist_directory}")

    def _init_collections(self):
        """Initialize ChromaDB collections."""
        try:
            # Main results collection
            self.results_collection = self.client.get_or_create_collection(
                name="sentiment_results",
                metadata={"description": "Sentiment analysis results with embeddings"}
            )

            # Aggregated results collection
            self.aggregated_collection = self.client.get_or_create_collection(
                name="aggregated_results",
                metadata={"description": "Aggregated sentiment analysis results"}
            )

            # Metadata collection for quick lookups
            self.metadata_collection = self.client.get_or_create_collection(
                name="result_metadata",
                metadata={"description": "Metadata for sentiment analysis results"}
            )

            # Semantic search collection for enhanced search capabilities
            self.semantic_collection = self.client.get_or_create_collection(
                name="semantic_search",
                metadata={
                    "description": "Semantic search index for all content types"
                }
            )

            # Multi-language search collection
            self.multilingual_collection = self.client.get_or_create_collection(
                name="multilingual_search",
                metadata={
                    "description": "Multi-language semantic search index"
                }
            )

            logger.info("ChromaDB collections initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collections: {e}")
            raise

    async def store_result(self, result: AnalysisResult) -> str:
        """Store a sentiment analysis result in the vector database."""
        try:
            # Generate unique ID if not present
            if not hasattr(result, 'id'):
                result.id = str(uuid.uuid4())

            # Create document for storage
            document = self._result_to_document(result)

            # Store in results collection
            self.results_collection.add(
                documents=[document["text"]],
                metadatas=[document["metadata"]],
                ids=[result.id]
            )

            # Store metadata separately for quick access
            self.metadata_collection.add(
                documents=[json.dumps(document["metadata"])],
                metadatas=[document["metadata"]],
                ids=[f"meta_{result.id}"]
            )

            # Index for semantic search
            await self._index_for_semantic_search(result.id, document)

            logger.info(f"Stored result {result.id} in vector database")
            return result.id

        except Exception as e:
            logger.error(f"Failed to store result in vector database: {e}")
            raise

    async def _index_for_semantic_search(self, result_id: str, document: Dict[str, Any]):
        """Index content for semantic search."""
        try:
            # Create semantic search document
            semantic_doc = {
                "content": document["text"],
                "content_type": document["metadata"].get("content_type", "text"),
                "language": document["metadata"].get("language", "en"),
                "source_id": result_id,
                "timestamp": document["metadata"].get("timestamp", datetime.now().isoformat()),
                "sentiment": document["metadata"].get("sentiment_label", "unknown"),
                "confidence": document["metadata"].get("sentiment_confidence", 0.0)
            }

            # Store in semantic search collection
            self.semantic_collection.add(
                documents=[semantic_doc["content"]],
                metadatas=[semantic_doc],
                ids=[f"semantic_{result_id}"]
            )

            # Store in multilingual collection if not English
            if semantic_doc["language"] != "en":
                self.multilingual_collection.add(
                    documents=[semantic_doc["content"]],
                    metadatas=[semantic_doc],
                    ids=[f"multilingual_{result_id}"]
                )

            logger.debug(f"Indexed {result_id} for semantic search")

        except Exception as e:
            logger.error(f"Failed to index for semantic search: {e}")

    async def semantic_search(
        self,
        query: str,
        language: str = "en",
        content_types: Optional[List[str]] = None,
        n_results: int = 10,
        similarity_threshold: float = 0.7,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search across all indexed content.
        
        Args:
            query: Search query
            language: Language filter (use "all" for all languages)
            content_types: List of content types to search (None for all)
            n_results: Number of results to return
            similarity_threshold: Minimum similarity score
            include_metadata: Whether to include full metadata
            
        Returns:
            List of search results with similarity scores
        """
        try:
            start_time = datetime.now()
            
            # Determine which collection to search
            if language == "all":
                collection = self.semantic_collection
            elif language == "en":
                collection = self.semantic_collection
            else:
                collection = self.multilingual_collection

            # Perform semantic search
            search_results = collection.query(
                query_texts=[query],
                n_results=n_results * 2,  # Get more results for filtering
                where={"language": language} if language != "all" else None
            )

            # Process and filter results
            processed_results = []
            for i in range(len(search_results["ids"][0])):
                distance = search_results["distances"][0][i]
                similarity = 1.0 - distance
                
                if similarity >= similarity_threshold:
                    metadata = search_results["metadatas"][0][i]
                    
                    # Filter by content type if specified
                    if content_types and metadata.get("content_type") not in content_types:
                        continue
                    
                    result = {
                        "id": search_results["ids"][0][i],
                        "content": search_results["documents"][0][i],
                        "similarity": similarity,
                        "content_type": metadata.get("content_type", "unknown"),
                        "language": metadata.get("language", "en"),
                        "source_id": metadata.get("source_id"),
                        "timestamp": metadata.get("timestamp"),
                        "sentiment": metadata.get("sentiment", "unknown"),
                        "confidence": metadata.get("confidence", 0.0)
                    }
                    
                    if include_metadata:
                        result["metadata"] = metadata
                    
                    processed_results.append(result)
                    
                    # Stop if we have enough results
                    if len(processed_results) >= n_results:
                        break

            # Sort by similarity score
            processed_results.sort(key=lambda x: x["similarity"], reverse=True)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Semantic search completed in {processing_time:.3f}s, found {len(processed_results)} results")
            
            return processed_results

        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return []

    async def multi_language_semantic_search(
        self,
        query: str,
        target_languages: List[str] = None,
        n_results: int = 10,
        similarity_threshold: float = 0.7
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform semantic search across multiple languages.
        
        Args:
            query: Search query
            target_languages: List of target languages (None for all)
            n_results: Number of results per language
            similarity_threshold: Minimum similarity score
            
        Returns:
            Dictionary with language as key and results as value
        """
        try:
            if target_languages is None:
                target_languages = ["en", "zh", "ru", "ja", "ko", "ar", "hi"]
            
            results = {}
            
            for language in target_languages:
                language_results = await self.semantic_search(
                    query=query,
                    language=language,
                    n_results=n_results,
                    similarity_threshold=similarity_threshold
                )
                
                if language_results:
                    results[language] = language_results
            
            logger.info(f"Multi-language search completed for {len(results)} languages")
            return results

        except Exception as e:
            logger.error(f"Failed to perform multi-language semantic search: {e}")
            return {}

    async def search_by_concept(
        self,
        concept: str,
        n_results: int = 10,
        similarity_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search for content related to a specific concept, even if exact terms don't match.
        
        Args:
            concept: Concept to search for
            n_results: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of conceptually related content
        """
        try:
            # Use a lower threshold for conceptual search
            results = await self.semantic_search(
                query=concept,
                language="all",
                n_results=n_results,
                similarity_threshold=similarity_threshold
            )
            
            # Filter for conceptual relevance
            conceptual_results = []
            for result in results:
                # Additional filtering could be added here based on content analysis
                conceptual_results.append(result)
            
            logger.info(f"Conceptual search found {len(conceptual_results)} related results")
            return conceptual_results

        except Exception as e:
            logger.error(f"Failed to perform conceptual search: {e}")
            return []

    async def search_across_content_types(
        self,
        query: str,
        content_types: List[str],
        n_results: int = 10,
        similarity_threshold: float = 0.7
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across specific content types.
        
        Args:
            query: Search query
            content_types: List of content types to search
            n_results: Number of results per content type
            similarity_threshold: Minimum similarity score
            
        Returns:
            Dictionary with content type as key and results as value
        """
        try:
            results = {}
            
            for content_type in content_types:
                type_results = await self.semantic_search(
                    query=query,
                    content_types=[content_type],
                    n_results=n_results,
                    similarity_threshold=similarity_threshold
                )
                
                if type_results:
                    results[content_type] = type_results
            
            logger.info(f"Cross-content-type search completed for {len(results)} types")
            return results

        except Exception as e:
            logger.error(f"Failed to perform cross-content-type search: {e}")
            return {}

    async def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search index."""
        try:
            stats = {
                "total_documents": self.semantic_collection.count(),
                "multilingual_documents": self.multilingual_collection.count(),
                "languages": {},
                "content_types": {},
                "recent_activity": {}
            }
            
            # Get sample documents for analysis
            sample_results = self.semantic_collection.get(
                limit=min(1000, self.semantic_collection.count())
            )
            
            if sample_results["metadatas"]:
                # Analyze languages
                for metadata in sample_results["metadatas"]:
                    lang = metadata.get("language", "unknown")
                    stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
                    
                    content_type = metadata.get("content_type", "unknown")
                    stats["content_types"][content_type] = stats["content_types"].get(content_type, 0) + 1
            
            logger.info(f"Search statistics: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to get search statistics: {e}")
            return {}

    async def find_similar_content(
        self, 
        content: str, 
        threshold: float = 0.95,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar content in the vector database."""
        try:
            # Search for similar content
            search_results = self.results_collection.query(
                query_texts=[content],
                n_results=n_results
            )

            similar_results = []
            for i in range(len(search_results["ids"][0])):
                distance = search_results["distances"][0][i]
                similarity = 1.0 - distance  # Convert distance to similarity
                
                if similarity >= threshold:
                    result = {
                        "id": search_results["ids"][0][i],
                        "text": search_results["documents"][0][i],
                        "metadata": search_results["metadatas"][0][i],
                        "similarity": similarity,
                        "distance": distance
                    }
                    similar_results.append(result)

            logger.info(f"Found {len(similar_results)} similar results")
            return similar_results

        except Exception as e:
            logger.error(f"Failed to find similar content: {e}")
            return []

    async def check_content_duplicate(
        self, 
        content: str, 
        threshold: float = 0.98
    ) -> Optional[Dict[str, Any]]:
        """Check if content is a duplicate of existing content."""
        try:
            similar_results = await self.find_similar_content(content, threshold, 1)
            
            if similar_results:
                return similar_results[0]
            
            return None

        except Exception as e:
            logger.error(f"Failed to check content duplicate: {e}")
            return None

    def sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata to ensure ChromaDB compatibility."""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                sanitized[key] = value
            elif isinstance(value, dict):
                # Convert dict to JSON string
                sanitized[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, list):
                # Convert list to JSON string
                sanitized[key] = json.dumps(value, ensure_ascii=False)
            else:
                # Convert other types to string
                sanitized[key] = str(value)
        return sanitized

    def _result_to_document(self, result: AnalysisResult) -> Dict[str, Any]:
        """Convert AnalysisResult to document format for ChromaDB."""
        # Create text representation for embedding
        # Prioritize full content over summaries for better search and retrieval
        text_content = self._get_full_content_for_storage(result)

        # Create metadata
        metadata = {
            "request_id": result.request_id,
            "data_type": (
                result.data_type.value if hasattr(result.data_type, 'value')
                else str(result.data_type)
            ),
            "sentiment_label": result.sentiment.label,
            "sentiment_confidence": result.sentiment.confidence,
            "processing_time": result.processing_time,
            "status": result.status,
            "model_used": result.model_used,
            "language": getattr(result, 'language', 'en'),
            "timestamp": datetime.now().isoformat(),
            "agent_id": result.metadata.get("agent_id", "unknown"),
            "method": result.metadata.get("method", "unknown"),
            "quality_score": result.quality_score,
            "reflection_enabled": result.reflection_enabled,
            "content_type": result.metadata.get(
                "content_type", "full_content"
            ),
            "has_full_transcription": result.metadata.get(
                "has_full_transcription", False
            ),
            "has_translation": result.metadata.get("has_translation", False)
        }

        # Add sentiment scores if available
        if hasattr(result.sentiment, 'scores') and result.sentiment.scores:
            metadata["sentiment_scores"] = json.dumps(result.sentiment.scores)

        # Add reasoning if available
        if hasattr(result.sentiment, 'reasoning') and result.sentiment.reasoning:
            metadata["reasoning"] = result.sentiment.reasoning[:500]

        # Filter out None values from metadata (ChromaDB doesn't accept None values)
        metadata = {k: v for k, v in metadata.items() if v is not None}

        # Sanitize metadata for ChromaDB compatibility
        metadata = self.sanitize_metadata(metadata)

        return {
            "text": text_content,
            "metadata": metadata
        }

    def _get_full_content_for_storage(self, result: AnalysisResult) -> str:
        """Get the most appropriate content for vector storage, prioritizing full content."""
        # Check if we have full transcription/translation in metadata
        full_transcription = result.metadata.get("full_transcription")
        full_translation = result.metadata.get("full_translation")
        
        # Priority order: full transcription > full translation > extracted_text > raw_content
        if full_transcription:
            return full_transcription
        elif full_translation:
            return full_translation
        elif result.extracted_text:
            # Check if extracted_text is actually full content (not a summary)
            if self._is_full_content(result.extracted_text, result.metadata):
                return result.extracted_text
            else:
                # If it's a summary, try to get full content from metadata
                full_content = result.metadata.get("full_content")
                if full_content:
                    return full_content
                # Fall back to extracted_text if no full content available
                return result.extracted_text
        else:
            return str(result.raw_content) or ""

    def _is_full_content(self, text: str, metadata: Dict[str, Any]) -> bool:
        """Determine if the text is full content or a summary."""
        # Check metadata flags first
        if metadata.get("is_summary", False):
            return False
        if metadata.get("is_full_content", True):
            return True
        
        # Heuristic: if text is very short compared to expected full content, it's likely a summary
        expected_min_length = metadata.get("expected_min_length", 100)
        if len(text) < expected_min_length:
            return False
        
        # Check for summary indicators in text
        summary_indicators = [
            "summary", "key points", "main points", "overview", "brief",
            "in summary", "to summarize", "key takeaways"
        ]
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in summary_indicators):
            return False
        
        return True

    async def search_similar_results(
        self,
        query: str,
        n_results: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar results based on text content."""
        try:
            # Search in results collection
            search_results = self.results_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata
            )

            # Format results
            formatted_results = []
            for i in range(len(search_results["ids"][0])):
                result = {
                    "id": search_results["ids"][0][i],
                    "text": search_results["documents"][0][i],
                    "metadata": search_results["metadatas"][0][i],
                    "distance": (
                        search_results["distances"][0][i]
                        if "distances" in search_results else None
                    )
                }
                formatted_results.append(result)

            logger.info(f"Found {len(formatted_results)} similar results for query")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search similar results: {e}")
            return []

    async def get_results_by_filter(
        self,
        filter_metadata: Dict[str, Any],
        n_results: int = 100
    ) -> List[Dict[str, Any]]:
        """Get results filtered by metadata."""
        try:
            results = self.results_collection.get(
                where=filter_metadata,
                limit=n_results
            )

            formatted_results = []
            for i in range(len(results["ids"])):
                result = {
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i]
                }
                formatted_results.append(result)

            logger.info(f"Retrieved {len(formatted_results)} results with filter")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to get results by filter: {e}")
            return []

    async def aggregate_results(
        self,
        filter_metadata: Optional[Dict[str, Any]] = None,
        group_by: List[str] = None
    ) -> Dict[str, Any]:
        """Aggregate sentiment analysis results."""
        try:
            # Get all results (or filtered results)
            if filter_metadata:
                results = await self.get_results_by_filter(
                    filter_metadata, n_results=10000
                )
            else:
                results = await self.get_results_by_filter({}, n_results=10000)

            if not results:
                return {"error": "No results found for aggregation"}

            # Basic aggregation
            aggregation = {
                "total_results": len(results),
                "sentiment_distribution": {},
                "confidence_stats": {
                    "min": float('inf'),
                    "max": float('-inf'),
                    "avg": 0.0,
                    "total": 0.0
                },
                "processing_time_stats": {
                    "min": float('inf'),
                    "max": float('-inf'),
                    "avg": 0.0,
                    "total": 0.0
                },
                "data_type_distribution": {},
                "model_distribution": {},
                "agent_distribution": {},
                "timestamp_range": {
                    "earliest": None,
                    "latest": None
                }
            }

            # Process each result
            for result in results:
                metadata = result["metadata"]

                # Sentiment distribution
                sentiment = metadata.get("sentiment_label", "unknown")
                aggregation["sentiment_distribution"][sentiment] = (
                    aggregation["sentiment_distribution"].get(sentiment, 0) + 1
                )

                # Confidence stats
                confidence = metadata.get("sentiment_confidence", 0.0)
                if confidence > 0:
                    aggregation["confidence_stats"]["min"] = min(
                        aggregation["confidence_stats"]["min"], confidence
                    )
                    aggregation["confidence_stats"]["max"] = max(
                        aggregation["confidence_stats"]["max"], confidence
                    )
                    aggregation["confidence_stats"]["total"] += confidence

                # Processing time stats
                processing_time = metadata.get("processing_time", 0.0)
                if processing_time > 0:
                    aggregation["processing_time_stats"]["min"] = min(
                        aggregation["processing_time_stats"]["min"], processing_time
                    )
                    aggregation["processing_time_stats"]["max"] = max(
                        aggregation["processing_time_stats"]["max"], processing_time
                    )
                    aggregation["processing_time_stats"]["total"] += processing_time

                # Data type distribution
                data_type = metadata.get("data_type", "unknown")
                aggregation["data_type_distribution"][data_type] = (
                    aggregation["data_type_distribution"].get(data_type, 0) + 1
                )

                # Model distribution
                model = metadata.get("model_used", "unknown")
                aggregation["model_distribution"][model] = (
                    aggregation["model_distribution"].get(model, 0) + 1
                )

                # Agent distribution
                agent = metadata.get("agent_id", "unknown")
                aggregation["agent_distribution"][agent] = (
                    aggregation["agent_distribution"].get(agent, 0) + 1
                )

                # Timestamp range
                timestamp = metadata.get("timestamp")
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        if (not aggregation["timestamp_range"]["earliest"] or
                                dt < aggregation["timestamp_range"]["earliest"]):
                            aggregation["timestamp_range"]["earliest"] = dt
                        if (not aggregation["timestamp_range"]["latest"] or
                                dt > aggregation["timestamp_range"]["latest"]):
                            aggregation["timestamp_range"]["latest"] = dt
                    except Exception:
                        pass

            # Calculate averages
            if aggregation["confidence_stats"]["total"] > 0:
                aggregation["confidence_stats"]["avg"] = (
                    aggregation["confidence_stats"]["total"] / len(results)
                )
                aggregation["confidence_stats"]["min"] = (
                    aggregation["confidence_stats"]["min"]
                    if aggregation["confidence_stats"]["min"] != float('inf') else 0.0
                )
                aggregation["confidence_stats"]["max"] = (
                    aggregation["confidence_stats"]["max"]
                    if aggregation["confidence_stats"]["max"] != float('-inf') else 0.0
                )

            if aggregation["processing_time_stats"]["total"] > 0:
                aggregation["processing_time_stats"]["avg"] = (
                    aggregation["processing_time_stats"]["total"] / len(results)
                )
                aggregation["processing_time_stats"]["min"] = (
                    aggregation["processing_time_stats"]["min"]
                    if aggregation["processing_time_stats"]["min"] != float('inf') else 0.0
                )
                aggregation["processing_time_stats"]["max"] = (
                    aggregation["processing_time_stats"]["max"]
                    if aggregation["processing_time_stats"]["max"] != float('-inf') else 0.0
                )

            # Convert datetime objects to strings for JSON serialization
            if aggregation["timestamp_range"]["earliest"]:
                aggregation["timestamp_range"]["earliest"] = (
                    aggregation["timestamp_range"]["earliest"].isoformat()
                )
            if aggregation["timestamp_range"]["latest"]:
                aggregation["timestamp_range"]["latest"] = (
                    aggregation["timestamp_range"]["latest"].isoformat()
                )

            # Store aggregated results
            aggregation_id = f"agg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.aggregated_collection.add(
                documents=[json.dumps(aggregation)],
                metadatas=[{
                    "aggregation_id": aggregation_id,
                    "timestamp": datetime.now().isoformat(),
                    "filter_metadata": filter_metadata or {},
                    "group_by": group_by or []
                }],
                ids=[aggregation_id]
            )

            logger.info(
                f"Generated aggregation {aggregation_id} with {len(results)} results"
            )
            return aggregation

        except Exception as e:
            logger.error(f"Failed to aggregate results: {e}")
            return {"error": f"Aggregation failed: {str(e)}"}

    async def get_aggregation_history(
        self,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get history of aggregations."""
        try:
            results = self.aggregated_collection.get(limit=limit)

            formatted_results = []
            for i in range(len(results["ids"])):
                result = {
                    "id": results["ids"][i],
                    "data": json.loads(results["documents"][i]),
                    "metadata": results["metadatas"][i]
                }
                formatted_results.append(result)

            # Sort by timestamp (newest first)
            formatted_results.sort(
                key=lambda x: x["metadata"]["timestamp"],
                reverse=True
            )

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to get aggregation history: {e}")
            return []

    async def delete_result(self, result_id: str) -> bool:
        """Delete a specific result from the database."""
        try:
            # Delete from all collections
            self.results_collection.delete(ids=[result_id])
            self.metadata_collection.delete(ids=[f"meta_{result_id}"])
            self.semantic_collection.delete(ids=[f"semantic_{result_id}"])
            self.multilingual_collection.delete(ids=[f"multilingual_{result_id}"])

            logger.info(f"Deleted result {result_id} from vector database")
            return True

        except Exception as e:
            logger.error(f"Failed to delete result {result_id}: {e}")
            return False

    async def clear_database(self) -> bool:
        """Clear all data from the database."""
        try:
            # Reset all collections
            self.client.reset()

            # Reinitialize collections
            self._init_collections()

            logger.info("Vector database cleared and reinitialized")
            return True

        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            stats = {
                "persist_directory": self.persist_directory,
                "collections": {},
                "total_documents": 0
            }

            # Get collection info
            for collection_name in [
                "sentiment_results", "aggregated_results", "result_metadata",
                "semantic_search", "multilingual_search"
            ]:
                try:
                    collection = self.client.get_collection(collection_name)
                    count = collection.count()
                    stats["collections"][collection_name] = {
                        "document_count": count,
                        "metadata": collection.metadata
                    }
                    stats["total_documents"] += count
                except Exception as e:
                    stats["collections"][collection_name] = {
                        "error": str(e),
                        "document_count": 0
                    }

            return stats

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": f"Failed to get stats: {str(e)}"}

    async def add_texts(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add texts to a specific collection."""
        try:
            # Get or create collection
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": f"Collection for {collection_name}"}
            )

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in texts]

            # Prepare metadatas
            if metadatas is None:
                metadatas = [{} for _ in texts]

            # Add to collection
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(texts)} texts to collection {collection_name}")
            return ids

        except Exception as e:
            logger.error(f"Failed to add texts to collection {collection_name}: {e}")
            raise

    async def query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query a specific collection."""
        try:
            # Get collection
            collection = self.client.get_collection(collection_name)

            # Prepare query parameters
            query_params = {
                "query_texts": [query_text],
                "n_results": n_results
            }

            # Add filter if provided
            if filter_metadata:
                query_params["where"] = filter_metadata

            # Perform query
            results = collection.query(**query_params)

            # Format results
            formatted_results = []
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    result = {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": 1.0 - results["distances"][0][i] if results["distances"] else 1.0
                    }
                    formatted_results.append(result)

            logger.info(f"Query returned {len(formatted_results)} results from {collection_name}")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to query collection {collection_name}: {e}")
            return []

    async def store_content(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store arbitrary content in the vector database."""
        try:
            # Generate unique ID
            content_id = str(uuid.uuid4())
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            # Add default metadata
            metadata.update({
                "content_type": "text",
                "language": "en",
                "timestamp": datetime.now().isoformat(),
                "source": "manual_upload"
            })
            
            # Sanitize metadata
            sanitized_metadata = self.sanitize_metadata(metadata)
            
            # Store in semantic search collection
            self.semantic_collection.add(
                documents=[content],
                metadatas=[sanitized_metadata],
                ids=[content_id]
            )
            
            # Store in multilingual collection if not English
            if sanitized_metadata.get("language", "en") != "en":
                self.multilingual_collection.add(
                    documents=[content],
                    metadatas=[sanitized_metadata],
                    ids=[f"multilingual_{content_id}"]
                )
            
            # Store metadata separately for quick access
            self.metadata_collection.add(
                documents=[json.dumps(sanitized_metadata)],
                metadatas=[sanitized_metadata],
                ids=[f"meta_{content_id}"]
            )
            
            logger.info(f"Stored content {content_id} in vector database")
            return content_id
            
        except Exception as e:
            logger.error(f"Failed to store content in vector database: {e}")
            raise


# Global instance
vector_db = VectorDBManager()
