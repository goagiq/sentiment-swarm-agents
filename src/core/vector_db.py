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
        self.persist_directory = persist_directory or str(settings.paths.cache_dir / "chroma_db")
        
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
            
            logger.info(f"Stored result {result.id} in vector database")
            return result.id
            
        except Exception as e:
            logger.error(f"Failed to store result in vector database: {e}")
            raise
    
    def _result_to_document(self, result: AnalysisResult) -> Dict[str, Any]:
        """Convert AnalysisResult to document format for ChromaDB."""
        # Create text representation for embedding
        text_content = result.extracted_text or str(result.raw_content) or ""
        
        # Create metadata
        metadata = {
            "request_id": result.request_id,
            "data_type": result.data_type.value if hasattr(result.data_type, 'value') else str(result.data_type),
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
            "reflection_enabled": result.reflection_enabled
        }
        
        # Add sentiment scores if available
        if hasattr(result.sentiment, 'scores') and result.sentiment.scores:
            metadata["sentiment_scores"] = json.dumps(result.sentiment.scores)
        
        # Add reasoning if available
        if hasattr(result.sentiment, 'reasoning') and result.sentiment.reasoning:
            metadata["reasoning"] = result.sentiment.reasoning[:500]  # Truncate long reasoning
        
        # Filter out None values from metadata (ChromaDB doesn't accept None values)
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        return {
            "text": text_content,
            "metadata": metadata
        }
    
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
                    "distance": search_results["distances"][0][i] if "distances" in search_results else None
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
                results = await self.get_results_by_filter(filter_metadata, n_results=10000)
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
                aggregation["sentiment_distribution"][sentiment] = \
                    aggregation["sentiment_distribution"].get(sentiment, 0) + 1
                
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
                aggregation["data_type_distribution"][data_type] = \
                    aggregation["data_type_distribution"].get(data_type, 0) + 1
                
                # Model distribution
                model = metadata.get("model_used", "unknown")
                aggregation["model_distribution"][model] = \
                    aggregation["model_distribution"].get(model, 0) + 1
                
                # Agent distribution
                agent = metadata.get("agent_id", "unknown")
                aggregation["agent_distribution"][agent] = \
                    aggregation["agent_distribution"].get(agent, 0) + 1
                
                # Timestamp range
                timestamp = metadata.get("timestamp")
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        if not aggregation["timestamp_range"]["earliest"] or dt < aggregation["timestamp_range"]["earliest"]:
                            aggregation["timestamp_range"]["earliest"] = dt
                        if not aggregation["timestamp_range"]["latest"] or dt > aggregation["timestamp_range"]["latest"]:
                            aggregation["timestamp_range"]["latest"] = dt
                    except:
                        pass
            
            # Calculate averages
            if aggregation["confidence_stats"]["total"] > 0:
                aggregation["confidence_stats"]["avg"] = \
                    aggregation["confidence_stats"]["total"] / len(results)
                aggregation["confidence_stats"]["min"] = \
                    aggregation["confidence_stats"]["min"] if aggregation["confidence_stats"]["min"] != float('inf') else 0.0
                aggregation["confidence_stats"]["max"] = \
                    aggregation["confidence_stats"]["max"] if aggregation["confidence_stats"]["max"] != float('-inf') else 0.0
            
            if aggregation["processing_time_stats"]["total"] > 0:
                aggregation["processing_time_stats"]["avg"] = \
                    aggregation["processing_time_stats"]["total"] / len(results)
                aggregation["processing_time_stats"]["min"] = \
                    aggregation["processing_time_stats"]["min"] if aggregation["processing_time_stats"]["min"] != float('inf') else 0.0
                aggregation["processing_time_stats"]["max"] = \
                    aggregation["processing_time_stats"]["max"] if aggregation["processing_time_stats"]["max"] != float('-inf') else 0.0
            
            # Convert datetime objects to strings for JSON serialization
            if aggregation["timestamp_range"]["earliest"]:
                aggregation["timestamp_range"]["earliest"] = aggregation["timestamp_range"]["earliest"].isoformat()
            if aggregation["timestamp_range"]["latest"]:
                aggregation["timestamp_range"]["latest"] = aggregation["timestamp_range"]["latest"].isoformat()
            
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
            
            logger.info(f"Generated aggregation {aggregation_id} with {len(results)} results")
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
            for collection_name in ["sentiment_results", "aggregated_results", "result_metadata"]:
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


# Global instance
vector_db = VectorDBManager()
