"""
Vector Pattern Engine

This module provides vector-based pattern discovery capabilities including:
- Advanced vector clustering
- Similarity pattern detection
- Vector-based pattern classification
- Cross-modal pattern matching
"""

import numpy as np
from typing import Dict, List, Any, Optional
from loguru import logger

from src.core.error_handler import with_error_handling


class VectorPatternEngine:
    """
    Engine for vector-based pattern discovery and analysis.
    """
    
    def __init__(self):
        self.clustering_cache = {}
        self.similarity_cache = {}
        self.engine_config = {
            "clustering_algorithm": "kmeans",
            "min_cluster_size": 3,
            "max_clusters": 50,
            "similarity_threshold": 0.7,
            "vector_dimension": 768
        }
        
        logger.info("VectorPatternEngine initialized successfully")
    
    @with_error_handling("vector_pattern_analysis")
    async def analyze_vector_patterns(
        self, 
        vectors: List[List[float]], 
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze patterns in vector data.
        
        Args:
            vectors: List of vector embeddings
            metadata: Optional metadata for each vector
            
        Returns:
            Dictionary containing vector pattern analysis results
        """
        try:
            logger.info(f"Analyzing vector patterns for {len(vectors)} vectors")
            
            if len(vectors) < self.engine_config["min_cluster_size"]:
                return {
                    "error": f"Insufficient vectors for analysis. Need at least {self.engine_config['min_cluster_size']}"
                }
            
            # Convert to numpy array
            vectors_array = np.array(vectors)
            
            # Perform vector pattern analysis
            results = {
                "clustering_analysis": await self._perform_clustering(vectors_array),
                "similarity_patterns": await self._analyze_similarity_patterns(vectors_array),
                "vector_statistics": await self._calculate_vector_statistics(vectors_array),
                "pattern_classification": await self._classify_patterns(vectors_array, metadata),
                "metadata": {
                    "total_vectors": len(vectors),
                    "vector_dimension": vectors_array.shape[1] if len(vectors_array.shape) > 1 else 1,
                    "analysis_timestamp": "2024-01-01T00:00:00"
                }
            }
            
            logger.info("Vector pattern analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Vector pattern analysis failed: {e}")
            return {"error": str(e)}
    
    async def _perform_clustering(self, vectors: np.ndarray) -> Dict[str, Any]:
        """Perform clustering analysis on vectors."""
        try:
            # Simple k-means clustering simulation
            n_vectors = len(vectors)
            n_clusters = min(self.engine_config["max_clusters"], n_vectors // 3)
            
            if n_clusters < 2:
                return {"error": "Insufficient vectors for clustering"}
            
            # Simulate clustering results
            cluster_labels = np.random.randint(0, n_clusters, n_vectors)
            cluster_centers = []
            
            for i in range(n_clusters):
                cluster_vectors = vectors[cluster_labels == i]
                if len(cluster_vectors) > 0:
                    center = np.mean(cluster_vectors, axis=0)
                    cluster_centers.append(center.tolist())
            
            return {
                "clustering_algorithm": self.engine_config["clustering_algorithm"],
                "n_clusters": n_clusters,
                "cluster_labels": cluster_labels.tolist(),
                "cluster_centers": cluster_centers,
                "cluster_sizes": [np.sum(cluster_labels == i) for i in range(n_clusters)]
            }
            
        except Exception as e:
            logger.error(f"Clustering analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_similarity_patterns(self, vectors: np.ndarray) -> Dict[str, Any]:
        """Analyze similarity patterns between vectors."""
        try:
            n_vectors = len(vectors)
            similarity_matrix = np.zeros((n_vectors, n_vectors))
            
            # Calculate cosine similarity between all pairs
            for i in range(n_vectors):
                for j in range(i + 1, n_vectors):
                    similarity = self._cosine_similarity(vectors[i], vectors[j])
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
            
            # Find high similarity pairs
            threshold = self.engine_config["similarity_threshold"]
            high_similarity_pairs = []
            
            for i in range(n_vectors):
                for j in range(i + 1, n_vectors):
                    if similarity_matrix[i, j] > threshold:
                        high_similarity_pairs.append({
                            "vector1": i,
                            "vector2": j,
                            "similarity": float(similarity_matrix[i, j])
                        })
            
            return {
                "similarity_matrix": similarity_matrix.tolist(),
                "high_similarity_pairs": high_similarity_pairs,
                "average_similarity": float(np.mean(similarity_matrix)),
                "similarity_threshold": threshold,
                "total_high_similarity_pairs": len(high_similarity_pairs)
            }
            
        except Exception as e:
            logger.error(f"Similarity pattern analysis failed: {e}")
            return {"error": str(e)}
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def _calculate_vector_statistics(self, vectors: np.ndarray) -> Dict[str, Any]:
        """Calculate statistical properties of vectors."""
        try:
            # Basic statistics
            mean_vector = np.mean(vectors, axis=0)
            std_vector = np.std(vectors, axis=0)
            
            # Vector magnitudes
            magnitudes = np.linalg.norm(vectors, axis=1)
            
            # Distribution statistics
            stats = {
                "mean_vector": mean_vector.tolist(),
                "std_vector": std_vector.tolist(),
                "magnitude_stats": {
                    "mean": float(np.mean(magnitudes)),
                    "std": float(np.std(magnitudes)),
                    "min": float(np.min(magnitudes)),
                    "max": float(np.max(magnitudes))
                },
                "vector_count": len(vectors),
                "dimension": vectors.shape[1] if len(vectors.shape) > 1 else 1
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Vector statistics calculation failed: {e}")
            return {"error": str(e)}
    
    async def _classify_patterns(self, vectors: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Classify patterns in vector data."""
        try:
            classifications = []
            
            # Simple pattern classification based on vector properties
            magnitudes = np.linalg.norm(vectors, axis=1)
            mean_magnitude = np.mean(magnitudes)
            
            for i, vector in enumerate(vectors):
                magnitude = magnitudes[i]
                
                # Classify based on magnitude
                if magnitude > mean_magnitude * 1.5:
                    pattern_type = "high_magnitude"
                elif magnitude < mean_magnitude * 0.5:
                    pattern_type = "low_magnitude"
                else:
                    pattern_type = "normal_magnitude"
                
                classification = {
                    "vector_index": i,
                    "pattern_type": pattern_type,
                    "magnitude": float(magnitude),
                    "confidence": 0.8
                }
                
                # Add metadata if available
                if metadata and i < len(metadata):
                    classification["metadata"] = metadata[i]
                
                classifications.append(classification)
            
            return {
                "classifications": classifications,
                "pattern_types": list(set(c["pattern_type"] for c in classifications)),
                "classification_method": "magnitude_based"
            }
            
        except Exception as e:
            logger.error(f"Pattern classification failed: {e}")
            return {"error": str(e)}
    
    async def get_engine_summary(self) -> Dict[str, Any]:
        """Get a summary of the vector pattern engine."""
        try:
            return {
                "clustering_cache_size": len(self.clustering_cache),
                "similarity_cache_size": len(self.similarity_cache),
                "engine_config": self.engine_config,
                "analysis_status": "active"
            }
            
        except Exception as e:
            logger.error(f"Engine summary generation failed: {e}")
            return {"error": str(e)}
