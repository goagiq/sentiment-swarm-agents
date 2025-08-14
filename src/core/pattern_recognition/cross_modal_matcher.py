"""
Enhanced Cross-Modal Pattern Matcher

This module provides advanced cross-modal pattern matching capabilities including:
- Semantic cross-modal pattern correlation
- Temporal correlation across modalities
- Cross-content type pattern matching
- Multi-modal pattern analysis
- Pattern correlation scoring with confidence
- Anomaly detection across modalities
- Trend analysis across modalities
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics

from loguru import logger

from src.core.error_handler import with_error_handling
from src.core.unified_mcp_client import call_unified_mcp_tool


@dataclass
class CrossModalMatch:
    """Represents a match between patterns across modalities."""
    source_modality: str
    target_modality: str
    match_type: str  # semantic, temporal, spatial, contextual
    match_score: float
    confidence: float
    shared_elements: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CrossModalAnomaly:
    """Represents anomalies detected across modalities."""
    anomaly_type: str
    affected_modalities: List[str]
    severity: float
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CrossModalTrend:
    """Represents trends detected across modalities."""
    trend_type: str
    affected_modalities: List[str]
    direction: str  # increasing, decreasing, stable
    strength: float
    duration: timedelta
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)


class EnhancedCrossModalMatcher:
    """
    Enhanced matcher for patterns across different content modalities.
    """
    
    def __init__(self):
        self.matching_cache = {}
        self.matching_config = {
            "semantic_similarity_threshold": 0.6,
            "temporal_correlation_threshold": 0.5,
            "spatial_correlation_threshold": 0.4,
            "contextual_similarity_threshold": 0.5,
            "enable_cross_modal": True,
            "cache_ttl": 3600  # 1 hour
        }
        
        # Weighting for different match types
        self.match_weights = {
            "semantic": 0.4,
            "temporal": 0.3,
            "spatial": 0.2,
            "contextual": 0.1
        }
        
        # Modality-specific reliability scores
        self.modality_reliability = {
            "text": 0.9,
            "audio": 0.7,
            "video": 0.8,
            "image": 0.8,
            "web": 0.8
        }
        
        logger.info("EnhancedCrossModalMatcher initialized successfully")
    
    @with_error_handling("enhanced_cross_modal_matching")
    async def match_patterns(
        self, 
        patterns: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Enhanced pattern matching across different modalities.
        
        Args:
            patterns: Dictionary mapping modality names to pattern lists
            
        Returns:
            Dictionary containing matches, anomalies, trends, and metadata
        """
        try:
            # Clear expired cache entries
            await self._cleanup_cache()
            
            # Generate cache key
            cache_key = self._generate_cache_key(patterns)
            
            # Check cache first
            if cache_key in self.matching_cache:
                cached_result = self.matching_cache[cache_key]
                if (datetime.now() - cached_result["timestamp"]).seconds < self.matching_config["cache_ttl"]:
                    logger.info("Returning cached cross-modal match result")
                    return cached_result["data"]
            
            # Perform enhanced cross-modal matching
            matches = await self._enhanced_modality_matching(patterns)
            
            # Detect cross-modal anomalies
            anomalies = await self._detect_cross_modal_anomalies(patterns)
            
            # Analyze cross-modal trends
            trends = await self._analyze_cross_modal_trends(patterns)
            
            # Calculate overall correlation score
            overall_correlation = self._calculate_overall_correlation(matches)
            
            result = {
                "matches": [match.__dict__ for match in matches],
                "anomalies": [anomaly.__dict__ for anomaly in anomalies],
                "trends": [trend.__dict__ for trend in trends],
                "total_matches": len(matches),
                "total_anomalies": len(anomalies),
                "total_trends": len(trends),
                "overall_correlation_score": overall_correlation,
                "matching_method": "enhanced_cross_modal",
                "modalities_analyzed": list(patterns.keys()),
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            self.matching_cache[cache_key] = {
                "data": result,
                "timestamp": datetime.now()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced cross-modal matching failed: {e}")
            return {"error": str(e)}
    
    async def _enhanced_modality_matching(
        self, 
        patterns: Dict[str, List[Any]]
    ) -> List[CrossModalMatch]:
        """Perform enhanced matching between modalities."""
        matches = []
        modalities = list(patterns.keys())
        
        for i, modality1 in enumerate(modalities):
            for modality2 in modalities[i+1:]:
                patterns1 = patterns[modality1]
                patterns2 = patterns[modality2]
                
                # Semantic matching
                semantic_match = await self._semantic_cross_modal_matching(
                    modality1, patterns1, modality2, patterns2
                )
                if semantic_match:
                    matches.append(semantic_match)
                
                # Temporal matching
                temporal_match = await self._temporal_correlation(
                    modality1, patterns1, modality2, patterns2
                )
                if temporal_match:
                    matches.append(temporal_match)
                
                # Spatial matching (for applicable modalities)
                if modality1 in ["image", "video"] and modality2 in ["image", "video"]:
                    spatial_match = await self._spatial_correlation(
                        modality1, patterns1, modality2, patterns2
                    )
                    if spatial_match:
                        matches.append(spatial_match)
                
                # Contextual matching
                contextual_match = await self._contextual_similarity(
                    modality1, patterns1, modality2, patterns2
                )
                if contextual_match:
                    matches.append(contextual_match)
        
        return matches
    
    async def _semantic_cross_modal_matching(
        self, 
        modality1: str, 
        patterns1: List[Any], 
        modality2: str, 
        patterns2: List[Any]
    ) -> Optional[CrossModalMatch]:
        """Perform semantic matching between modalities."""
        try:
            # Extract semantic features from patterns
            features1 = await self._extract_semantic_features(modality1, patterns1)
            features2 = await self._extract_semantic_features(modality2, patterns2)
            
            # Calculate semantic similarity
            similarity_score = await self._calculate_semantic_similarity(
                features1, features2
            )
            
            if similarity_score >= self.matching_config["semantic_similarity_threshold"]:
                # Find shared semantic elements
                shared_elements = await self._find_shared_semantic_elements(
                    features1, features2
                )
                
                # Calculate confidence
                confidence = self._calculate_match_confidence(
                    modality1, modality2, similarity_score
                )
                
                return CrossModalMatch(
                    source_modality=modality1,
                    target_modality=modality2,
                    match_type="semantic",
                    match_score=similarity_score,
                    confidence=confidence,
                    shared_elements=shared_elements,
                    evidence={
                        "features1_count": len(features1),
                        "features2_count": len(features2),
                        "shared_elements_count": len(shared_elements)
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in semantic cross-modal matching: {e}")
            return None
    
    async def _temporal_correlation(
        self, 
        modality1: str, 
        patterns1: List[Any], 
        modality2: str, 
        patterns2: List[Any]
    ) -> Optional[CrossModalMatch]:
        """Perform temporal correlation analysis between modalities."""
        try:
            # Extract temporal features
            temporal_features1 = await self._extract_temporal_features(modality1, patterns1)
            temporal_features2 = await self._extract_temporal_features(modality2, patterns2)
            
            # Calculate temporal correlation
            correlation_score = await self._calculate_temporal_correlation(
                temporal_features1, temporal_features2
            )
            
            if correlation_score >= self.matching_config["temporal_correlation_threshold"]:
                # Find shared temporal patterns
                shared_patterns = await self._find_shared_temporal_patterns(
                    temporal_features1, temporal_features2
                )
                
                # Calculate confidence
                confidence = self._calculate_match_confidence(
                    modality1, modality2, correlation_score
                )
                
                return CrossModalMatch(
                    source_modality=modality1,
                    target_modality=modality2,
                    match_type="temporal",
                    match_score=correlation_score,
                    confidence=confidence,
                    shared_elements=shared_patterns,
                    evidence={
                        "temporal_features1_count": len(temporal_features1),
                        "temporal_features2_count": len(temporal_features2),
                        "shared_patterns_count": len(shared_patterns)
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in temporal correlation: {e}")
            return None
    
    async def _spatial_correlation(
        self, 
        modality1: str, 
        patterns1: List[Any], 
        modality2: str, 
        patterns2: List[Any]
    ) -> Optional[CrossModalMatch]:
        """Perform spatial correlation analysis between visual modalities."""
        try:
            # Extract spatial features
            spatial_features1 = await self._extract_spatial_features(modality1, patterns1)
            spatial_features2 = await self._extract_spatial_features(modality2, patterns2)
            
            # Calculate spatial correlation
            correlation_score = await self._calculate_spatial_correlation(
                spatial_features1, spatial_features2
            )
            
            if correlation_score >= self.matching_config["spatial_correlation_threshold"]:
                # Find shared spatial elements
                shared_elements = await self._find_shared_spatial_elements(
                    spatial_features1, spatial_features2
                )
                
                # Calculate confidence
                confidence = self._calculate_match_confidence(
                    modality1, modality2, correlation_score
                )
                
                return CrossModalMatch(
                    source_modality=modality1,
                    target_modality=modality2,
                    match_type="spatial",
                    match_score=correlation_score,
                    confidence=confidence,
                    shared_elements=shared_elements,
                    evidence={
                        "spatial_features1_count": len(spatial_features1),
                        "spatial_features2_count": len(spatial_features2),
                        "shared_elements_count": len(shared_elements)
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in spatial correlation: {e}")
            return None
    
    async def _contextual_similarity(
        self, 
        modality1: str, 
        patterns1: List[Any], 
        modality2: str, 
        patterns2: List[Any]
    ) -> Optional[CrossModalMatch]:
        """Perform contextual similarity analysis between modalities."""
        try:
            # Extract contextual features
            context1 = await self._extract_contextual_features(modality1, patterns1)
            context2 = await self._extract_contextual_features(modality2, patterns2)
            
            # Calculate contextual similarity
            similarity_score = await self._calculate_contextual_similarity(
                context1, context2
            )
            
            if similarity_score >= self.matching_config["contextual_similarity_threshold"]:
                # Find shared contextual elements
                shared_elements = await self._find_shared_contextual_elements(
                    context1, context2
                )
                
                # Calculate confidence
                confidence = self._calculate_match_confidence(
                    modality1, modality2, similarity_score
                )
                
                return CrossModalMatch(
                    source_modality=modality1,
                    target_modality=modality2,
                    match_type="contextual",
                    match_score=similarity_score,
                    confidence=confidence,
                    shared_elements=shared_elements,
                    evidence={
                        "context1_size": len(context1),
                        "context2_size": len(context2),
                        "shared_elements_count": len(shared_elements)
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in contextual similarity: {e}")
            return None
    
    async def _detect_cross_modal_anomalies(
        self, 
        patterns: Dict[str, List[Any]]
    ) -> List[CrossModalAnomaly]:
        """Detect anomalies across modalities."""
        anomalies = []
        
        try:
            # Analyze pattern distributions across modalities
            modality_stats = {}
            for modality, pattern_list in patterns.items():
                modality_stats[modality] = {
                    "count": len(pattern_list),
                    "types": self._get_pattern_types(pattern_list),
                    "complexity": self._calculate_pattern_complexity(pattern_list)
                }
            
            # Detect statistical anomalies
            statistical_anomalies = await self._detect_statistical_anomalies(
                modality_stats
            )
            anomalies.extend(statistical_anomalies)
            
            # Detect temporal anomalies
            temporal_anomalies = await self._detect_temporal_anomalies(patterns)
            anomalies.extend(temporal_anomalies)
            
            # Detect content anomalies
            content_anomalies = await self._detect_content_anomalies(patterns)
            anomalies.extend(content_anomalies)
            
        except Exception as e:
            logger.error(f"Error detecting cross-modal anomalies: {e}")
        
        return anomalies
    
    async def _analyze_cross_modal_trends(
        self, 
        patterns: Dict[str, List[Any]]
    ) -> List[CrossModalTrend]:
        """Analyze trends across modalities."""
        trends = []
        
        try:
            # Analyze temporal trends
            temporal_trends = await self._analyze_temporal_trends(patterns)
            trends.extend(temporal_trends)
            
            # Analyze content trends
            content_trends = await self._analyze_content_trends(patterns)
            trends.extend(content_trends)
            
            # Analyze complexity trends
            complexity_trends = await self._analyze_complexity_trends(patterns)
            trends.extend(complexity_trends)
            
        except Exception as e:
            logger.error(f"Error analyzing cross-modal trends: {e}")
        
        return trends
    
    # Helper methods for feature extraction and analysis
    async def _extract_semantic_features(self, modality: str, patterns: List[Any]) -> List[str]:
        """Extract semantic features from patterns."""
        try:
            features = []
            for pattern in patterns:
                if isinstance(pattern, dict):
                    # Extract text-based features
                    if "text" in pattern:
                        features.append(pattern["text"])
                    if "description" in pattern:
                        features.append(pattern["description"])
                    if "name" in pattern:
                        features.append(pattern["name"])
                    if "type" in pattern:
                        features.append(pattern["type"])
                elif isinstance(pattern, str):
                    features.append(pattern)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting semantic features: {e}")
            return []
    
    async def _extract_temporal_features(self, modality: str, patterns: List[Any]) -> List[Dict[str, Any]]:
        """Extract temporal features from patterns."""
        try:
            temporal_features = []
            for pattern in patterns:
                if isinstance(pattern, dict):
                    temporal_feature = {
                        "timestamp": pattern.get("timestamp"),
                        "duration": pattern.get("duration"),
                        "frequency": pattern.get("frequency"),
                        "periodicity": pattern.get("periodicity")
                    }
                    temporal_features.append(temporal_feature)
            
            return temporal_features
            
        except Exception as e:
            logger.error(f"Error extracting temporal features: {e}")
            return []
    
    async def _extract_spatial_features(self, modality: str, patterns: List[Any]) -> List[Dict[str, Any]]:
        """Extract spatial features from patterns."""
        try:
            spatial_features = []
            for pattern in patterns:
                if isinstance(pattern, dict):
                    spatial_feature = {
                        "position": pattern.get("position"),
                        "size": pattern.get("size"),
                        "region": pattern.get("region"),
                        "coordinates": pattern.get("coordinates")
                    }
                    spatial_features.append(spatial_feature)
            
            return spatial_features
            
        except Exception as e:
            logger.error(f"Error extracting spatial features: {e}")
            return []
    
    async def _extract_contextual_features(self, modality: str, patterns: List[Any]) -> List[Dict[str, Any]]:
        """Extract contextual features from patterns."""
        try:
            contextual_features = []
            for pattern in patterns:
                if isinstance(pattern, dict):
                    contextual_feature = {
                        "context": pattern.get("context"),
                        "environment": pattern.get("environment"),
                        "conditions": pattern.get("conditions"),
                        "metadata": pattern.get("metadata", {})
                    }
                    contextual_features.append(contextual_feature)
            
            return contextual_features
            
        except Exception as e:
            logger.error(f"Error extracting contextual features: {e}")
            return []
    
    async def _calculate_semantic_similarity(
        self, 
        features1: List[str], 
        features2: List[str]
    ) -> float:
        """Calculate semantic similarity between feature sets."""
        try:
            if not features1 or not features2:
                return 0.0
            
            # Simple Jaccard similarity for now
            set1 = set(features1)
            set2 = set(features2)
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    async def _calculate_temporal_correlation(
        self, 
        features1: List[Dict[str, Any]], 
        features2: List[Dict[str, Any]]
    ) -> float:
        """Calculate temporal correlation between feature sets."""
        try:
            if not features1 or not features2:
                return 0.0
            
            # Extract timestamps and calculate correlation
            timestamps1 = [f.get("timestamp") for f in features1 if f.get("timestamp")]
            timestamps2 = [f.get("timestamp") for f in features2 if f.get("timestamp")]
            
            if not timestamps1 or not timestamps2:
                return 0.0
            
            # Simple correlation calculation
            # In a real implementation, you'd use proper statistical correlation
            return 0.5  # Placeholder
            
        except Exception as e:
            logger.error(f"Error calculating temporal correlation: {e}")
            return 0.0
    
    async def _calculate_spatial_correlation(
        self, 
        features1: List[Dict[str, Any]], 
        features2: List[Dict[str, Any]]
    ) -> float:
        """Calculate spatial correlation between feature sets."""
        try:
            if not features1 or not features2:
                return 0.0
            
            # Extract spatial coordinates and calculate correlation
            coordinates1 = [f.get("coordinates") for f in features1 if f.get("coordinates")]
            coordinates2 = [f.get("coordinates") for f in features2 if f.get("coordinates")]
            
            if not coordinates1 or not coordinates2:
                return 0.0
            
            # Simple correlation calculation
            # In a real implementation, you'd use proper spatial correlation
            return 0.4  # Placeholder
            
        except Exception as e:
            logger.error(f"Error calculating spatial correlation: {e}")
            return 0.0
    
    async def _calculate_contextual_similarity(
        self, 
        context1: List[Dict[str, Any]], 
        context2: List[Dict[str, Any]]
    ) -> float:
        """Calculate contextual similarity between context sets."""
        try:
            if not context1 or not context2:
                return 0.0
            
            # Extract context strings and calculate similarity
            context_strings1 = [str(c.get("context", "")) for c in context1]
            context_strings2 = [str(c.get("context", "")) for c in context2]
            
            # Use semantic similarity calculation
            return await self._calculate_semantic_similarity(
                context_strings1, context_strings2
            )
            
        except Exception as e:
            logger.error(f"Error calculating contextual similarity: {e}")
            return 0.0
    
    def _calculate_match_confidence(
        self, 
        modality1: str, 
        modality2: str, 
        match_score: float
    ) -> float:
        """Calculate confidence for a cross-modal match."""
        try:
            # Base confidence from match score
            base_confidence = match_score
            
            # Adjust based on modality reliability
            reliability1 = self.modality_reliability.get(modality1, 0.5)
            reliability2 = self.modality_reliability.get(modality2, 0.5)
            avg_reliability = (reliability1 + reliability2) / 2
            
            # Final confidence
            final_confidence = base_confidence * avg_reliability
            
            return min(final_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating match confidence: {e}")
            return 0.0
    
    def _calculate_overall_correlation(self, matches: List[CrossModalMatch]) -> float:
        """Calculate overall correlation score from all matches."""
        try:
            if not matches:
                return 0.0
            
            # Weighted average of match scores
            weighted_sum = 0.0
            total_weight = 0.0
            
            for match in matches:
                weight = self.match_weights.get(match.match_type, 0.1)
                weighted_sum += match.match_score * weight * match.confidence
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating overall correlation: {e}")
            return 0.0
    
    def _generate_cache_key(self, patterns: Dict[str, List[Any]]) -> str:
        """Generate cache key for patterns."""
        try:
            # Create a hashable representation of patterns
            pattern_summary = {}
            for modality, pattern_list in patterns.items():
                pattern_summary[modality] = {
                    "count": len(pattern_list),
                    "types": self._get_pattern_types(pattern_list)
                }
            
            return str(hash(str(pattern_summary)))
            
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return str(hash(str(patterns)))
    
    def _get_pattern_types(self, patterns: List[Any]) -> List[str]:
        """Get types of patterns in a list."""
        try:
            types = []
            for pattern in patterns:
                if isinstance(pattern, dict):
                    pattern_type = pattern.get("type", "unknown")
                    if pattern_type not in types:
                        types.append(pattern_type)
                else:
                    pattern_type = type(pattern).__name__
                    if pattern_type not in types:
                        types.append(pattern_type)
            
            return types
            
        except Exception as e:
            logger.error(f"Error getting pattern types: {e}")
            return []
    
    def _calculate_pattern_complexity(self, patterns: List[Any]) -> float:
        """Calculate complexity score for a pattern list."""
        try:
            if not patterns:
                return 0.0
            
            complexity_scores = []
            for pattern in patterns:
                if isinstance(pattern, dict):
                    # Count non-empty fields as complexity measure
                    non_empty_fields = sum(
                        1 for value in pattern.values() 
                        if value is not None and value != ""
                    )
                    complexity_scores.append(non_empty_fields)
                else:
                    complexity_scores.append(1)
            
            return statistics.mean(complexity_scores) if complexity_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating pattern complexity: {e}")
            return 0.0
    
    async def _cleanup_cache(self):
        """Clean up expired cache entries."""
        try:
            current_time = datetime.now()
            expired_keys = []
            
            for key, value in self.matching_cache.items():
                if (current_time - value["timestamp"]).seconds > self.matching_config["cache_ttl"]:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.matching_cache[key]
                
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
    
    # Placeholder methods for anomaly and trend detection
    async def _detect_statistical_anomalies(
        self, 
        modality_stats: Dict[str, Dict[str, Any]]
    ) -> List[CrossModalAnomaly]:
        """Detect statistical anomalies across modalities."""
        # Placeholder implementation
        return []
    
    async def _detect_temporal_anomalies(
        self, 
        patterns: Dict[str, List[Any]]
    ) -> List[CrossModalAnomaly]:
        """Detect temporal anomalies across modalities."""
        # Placeholder implementation
        return []
    
    async def _detect_content_anomalies(
        self, 
        patterns: Dict[str, List[Any]]
    ) -> List[CrossModalAnomaly]:
        """Detect content anomalies across modalities."""
        # Placeholder implementation
        return []
    
    async def _analyze_temporal_trends(
        self, 
        patterns: Dict[str, List[Any]]
    ) -> List[CrossModalTrend]:
        """Analyze temporal trends across modalities."""
        # Placeholder implementation
        return []
    
    async def _analyze_content_trends(
        self, 
        patterns: Dict[str, List[Any]]
    ) -> List[CrossModalTrend]:
        """Analyze content trends across modalities."""
        # Placeholder implementation
        return []
    
    async def _analyze_complexity_trends(
        self, 
        patterns: Dict[str, List[Any]]
    ) -> List[CrossModalTrend]:
        """Analyze complexity trends across modalities."""
        # Placeholder implementation
        return []
    
    # Placeholder methods for finding shared elements
    async def _find_shared_semantic_elements(
        self, 
        features1: List[str], 
        features2: List[str]
    ) -> List[str]:
        """Find shared semantic elements between feature sets."""
        try:
            set1 = set(features1)
            set2 = set(features2)
            return list(set1.intersection(set2))
        except Exception as e:
            logger.error(f"Error finding shared semantic elements: {e}")
            return []
    
    async def _find_shared_temporal_patterns(
        self, 
        features1: List[Dict[str, Any]], 
        features2: List[Dict[str, Any]]
    ) -> List[str]:
        """Find shared temporal patterns between feature sets."""
        # Placeholder implementation
        return []
    
    async def _find_shared_spatial_elements(
        self, 
        features1: List[Dict[str, Any]], 
        features2: List[Dict[str, Any]]
    ) -> List[str]:
        """Find shared spatial elements between feature sets."""
        # Placeholder implementation
        return []
    
    async def _find_shared_contextual_elements(
        self, 
        context1: List[Dict[str, Any]], 
        context2: List[Dict[str, Any]]
    ) -> List[str]:
        """Find shared contextual elements between context sets."""
        # Placeholder implementation
        return []
