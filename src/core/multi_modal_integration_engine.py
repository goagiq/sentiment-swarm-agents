"""
Multi-Modal Integration Engine

Provides advanced cross-modal integration capabilities including:
- Semantic alignment across modalities
- Cross-modal confidence scoring
- Unified decision context building
- Multi-modal insight correlation
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from loguru import logger

from src.core.models import DataType, AnalysisRequest
from src.core.error_handler import with_error_handling
from src.core.unified_mcp_client import call_unified_mcp_tool


@dataclass
class ModalityInsight:
    """Represents insights extracted from a specific modality."""
    modality: str
    content_type: DataType
    entities: List[Dict[str, Any]] = field(default_factory=list)
    sentiment: Optional[Dict[str, Any]] = None
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossModalCorrelation:
    """Represents correlations between different modalities."""
    source_modality: str
    target_modality: str
    correlation_type: str  # semantic, temporal, spatial, etc.
    correlation_score: float
    shared_entities: List[str] = field(default_factory=list)
    shared_patterns: List[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiModalDecisionContext:
    """Unified decision context built from multiple modalities."""
    modality_insights: Dict[str, ModalityInsight] = field(default_factory=dict)
    cross_modal_correlations: List[CrossModalCorrelation] = field(default_factory=list)
    unified_entities: List[Dict[str, Any]] = field(default_factory=list)
    unified_patterns: List[Dict[str, Any]] = field(default_factory=list)
    overall_confidence: float = 0.0
    modality_weights: Dict[str, float] = field(default_factory=dict)
    decision_factors: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class MultiModalIntegrationEngine:
    """
    Advanced engine for integrating insights across multiple modalities.
    """
    
    def __init__(self):
        self.modality_processors = {
            DataType.TEXT: self._process_text_modality,
            DataType.AUDIO: self._process_audio_modality,
            DataType.VIDEO: self._process_video_modality,
            DataType.IMAGE: self._process_image_modality,
            DataType.WEBPAGE: self._process_web_modality
        }
        
        self.correlation_weights = {
            "semantic": 0.4,
            "temporal": 0.3,
            "spatial": 0.2,
            "contextual": 0.1
        }
        
        self.modality_reliability_weights = {
            DataType.TEXT: 0.9,
            DataType.IMAGE: 0.8,
            DataType.VIDEO: 0.7,
            DataType.AUDIO: 0.6,
            DataType.WEBPAGE: 0.8
        }
        
        logger.info("MultiModalIntegrationEngine initialized successfully")
    
    @with_error_handling("multi_modal_integration")
    async def build_unified_context(
        self, 
        requests: List[AnalysisRequest]
    ) -> MultiModalDecisionContext:
        """
        Build unified decision context from multiple modality requests.
        
        Args:
            requests: List of analysis requests from different modalities
            
        Returns:
            Unified decision context combining all modality insights
        """
        try:
            # Extract insights from each modality
            modality_insights = {}
            for request in requests:
                insight = await self._extract_modality_insight(request)
                if insight:
                    modality_insights[request.data_type] = insight
            
            # Find cross-modal correlations
            correlations = await self._find_cross_modal_correlations(modality_insights)
            
            # Build unified entities and patterns
            unified_entities = await self._build_unified_entities(modality_insights)
            unified_patterns = await self._build_unified_patterns(modality_insights)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                modality_insights, correlations
            )
            
            # Build decision factors
            decision_factors = await self._build_decision_factors(
                modality_insights, correlations
            )
            
            return MultiModalDecisionContext(
                modality_insights=modality_insights,
                cross_modal_correlations=correlations,
                unified_entities=unified_entities,
                unified_patterns=unified_patterns,
                overall_confidence=overall_confidence,
                modality_weights=self.modality_reliability_weights,
                decision_factors=decision_factors
            )
            
        except Exception as e:
            logger.error(f"Error building unified context: {e}")
            raise
    
    async def _extract_modality_insight(self, request: AnalysisRequest) -> Optional[ModalityInsight]:
        """Extract insights from a specific modality."""
        try:
            processor = self.modality_processors.get(request.data_type)
            if not processor:
                logger.warning(f"No processor for modality: {request.data_type}")
                return None
            
            return await processor(request)
            
        except Exception as e:
            logger.error(f"Error extracting insight from {request.data_type}: {e}")
            return None
    
    async def _process_text_modality(self, request: AnalysisRequest) -> ModalityInsight:
        """Process text modality and extract insights."""
        try:
            # Extract entities using knowledge graph
            entities_result = await call_unified_mcp_tool(
                "extract_entities",
                {
                    "text": request.content,
                    "language": request.language,
                    "entity_types": ["business", "market", "risk", "opportunity"]
                }
            )
            
            # Extract sentiment
            sentiment_result = await call_unified_mcp_tool(
                "analyze_sentiment",
                {
                    "text": request.content,
                    "language": request.language
                }
            )
            
            # Extract patterns
            patterns_result = await call_unified_mcp_tool(
                "analyze_patterns",
                {
                    "content": request.content,
                    "pattern_types": ["trends", "anomalies", "seasonality"]
                }
            )
            
            # Calculate confidence based on content quality
            confidence = self._calculate_text_confidence(request.content)
            
            return ModalityInsight(
                modality="text",
                content_type=request.data_type,
                entities=entities_result.get("entities", []) if entities_result.get("success") else [],
                sentiment=sentiment_result if sentiment_result.get("success") else None,
                patterns=patterns_result.get("patterns", []) if patterns_result.get("success") else [],
                confidence=confidence,
                metadata={"content_length": len(str(request.content))}
            )
            
        except Exception as e:
            logger.error(f"Error processing text modality: {e}")
            return ModalityInsight(
                modality="text",
                content_type=request.data_type,
                confidence=0.0
            )
    
    async def _process_audio_modality(self, request: AnalysisRequest) -> ModalityInsight:
        """Process audio modality and extract insights."""
        try:
            # Extract audio insights using audio agent
            audio_result = await call_unified_mcp_tool(
                "analyze_audio",
                {
                    "audio_content": request.content,
                    "analysis_types": ["transcription", "sentiment", "entities", "patterns"]
                }
            )
            
            confidence = self._calculate_audio_confidence(request.content)
            
            return ModalityInsight(
                modality="audio",
                content_type=request.data_type,
                entities=audio_result.get("entities", []) if audio_result.get("success") else [],
                sentiment=audio_result.get("sentiment") if audio_result.get("success") else None,
                patterns=audio_result.get("patterns", []) if audio_result.get("success") else [],
                confidence=confidence,
                metadata={"audio_duration": audio_result.get("duration", 0)}
            )
            
        except Exception as e:
            logger.error(f"Error processing audio modality: {e}")
            return ModalityInsight(
                modality="audio",
                content_type=request.data_type,
                confidence=0.0
            )
    
    async def _process_video_modality(self, request: AnalysisRequest) -> ModalityInsight:
        """Process video modality and extract insights."""
        try:
            # Extract video insights using vision agent
            video_result = await call_unified_mcp_tool(
                "analyze_video",
                {
                    "video_content": request.content,
                    "analysis_types": ["objects", "actions", "sentiment", "text", "audio"]
                }
            )
            
            confidence = self._calculate_video_confidence(request.content)
            
            return ModalityInsight(
                modality="video",
                content_type=request.data_type,
                entities=video_result.get("entities", []) if video_result.get("success") else [],
                sentiment=video_result.get("sentiment") if video_result.get("success") else None,
                patterns=video_result.get("patterns", []) if video_result.get("success") else [],
                confidence=confidence,
                metadata={"video_duration": video_result.get("duration", 0)}
            )
            
        except Exception as e:
            logger.error(f"Error processing video modality: {e}")
            return ModalityInsight(
                modality="video",
                content_type=request.data_type,
                confidence=0.0
            )
    
    async def _process_image_modality(self, request: AnalysisRequest) -> ModalityInsight:
        """Process image modality and extract insights."""
        try:
            # Extract image insights using vision agent
            image_result = await call_unified_mcp_tool(
                "analyze_image",
                {
                    "image_content": request.content,
                    "analysis_types": ["objects", "text", "sentiment", "patterns"]
                }
            )
            
            confidence = self._calculate_image_confidence(request.content)
            
            return ModalityInsight(
                modality="image",
                content_type=request.data_type,
                entities=image_result.get("entities", []) if image_result.get("success") else [],
                sentiment=image_result.get("sentiment") if image_result.get("success") else None,
                patterns=image_result.get("patterns", []) if image_result.get("success") else [],
                confidence=confidence,
                metadata={"image_resolution": image_result.get("resolution", "unknown")}
            )
            
        except Exception as e:
            logger.error(f"Error processing image modality: {e}")
            return ModalityInsight(
                modality="image",
                content_type=request.data_type,
                confidence=0.0
            )
    
    async def _process_web_modality(self, request: AnalysisRequest) -> ModalityInsight:
        """Process web modality and extract insights."""
        try:
            # Extract web insights using web agent
            web_result = await call_unified_mcp_tool(
                "analyze_webpage",
                {
                    "url": request.content,
                    "analysis_types": ["content", "sentiment", "entities", "patterns"]
                }
            )
            
            confidence = self._calculate_web_confidence(request.content)
            
            return ModalityInsight(
                modality="web",
                content_type=request.data_type,
                entities=web_result.get("entities", []) if web_result.get("success") else [],
                sentiment=web_result.get("sentiment") if web_result.get("success") else None,
                patterns=web_result.get("patterns", []) if web_result.get("success") else [],
                confidence=confidence,
                metadata={"page_title": web_result.get("title", "unknown")}
            )
            
        except Exception as e:
            logger.error(f"Error processing web modality: {e}")
            return ModalityInsight(
                modality="web",
                content_type=request.data_type,
                confidence=0.0
            )
    
    async def _find_cross_modal_correlations(
        self, 
        modality_insights: Dict[str, ModalityInsight]
    ) -> List[CrossModalCorrelation]:
        """Find correlations between different modalities."""
        correlations = []
        modalities = list(modality_insights.keys())
        
        for i, modality1 in enumerate(modalities):
            for modality2 in modalities[i+1:]:
                insight1 = modality_insights[modality1]
                insight2 = modality_insights[modality2]
                
                # Semantic correlation
                semantic_corr = await self._calculate_semantic_correlation(insight1, insight2)
                if semantic_corr["score"] > 0.3:
                    correlations.append(CrossModalCorrelation(
                        source_modality=modality1,
                        target_modality=modality2,
                        correlation_type="semantic",
                        correlation_score=semantic_corr["score"],
                        shared_entities=semantic_corr["shared_entities"],
                        confidence=semantic_corr["confidence"]
                    ))
                
                # Temporal correlation
                temporal_corr = await self._calculate_temporal_correlation(insight1, insight2)
                if temporal_corr["score"] > 0.3:
                    correlations.append(CrossModalCorrelation(
                        source_modality=modality1,
                        target_modality=modality2,
                        correlation_type="temporal",
                        correlation_score=temporal_corr["score"],
                        shared_patterns=temporal_corr["shared_patterns"],
                        confidence=temporal_corr["confidence"]
                    ))
        
        return correlations
    
    async def _calculate_semantic_correlation(
        self, 
        insight1: ModalityInsight, 
        insight2: ModalityInsight
    ) -> Dict[str, Any]:
        """Calculate semantic correlation between two modality insights."""
        try:
            # Extract entity names
            entities1 = [e.get("name", "") for e in insight1.entities]
            entities2 = [e.get("name", "") for e in insight2.entities]
            
            # Find shared entities
            shared_entities = list(set(entities1) & set(entities2))
            
            # Calculate semantic similarity
            if entities1 and entities2:
                similarity_score = len(shared_entities) / max(len(entities1), len(entities2))
            else:
                similarity_score = 0.0
            
            # Calculate confidence based on entity quality
            confidence = min(insight1.confidence, insight2.confidence) * 0.8
            
            return {
                "score": similarity_score,
                "shared_entities": shared_entities,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error calculating semantic correlation: {e}")
            return {"score": 0.0, "shared_entities": [], "confidence": 0.0}
    
    async def _calculate_temporal_correlation(
        self, 
        insight1: ModalityInsight, 
        insight2: ModalityInsight
    ) -> Dict[str, Any]:
        """Calculate temporal correlation between two modality insights."""
        try:
            # Extract temporal patterns
            patterns1 = [p for p in insight1.patterns if p.get("type") == "temporal"]
            patterns2 = [p for p in insight2.patterns if p.get("type") == "temporal"]
            
            # Find shared temporal patterns
            shared_patterns = []
            for p1 in patterns1:
                for p2 in patterns2:
                    if self._patterns_overlap(p1, p2):
                        shared_patterns.append(p1.get("name", ""))
            
            # Calculate temporal similarity
            if patterns1 and patterns2:
                similarity_score = len(shared_patterns) / max(len(patterns1), len(patterns2))
            else:
                similarity_score = 0.0
            
            # Calculate confidence
            confidence = min(insight1.confidence, insight2.confidence) * 0.7
            
            return {
                "score": similarity_score,
                "shared_patterns": shared_patterns,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error calculating temporal correlation: {e}")
            return {"score": 0.0, "shared_patterns": [], "confidence": 0.0}
    
    def _patterns_overlap(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> bool:
        """Check if two patterns overlap temporally."""
        try:
            start1 = pattern1.get("start_time", 0)
            end1 = pattern1.get("end_time", 0)
            start2 = pattern2.get("start_time", 0)
            end2 = pattern2.get("end_time", 0)
            
            return not (end1 < start2 or end2 < start1)
        except:
            return False
    
    async def _build_unified_entities(
        self, 
        modality_insights: Dict[str, ModalityInsight]
    ) -> List[Dict[str, Any]]:
        """Build unified entity list from all modalities."""
        unified_entities = {}
        
        for modality, insight in modality_insights.items():
            for entity in insight.entities:
                entity_name = entity.get("name", "")
                if entity_name:
                    if entity_name not in unified_entities:
                        unified_entities[entity_name] = {
                            "name": entity_name,
                            "type": entity.get("type", "unknown"),
                            "modalities": [],
                            "confidence_scores": [],
                            "metadata": {}
                        }
                    
                    unified_entities[entity_name]["modalities"].append(modality)
                    unified_entities[entity_name]["confidence_scores"].append(insight.confidence)
                    
                    # Merge metadata
                    for key, value in entity.get("metadata", {}).items():
                        if key not in unified_entities[entity_name]["metadata"]:
                            unified_entities[entity_name]["metadata"][key] = []
                        unified_entities[entity_name]["metadata"][key].append(value)
        
        # Calculate unified confidence scores
        for entity_name, entity_data in unified_entities.items():
            if entity_data["confidence_scores"]:
                entity_data["unified_confidence"] = sum(entity_data["confidence_scores"]) / len(entity_data["confidence_scores"])
            else:
                entity_data["unified_confidence"] = 0.0
        
        return list(unified_entities.values())
    
    async def _build_unified_patterns(
        self, 
        modality_insights: Dict[str, ModalityInsight]
    ) -> List[Dict[str, Any]]:
        """Build unified pattern list from all modalities."""
        unified_patterns = {}
        
        for modality, insight in modality_insights.items():
            for pattern in insight.patterns:
                pattern_name = pattern.get("name", "")
                if pattern_name:
                    if pattern_name not in unified_patterns:
                        unified_patterns[pattern_name] = {
                            "name": pattern_name,
                            "type": pattern.get("type", "unknown"),
                            "modalities": [],
                            "confidence_scores": [],
                            "metadata": {}
                        }
                    
                    unified_patterns[pattern_name]["modalities"].append(modality)
                    unified_patterns[pattern_name]["confidence_scores"].append(insight.confidence)
                    
                    # Merge metadata
                    for key, value in pattern.get("metadata", {}).items():
                        if key not in unified_patterns[pattern_name]["metadata"]:
                            unified_patterns[pattern_name]["metadata"][key] = []
                        unified_patterns[pattern_name]["metadata"][key].append(value)
        
        # Calculate unified confidence scores
        for pattern_name, pattern_data in unified_patterns.items():
            if pattern_data["confidence_scores"]:
                pattern_data["unified_confidence"] = sum(pattern_data["confidence_scores"]) / len(pattern_data["confidence_scores"])
            else:
                pattern_data["unified_confidence"] = 0.0
        
        return list(unified_patterns.values())
    
    def _calculate_overall_confidence(
        self, 
        modality_insights: Dict[str, ModalityInsight],
        correlations: List[CrossModalCorrelation]
    ) -> float:
        """Calculate overall confidence score for the multi-modal context."""
        try:
            # Base confidence from modality insights
            modality_confidences = [
                insight.confidence * self.modality_reliability_weights.get(insight.content_type, 0.5)
                for insight in modality_insights.values()
            ]
            
            if not modality_confidences:
                return 0.0
            
            base_confidence = sum(modality_confidences) / len(modality_confidences)
            
            # Boost from correlations
            correlation_boost = 0.0
            if correlations:
                avg_correlation_score = sum(c.correlation_score for c in correlations) / len(correlations)
                correlation_boost = avg_correlation_score * 0.2
            
            # Final confidence (capped at 1.0)
            final_confidence = min(base_confidence + correlation_boost, 1.0)
            
            return final_confidence
            
        except Exception as e:
            logger.error(f"Error calculating overall confidence: {e}")
            return 0.0
    
    async def _build_decision_factors(
        self, 
        modality_insights: Dict[str, ModalityInsight],
        correlations: List[CrossModalCorrelation]
    ) -> List[Dict[str, Any]]:
        """Build decision factors from multi-modal insights."""
        decision_factors = []
        
        # Add high-confidence entities as decision factors
        for insight in modality_insights.values():
            for entity in insight.entities:
                if insight.confidence > 0.7:
                    decision_factors.append({
                        "type": "entity",
                        "name": entity.get("name", ""),
                        "modality": insight.modality,
                        "confidence": insight.confidence,
                        "impact": "high" if entity.get("type") in ["risk", "opportunity"] else "medium"
                    })
        
        # Add strong correlations as decision factors
        for correlation in correlations:
            if correlation.correlation_score > 0.6:
                decision_factors.append({
                    "type": "correlation",
                    "description": f"Strong {correlation.correlation_type} correlation between {correlation.source_modality} and {correlation.target_modality}",
                    "confidence": correlation.confidence,
                    "impact": "high"
                })
        
        return decision_factors
    
    def _calculate_text_confidence(self, content: Any) -> float:
        """Calculate confidence for text modality."""
        try:
            text = str(content)
            if len(text) < 10:
                return 0.3
            elif len(text) < 100:
                return 0.6
            else:
                return 0.9
        except:
            return 0.0
    
    def _calculate_audio_confidence(self, content: Any) -> float:
        """Calculate confidence for audio modality."""
        try:
            # This would need actual audio analysis
            return 0.7
        except:
            return 0.0
    
    def _calculate_video_confidence(self, content: Any) -> float:
        """Calculate confidence for video modality."""
        try:
            # This would need actual video analysis
            return 0.8
        except:
            return 0.0
    
    def _calculate_image_confidence(self, content: Any) -> float:
        """Calculate confidence for image modality."""
        try:
            # This would need actual image analysis
            return 0.8
        except:
            return 0.0
    
    def _calculate_web_confidence(self, content: Any) -> float:
        """Calculate confidence for web modality."""
        try:
            # This would need actual web analysis
            return 0.8
        except:
            return 0.0
