"""
Relationship Optimizer for Phase 3 Advanced Features.
Implements relationship quality assessment and optimization algorithms.
"""

from typing import Dict, List, Optional, Any
import re
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class RelationshipQuality:
    """Quality assessment for a relationship."""
    source: str
    target: str
    relationship_type: str
    quality_score: float
    confidence: float
    evidence_strength: float
    redundancy_score: float
    overall_score: float


class RelationshipOptimizer:
    """Advanced relationship optimizer for Chinese content."""
    
    def __init__(self):
        # Quality assessment criteria
        self.quality_criteria = {
            "evidence_threshold": 0.3,
            "confidence_threshold": 0.5,
            "redundancy_penalty": 0.2,
            "semantic_coherence_weight": 0.4,
            "context_relevance_weight": 0.3,
            "structural_validity_weight": 0.3
        }
        
        # Chinese-specific relationship validation patterns
        self.validation_patterns = {
            "WORKS_FOR": {
                "required_context": ["工作", "任职", "就职", "担任", "职位"],
                "entity_types": ["PERSON", "ORGANIZATION"],
                "weight": 0.8
            },
            "LOCATED_IN": {
                "required_context": ["位于", "坐落", "地处", "位置", "地址"],
                "entity_types": ["ORGANIZATION", "LOCATION"],
                "weight": 0.7
            },
            "RELATED_TO": {
                "required_context": ["相关", "联系", "关联", "涉及", "关于"],
                "entity_types": ["CONCEPT", "CONCEPT"],
                "weight": 0.6
            },
            "COLLABORATES_WITH": {
                "required_context": ["合作", "协作", "联合", "共同", "伙伴"],
                "entity_types": ["ORGANIZATION", "ORGANIZATION"],
                "weight": 0.7
            },
            "CONTAINS": {
                "required_context": ["包含", "包括", "涵盖", "组成", "构成"],
                "entity_types": ["ORGANIZATION", "ORGANIZATION"],
                "weight": 0.8
            }
        }
        
        # Redundancy detection patterns
        self.redundancy_patterns = [
            r"([^。！？]*?{entity1}[^。！？]*?{entity2}[^。！？]*?[。！？])",
            r"([^。！？]*?{entity2}[^。！？]*?{entity1}[^。！？]*?[。！？])"
        ]
    
    def optimize_relationships(
        self, 
        relationships: List[Dict], 
        entities: List[Dict], 
        text: str
    ) -> List[Dict]:
        """Optimize relationships based on quality assessment."""
        # Assess quality of each relationship
        quality_assessments = []
        for rel in relationships:
            quality = self._assess_relationship_quality(rel, entities, text)
            quality_assessments.append(quality)
        
        # Filter relationships based on quality thresholds
        filtered_relationships = self._filter_by_quality(quality_assessments)
        
        # Remove redundant relationships
        deduplicated_relationships = self._remove_redundancies(filtered_relationships)
        
        # Sort by quality score
        optimized_relationships = sorted(
            deduplicated_relationships, 
            key=lambda x: x["quality_score"], 
            reverse=True
        )
        
        return optimized_relationships
    
    def _assess_relationship_quality(
        self, 
        relationship: Dict, 
        entities: List[Dict], 
        text: str
    ) -> RelationshipQuality:
        """Assess the quality of a relationship."""
        source = relationship.get("source", "")
        target = relationship.get("target", "")
        rel_type = relationship.get("relationship_type", "")
        
        # Calculate individual quality metrics
        evidence_strength = self._calculate_evidence_strength(source, target, rel_type, text)
        confidence = self._calculate_confidence(source, target, rel_type, entities)
        redundancy_score = self._calculate_redundancy_score(source, target, text)
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_score(
            evidence_strength, confidence, redundancy_score
        )
        
        return RelationshipQuality(
            source=source,
            target=target,
            relationship_type=rel_type,
            quality_score=overall_score,
            confidence=confidence,
            evidence_strength=evidence_strength,
            redundancy_score=redundancy_score,
            overall_score=overall_score
        )
    
    def _calculate_evidence_strength(
        self, 
        source: str, 
        target: str, 
        rel_type: str, 
        text: str
    ) -> float:
        """Calculate evidence strength for a relationship."""
        # Check if relationship type has validation patterns
        if rel_type not in self.validation_patterns:
            return 0.5  # Default strength for unknown relationship types
        
        pattern = self.validation_patterns[rel_type]
        required_context = pattern.get("required_context", [])
        
        # Find context around both entities
        source_pos = text.find(source)
        target_pos = text.find(target)
        
        if source_pos == -1 or target_pos == -1:
            return 0.1
        
        # Get context window
        start = min(source_pos, target_pos) - 100
        end = max(source_pos, target_pos) + 100
        context = text[max(0, start):min(len(text), end)]
        
        # Check for required context indicators
        context_matches = 0
        for indicator in required_context:
            if indicator in context:
                context_matches += 1
        
        # Calculate evidence strength
        if required_context:
            evidence_strength = context_matches / len(required_context)
        else:
            evidence_strength = 0.5
        
        # Apply relationship type weight
        weight = pattern.get("weight", 0.5)
        return evidence_strength * weight
    
    def _calculate_confidence(
        self, 
        source: str, 
        target: str, 
        rel_type: str, 
        entities: List[Dict]
    ) -> float:
        """Calculate confidence for a relationship."""
        # Find entity types
        source_type = self._get_entity_type(source, entities)
        target_type = self._get_entity_type(target, entities)
        
        # Check if relationship type is valid for these entity types
        if rel_type in self.validation_patterns:
            pattern = self.validation_patterns[rel_type]
            valid_types = pattern.get("entity_types", [])
            
            if source_type in valid_types and target_type in valid_types:
                return 0.8
            elif source_type in valid_types or target_type in valid_types:
                return 0.6
            else:
                return 0.3
        else:
            return 0.5
    
    def _calculate_redundancy_score(
        self, 
        source: str, 
        target: str, 
        text: str
    ) -> float:
        """Calculate redundancy score (lower is better)."""
        # Count how many times this entity pair appears together
        sentences = re.split(r'[。！？]', text)
        
        co_occurrence_count = 0
        for sentence in sentences:
            if source in sentence and target in sentence:
                co_occurrence_count += 1
        
        # Normalize redundancy score (0 = no redundancy, 1 = high redundancy)
        if co_occurrence_count <= 1:
            return 0.0
        elif co_occurrence_count <= 3:
            return 0.3
        elif co_occurrence_count <= 5:
            return 0.6
        else:
            return 1.0
    
    def _calculate_overall_score(
        self, 
        evidence_strength: float, 
        confidence: float, 
        redundancy_score: float
    ) -> float:
        """Calculate overall quality score."""
        # Apply redundancy penalty
        redundancy_penalty = redundancy_score * self.quality_criteria["redundancy_penalty"]
        
        # Calculate weighted score
        overall_score = (
            evidence_strength * self.quality_criteria["semantic_coherence_weight"] +
            confidence * self.quality_criteria["context_relevance_weight"] +
            (1 - redundancy_penalty) * self.quality_criteria["structural_validity_weight"]
        )
        
        return max(0.0, min(1.0, overall_score))
    
    def _get_entity_type(self, entity_name: str, entities: List[Dict]) -> str:
        """Get entity type from entity list."""
        for entity in entities:
            if entity.get("text", entity.get("name", "")) == entity_name:
                return entity.get("type", "CONCEPT").upper()
        return "CONCEPT"
    
    def _filter_by_quality(
        self, 
        quality_assessments: List[RelationshipQuality]
    ) -> List[Dict]:
        """Filter relationships based on quality thresholds."""
        filtered = []
        
        for assessment in quality_assessments:
            # Check quality thresholds
            if (assessment.overall_score >= self.quality_criteria["evidence_threshold"] and
                assessment.confidence >= self.quality_criteria["confidence_threshold"]):
                
                filtered.append({
                    "source": assessment.source,
                    "target": assessment.target,
                    "relationship_type": assessment.relationship_type,
                    "quality_score": assessment.overall_score,
                    "confidence": assessment.confidence,
                    "evidence_strength": assessment.evidence_strength
                })
        
        return filtered
    
    def _remove_redundancies(self, relationships: List[Dict]) -> List[Dict]:
        """Remove redundant relationships."""
        # Group relationships by entity pairs
        relationship_groups = defaultdict(list)
        
        for rel in relationships:
            # Create canonical key (sorted entity names)
            entity_pair = tuple(sorted([rel["source"], rel["target"]]))
            relationship_groups[entity_pair].append(rel)
        
        # Keep only the best relationship for each entity pair
        deduplicated = []
        
        for entity_pair, rels in relationship_groups.items():
            if len(rels) == 1:
                deduplicated.append(rels[0])
            else:
                # Keep the relationship with highest quality score
                best_rel = max(rels, key=lambda x: x["quality_score"])
                deduplicated.append(best_rel)
        
        return deduplicated
    
    def get_optimization_statistics(
        self, 
        original_relationships: List[Dict], 
        optimized_relationships: List[Dict]
    ) -> Dict[str, Any]:
        """Get statistics about the optimization process."""
        original_count = len(original_relationships)
        optimized_count = len(optimized_relationships)
        
        if original_count == 0:
            return {
                "original_count": 0,
                "optimized_count": 0,
                "removed_count": 0,
                "quality_improvement": 0.0,
                "redundancy_reduction": 0.0
            }
        
        # Calculate quality improvement
        original_avg_quality = sum(
            rel.get("quality_score", 0.5) for rel in original_relationships
        ) / original_count
        
        optimized_avg_quality = sum(
            rel.get("quality_score", 0.5) for rel in optimized_relationships
        ) / optimized_count if optimized_count > 0 else 0.0
        
        quality_improvement = optimized_avg_quality - original_avg_quality
        
        # Calculate redundancy reduction
        redundancy_reduction = (original_count - optimized_count) / original_count
        
        return {
            "original_count": original_count,
            "optimized_count": optimized_count,
            "removed_count": original_count - optimized_count,
            "quality_improvement": quality_improvement,
            "redundancy_reduction": redundancy_reduction,
            "average_quality_before": original_avg_quality,
            "average_quality_after": optimized_avg_quality
        }
    
    def validate_relationship_structure(self, relationship: Dict) -> Dict[str, Any]:
        """Validate the structure of a relationship."""
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        required_fields = ["source", "target", "relationship_type"]
        for field in required_fields:
            if field not in relationship or not relationship[field]:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"Missing required field: {field}")
        
        # Check for self-relationships
        if (relationship.get("source") == relationship.get("target") and
            relationship.get("source")):
            validation_result["warnings"].append("Self-relationship detected")
        
        # Check relationship type validity
        rel_type = relationship.get("relationship_type", "")
        if rel_type and rel_type not in self.validation_patterns:
            validation_result["warnings"].append(f"Unknown relationship type: {rel_type}")
        
        return validation_result
