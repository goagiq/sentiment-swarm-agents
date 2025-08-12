"""
Semantic Similarity Analyzer for Phase 3 Advanced Features.
Implements word embedding-based similarity analysis for Chinese entities and 
relationships.
"""

from typing import Dict, List, Optional, Any
import re
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SimilarityResult:
    """Result of semantic similarity analysis."""
    entity1: str
    entity2: str
    similarity_score: float
    similarity_type: str
    confidence: float
    context_evidence: List[str]


class SemanticSimilarityAnalyzer:
    """Advanced semantic similarity analyzer for Chinese content."""
    
    def __init__(self):
        # Chinese-specific semantic patterns for similarity analysis
        self.semantic_patterns = {
            "technology": {
                "keywords": ["技术", "科技", "人工智能", "机器学习", "深度学习", "算法", "软件", "硬件"],
                "weight": 0.8
            },
            "business": {
                "keywords": ["公司", "企业", "商业", "市场", "经济", "投资", "融资", "创业"],
                "weight": 0.7
            },
            "academic": {
                "keywords": ["大学", "学院", "研究", "学术", "教育", "教授", "博士", "论文"],
                "weight": 0.8
            },
            "government": {
                "keywords": ["政府", "部门", "机构", "政策", "法规", "官员", "部长", "委员会"],
                "weight": 0.7
            },
            "location": {
                "keywords": ["城市", "地区", "省份", "国家", "区域", "中心", "区", "街道"],
                "weight": 0.6
            }
        }
        
        # Chinese-specific relationship indicators
        self.relationship_indicators = {
            "hierarchical": ["包含", "属于", "下属", "上级", "管理", "领导"],
            "collaborative": ["合作", "联合", "共同", "协作", "伙伴", "联盟"],
            "competitive": ["竞争", "对手", "对抗", "挑战", "超越"],
            "supportive": ["支持", "帮助", "促进", "推动", "协助"],
            "oppositional": ["反对", "抵制", "抗议", "冲突", "分歧"]
        }
        
        # Context window size for similarity analysis
        self.context_window = 100  # characters
        
    def analyze_semantic_similarity(self, entities: List[Dict], text: str) -> List[SimilarityResult]:
        """Analyze semantic similarity between entities."""
        results = []
        
        # Convert entities to simple list
        entity_list = [entity.get("text", entity.get("name", "")) for entity in entities]
        
        # Generate all entity pairs
        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                entity1, entity2 = entity_list[i], entity_list[j]
                
                # Calculate different types of similarity
                lexical_similarity = self._calculate_lexical_similarity(entity1, entity2)
                semantic_similarity = self._calculate_semantic_similarity(entity1, entity2, text)
                contextual_similarity = self._calculate_contextual_similarity(entity1, entity2, text)
                
                # Combine similarities with weights
                combined_score = (
                    lexical_similarity * 0.2 +
                    semantic_similarity * 0.5 +
                    contextual_similarity * 0.3
                )
                
                # Get context evidence
                context_evidence = self._get_context_evidence(entity1, entity2, text)
                
                # Determine similarity type
                similarity_type = self._determine_similarity_type(
                    lexical_similarity, semantic_similarity, contextual_similarity
                )
                
                # Calculate confidence based on evidence strength
                confidence = self._calculate_confidence(combined_score, len(context_evidence))
                
                results.append(SimilarityResult(
                    entity1=entity1,
                    entity2=entity2,
                    similarity_score=combined_score,
                    similarity_type=similarity_type,
                    confidence=confidence,
                    context_evidence=context_evidence
                ))
        
        return results
    
    def _calculate_lexical_similarity(self, entity1: str, entity2: str) -> float:
        """Calculate lexical similarity between entities."""
        # Simple character-based similarity for Chinese
        chars1 = set(entity1)
        chars2 = set(entity2)
        
        if not chars1 or not chars2:
            return 0.0
        
        intersection = chars1 & chars2
        union = chars1 | chars2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_semantic_similarity(self, entity1: str, entity2: str, text: str) -> float:
        """Calculate semantic similarity based on category matching."""
        category1 = self._get_entity_category(entity1)
        category2 = self._get_entity_category(entity2)
        
        if category1 == category2 and category1:
            return self.semantic_patterns[category1]["weight"]
        elif category1 and category2:
            # Different categories but both have semantic meaning
            return 0.3
        else:
            return 0.1
    
    def _calculate_contextual_similarity(self, entity1: str, entity2: str, text: str) -> float:
        """Calculate contextual similarity based on text proximity and context."""
        # Find positions of entities in text
        pos1 = text.find(entity1)
        pos2 = text.find(entity2)
        
        if pos1 == -1 or pos2 == -1:
            return 0.0
        
        # Calculate distance
        distance = abs(pos1 - pos2)
        
        # Normalize distance (closer = higher similarity)
        max_distance = len(text) * 0.1  # 10% of text length
        distance_score = max(0, 1 - (distance / max_distance))
        
        # Check for relationship indicators in context
        context_score = self._get_context_relationship_score(entity1, entity2, text)
        
        return (distance_score * 0.6) + (context_score * 0.4)
    
    def _get_entity_category(self, entity: str) -> Optional[str]:
        """Get the semantic category of an entity."""
        for category, pattern_info in self.semantic_patterns.items():
            for keyword in pattern_info["keywords"]:
                if keyword in entity:
                    return category
        return None
    
    def _get_context_relationship_score(self, entity1: str, entity2: str, text: str) -> float:
        """Get relationship score based on context indicators."""
        # Find context around both entities
        pos1 = text.find(entity1)
        pos2 = text.find(entity2)
        
        if pos1 == -1 or pos2 == -1:
            return 0.0
        
        # Get context window around both entities
        start = min(pos1, pos2) - self.context_window
        end = max(pos1, pos2) + self.context_window
        context = text[max(0, start):min(len(text), end)]
        
        # Check for relationship indicators
        total_score = 0.0
        indicator_count = 0
        
        for rel_type, indicators in self.relationship_indicators.items():
            for indicator in indicators:
                if indicator in context:
                    # Different weights for different relationship types
                    weights = {
                        "hierarchical": 0.8,
                        "collaborative": 0.7,
                        "competitive": 0.6,
                        "supportive": 0.7,
                        "oppositional": 0.6
                    }
                    total_score += weights.get(rel_type, 0.5)
                    indicator_count += 1
        
        return total_score / max(indicator_count, 1)
    
    def _get_context_evidence(self, entity1: str, entity2: str, text: str) -> List[str]:
        """Get contextual evidence for entity similarity."""
        evidence = []
        
        # Find sentences containing both entities
        sentences = re.split(r'[。！？]', text)
        
        for sentence in sentences:
            if entity1 in sentence and entity2 in sentence:
                # Clean up sentence
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 10:  # Only meaningful sentences
                    evidence.append(clean_sentence)
        
        # Limit evidence to top 3 most relevant
        return evidence[:3]
    
    def _determine_similarity_type(self, lexical: float, semantic: float, contextual: float) -> str:
        """Determine the type of similarity based on scores."""
        if semantic > 0.6:
            return "semantic"
        elif contextual > 0.5:
            return "contextual"
        elif lexical > 0.3:
            return "lexical"
        else:
            return "weak"
    
    def _calculate_confidence(self, similarity_score: float, evidence_count: int) -> float:
        """Calculate confidence based on similarity score and evidence."""
        base_confidence = similarity_score
        
        # Boost confidence based on evidence count
        evidence_boost = min(evidence_count * 0.1, 0.3)
        
        return min(base_confidence + evidence_boost, 1.0)
    
    def get_similarity_statistics(self, results: List[SimilarityResult]) -> Dict[str, Any]:
        """Get statistics about similarity analysis results."""
        if not results:
            return {
                "total_pairs": 0,
                "average_similarity": 0.0,
                "high_similarity_pairs": 0,
                "similarity_types": {},
                "average_confidence": 0.0
            }
        
        total_pairs = len(results)
        average_similarity = sum(r.similarity_score for r in results) / total_pairs
        high_similarity_pairs = len([r for r in results if r.similarity_score > 0.7])
        average_confidence = sum(r.confidence for r in results) / total_pairs
        
        # Count similarity types
        similarity_types = defaultdict(int)
        for result in results:
            similarity_types[result.similarity_type] += 1
        
        return {
            "total_pairs": total_pairs,
            "average_similarity": average_similarity,
            "high_similarity_pairs": high_similarity_pairs,
            "similarity_types": dict(similarity_types),
            "average_confidence": average_confidence
        }
    
    def filter_high_similarity_pairs(self, results: List[SimilarityResult], threshold: float = 0.6) -> List[SimilarityResult]:
        """Filter results to only include high similarity pairs."""
        return [r for r in results if r.similarity_score >= threshold]
    
    def get_relationship_suggestions(self, results: List[SimilarityResult]) -> List[Dict[str, Any]]:
        """Generate relationship suggestions based on similarity analysis."""
        suggestions = []
        
        for result in results:
            if result.similarity_score > 0.5:  # Only suggest for meaningful similarities
                relationship_type = self._suggest_relationship_type(result)
                
                suggestions.append({
                    "entity1": result.entity1,
                    "entity2": result.entity2,
                    "suggested_relationship": relationship_type,
                    "confidence": result.confidence,
                    "evidence": result.context_evidence,
                    "similarity_score": result.similarity_score
                })
        
        return suggestions
    
    def _suggest_relationship_type(self, result: SimilarityResult) -> str:
        """Suggest relationship type based on similarity analysis."""
        if result.similarity_type == "semantic":
            return "RELATED_TO"
        elif result.similarity_type == "contextual":
            # Analyze context evidence for specific relationship types
            for evidence in result.context_evidence:
                for rel_type, indicators in self.relationship_indicators.items():
                    for indicator in indicators:
                        if indicator in evidence:
                            if rel_type == "hierarchical":
                                return "CONTAINS"
                            elif rel_type == "collaborative":
                                return "COLLABORATES_WITH"
                            elif rel_type == "competitive":
                                return "COMPETES_WITH"
                            elif rel_type == "supportive":
                                return "SUPPORTS"
                            elif rel_type == "oppositional":
                                return "OPPOSES"
            return "RELATED_TO"
        else:
            return "WEAKLY_RELATED"
