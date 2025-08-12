"""
Chinese Advanced Fallback Strategies for Phase 2 orphan node reduction.
Implements multi-level fallback mechanisms to ensure maximum relationship coverage.
"""

from typing import Dict, List, Any, Tuple, Optional
import re
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class FallbackResult:
    """Result of fallback strategy application."""
    relationships: List[Dict[str, Any]]
    strategy_used: str
    confidence: float
    entities_covered: int


class ChineseFallbackStrategies:
    """Advanced fallback strategies for Chinese content to reduce orphan nodes."""
    
    def __init__(self):
        # Chinese-specific fallback templates
        self.fallback_templates = {
            "hierarchical": {
                "person_organization": "WORKS_FOR",
                "person_location": "LIVES_IN",
                "organization_location": "HEADQUARTERED_IN",
                "person_concept": "EXPERT_IN",
                "organization_concept": "SPECIALIZES_IN"
            },
            "proximity": {
                "default": "NEAR_TO",
                "same_sentence": "RELATED_TO",
                "adjacent": "CONNECTED_TO"
            },
            "template": {
                "person_person": "COLLEAGUE_OF",
                "organization_organization": "PARTNER_OF",
                "location_location": "NEAR_TO",
                "concept_concept": "RELATED_TO"
            },
            "semantic": {
                "technology": "TECHNOLOGY_RELATED",
                "business": "BUSINESS_RELATED",
                "academic": "ACADEMIC_RELATED",
                "government": "GOVERNMENT_RELATED"
            }
        }
        
        # Chinese-specific rule patterns
        self.rule_patterns = {
            "company_employee": [
                r"([^。！？]*?{company}[^。！？]*?{person}[^。！？]*?[。！？])",
                r"([^。！？]*?{person}[^。！？]*?{company}[^。！？]*?[。！？])"
            ],
            "location_organization": [
                r"([^。！？]*?{location}[^。！？]*?{organization}[^。！？]*?[。！？])",
                r"([^。！？]*?{organization}[^。！？]*?{location}[^。！？]*?[。！？])"
            ],
            "person_expertise": [
                r"([^。！？]*?{person}[^。！？]*?{concept}[^。！？]*?[。！？])",
                r"([^。！？]*?{concept}[^。！？]*?{person}[^。！？]*?[。！？])"
            ]
        }
    
    def apply_fallback_strategies(self, entities: List[Dict], text: str, 
                                 existing_relationships: List[Dict] = None) -> List[FallbackResult]:
        """Apply multiple fallback strategies to create relationships."""
        if existing_relationships is None:
            existing_relationships = []
        
        results = []
        
        # Get entities that don't have relationships
        orphan_entities = self._get_orphan_entities(entities, existing_relationships)
        
        if not orphan_entities:
            return results
        
        # Apply fallback strategies in order of preference
        strategies = [
            ("hierarchical", self._hierarchical_fallback),
            ("proximity", self._proximity_fallback),
            ("template", self._template_fallback),
            ("semantic", self._semantic_fallback)
        ]
        
        for strategy_name, strategy_func in strategies:
            result = strategy_func(orphan_entities, text)
            if result.relationships:
                results.append(result)
                
                # Update orphan entities for next strategy
                orphan_entities = self._get_orphan_entities(entities, 
                                                          existing_relationships + 
                                                          [r for res in results for r in res.relationships])
                
                if not orphan_entities:
                    break
        
        return results
    
    def _get_orphan_entities(self, entities: List[Dict], 
                           relationships: List[Dict]) -> List[Dict]:
        """Get entities that don't have any relationships."""
        connected_entities = set()
        
        for rel in relationships:
            connected_entities.add(rel.get("source", rel.get("from", "")))
            connected_entities.add(rel.get("target", rel.get("to", "")))
        
        orphan_entities = []
        for entity in entities:
            entity_name = entity.get("text", entity.get("name", ""))
            if entity_name not in connected_entities:
                orphan_entities.append(entity)
        
        return orphan_entities
    
    def _hierarchical_fallback(self, entities: List[Dict], text: str) -> FallbackResult:
        """Apply hierarchical fallback strategy."""
        relationships = []
        
        # Group entities by type
        by_type = defaultdict(list)
        for entity in entities:
            entity_type = entity.get("type", "CONCEPT").upper()
            by_type[entity_type].append(entity)
        
        # Create hierarchical relationships
        templates = self.fallback_templates["hierarchical"]
        
        # Person -> Organization relationships
        persons = by_type.get("PERSON", [])
        organizations = by_type.get("ORGANIZATION", [])
        
        for person in persons:
            for org in organizations:
                if self._are_entities_related(person, org, text):
                    relationships.append({
                        "source": person.get("text", person.get("name", "")),
                        "target": org.get("text", org.get("name", "")),
                        "relationship_type": templates["person_organization"],
                        "confidence": 0.7,
                        "strategy": "hierarchical"
                    })
        
        # Person -> Location relationships
        locations = by_type.get("LOCATION", [])
        for person in persons:
            for location in locations:
                if self._are_entities_related(person, location, text):
                    relationships.append({
                        "source": person.get("text", person.get("name", "")),
                        "target": location.get("text", location.get("name", "")),
                        "relationship_type": templates["person_location"],
                        "confidence": 0.6,
                        "strategy": "hierarchical"
                    })
        
        # Organization -> Location relationships
        for org in organizations:
            for location in locations:
                if self._are_entities_related(org, location, text):
                    relationships.append({
                        "source": org.get("text", org.get("name", "")),
                        "target": location.get("text", location.get("name", "")),
                        "relationship_type": templates["organization_location"],
                        "confidence": 0.7,
                        "strategy": "hierarchical"
                    })
        
        # Person -> Concept relationships
        concepts = by_type.get("CONCEPT", [])
        for person in persons:
            for concept in concepts:
                if self._are_entities_related(person, concept, text):
                    relationships.append({
                        "source": person.get("text", person.get("name", "")),
                        "target": concept.get("text", concept.get("name", "")),
                        "relationship_type": templates["person_concept"],
                        "confidence": 0.6,
                        "strategy": "hierarchical"
                    })
        
        # Organization -> Concept relationships
        for org in organizations:
            for concept in concepts:
                if self._are_entities_related(org, concept, text):
                    relationships.append({
                        "source": org.get("text", org.get("name", "")),
                        "target": concept.get("text", concept.get("name", "")),
                        "relationship_type": templates["organization_concept"],
                        "confidence": 0.6,
                        "strategy": "hierarchical"
                    })
        
        return FallbackResult(
            relationships=relationships,
            strategy_used="hierarchical",
            confidence=0.7,
            entities_covered=len(set([r["source"] for r in relationships] + [r["target"] for r in relationships]))
        )
    
    def _proximity_fallback(self, entities: List[Dict], text: str) -> FallbackResult:
        """Apply proximity-based fallback strategy."""
        relationships = []
        
        # Find entities that appear close together in text
        entity_positions = []
        for entity in entities:
            entity_text = entity.get("text", entity.get("name", ""))
            position = text.find(entity_text)
            if position != -1:
                entity_positions.append((entity, position))
        
        # Sort by position
        entity_positions.sort(key=lambda x: x[1])
        
        # Create relationships between nearby entities
        proximity_threshold = 150  # characters
        
        for i in range(len(entity_positions)):
            for j in range(i + 1, len(entity_positions)):
                entity1, pos1 = entity_positions[i]
                entity2, pos2 = entity_positions[j]
                
                if pos2 - pos1 <= proximity_threshold:
                    # Determine relationship type based on proximity
                    if pos2 - pos1 <= 50:
                        rel_type = "CONNECTED_TO"
                        confidence = 0.6
                    elif pos2 - pos1 <= 100:
                        rel_type = "NEAR_TO"
                        confidence = 0.5
                    else:
                        rel_type = "RELATED_TO"
                        confidence = 0.4
                    
                    relationships.append({
                        "source": entity1.get("text", entity1.get("name", "")),
                        "target": entity2.get("text", entity2.get("name", "")),
                        "relationship_type": rel_type,
                        "confidence": confidence,
                        "strategy": "proximity"
                    })
        
        return FallbackResult(
            relationships=relationships,
            strategy_used="proximity",
            confidence=0.5,
            entities_covered=len(set([r["source"] for r in relationships] + [r["target"] for r in relationships]))
        )
    
    def _template_fallback(self, entities: List[Dict], text: str) -> FallbackResult:
        """Apply template-based fallback strategy."""
        relationships = []
        
        # Group entities by type
        by_type = defaultdict(list)
        for entity in entities:
            entity_type = entity.get("type", "CONCEPT").upper()
            by_type[entity_type].append(entity)
        
        templates = self.fallback_templates["template"]
        
        # Create relationships within each entity type
        for entity_type, entity_list in by_type.items():
            if len(entity_list) > 1:
                template_key = f"{entity_type.lower()}_{entity_type.lower()}"
                rel_type = templates.get(template_key, "RELATED_TO")
                
                # Create relationships between entities of the same type
                for i in range(len(entity_list)):
                    for j in range(i + 1, len(entity_list)):
                        relationships.append({
                            "source": entity_list[i].get("text", entity_list[i].get("name", "")),
                            "target": entity_list[j].get("text", entity_list[j].get("name", "")),
                            "relationship_type": rel_type,
                            "confidence": 0.4,
                            "strategy": "template"
                        })
        
        return FallbackResult(
            relationships=relationships,
            strategy_used="template",
            confidence=0.4,
            entities_covered=len(set([r["source"] for r in relationships] + [r["target"] for r in relationships]))
        )
    
    def _semantic_fallback(self, entities: List[Dict], text: str) -> FallbackResult:
        """Apply semantic-based fallback strategy."""
        relationships = []
        
        # Chinese-specific semantic categories
        semantic_categories = {
            "technology": [r"技术", r"科技", r"人工智能", r"机器学习", r"算法"],
            "business": [r"公司", r"企业", r"商业", r"市场", r"经济"],
            "academic": [r"大学", r"研究", r"学术", r"教育", r"教授"],
            "government": [r"政府", r"部门", r"机构", r"政策", r"法规"]
        }
        
        # Categorize entities
        categorized_entities = defaultdict(list)
        
        for entity in entities:
            entity_text = entity.get("text", entity.get("name", ""))
            for category, patterns in semantic_categories.items():
                for pattern in patterns:
                    if re.search(pattern, entity_text):
                        categorized_entities[category].append(entity)
                        break
        
        # Create relationships within semantic categories
        templates = self.fallback_templates["semantic"]
        
        for category, entity_list in categorized_entities.items():
            if len(entity_list) > 1:
                rel_type = templates.get(category, "RELATED_TO")
                
                for i in range(len(entity_list)):
                    for j in range(i + 1, len(entity_list)):
                        relationships.append({
                            "source": entity_list[i].get("text", entity_list[i].get("name", "")),
                            "target": entity_list[j].get("text", entity_list[j].get("name", "")),
                            "relationship_type": rel_type,
                            "confidence": 0.3,
                            "strategy": "semantic"
                        })
        
        return FallbackResult(
            relationships=relationships,
            strategy_used="semantic",
            confidence=0.3,
            entities_covered=len(set([r["source"] for r in relationships] + [r["target"] for r in relationships]))
        )
    
    def _are_entities_related(self, entity1: Dict, entity2: Dict, text: str) -> bool:
        """Check if two entities are mentioned together in the text."""
        entity1_text = entity1.get("text", entity1.get("name", ""))
        entity2_text = entity2.get("text", entity2.get("name", ""))
        
        # Check if they appear in the same sentence
        sentences = re.split(r'[。！？]', text)
        for sentence in sentences:
            if entity1_text in sentence and entity2_text in sentence:
                return True
        
        # Check if they appear close together
        pos1 = text.find(entity1_text)
        pos2 = text.find(entity2_text)
        
        if pos1 != -1 and pos2 != -1:
            if abs(pos1 - pos2) < 200:
                return True
        
        return False
    
    def get_fallback_statistics(self, results: List[FallbackResult]) -> Dict[str, Any]:
        """Get statistics about fallback strategy results."""
        total_relationships = sum(len(result.relationships) for result in results)
        total_entities_covered = sum(result.entities_covered for result in results)
        
        strategy_stats = defaultdict(lambda: {"count": 0, "confidence": 0.0})
        
        for result in results:
            strategy_stats[result.strategy_used]["count"] += len(result.relationships)
            strategy_stats[result.strategy_used]["confidence"] = result.confidence
        
        return {
            "total_fallback_relationships": total_relationships,
            "total_entities_covered": total_entities_covered,
            "strategies_used": len(results),
            "strategy_breakdown": dict(strategy_stats),
            "average_confidence": sum(result.confidence for result in results) / len(results) if results else 0
        }
