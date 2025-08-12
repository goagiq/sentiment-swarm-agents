"""
Chinese Hierarchical Relationship Creator for Phase 2 orphan node reduction.
Implements advanced relationship creation algorithms specifically for Chinese content.
"""

from typing import Dict, List, Any, Tuple, Optional
import re
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Entity:
    """Entity with position and context information."""
    text: str
    entity_type: str
    position: int
    confidence: float = 0.7
    context: str = ""


@dataclass
class Relationship:
    """Relationship between entities."""
    source: str
    target: str
    relationship_type: str
    confidence: float
    description: str = ""


class ChineseHierarchicalRelationshipCreator:
    """Advanced relationship creator for Chinese content to reduce orphan nodes."""
    
    def __init__(self):
        # Chinese-specific relationship templates
        self.relationship_templates = {
            "person_organization": {
                "WORKS_FOR": 0.9,
                "FOUNDED": 0.8,
                "MANAGES": 0.8,
                "CONSULTS_FOR": 0.7
            },
            "person_location": {
                "LIVES_IN": 0.8,
                "WORKS_IN": 0.8,
                "VISITS": 0.6
            },
            "organization_location": {
                "HEADQUARTERED_IN": 0.9,
                "HAS_BRANCH_IN": 0.8,
                "OPERATES_IN": 0.8
            },
            "organization_organization": {
                "SUBSIDIARY_OF": 0.9,
                "PARTNER_OF": 0.8,
                "COMPETES_WITH": 0.7,
                "COLLABORATES_WITH": 0.8
            },
            "concept_concept": {
                "RELATED_TO": 0.7,
                "SIMILAR_TO": 0.6,
                "OPPOSES": 0.7,
                "ENHANCES": 0.7
            },
            "person_concept": {
                "EXPERT_IN": 0.8,
                "STUDIES": 0.7,
                "DEVELOPED": 0.8,
                "PROMOTES": 0.7
            },
            "organization_concept": {
                "SPECIALIZES_IN": 0.8,
                "IMPLEMENTS": 0.8,
                "RESEARCHES": 0.8,
                "PROVIDES": 0.7
            }
        }
        
        # Chinese-specific context patterns
        self.context_patterns = {
            "WORKS_FOR": [
                r"在.*工作", r"就职于", r"任职于", r"担任.*职位",
                r"为.*工作", r"在.*担任", r"就职.*公司"
            ],
            "LIVES_IN": [
                r"居住在", r"住在", r"定居于", r"位于",
                r"在.*生活", r"家住.*", r"居住在.*"
            ],
            "HEADQUARTERED_IN": [
                r"总部位于", r"总部设在", r"总部在",
                r"公司位于", r"企业位于", r"机构位于"
            ],
            "EXPERT_IN": [
                r"专家", r"权威", r"精通", r"擅长",
                r"在.*领域", r"专业.*", r"研究.*"
            ]
        }
    
    def create_hierarchical_relationships(self, entities: List[Dict], text: str) -> List[Relationship]:
        """Create hierarchical relationships for Chinese entities."""
        relationships = []
        
        # Convert entities to Entity objects with position information
        entity_objects = self._extract_entities_with_positions(entities, text)
        
        # Create different types of relationships
        relationships.extend(self._create_parent_child_relationships(entity_objects, text))
        relationships.extend(self._create_sibling_relationships(entity_objects, text))
        relationships.extend(self._create_location_hierarchies(entity_objects, text))
        relationships.extend(self._create_organization_hierarchies(entity_objects, text))
        relationships.extend(self._create_concept_relationships(entity_objects, text))
        
        # Remove duplicates and return
        return self._deduplicate_relationships(relationships)
    
    def _extract_entities_with_positions(self, entities: List[Dict], text: str) -> List[Entity]:
        """Extract entities with their positions in the text."""
        entity_objects = []
        
        for entity in entities:
            entity_text = entity.get("text", entity.get("name", ""))
            entity_type = entity.get("type", "CONCEPT").upper()
            
            # Find position in text
            position = text.find(entity_text)
            if position != -1:
                # Get context around the entity
                start = max(0, position - 50)
                end = min(len(text), position + len(entity_text) + 50)
                context = text[start:end]
                
                entity_objects.append(Entity(
                    text=entity_text,
                    entity_type=entity_type,
                    position=position,
                    confidence=entity.get("confidence", 0.7),
                    context=context
                ))
        
        return entity_objects
    
    def _create_parent_child_relationships(self, entities: List[Entity], text: str) -> List[Relationship]:
        """Create parent-child relationships based on entity types and context."""
        relationships = []
        
        # Group entities by type
        by_type = defaultdict(list)
        for entity in entities:
            by_type[entity.entity_type].append(entity)
        
        # Create organization -> person relationships
        organizations = by_type.get("ORGANIZATION", [])
        persons = by_type.get("PERSON", [])
        
        for person in persons:
            for org in organizations:
                # Check if person and organization are mentioned together
                if self._are_entities_related(person, org, text):
                    relationship_type = self._determine_relationship_type(person, org, text)
                    confidence = self._calculate_relationship_confidence(person, org, relationship_type)
                    
                    relationships.append(Relationship(
                        source=person.text,
                        target=org.text,
                        relationship_type=relationship_type,
                        confidence=confidence,
                        description=f"{person.text} {relationship_type.lower().replace('_', ' ')} {org.text}"
                    ))
        
        return relationships
    
    def _create_sibling_relationships(self, entities: List[Entity], text: str) -> List[Relationship]:
        """Create sibling relationships within the same category."""
        relationships = []
        
        # Group entities by type
        by_type = defaultdict(list)
        for entity in entities:
            by_type[entity.entity_type].append(entity)
        
        # Create relationships within each category
        for entity_type, entity_list in by_type.items():
            if len(entity_list) > 1:
                # Create relationships between entities of the same type
                for i in range(len(entity_list)):
                    for j in range(i + 1, len(entity_list)):
                        entity1 = entity_list[i]
                        entity2 = entity_list[j]
                        
                        # Check if they are mentioned close together
                        if abs(entity1.position - entity2.position) < 200:
                            relationship_type = self._get_sibling_relationship_type(entity_type)
                            confidence = 0.6  # Lower confidence for sibling relationships
                            
                            relationships.append(Relationship(
                                source=entity1.text,
                                target=entity2.text,
                                relationship_type=relationship_type,
                                confidence=confidence,
                                description=f"{entity1.text} and {entity2.text} are related {entity_type.lower()}s"
                            ))
        
        return relationships
    
    def _create_location_hierarchies(self, entities: List[Entity], text: str) -> List[Relationship]:
        """Create location-based hierarchical relationships."""
        relationships = []
        
        locations = [e for e in entities if e.entity_type == "LOCATION"]
        organizations = [e for e in entities if e.entity_type == "ORGANIZATION"]
        persons = [e for e in entities if e.entity_type == "PERSON"]
        
        # Create organization -> location relationships
        for org in organizations:
            for location in locations:
                if self._are_entities_related(org, location, text):
                    relationships.append(Relationship(
                        source=org.text,
                        target=location.text,
                        relationship_type="HEADQUARTERED_IN",
                        confidence=0.8,
                        description=f"{org.text} is headquartered in {location.text}"
                    ))
        
        # Create person -> location relationships
        for person in persons:
            for location in locations:
                if self._are_entities_related(person, location, text):
                    relationships.append(Relationship(
                        source=person.text,
                        target=location.text,
                        relationship_type="LIVES_IN",
                        confidence=0.7,
                        description=f"{person.text} lives in {location.text}"
                    ))
        
        return relationships
    
    def _create_organization_hierarchies(self, entities: List[Entity], text: str) -> List[Relationship]:
        """Create organization-based hierarchical relationships."""
        relationships = []
        
        organizations = [e for e in entities if e.entity_type == "ORGANIZATION"]
        
        # Create parent-child organization relationships
        for i, org1 in enumerate(organizations):
            for j, org2 in enumerate(organizations):
                if i != j:
                    # Check if one organization is mentioned as part of another
                    if self._is_sub_organization(org1, org2, text):
                        relationships.append(Relationship(
                            source=org1.text,
                            target=org2.text,
                            relationship_type="SUBSIDIARY_OF",
                            confidence=0.8,
                            description=f"{org1.text} is a subsidiary of {org2.text}"
                        ))
        
        return relationships
    
    def _create_concept_relationships(self, entities: List[Entity], text: str) -> List[Relationship]:
        """Create concept-based relationships."""
        relationships = []
        
        concepts = [e for e in entities if e.entity_type == "CONCEPT"]
        persons = [e for e in entities if e.entity_type == "PERSON"]
        organizations = [e for e in entities if e.entity_type == "ORGANIZATION"]
        
        # Create person -> concept relationships
        for person in persons:
            for concept in concepts:
                if self._are_entities_related(person, concept, text):
                    relationships.append(Relationship(
                        source=person.text,
                        target=concept.text,
                        relationship_type="EXPERT_IN",
                        confidence=0.7,
                        description=f"{person.text} is an expert in {concept.text}"
                    ))
        
        # Create organization -> concept relationships
        for org in organizations:
            for concept in concepts:
                if self._are_entities_related(org, concept, text):
                    relationships.append(Relationship(
                        source=org.text,
                        target=concept.text,
                        relationship_type="SPECIALIZES_IN",
                        confidence=0.7,
                        description=f"{org.text} specializes in {concept.text}"
                    ))
        
        # Create concept -> concept relationships
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts):
                if i != j and self._are_concepts_related(concept1, concept2, text):
                    relationships.append(Relationship(
                        source=concept1.text,
                        target=concept2.text,
                        relationship_type="RELATED_TO",
                        confidence=0.6,
                        description=f"{concept1.text} is related to {concept2.text}"
                    ))
        
        return relationships
    
    def _are_entities_related(self, entity1: Entity, entity2: Entity, text: str) -> bool:
        """Check if two entities are mentioned together in the text."""
        # Check if entities are mentioned close together
        if abs(entity1.position - entity2.position) < 100:
            return True
        
        # Check if they appear in the same sentence
        sentences = re.split(r'[。！？]', text)
        for sentence in sentences:
            if entity1.text in sentence and entity2.text in sentence:
                return True
        
        return False
    
    def _determine_relationship_type(self, entity1: Entity, entity2: Entity, text: str) -> str:
        """Determine the type of relationship between two entities."""
        entity_type_pair = f"{entity1.entity_type.lower()}_{entity2.entity_type.lower()}"
        
        if entity_type_pair in self.relationship_templates:
            # Check context for specific relationship types
            for rel_type, patterns in self.context_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, entity1.context) or re.search(pattern, entity2.context):
                        return rel_type
            
            # Return default relationship type
            return list(self.relationship_templates[entity_type_pair].keys())[0]
        
        return "RELATED_TO"
    
    def _calculate_relationship_confidence(self, entity1: Entity, entity2: Entity, relationship_type: str) -> float:
        """Calculate confidence score for a relationship."""
        entity_type_pair = f"{entity1.entity_type.lower()}_{entity2.entity_type.lower()}"
        
        if entity_type_pair in self.relationship_templates:
            return self.relationship_templates[entity_type_pair].get(relationship_type, 0.7)
        
        return 0.6
    
    def _get_sibling_relationship_type(self, entity_type: str) -> str:
        """Get relationship type for sibling entities."""
        sibling_types = {
            "PERSON": "COLLEAGUE_OF",
            "ORGANIZATION": "PARTNER_OF",
            "LOCATION": "NEAR_TO",
            "CONCEPT": "RELATED_TO"
        }
        return sibling_types.get(entity_type, "RELATED_TO")
    
    def _is_sub_organization(self, org1: Entity, org2: Entity, text: str) -> bool:
        """Check if one organization is a subsidiary of another."""
        # Check for subsidiary indicators in context
        subsidiary_patterns = [
            r"子公司", r"分公司", r"分支机构", r"下属", r"附属"
        ]
        
        for pattern in subsidiary_patterns:
            if re.search(pattern, org1.context) and org2.text in org1.context:
                return True
        
        return False
    
    def _are_concepts_related(self, concept1: Entity, concept2: Entity, text: str) -> bool:
        """Check if two concepts are related."""
        # Check if concepts appear in similar contexts
        if abs(concept1.position - concept2.position) < 150:
            return True
        
        # Check for concept relationship indicators
        relationship_indicators = [
            r"相关", r"类似", r"相似", r"联系", r"关系"
        ]
        
        for indicator in relationship_indicators:
            if re.search(indicator, concept1.context) and concept2.text in concept1.context:
                return True
        
        return False
    
    def _deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Remove duplicate relationships."""
        seen = set()
        unique_relationships = []
        
        for rel in relationships:
            rel_key = (rel.source, rel.target, rel.relationship_type)
            if rel_key not in seen:
                seen.add(rel_key)
                unique_relationships.append(rel)
        
        return unique_relationships
