"""
Chinese Entity Clustering for Phase 2 orphan node reduction.
Implements semantic clustering, proximity analysis, and co-occurrence analysis for Chinese entities.
"""

from typing import Dict, List, Any, Tuple, Optional, Set
import re
from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass
class EntityCluster:
    """A cluster of related entities."""
    entities: List[str]
    cluster_type: str
    confidence: float
    relationships: List[Tuple[str, str, str]]  # (source, target, relationship_type)


class ChineseEntityClustering:
    """Advanced entity clustering for Chinese content to reduce orphan nodes."""
    
    def __init__(self):
        # Chinese-specific semantic similarity patterns
        self.semantic_patterns = {
            "technology": [
                r"技术", r"科技", r"人工智能", r"机器学习", r"深度学习",
                r"算法", r"软件", r"硬件", r"系统", r"平台"
            ],
            "business": [
                r"公司", r"企业", r"商业", r"市场", r"经济",
                r"投资", r"融资", r"创业", r"管理", r"战略"
            ],
            "academic": [
                r"大学", r"学院", r"研究", r"学术", r"教育",
                r"教授", r"博士", r"论文", r"实验室", r"学科"
            ],
            "government": [
                r"政府", r"部门", r"机构", r"政策", r"法规",
                r"官员", r"部长", r"委员会", r"办公室", r"局"
            ],
            "location": [
                r"城市", r"地区", r"省份", r"国家", r"区域",
                r"中心", r"区", r"街道", r"广场", r"园区"
            ]
        }
        
        # Chinese-specific proximity indicators
        self.proximity_indicators = [
            r"和", r"与", r"及", r"以及", r"还有", r"包括",
            r"其中", r"包括", r"例如", r"比如", r"特别是"
        ]
        
        # Chinese-specific co-occurrence patterns
        self.co_occurrence_patterns = [
            r"([^。！？]*?{entity1}[^。！？]*?{entity2}[^。！？]*?[。！？])",
            r"([^。！？]*?{entity2}[^。！？]*?{entity1}[^。！？]*?[。！？])"
        ]
    
    def cluster_entities(self, entities: List[Dict], text: str) -> List[EntityCluster]:
        """Cluster entities using multiple algorithms."""
        clusters = []
        
        # Convert entities to simple list
        entity_list = [entity.get("text", entity.get("name", "")) for entity in entities]
        entity_types = [entity.get("type", "CONCEPT").upper() for entity in entities]
        
        # Apply different clustering algorithms
        clusters.extend(self._semantic_clustering(entity_list, entity_types, text))
        clusters.extend(self._proximity_clustering(entity_list, entity_types, text))
        clusters.extend(self._co_occurrence_clustering(entity_list, entity_types, text))
        clusters.extend(self._category_clustering(entity_list, entity_types, text))
        
        # Merge overlapping clusters
        return self._merge_clusters(clusters)
    
    def _semantic_clustering(self, entities: List[str], entity_types: List[str], text: str) -> List[EntityCluster]:
        """Cluster entities based on semantic similarity."""
        clusters = []
        
        # Group entities by semantic category
        semantic_groups = defaultdict(list)
        
        for i, entity in enumerate(entities):
            entity_type = entity_types[i]
            semantic_category = self._get_semantic_category(entity, text)
            
            if semantic_category:
                semantic_groups[semantic_category].append(entity)
        
        # Create clusters for each semantic group
        for category, entity_group in semantic_groups.items():
            if len(entity_group) > 1:
                # Create relationships within the cluster
                relationships = []
                for i in range(len(entity_group)):
                    for j in range(i + 1, len(entity_group)):
                        relationship_type = self._get_semantic_relationship_type(category)
                        relationships.append((entity_group[i], entity_group[j], relationship_type))
                
                clusters.append(EntityCluster(
                    entities=entity_group,
                    cluster_type=f"semantic_{category}",
                    confidence=0.8,
                    relationships=relationships
                ))
        
        return clusters
    
    def _proximity_clustering(self, entities: List[str], entity_types: List[str], text: str) -> List[EntityCluster]:
        """Cluster entities based on proximity in text."""
        clusters = []
        
        # Find entities that appear close together
        entity_positions = []
        for entity in entities:
            position = text.find(entity)
            if position != -1:
                entity_positions.append((entity, position))
        
        # Sort by position
        entity_positions.sort(key=lambda x: x[1])
        
        # Group entities that are close together
        proximity_threshold = 200  # characters
        current_cluster = []
        
        for i, (entity, position) in enumerate(entity_positions):
            if not current_cluster:
                current_cluster = [entity]
            else:
                # Check if this entity is close to the last entity in current cluster
                last_entity, last_position = entity_positions[i - 1]
                if position - last_position <= proximity_threshold:
                    current_cluster.append(entity)
                else:
                    # Create cluster if it has multiple entities
                    if len(current_cluster) > 1:
                        relationships = self._create_proximity_relationships(current_cluster)
                        clusters.append(EntityCluster(
                            entities=current_cluster.copy(),
                            cluster_type="proximity",
                            confidence=0.7,
                            relationships=relationships
                        ))
                    current_cluster = [entity]
        
        # Handle the last cluster
        if len(current_cluster) > 1:
            relationships = self._create_proximity_relationships(current_cluster)
            clusters.append(EntityCluster(
                entities=current_cluster,
                cluster_type="proximity",
                confidence=0.7,
                relationships=relationships
            ))
        
        return clusters
    
    def _co_occurrence_clustering(self, entities: List[str], entity_types: List[str], text: str) -> List[EntityCluster]:
        """Cluster entities based on co-occurrence in sentences."""
        clusters = []
        
        # Split text into sentences
        sentences = re.split(r'[。！？]', text)
        
        # Find entities that co-occur in the same sentence
        co_occurrence_groups = defaultdict(set)
        
        for sentence in sentences:
            sentence_entities = []
            for entity in entities:
                if entity in sentence:
                    sentence_entities.append(entity)
            
            # Create relationships between all entities in the sentence
            if len(sentence_entities) > 1:
                for i in range(len(sentence_entities)):
                    for j in range(i + 1, len(sentence_entities)):
                        entity1, entity2 = sentence_entities[i], sentence_entities[j]
                        co_occurrence_groups[(entity1, entity2)].add(sentence)
        
        # Create clusters for frequently co-occurring entities
        for (entity1, entity2), sentences in co_occurrence_groups.items():
            if len(sentences) >= 1:  # At least one co-occurrence
                relationship_type = self._get_co_occurrence_relationship_type(entity1, entity2, text)
                clusters.append(EntityCluster(
                    entities=[entity1, entity2],
                    cluster_type="co_occurrence",
                    confidence=0.6 + min(len(sentences) * 0.1, 0.3),  # Higher confidence for more co-occurrences
                    relationships=[(entity1, entity2, relationship_type)]
                ))
        
        return clusters
    
    def _category_clustering(self, entities: List[str], entity_types: List[str], text: str) -> List[EntityCluster]:
        """Cluster entities based on their entity type categories."""
        clusters = []
        
        # Group entities by type
        type_groups = defaultdict(list)
        for entity, entity_type in zip(entities, entity_types):
            type_groups[entity_type].append(entity)
        
        # Create clusters for each entity type
        for entity_type, entity_group in type_groups.items():
            if len(entity_group) > 1:
                relationships = self._create_category_relationships(entity_group, entity_type)
                clusters.append(EntityCluster(
                    entities=entity_group,
                    cluster_type=f"category_{entity_type.lower()}",
                    confidence=0.7,
                    relationships=relationships
                ))
        
        return clusters
    
    def _get_semantic_category(self, entity: str, text: str) -> Optional[str]:
        """Determine the semantic category of an entity."""
        for category, patterns in self.semantic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, entity):
                    return category
        
        return None
    
    def _get_semantic_relationship_type(self, category: str) -> str:
        """Get relationship type for semantic clusters."""
        relationship_types = {
            "technology": "RELATED_TECHNOLOGY",
            "business": "BUSINESS_RELATED",
            "academic": "ACADEMIC_COLLABORATION",
            "government": "GOVERNMENT_RELATED",
            "location": "GEOGRAPHICALLY_RELATED"
        }
        return relationship_types.get(category, "RELATED_TO")
    
    def _create_proximity_relationships(self, entities: List[str]) -> List[Tuple[str, str, str]]:
        """Create relationships for proximity-based clusters."""
        relationships = []
        
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                relationships.append((entities[i], entities[j], "NEAR_TO"))
        
        return relationships
    
    def _get_co_occurrence_relationship_type(self, entity1: str, entity2: str, text: str) -> str:
        """Determine relationship type for co-occurring entities."""
        # Check for specific relationship indicators in the context
        context_patterns = {
            r"合作": "COLLABORATES_WITH",
            r"竞争": "COMPETES_WITH",
            r"支持": "SUPPORTS",
            r"反对": "OPPOSES",
            r"包含": "CONTAINS",
            r"属于": "BELONGS_TO"
        }
        
        # Find sentences containing both entities
        sentences = re.split(r'[。！？]', text)
        for sentence in sentences:
            if entity1 in sentence and entity2 in sentence:
                for pattern, rel_type in context_patterns.items():
                    if re.search(pattern, sentence):
                        return rel_type
        
        return "RELATED_TO"
    
    def _create_category_relationships(self, entities: List[str], entity_type: str) -> List[Tuple[str, str, str]]:
        """Create relationships for category-based clusters."""
        relationships = []
        
        relationship_types = {
            "PERSON": "COLLEAGUE_OF",
            "ORGANIZATION": "PARTNER_OF",
            "LOCATION": "NEAR_TO",
            "CONCEPT": "RELATED_TO"
        }
        
        rel_type = relationship_types.get(entity_type, "RELATED_TO")
        
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                relationships.append((entities[i], entities[j], rel_type))
        
        return relationships
    
    def _merge_clusters(self, clusters: List[EntityCluster]) -> List[EntityCluster]:
        """Merge overlapping clusters."""
        if not clusters:
            return clusters
        
        # Sort clusters by confidence (highest first)
        clusters.sort(key=lambda x: x.confidence, reverse=True)
        
        merged_clusters = []
        used_entities = set()
        
        for cluster in clusters:
            # Check if this cluster overlaps significantly with existing clusters
            should_add = True
            
            for existing_cluster in merged_clusters:
                overlap = set(cluster.entities) & set(existing_cluster.entities)
                if len(overlap) > 0:
                    # If there's significant overlap, merge them
                    if len(overlap) >= min(len(cluster.entities), len(existing_cluster.entities)) * 0.5:
                        should_add = False
                        # Merge entities and relationships
                        existing_cluster.entities = list(set(existing_cluster.entities + cluster.entities))
                        existing_cluster.relationships.extend(cluster.relationships)
                        # Update confidence to average
                        existing_cluster.confidence = (existing_cluster.confidence + cluster.confidence) / 2
                        break
            
            if should_add:
                merged_clusters.append(cluster)
                used_entities.update(cluster.entities)
        
        return merged_clusters
    
    def get_cluster_statistics(self, clusters: List[EntityCluster]) -> Dict[str, Any]:
        """Get statistics about the clustering results."""
        total_entities = sum(len(cluster.entities) for cluster in clusters)
        total_relationships = sum(len(cluster.relationships) for cluster in clusters)
        
        cluster_types = defaultdict(int)
        for cluster in clusters:
            cluster_types[cluster.cluster_type] += 1
        
        return {
            "total_clusters": len(clusters),
            "total_entities_clustered": total_entities,
            "total_relationships_created": total_relationships,
            "cluster_types": dict(cluster_types),
            "average_cluster_size": total_entities / len(clusters) if clusters else 0,
            "average_confidence": sum(cluster.confidence for cluster in clusters) / len(clusters) if clusters else 0
        }
