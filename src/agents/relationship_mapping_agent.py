"""
Relationship Mapping Agent for mapping relationships between entities.
Extracted from the knowledge graph agent to provide focused relationship mapping capabilities.
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path

import logging

from src.agents.base_agent import StrandsBaseAgent
from src.core.models import (
    AnalysisRequest, 
    AnalysisResult, 
    DataType, 
    SentimentResult,
    ProcessingStatus
)
from src.core.processing_service import ProcessingService
from src.core.error_handling_service import ErrorHandlingService, ErrorContext
from src.core.model_management_service import ModelManagementService
from src.core.strands_mock import tool

# Configure logger
logger = logging.getLogger(__name__)


class RelationshipMappingAgent(StrandsBaseAgent):
    """Agent for mapping relationships between entities."""
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        **kwargs
    ):
        # Initialize services
        self.model_management_service = ModelManagementService()
        self.processing_service = ProcessingService()
        self.error_handling_service = ErrorHandlingService()
        
        # Set model name before calling super().__init__
        self.model_name = model_name or self.model_management_service.get_best_model("text")
        
        super().__init__(
            model_name=self.model_name,
            **kwargs
        )
        
        # Relationship categories
        self.relationship_categories = {
            "hierarchical": ["is_a", "part_of", "subclass_of", "contains", "belongs_to"],
            "temporal": ["before", "after", "during", "starts", "ends", "overlaps"],
            "spatial": ["located_in", "near", "adjacent_to", "within", "outside"],
            "causal": ["causes", "results_in", "leads_to", "triggers", "prevents"],
            "functional": ["uses", "produces", "consumes", "manages", "supports"],
            "social": ["works_with", "reports_to", "collaborates_with", "mentors"],
            "ownership": ["owns", "belongs_to", "manages", "controls", "operates"],
            "other": ["related_to", "associated_with", "connected_to", "similar_to"]
        }
        
        # Agent metadata
        self.metadata.update({
            "agent_type": "relationship_mapping",
            "model": self.model_name,
            "capabilities": [
                "relationship_extraction",
                "relationship_categorization",
                "relationship_validation",
                "bidirectional_mapping",
                "relationship_scoring"
            ],
            "supported_data_types": [
                DataType.TEXT,
                DataType.AUDIO,
                DataType.VIDEO,
                DataType.WEBPAGE,
                DataType.PDF,
                DataType.SOCIAL_MEDIA
            ],
            "relationship_categories": list(self.relationship_categories.keys())
        })
        
        logger.info(
            f"Relationship Mapping Agent {self.agent_id} initialized with model "
            f"{self.model_name}"
        )
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            self.map_relationships,
            self.extract_relationships,
            self.categorize_relationships,
            self.validate_relationships,
            self.find_relationship_paths,
            self.get_relationship_statistics
        ]
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        return request.data_type in [
            DataType.TEXT,
            DataType.AUDIO,
            DataType.VIDEO,
            DataType.WEBPAGE,
            DataType.PDF,
            DataType.SOCIAL_MEDIA
        ]
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process the analysis request."""
        context = ErrorContext(
            agent_id=self.agent_id,
            operation="relationship_mapping",
            request_id=request.id
        )
        
        return await self.error_handling_service.safe_execute_async(
            self._map_relationships_from_request,
            request,
            context,
            AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(label="neutral", confidence=0.0),
                status=ProcessingStatus.FAILED
            )
        )
    
    async def _map_relationships_from_request(
        self, 
        request: AnalysisRequest, 
        context: ErrorContext
    ) -> AnalysisResult:
        """Extract relationships from the analysis request."""
        # Extract text content
        text_content = await self._extract_text_content(request)
        
        # Extract entities first (assuming they're provided or we extract them)
        entities = await self._extract_entities_from_text(text_content)
        
        # Map relationships between entities
        relationships = await self._extract_relationships_from_text(
            text_content, entities
        )
        
        # Create sentiment from relationships
        sentiment = await self._create_sentiment_from_relationships(relationships)
        
        # Prepare metadata
        metadata = {
            "entities_count": len(entities),
            "relationships_count": len(relationships),
            "relationship_categories": self._count_relationships_by_category(relationships),
            "entities": entities,
            "relationships": relationships,
            "model_used": self.model_name
        }
        
        return AnalysisResult(
            request_id=request.id,
            data_type=request.data_type,
            sentiment=sentiment,
            status=ProcessingStatus.COMPLETED,
            metadata=metadata
        )
    
    async def _extract_text_content(self, request: AnalysisRequest) -> str:
        """Extract text content from the request."""
        return self.processing_service.extract_text_content(request.content)
    
    async def _extract_entities_from_text(self, text: str) -> List[Dict]:
        """Extract entities from text using a simple pattern-based approach."""
        # This is a simplified entity extraction - in practice, you might use
        # the EntityExtractionAgent or a more sophisticated approach
        entities = []
        
        # Simple pattern matching for common entity types
        patterns = {
            "person": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            "organization": r'\b[A-Z][A-Za-z\s&]+(?:Inc|Corp|LLC|Ltd|Company|Organization)\b',
            "location": r'\b[A-Z][a-z]+(?: City| State| Country| Province)\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    "name": match.group(),
                    "type": entity_type,
                    "confidence": 0.7,
                    "position": match.span()
                })
        
        return entities
    
    async def _extract_relationships_from_text(
        self, 
        text: str, 
        entities: List[Dict]
    ) -> List[Dict]:
        """Extract relationships between entities from text."""
        if len(entities) < 2:
            return []
        
        relationships = []
        
        # Create entity pairs
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Look for relationship patterns between these entities
                relationship = await self._find_relationship_between_entities(
                    text, entity1, entity2
                )
                if relationship:
                    relationships.append(relationship)
        
        return relationships
    
    async def _find_relationship_between_entities(
        self, 
        text: str, 
        entity1: Dict, 
        entity2: Dict
    ) -> Optional[Dict]:
        """Find relationship between two entities in text."""
        # Get text between entities
        start = min(entity1["position"][0], entity2["position"][0])
        end = max(entity1["position"][1], entity2["position"][1])
        
        # Extract context around entities
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)
        context = text[context_start:context_end]
        
        # Look for relationship indicators
        relationship_indicators = [
            "works for", "manages", "reports to", "collaborates with",
            "is part of", "belongs to", "owns", "founded", "located in",
            "based in", "from", "to", "with", "and", "or"
        ]
        
        for indicator in relationship_indicators:
            if indicator.lower() in context.lower():
                return {
                    "source": entity1["name"],
                    "target": entity2["name"],
                    "relationship_type": indicator,
                    "confidence": 0.6,
                    "context": context,
                    "category": self._categorize_relationship(indicator)
                }
        
        return None
    
    def _categorize_relationship(self, relationship_type: str) -> str:
        """Categorize a relationship type."""
        relationship_lower = relationship_type.lower()
        
        for category, indicators in self.relationship_categories.items():
            for indicator in indicators:
                if indicator in relationship_lower:
                    return category
        
        return "other"
    
    def _count_relationships_by_category(self, relationships: List[Dict]) -> Dict[str, int]:
        """Count relationships by category."""
        counts = {}
        for relationship in relationships:
            category = relationship.get("category", "other")
            counts[category] = counts.get(category, 0) + 1
        return counts
    
    async def _create_sentiment_from_relationships(
        self, 
        relationships: List[Dict]
    ) -> SentimentResult:
        """Create sentiment from relationships."""
        if not relationships:
            return SentimentResult(label="neutral", confidence=0.5)
        
        # Simple sentiment analysis based on relationship types
        positive_indicators = ["collaborates", "supports", "helps", "works with"]
        negative_indicators = ["conflicts", "opposes", "competes", "against"]
        
        positive_count = 0
        negative_count = 0
        
        for relationship in relationships:
            rel_type = relationship.get("relationship_type", "").lower()
            
            if any(indicator in rel_type for indicator in positive_indicators):
                positive_count += 1
            elif any(indicator in rel_type for indicator in negative_indicators):
                negative_count += 1
        
        total = len(relationships)
        if total == 0:
            return SentimentResult(label="neutral", confidence=0.5)
        
        positive_ratio = positive_count / total
        negative_ratio = negative_count / total
        
        if positive_ratio > negative_ratio:
            return SentimentResult(
                label="positive", 
                confidence=positive_ratio
            )
        elif negative_ratio > positive_ratio:
            return SentimentResult(
                label="negative", 
                confidence=negative_ratio
            )
        else:
            return SentimentResult(label="neutral", confidence=0.5)
    
    @tool
    async def map_relationships(self, text: str, entities: List[Dict]) -> dict:
        """Map relationships between entities in text."""
        try:
            logger.info(f"Mapping relationships for {len(entities)} entities")
            
            # Extract relationships
            relationships = await self._extract_relationships_from_text(text, entities)
            
            # Categorize relationships
            categorized_relationships = []
            for relationship in relationships:
                relationship["category"] = self._categorize_relationship(
                    relationship["relationship_type"]
                )
                categorized_relationships.append(relationship)
            
            # Generate statistics
            stats = self._count_relationships_by_category(categorized_relationships)
            
            return {
                "success": True,
                "relationships": categorized_relationships,
                "statistics": {
                    "total_relationships": len(categorized_relationships),
                    "relationships_by_category": stats,
                    "entities_involved": len(set(
                        [r["source"] for r in categorized_relationships] +
                        [r["target"] for r in categorized_relationships]
                    ))
                },
                "model_used": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error mapping relationships: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "relationships": [],
                "statistics": {}
            }
    
    @tool
    async def extract_relationships(self, text: str) -> dict:
        """Extract relationships from text without requiring pre-extracted entities."""
        try:
            logger.info("Extracting relationships from text")
            
            # Extract entities first
            entities = await self._extract_entities_from_text(text)
            
            # Extract relationships
            relationships = await self._extract_relationships_from_text(text, entities)
            
            return {
                "success": True,
                "entities": entities,
                "relationships": relationships,
                "statistics": {
                    "entities_count": len(entities),
                    "relationships_count": len(relationships)
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting relationships: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "entities": [],
                "relationships": []
            }
    
    @tool
    async def categorize_relationships(self, relationships: List[Dict]) -> dict:
        """Categorize relationships by type."""
        try:
            categorized = []
            for relationship in relationships:
                relationship["category"] = self._categorize_relationship(
                    relationship["relationship_type"]
                )
                categorized.append(relationship)
            
            stats = self._count_relationships_by_category(categorized)
            
            return {
                "success": True,
                "categorized_relationships": categorized,
                "statistics": stats
            }
            
        except Exception as e:
            logger.error(f"Error categorizing relationships: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "categorized_relationships": []
            }
    
    @tool
    async def validate_relationships(self, relationships: List[Dict]) -> dict:
        """Validate relationships for consistency and completeness."""
        try:
            valid_relationships = []
            invalid_relationships = []
            
            for relationship in relationships:
                # Check required fields
                required_fields = ["source", "target", "relationship_type"]
                missing_fields = [
                    field for field in required_fields 
                    if field not in relationship or not relationship[field]
                ]
                
                if missing_fields:
                    relationship["validation_errors"] = f"Missing fields: {missing_fields}"
                    invalid_relationships.append(relationship)
                else:
                    valid_relationships.append(relationship)
            
            return {
                "success": True,
                "valid_relationships": valid_relationships,
                "invalid_relationships": invalid_relationships,
                "validation_rate": len(valid_relationships) / len(relationships) if relationships else 0
            }
            
        except Exception as e:
            logger.error(f"Error validating relationships: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "valid_relationships": [],
                "invalid_relationships": []
            }
    
    @tool
    async def find_relationship_paths(self, source: str, target: str, relationships: List[Dict]) -> dict:
        """Find paths between two entities through relationships."""
        try:
            # Build a simple graph from relationships
            graph = {}
            for rel in relationships:
                source_entity = rel["source"]
                target_entity = rel["target"]
                
                if source_entity not in graph:
                    graph[source_entity] = []
                if target_entity not in graph:
                    graph[target_entity] = []
                
                graph[source_entity].append(target_entity)
                graph[target_entity].append(source_entity)
            
            # Simple BFS to find paths
            def find_paths(start, end, max_depth=3):
                if start not in graph or end not in graph:
                    return []
                
                queue = [(start, [start])]
                paths = []
                
                while queue and len(paths) < 10:  # Limit to 10 paths
                    current, path = queue.pop(0)
                    
                    if current == end and len(path) > 1:
                        paths.append(path)
                        continue
                    
                    if len(path) >= max_depth:
                        continue
                    
                    for neighbor in graph.get(current, []):
                        if neighbor not in path:
                            queue.append((neighbor, path + [neighbor]))
                
                return paths
            
            paths = find_paths(source, target)
            
            return {
                "success": True,
                "source": source,
                "target": target,
                "paths": paths,
                "path_count": len(paths)
            }
            
        except Exception as e:
            logger.error(f"Error finding relationship paths: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "paths": []
            }
    
    @tool
    async def get_relationship_statistics(self) -> dict:
        """Get statistics about relationship mapping capabilities."""
        return {
            "success": True,
            "agent_id": self.agent_id,
            "model": self.model_name,
            "relationship_categories": list(self.relationship_categories.keys()),
            "category_count": len(self.relationship_categories),
            "capabilities": self.metadata["capabilities"]
        }
