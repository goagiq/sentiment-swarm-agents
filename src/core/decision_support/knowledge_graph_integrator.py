"""
Knowledge Graph Integrator for Decision Support

Provides integration between the decision support system and knowledge graph
agent, enabling dynamic context extraction and entity-based decision enhancement.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.config.decision_support_config import (
    get_knowledge_graph_config,
    get_language_entity_patterns,
    get_language_reasoning_patterns
)
from src.core.models import AnalysisRequest
from src.core.unified_mcp_client import call_unified_mcp_tool

logger = logging.getLogger(__name__)


@dataclass
class EntityContext:
    """Context information extracted from knowledge graph entities."""
    entity_id: str
    entity_name: str
    entity_type: str
    confidence: float
    relationships: List[Dict[str, Any]]
    attributes: Dict[str, Any]
    language: str
    created_at: datetime


@dataclass
class DecisionContext:
    """Comprehensive decision context from knowledge graph."""
    business_entities: List[EntityContext]
    market_entities: List[EntityContext]
    risk_entities: List[EntityContext]
    opportunity_entities: List[EntityContext]
    constraint_entities: List[EntityContext]
    goal_entities: List[EntityContext]
    relationship_network: Dict[str, List[str]]
    historical_patterns: List[Dict[str, Any]]
    confidence_score: float
    language: str
    extracted_at: datetime


class KnowledgeGraphIntegrator:
    """Integrates knowledge graph capabilities with decision support system."""
    
    def __init__(self):
        self.config = get_knowledge_graph_config()
        self.cache: Dict[str, DecisionContext] = {}
        self.cache_ttl = 3600  # 1 hour
        
    async def extract_decision_context(
        self, 
        request: AnalysisRequest,
        language: str = "en"
    ) -> DecisionContext:
        """
        Extract comprehensive decision context from knowledge graph.
        
        Args:
            request: Analysis request containing content to analyze
            language: Language code for context extraction
            
        Returns:
            DecisionContext with extracted information
        """
        try:
            # Check cache first
            cache_key = f"{request.id}_{language}"
            if cache_key in self.cache:
                cached_context = self.cache[cache_key]
                if (datetime.now() - cached_context.extracted_at).seconds < self.cache_ttl:
                    logger.debug(f"Using cached decision context for {request.id}")
                    return cached_context
            
            # Extract entities from content
            entities = await self._extract_entities_from_content(
                request.content, language
            )
            
            # Query knowledge graph for related information
            related_entities = await self._query_related_entities(entities, language)
            
            # Build relationship network
            relationship_network = await self._build_relationship_network(
                entities + related_entities
            )
            
            # Analyze historical patterns
            historical_patterns = await self._analyze_historical_patterns(
                entities, language
            )
            
            # Categorize entities by type
            categorized_entities = self._categorize_entities(entities + related_entities)
            
            # Calculate confidence score
            confidence_score = self._calculate_context_confidence(
                entities, related_entities, historical_patterns
            )
            
            # Create decision context
            context = DecisionContext(
                business_entities=categorized_entities.get("business", []),
                market_entities=categorized_entities.get("market", []),
                risk_entities=categorized_entities.get("risk", []),
                opportunity_entities=categorized_entities.get("opportunity", []),
                constraint_entities=categorized_entities.get("constraint", []),
                goal_entities=categorized_entities.get("goal", []),
                relationship_network=relationship_network,
                historical_patterns=historical_patterns,
                confidence_score=confidence_score,
                language=language,
                extracted_at=datetime.now()
            )
            
            # Cache the context
            self.cache[cache_key] = context
            
            logger.info(f"Extracted decision context for {request.id} with {len(entities)} entities")
            return context
            
        except Exception as e:
            logger.error(f"Error extracting decision context: {e}")
            # Return empty context on error
            return DecisionContext(
                business_entities=[],
                market_entities=[],
                risk_entities=[],
                opportunity_entities=[],
                constraint_entities=[],
                goal_entities=[],
                relationship_network={},
                historical_patterns=[],
                confidence_score=0.0,
                language=language,
                extracted_at=datetime.now()
            )
    
    async def _extract_entities_from_content(
        self, 
        content: str, 
        language: str
    ) -> List[Dict[str, Any]]:
        """Extract entities from content using knowledge graph agent."""
        try:
            # Use MCP tool to extract entities
            result = await call_unified_mcp_tool(
                "extract_entities",
                content=content,
                language=language,
                entity_types=self.config.entity_types
            )
            
            if result and "entities" in result:
                return result["entities"]
            else:
                logger.warning("No entities extracted from content")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    async def _query_related_entities(
        self, 
        entities: List[Dict[str, Any]], 
        language: str
    ) -> List[Dict[str, Any]]:
        """Query knowledge graph for entities related to extracted entities."""
        try:
            related_entities = []
            
            for entity in entities[:self.config.max_entities_per_query]:
                # Query knowledge graph for related entities
                result = await call_unified_mcp_tool(
                    "query_knowledge_graph",
                    query=entity.get("name", ""),
                    entity_types=self.config.entity_types,
                    max_results=self.config.max_relationships_per_entity,
                    language=language
                )
                
                if result and "related_entities" in result:
                    related_entities.extend(result["related_entities"])
            
            # Remove duplicates
            seen = set()
            unique_entities = []
            for entity in related_entities:
                entity_id = entity.get("id", entity.get("name", ""))
                if entity_id not in seen:
                    seen.add(entity_id)
                    unique_entities.append(entity)
            
            return unique_entities
            
        except Exception as e:
            logger.error(f"Error querying related entities: {e}")
            return []
    
    async def _build_relationship_network(
        self, 
        entities: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Build relationship network from entities."""
        try:
            network = {}
            
            for entity in entities:
                entity_id = entity.get("id", entity.get("name", ""))
                relationships = entity.get("relationships", [])
                
                network[entity_id] = []
                for rel in relationships:
                    target_id = rel.get("target_id", rel.get("target_name", ""))
                    if target_id:
                        network[entity_id].append(target_id)
            
            return network
            
        except Exception as e:
            logger.error(f"Error building relationship network: {e}")
            return {}
    
    async def _analyze_historical_patterns(
        self, 
        entities: List[Dict[str, Any]], 
        language: str
    ) -> List[Dict[str, Any]]:
        """Analyze historical patterns related to entities."""
        try:
            patterns = []
            
            # Get language-specific patterns
            entity_patterns = get_language_entity_patterns(language)
            
            for entity in entities:
                entity_type = entity.get("type", "").lower()
                
                # Check for business patterns
                if entity_type in ["organization", "product", "market"]:
                    business_patterns = await self._find_business_patterns(
                        entity, language
                    )
                    patterns.extend(business_patterns)
                
                # Check for risk patterns
                if entity_type in ["risk", "threat", "vulnerability"]:
                    risk_patterns = await self._find_risk_patterns(
                        entity, language
                    )
                    patterns.extend(risk_patterns)
                
                # Check for opportunity patterns
                if entity_type in ["opportunity", "potential", "prospect"]:
                    opportunity_patterns = await self._find_opportunity_patterns(
                        entity, language
                    )
                    patterns.extend(opportunity_patterns)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing historical patterns: {e}")
            return []
    
    async def _find_business_patterns(
        self, 
        entity: Dict[str, Any], 
        language: str
    ) -> List[Dict[str, Any]]:
        """Find business-related patterns for an entity."""
        try:
            # Query for business patterns
            result = await call_unified_mcp_tool(
                "analyze_decision_patterns",
                entity_name=entity.get("name", ""),
                pattern_type="business_patterns",
                language=language,
                time_window=self.config.context_time_window
            )
            
            if result and "patterns" in result:
                return result["patterns"]
            return []
            
        except Exception as e:
            logger.error(f"Error finding business patterns: {e}")
            return []
    
    async def _find_risk_patterns(
        self, 
        entity: Dict[str, Any], 
        language: str
    ) -> List[Dict[str, Any]]:
        """Find risk-related patterns for an entity."""
        try:
            # Query for risk patterns
            result = await call_unified_mcp_tool(
                "analyze_decision_patterns",
                entity_name=entity.get("name", ""),
                pattern_type="risk_patterns",
                language=language,
                time_window=self.config.context_time_window
            )
            
            if result and "patterns" in result:
                return result["patterns"]
            return []
            
        except Exception as e:
            logger.error(f"Error finding risk patterns: {e}")
            return []
    
    async def _find_opportunity_patterns(
        self, 
        entity: Dict[str, Any], 
        language: str
    ) -> List[Dict[str, Any]]:
        """Find opportunity-related patterns for an entity."""
        try:
            # Query for opportunity patterns
            result = await call_unified_mcp_tool(
                "analyze_decision_patterns",
                entity_name=entity.get("name", ""),
                pattern_type="opportunity_patterns",
                language=language,
                time_window=self.config.context_time_window
            )
            
            if result and "patterns" in result:
                return result["patterns"]
            return []
            
        except Exception as e:
            logger.error(f"Error finding opportunity patterns: {e}")
            return []
    
    def _categorize_entities(
        self, 
        entities: List[Dict[str, Any]]
    ) -> Dict[str, List[EntityContext]]:
        """Categorize entities by their type and role."""
        try:
            categorized = {
                "business": [],
                "market": [],
                "risk": [],
                "opportunity": [],
                "constraint": [],
                "goal": []
            }
            
            for entity in entities:
                entity_type = entity.get("type", "").lower()
                entity_context = EntityContext(
                    entity_id=entity.get("id", ""),
                    entity_name=entity.get("name", ""),
                    entity_type=entity_type,
                    confidence=entity.get("confidence", 0.0),
                    relationships=entity.get("relationships", []),
                    attributes=entity.get("attributes", {}),
                    language=entity.get("language", "en"),
                    created_at=datetime.now()
                )
                
                # Categorize based on entity type
                if entity_type in ["organization", "company", "business"]:
                    categorized["business"].append(entity_context)
                elif entity_type in ["market", "industry", "sector"]:
                    categorized["market"].append(entity_context)
                elif entity_type in ["risk", "threat", "vulnerability"]:
                    categorized["risk"].append(entity_context)
                elif entity_type in ["opportunity", "potential", "prospect"]:
                    categorized["opportunity"].append(entity_context)
                elif entity_type in ["constraint", "limitation", "barrier"]:
                    categorized["constraint"].append(entity_context)
                elif entity_type in ["goal", "objective", "target"]:
                    categorized["goal"].append(entity_context)
            
            return categorized
            
        except Exception as e:
            logger.error(f"Error categorizing entities: {e}")
            return {key: [] for key in ["business", "market", "risk", "opportunity", "constraint", "goal"]}
    
    def _calculate_context_confidence(
        self,
        entities: List[Dict[str, Any]],
        related_entities: List[Dict[str, Any]],
        historical_patterns: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for the extracted context."""
        try:
            # Base confidence from entity extraction
            entity_confidence = sum(
                entity.get("confidence", 0.0) for entity in entities
            ) / max(len(entities), 1)
            
            # Relationship confidence
            relationship_confidence = min(
                len(related_entities) / self.config.max_entities_per_query, 1.0
            )
            
            # Pattern confidence
            pattern_confidence = min(
                len(historical_patterns) / 10, 1.0  # Normalize to 0-1
            )
            
            # Weighted average
            confidence = (
                entity_confidence * 0.5 +
                relationship_confidence * 0.3 +
                pattern_confidence * 0.2
            )
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating context confidence: {e}")
            return 0.0
    
    async def get_entity_insights(
        self, 
        entity_name: str, 
        language: str = "en"
    ) -> Dict[str, Any]:
        """Get detailed insights about a specific entity."""
        try:
            # Query knowledge graph for entity insights
            result = await call_unified_mcp_tool(
                "get_entity_context",
                entity_name=entity_name,
                language=language,
                include_relationships=True,
                include_attributes=True
            )
            
            if result:
                return result
            else:
                return {"entity_name": entity_name, "insights": []}
                
        except Exception as e:
            logger.error(f"Error getting entity insights: {e}")
            return {"entity_name": entity_name, "insights": [], "error": str(e)}
    
    async def find_similar_decisions(
        self, 
        context: DecisionContext, 
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        """Find similar historical decisions based on context."""
        try:
            # Extract key entities for similarity search
            key_entities = []
            for entity_list in [
                context.business_entities,
                context.market_entities,
                context.risk_entities,
                context.opportunity_entities
            ]:
                key_entities.extend([e.entity_name for e in entity_list])
            
            if not key_entities:
                return []
            
            # Search for similar decisions
            result = await call_unified_mcp_tool(
                "query_knowledge_graph",
                query=" ".join(key_entities[:5]),  # Use top 5 entities
                entity_types=["DECISION", "RECOMMENDATION", "OUTCOME"],
                max_results=10,
                language=language
            )
            
            if result and "related_entities" in result:
                return result["related_entities"]
            return []
            
        except Exception as e:
            logger.error(f"Error finding similar decisions: {e}")
            return []
    
    def clear_cache(self):
        """Clear the context cache."""
        self.cache.clear()
        logger.info("Decision context cache cleared")
