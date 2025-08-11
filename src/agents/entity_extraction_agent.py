"""
Entity Extraction Agent for extracting entities from text content.
Extracted from the knowledge graph agent to provide focused entity extraction capabilities.
"""

import asyncio
import json
import re
from typing import Dict, List, Optional
import logging

from src.agents.base_agent import StrandsBaseAgent
from src.core.strands_mock import tool
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

# Configure logger
logger = logging.getLogger(__name__)


class EntityExtractionAgent(StrandsBaseAgent):
    """Agent for extracting entities from text content."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
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

        # Processing settings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Entity categories
        self.entity_categories = {
            "person": ["person", "individual", "human", "name"],
            "organization": ["organization", "company", "corporation", "institution", "agency"],
            "location": ["location", "place", "city", "country", "region", "address"],
            "event": ["event", "conference", "meeting", "ceremony", "festival"],
            "product": ["product", "item", "goods", "service", "software"],
            "technology": ["technology", "software", "platform", "system", "tool"],
            "concept": ["concept", "idea", "theory", "principle", "methodology"],
            "date": ["date", "time", "period", "era", "year"],
            "quantity": ["quantity", "amount", "number", "measure", "value"],
            "other": ["other", "miscellaneous", "unknown"]
        }

        # Agent metadata
        self.metadata.update({
            "agent_type": "entity_extraction",
            "model": self.model_name,
            "capabilities": [
                "entity_extraction",
                "entity_categorization",
                "chunk_based_processing",
                "enhanced_entity_detection"
            ],
            "supported_data_types": [
                DataType.TEXT,
                DataType.AUDIO,
                DataType.VIDEO,
                DataType.WEBPAGE,
                DataType.PDF,
                DataType.SOCIAL_MEDIA
            ],
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "entity_categories": list(self.entity_categories.keys())
        })

        logger.info(f"Entity Extraction Agent {self.agent_id} initialized with model {self.model_name}")

    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            self.extract_entities,
            self.extract_entities_enhanced,
            self.categorize_entities,
            self.extract_entities_from_chunks,
            self.get_entity_statistics
        ]

    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        return request.data_type in self.metadata["supported_data_types"]

    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process the analysis request."""
        context = ErrorContext(
            agent_id=self.agent_id,
            request_id=request.request_id,
            operation="entity_extraction"
        )

        try:
            return await self._extract_entities_from_request(request, context)
        except Exception as e:
            return self.error_handling_service.handle_error(
                e, context, "Entity extraction failed"
            )

    async def _extract_entities_from_request(
        self,
        request: AnalysisRequest,
        context: ErrorContext
    ) -> AnalysisResult:
        """Extract entities from the analysis request."""
        try:
            # Extract text content from request
            text_content = await self._extract_text_content(request)

            # Extract entities using enhanced method
            entities_result = await self.extract_entities_enhanced(text_content)

            # Create sentiment result from entities
            sentiment = self._create_sentiment_from_entities(entities_result["entities"])

            return AnalysisResult(
                request_id=request.request_id,
                agent_id=self.agent_id,
                status=ProcessingStatus.COMPLETED,
                data_type=request.data_type,
                content=text_content,
                entities=entities_result["entities"],
                sentiment=sentiment,
                metadata={
                    "entity_count": len(entities_result["entities"]),
                    "categories_found": entities_result["categories_found"],
                    "processing_method": "enhanced_entity_extraction"
                }
            )

        except Exception as e:
            return self.error_handling_service.handle_error(e, context, "Entity extraction failed")

    async def _extract_text_content(self, request: AnalysisRequest) -> str:
        """Extract text content from the request."""
        return self.processing_service.extract_text_content(request)

    @tool("extract_entities", "Extract entities from text using basic extraction")
    async def extract_entities(self, text: str) -> dict:
        """Extract entities from text using basic extraction."""
        try:
            prompt = self._create_entity_extraction_prompt(text)
            response = await self._call_model(prompt)
            entities = self._parse_entities_from_response(response)

            return {
                "entities": entities,
                "count": len(entities),
                "categories_found": list(set(entity.get("category", "unknown") for entity in entities))
            }

        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {"entities": [], "count": 0, "categories_found": [], "error": str(e)}

    @tool("extract_entities_enhanced", "Extract entities from text using enhanced extraction with categorization")
    async def extract_entities_enhanced(self, text: str) -> dict:
        """Extract entities from text using enhanced extraction with categorization."""
        try:
            # First extract basic entities
            basic_result = await self.extract_entities(text)
            entities = basic_result["entities"]

            # Categorize entities
            categorized_entities = self._categorize_entities(entities)

            # Merge similar entities
            merged_entities = self._merge_similar_entities(categorized_entities)

            # Add context and relationships
            for entity in merged_entities:
                entity["context"] = self._extract_entity_context(entity, text)
                entity["relationships"] = self._find_entity_relationships(entity, merged_entities)
                entity["confidence"] = self._calculate_entity_confidence(entity)

            return {
                "entities": merged_entities,
                "count": len(merged_entities),
                "categories_found": list(set(entity.get("category", "unknown") for entity in merged_entities)),
                "statistics": self._count_entities_by_category(merged_entities)
            }

        except Exception as e:
            logger.error(f"Error in enhanced entity extraction: {e}")
            return {"entities": [], "count": 0, "categories_found": [], "error": str(e)}

    @tool("categorize_entities", "Categorize a list of entities")
    async def categorize_entities(self, entities: List[Dict]) -> dict:
        """Categorize a list of entities."""
        try:
            categorized = self._categorize_entities(entities)
            return {
                "entities": categorized,
                "categories": list(set(entity.get("category", "unknown") for entity in categorized)),
                "statistics": self._count_entities_by_category(categorized)
            }

        except Exception as e:
            logger.error(f"Error categorizing entities: {e}")
            return {"entities": [], "categories": [], "statistics": {}, "error": str(e)}

    @tool("extract_entities_from_chunks", "Extract entities from multiple text chunks")
    async def extract_entities_from_chunks(self, chunks: List[str]) -> dict:
        """Extract entities from multiple text chunks."""
        try:
            all_entities = []
            chunk_results = []

            for i, chunk in enumerate(chunks):
                chunk_result = await self.extract_entities_enhanced(chunk)
                all_entities.extend(chunk_result["entities"])
                chunk_results.append({
                    "chunk_index": i,
                    "entities": chunk_result["entities"],
                    "count": chunk_result["count"]
                })

            # Merge entities across chunks
            merged_entities = self._merge_similar_entities(all_entities)

            return {
                "entities": merged_entities,
                "total_count": len(merged_entities),
                "chunk_results": chunk_results,
                "categories_found": list(set(entity.get("category", "unknown") for entity in merged_entities))
            }

        except Exception as e:
            logger.error(f"Error extracting entities from chunks: {e}")
            return {"entities": [], "total_count": 0, "chunk_results": [], "error": str(e)}

    @tool("get_entity_statistics", "Get statistics about entity extraction capabilities")
    async def get_entity_statistics(self) -> dict:
        """Get statistics about entity extraction capabilities."""
        return {
            "entity_categories": self.entity_categories,
            "supported_data_types": self.metadata["supported_data_types"],
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "model": self.model_name
        }

    def _create_entity_extraction_prompt(self, text: str) -> str:
        """Create a prompt for entity extraction."""
        return f"""
        Extract entities from the following text. For each entity, provide:
        - name: The entity name
        - type: The type of entity (person, organization, location, etc.)
        - importance: high, medium, or low
        - description: Brief description of the entity

        Text: {text}

        Return the entities in JSON format:
        {{
            "entities": [
                {{
                    "name": "entity_name",
                    "type": "entity_type",
                    "importance": "high|medium|low",
                    "description": "brief_description"
                }}
            ]
        }}
        """

    def _parse_entities_from_response(self, response: str) -> List[Dict]:
        """Parse entities from model response."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("entities", [])
            return []
        except Exception as e:
            logger.error(f"Error parsing entities from response: {e}")
            return []

    def _merge_similar_entities(self, entities: List[Dict]) -> List[Dict]:
        """Merge similar entities based on name similarity."""
        if not entities:
            return []

        merged = []
        processed = set()

        for i, entity1 in enumerate(entities):
            if i in processed:
                continue

            similar_entities = [entity1]
            processed.add(i)

            for j, entity2 in enumerate(entities[i+1:], i+1):
                if j in processed:
                    continue

                # Check if entities are similar (same name or very similar)
                if self._are_entities_similar(entity1, entity2):
                    similar_entities.append(entity2)
                    processed.add(j)

            # Merge similar entities
            merged_entity = self._merge_entity_group(similar_entities)
            merged.append(merged_entity)

        return merged

    def _are_entities_similar(self, entity1: Dict, entity2: Dict) -> bool:
        """Check if two entities are similar."""
        name1 = entity1.get("name", "").lower()
        name2 = entity2.get("name", "").lower()

        # Exact match
        if name1 == name2:
            return True

        # Check for partial matches
        if name1 in name2 or name2 in name1:
            return True

        # Check for acronyms
        if len(name1) <= 3 and name1.upper() == name1:
            if name1.lower() in name2.lower():
                return True

        if len(name2) <= 3 and name2.upper() == name2:
            if name2.lower() in name1.lower():
                return True

        return False

    def _merge_entity_group(self, entities: List[Dict]) -> Dict:
        """Merge a group of similar entities into one."""
        if not entities:
            return {}

        # Use the first entity as base
        merged = entities[0].copy()

        # Merge importance levels
        importance_levels = [entity.get("importance", "low") for entity in entities]
        merged["importance"] = self._merge_importance(importance_levels[0], importance_levels[-1])

        # Merge descriptions
        descriptions = [entity.get("description", "") for entity in entities if entity.get("description")]
        if descriptions:
            merged["description"] = " | ".join(descriptions)

        # Merge types if different
        types = list(set(entity.get("type", "") for entity in entities if entity.get("type")))
        if len(types) > 1:
            merged["type"] = " | ".join(types)

        return merged

    def _merge_importance(self, importance1: str, importance2: str) -> str:
        """Merge two importance levels."""
        importance_map = {"low": 1, "medium": 2, "high": 3}
        level1 = importance_map.get(importance1.lower(), 1)
        level2 = importance_map.get(importance2.lower(), 1)
        max_level = max(level1, level2)
        return {1: "low", 2: "medium", 3: "high"}[max_level]

    def _categorize_entities(self, entities: List[Dict]) -> List[Dict]:
        """Categorize entities based on their type and description."""
        for entity in entities:
            entity["category"] = self._determine_entity_category(entity)
        return entities

    def _determine_entity_category(self, entity: Dict) -> str:
        """Determine the category of an entity."""
        entity_type = entity.get("type", "").lower()
        entity_name = entity.get("name", "").lower()
        description = entity.get("description", "").lower()

        # Check each category
        for category, keywords in self.entity_categories.items():
            for keyword in keywords:
                if (keyword in entity_type or keyword in entity_name or
                    keyword in description):
                    return category

        return "other"

    def _count_entities_by_category(self, entities: List[Dict]) -> Dict[str, int]:
        """Count entities by category."""
        counts = {}
        for entity in entities:
            category = entity.get("category", "unknown")
            counts[category] = counts.get(category, 0) + 1
        return counts

    def _calculate_entity_confidence(self, entity: Dict) -> float:
        """Calculate confidence score for an entity."""
        confidence = 0.5  # Base confidence

        # Boost confidence based on entity properties
        if entity.get("description"):
            confidence += 0.2

        if entity.get("context"):
            confidence += 0.1

        if entity.get("relationships"):
            confidence += 0.1

        importance = entity.get("importance", "low").lower()
        if importance == "high":
            confidence += 0.1
        elif importance == "medium":
            confidence += 0.05

        return min(confidence, 1.0)

    def _extract_entity_context(self, entity: Dict, text: str) -> str:
        """Extract context around an entity in the text."""
        entity_name = entity.get("name", "")
        if not entity_name:
            return ""

        # Find entity position in text
        pos = text.lower().find(entity_name.lower())
        if pos == -1:
            return ""

        # Extract context (50 characters before and after)
        start = max(0, pos - 50)
        end = min(len(text), pos + len(entity_name) + 50)
        context = text[start:end]

        return context.strip()

    def _find_entity_relationships(self, entity: Dict, all_entities: List[Dict]) -> List[str]:
        """Find relationships between entities."""
        relationships = []
        entity_name = entity.get("name", "").lower()

        for other_entity in all_entities:
            if other_entity == entity:
                continue

            other_name = other_entity.get("name", "").lower()
            if other_name in entity_name or entity_name in other_name:
                relationships.append(f"similar_to:{other_entity['name']}")

        return relationships

    def _create_sentiment_from_entities(self, entities: List[Dict]) -> SentimentResult:
        """Create a sentiment result from extracted entities."""
        if not entities:
            return SentimentResult(
                overall_sentiment="neutral",
                confidence=0.5,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0
            )

        # Simple sentiment based on entity importance
        high_importance_count = sum(1 for e in entities if e.get("importance") == "high")
        total_count = len(entities)

        if total_count == 0:
            return SentimentResult(
                overall_sentiment="neutral",
                confidence=0.5,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0
            )

        # Calculate sentiment based on entity importance
        importance_ratio = high_importance_count / total_count

        if importance_ratio > 0.7:
            sentiment = "positive"
            positive_score = 0.8
            negative_score = 0.1
            neutral_score = 0.1
        elif importance_ratio > 0.3:
            sentiment = "neutral"
            positive_score = 0.3
            negative_score = 0.2
            neutral_score = 0.5
        else:
            sentiment = "negative"
            positive_score = 0.1
            negative_score = 0.7
            neutral_score = 0.2

        return SentimentResult(
            overall_sentiment=sentiment,
            confidence=0.6,
            positive_score=positive_score,
            negative_score=negative_score,
            neutral_score=neutral_score
        )
