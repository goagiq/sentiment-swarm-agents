"""
Knowledge Graph Coordinator for orchestrating specialized knowledge graph agents.
Replaces the monolithic Knowledge Graph Agent with a coordinator pattern.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

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
from src.agents.entity_extraction_agent import EntityExtractionAgent
from src.agents.relationship_mapping_agent import RelationshipMappingAgent
from src.agents.graph_analysis_agent import GraphAnalysisAgent
from src.agents.graph_visualization_agent import GraphVisualizationAgent

# Configure logger
logger = logging.getLogger(__name__)


class KnowledgeGraphCoordinator(StrandsBaseAgent):
    """Coordinator for specialized knowledge graph agents."""
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        graph_storage_path: Optional[str] = None,
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
        
        # Set graph storage path
        self.graph_storage_path = Path(graph_storage_path or "./data/knowledge_graphs")
        self.graph_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize specialized agents
        self.entity_agent = EntityExtractionAgent(model_name=self.model_name)
        self.relationship_agent = RelationshipMappingAgent(model_name=self.model_name)
        self.analysis_agent = GraphAnalysisAgent(model_name=self.model_name)
        self.visualization_agent = GraphVisualizationAgent(
            model_name=self.model_name,
            output_dir=str(self.graph_storage_path / "visualizations")
        )
        
        # Agent metadata
        self.metadata.update({
            "agent_type": "knowledge_graph_coordinator",
            "model": self.model_name,
            "capabilities": [
                "entity_extraction",
                "relationship_mapping",
                "graph_analysis",
                "graph_visualization",
                "coordinated_processing",
                "workflow_orchestration"
            ],
            "supported_data_types": [
                DataType.TEXT,
                DataType.AUDIO,
                DataType.VIDEO,
                DataType.WEBPAGE,
                DataType.PDF,
                DataType.SOCIAL_MEDIA
            ],
            "specialized_agents": [
                "EntityExtractionAgent",
                "RelationshipMappingAgent", 
                "GraphAnalysisAgent",
                "GraphVisualizationAgent"
            ],
            "graph_storage_path": str(self.graph_storage_path)
        })
        
        logger.info(
            f"Knowledge Graph Coordinator {self.agent_id} initialized with model "
            f"{self.model_name}"
        )
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            self.extract_entities,
            self.map_relationships,
            self.analyze_graph,
            self.visualize_graph,
            self.generate_comprehensive_report,
            self.query_knowledge_graph,
            self.get_coordinator_status
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
        """Process the analysis request using coordinated agents."""
        context = ErrorContext(
            agent_id=self.agent_id,
            operation="knowledge_graph_coordination",
            request_id=request.id
        )
        
        return await self.error_handling_service.safe_execute_async(
            self._coordinate_processing,
            request,
            context,
            AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(label="neutral", confidence=0.0),
                status=ProcessingStatus.FAILED
            )
        )
    
    async def _coordinate_processing(
        self, 
        request: AnalysisRequest, 
        context: ErrorContext
    ) -> AnalysisResult:
        """Coordinate processing across specialized agents."""
        # Extract text content
        text_content = await self._extract_text_content(request)
        
        # Step 1: Extract entities
        entity_result = await self.entity_agent.process(request)
        entities = entity_result.metadata.get("entities", [])
        
        # Step 2: Map relationships
        relationship_request = AnalysisRequest(
            data_type=request.data_type,
            content=text_content,
            language=request.language
        )
        relationship_result = await self.relationship_agent.process(relationship_request)
        relationships = relationship_result.metadata.get("relationships", [])
        
        # Step 3: Build graph data
        graph_data = await self._build_graph_data(entities, relationships)
        
        # Step 4: Analyze graph
        analysis_request = AnalysisRequest(
            data_type=request.data_type,
            content=json.dumps(graph_data),
            language=request.language
        )
        analysis_result = await self.analysis_agent.process(analysis_request)
        
        # Step 5: Generate visualizations
        visualization_request = AnalysisRequest(
            data_type=request.data_type,
            content=json.dumps(graph_data),
            language=request.language
        )
        visualization_result = await self.visualization_agent.process(visualization_request)
        
        # Combine results
        combined_metadata = {
            "entities": entities,
            "relationships": relationships,
            "graph_data": graph_data,
            "analysis": analysis_result.metadata,
            "visualizations": visualization_result.metadata.get("visualizations", {}),
            "processing_steps": [
                "entity_extraction",
                "relationship_mapping", 
                "graph_analysis",
                "graph_visualization"
            ],
            "agents_used": [
                self.entity_agent.agent_id,
                self.relationship_agent.agent_id,
                self.analysis_agent.agent_id,
                self.visualization_agent.agent_id
            ]
        }
        
        # Create sentiment from combined results
        sentiment = await self._create_combined_sentiment([
            entity_result.sentiment,
            relationship_result.sentiment,
            analysis_result.sentiment,
            visualization_result.sentiment
        ])
        
        return AnalysisResult(
            request_id=request.id,
            data_type=request.data_type,
            sentiment=sentiment,
            status=ProcessingStatus.COMPLETED,
            metadata=combined_metadata
        )
    
    async def _extract_text_content(self, request: AnalysisRequest) -> str:
        """Extract text content from the request."""
        return self.processing_service.extract_text_content(request.content)
    
    async def _build_graph_data(self, entities: List[Dict], relationships: List[Dict]) -> Dict:
        """Build graph data from entities and relationships."""
        # Create nodes from entities
        nodes = []
        for entity in entities:
            nodes.append({
                "id": entity.get("name", str(entity)),
                "type": entity.get("type", "unknown"),
                "confidence": entity.get("confidence", 0.0),
                "category": entity.get("category", "other")
            })
        
        # Create edges from relationships
        edges = []
        for relationship in relationships:
            edges.append({
                "source": relationship.get("source", ""),
                "target": relationship.get("target", ""),
                "relationship_type": relationship.get("relationship_type", "related_to"),
                "confidence": relationship.get("confidence", 0.0),
                "category": relationship.get("category", "other")
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "entity_count": len(entities),
                "relationship_count": len(relationships),
                "generated_by": "KnowledgeGraphCoordinator"
            }
        }
    
    async def _create_combined_sentiment(self, sentiments: List[SentimentResult]) -> SentimentResult:
        """Create combined sentiment from multiple agent results."""
        if not sentiments:
            return SentimentResult(label="neutral", confidence=0.5)
        
        # Simple averaging approach
        positive_count = sum(1 for s in sentiments if s.label == "positive")
        negative_count = sum(1 for s in sentiments if s.label == "negative")
        neutral_count = sum(1 for s in sentiments if s.label == "neutral")
        
        total = len(sentiments)
        avg_confidence = sum(s.confidence for s in sentiments) / total
        
        if positive_count > negative_count and positive_count > neutral_count:
            return SentimentResult(label="positive", confidence=avg_confidence)
        elif negative_count > positive_count and negative_count > neutral_count:
            return SentimentResult(label="negative", confidence=avg_confidence)
        else:
            return SentimentResult(label="neutral", confidence=avg_confidence)
    
    @tool
    async def extract_entities(self, text: str) -> dict:
        """Extract entities from text using the Entity Extraction Agent."""
        try:
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=text
            )
            
            result = await self.entity_agent.process(request)
            
            return {
                "success": True,
                "entities": result.metadata.get("entities", []),
                "statistics": result.metadata.get("entity_statistics", {}),
                "agent_used": self.entity_agent.agent_id
            }
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    async def map_relationships(self, text: str, entities: List[Dict]) -> dict:
        """Map relationships between entities using the Relationship Mapping Agent."""
        try:
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=text
            )
            
            # Create a request with entities included
            request.metadata = {"entities": entities}
            
            result = await self.relationship_agent.process(request)
            
            return {
                "success": True,
                "relationships": result.metadata.get("relationships", []),
                "statistics": result.metadata.get("relationship_statistics", {}),
                "agent_used": self.relationship_agent.agent_id
            }
        except Exception as e:
            logger.error(f"Error mapping relationships: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    async def analyze_graph(self, graph_data: Dict) -> dict:
        """Analyze graph using the Graph Analysis Agent."""
        try:
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=json.dumps(graph_data)
            )
            
            result = await self.analysis_agent.process(request)
            
            return {
                "success": True,
                "analysis": result.metadata,
                "agent_used": self.analysis_agent.agent_id
            }
        except Exception as e:
            logger.error(f"Error analyzing graph: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    async def visualize_graph(self, graph_data: Dict) -> dict:
        """Visualize graph using the Graph Visualization Agent."""
        try:
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=json.dumps(graph_data)
            )
            
            result = await self.visualization_agent.process(request)
            
            return {
                "success": True,
                "visualizations": result.metadata.get("visualizations", {}),
                "agent_used": self.visualization_agent.agent_id
            }
        except Exception as e:
            logger.error(f"Error visualizing graph: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    async def generate_comprehensive_report(self, text: str) -> dict:
        """Generate a comprehensive knowledge graph report."""
        try:
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=text
            )
            
            result = await self.process(request)
            
            return {
                "success": True,
                "report": {
                    "entities": result.metadata.get("entities", []),
                    "relationships": result.metadata.get("relationships", []),
                    "analysis": result.metadata.get("analysis", {}),
                    "visualizations": result.metadata.get("visualizations", {}),
                    "processing_steps": result.metadata.get("processing_steps", []),
                    "agents_used": result.metadata.get("agents_used", [])
                },
                "sentiment": {
                    "label": result.sentiment.label,
                    "confidence": result.sentiment.confidence
                }
            }
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    async def query_knowledge_graph(self, query: str) -> dict:
        """Query the knowledge graph (placeholder for future implementation)."""
        try:
            return {
                "success": True,
                "message": "Knowledge graph querying requires persistent graph storage",
                "query": query,
                "capabilities": [
                    "entity_search",
                    "relationship_traversal",
                    "path_finding",
                    "similarity_search"
                ]
            }
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    async def get_coordinator_status(self) -> dict:
        """Get status of the coordinator and all specialized agents."""
        try:
            agent_statuses = {
                "entity_agent": {
                    "agent_id": self.entity_agent.agent_id,
                    "status": "active",
                    "capabilities": self.entity_agent.metadata.get("capabilities", [])
                },
                "relationship_agent": {
                    "agent_id": self.relationship_agent.agent_id,
                    "status": "active",
                    "capabilities": self.relationship_agent.metadata.get("capabilities", [])
                },
                "analysis_agent": {
                    "agent_id": self.analysis_agent.agent_id,
                    "status": "active",
                    "capabilities": self.analysis_agent.metadata.get("capabilities", [])
                },
                "visualization_agent": {
                    "agent_id": self.visualization_agent.agent_id,
                    "status": "active",
                    "capabilities": self.visualization_agent.metadata.get("capabilities", [])
                }
            }
            
            return {
                "success": True,
                "coordinator_id": self.agent_id,
                "model": self.model_name,
                "graph_storage_path": str(self.graph_storage_path),
                "agent_statuses": agent_statuses,
                "total_agents": len(agent_statuses),
                "capabilities": self.metadata.get("capabilities", [])
            }
        except Exception as e:
            logger.error(f"Error getting coordinator status: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
