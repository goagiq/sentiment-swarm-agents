"""
Graph Analysis Agent for analyzing graph structures and properties.
Extracted from the knowledge graph agent to provide focused graph analysis capabilities.
"""

import asyncio
import json
import networkx as nx
import matplotlib.pyplot as plt
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

# Configure logger
logger = logging.getLogger(__name__)


class GraphAnalysisAgent(StrandsBaseAgent):
    """Agent for analyzing graph structures and properties."""
    
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
        
        # Analysis capabilities
        self.analysis_types = {
            "centrality": ["degree", "betweenness", "closeness", "eigenvector"],
            "community": ["modularity", "clustering", "connected_components"],
            "structural": ["density", "diameter", "average_path_length", "clustering_coefficient"],
            "topological": ["degree_distribution", "assortativity", "reciprocity"]
        }
        
        # Agent metadata
        self.metadata.update({
            "agent_type": "graph_analysis",
            "model": self.model_name,
            "capabilities": [
                "centrality_analysis",
                "community_detection",
                "structural_analysis",
                "topological_analysis",
                "graph_metrics",
                "path_analysis"
            ],
            "supported_data_types": [
                DataType.TEXT,
                DataType.AUDIO,
                DataType.VIDEO,
                DataType.WEBPAGE,
                DataType.PDF,
                DataType.SOCIAL_MEDIA
            ],
            "analysis_types": list(self.analysis_types.keys())
        })
        
        logger.info(
            f"Graph Analysis Agent {self.agent_id} initialized with model "
            f"{self.model_name}"
        )
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            self.analyze_graph_communities,
            self.analyze_centrality,
            self.analyze_graph_structure,
            self.find_entity_paths,
            self.get_graph_metrics,
            self.detect_communities,
            self.analyze_connectivity
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
            operation="graph_analysis",
            request_id=request.id
        )
        
        return await self.error_handling_service.safe_execute_async(
            self._analyze_graph_from_request,
            request,
            context,
            AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(label="neutral", confidence=0.0),
                status=ProcessingStatus.FAILED
            )
        )
    
    async def _analyze_graph_from_request(
        self, 
        request: AnalysisRequest, 
        context: ErrorContext
    ) -> AnalysisResult:
        """Analyze graph from the analysis request."""
        # Extract graph data from request
        graph_data = await self._extract_graph_data(request)
        
        if not graph_data:
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(label="neutral", confidence=0.0),
                status=ProcessingStatus.FAILED,
                metadata={"error": "No graph data found in request"}
            )
        
        # Build NetworkX graph
        graph = await self._build_graph(graph_data)
        
        # Perform comprehensive analysis
        analysis_results = await self._perform_comprehensive_analysis(graph)
        
        # Create sentiment from analysis
        sentiment = await self._create_sentiment_from_analysis(analysis_results)
        
        return AnalysisResult(
            request_id=request.id,
            data_type=request.data_type,
            sentiment=sentiment,
            status=ProcessingStatus.COMPLETED,
            metadata=analysis_results
        )
    
    async def _extract_graph_data(self, request: AnalysisRequest) -> Dict:
        """Extract graph data from the request."""
        content = self.processing_service.extract_text_content(request.content)
        
        # Try to parse as JSON first
        try:
            if isinstance(content, str):
                return json.loads(content)
            elif isinstance(content, dict):
                return content
        except (json.JSONDecodeError, TypeError):
            pass
        
        # If not JSON, assume it's a text description
        return {"description": content}
    
    async def _build_graph(self, graph_data: Dict) -> nx.Graph:
        """Build a NetworkX graph from graph data."""
        graph = nx.Graph()
        
        # Add nodes and edges from graph data
        if "nodes" in graph_data:
            for node in graph_data["nodes"]:
                node_id = node.get("id", str(node))
                graph.add_node(node_id, **node)
        
        if "edges" in graph_data:
            for edge in graph_data["edges"]:
                source = edge.get("source")
                target = edge.get("target")
                if source and target:
                    graph.add_edge(source, target, **edge)
        
        return graph
    
    async def _perform_comprehensive_analysis(self, graph: nx.Graph) -> Dict:
        """Perform comprehensive graph analysis."""
        if graph.number_of_nodes() == 0:
            return {"error": "Empty graph"}
        
        analysis = {
            "basic_stats": self._get_basic_stats(graph),
            "centrality": self._analyze_centrality(graph),
            "communities": self._detect_communities(graph),
            "connectivity": self._analyze_connectivity(graph),
            "structural": self._analyze_structure(graph)
        }
        
        return analysis
    
    def _get_basic_stats(self, graph: nx.Graph) -> Dict:
        """Get basic graph statistics."""
        return {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_connected": nx.is_connected(graph),
            "connected_components": nx.number_connected_components(graph),
            "average_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
        }
    
    def _analyze_centrality(self, graph: nx.Graph) -> Dict:
        """Analyze centrality measures."""
        if graph.number_of_nodes() == 0:
            return {}
        
        centrality_measures = {}
        
        # Degree centrality
        degree_centrality = nx.degree_centrality(graph)
        centrality_measures["degree"] = {
            "max": max(degree_centrality.values()) if degree_centrality else 0,
            "min": min(degree_centrality.values()) if degree_centrality else 0,
            "average": sum(degree_centrality.values()) / len(degree_centrality) if degree_centrality else 0,
            "top_nodes": sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        # Betweenness centrality (only for connected graphs)
        if nx.is_connected(graph):
            betweenness_centrality = nx.betweenness_centrality(graph)
            centrality_measures["betweenness"] = {
                "max": max(betweenness_centrality.values()) if betweenness_centrality else 0,
                "min": min(betweenness_centrality.values()) if betweenness_centrality else 0,
                "average": sum(betweenness_centrality.values()) / len(betweenness_centrality) if betweenness_centrality else 0,
                "top_nodes": sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        
        # Closeness centrality (only for connected graphs)
        if nx.is_connected(graph):
            closeness_centrality = nx.closeness_centrality(graph)
            centrality_measures["closeness"] = {
                "max": max(closeness_centrality.values()) if closeness_centrality else 0,
                "min": min(closeness_centrality.values()) if closeness_centrality else 0,
                "average": sum(closeness_centrality.values()) / len(closeness_centrality) if closeness_centrality else 0,
                "top_nodes": sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        
        return centrality_measures
    
    def _detect_communities(self, graph: nx.Graph) -> Dict:
        """Detect communities in the graph."""
        if graph.number_of_nodes() == 0:
            return {}
        
        communities = {}
        
        # Connected components
        connected_components = list(nx.connected_components(graph))
        communities["connected_components"] = {
            "count": len(connected_components),
            "sizes": [len(comp) for comp in connected_components],
            "largest_component_size": max(len(comp) for comp in connected_components) if connected_components else 0
        }
        
        # Clustering coefficient
        clustering_coefficient = nx.average_clustering(graph)
        communities["clustering"] = {
            "average_clustering_coefficient": clustering_coefficient,
            "global_clustering_coefficient": nx.transitivity(graph)
        }
        
        # Try community detection algorithms
        try:
            # Louvain community detection
            from community import community_louvain
            partition = community_louvain.best_partition(graph)
            communities["louvain"] = {
                "communities": len(set(partition.values())),
                "modularity": nx.community.modularity(graph, [set(n for n in graph.nodes() if partition[n] == com) for com in set(partition.values())])
            }
        except ImportError:
            communities["louvain"] = {"error": "community module not available"}
        
        return communities
    
    def _analyze_connectivity(self, graph: nx.Graph) -> Dict:
        """Analyze graph connectivity."""
        if graph.number_of_nodes() == 0:
            return {}
        
        connectivity = {}
        
        # Basic connectivity
        connectivity["is_connected"] = nx.is_connected(graph)
        connectivity["connected_components"] = nx.number_connected_components(graph)
        
        # Path analysis
        if nx.is_connected(graph):
            connectivity["diameter"] = nx.diameter(graph)
            connectivity["average_path_length"] = nx.average_shortest_path_length(graph)
        else:
            connectivity["diameter"] = "infinite (disconnected)"
            connectivity["average_path_length"] = "infinite (disconnected)"
        
        # Bridge analysis
        bridges = list(nx.bridges(graph))
        connectivity["bridges"] = len(bridges)
        
        return connectivity
    
    def _analyze_structure(self, graph: nx.Graph) -> Dict:
        """Analyze graph structure."""
        if graph.number_of_nodes() == 0:
            return {}
        
        structure = {}
        
        # Degree distribution
        degrees = [d for n, d in graph.degree()]
        structure["degree_distribution"] = {
            "min_degree": min(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
            "average_degree": sum(degrees) / len(degrees) if degrees else 0
        }
        
        # Graph density
        structure["density"] = nx.density(graph)
        
        # Assortativity
        try:
            structure["assortativity"] = nx.degree_assortativity_coefficient(graph)
        except:
            structure["assortativity"] = "cannot compute"
        
        return structure
    
    async def _create_sentiment_from_analysis(self, analysis_results: Dict) -> SentimentResult:
        """Create sentiment from graph analysis results."""
        if "error" in analysis_results:
            return SentimentResult(label="neutral", confidence=0.5)
        
        # Simple sentiment based on graph properties
        basic_stats = analysis_results.get("basic_stats", {})
        
        if basic_stats.get("is_connected", False):
            # Connected graphs are generally positive
            return SentimentResult(label="positive", confidence=0.7)
        else:
            # Disconnected graphs might indicate fragmentation
            return SentimentResult(label="neutral", confidence=0.5)
    
    @tool
    async def analyze_graph_communities(self) -> dict:
        """Analyze communities in the graph."""
        try:
            # This would typically work with a stored graph
            # For now, return a placeholder
            return {
                "success": True,
                "message": "Graph community analysis requires a graph to be loaded",
                "capabilities": [
                    "connected_components",
                    "clustering_coefficient",
                    "louvain_communities",
                    "modularity_analysis"
                ]
            }
        except Exception as e:
            logger.error(f"Error analyzing graph communities: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    async def analyze_centrality(self, graph_data: Dict) -> dict:
        """Analyze centrality measures of a graph."""
        try:
            graph = await self._build_graph(graph_data)
            centrality_analysis = self._analyze_centrality(graph)
            
            return {
                "success": True,
                "centrality_analysis": centrality_analysis,
                "graph_stats": {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges()
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing centrality: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    async def analyze_graph_structure(self, graph_data: Dict) -> dict:
        """Analyze the structure of a graph."""
        try:
            graph = await self._build_graph(graph_data)
            structure_analysis = self._analyze_structure(graph)
            connectivity_analysis = self._analyze_connectivity(graph)
            
            return {
                "success": True,
                "structure_analysis": structure_analysis,
                "connectivity_analysis": connectivity_analysis,
                "graph_stats": {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges()
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing graph structure: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    async def find_entity_paths(self, source: str, target: str, graph_data: Dict) -> dict:
        """Find paths between two entities in a graph."""
        try:
            graph = await self._build_graph(graph_data)
            
            if source not in graph or target not in graph:
                return {
                    "success": False,
                    "error": f"Source '{source}' or target '{target}' not found in graph"
                }
            
            # Find all simple paths
            try:
                paths = list(nx.all_simple_paths(graph, source, target, cutoff=5))
            except nx.NetworkXNoPath:
                paths = []
            
            # Find shortest path
            try:
                shortest_path = nx.shortest_path(graph, source, target)
                shortest_length = len(shortest_path) - 1
            except nx.NetworkXNoPath:
                shortest_path = None
                shortest_length = None
            
            return {
                "success": True,
                "source": source,
                "target": target,
                "total_paths": len(paths),
                "shortest_path": shortest_path,
                "shortest_length": shortest_length,
                "all_paths": paths[:10]  # Limit to first 10 paths
            }
        except Exception as e:
            logger.error(f"Error finding entity paths: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    async def get_graph_metrics(self, graph_data: Dict) -> dict:
        """Get comprehensive graph metrics."""
        try:
            graph = await self._build_graph(graph_data)
            
            if graph.number_of_nodes() == 0:
                return {
                    "success": False,
                    "error": "Empty graph"
                }
            
            # Perform all analyses
            analysis = await self._perform_comprehensive_analysis(graph)
            
            return {
                "success": True,
                "metrics": analysis,
                "summary": {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges(),
                    "density": nx.density(graph),
                    "is_connected": nx.is_connected(graph),
                    "connected_components": nx.number_connected_components(graph)
                }
            }
        except Exception as e:
            logger.error(f"Error getting graph metrics: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    async def detect_communities(self, graph_data: Dict) -> dict:
        """Detect communities in a graph."""
        try:
            graph = await self._build_graph(graph_data)
            communities = self._detect_communities(graph)
            
            return {
                "success": True,
                "communities": communities,
                "graph_stats": {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges()
                }
            }
        except Exception as e:
            logger.error(f"Error detecting communities: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    async def analyze_connectivity(self, graph_data: Dict) -> dict:
        """Analyze graph connectivity."""
        try:
            graph = await self._build_graph(graph_data)
            connectivity = self._analyze_connectivity(graph)
            
            return {
                "success": True,
                "connectivity": connectivity,
                "graph_stats": {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges()
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing connectivity: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
