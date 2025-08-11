"""
Graph Visualization Agent for creating visual representations of graphs.
Extracted from the knowledge graph agent to provide focused graph visualization capabilities.
"""

import asyncio
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


class GraphVisualizationAgent(StrandsBaseAgent):
    """Agent for creating visual representations of graphs."""
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        output_dir: Optional[str] = None,
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
        
        # Set output directory
        self.output_dir = Path(output_dir or "./Results/graph_visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualization styles
        self.visualization_styles = {
            "default": {
                "node_size": 1000,
                "node_color": "lightblue",
                "edge_color": "gray",
                "font_size": 8,
                "figsize": (12, 8)
            },
            "professional": {
                "node_size": 800,
                "node_color": "#2E86AB",
                "edge_color": "#A23B72",
                "font_size": 10,
                "figsize": (14, 10)
            },
            "minimal": {
                "node_size": 600,
                "node_color": "white",
                "edge_color": "black",
                "font_size": 6,
                "figsize": (10, 6)
            }
        }
        
        # Agent metadata
        self.metadata.update({
            "agent_type": "graph_visualization",
            "model": self.model_name,
            "capabilities": [
                "graph_visualization",
                "report_generation",
                "multiple_formats",
                "custom_styling",
                "interactive_plots"
            ],
            "supported_data_types": [
                DataType.TEXT,
                DataType.AUDIO,
                DataType.VIDEO,
                DataType.WEBPAGE,
                DataType.PDF,
                DataType.SOCIAL_MEDIA
            ],
            "output_formats": ["png", "svg", "html", "markdown"],
            "visualization_styles": list(self.visualization_styles.keys())
        })
        
        logger.info(
            f"Graph Visualization Agent {self.agent_id} initialized with model "
            f"{self.model_name}"
        )
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            self.generate_graph_report,
            self.visualize_graph,
            self.create_interactive_plot,
            self.generate_graph_summary,
            self.export_graph_data,
            self.get_visualization_options
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
            operation="graph_visualization",
            request_id=request.id
        )
        
        return await self.error_handling_service.safe_execute_async(
            self._visualize_graph_from_request,
            request,
            context,
            AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(label="neutral", confidence=0.0),
                status=ProcessingStatus.FAILED
            )
        )
    
    async def _visualize_graph_from_request(
        self, 
        request: AnalysisRequest, 
        context: ErrorContext
    ) -> AnalysisResult:
        """Create visualizations from the analysis request."""
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
        
        # Generate visualizations
        visualization_results = await self._generate_visualizations(graph, request.id)
        
        # Create sentiment from visualization
        sentiment = await self._create_sentiment_from_visualization(visualization_results)
        
        return AnalysisResult(
            request_id=request.id,
            data_type=request.data_type,
            sentiment=sentiment,
            status=ProcessingStatus.COMPLETED,
            metadata=visualization_results
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
    
    async def _generate_visualizations(self, graph: nx.Graph, request_id: str) -> Dict:
        """Generate various visualizations of the graph."""
        if graph.number_of_nodes() == 0:
            return {"error": "Empty graph"}
        
        results = {
            "request_id": request_id,
            "graph_stats": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges()
            },
            "visualizations": {}
        }
        
        # Generate PNG visualization
        png_path = await self._generate_png_visualization(graph, request_id)
        if png_path:
            results["visualizations"]["png"] = str(png_path)
        
        # Generate HTML visualization
        html_path = await self._generate_html_visualization(graph, request_id)
        if html_path:
            results["visualizations"]["html"] = str(html_path)
        
        # Generate markdown report
        md_path = await self._generate_markdown_report(graph, request_id)
        if md_path:
            results["visualizations"]["markdown"] = str(md_path)
        
        return results
    
    async def _generate_png_visualization(self, graph: nx.Graph, request_id: str) -> Optional[Path]:
        """Generate PNG visualization of the graph."""
        try:
            # Set up the plot
            plt.figure(figsize=(12, 8))
            
            # Choose layout
            if graph.number_of_nodes() <= 20:
                pos = nx.spring_layout(graph, k=1, iterations=50)
            else:
                pos = nx.kamada_kawai_layout(graph)
            
            # Draw the graph
            nx.draw(
                graph, pos,
                with_labels=True,
                node_color='lightblue',
                node_size=1000,
                font_size=8,
                font_weight='bold',
                edge_color='gray',
                width=1,
                alpha=0.7
            )
            
            # Add title
            plt.title(f"Graph Visualization - Request {request_id}", fontsize=14, fontweight='bold')
            
            # Save the plot
            output_path = self.output_dir / f"graph_{request_id}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Generated PNG visualization: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating PNG visualization: {str(e)}")
            return None
    
    async def _generate_html_visualization(self, graph: nx.Graph, request_id: str) -> Optional[Path]:
        """Generate HTML visualization of the graph."""
        try:
            # Create HTML content
            html_content = self._create_html_template(graph, request_id)
            
            # Save HTML file
            output_path = self.output_dir / f"graph_{request_id}.html"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Generated HTML visualization: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating HTML visualization: {str(e)}")
            return None
    
    async def _generate_markdown_report(self, graph: nx.Graph, request_id: str) -> Optional[Path]:
        """Generate markdown report of the graph."""
        try:
            # Calculate basic statistics
            stats = {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "density": nx.density(graph),
                "is_connected": nx.is_connected(graph),
                "connected_components": nx.number_connected_components(graph)
            }
            
            # Get node and edge lists
            nodes = list(graph.nodes())
            edges = list(graph.edges())
            
            # Create markdown content
            md_content = f"""# Graph Analysis Report - Request {request_id}

## Graph Statistics
- **Nodes**: {stats['nodes']}
- **Edges**: {stats['edges']}
- **Density**: {stats['density']:.4f}
- **Connected**: {stats['is_connected']}
- **Connected Components**: {stats['connected_components']}

## Nodes
{chr(10).join([f"- {node}" for node in nodes[:20]])}
{f"- ... and {len(nodes) - 20} more nodes" if len(nodes) > 20 else ""}

## Edges
{chr(10).join([f"- {edge[0]} ↔ {edge[1]}" for edge in edges[:20]])}
{f"- ... and {len(edges) - 20} more edges" if len(edges) > 20 else ""}

## Visualization Files
- PNG: `graph_{request_id}.png`
- HTML: `graph_{request_id}.html`

Generated by Graph Visualization Agent on {asyncio.get_event_loop().time()}
"""
            
            # Save markdown file
            output_path = self.output_dir / f"graph_{request_id}.md"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            logger.info(f"Generated markdown report: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating markdown report: {str(e)}")
            return None
    
    def _create_html_template(self, graph: nx.Graph, request_id: str) -> str:
        """Create HTML template for graph visualization."""
        # Convert graph to JSON for JavaScript
        graph_json = nx.node_link_data(graph)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph Visualization - Request {request_id}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        #graph-container {{
            width: 100%;
            height: 600px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Graph Visualization - Request {request_id}</h1>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{graph.number_of_nodes()}</div>
                <div class="stat-label">Nodes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{graph.number_of_edges()}</div>
                <div class="stat-label">Edges</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{nx.density(graph):.4f}</div>
                <div class="stat-label">Density</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{'Yes' if nx.is_connected(graph) else 'No'}</div>
                <div class="stat-label">Connected</div>
            </div>
        </div>
        
        <div id="graph-container"></div>
    </div>

    <script>
        // Graph data
        const graphData = {json.dumps(graph_json)};
        
        // Set up the visualization
        const width = document.getElementById('graph-container').clientWidth;
        const height = 600;
        
        const svg = d3.select('#graph-container')
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        // Create force simulation
        const simulation = d3.forceSimulation(graphData.nodes)
            .force('link', d3.forceLink(graphData.links).id(d => d.id))
            .force('charge', d3.forceManyBody().strength(-100))
            .force('center', d3.forceCenter(width / 2, height / 2));
        
        // Create links
        const link = svg.append('g')
            .selectAll('line')
            .data(graphData.links)
            .enter().append('line')
            .attr('stroke', '#999')
            .attr('stroke-opacity', 0.6)
            .attr('stroke-width', 1);
        
        // Create nodes
        const node = svg.append('g')
            .selectAll('circle')
            .data(graphData.nodes)
            .enter().append('circle')
            .attr('r', 5)
            .attr('fill', '#69b3a2')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));
        
        // Add labels
        const label = svg.append('g')
            .selectAll('text')
            .data(graphData.nodes)
            .enter().append('text')
            .text(d => d.id)
            .attr('font-size', '12px')
            .attr('dx', 12)
            .attr('dy', 4);
        
        // Update positions on simulation tick
        simulation.on('tick', () => {{
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
            
            label
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        }});
        
        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    </script>
</body>
</html>
"""
        
        return html_template
    
    async def _create_sentiment_from_visualization(self, visualization_results: Dict) -> SentimentResult:
        """Create sentiment from visualization results."""
        if "error" in visualization_results:
            return SentimentResult(label="neutral", confidence=0.5)
        
        # Simple sentiment based on visualization success
        if visualization_results.get("visualizations"):
            return SentimentResult(label="positive", confidence=0.8)
        else:
            return SentimentResult(label="neutral", confidence=0.5)
    
    @tool
    async def generate_graph_report(self, output_path: Optional[str] = None) -> dict:
        """Generate a comprehensive graph report with visualizations."""
        try:
            # This would typically work with a stored graph
            # For now, return a placeholder
            return {
                "success": True,
                "message": "Graph report generation requires a graph to be loaded",
                "capabilities": [
                    "png_visualization",
                    "html_interactive",
                    "markdown_report",
                    "statistics_summary"
                ],
                "output_formats": ["png", "html", "markdown"],
                "output_directory": str(self.output_dir)
            }
        except Exception as e:
            logger.error(f"Error generating graph report: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    async def visualize_graph(self, graph_data: Dict, style: str = "default") -> dict:
        """Create visualizations of a graph."""
        try:
            graph = await self._build_graph(graph_data)
            
            if graph.number_of_nodes() == 0:
                return {
                    "success": False,
                    "error": "Empty graph"
                }
            
            # Generate visualizations
            request_id = f"viz_{int(asyncio.get_event_loop().time())}"
            results = await self._generate_visualizations(graph, request_id)
            
            return {
                "success": True,
                "visualizations": results["visualizations"],
                "graph_stats": results["graph_stats"],
                "style_used": style,
                "output_directory": str(self.output_dir)
            }
        except Exception as e:
            logger.error(f"Error visualizing graph: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    async def create_interactive_plot(self, graph_data: Dict) -> dict:
        """Create an interactive plot of the graph."""
        try:
            graph = await self._build_graph(graph_data)
            
            if graph.number_of_nodes() == 0:
                return {
                    "success": False,
                    "error": "Empty graph"
                }
            
            # Generate HTML visualization
            request_id = f"interactive_{int(asyncio.get_event_loop().time())}"
            html_path = await self._generate_html_visualization(graph, request_id)
            
            return {
                "success": True,
                "interactive_plot": str(html_path) if html_path else None,
                "graph_stats": {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges()
                }
            }
        except Exception as e:
            logger.error(f"Error creating interactive plot: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    async def generate_graph_summary(self, graph_data: Dict) -> dict:
        """Generate a summary of the graph."""
        try:
            graph = await self._build_graph(graph_data)
            
            if graph.number_of_nodes() == 0:
                return {
                    "success": False,
                    "error": "Empty graph"
                }
            
            # Calculate statistics
            stats = {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "density": nx.density(graph),
                "is_connected": nx.is_connected(graph),
                "connected_components": nx.number_connected_components(graph),
                "average_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
            }
            
            return {
                "success": True,
                "summary": stats,
                "node_list": list(graph.nodes())[:10],  # First 10 nodes
                "edge_list": list(graph.edges())[:10]   # First 10 edges
            }
        except Exception as e:
            logger.error(f"Error generating graph summary: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    async def export_graph_data(self, graph_data: Dict, format: str = "json") -> dict:
        """Export graph data in various formats."""
        try:
            graph = await self._build_graph(graph_data)
            
            if graph.number_of_nodes() == 0:
                return {
                    "success": False,
                    "error": "Empty graph"
                }
            
            # Export based on format
            if format.lower() == "json":
                export_data = nx.node_link_data(graph)
            elif format.lower() == "edgelist":
                export_data = list(graph.edges())
            elif format.lower() == "adjacency":
                export_data = dict(graph.adjacency())
            else:
                return {
                    "success": False,
                    "error": f"Unsupported format: {format}"
                }
            
            return {
                "success": True,
                "format": format,
                "data": export_data,
                "graph_stats": {
                    "nodes": graph.number_of_nodes(),
                    "edges": graph.number_of_edges()
                }
            }
        except Exception as e:
            logger.error(f"Error exporting graph data: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @tool
    async def get_visualization_options(self) -> dict:
        """Get available visualization options and styles."""
        return {
            "success": True,
            "agent_id": self.agent_id,
            "model": self.model_name,
            "visualization_styles": self.visualization_styles,
            "output_formats": ["png", "svg", "html", "markdown"],
            "capabilities": self.metadata["capabilities"],
            "output_directory": str(self.output_dir)
        }
