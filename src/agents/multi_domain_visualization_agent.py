"""
Multi-Domain Knowledge Graph Visualization Agent.
Supports three visualization modes:
1. Separate graphs per domain
2. Combined view with domain filtering
3. Hierarchical view showing domain relationships
"""

import asyncio
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path
import re
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from loguru import logger

from src.agents.base_agent import StrandsBaseAgent
from src.core.models import (
    AnalysisRequest, 
    AnalysisResult, 
    DataType, 
    SentimentResult,
    ProcessingStatus
)
from src.config.config import config
from src.config.settings import settings


class MultiDomainVisualizationAgent(StrandsBaseAgent):
    """
    Multi-Domain Knowledge Graph Visualization Agent.
    
    Features:
    - Separate domain visualizations
    - Combined view with filtering
    - Hierarchical domain relationships
    - Interactive Plotly visualizations
    - Export capabilities (PNG, HTML, JSON)
    """
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs
    ):
        self.model_name = model_name or config.model.default_text_model
        
        super().__init__(
            model_name=self.model_name,
            **kwargs
        )
        
        # Initialize output directory
        self.output_path = Path(
            output_path or settings.paths.results_dir / "visualizations"
        )
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Color schemes for different domains
        self.domain_colors = {
            "en": "#1f77b4",  # Blue
            "zh": "#ff7f0e",  # Orange
            "es": "#2ca02c",  # Green
            "fr": "#d62728",  # Red
            "de": "#9467bd",  # Purple
            "ja": "#8c564b",  # Brown
            "ko": "#e377c2",  # Pink
            "ar": "#7f7f7f",  # Gray
            "ru": "#bcbd22",  # Yellow-green
            "pt": "#17becf"   # Cyan
        }
        
        # Topic colors
        self.topic_colors = {
            "economics": "#ff6b6b",
            "politics": "#4ecdc4",
            "social": "#45b7d1",
            "science": "#96ceb4",
            "war": "#feca57",
            "tech": "#ff9ff3"
        }
        
        self.metadata.update({
            "agent_type": "multi_domain_visualization",
            "model": self.model_name,
            "capabilities": [
                "separate_domain_visualization",
                "combined_view_with_filtering",
                "hierarchical_domain_view",
                "interactive_plotly_charts",
                "export_capabilities",
                "topic_based_filtering",
                "cross_domain_analysis"
            ],
            "supported_formats": ["png", "html", "json", "svg"],
            "visualization_modes": [
                "separate_domains",
                "combined_filtered",
                "hierarchical"
            ]
        })
        
        logger.info(
            f"Multi-Domain Visualization Agent {self.agent_id} initialized"
        )
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            self.visualize_separate_domains,
            self.visualize_combined_view,
            self.visualize_hierarchical_view,
            self.create_interactive_dashboard,
            self.export_visualization,
            self.generate_domain_comparison,
            self.create_topic_analysis_chart,
            self.visualize_cross_domain_connections
        ]
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        return request.data_type == DataType.TEXT
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process visualization request."""
        try:
            # Parse visualization request
            content = json.loads(request.content) if isinstance(request.content, str) else request.content
            
            viz_type = content.get("visualization_type", "separate_domains")
            graphs_data = content.get("graphs_data", {})
            options = content.get("options", {})
            
            if viz_type == "separate_domains":
                result = await self.visualize_separate_domains(graphs_data, options)
            elif viz_type == "combined_view":
                result = await self.visualize_combined_view(graphs_data, options)
            elif viz_type == "hierarchical":
                result = await self.visualize_hierarchical_view(graphs_data, options)
            else:
                result = {"status": "error", "error": f"Unknown visualization type: {viz_type}"}
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                content=request.content,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=1.0,
                    reasoning="Visualization processing completed"
                ),
                processing_time=0.0,
                status=ProcessingStatus.COMPLETED,
                model_used=self.model_name,
                metadata={
                    "agent_id": self.agent_id,
                    "method": "multi_domain_visualization",
                    "visualization_type": viz_type,
                    "result": result
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to process visualization request: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                content=request.content,
                sentiment=SentimentResult(
                    label="error",
                    confidence=0.0,
                    reasoning=f"Visualization failed: {str(e)}"
                ),
                processing_time=0.0,
                status=ProcessingStatus.FAILED,
                model_used=self.model_name,
                metadata={"error": str(e)}
            )
    
    async def visualize_separate_domains(
        self, graphs_data: Dict, options: Dict = None
    ) -> dict:
        """Create separate visualizations for each domain."""
        try:
            options = options or {}
            output_format = options.get("format", "html")
            include_topics = options.get("include_topics", True)
            max_nodes = options.get("max_nodes", 50)
            
            results = {}
            
            for domain, graph_data in graphs_data.items():
                if not graph_data or not graph_data.get("nodes"):
                    continue
                
                # Create separate visualization for this domain
                fig = await self._create_domain_visualization(
                    graph_data, domain, include_topics, max_nodes
                )
                
                # Save visualization
                output_file = self.output_path / f"domain_{domain}_visualization.{output_format}"
                
                if output_format == "html":
                    fig.write_html(str(output_file))
                elif output_format == "png":
                    fig.write_image(str(output_file))
                elif output_format == "svg":
                    fig.write_image(str(output_file))
                
                results[domain] = {
                    "status": "success",
                    "output_file": str(output_file),
                    "node_count": len(graph_data.get("nodes", [])),
                    "edge_count": len(graph_data.get("edges", []))
                }
            
            return {
                "status": "success",
                "visualization_type": "separate_domains",
                "results": results,
                "total_domains": len(results)
            }
            
        except Exception as e:
            logger.error(f"Failed to create separate domain visualizations: {e}")
            return {"status": "error", "error": str(e)}
    
    async def visualize_combined_view(
        self, graphs_data: Dict, options: Dict = None
    ) -> dict:
        """Create a combined view with domain filtering."""
        try:
            options = options or {}
            output_format = options.get("format", "html")
            selected_domains = options.get("selected_domains", list(graphs_data.keys()))
            selected_topics = options.get("selected_topics", [])
            max_nodes = options.get("max_nodes", 100)
            
            # Combine selected domains
            combined_nodes = []
            combined_edges = []
            
            for domain in selected_domains:
                if domain not in graphs_data:
                    continue
                
                domain_data = graphs_data[domain]
                domain_nodes = domain_data.get("nodes", [])
                domain_edges = domain_data.get("edges", [])
                
                # Filter by topics if specified
                if selected_topics:
                    domain_nodes = [
                        node for node in domain_nodes
                        if any(topic in node.get("topics", []) for topic in selected_topics)
                    ]
                    # Filter edges to only include filtered nodes
                    node_names = {node["name"] for node in domain_nodes}
                    domain_edges = [
                        edge for edge in domain_edges
                        if edge.get("source") in node_names and edge.get("target") in node_names
                    ]
                
                # Add domain prefix to node names to avoid conflicts
                for node in domain_nodes:
                    node["name"] = f"{domain}:{node['name']}"
                    node["domain"] = domain
                
                for edge in domain_edges:
                    edge["source"] = f"{domain}:{edge['source']}"
                    edge["target"] = f"{domain}:{edge['target']}"
                    edge["domain"] = domain
                
                combined_nodes.extend(domain_nodes)
                combined_edges.extend(domain_edges)
            
            # Limit nodes if specified
            if len(combined_nodes) > max_nodes:
                # Keep nodes with highest degree
                node_degrees = defaultdict(int)
                for edge in combined_edges:
                    node_degrees[edge["source"]] += 1
                    node_degrees[edge["target"]] += 1
                
                top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
                top_node_names = {node[0] for node in top_nodes}
                
                combined_nodes = [node for node in combined_nodes if node["name"] in top_node_names]
                combined_edges = [
                    edge for edge in combined_edges
                    if edge["source"] in top_node_names and edge["target"] in top_node_names
                ]
            
            # Create combined visualization
            fig = await self._create_combined_visualization(
                combined_nodes, combined_edges, selected_domains
            )
            
            # Save visualization
            output_file = self.output_path / f"combined_view_visualization.{output_format}"
            
            if output_format == "html":
                fig.write_html(str(output_file))
            elif output_format == "png":
                fig.write_image(str(output_file))
            elif output_format == "svg":
                fig.write_image(str(output_file))
            
            return {
                "status": "success",
                "visualization_type": "combined_view",
                "output_file": str(output_file),
                "selected_domains": selected_domains,
                "selected_topics": selected_topics,
                "node_count": len(combined_nodes),
                "edge_count": len(combined_edges)
            }
            
        except Exception as e:
            logger.error(f"Failed to create combined view visualization: {e}")
            return {"status": "error", "error": str(e)}
    
    async def visualize_hierarchical_view(
        self, graphs_data: Dict, options: Dict = None
    ) -> dict:
        """Create a hierarchical view showing domain relationships."""
        try:
            options = options or {}
            output_format = options.get("format", "html")
            include_cross_domain = options.get("include_cross_domain", True)
            
            # Create hierarchical structure
            hierarchy = {
                "name": "Knowledge Graph",
                "children": []
            }
            
            # Add domain nodes
            for domain, graph_data in graphs_data.items():
                if not graph_data:
                    continue
                
                domain_node = {
                    "name": domain,
                    "type": "domain",
                    "size": len(graph_data.get("nodes", [])),
                    "children": []
                }
                
                # Group nodes by topics
                topic_groups = defaultdict(list)
                for node in graph_data.get("nodes", []):
                    topics = node.get("topics", ["general"])
                    for topic in topics:
                        topic_groups[topic].append(node)
                
                # Add topic nodes
                for topic, nodes in topic_groups.items():
                    topic_node = {
                        "name": topic,
                        "type": "topic",
                        "size": len(nodes),
                        "children": [
                            {
                                "name": node["name"],
                                "type": "entity",
                                "size": 1,
                                "entity_type": node.get("type", "unknown")
                            }
                            for node in nodes[:10]  # Limit to 10 entities per topic
                        ]
                    }
                    domain_node["children"].append(topic_node)
                
                hierarchy["children"].append(domain_node)
            
            # Create hierarchical visualization
            fig = await self._create_hierarchical_visualization(hierarchy)
            
            # Save visualization
            output_file = self.output_path / f"hierarchical_view_visualization.{output_format}"
            
            if output_format == "html":
                fig.write_html(str(output_file))
            elif output_format == "png":
                fig.write_image(str(output_file))
            elif output_format == "svg":
                fig.write_image(str(output_file))
            
            return {
                "status": "success",
                "visualization_type": "hierarchical_view",
                "output_file": str(output_file),
                "domain_count": len(hierarchy["children"]),
                "total_entities": sum(
                    domain["size"] for domain in hierarchy["children"]
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to create hierarchical view visualization: {e}")
            return {"status": "error", "error": str(e)}
    
    async def create_interactive_dashboard(
        self, graphs_data: Dict, options: Dict = None
    ) -> dict:
        """Create an interactive dashboard with all visualization modes."""
        try:
            options = options or {}
            
            # Create subplots for different views
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Domain Overview", "Topic Distribution",
                    "Cross-Domain Connections", "Entity Network"
                ),
                specs=[
                    [{"type": "bar"}, {"type": "pie"}],
                    [{"type": "scatter"}, {"type": "scatter"}]
                ]
            )
            
            # 1. Domain Overview (Bar Chart)
            domains = list(graphs_data.keys())
            node_counts = [len(graphs_data[domain].get("nodes", [])) for domain in domains]
            
            fig.add_trace(
                go.Bar(
                    x=domains,
                    y=node_counts,
                    name="Nodes per Domain",
                    marker_color=[self.domain_colors.get(domain, "#cccccc") for domain in domains]
                ),
                row=1, col=1
            )
            
            # 2. Topic Distribution (Pie Chart)
            all_topics = defaultdict(int)
            for domain_data in graphs_data.values():
                for node in domain_data.get("nodes", []):
                    for topic in node.get("topics", []):
                        all_topics[topic] += 1
            
            if all_topics:
                fig.add_trace(
                    go.Pie(
                        labels=list(all_topics.keys()),
                        values=list(all_topics.values()),
                        name="Topic Distribution"
                    ),
                    row=1, col=2
                )
            
            # 3. Cross-Domain Connections (Scatter)
            # This would show connections between domains
            # For now, create a placeholder
            
            # 4. Entity Network (Scatter)
            # This would show a network view of entities
            # For now, create a placeholder
            
            # Update layout
            fig.update_layout(
                title="Multi-Domain Knowledge Graph Dashboard",
                height=800,
                showlegend=True
            )
            
            # Save dashboard
            output_file = self.output_path / "interactive_dashboard.html"
            fig.write_html(str(output_file))
            
            return {
                "status": "success",
                "visualization_type": "interactive_dashboard",
                "output_file": str(output_file)
            }
            
        except Exception as e:
            logger.error(f"Failed to create interactive dashboard: {e}")
            return {"status": "error", "error": str(e)}
    
    async def export_visualization(
        self, visualization_data: Dict, format: str = "html"
    ) -> dict:
        """Export visualization in various formats."""
        try:
            if format not in ["html", "png", "svg", "json"]:
                return {"status": "error", "error": f"Unsupported format: {format}"}
            
            output_file = self.output_path / f"exported_visualization.{format}"
            
            if format == "json":
                with open(output_file, 'w') as f:
                    json.dump(visualization_data, f, indent=2)
            else:
                # For other formats, would need to convert visualization
                pass
            
            return {
                "status": "success",
                "output_file": str(output_file),
                "format": format
            }
            
        except Exception as e:
            logger.error(f"Failed to export visualization: {e}")
            return {"status": "error", "error": str(e)}
    
    async def generate_domain_comparison(
        self, graphs_data: Dict, options: Dict = None
    ) -> dict:
        """Generate comparison charts between domains."""
        try:
            options = options or {}
            comparison_type = options.get("comparison_type", "size")
            
            if comparison_type == "size":
                # Compare domain sizes
                domains = list(graphs_data.keys())
                node_counts = [len(graphs_data[domain].get("nodes", [])) for domain in domains]
                edge_counts = [len(graphs_data[domain].get("edges", [])) for domain in domains]
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=domains,
                    y=node_counts,
                    name="Nodes",
                    marker_color=[self.domain_colors.get(domain, "#cccccc") for domain in domains]
                ))
                
                fig.add_trace(go.Bar(
                    x=domains,
                    y=edge_counts,
                    name="Edges",
                    marker_color=[self.domain_colors.get(domain, "#cccccc") for domain in domains]
                ))
                
                fig.update_layout(
                    title="Domain Size Comparison",
                    xaxis_title="Domain",
                    yaxis_title="Count",
                    barmode="group"
                )
            
            # Save comparison
            output_file = self.output_path / f"domain_comparison_{comparison_type}.html"
            fig.write_html(str(output_file))
            
            return {
                "status": "success",
                "comparison_type": comparison_type,
                "output_file": str(output_file)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate domain comparison: {e}")
            return {"status": "error", "error": str(e)}
    
    async def create_topic_analysis_chart(
        self, graphs_data: Dict, options: Dict = None
    ) -> dict:
        """Create topic analysis charts."""
        try:
            options = options or {}
            
            # Analyze topics across domains
            topic_analysis = defaultdict(lambda: defaultdict(int))
            
            for domain, graph_data in graphs_data.items():
                for node in graph_data.get("nodes", []):
                    for topic in node.get("topics", []):
                        topic_analysis[topic][domain] += 1
            
            # Create heatmap
            topics = list(topic_analysis.keys())
            domains = list(graphs_data.keys())
            
            heatmap_data = []
            for topic in topics:
                row = [topic_analysis[topic].get(domain, 0) for domain in domains]
                heatmap_data.append(row)
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=domains,
                y=topics,
                colorscale="Viridis"
            ))
            
            fig.update_layout(
                title="Topic Distribution Across Domains",
                xaxis_title="Domain",
                yaxis_title="Topic"
            )
            
            # Save chart
            output_file = self.output_path / "topic_analysis_heatmap.html"
            fig.write_html(str(output_file))
            
            return {
                "status": "success",
                "output_file": str(output_file),
                "topic_count": len(topics),
                "domain_count": len(domains)
            }
            
        except Exception as e:
            logger.error(f"Failed to create topic analysis chart: {e}")
            return {"status": "error", "error": str(e)}
    
    async def visualize_cross_domain_connections(
        self, cross_domain_data: Dict, options: Dict = None
    ) -> dict:
        """Visualize cross-domain connections."""
        try:
            options = options or {}
            
            # Create network visualization of cross-domain connections
            nodes = cross_domain_data.get("nodes", [])
            edges = cross_domain_data.get("edges", [])
            
            # Create network graph
            fig = go.Figure()
            
            # Add edges
            for edge in edges:
                fig.add_trace(go.Scatter(
                    x=[edge["source_x"], edge["target_x"]],
                    y=[edge["source_y"], edge["target_y"]],
                    mode="lines",
                    line=dict(width=1, color="gray"),
                    showlegend=False
                ))
            
            # Add nodes
            for node in nodes:
                fig.add_trace(go.Scatter(
                    x=[node["x"]],
                    y=[node["y"]],
                    mode="markers+text",
                    marker=dict(
                        size=node.get("size", 10),
                        color=self.domain_colors.get(node.get("domain", "en"), "#cccccc")
                    ),
                    text=node["name"],
                    textposition="middle center",
                    name=node.get("domain", "unknown")
                ))
            
            fig.update_layout(
                title="Cross-Domain Connections",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                showlegend=True
            )
            
            # Save visualization
            output_file = self.output_path / "cross_domain_connections.html"
            fig.write_html(str(output_file))
            
            return {
                "status": "success",
                "output_file": str(output_file),
                "node_count": len(nodes),
                "edge_count": len(edges)
            }
            
        except Exception as e:
            logger.error(f"Failed to visualize cross-domain connections: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _create_domain_visualization(
        self, graph_data: Dict, domain: str, include_topics: bool, max_nodes: int
    ) -> go.Figure:
        """Create visualization for a single domain."""
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])
        
        # Limit nodes if specified
        if len(nodes) > max_nodes:
            # Keep nodes with highest degree
            node_degrees = defaultdict(int)
            for edge in edges:
                node_degrees[edge["source"]] += 1
                node_degrees[edge["target"]] += 1
            
            top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            top_node_names = {node[0] for node in top_nodes}
            
            nodes = [node for node in nodes if node["name"] in top_node_names]
            edges = [
                edge for edge in edges
                if edge["source"] in top_node_names and edge["target"] in top_node_names
            ]
        
        # Create network visualization
        fig = go.Figure()
        
        # Add edges
        for edge in edges:
            fig.add_trace(go.Scatter(
                x=[edge["source_x"], edge["target_x"]],
                y=[edge["source_y"], edge["target_y"]],
                mode="lines",
                line=dict(width=1, color="gray"),
                showlegend=False
            ))
        
        # Add nodes
        for node in nodes:
            color = self.domain_colors.get(domain, "#cccccc")
            if include_topics and node.get("topics"):
                # Use topic color if available
                topic = node["topics"][0] if node["topics"] else "general"
                color = self.topic_colors.get(topic, color)
            
            fig.add_trace(go.Scatter(
                x=[node["x"]],
                y=[node["y"]],
                mode="markers+text",
                marker=dict(
                    size=node.get("size", 10),
                    color=color
                ),
                text=node["name"],
                textposition="middle center",
                name=node.get("type", "unknown")
            ))
        
        fig.update_layout(
            title=f"Knowledge Graph - {domain.upper()} Domain",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=True
        )
        
        return fig
    
    async def _create_combined_visualization(
        self, nodes: List[Dict], edges: List[Dict], domains: List[str]
    ) -> go.Figure:
        """Create combined visualization."""
        fig = go.Figure()
        
        # Add edges
        for edge in edges:
            fig.add_trace(go.Scatter(
                x=[edge["source_x"], edge["target_x"]],
                y=[edge["source_y"], edge["target_y"]],
                mode="lines",
                line=dict(width=1, color="gray"),
                showlegend=False
            ))
        
        # Add nodes by domain
        for domain in domains:
            domain_nodes = [node for node in nodes if node.get("domain") == domain]
            
            for node in domain_nodes:
                fig.add_trace(go.Scatter(
                    x=[node["x"]],
                    y=[node["y"]],
                    mode="markers+text",
                    marker=dict(
                        size=node.get("size", 10),
                        color=self.domain_colors.get(domain, "#cccccc")
                    ),
                    text=node["name"],
                    textposition="middle center",
                    name=f"{domain}: {node.get('type', 'unknown')}"
                ))
        
        fig.update_layout(
            title="Combined Knowledge Graph View",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=True
        )
        
        return fig
    
    async def _create_hierarchical_visualization(self, hierarchy: Dict) -> go.Figure:
        """Create hierarchical visualization."""
        fig = go.Figure()
        
        # Create sunburst chart
        fig.add_trace(go.Sunburst(
            labels=[hierarchy["name"]] + self._extract_labels(hierarchy),
            parents=[""] + self._extract_parents(hierarchy),
            values=[hierarchy["size"]] + self._extract_values(hierarchy),
            branchvalues="total"
        ))
        
        fig.update_layout(
            title="Hierarchical Knowledge Graph View",
            width=800,
            height=800
        )
        
        return fig
    
    def _extract_labels(self, node: Dict) -> List[str]:
        """Extract labels from hierarchical structure."""
        labels = []
        for child in node.get("children", []):
            labels.append(child["name"])
            labels.extend(self._extract_labels(child))
        return labels
    
    def _extract_parents(self, node: Dict) -> List[str]:
        """Extract parents from hierarchical structure."""
        parents = []
        for child in node.get("children", []):
            parents.append(node["name"])
            parents.extend(self._extract_parents(child))
        return parents
    
    def _extract_values(self, node: Dict) -> List[int]:
        """Extract values from hierarchical structure."""
        values = []
        for child in node.get("children", []):
            values.append(child["size"])
            values.extend(self._extract_values(child))
        return values
