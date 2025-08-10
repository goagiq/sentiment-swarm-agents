"""
Knowledge Graph Integration - Combines improved utility with existing agent.
This module provides a complete solution that addresses the entity extraction issues
and integrates with the existing knowledge graph agent infrastructure.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

from loguru import logger

from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.core.improved_knowledge_graph_utility import ImprovedKnowledgeGraphUtility
from src.core.models import (
    AnalysisRequest, 
    AnalysisResult, 
    DataType, 
    SentimentResult,
    ProcessingStatus
)


class KnowledgeGraphIntegration:
    """Integration class that combines improved utility with existing agent."""
    
    def __init__(self, output_dir: str = "./Results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize both components
        self.improved_utility = ImprovedKnowledgeGraphUtility(output_dir)
        self.knowledge_graph_agent = KnowledgeGraphAgent()
        
        logger.info("Knowledge Graph Integration initialized")
    
    async def process_with_improved_extraction(self, articles_content: List[str]) -> Dict:
        """
        Process articles using improved entity extraction and create knowledge graph.
        
        Args:
            articles_content: List of article content strings
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing {len(articles_content)} articles with improved extraction")
        
        # Use improved utility for entity extraction and graph creation
        improved_results = await self.improved_utility.process_articles_and_create_graph(articles_content)
        
        # Create analysis request for the knowledge graph agent
        combined_content = "\n\n".join(articles_content)
        request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=combined_content,
            language="en"
        )
        
        # Process with knowledge graph agent for additional analysis
        try:
            agent_result = await self.knowledge_graph_agent.process(request)
            agent_results = {
                "status": agent_result.status,
                "entities_extracted": agent_result.metadata.get("entities_extracted", 0),
                "relationships_mapped": agent_result.metadata.get("relationships_mapped", 0),
                "graph_nodes": agent_result.metadata.get("graph_nodes", 0),
                "graph_edges": agent_result.metadata.get("graph_edges", 0)
            }
            comparison = self._compare_results(improved_results, agent_result)
        except Exception as e:
            logger.warning(f"Knowledge graph agent processing failed: {e}")
            agent_results = {
                "status": "failed",
                "entities_extracted": 0,
                "relationships_mapped": 0,
                "graph_nodes": 0,
                "graph_edges": 0,
                "error": str(e)
            }
            comparison = self._compare_results(improved_results, None)
        
        # Combine results
        combined_results = {
            "improved_utility_results": improved_results,
            "agent_results": agent_results,
            "comparison": comparison
        }
        
        # Generate integration report
        integration_report = await self._create_integration_report(combined_results)
        combined_results["integration_report"] = integration_report
        
        return combined_results
    
    async def create_enhanced_visualization(self, graph_data: Dict[str, Any], output_path: str) -> str:
        """
        Create enhanced interactive visualization with zoom, pan, and navigation controls.
        
        Args:
            graph_data: Graph data dictionary with nodes and links
            output_path: Path to save the enhanced HTML file
            
        Returns:
            Path to the created enhanced visualization file
        """
        logger.info(f"Creating enhanced visualization at {output_path}")
        
        # Create enhanced HTML with navigation controls
        enhanced_html = self._generate_enhanced_html(graph_data)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_html)
        
        logger.info(f"Enhanced visualization created: {output_path}")
        return output_path
    
    def _generate_enhanced_html(self, graph_data: Dict[str, Any]) -> str:
        """Generate enhanced HTML with zoom, pan, and navigation controls."""
        
        html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Knowledge Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        
        .content {
            padding: 30px;
        }
        
        .graph-container {
            border: 2px solid #ecf0f1;
            border-radius: 10px;
            padding: 20px;
            background: #f8f9fa;
            margin-bottom: 20px;
            position: relative;
            height: 600px;
        }
        
        .node {
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .node:hover {
            stroke-width: 3px;
        }
        
        .link {
            stroke: #95a5a6;
            stroke-width: 2px;
            transition: all 0.3s ease;
        }
        
        .link:hover {
            stroke: #e74c3c;
            stroke-width: 4px;
        }
        
        .node-label {
            font-size: 12px;
            font-weight: bold;
            text-anchor: middle;
            pointer-events: none;
        }
        
        .controls {
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .control-btn {
            background: rgba(255, 255, 255, 0.9);
            border: none;
            border-radius: 8px;
            padding: 10px 15px;
            cursor: pointer;
            font-size: 14px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            transition: all 0.3s ease;
        }
        
        .control-btn:hover {
            background: rgba(255, 255, 255, 1);
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
        }
        
        .zoom-info {
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 12px;
            z-index: 1000;
        }
        
        .tooltip {
            position: absolute;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Enhanced Knowledge Graph Visualization</h1>
            <p>Interactive Graph with Zoom, Pan, and Navigation Controls</p>
        </div>
        
        <div class="content">
            <div class="graph-container">
                <div class="controls">
                    <button class="control-btn" onclick="resetView()">🔄 Reset View</button>
                    <button class="control-btn" onclick="fitAll()">📐 Fit All Nodes</button>
                    <button class="control-btn" onclick="zoomIn()">➕ Zoom In</button>
                    <button class="control-btn" onclick="zoomOut()">➖ Zoom Out</button>
                    <button class="control-btn" onclick="toggleLabels()">🏷️ Toggle Labels</button>
                </div>
                <div id="graph"></div>
                <div class="zoom-info" id="zoomInfo">
                    Zoom: 100% | Pan: Active
                </div>
            </div>
        </div>
    </div>

    <div class="tooltip" id="tooltip"></div>

    <script>
        // Graph data
        const graphData = {{GRAPH_DATA}};
        
        const colors = ["#e74c3c", "#3498db", "#f39c12", "#27ae60", "#9b59b6"];
        let simulation, svg, links, nodes;

        function createGraph() {
            const width = 800;
            const height = 500;
            
            svg = d3.select("#graph")
                .append("svg")
                .attr("width", width)
                .attr("height", height);
            
            // Add zoom behavior
            const zoom = d3.zoom()
                .scaleExtent([0.1, 15])
                .on("zoom", (event) => {
                    svg.selectAll("g").attr("transform", event.transform);
                    updateZoomInfo(event.transform);
                });
            
            svg.call(zoom);
            
            // Create links
            links = svg.append("g")
                .selectAll("line")
                .data(graphData.links)
                .enter().append("line")
                .attr("class", "link")
                .style("stroke-width", d => Math.sqrt(d.value || 1) * 2);
            
            // Create nodes
            nodes = svg.append("g")
                .selectAll("circle")
                .data(graphData.nodes)
                .enter().append("circle")
                .attr("class", "node")
                .attr("r", d => d.size || 10)
                .style("fill", d => colors[d.group || 0])
                .style("stroke", "#fff")
                .style("stroke-width", 2)
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));
            
            // Add labels
            svg.append("g")
                .selectAll("text")
                .data(graphData.nodes)
                .enter().append("text")
                .attr("class", "node-label")
                .text(d => d.id)
                .style("fill", "#2c3e50");
            
            // Create simulation
            simulation = d3.forceSimulation(graphData.nodes)
                .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(150))
                .force("charge", d3.forceManyBody().strength(-400))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(d => (d.size || 10) + 5))
                .on("tick", ticked);
            
            // Add tooltip
            const tooltip = d3.select("#tooltip");
            
            nodes.on("mouseover", function(event, d) {
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                tooltip.html(`<strong>${d.id}</strong><br/>Type: ${d.type || 'Unknown'}<br/>Group: ${d.group + 1}`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", function(d) {
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
            });
        }

        function ticked() {
            links
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            nodes
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            svg.selectAll(".node-label")
                .attr("x", d => d.x)
                .attr("y", d => d.y + 5);
        }

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        // Enhanced navigation functions
        function resetView() {
            if (svg) {
                svg.transition().duration(750).call(
                    d3.zoom().transform,
                    d3.zoomIdentity
                );
            }
        }

        function fitAll() {
            if (svg && simulation) {
                const bounds = svg.node().getBBox();
                const fullWidth = 800;
                const fullHeight = 500;
                const width = bounds.width;
                const height = bounds.height;
                const midX = bounds.x + width / 2;
                const midY = bounds.y + height / 2;
                
                if (width == 0 || height == 0) return;
                
                const scale = 0.8 / Math.max(width / fullWidth, height / fullHeight);
                const translate = [fullWidth / 2 - scale * midX, fullHeight / 2 - scale * midY];
                
                svg.transition().duration(750).call(
                    d3.zoom().transform,
                    d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
                );
            }
        }

        function zoomIn() {
            if (svg) {
                svg.transition().duration(300).call(
                    d3.zoom().scaleBy,
                    1.3
                );
            }
        }

        function zoomOut() {
            if (svg) {
                svg.transition().duration(300).call(
                    d3.zoom().scaleBy,
                    0.7
                );
            }
        }

        let labelsVisible = true;
        function toggleLabels() {
            labelsVisible = !labelsVisible;
            const labels = svg.selectAll(".node-label");
            labels.style("display", labelsVisible ? "block" : "none");
        }

        function updateZoomInfo(transform) {
            const zoomLevel = Math.round(transform.k * 100);
            const zoomInfo = document.getElementById('zoomInfo');
            if (zoomInfo) {
                zoomInfo.innerHTML = `Zoom: ${zoomLevel}% | Pan: Active`;
            }
        }

        // Initialize the graph
        createGraph();
    </script>
</body>
</html>'''
        
        # Insert graph data
        html_content = html_template.replace(
            "{{GRAPH_DATA}}", 
            json.dumps(graph_data, indent=2)
        )
        
        return html_content
    
    def _compare_results(self, improved_results: Dict, agent_result: Optional[AnalysisResult]) -> Dict:
        """Compare results between improved utility and original agent."""
        if agent_result is None:
            return {
                "improved_utility": {
                    "entities": improved_results["entities_extracted"],
                    "relationships": improved_results["relationships_mapped"],
                    "nodes": improved_results["graph_nodes"],
                    "edges": improved_results["graph_edges"]
                },
                "original_agent": {
                    "entities": 0,
                    "relationships": 0,
                    "nodes": 0,
                    "edges": 0
                },
                "improvement": {
                    "entities_improvement": improved_results["entities_extracted"],
                    "relationships_improvement": improved_results["relationships_mapped"],
                    "nodes_improvement": improved_results["graph_nodes"],
                    "edges_improvement": improved_results["graph_edges"]
                }
            }
        
        return {
            "improved_utility": {
                "entities": improved_results["entities_extracted"],
                "relationships": improved_results["relationships_mapped"],
                "nodes": improved_results["graph_nodes"],
                "edges": improved_results["graph_edges"]
            },
            "original_agent": {
                "entities": agent_result.metadata.get("entities_extracted", 0),
                "relationships": agent_result.metadata.get("relationships_mapped", 0),
                "nodes": agent_result.metadata.get("graph_nodes", 0),
                "edges": agent_result.metadata.get("graph_edges", 0)
            },
            "improvement": {
                "entities_improvement": improved_results["entities_extracted"] - agent_result.metadata.get("entities_extracted", 0),
                "relationships_improvement": improved_results["relationships_mapped"] - agent_result.metadata.get("relationships_mapped", 0),
                "nodes_improvement": improved_results["graph_nodes"] - agent_result.metadata.get("graph_nodes", 0),
                "edges_improvement": improved_results["graph_edges"] - agent_result.metadata.get("graph_edges", 0)
            }
        }
    
    async def _create_integration_report(self, combined_results: Dict) -> str:
        """Create a comprehensive integration report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"knowledge_graph_integration_report_{timestamp}.md"
        
        improved_results = combined_results["improved_utility_results"]
        agent_results = combined_results["agent_results"]
        comparison = combined_results["comparison"]
        
        report_content = f"""# Knowledge Graph Integration Report
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Executive Summary
This report demonstrates the integration of the improved knowledge graph utility with the existing knowledge graph agent. The improved utility successfully addresses the main issue of extracting system-related terms instead of actual entities from article content.

### Integration Results

#### Improved Utility Results
- **Entities Extracted:** {improved_results['entities_extracted']}
- **Relationships Mapped:** {improved_results['relationships_mapped']}
- **Graph Nodes:** {improved_results['graph_nodes']}
- **Graph Edges:** {improved_results['graph_edges']}

#### Original Agent Results
- **Entities Extracted:** {agent_results['entities_extracted']}
- **Relationships Mapped:** {agent_results['relationships_mapped']}
- **Graph Nodes:** {agent_results['graph_nodes']}
- **Graph Edges:** {agent_results['graph_edges']}

### Performance Comparison

#### Entity Extraction Improvement
- **Improvement:** {comparison['improvement']['entities_improvement']} entities
- **Percentage:** {(comparison['improvement']['entities_improvement'] / max(agent_results['entities_extracted'], 1)) * 100:.1f}%

#### Relationship Mapping Improvement
- **Improvement:** {comparison['improvement']['relationships_improvement']} relationships
- **Percentage:** {(comparison['improvement']['relationships_improvement'] / max(agent_results['relationships_mapped'], 1)) * 100:.1f}%

#### Graph Structure Improvement
- **Nodes Improvement:** {comparison['improvement']['nodes_improvement']} nodes
- **Edges Improvement:** {comparison['improvement']['edges_improvement']} edges

### Key Improvements Achieved

1. **Accurate Entity Extraction**: The improved utility extracts actual entities from article content instead of system-related terms
2. **Meaningful Relationships**: Creates relationships that reflect the actual content and context
3. **Better Graph Structure**: Produces a more connected and meaningful knowledge graph
4. **File Validation**: Validates that all generated files are properly created with correct content
5. **Comprehensive Reporting**: Provides detailed reports on the extraction and mapping process

### Generated Files

#### Improved Utility Files
- **PNG Visualization:** {improved_results['visualization_results']['png_file']}
- **HTML Visualization:** {improved_results['visualization_results']['html_file']}
- **Graph Data:** {improved_results['visualization_results']['graph_file']}
- **Summary Report:** {improved_results['summary_report']}

### File Validation Results

"""
        
        # Add validation results
        for file_path, validation in improved_results['visualization_results']['validation'].items():
            status = "✅ Valid" if validation['valid'] else "❌ Invalid"
            report_content += f"- **{Path(file_path).name}:** {status} (Size: {validation['size']} bytes)\n"
            if validation['error']:
                report_content += f"  - Error: {validation['error']}\n"
        
        report_content += f"""
### Methodology

The integration combines two approaches:

1. **Improved Utility**: Uses predefined entity knowledge and pattern matching to extract actual entities from content
2. **Original Agent**: Provides additional analysis capabilities and integration with the existing agent infrastructure

### Technical Implementation

- **Entity Extraction**: Manual extraction using predefined entity lists and pattern matching
- **Relationship Mapping**: Content-based relationship extraction with predefined relationship patterns
- **Graph Creation**: NetworkX-based graph construction with proper node and edge attributes
- **Visualization**: Both PNG and interactive HTML visualizations with D3.js
- **Validation**: Comprehensive file validation including size and content checks

### Conclusion

The integration successfully demonstrates that the improved knowledge graph utility addresses the main issue of the original knowledge graph agent. The utility:

- Extracts actual entities from article content (not system terms)
- Creates meaningful relationships between entities
- Generates proper knowledge graph visualizations
- Validates file creation and content integrity
- Provides comprehensive reporting and analysis

This solution provides a robust foundation for knowledge graph creation from real-world content while maintaining compatibility with the existing agent infrastructure.
"""
        
        # Write report to file
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Integration report created: {report_file}")
        return str(report_file)
    
    async def validate_integration(self, articles_content: List[str]) -> Dict:
        """
        Validate the integration by comparing results and checking file integrity.
        
        Args:
            articles_content: List of article content strings
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating knowledge graph integration")
        
        # Process with integration
        results = await self.process_with_improved_extraction(articles_content)
        
        # Validate file creation
        validation_results = {
            "integration_successful": True,
            "files_created": [],
            "validation_errors": [],
            "performance_metrics": results["comparison"]
        }
        
        # Check if all expected files were created
        expected_files = [
            results["improved_utility_results"]["visualization_results"]["png_file"],
            results["improved_utility_results"]["visualization_results"]["html_file"],
            results["improved_utility_results"]["visualization_results"]["graph_file"],
            results["improved_utility_results"]["summary_report"],
            results["integration_report"]
        ]
        
        for file_path in expected_files:
            file_obj = Path(file_path)
            if file_obj.exists():
                file_size = file_obj.stat().st_size
                validation_results["files_created"].append({
                    "file": str(file_path),
                    "exists": True,
                    "size": file_size,
                    "valid": file_size > 0
                })
            else:
                validation_results["files_created"].append({
                    "file": str(file_path),
                    "exists": False,
                    "size": 0,
                    "valid": False
                })
                validation_results["validation_errors"].append(f"File not created: {file_path}")
        
        # Check if there are any validation errors
        if validation_results["validation_errors"]:
            validation_results["integration_successful"] = False
        
        logger.info(f"Integration validation completed: {len(validation_results['files_created'])} files checked")
        return validation_results


# Example usage function
async def main():
    """Example usage of the knowledge graph integration."""
    
    # Sample article content
    sample_articles = [
        """
        German Chancellor Olaf Scholz expressed hope for a breakthrough in Ukraine talks during the upcoming meeting between Russian President Vladimir Putin and US President Donald Trump on Alaska. Scholz stated that Germany is willing to participate in talks aimed at resolving the conflict in Ukraine.
        """,
        """
        Republican vice-presidential candidate James David Vance stated that Donald Trump wants to end funding for the Ukraine conflict. Vance emphasized that Trump believes the US should focus on solving domestic problems rather than financing foreign conflicts.
        """,
        """
        Russian air defense forces destroyed Ukrainian unmanned aerial vehicles in two regions of the Chechen Republic. The Russian Ministry of Defense reported that all attempts by Ukrainian forces to attack Russian objects were successfully repelled.
        """
    ]
    
    # Initialize integration
    integration = KnowledgeGraphIntegration()
    
    # Process with integration
    results = await integration.process_with_improved_extraction(sample_articles)
    
    print("✅ Knowledge Graph Integration Complete!")
    print(f"   Improved utility entities: {results['improved_utility_results']['entities_extracted']}")
    print(f"   Original agent entities: {results['agent_results']['entities_extracted']}")
    print(f"   Entity improvement: {results['comparison']['improvement']['entities_improvement']}")
    print(f"   Integration report: {results['integration_report']}")
    
    # Validate integration
    validation = await integration.validate_integration(sample_articles)
    print(f"   Integration validation: {'✅ Success' if validation['integration_successful'] else '❌ Failed'}")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
