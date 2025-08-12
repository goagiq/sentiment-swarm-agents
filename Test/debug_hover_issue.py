#!/usr/bin/env python3
"""
Debug script to test hover functionality and identify tooltip issues.
"""

import asyncio
from pathlib import Path
from loguru import logger

from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.core.models import AnalysisRequest, DataType


async def test_hover_functionality():
    """Test hover functionality with a simple graph."""
    logger.info("=== Testing Hover Functionality ===")
    
    # Initialize agent
    agent = KnowledgeGraphAgent()
    
    # Create a simple test with known entities
    test_text = """
    John Smith works at Microsoft Corporation in Seattle, Washington.
    Microsoft is a technology company based in Redmond, Washington.
    Seattle is a city in Washington state.
    """
    
    # Create analysis request
    request = AnalysisRequest(
        content=test_text,
        data_type=DataType.TEXT,
        language="en",
        analysis_type="knowledge_graph"
    )
    
    # Process the request
    logger.info("Processing test text...")
    result = await agent.process_request(request)
    
    # Generate graph report
    logger.info("Generating graph report...")
    report_result = await agent.generate_graph_report(
        output_path="hover_test_report",
        target_language="en"
    )
    
    if report_result:
        logger.info("✅ Graph report generated successfully")
        
        # Check if files exist
        html_file = Path("Results/reports/hover_test_report.html")
        
        if html_file.exists():
            logger.info(f"HTML file created: {html_file}")
            
            # Read and analyze the HTML content
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Check for tooltip elements
            has_tooltip_div = 'id="tooltip"' in html_content
            has_tooltip_css = '.tooltip' in html_content
            has_mouseover_events = 'mouseover' in html_content
            has_mouseout_events = 'mouseout' in html_content
            
            logger.info(f"Tooltip div: {has_tooltip_div}")
            logger.info(f"Tooltip CSS: {has_tooltip_css}")
            logger.info(f"Mouseover events: {has_mouseover_events}")
            logger.info(f"Mouseout events: {has_mouseout_events}")
            
            # Check for node data
            has_nodes_data = 'const nodes =' in html_content
            has_node_elements = '.selectAll(\'circle\')' in html_content
            
            logger.info(f"Nodes data: {has_nodes_data}")
            logger.info(f"Node elements: {has_node_elements}")
            
            # Check for specific tooltip functionality
            has_tooltip_content = 'tooltipContent' in html_content
            has_tooltip_display = 'tooltip.style(\'display\', \'block\')' in html_content
            
            logger.info(f"Tooltip content generation: {has_tooltip_content}")
            logger.info(f"Tooltip display setting: {has_tooltip_display}")
            
            # Check for potential issues
            if has_tooltip_div and has_mouseover_events and has_nodes_data:
                logger.info("✅ All tooltip components present")
            else:
                logger.warning("⚠️ Missing tooltip components")
                
            # Check for CSS issues
            if has_tooltip_css:
                # Look for specific CSS properties
                has_position = 'position: absolute' in html_content
                has_z_index = 'z-index' in html_content
                has_pointer_events = 'pointer-events' in html_content
                
                logger.info(f"Tooltip position: {has_position}")
                logger.info(f"Tooltip z-index: {has_z_index}")
                logger.info(f"Tooltip pointer-events: {has_pointer_events}")
            
        else:
            logger.error("❌ HTML file not created")
    else:
        logger.error("❌ Failed to generate graph report")


async def test_simple_html():
    """Create a simple test HTML to verify tooltip functionality."""
    logger.info("=== Creating Simple Test HTML ===")
    
    simple_html = """<!DOCTYPE html>
<html>
<head>
    <title>Simple Tooltip Test</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        .tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
        }
        .node {
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div id="graph" style="width: 600px; height: 400px; border: 1px solid #ccc;"></div>
    <div id="tooltip" class="tooltip" style="display: none;"></div>
    
    <script>
        const data = [
            {id: "John", name: "John Smith", type: "PERSON"},
            {id: "Microsoft", name: "Microsoft Corp", type: "ORGANIZATION"},
            {id: "Seattle", name: "Seattle", type: "LOCATION"}
        ];
        
        const svg = d3.select('#graph')
            .append('svg')
            .attr('width', 600)
            .attr('height', 400);
        
        const nodes = svg.selectAll('circle')
            .data(data)
            .enter()
            .append('circle')
            .attr('cx', (d, i) => 100 + i * 200)
            .attr('cy', 200)
            .attr('r', 20)
            .attr('fill', 'steelblue')
            .attr('class', 'node');
        
        const tooltip = d3.select('#tooltip');
        
        nodes.on('mouseover', function(event, d) {
            tooltip.style('display', 'block')
                .html(`<strong>${d.name}</strong><br/>Type: ${d.type}`)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 10) + 'px');
        })
        .on('mouseout', function() {
            tooltip.style('display', 'none');
        });
    </script>
</body>
</html>"""
    
    # Write the simple test HTML
    test_file = Path("Test/simple_tooltip_test.html")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(simple_html)
    
    logger.info(f"✅ Simple test HTML created: {test_file}")
    logger.info("Open this file in a browser to test basic tooltip functionality")


async def main():
    """Run all tests."""
    logger.info("Starting hover functionality debug...")
    
    # Test 1: Simple HTML test
    await test_simple_html()
    
    # Test 2: Full graph hover test
    await test_hover_functionality()
    
    logger.info("=== Debug Summary ===")
    logger.info("Check the generated files:")
    logger.info("- Test/simple_tooltip_test.html (simple test)")
    logger.info("- Results/reports/hover_test_report.html (full test)")


if __name__ == "__main__":
    asyncio.run(main())
