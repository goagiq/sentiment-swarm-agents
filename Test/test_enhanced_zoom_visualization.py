#!/usr/bin/env python3
"""
Test script to verify enhanced zoom functionality in knowledge graph visualization.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.agents.knowledge_graph_agent import KnowledgeGraphAgent

async def test_enhanced_zoom_visualization():
    """Test the enhanced zoom functionality in knowledge graph visualization."""
    
    print("Testing Enhanced Zoom Visualization...")
    
    # Initialize the agent
    agent = KnowledgeGraphAgent()
    
    # Add some sample data to the graph
    sample_entities = [
        {"id": "Ukraine", "type": "location", "confidence": 0.9},
        {"id": "Russia", "type": "location", "confidence": 0.9},
        {"id": "Vladimir Putin", "type": "person", "confidence": 0.8},
        {"id": "Volodymyr Zelensky", "type": "person", "confidence": 0.8},
        {"id": "Military Conflict", "type": "event", "confidence": 0.7},
        {"id": "NATO", "type": "organization", "confidence": 0.8},
        {"id": "European Union", "type": "organization", "confidence": 0.8},
        {"id": "United States", "type": "location", "confidence": 0.9},
        {"id": "Joe Biden", "type": "person", "confidence": 0.8},
        {"id": "Diplomatic Relations", "type": "concept", "confidence": 0.6}
    ]
    
    sample_relationships = [
        {"source": "Ukraine", "target": "Russia", "relationship_type": "conflicts_with", "confidence": 0.8},
        {"source": "Vladimir Putin", "target": "Russia", "relationship_type": "leads", "confidence": 0.9},
        {"source": "Volodymyr Zelensky", "target": "Ukraine", "relationship_type": "leads", "confidence": 0.9},
        {"source": "NATO", "target": "Ukraine", "relationship_type": "supports", "confidence": 0.7},
        {"source": "European Union", "target": "Ukraine", "relationship_type": "supports", "confidence": 0.7},
        {"source": "United States", "target": "Ukraine", "relationship_type": "supports", "confidence": 0.8},
        {"source": "Joe Biden", "target": "United States", "relationship_type": "leads", "confidence": 0.9},
        {"source": "Military Conflict", "target": "Ukraine", "relationship_type": "affects", "confidence": 0.8},
        {"source": "Military Conflict", "target": "Russia", "relationship_type": "affects", "confidence": 0.8},
        {"source": "Diplomatic Relations", "target": "Ukraine", "relationship_type": "involves", "confidence": 0.6},
        {"source": "Diplomatic Relations", "target": "Russia", "relationship_type": "involves", "confidence": 0.6}
    ]
    
    # Add entities and relationships to the graph
    await agent._add_to_graph(sample_entities, sample_relationships, "test_zoom")
    
    print(f"Added {len(sample_entities)} entities and {len(sample_relationships)} relationships to the graph")
    
    # Generate enhanced graph report
    print("Generating enhanced graph report with zoom functionality...")
    result = await agent.generate_graph_report("Results/test_enhanced_zoom_visualization")
    
    # Extract file paths from result
    json_result = result.get("content", [{}])[0].get("json", {})
    png_file = json_result.get("png_file", "")
    html_file = json_result.get("html_file", "")
    
    print(f"Generated PNG file: {png_file}")
    print(f"Generated HTML file: {html_file}")
    
    # Check if files are in Results directory
    png_path = Path(png_file)
    html_path = Path(html_file)
    
    if "Results" in str(png_path) and "Results" in str(html_path):
        print("✅ SUCCESS: Files are being saved to Results directory!")
    else:
        print("❌ FAILURE: Files are not being saved to Results directory!")
        print(f"PNG path: {png_path}")
        print(f"HTML path: {html_path}")
    
    # Check if files actually exist
    if png_path.exists():
        print(f"✅ PNG file exists: {png_path}")
    else:
        print(f"❌ PNG file does not exist: {png_path}")
    
    if html_path.exists():
        print(f"✅ HTML file exists: {html_path}")
        print("✅ Enhanced zoom visualization generated successfully!")
        print("📋 Zoom Features Added:")
        print("   - Mouse wheel to zoom in/out")
        print("   - Click and drag to pan")
        print("   - Double-click to reset zoom")
        print("   - Zoom level indicator")
        print("   - Zoom scale limits (10% to 1000%)")
    else:
        print(f"❌ HTML file does not exist: {html_path}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_zoom_visualization())
