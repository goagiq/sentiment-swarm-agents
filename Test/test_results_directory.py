#!/usr/bin/env python3
"""
Test script to verify that the knowledge graph agent saves files to the 
Results directory by default.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.agents.knowledge_graph_agent import KnowledgeGraphAgent


async def test_results_directory():
    """Test that the knowledge graph agent saves files to Results directory."""
    
    print("Testing Knowledge Graph Agent Results Directory...")
    
    # Initialize the agent
    agent = KnowledgeGraphAgent()
    
    # Test the generate_graph_report method with default path
    print("Generating graph report with default path...")
    result = await agent.generate_graph_report()
    
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
    else:
        print(f"❌ HTML file does not exist: {html_path}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    asyncio.run(test_results_directory())
