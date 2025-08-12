#!/usr/bin/env python3
"""
Generate knowledge graph report for Russian content.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.knowledge_graph_agent import KnowledgeGraphAgent


async def main():
    """Generate knowledge graph report for Russian content."""
    print("Generating knowledge graph report for Russian content...")
    
    # Initialize knowledge graph agent
    kg_agent = KnowledgeGraphAgent()
    
    # Generate report
    report_path = "Results/russian_oliver_knowledge_graph_report.html"
    
    try:
        result = await kg_agent.generate_graph_report(
            output_path=report_path,
            target_language="ru"
        )
        
        if result.get("success", False):
            print(f"✓ Knowledge graph report generated successfully!")
            print(f"  - Report saved to: {report_path}")
            
            # Print summary information
            entities = result.get('entities_extracted', 0)
            relationships = result.get('relationships_mapped', 0)
            nodes = result.get('graph_nodes', 0)
            edges = result.get('graph_edges', 0)
            
            print(f"  - Entities extracted: {entities}")
            print(f"  - Relationships mapped: {relationships}")
            print(f"  - Graph nodes: {nodes}")
            print(f"  - Graph edges: {edges}")
            
            # Check if files were created
            html_file = Path(report_path)
            png_file = Path(report_path.replace('.html', '.png'))
            md_file = Path(report_path.replace('.html', '.md'))
            
            if html_file.exists():
                print(f"  - HTML report: {html_file}")
            if png_file.exists():
                print(f"  - PNG visualization: {png_file}")
            if md_file.exists():
                print(f"  - Markdown report: {md_file}")
                
        else:
            print("✗ Failed to generate knowledge graph report")
            error_msg = result.get('error', 'Unknown error')
            print(f"  Error: {error_msg}")
            
    except Exception as e:
        print(f"✗ Report generation failed with exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
