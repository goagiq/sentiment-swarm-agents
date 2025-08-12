#!/usr/bin/env python3
"""
Comprehensive test script to regenerate the knowledge graph report 
using the improved entity extraction and relationship mapping methods.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.knowledge_graph_agent import KnowledgeGraphAgent
from core.models import AnalysisRequest, DataType
from loguru import logger

async def regenerate_graph_report():
    """Regenerate the knowledge graph report using improved methods."""
    
    # Sample text content for testing - using more comprehensive content
    test_content = """
    Apple Inc. is a technology company based in Cupertino, California. 
    Steve Jobs and Steve Wozniak founded Apple in 1976. 
    The company is known for products like the iPhone, iPad, and MacBook. 
    Tim Cook is the current CEO of Apple. 
    The company has offices in San Francisco and New York.
    
    Microsoft Corporation is another major technology company headquartered in Redmond, Washington.
    Bill Gates and Paul Allen founded Microsoft in 1975.
    Satya Nadella is the current CEO of Microsoft.
    The company is known for Windows, Office, and Azure cloud services.
    
    Google LLC is a technology company based in Mountain View, California.
    Larry Page and Sergey Brin founded Google in 1998.
    Sundar Pichai is the current CEO of Google.
    The company is known for search, Android, and cloud computing services.
    """
    
    print("=== Regenerating Knowledge Graph Report with Improved Methods ===")
    
    try:
        # Initialize the KnowledgeGraphAgent
        print("Initializing KnowledgeGraphAgent...")
        kg_agent = KnowledgeGraphAgent()
        print("✓ KnowledgeGraphAgent initialized successfully")
        
        # Create analysis request
        request = AnalysisRequest(
            content=test_content,
            data_type=DataType.TEXT,
            language="en"
        )
        
        print(f"\nProcessing content with {len(test_content)} characters...")
        
        # Process the content
        result = await kg_agent.process(request)
        
        print(f"Processing completed with status: {result.status}")
        
        if result.status == "completed":
            print("✓ Content processing successful")
            
            # Generate the graph report
            print("\n--- Generating Graph Report ---")
            
            # Set output path for the report
            output_path = "Results/reports/regenerated_knowledge_graph_report.md"
            
            report_result = await kg_agent.generate_graph_report(
                output_path=output_path,
                target_language="en"
            )
            
            print(f"Report generation result: {report_result}")
            
            # Check if the report file was created
            report_file = Path(output_path)
            if report_file.exists():
                print(f"✓ Graph report generated successfully: {output_path}")
                
                # Read and display a summary of the report
                with open(report_file, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                
                print(f"\nReport file size: {len(report_content)} characters")
                
                # Show key statistics from the report
                lines = report_content.split('\n')
                for line in lines:
                    if any(keyword in line.lower() for keyword in [
                        'total nodes', 'total edges', 'connected components',
                        'node types', 'language distribution'
                    ]):
                        print(f"  {line.strip()}")
                
            else:
                print(f"⚠ Report file not found at: {output_path}")
            
            # Also generate PNG and HTML reports
            print("\n--- Generating Additional Report Formats ---")
            
            # Generate PNG report
            png_output = "Results/reports/regenerated_knowledge_graph_report.png"
            png_result = await kg_agent.generate_graph_report(
                output_path=png_output,
                target_language="en"
            )
            print(f"PNG report result: {png_result}")
            
            # Generate HTML report
            html_output = "Results/reports/regenerated_knowledge_graph_report.html"
            html_result = await kg_agent.generate_graph_report(
                output_path=html_output,
                target_language="en"
            )
            print(f"HTML report result: {html_result}")
            
            # Check current graph statistics
            print("\n--- Current Graph Statistics ---")
            graph_stats = kg_agent._get_graph_stats()
            print(f"Nodes: {graph_stats.get('nodes', 0)}")
            print(f"Edges: {graph_stats.get('edges', 0)}")
            print(f"Connected Components: {graph_stats.get('connected_components', 0)}")
            print(f"Node Types: {graph_stats.get('node_types', {})}")
            
        else:
            print(f"⚠ Processing failed with status: {result.status}")
            if hasattr(result, 'error') and result.error:
                print(f"Error: {result.error}")
        
        print("\n=== Report Regeneration Complete ===")
        
    except Exception as e:
        print(f"❌ Report regeneration failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(regenerate_graph_report())
