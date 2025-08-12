#!/usr/bin/env python3
"""
Comprehensive test script to regenerate the knowledge graph report 
with enhanced visualization like the classical Chinese report.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.knowledge_graph_agent import KnowledgeGraphAgent
from core.models import AnalysisRequest, DataType

async def regenerate_enhanced_report():
    """Regenerate the knowledge graph report with enhanced visualization."""
    
    # Comprehensive test content with clear entities and relationships
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
    
    Tesla Inc. is an electric vehicle company led by Elon Musk.
    The company is headquartered in Austin, Texas.
    Tesla produces electric cars, solar panels, and energy storage systems.
    """
    
    print("=== Regenerating Enhanced Knowledge Graph Report ===")
    
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
            
            # Check current graph statistics before report generation
            print("\n--- Current Graph Statistics ---")
            graph_stats = kg_agent._get_graph_stats()
            print(f"Nodes: {graph_stats.get('nodes', 0)}")
            print(f"Edges: {graph_stats.get('edges', 0)}")
            print(f"Connected Components: {graph_stats.get('connected_components', 0)}")
            
            # Generate the enhanced graph report
            print("\n--- Generating Enhanced Graph Report ---")
            
            # Set output path for the report
            output_path = "Results/reports/enhanced_technology_report.md"
            
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
            
            # Generate enhanced HTML report with D3.js visualization
            print("\n--- Generating Enhanced HTML Report ---")
            
            html_output = "Results/reports/enhanced_technology_report.html"
            html_result = await kg_agent.generate_graph_report(
                output_path=html_output,
                target_language="en"
            )
            print(f"HTML report result: {html_result}")
            
            # Generate PNG report
            print("\n--- Generating PNG Report ---")
            
            png_output = "Results/reports/enhanced_technology_report.png"
            png_result = await kg_agent.generate_graph_report(
                output_path=png_output,
                target_language="en"
            )
            print(f"PNG report result: {png_result}")
            
            # Final graph statistics
            print("\n--- Final Graph Statistics ---")
            final_stats = kg_agent._get_graph_stats()
            print(f"Final Nodes: {final_stats.get('nodes', 0)}")
            print(f"Final Edges: {final_stats.get('edges', 0)}")
            print(f"Final Connected Components: {final_stats.get('connected_components', 0)}")
            
            if final_stats.get('edges', 0) > 0:
                print("✓ SUCCESS: Graph now has relationships!")
            else:
                print("⚠ WARNING: Graph still has no relationships")
                
                # Try to manually create some relationships for testing
                print("\n--- Attempting Manual Relationship Creation ---")
                try:
                    # Get current entities from the graph
                    entities = list(kg_agent.graph.nodes(data=True))
                    if entities:
                        print(f"Found {len(entities)} entities in graph")
                        
                        # Create some basic relationships
                        relationships_created = 0
                        for i, (node1, data1) in enumerate(entities[:5]):
                            for j, (node2, data2) in enumerate(entities[i+1:6]):
                                if node1 != node2:
                                    kg_agent.graph.add_edge(node1, node2, 
                                                          relationship_type="RELATED_TO",
                                                          confidence=0.5,
                                                          description=f"Entities in same context")
                                    relationships_created += 1
                        
                        print(f"Created {relationships_created} manual relationships")
                        kg_agent._save_graph()
                        
                        # Generate report again
                        print("\n--- Regenerating Report with Manual Relationships ---")
                        final_report_result = await kg_agent.generate_graph_report(
                            output_path="Results/reports/enhanced_technology_report_with_relationships.html",
                            target_language="en"
                        )
                        print(f"Final report result: {final_report_result}")
                        
                except Exception as e:
                    print(f"Manual relationship creation failed: {e}")
            
        else:
            print(f"⚠ Processing failed with status: {result.status}")
            if hasattr(result, 'error') and result.error:
                print(f"Error: {result.error}")
        
        print("\n=== Enhanced Report Regeneration Complete ===")
        
    except Exception as e:
        print(f"❌ Enhanced report regeneration failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(regenerate_enhanced_report())
