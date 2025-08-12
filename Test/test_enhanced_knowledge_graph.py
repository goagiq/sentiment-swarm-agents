#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.enhanced_knowledge_graph_agent import EnhancedKnowledgeGraphAgent
from src.core.models import AnalysisRequest, DataType

async def test_enhanced_knowledge_graph():
    """Test the enhanced knowledge graph agent with Classical Chinese content."""
    
    # Read the extracted text
    try:
        with open("extracted_text.txt", "r", encoding="utf-8") as f:
            text_content = f.read()
        print(f"Loaded text content: {len(text_content)} characters")
    except FileNotFoundError:
        print("extracted_text.txt not found. Please run extract_pdf_text.py first.")
        return
    
    # Initialize the enhanced knowledge graph agent
    agent = EnhancedKnowledgeGraphAgent()
    
    # Create analysis request
    request = AnalysisRequest(
        id="classical_chinese_test",
        content=text_content,
        data_type=DataType.TEXT
    )
    
    print("Processing Classical Chinese content with enhanced knowledge graph agent...")
    
    # Process the content
    result = await agent.process(request)
    
    print(f"\nProcessing completed!")
    print(f"Status: {result.status}")
    print(f"Processing time: {result.processing_time:.2f} seconds")
    print(f"Content: {result.content}")
    
    # Print metadata
    print(f"\nMetadata:")
    for key, value in result.metadata.items():
        print(f"  {key}: {value}")
    
    # Generate graph report
    print("\nGenerating enhanced graph report...")
    report_result = await agent.generate_graph_report("enhanced_classical_chinese_report.html")
    print(f"Report generated: {report_result}")
    
    # Get graph statistics
    stats = agent._get_graph_stats()
    print(f"\nEnhanced Graph Statistics:")
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Density: {stats['density']:.4f}")
    print(f"  Connected Components: {stats['connected_components']}")
    
    # Show some sample entities and relationships
    print(f"\nSample Entities:")
    entities = list(agent.graph.nodes())[:10]
    for entity in entities:
        node_data = agent.graph.nodes[entity]
        print(f"  {entity} ({node_data.get('type', 'unknown')}) - {node_data.get('domain', 'general')}")
    
    print(f"\nSample Relationships:")
    edges = list(agent.graph.edges(data=True))[:10]
    for source, target, data in edges:
        print(f"  {source} --[{data.get('relationship_type', 'related')}]--> {target}")

if __name__ == "__main__":
    asyncio.run(test_enhanced_knowledge_graph())
