#!/usr/bin/env python3
"""
Simple test to verify edge creation is working.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.knowledge_graph_agent import KnowledgeGraphAgent
from core.models import AnalysisRequest, DataType


async def test_simple_edge_creation():
    """Test simple edge creation with new entities."""
    print("🔍 Testing Simple Edge Creation...")
    
    # Initialize knowledge graph agent
    kg_agent = KnowledgeGraphAgent()
    
    # Get initial graph stats
    initial_nodes = kg_agent.graph.number_of_nodes()
    initial_edges = kg_agent.graph.number_of_edges()
    print(f"📊 Initial graph: {initial_nodes} nodes, {initial_edges} edges")
    
    # Test text with completely new entities
    test_text = """
    Alice Johnson works for TechCorp Inc. 
    TechCorp Inc. is located in Silicon Valley. 
    Bob Smith collaborates with Alice Johnson at TechCorp Inc.
    """
    
    print(f"📝 Test text: {test_text.strip()}")
    
    # Create analysis request
    request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=test_text,
        language="en"
    )
    
    # Process with knowledge graph agent
    print("🧠 Processing with knowledge graph agent...")
    result = await kg_agent.process(request)
    
    # Check results
    print(f"✅ Processing completed")
    print(f"📊 Entities extracted: {result.metadata.get('entities_extracted', 0)}")
    print(f"🔗 Relationships extracted: {result.metadata.get('relationships_extracted', 0)}")
    
    # Check final graph stats
    final_nodes = kg_agent.graph.number_of_nodes()
    final_edges = kg_agent.graph.number_of_edges()
    print(f"📈 Final graph: {final_nodes} nodes, {final_edges} edges")
    print(f"📈 Changes: +{final_nodes - initial_nodes} nodes, +{final_edges - initial_edges} edges")
    
    # Check if new entities were added
    entities_in_graph = list(kg_agent.graph.nodes())
    new_entities = [e for e in entities_in_graph if "Alice" in e or "Bob" in e or "TechCorp" in e or "Silicon" in e]
    print(f"🏷️  New entities found: {new_entities}")
    
    # Check if new edges were created
    edges_in_graph = list(kg_agent.graph.edges())
    new_edges = [e for e in edges_in_graph if any(name in str(e) for name in ["Alice", "Bob", "TechCorp", "Silicon"])]
    print(f"🔗 New edges found: {new_edges}")
    
    return result


async def main():
    """Run the test."""
    print("🚀 Starting Simple Edge Creation Test")
    print("=" * 50)
    
    await test_simple_edge_creation()
    
    print("\n" + "=" * 50)
    print("✅ Test completed")


if __name__ == "__main__":
    asyncio.run(main())
