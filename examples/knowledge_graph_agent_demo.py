"""
Demo script for Knowledge Graph Agent functionality.
"""

import asyncio
import json
from pathlib import Path

from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.core.models import AnalysisRequest, DataType


async def demo_knowledge_graph_agent():
    """Demonstrate knowledge graph agent capabilities."""
    
    print("🚀 Knowledge Graph Agent Demo")
    print("=" * 50)
    
    # Initialize the knowledge graph agent
    agent = KnowledgeGraphAgent()
    
    # Sample text content for demonstration
    sample_texts = [
        {
            "content": """
            Apple Inc. is a technology company founded by Steve Jobs and Steve Wozniak in 1976. 
            The company is headquartered in Cupertino, California and is known for its iPhone, 
            iPad, and Mac computers. Tim Cook is the current CEO of Apple, succeeding Steve Jobs 
            after his passing in 2011. Apple competes with companies like Microsoft, Google, and 
            Samsung in the technology market.
            """,
            "data_type": DataType.TEXT,
            "description": "Technology company information"
        },
        {
            "content": """
            The COVID-19 pandemic, caused by the SARS-CoV-2 virus, began in Wuhan, China in late 2019. 
            The World Health Organization declared it a global pandemic in March 2020. 
            Dr. Anthony Fauci, director of the National Institute of Allergy and Infectious Diseases, 
            became a key figure in the US response. Vaccines were developed by companies including 
            Pfizer, Moderna, and Johnson & Johnson.
            """,
            "data_type": DataType.TEXT,
            "description": "COVID-19 pandemic information"
        }
    ]
    
    # Process each sample text
    for i, sample in enumerate(sample_texts, 1):
        print(f"\n📝 Processing Sample {i}: {sample['description']}")
        print("-" * 40)
        
        # Create analysis request
        request = AnalysisRequest(
            data_type=sample["data_type"],
            content=sample["content"],
            language="en"
        )
        
        # Process the request
        result = await agent.process(request)
        
        print(f"✅ Processing completed!")
        print(f"   Entities extracted: {result.metadata.get('entities_extracted', 0)}")
        print(f"   Relationships mapped: {result.metadata.get('relationships_mapped', 0)}")
        print(f"   Graph nodes: {result.metadata.get('graph_nodes', 0)}")
        print(f"   Graph edges: {result.metadata.get('graph_edges', 0)}")
        
        # Show graph analysis
        graph_analysis = result.metadata.get('graph_analysis', {})
        if graph_analysis:
            print(f"   New entities: {graph_analysis.get('new_entities', 0)}")
            print(f"   New relationships: {graph_analysis.get('new_relationships', 0)}")
    
    # Demonstrate individual tools
    print(f"\n🔧 Testing Individual Tools")
    print("=" * 50)
    
    # Test entity extraction
    print("\n1. Testing Entity Extraction:")
    test_text = "Elon Musk is the CEO of Tesla and SpaceX. He also owns X (formerly Twitter)."
    entities_result = await agent.extract_entities(test_text)
    print(f"   Input: {test_text}")
    print(f"   Result: {json.dumps(entities_result, indent=2)}")
    
    # Test relationship mapping
    print("\n2. Testing Relationship Mapping:")
    entities = [
        {"name": "Elon Musk", "type": "person", "confidence": 0.9},
        {"name": "Tesla", "type": "company", "confidence": 0.9},
        {"name": "SpaceX", "type": "company", "confidence": 0.9},
        {"name": "X", "type": "company", "confidence": 0.8}
    ]
    relationships_result = await agent.map_relationships(test_text, entities)
    print(f"   Input entities: {entities}")
    print(f"   Result: {json.dumps(relationships_result, indent=2)}")
    
    # Test graph querying
    print("\n3. Testing Graph Querying:")
    query_result = await agent.query_knowledge_graph("technology companies")
    print(f"   Query: 'technology companies'")
    print(f"   Result: {json.dumps(query_result, indent=2)}")
    
    # Test community analysis
    print("\n4. Testing Community Analysis:")
    communities_result = await agent.analyze_graph_communities()
    print(f"   Result: {json.dumps(communities_result, indent=2)}")
    
    # Test entity context
    print("\n5. Testing Entity Context:")
    context_result = await agent.get_entity_context("Apple")
    print(f"   Entity: 'Apple'")
    print(f"   Result: {json.dumps(context_result, indent=2)}")
    
    # Generate graph report
    print("\n6. Generating Graph Report:")
    report_result = await agent.generate_graph_report()
    print(f"   Result: {json.dumps(report_result, indent=2)}")
    
    # Test path finding
    print("\n7. Testing Path Finding:")
    path_result = await agent.find_entity_paths("Steve Jobs", "Tim Cook")
    print(f"   Path from 'Steve Jobs' to 'Tim Cook'")
    print(f"   Result: {json.dumps(path_result, indent=2)}")
    
    # Show final graph statistics
    print(f"\n📊 Final Graph Statistics")
    print("=" * 50)
    stats = agent._get_graph_stats()
    print(f"   Total nodes: {stats['nodes']}")
    print(f"   Total edges: {stats['edges']}")
    print(f"   Graph density: {stats['density']:.4f}")
    print(f"   Average clustering: {stats['average_clustering']:.4f}")
    print(f"   Connected components: {stats['connected_components']}")
    
    print(f"\n✅ Knowledge Graph Agent Demo Completed!")
    print(f"   Graph saved to: {agent.graph_file}")


if __name__ == "__main__":
    asyncio.run(demo_knowledge_graph_agent())
