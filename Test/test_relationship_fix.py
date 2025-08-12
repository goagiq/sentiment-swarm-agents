"""
Test script to verify relationship addition to knowledge graph.
"""

import asyncio
import json
from src.agents.knowledge_graph_agent import KnowledgeGraphAgent


async def test_relationship_addition():
    """Test that relationships are properly added to the graph."""
    print("Testing relationship addition to knowledge graph...")
    
    # Create a knowledge graph agent
    agent = KnowledgeGraphAgent()
    
    # Test entities
    test_entities = [
        {"name": "中国", "type": "LOCATION", "confidence": 0.9},
        {"name": "科技", "type": "CONCEPT", "confidence": 0.8},
        {"name": "发展", "type": "CONCEPT", "confidence": 0.8}
    ]
    
    # Test relationships
    test_relationships = [
        {
            "source": "中国",
            "target": "科技",
            "relationship_type": "RELATED_TO",
            "confidence": 0.7,
            "description": "中国与科技相关"
        },
        {
            "source": "科技",
            "target": "发展",
            "relationship_type": "LEADS_TO",
            "confidence": 0.8,
            "description": "科技导致发展"
        }
    ]
    
    print(f"Initial graph state: {agent.graph.number_of_nodes()} nodes, {agent.graph.number_of_edges()} edges")
    
    # Add entities and relationships to the graph
    await agent._add_to_graph(test_entities, test_relationships, "test_request_001", "zh")
    
    print(f"After adding entities and relationships: {agent.graph.number_of_nodes()} nodes, {agent.graph.number_of_edges()} edges")
    
    # Check if relationships were added
    edges = list(agent.graph.edges(data=True))
    print(f"Edges in graph: {len(edges)}")
    
    for source, target, data in edges:
        print(f"Edge: {source} -> {target} (type: {data.get('relationship_type', 'unknown')})")
    
    # Test relationship mapping with Chinese content
    print("\nTesting relationship mapping with Chinese content...")
    chinese_text = "中国科技发展迅速，在人工智能领域取得了重大突破。"
    chinese_entities = [{"name": "中国", "type": "LOCATION"}, {"name": "科技", "type": "CONCEPT"}, {"name": "发展", "type": "CONCEPT"}]
    
    relationship_result = await agent.map_relationships(chinese_text, chinese_entities, "zh")
    print(f"Relationship mapping result: {relationship_result}")
    
    # Check if the relationships from mapping are valid
    relationship_json = relationship_result.get("content", [{}])[0].get("json", {})
    relationships = relationship_json.get("relationships", [])
    
    print(f"Extracted {len(relationships)} relationships from mapping")
    for rel in relationships:
        print(f"  {rel.get('source')} -> {rel.get('target')} ({rel.get('relationship_type')})")
    
    # Test adding these relationships to the graph
    if relationships:
        print("\nAdding mapped relationships to graph...")
        await agent._add_to_graph(chinese_entities, relationships, "test_request_002", "zh")
        
        print(f"Final graph state: {agent.graph.number_of_nodes()} nodes, {agent.graph.number_of_edges()} edges")
        
        # List all edges
        all_edges = list(agent.graph.edges(data=True))
        print(f"All edges in graph: {len(all_edges)}")
        for source, target, data in all_edges:
            print(f"  {source} -> {target} (type: {data.get('relationship_type', 'unknown')})")
    
    return agent


async def test_graph_persistence():
    """Test that the graph persists relationships correctly."""
    print("\n" + "="*50)
    print("Testing graph persistence...")
    
    # Create a new agent instance
    agent = KnowledgeGraphAgent()
    
    print(f"Loaded graph: {agent.graph.number_of_nodes()} nodes, {agent.graph.number_of_edges()} edges")
    
    # List all edges
    edges = list(agent.graph.edges(data=True))
    print(f"Edges in loaded graph: {len(edges)}")
    for source, target, data in edges:
        print(f"  {source} -> {target} (type: {data.get('relationship_type', 'unknown')})")


if __name__ == "__main__":
    async def main():
        # Test relationship addition
        agent = await test_relationship_addition()
        
        # Test graph persistence
        await test_graph_persistence()
    
    asyncio.run(main())
