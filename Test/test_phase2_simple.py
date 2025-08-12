"""
Simple test for Phase 2: Graph Storage Enhancement.
Directly tests language metadata storage without relying on entity extraction.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.knowledge_graph_agent import KnowledgeGraphAgent


async def test_direct_language_metadata():
    """Test language metadata storage directly by adding nodes manually."""
    print("🧪 Testing Direct Language Metadata Storage...")
    
    agent = KnowledgeGraphAgent()
    
    # Add nodes directly with language metadata
    test_nodes = [
        ("Beijing", {"type": "LOCATION", "language": "zh", "original_text": "北京"}),
        ("Washington D.C.", {"type": "LOCATION", "language": "en", "original_text": "Washington D.C."}),
        ("清华大学", {"type": "ORGANIZATION", "language": "zh", "original_text": "清华大学"}),
        ("Harvard University", {"type": "ORGANIZATION", "language": "en", "original_text": "Harvard University"}),
    ]
    
    # Add nodes to graph
    for node_name, attributes in test_nodes:
        agent.graph.add_node(node_name, **attributes)
        print(f"✅ Added node '{node_name}' with language: {attributes['language']}")
    
    # Add edges with language metadata
    test_edges = [
        ("Beijing", "清华大学", {"relationship_type": "LOCATED_IN", "language": "zh"}),
        ("Washington D.C.", "Harvard University", {"relationship_type": "LOCATED_IN", "language": "en"}),
    ]
    
    for source, target, attributes in test_edges:
        agent.graph.add_edge(source, target, **attributes)
        print(f"✅ Added edge '{source}' -> '{target}' with language: {attributes['language']}")
    
    # Check language metadata
    nodes_with_language = 0
    total_nodes = agent.graph.number_of_nodes()
    
    for node, attrs in agent.graph.nodes(data=True):
        if "language" in attrs:
            nodes_with_language += 1
            print(f"✅ Node '{node}' has language: {attrs['language']}")
            if "original_text" in attrs:
                print(f"   Original text: {attrs['original_text']}")
    
    edges_with_language = 0
    total_edges = agent.graph.number_of_edges()
    
    for source, target, attrs in agent.graph.edges(data=True):
        if "language" in attrs:
            edges_with_language += 1
            print(f"✅ Edge '{source}' -> '{target}' has language: {attrs['language']}")
    
    # Check language statistics
    stats = agent._get_graph_stats()
    print(f"\n📊 Graph stats: {stats}")
    
    if "languages" in stats:
        languages = stats['languages']
        print(f"🌐 Languages in graph: {list(languages.keys())}")
        print(f"   English nodes: {languages.get('en', {}).get('nodes', 0)}")
        print(f"   Chinese nodes: {languages.get('zh', {}).get('nodes', 0)}")
    
    print(f"\n📊 Summary:")
    print(f"   Nodes with language metadata: {nodes_with_language}/{total_nodes}")
    print(f"   Edges with language metadata: {edges_with_language}/{total_edges}")
    
    return (nodes_with_language == total_nodes and 
            edges_with_language == total_edges and
            "languages" in stats and
            len(stats["languages"]) >= 2)


async def test_graph_serialization():
    """Test that language metadata persists through graph serialization."""
    print("\n🧪 Testing Graph Serialization...")
    
    agent = KnowledgeGraphAgent()
    
    # Add test nodes with language metadata
    agent.graph.add_node("TestNode", type="TEST", language="zh", original_text="测试节点")
    agent.graph.add_node("TestNode2", type="TEST", language="en", original_text="Test Node")
    
    # Save graph
    print("💾 Saving graph...")
    agent._save_graph()
    
    # Create new agent instance to test loading
    new_agent = KnowledgeGraphAgent()
    
    # Check if language metadata is preserved
    loaded_nodes_with_language = 0
    for node, attrs in new_agent.graph.nodes(data=True):
        if "language" in attrs:
            loaded_nodes_with_language += 1
            print(f"✅ Loaded node '{node}' with language: {attrs['language']}")
    
    print(f"📊 Loaded graph: {new_agent.graph.number_of_nodes()} nodes")
    print(f"   Nodes with language metadata: {loaded_nodes_with_language}")
    
    return loaded_nodes_with_language == 2


async def main():
    """Run Phase 2 tests."""
    print("🚀 Starting Phase 2: Graph Storage Enhancement Tests...\n")
    
    tests = [
        ("Direct Language Metadata Storage", test_direct_language_metadata),
        ("Graph Serialization", test_graph_serialization),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with error: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*50)
    print("📊 PHASE 2 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 2 tests passed! Graph storage enhancement is working correctly.")
    else:
        print("⚠️  Some Phase 2 tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(main())
