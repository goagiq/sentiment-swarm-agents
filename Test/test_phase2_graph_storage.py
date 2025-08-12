"""
Test file for Phase 2: Graph Storage Enhancement.
Tests language metadata storage in the knowledge graph.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.core.models import AnalysisRequest, DataType


async def test_language_metadata_storage():
    """Test that language metadata is properly stored in graph nodes and edges."""
    print("🧪 Testing Language Metadata Storage...")
    
    # Initialize the knowledge graph agent
    agent = KnowledgeGraphAgent()
    
    # Test Chinese content
    chinese_text = """
    人工智能技术正在快速发展。北京是中国的首都，有很多著名的大学和研究机构。
    清华大学和北京大学都是世界知名的高等学府。人工智能在医疗、教育、交通等领域都有广泛应用。
    """
    
    # Create analysis request
    request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=chinese_text,
        language="zh"
    )
    
    print(f"📝 Processing Chinese text...")
    
    # Process the request
    result = await agent.process(request)
    
    print(f"🔍 Detected language: {request.language}")
    print(f"📊 Graph stats: {agent.metadata['graph_stats']}")
    
    # Check if language metadata is stored in nodes
    nodes_with_language = 0
    total_nodes = agent.graph.number_of_nodes()
    
    for node, attrs in agent.graph.nodes(data=True):
        if "language" in attrs:
            nodes_with_language += 1
            print(f"✅ Node '{node}' has language: {attrs['language']}")
            if "original_text" in attrs:
                print(f"   Original text: {attrs['original_text'][:50]}...")
    
    # Check if language metadata is stored in edges
    edges_with_language = 0
    total_edges = agent.graph.number_of_edges()
    
    for source, target, attrs in agent.graph.edges(data=True):
        if "language" in attrs:
            edges_with_language += 1
            print(f"✅ Edge '{source}' -> '{target}' has language: {attrs['language']}")
    
    print(f"\n📊 Language Metadata Summary:")
    print(f"   Nodes with language metadata: {nodes_with_language}/{total_nodes}")
    print(f"   Edges with language metadata: {edges_with_language}/{total_edges}")
    
    # Verify language statistics are included
    stats = agent.metadata['graph_stats']
    if "languages" in stats:
        print(f"   Language distribution: {stats['languages']}")
        print(f"   Total languages: {stats['total_languages']}")
    
    return nodes_with_language > 0 and edges_with_language > 0


async def test_multilingual_graph():
    """Test adding content in multiple languages to the same graph."""
    print("\n🧪 Testing Multilingual Graph...")
    
    agent = KnowledgeGraphAgent()
    
    # Add English content
    english_text = """
    Artificial intelligence technology is developing rapidly. Washington D.C. is the capital of the United States.
    Harvard University and MIT are world-renowned institutions of higher learning. 
    AI has widespread applications in healthcare, education, transportation, and other fields.
    """
    
    english_request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=english_text,
        language="en"
    )
    
    print("📝 Adding English content...")
    await agent.process(english_request)
    
    # Add Chinese content
    chinese_text = """
    人工智能技术正在快速发展。北京是中国的首都，有很多著名的大学和研究机构。
    清华大学和北京大学都是世界知名的高等学府。
    """
    
    chinese_request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=chinese_text,
        language="zh"
    )
    
    print("📝 Adding Chinese content...")
    await agent.process(chinese_request)
    
    # Check language distribution
    stats = agent.metadata['graph_stats']
    print(f"📊 Final graph stats: {stats}")
    
    if "languages" in stats:
        languages = stats['languages']
        print(f"🌐 Languages in graph: {list(languages.keys())}")
        
        # Check if both languages are present
        has_english = "en" in languages
        has_chinese = "zh" in languages
        
        print(f"   English nodes: {languages.get('en', {}).get('nodes', 0)}")
        print(f"   Chinese nodes: {languages.get('zh', {}).get('nodes', 0)}")
        
        return has_english and has_chinese
    else:
        print("❌ No language statistics found")
        return False


async def test_graph_serialization():
    """Test that language metadata persists through graph serialization."""
    print("\n🧪 Testing Graph Serialization...")
    
    agent = KnowledgeGraphAgent()
    
    # Add some content
    test_text = "This is a test. 这是一个测试。"
    request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=test_text,
        language="auto"
    )
    
    print("📝 Adding test content...")
    await agent.process(request)
    
    # Check initial state
    initial_nodes = agent.graph.number_of_nodes()
    initial_edges = agent.graph.number_of_edges()
    print(f"📊 Initial graph: {initial_nodes} nodes, {initial_edges} edges")
    
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
    
    loaded_edges_with_language = 0
    for source, target, attrs in new_agent.graph.edges(data=True):
        if "language" in attrs:
            loaded_edges_with_language += 1
    
    print(f"📊 Loaded graph: {new_agent.graph.number_of_nodes()} nodes, {new_agent.graph.number_of_edges()} edges")
    print(f"   Nodes with language metadata: {loaded_nodes_with_language}")
    print(f"   Edges with language metadata: {loaded_edges_with_language}")
    
    return (new_agent.graph.number_of_nodes() == initial_nodes and 
            new_agent.graph.number_of_edges() == initial_edges and
            loaded_nodes_with_language > 0)


async def main():
    """Run all Phase 2 tests."""
    print("🚀 Starting Phase 2: Graph Storage Enhancement Tests...\n")
    
    tests = [
        ("Language Metadata Storage", test_language_metadata_storage),
        ("Multilingual Graph", test_multilingual_graph),
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
