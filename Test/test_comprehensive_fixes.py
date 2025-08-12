#!/usr/bin/env python3
"""
Comprehensive test to verify edge creation and Russian PDF processing fixes.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.knowledge_graph_agent import KnowledgeGraphAgent
from agents.entity_extraction_agent import EntityExtractionAgent
from core.models import AnalysisRequest, DataType
from config.language_specific_config import should_use_enhanced_extraction, get_language_config


async def test_edge_creation_fix():
    """Test that edge creation is working properly."""
    print("\n🔍 Testing Edge Creation Fix...")
    
    # Initialize knowledge graph agent
    kg_agent = KnowledgeGraphAgent()
    
    # Get initial graph stats
    initial_nodes = kg_agent.graph.number_of_nodes()
    initial_edges = kg_agent.graph.number_of_edges()
    print(f"📊 Initial graph: {initial_nodes} nodes, {initial_edges} edges")
    
    # Test text with new entities
    test_text = """
    Sarah Wilson works for Innovation Labs. 
    Innovation Labs is located in Boston, Massachusetts. 
    David Chen collaborates with Sarah Wilson at Innovation Labs.
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
    new_entities = [e for e in entities_in_graph if any(name in e for name in ["Sarah", "Wilson", "Innovation", "Labs", "David", "Chen", "Boston"])]
    print(f"🏷️  New entities found: {new_entities}")
    
    # Check if new edges were created
    edges_in_graph = list(kg_agent.graph.edges())
    new_edges = [e for e in edges_in_graph if any(name in str(e) for name in ["Sarah", "Wilson", "Innovation", "Labs", "David", "Chen", "Boston"])]
    print(f"🔗 New edges found: {new_edges}")
    
    # Verify that edges were actually created
    if final_edges > initial_edges:
        print("✅ SUCCESS: Edge creation is working!")
        return True
    else:
        print("❌ FAILURE: No new edges were created")
        return False


async def test_russian_pdf_processing_fix():
    """Test that Russian PDF processing is working properly."""
    print("\n🔍 Testing Russian PDF Processing Fix...")
    
    # Initialize knowledge graph agent
    kg_agent = KnowledgeGraphAgent()
    
    # Get initial graph stats
    initial_nodes = kg_agent.graph.number_of_nodes()
    initial_edges = kg_agent.graph.number_of_edges()
    print(f"📊 Initial graph: {initial_nodes} nodes, {initial_edges} edges")
    
    # Test Russian text
    russian_text = """
    Анна Петрова работает в компании РосТех. 
    РосТех расположена в Москве. 
    Иван Сидоров сотрудничает с Анной Петровой в РосТех.
    """
    
    print(f"📝 Russian test text: {russian_text.strip()}")
    
    # Check language configuration
    print(f"🌍 Russian language config: {get_language_config('ru')}")
    print(f"🔧 Enhanced extraction enabled: {should_use_enhanced_extraction('ru')}")
    
    # Create analysis request
    request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=russian_text,
        language="ru"
    )
    
    # Process with knowledge graph agent
    print("🧠 Processing Russian text with knowledge graph agent...")
    result = await kg_agent.process(request)
    
    # Check results
    print(f"✅ Russian processing completed")
    print(f"📊 Entities extracted: {result.metadata.get('entities_extracted', 0)}")
    print(f"🔗 Relationships extracted: {result.metadata.get('relationships_extracted', 0)}")
    
    # Check final graph stats
    final_nodes = kg_agent.graph.number_of_nodes()
    final_edges = kg_agent.graph.number_of_edges()
    print(f"📈 Final graph: {final_nodes} nodes, {final_edges} edges")
    print(f"📈 Changes: +{final_nodes - initial_nodes} nodes, +{final_edges - initial_edges} edges")
    
    # Check if Russian entities were added
    entities_in_graph = list(kg_agent.graph.nodes())
    russian_entities = [e for e in entities_in_graph if any(name in e for name in ["Анна", "Петрова", "РосТех", "Иван", "Сидоров", "Москва"])]
    print(f"🏷️  Russian entities found: {russian_entities}")
    
    # Check if Russian edges were created
    edges_in_graph = list(kg_agent.graph.edges())
    russian_edges = [e for e in edges_in_graph if any(name in str(e) for name in ["Анна", "Петрова", "РосТех", "Иван", "Сидоров", "Москва"])]
    print(f"🔗 Russian edges found: {russian_edges}")
    
    # Verify that Russian processing worked
    if result.metadata.get('entities_extracted', 0) > 0:
        print("✅ SUCCESS: Russian PDF processing is working!")
        return True
    else:
        print("❌ FAILURE: No Russian entities were extracted")
        return False


async def test_entity_extraction_agent_fix():
    """Test that entity extraction agent is working properly."""
    print("\n🔍 Testing Entity Extraction Agent Fix...")
    
    # Initialize entity extraction agent
    entity_agent = EntityExtractionAgent()
    
    # Test Russian text
    russian_text = """
    Анна Петрова работает в компании РосТех. 
    РосТех расположена в Москве.
    """
    
    print(f"📝 Testing Russian entity extraction...")
    
    try:
        # Test if the enhanced Russian extraction method exists
        if hasattr(entity_agent, '_extract_russian_entities_enhanced'):
            print("✅ Enhanced Russian extraction method exists")
            result = await entity_agent._extract_russian_entities_enhanced(russian_text)
            print(f"📊 Enhanced extraction result: {result}")
            
            # Check if entities were extracted
            entities = result.get("entities", [])
            if entities:
                print(f"✅ SUCCESS: {len(entities)} Russian entities extracted")
                return True
            else:
                print("❌ FAILURE: No Russian entities extracted")
                return False
        else:
            print("❌ FAILURE: Enhanced Russian extraction method does not exist")
            return False
            
    except Exception as e:
        print(f"❌ ERROR in entity extraction: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Comprehensive Fixes Test")
    print("=" * 60)
    
    # Test 1: Edge creation fix
    edge_success = await test_edge_creation_fix()
    
    # Test 2: Russian PDF processing fix
    russian_success = await test_russian_pdf_processing_fix()
    
    # Test 3: Entity extraction agent fix
    entity_success = await test_entity_extraction_agent_fix()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print(f"🔗 Edge Creation: {'✅ PASSED' if edge_success else '❌ FAILED'}")
    print(f"🇷🇺 Russian PDF Processing: {'✅ PASSED' if russian_success else '❌ FAILED'}")
    print(f"🏷️  Entity Extraction Agent: {'✅ PASSED' if entity_success else '❌ FAILED'}")
    
    if edge_success and russian_success and entity_success:
        print("\n🎉 ALL TESTS PASSED! Both issues have been fixed successfully!")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
    
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
