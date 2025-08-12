#!/usr/bin/env python3
"""
Test script to diagnose and fix edge creation and Russian PDF processing issues.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.knowledge_graph_agent import KnowledgeGraphAgent
from agents.entity_extraction_agent import EntityExtractionAgent
from core.models import AnalysisRequest, DataType
from config.language_specific_config import should_use_enhanced_extraction, get_language_config


async def test_edge_creation_issue():
    """Test edge creation to identify the issue."""
    print("\n🔍 Testing Edge Creation Issue...")
    
    # Initialize knowledge graph agent
    kg_agent = KnowledgeGraphAgent()
    
    # Test text with entities
    test_text = """
    John Smith works for Microsoft Corporation. 
    Microsoft is located in Seattle, Washington. 
    John Smith lives in Seattle and collaborates with Jane Doe.
    Jane Doe is the CEO of Apple Inc.
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
    
    # Check graph stats
    graph_stats = kg_agent._get_graph_stats()
    print(f"📈 Graph stats - Nodes: {graph_stats.get('nodes', 0)}, Edges: {graph_stats.get('edges', 0)}")
    
    # Check if entities were added to graph
    entities_in_graph = list(kg_agent.graph.nodes())
    print(f"🏷️  Entities in graph: {len(entities_in_graph)}")
    if entities_in_graph:
        print(f"   Sample entities: {entities_in_graph[:5]}")
    
    # Check if edges were created
    edges_in_graph = list(kg_agent.graph.edges())
    print(f"🔗 Edges in graph: {len(edges_in_graph)}")
    if edges_in_graph:
        print(f"   Sample edges: {edges_in_graph[:5]}")
    
    return result


async def test_russian_pdf_processing():
    """Test Russian PDF processing to identify the issue."""
    print("\n🔍 Testing Russian PDF Processing...")
    
    # Initialize knowledge graph agent
    kg_agent = KnowledgeGraphAgent()
    
    # Test Russian text
    russian_text = """
    Владимир Путин является президентом России. 
    Россия расположена в Европе и Азии. 
    Газпром является крупнейшей энергетической компанией России.
    Дмитрий Медведев работает в правительстве России.
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
    
    # Check graph stats
    graph_stats = kg_agent._get_graph_stats()
    print(f"📈 Graph stats - Nodes: {graph_stats.get('nodes', 0)}, Edges: {graph_stats.get('edges', 0)}")
    
    return result


async def test_entity_extraction_agent():
    """Test entity extraction agent for Russian."""
    print("\n🔍 Testing Entity Extraction Agent for Russian...")
    
    # Initialize entity extraction agent
    entity_agent = EntityExtractionAgent()
    
    # Test Russian text
    russian_text = """
    Владимир Путин является президентом России. 
    Россия расположена в Европе и Азии. 
    Газпром является крупнейшей энергетической компанией России.
    """
    
    print(f"📝 Testing Russian entity extraction...")
    
    try:
        # Test if the enhanced Russian extraction method exists
        if hasattr(entity_agent, '_extract_russian_entities_enhanced'):
            print("✅ Enhanced Russian extraction method exists")
            result = await entity_agent._extract_russian_entities_enhanced(russian_text)
            print(f"📊 Enhanced extraction result: {result}")
        else:
            print("❌ Enhanced Russian extraction method does not exist")
            
        # Test regular extraction
        request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=russian_text,
            language="ru"
        )
        result = await entity_agent.process(request)
        print(f"📊 Regular extraction result: {result.metadata.get('entities_extracted', 0)} entities")
        
    except Exception as e:
        print(f"❌ Error in entity extraction: {e}")
    
    return entity_agent


async def test_relationship_mapping():
    """Test relationship mapping specifically."""
    print("\n🔍 Testing Relationship Mapping...")
    
    # Initialize knowledge graph agent
    kg_agent = KnowledgeGraphAgent()
    
    # Test entities
    test_entities = [
        {"name": "John Smith", "type": "PERSON", "confidence": 0.8},
        {"name": "Microsoft", "type": "ORGANIZATION", "confidence": 0.9},
        {"name": "Seattle", "type": "LOCATION", "confidence": 0.7},
        {"name": "Jane Doe", "type": "PERSON", "confidence": 0.8}
    ]
    
    test_text = "John Smith works for Microsoft in Seattle. Jane Doe collaborates with John Smith."
    
    print(f"📝 Test text: {test_text}")
    print(f"🏷️  Test entities: {[e['name'] for e in test_entities]}")
    
    # Test relationship mapping
    try:
        result = await kg_agent.map_relationships(test_text, test_entities, "en")
        print(f"🔗 Relationship mapping result: {result}")
        
        # Check if relationships were created
        relationships = result.get("content", [{}])[0].get("json", {}).get("relationships", [])
        print(f"📊 Relationships created: {len(relationships)}")
        if relationships:
            print(f"   Sample relationships: {relationships[:3]}")
        
    except Exception as e:
        print(f"❌ Error in relationship mapping: {e}")
    
    return result


async def main():
    """Run all tests."""
    print("🚀 Starting Edge Creation and Russian PDF Processing Tests")
    print("=" * 60)
    
    # Test 1: Edge creation issue
    await test_edge_creation_issue()
    
    # Test 2: Russian PDF processing
    await test_russian_pdf_processing()
    
    # Test 3: Entity extraction agent
    await test_entity_extraction_agent()
    
    # Test 4: Relationship mapping
    await test_relationship_mapping()
    
    print("\n" + "=" * 60)
    print("✅ All tests completed")


if __name__ == "__main__":
    asyncio.run(main())
