#!/usr/bin/env python3
"""
Test script to verify Russian entity extraction with sample Russian text.
This script tests the Russian entity extraction with text that contains known Russian entities.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.entity_extraction_agent import EntityExtractionAgent
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from core.models import AnalysisRequest, DataType

async def test_russian_entity_extraction_with_sample():
    """Test Russian entity extraction with sample Russian text containing known entities."""
    print("🧪 Testing Russian Entity Extraction with Sample Text...")
    print("=" * 60)
    
    # Initialize agents
    entity_agent = EntityExtractionAgent()
    kg_agent = KnowledgeGraphAgent()
    
    # Sample Russian text with known entities
    russian_text = """
    Владимир Путин, президент России, встретился с Дмитрием Медведевым в Москве.
    Газпром и Сбербанк являются крупнейшими компаниями России.
    МГУ имени Ломоносова - ведущий университет страны.
    Искусственный интеллект и машинное обучение развиваются в России.
    Санкт-Петербург и Новосибирск - важные научные центры.
    """
    
    print(f"📝 Sample Russian text length: {len(russian_text)} characters")
    print(f"📝 Sample text: {russian_text[:200]}...")
    print()
    
    # Test entity extraction
    print("🔍 Extracting entities from Russian text...")
    extraction_result = await entity_agent.extract_entities_multilingual(russian_text, "ru")
    
    print(f"✅ Extraction completed!")
    print(f"📊 Found {extraction_result.get('count', 0)} entities")
    print(f"📋 Categories: {extraction_result.get('categories_found', [])}")
    print(f"📈 Statistics: {extraction_result.get('statistics', {})}")
    print()
    
    # Display extracted entities
    entities = extraction_result.get('entities', [])
    if entities:
        print("📋 Extracted Entities:")
        for i, entity in enumerate(entities, 1):
            print(f"  {i}. {entity.get('name', 'N/A')} ({entity.get('type', 'N/A')}) - Confidence: {entity.get('confidence', 0):.2f}")
    else:
        print("❌ No entities extracted")
    
    print()
    
    # Test knowledge graph processing
    print("🕸️ Testing knowledge graph processing...")
    try:
        # Create analysis request
        request = AnalysisRequest(
            content=russian_text,
            data_type=DataType.TEXT,
            language="ru",
            analysis_type="entity_extraction"
        )
        
        # Process with knowledge graph agent
        kg_result = await kg_agent.process_content(request)
        
        print(f"✅ Knowledge graph processing completed!")
        print(f"📊 Graph statistics: {kg_result.get('statistics', {})}")
        
        # Check if Russian entities are in the graph
        graph_data = kg_result.get('graph_data', {})
        nodes = graph_data.get('nodes', [])
        
        russian_nodes = [node for node in nodes if node.get('language') == 'ru']
        print(f"🇷🇺 Russian nodes in graph: {len(russian_nodes)}")
        
        if russian_nodes:
            print("📋 Russian entities in graph:")
            for node in russian_nodes[:5]:  # Show first 5
                print(f"  - {node.get('label', 'N/A')} ({node.get('type', 'N/A')})")
        
    except Exception as e:
        print(f"❌ Knowledge graph processing failed: {e}")
    
    print()
    print("=" * 60)
    print("✅ Russian entity extraction test completed!")

if __name__ == "__main__":
    asyncio.run(test_russian_entity_extraction_with_sample())
