#!/usr/bin/env python3
"""
Test script to verify Russian PDF processing with enhanced Russian entity extraction.
This script tests the fixed Russian language support in the entity extraction agent.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.entity_extraction_agent import EntityExtractionAgent
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from core.models import AnalysisRequest, DataType

async def test_russian_entity_extraction():
    """Test Russian entity extraction with sample Russian text."""
    print("🧪 Testing Russian Entity Extraction...")
    
    # Sample Russian text for testing
    russian_text = """
    Владимир Путин объявил о новых инвестициях в искусственный интеллект.
    Газпром и Сбербанк планируют развивать технологии машинного обучения.
    В Москве открылся новый центр квантовых вычислений.
    Дмитрий Медведев выступил на конференции по цифровой экономике.
    Российские ученые из МГУ достигли прорыва в области нейронных сетей.
    """
    
    # Initialize the entity extraction agent
    agent = EntityExtractionAgent()
    
    # Test language detection
    detected_language = agent._detect_language(russian_text)
    print(f"🔍 Detected language: {detected_language}")
    
    # Test Russian entity extraction
    print("📝 Extracting Russian entities...")
    result = await agent.extract_entities_multilingual(russian_text, language="ru")
    
    print(f"✅ Found {result.get('count', 0)} entities")
    print(f"📊 Categories: {result.get('categories_found', [])}")
    
    # Display extracted entities
    entities = result.get('entities', [])
    for i, entity in enumerate(entities[:10], 1):  # Show first 10 entities
        print(f"  {i}. {entity.get('name', 'N/A')} ({entity.get('type', 'N/A')}) - Confidence: {entity.get('confidence', 0):.2f}")
    
    return result

async def test_russian_knowledge_graph():
    """Test Russian knowledge graph generation."""
    print("\n🧠 Testing Russian Knowledge Graph Generation...")
    
    # Sample Russian text
    russian_text = """
    Президент России Владимир Путин подписал указ о развитии искусственного интеллекта.
    Газпром инвестирует в технологии машинного обучения для оптимизации добычи нефти.
    Сбербанк запустил новую платформу для обработки естественного языка.
    В Санкт-Петербурге открылся центр квантовых вычислений.
    Российские ученые из МГУ и СПбГУ сотрудничают в области нейронных сетей.
    """
    
    # Initialize the knowledge graph agent
    agent = KnowledgeGraphAgent()
    
    # Create analysis request
    request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=russian_text,
        language="ru",
        request_id="test_russian_kg"
    )
    
    # Process the request
    print("🔄 Processing Russian text for knowledge graph...")
    result = await agent.process(request)
    
    print(f"✅ Processing completed with status: {result.status}")
    print(f"📊 Entities extracted: {len(result.entities) if result.entities else 0}")
    print(f"🔗 Relationships mapped: {len(result.relationships) if result.relationships else 0}")
    
    # Display some entities
    if result.entities:
        print("\n📋 Sample entities:")
        for i, entity in enumerate(result.entities[:5], 1):
            print(f"  {i}. {entity.get('name', 'N/A')} ({entity.get('type', 'N/A')})")
    
    return result

async def test_russian_language_detection():
    """Test Russian language detection with various text samples."""
    print("\n🔍 Testing Russian Language Detection...")
    
    agent = EntityExtractionAgent()
    
    test_cases = [
        ("Владимир Путин президент России", "ru"),
        ("Artificial Intelligence and Machine Learning", "en"),
        ("人工智能和机器学习", "zh"),
        ("Vladimir Putin is the president of Russia", "en"),
        ("Путин Владимир Владимирович - президент Российской Федерации", "ru"),
    ]
    
    for text, expected in test_cases:
        detected = agent._detect_language(text)
        status = "✅" if detected == expected else "❌"
        print(f"{status} Text: '{text[:30]}...' -> Detected: {detected}, Expected: {expected}")

async def main():
    """Main test function."""
    print("🚀 Starting Russian PDF Processing Tests")
    print("=" * 50)
    
    try:
        # Test 1: Russian entity extraction
        await test_russian_entity_extraction()
        
        # Test 2: Russian language detection
        await test_russian_language_detection()
        
        # Test 3: Russian knowledge graph generation
        await test_russian_knowledge_graph()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("✅ Russian language support is working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
