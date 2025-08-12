#!/usr/bin/env python3
"""
Test script to check Russian PDF processing with knowledge graph agent.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.config.language_specific_regex_config import detect_language_from_text


async def test_russian_entity_extraction():
    """Test Russian entity extraction with knowledge graph agent."""
    print("=== Testing Russian Entity Extraction ===")
    
    # Initialize the knowledge graph agent
    agent = KnowledgeGraphAgent()
    
    # Russian test text
    russian_text = """
    Иван Петров работает в компании ООО "Технологии будущего" в городе Москва.
    Он изучает искусственный интеллект и машинное обучение.
    Доктор Сидоров преподает в Московском университете.
    Компания специализируется на разработке программного обеспечения.
    """
    
    print(f"Test text: {russian_text.strip()}")
    
    # Detect language
    detected_language = detect_language_from_text(russian_text)
    print(f"Detected language: {detected_language}")
    
    # Extract entities
    try:
        result = await agent.extract_entities(russian_text, detected_language)
        print(f"Entity extraction result: {result}")
        
        if "content" in result and result["content"]:
            entities = result["content"][0].get("json", {}).get("entities", [])
            print(f"Extracted {len(entities)} entities:")
            for entity in entities:
                print(f"  - {entity.get('text', 'N/A')} ({entity.get('type', 'N/A')})")
        else:
            print("No entities extracted")
            
    except Exception as e:
        print(f"Entity extraction failed: {e}")
        import traceback
        traceback.print_exc()


async def test_russian_relationship_mapping():
    """Test Russian relationship mapping."""
    print("\n=== Testing Russian Relationship Mapping ===")
    
    agent = KnowledgeGraphAgent()
    
    # Sample entities
    entities = [
        {"text": "Иван Петров", "type": "PERSON"},
        {"text": "ООО Технологии будущего", "type": "ORGANIZATION"},
        {"text": "Москва", "type": "LOCATION"},
        {"text": "искусственный интеллект", "type": "CONCEPT"}
    ]
    
    russian_text = """
    Иван Петров работает в компании ООО "Технологии будущего" в городе Москва.
    Он изучает искусственный интеллект и машинное обучение.
    """
    
    try:
        result = await agent.map_relationships(russian_text, entities, "ru")
        print(f"Relationship mapping result: {result}")
        
    except Exception as e:
        print(f"Relationship mapping failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests."""
    print("Testing Russian PDF Processing with Knowledge Graph Agent")
    print("=" * 60)
    
    try:
        await test_russian_entity_extraction()
        await test_russian_relationship_mapping()
        
        print("\n=== Summary ===")
        print("✓ All tests completed")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
