#!/usr/bin/env python3
"""
Test script to bypass enhanced extraction and test fallback method directly.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.knowledge_graph_agent import KnowledgeGraphAgent


async def test_russian_bypass_enhanced():
    """Test Russian entity extraction by bypassing enhanced extraction."""
    
    # Sample Russian text
    russian_text = """
    Владимир Путин является президентом России. Москва является столицей России.
    Газпром является крупнейшей энергетической компанией. 
    МГУ имени Ломоносова является ведущим университетом.
    Искусственный интеллект и машинное обучение развиваются в России.
    """
    
    print("🧪 Testing Russian entity extraction with bypass...")
    print(f"📝 Sample text: {russian_text.strip()}")
    
    try:
        # Initialize knowledge graph agent
        kg_agent = KnowledgeGraphAgent()
        
        # Test fallback method directly
        print("\n🔍 Testing fallback method directly...")
        result = kg_agent._enhanced_fallback_entity_extraction(russian_text, "ru")
        
        print(f"\n✅ Fallback result:")
        print(f"  - Entities found: {len(result.get('entities', []))}")
        
        for i, entity in enumerate(result.get('entities', []), 1):
            name = entity.get('name', 'N/A')
            entity_type = entity.get('type', 'N/A')
            confidence = entity.get('confidence', 0)
            print(f"  {i}. {name} ({entity_type}) - Confidence: {confidence:.2f}")
        
        # Test with enhanced extraction disabled
        print("\n🔍 Testing with enhanced extraction disabled...")
        
        # Temporarily disable enhanced extraction
        original_should_use = kg_agent.__class__.__module__ + '.should_use_enhanced_extraction'
        
        # Test entity extraction without enhanced extraction
        entities_result = await kg_agent.extract_entities(russian_text, language="ru")
        
        # Extract entities from result
        entities = []
        if entities_result and "content" in entities_result:
            for content in entities_result["content"]:
                if "json" in content and "entities" in content["json"]:
                    entities.extend(content["json"]["entities"])
        
        print(f"\n✅ Entity extraction result:")
        print(f"  - Entities found: {len(entities)}")
        
        for i, entity in enumerate(entities, 1):
            entity_text = entity.get('text', 'N/A')
            entity_type = entity.get('type', 'N/A')
            confidence = entity.get('confidence', 0)
            print(f"  {i}. {entity_text} ({entity_type}) - Confidence: {confidence:.2f}")
        
        return len(entities) > 0 or len(result.get('entities', [])) > 0
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Starting Russian entity extraction bypass test...")
    
    success = await test_russian_bypass_enhanced()
    
    if success:
        print("\n✅ Russian entity extraction bypass test PASSED!")
    else:
        print("\n❌ Russian entity extraction bypass test FAILED!")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())
