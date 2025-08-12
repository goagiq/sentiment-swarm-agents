#!/usr/bin/env python3
"""
Test script to verify Russian entity extraction fix.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.knowledge_graph_agent import KnowledgeGraphAgent
from core.models import AnalysisRequest, DataType


async def test_russian_entity_extraction():
    """Test Russian entity extraction with the knowledge graph agent."""
    
    # Sample Russian text
    russian_text = """
    Владимир Путин является президентом России. Москва является столицей России.
    Газпром является крупнейшей энергетической компанией. 
    МГУ имени Ломоносова является ведущим университетом.
    Искусственный интеллект и машинное обучение развиваются в России.
    """
    
    print("🧪 Testing Russian entity extraction...")
    print("📝 Sample text:", russian_text.strip())
    
    try:
        # Initialize knowledge graph agent
        kg_agent = KnowledgeGraphAgent()
        
        # Test entity extraction
        print("\n🔍 Extracting entities...")
        entities_result = await kg_agent.extract_entities(
            russian_text, language="ru"
        )
        
        # Extract entities from result
        entities = []
        if entities_result and "content" in entities_result:
            for content in entities_result["content"]:
                if "json" in content and "entities" in content["json"]:
                    entities.extend(content["json"]["entities"])
        
        print(f"\n✅ Found {len(entities)} entities:")
        for i, entity in enumerate(entities, 1):
            entity_text = entity.get('text', 'N/A')
            entity_type = entity.get('type', 'N/A')
            confidence = entity.get('confidence', 0)
            print(f"  {i}. {entity_text} ({entity_type}) - "
                  f"Confidence: {confidence:.2f}")
        
        # Test with analysis request
        print("\n🔍 Testing with AnalysisRequest...")
        request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=russian_text,
            language="ru"
        )
        
        result = await kg_agent.process(request)
        
        print("\n✅ Analysis result:")
        print(f"  - Status: {result.status}")
        print(f"  - Processing time: {result.processing_time:.2f}s")
        
        if result.metadata and "statistics" in result.metadata:
            stats = result.metadata["statistics"]
            print(f"  - Entities found: {stats.get('entities_found', 0)}")
            print(f"  - Entity types: {stats.get('entity_types', {})}")
            print(f"  - Language stats: {stats.get('language_stats', {})}")
        
        return len(entities) > 0
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Starting Russian entity extraction test...")
    
    success = await test_russian_entity_extraction()
    
    if success:
        print("\n✅ Russian entity extraction test PASSED!")
    else:
        print("\n❌ Russian entity extraction test FAILED!")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())
