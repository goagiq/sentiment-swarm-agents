#!/usr/bin/env python3
"""
Simple test script to test Russian entity extraction without enhanced extraction.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.entity_extraction_agent import EntityExtractionAgent


async def test_simple_russian_extraction():
    """Test simple Russian entity extraction."""
    
    # Sample Russian text
    russian_text = """
    Владимир Путин является президентом России. Москва является столицей России.
    Газпром является крупнейшей энергетической компанией. 
    МГУ имени Ломоносова является ведущим университетом.
    Искусственный интеллект и машинное обучение развиваются в России.
    """
    
    print("🧪 Testing simple Russian entity extraction...")
    print(f"📝 Sample text: {russian_text.strip()}")
    
    try:
        # Initialize entity extraction agent
        agent = EntityExtractionAgent()
        
        # Test direct Russian extraction
        print("\n🔍 Testing direct Russian extraction...")
        result = await agent._extract_russian_entities_enhanced(russian_text)
        
        print(f"\n✅ Extraction result:")
        print(f"  - Entities found: {len(result.get('entities', []))}")
        
        for i, entity in enumerate(result.get('entities', []), 1):
            name = entity.get('name', 'N/A')
            entity_type = entity.get('type', 'N/A')
            confidence = entity.get('confidence', 0)
            print(f"  {i}. {name} ({entity_type}) - Confidence: {confidence:.2f}")
        
        return len(result.get('entities', [])) > 0
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Starting simple Russian entity extraction test...")
    
    success = await test_simple_russian_extraction()
    
    if success:
        print("\n✅ Simple Russian entity extraction test PASSED!")
    else:
        print("\n❌ Simple Russian entity extraction test FAILED!")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())
