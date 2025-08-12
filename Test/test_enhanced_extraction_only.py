#!/usr/bin/env python3
"""
Test script to verify enhanced entity extraction works.
"""

import sys
import os
import asyncio

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

async def test_enhanced_extraction():
    """Test the enhanced entity extraction."""
    print("🧪 Testing Enhanced Entity Extraction")
    print("=" * 50)
    
    try:
        from src.agents.entity_extraction_agent import EntityExtractionAgent
        
        # Create agent
        agent = EntityExtractionAgent()
        
        # Test text with clear entities
        test_text = """
        Artificial Intelligence (AI) is transforming the world. 
        Companies like Google, Microsoft, and OpenAI are leading the development.
        Machine learning algorithms are being used in healthcare, finance, and education.
        Deep learning models like GPT-4 and BERT have revolutionized natural language processing.
        """
        
        print(f"Test text: {test_text.strip()}")
        
        # Test enhanced extraction
        result = await agent.extract_entities_enhanced(test_text)
        
        print(f"\nEnhanced extraction result:")
        print(f"  Total entities: {result.get('count', 0)}")
        print(f"  Categories found: {result.get('categories_found', [])}")
        
        if 'entities' in result and result['entities']:
            print(f"\nExtracted entities:")
            for entity in result['entities']:
                name = entity.get('name', 'N/A')
                entity_type = entity.get('type', 'N/A')
                confidence = entity.get('confidence', 'N/A')
                method = entity.get('extraction_method', 'N/A')
                print(f"  - {name} ({entity_type}) - confidence: {confidence} - method: {method}")
                
                # Check for overly long entities
                if len(name) > 50:
                    print(f"    ⚠️  Entity too long: {len(name)} characters")
                if "." in name or "，" in name:
                    print(f"    ⚠️  Entity contains sentence markers")
        else:
            print(f"  No entities found or unexpected result format: {result}")
            
    except Exception as e:
        print(f"❌ Error testing enhanced entity extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_extraction())
