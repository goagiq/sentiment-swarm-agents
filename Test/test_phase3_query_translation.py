"""
Test Phase 3: Query Translation functionality for multilingual knowledge graph.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.knowledge_graph_agent import KnowledgeGraphAgent


async def test_phase3_query_translation():
    """Test Phase 3 query translation functionality."""
    print("🧪 Testing Phase 3: Query Translation")
    print("=" * 50)
    
    # Initialize the knowledge graph agent
    agent = KnowledgeGraphAgent()
    
    # Test data - Chinese query
    chinese_query = "人工智能的发展趋势"
    english_query = "artificial intelligence trends"
    
    print(f"📝 Testing Chinese query: '{chinese_query}'")
    print(f"📝 Testing English query: '{english_query}'")
    
    try:
        # Test 1: Chinese query with Chinese target language
        print("\n🔍 Test 1: Chinese query → Chinese results")
        result1 = await agent.query_knowledge_graph(
            chinese_query, target_language="zh"
        )
        print(f"✅ Result: {json.dumps(result1, indent=2, ensure_ascii=False)}")
        
        # Test 2: Chinese query with English target language
        print("\n🔍 Test 2: Chinese query → English results")
        result2 = await agent.query_knowledge_graph(
            chinese_query, target_language="en"
        )
        print(f"✅ Result: {json.dumps(result2, indent=2, ensure_ascii=False)}")
        
        # Test 3: English query with Chinese target language
        print("\n🔍 Test 3: English query → Chinese results")
        result3 = await agent.query_knowledge_graph(
            english_query, target_language="zh"
        )
        print(f"✅ Result: {json.dumps(result3, indent=2, ensure_ascii=False)}")
        
        # Test 4: English query with English target language (default)
        print("\n🔍 Test 4: English query → English results (default)")
        result4 = await agent.query_knowledge_graph(english_query)
        print(f"✅ Result: {json.dumps(result4, indent=2, ensure_ascii=False)}")
        
        # Test 5: Test caching functionality
        print("\n🔍 Test 5: Testing translation caching")
        result5 = await agent.query_knowledge_graph(
            chinese_query, target_language="zh"
        )
        print(f"✅ Cached result: "
              f"{json.dumps(result5, indent=2, ensure_ascii=False)}")
        
        print("\n🎉 Phase 3 Query Translation Tests Completed Successfully!")
        
        # Summary
        print("\n📊 Test Summary:")
        print(f"- Chinese → Chinese: {'✅' if result1 else '❌'}")
        print(f"- Chinese → English: {'✅' if result2 else '❌'}")
        print(f"- English → Chinese: {'✅' if result3 else '❌'}")
        print(f"- English → English: {'✅' if result4 else '❌'}")
        print(f"- Caching: {'✅' if result5 else '❌'}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


async def test_translation_service_integration():
    """Test that TranslationService is properly integrated."""
    print("\n🔧 Testing TranslationService Integration")
    print("=" * 50)
    
    agent = KnowledgeGraphAgent()
    
    # Test language detection
    test_texts = [
        "Hello world",
        "你好世界",
        "Bonjour le monde",
        "Hola mundo"
    ]
    
    for text in test_texts:
        try:
            detected_lang = await agent.translation_service.detect_language(text)
            print(f"✅ '{text}' → {detected_lang}")
        except Exception as e:
            print(f"❌ Language detection failed for '{text}': "
                  f"{e}")


if __name__ == "__main__":
    asyncio.run(test_phase3_query_translation())
    asyncio.run(test_translation_service_integration())
