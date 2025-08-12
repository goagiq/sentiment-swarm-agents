"""
Test Phase 4: Multilingual Entity Extraction functionality for knowledge graph.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.knowledge_graph_agent import KnowledgeGraphAgent


async def test_phase4_multilingual_entity_extraction():
    """Test Phase 4 multilingual entity extraction functionality."""
    print("🧪 Testing Phase 4: Multilingual Entity Extraction")
    print("=" * 60)
    
    # Initialize the knowledge graph agent
    agent = KnowledgeGraphAgent()
    
    # Test data in different languages
    test_cases = [
        {
            "language": "en",
            "text": "Donald Trump and Joe Biden discussed trade policy with China. The US Government implemented new tariffs on imports from Beijing.",
            "expected_entities": ["Donald Trump", "Joe Biden", "China", "US Government", "Beijing"]
        },
        {
            "language": "zh",
            "text": "习近平主席和特朗普总统讨论了与中国的贸易政策。中国政府实施了新的关税政策。",
            "expected_entities": ["习近平", "特朗普", "中国", "中国政府"]
        },
        {
            "language": "ja",
            "text": "安倍晋三首相とトランプ大統領が中国との貿易政策について議論しました。日本政府は新しい関税を実施しました。",
            "expected_entities": ["安倍晋三", "トランプ", "中国", "日本政府"]
        },
        {
            "language": "ko",
            "text": "문재인 대통령과 트럼프 대통령이 중국과의 무역 정책에 대해 논의했습니다. 한국 정부는 새로운 관세를 시행했습니다.",
            "expected_entities": ["문재인", "트럼프", "중국", "한국 정부"]
        },
        {
            "language": "es",
            "text": "El Presidente Trump y el Presidente Biden discutieron la política comercial con China. El Gobierno de EE.UU. implementó nuevos aranceles.",
            "expected_entities": ["Trump", "Biden", "China", "Gobierno de EE.UU."]
        },
        {
            "language": "fr",
            "text": "Le Président Trump et le Président Biden ont discuté de la politique commerciale avec la Chine. Le Gouvernement américain a mis en œuvre de nouveaux tarifs.",
            "expected_entities": ["Trump", "Biden", "Chine", "Gouvernement américain"]
        }
    ]
    
    results = {}
    
    for i, test_case in enumerate(test_cases, 1):
        language = test_case["language"]
        text = test_case["text"]
        expected_entities = test_case["expected_entities"]
        
        print(f"\n🔍 Test {i}: {language.upper()} Entity Extraction")
        print(f"📝 Text: {text[:50]}...")
        
        try:
            # Test entity extraction
            result = await agent.extract_entities(text, language)
            json_data = result.get("content", [{}])[0].get("json", {})
            entities = json_data.get("entities", [])
            
            # Extract entity names
            extracted_entity_names = [entity.get("name", "") for entity in entities]
            
            print(f"✅ Extracted {len(entities)} entities:")
            for entity in entities:
                print(f"   - {entity.get('name', 'N/A')} ({entity.get('type', 'N/A')})")
            
            # Check if expected entities were found
            found_entities = []
            for expected in expected_entities:
                for extracted in extracted_entity_names:
                    if expected.lower() in extracted.lower() or extracted.lower() in expected.lower():
                        found_entities.append(expected)
                        break
            
            success_rate = len(found_entities) / len(expected_entities) if expected_entities else 1.0
            results[language] = {
                "success": success_rate > 0.5,  # Consider success if >50% found
                "extracted_count": len(entities),
                "expected_count": len(expected_entities),
                "found_count": len(found_entities),
                "success_rate": success_rate,
                "found_entities": found_entities,
                "missing_entities": [e for e in expected_entities if e not in found_entities]
            }
            
            print(f"📊 Success Rate: {success_rate:.2%}")
            print(f"   Found: {found_entities}")
            if results[language]["missing_entities"]:
                print(f"   Missing: {results[language]['missing_entities']}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results[language] = {
                "success": False,
                "error": str(e)
            }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Phase 4 Test Summary")
    print("=" * 60)
    
    total_tests = len(test_cases)
    successful_tests = sum(1 for r in results.values() if r.get("success", False))
    
    for language, result in results.items():
        status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
        if "error" in result:
            print(f"{language.upper()}: {status} - Error: {result['error']}")
        else:
            success_rate = result.get("success_rate", 0)
            print(f"{language.upper()}: {status} - {success_rate:.1%} success rate")
    
    print(f"\n🎯 Overall: {successful_tests}/{total_tests} languages passed")
    
    return results


async def test_language_specific_patterns():
    """Test language-specific entity detection patterns."""
    print("\n🔧 Testing Language-Specific Patterns")
    print("=" * 50)
    
    agent = KnowledgeGraphAgent()
    
    # Test Chinese patterns
    chinese_text = "习近平主席访问了北京清华大学。"
    print(f"📝 Chinese text: {chinese_text}")
    
    try:
        result = await agent.extract_entities(chinese_text, "zh")
        json_data = result.get("content", [{}])[0].get("json", {})
        entities = json_data.get("entities", [])
        
        print(f"✅ Extracted entities: {[e.get('name') for e in entities]}")
        
        # Test fallback extraction
        fallback_result = agent._enhanced_fallback_entity_extraction(chinese_text, "zh")
        print(f"✅ Fallback entities: {[e.get('name') for e in fallback_result.get('entities', [])]}")
        
    except Exception as e:
        print(f"❌ Chinese pattern test failed: {e}")


async def test_entity_type_categorization():
    """Test entity type categorization across languages."""
    print("\n🏷️ Testing Entity Type Categorization")
    print("=" * 50)
    
    agent = KnowledgeGraphAgent()
    
    test_texts = [
        ("en", "Microsoft Corporation is located in Washington state."),
        ("zh", "微软公司位于华盛顿州。"),
        ("ja", "マイクロソフト株式会社はワシントン州にあります。"),
        ("ko", "마이크로소프트 회사는 워싱턴 주에 있습니다。")
    ]
    
    for language, text in test_texts:
        print(f"\n📝 {language.upper()}: {text}")
        
        try:
            result = await agent.extract_entities(text, language)
            json_data = result.get("content", [{}])[0].get("json", {})
            entities = json_data.get("entities", [])
            
            for entity in entities:
                print(f"   - {entity.get('name', 'N/A')}: {entity.get('type', 'N/A')}")
                
        except Exception as e:
            print(f"❌ {language} categorization failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_phase4_multilingual_entity_extraction())
    asyncio.run(test_language_specific_patterns())
    asyncio.run(test_entity_type_categorization())
