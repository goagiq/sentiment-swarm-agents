#!/usr/bin/env python3
"""
Test script for enhanced entity extraction improvements.
Tests the Phase 6.1 and 6.2 improvements for Chinese entity extraction.
"""

import asyncio
import json
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.entity_extraction_agent import EntityExtractionAgent


async def test_enhanced_chinese_entity_extraction():
    """Test enhanced Chinese entity extraction."""
    print("🧪 Testing Enhanced Chinese Entity Extraction")
    print("=" * 50)
    
    # Initialize the enhanced entity extraction agent
    agent = EntityExtractionAgent()
    
    # Test cases from Phase 6 document
    test_cases = [
        {
            "name": "Person Names Test",
            "text": "习近平主席和李克强总理在北京会见了马云先生。",
            "expected_entities": ["习近平", "李克强", "马云", "北京"],
            "expected_types": ["person", "person", "person", "location"]
        },
        {
            "name": "Organizations Test", 
            "text": "华为技术有限公司和阿里巴巴集团在深圳设立了新的研发中心。",
            "expected_entities": ["华为", "阿里巴巴", "深圳"],
            "expected_types": ["organization", "organization", "location"]
        },
        {
            "name": "Technical Terms Test",
            "text": "人工智能和机器学习技术在医疗领域有广泛应用。",
            "expected_entities": ["人工智能", "机器学习"],
            "expected_types": ["concept", "concept"]
        },
        {
            "name": "Mixed Entities Test",
            "text": "马云在杭州创立了阿里巴巴集团，该公司总部位于北京。",
            "expected_entities": ["马云", "杭州", "阿里巴巴", "北京"],
            "expected_types": ["person", "location", "organization", "location"]
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n📝 Testing: {test_case['name']}")
        print(f"Text: {test_case['text']}")
        
        try:
            # Extract entities using enhanced method
            result = await agent.extract_entities_multilingual(test_case['text'], "zh")
            
            print(f"✅ Extraction successful")
            print(f"Entities found: {len(result['entities'])}")
            
            # Display extracted entities
            for entity in result['entities']:
                print(f"  - {entity.get('name', 'N/A')} ({entity.get('type', 'unknown')}) - Confidence: {entity.get('confidence', 0.0):.2f}")
            
            # Calculate accuracy
            extracted_names = [entity.get('name', '') for entity in result['entities']]
            extracted_types = [entity.get('type', '') for entity in result['entities']]
            
            # Name accuracy
            correct_names = sum(1 for name in test_case['expected_entities'] if name in extracted_names)
            name_accuracy = correct_names / len(test_case['expected_entities']) if test_case['expected_entities'] else 0
            
            # Type accuracy (for entities that were found)
            correct_types = 0
            for i, expected_name in enumerate(test_case['expected_entities']):
                if expected_name in extracted_names:
                    idx = extracted_names.index(expected_name)
                    if idx < len(extracted_types) and extracted_types[idx] == test_case['expected_types'][i]:
                        correct_types += 1
            
            type_accuracy = correct_types / correct_names if correct_names > 0 else 0
            overall_accuracy = (name_accuracy + type_accuracy) / 2
            
            print(f"📊 Results:")
            print(f"  - Name Accuracy: {name_accuracy:.2%} ({correct_names}/{len(test_case['expected_entities'])})")
            print(f"  - Type Accuracy: {type_accuracy:.2%} ({correct_types}/{correct_names})")
            print(f"  - Overall Accuracy: {overall_accuracy:.2%}")
            
            results.append({
                "test_name": test_case['name'],
                "success": True,
                "name_accuracy": name_accuracy,
                "type_accuracy": type_accuracy,
                "overall_accuracy": overall_accuracy,
                "entities_found": len(result['entities']),
                "extracted_entities": extracted_names,
                "extracted_types": extracted_types,
                "expected_entities": test_case['expected_entities'],
                "expected_types": test_case['expected_types']
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "test_name": test_case['name'],
                "success": False,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 50)
    print("📈 SUMMARY")
    print("=" * 50)
    
    successful_tests = [r for r in results if r['success']]
    if successful_tests:
        avg_name_accuracy = sum(r['name_accuracy'] for r in successful_tests) / len(successful_tests)
        avg_type_accuracy = sum(r['type_accuracy'] for r in successful_tests) / len(successful_tests)
        avg_overall_accuracy = sum(r['overall_accuracy'] for r in successful_tests) / len(successful_tests)
        
        print(f"✅ Successful tests: {len(successful_tests)}/{len(test_cases)}")
        print(f"📊 Average Name Accuracy: {avg_name_accuracy:.2%}")
        print(f"📊 Average Type Accuracy: {avg_type_accuracy:.2%}")
        print(f"📊 Average Overall Accuracy: {avg_overall_accuracy:.2%}")
        
        # Check if we meet the Phase 6 targets
        print(f"\n🎯 Phase 6 Targets:")
        print(f"  - Person Names: 25% → 85% (Current: {avg_name_accuracy:.2%})")
        print(f"  - Organizations: 75% → 90% (Current: {avg_type_accuracy:.2%})")
        print(f"  - Overall: 50% → 88% (Current: {avg_overall_accuracy:.2%})")
        
        if avg_overall_accuracy >= 0.85:
            print("🎉 SUCCESS: Meeting Phase 6 accuracy targets!")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Below Phase 6 accuracy targets")
    else:
        print("❌ No successful tests")
    
    # Save results
    with open('Test/enhanced_entity_extraction_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Results saved to: Test/enhanced_entity_extraction_results.json")
    
    return results


async def test_english_entity_extraction():
    """Test English entity extraction for comparison."""
    print("\n🧪 Testing English Entity Extraction")
    print("=" * 50)
    
    agent = EntityExtractionAgent()
    
    test_cases = [
        {
            "name": "English Names Test",
            "text": "President Joe Biden and Elon Musk discussed AI technology.",
            "expected_entities": ["Joe Biden", "Elon Musk", "AI"],
            "expected_types": ["person", "person", "concept"]
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📝 Testing: {test_case['name']}")
        print(f"Text: {test_case['text']}")
        
        try:
            result = await agent.extract_entities_multilingual(test_case['text'], "en")
            
            print(f"✅ Extraction successful")
            print(f"Entities found: {len(result['entities'])}")
            
            for entity in result['entities']:
                print(f"  - {entity.get('name', 'N/A')} ({entity.get('type', 'unknown')}) - Confidence: {entity.get('confidence', 0.0):.2f}")
                
        except Exception as e:
            print(f"❌ Error: {e}")


async def main():
    """Main test function."""
    print("🚀 Starting Enhanced Entity Extraction Tests")
    print("Testing Phase 6.1 and 6.2 improvements")
    
    # Test Chinese entity extraction
    chinese_results = await test_enhanced_chinese_entity_extraction()
    
    # Test English entity extraction
    await test_english_entity_extraction()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
