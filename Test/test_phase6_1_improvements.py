#!/usr/bin/env python3
"""
Test Phase 6.1 Entity Extraction Improvements
Test the enhanced entity extraction with improved prompts and patterns.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.enhanced_chinese_entity_extraction import EnhancedChineseEntityExtractor, ChineseEntityValidator
from config.entity_extraction_config import get_language_config, get_entity_types, get_common_entities, get_patterns


async def test_phase6_1_improvements():
    """Test Phase 6.1 entity extraction improvements."""
    print("🧪 Testing Phase 6.1 Entity Extraction Improvements")
    print("=" * 60)
    
    # Test 1: Configuration System
    print("\n1. Testing Configuration System")
    print("-" * 40)
    
    # Test Chinese configuration
    zh_config = get_language_config("zh")
    print(f"✅ Chinese config loaded: {zh_config.language_name}")
    print(f"   Entity types: {list(zh_config.entity_types.keys())}")
    
    # Test English configuration
    en_config = get_language_config("en")
    print(f"✅ English config loaded: {en_config.language_name}")
    print(f"   Entity types: {list(en_config.entity_types.keys())}")
    
    # Test 2: Enhanced Chinese Entity Extractor
    print("\n2. Testing Enhanced Chinese Entity Extractor")
    print("-" * 40)
    
    extractor = EnhancedChineseEntityExtractor()
    validator = ChineseEntityValidator()
    
    # Test Chinese text
    chinese_text = """
    习近平主席在北京会见马云，讨论了人工智能和数字经济的发展。
    华为技术有限公司在深圳总部发布了新的5G技术。
    清华大学和中科院的研究团队在量子计算领域取得了重大突破。
    """
    
    print("📝 Testing Chinese text extraction:")
    print(f"   Text: {chinese_text.strip()}")
    
    # Test pattern-based extraction
    pattern_entities = extractor._extract_with_patterns(chinese_text)
    print(f"   Pattern-based entities found: {len(pattern_entities)}")
    for entity in pattern_entities[:5]:  # Show first 5
        print(f"     - {entity.text} ({entity.entity_type}, conf: {entity.confidence})")
    
    # Test dictionary-based extraction
    dict_entities = extractor._extract_with_dictionary(chinese_text)
    print(f"   Dictionary-based entities found: {len(dict_entities)}")
    for entity in dict_entities[:5]:  # Show first 5
        print(f"     - {entity.text} ({entity.entity_type}, conf: {entity.confidence})")
    
    # Test 3: Entity Validation
    print("\n3. Testing Entity Validation")
    print("-" * 40)
    
    test_cases = [
        ("习近平", "PERSON"),
        ("华为", "ORGANIZATION"),
        ("北京", "LOCATION"),
        ("人工智能", "CONCEPT"),
        ("普通词汇", "UNKNOWN"),
    ]
    
    for text, expected_type in test_cases:
        if expected_type == "PERSON":
            is_valid = validator.validate_person_name(text)
        elif expected_type == "ORGANIZATION":
            is_valid = validator.validate_organization_name(text)
        elif expected_type == "LOCATION":
            is_valid = validator.validate_location_name(text)
        elif expected_type == "CONCEPT":
            is_valid = validator.validate_technical_term(text)
        else:
            is_valid = False
        
        status = "✅" if is_valid else "❌"
        print(f"   {status} '{text}' ({expected_type}): {is_valid}")
    
    # Test 4: Multi-Strategy Pipeline
    print("\n4. Testing Multi-Strategy Pipeline")
    print("-" * 40)
    
    # Test the enhanced extraction method
    try:
        entities = await extractor.extract_entities_enhanced(chinese_text)
        print(f"   Enhanced extraction found: {len(entities)} entities")
        
        # Group by type
        by_type = {}
        for entity in entities:
            if entity.entity_type not in by_type:
                by_type[entity.entity_type] = []
            by_type[entity.entity_type].append(entity)
        
        for entity_type, entity_list in by_type.items():
            print(f"     {entity_type}: {len(entity_list)} entities")
            for entity in entity_list[:3]:  # Show first 3 of each type
                print(f"       - {entity.text} (conf: {entity.confidence})")
    
    except Exception as e:
        print(f"   ❌ Enhanced extraction failed: {e}")
    
    # Test 5: Performance Test
    print("\n5. Testing Performance")
    print("-" * 40)
    
    import time
    
    # Test pattern extraction performance
    start_time = time.time()
    for _ in range(100):
        extractor._extract_with_patterns(chinese_text)
    pattern_time = time.time() - start_time
    
    # Test dictionary extraction performance
    start_time = time.time()
    for _ in range(100):
        extractor._extract_with_dictionary(chinese_text)
    dict_time = time.time() - start_time
    
    print(f"   Pattern extraction (100x): {pattern_time:.3f}s")
    print(f"   Dictionary extraction (100x): {dict_time:.3f}s")
    print(f"   Average per extraction: {(pattern_time + dict_time) / 200:.6f}s")
    
    # Test 6: Accuracy Assessment
    print("\n6. Accuracy Assessment")
    print("-" * 40)
    
    # Expected entities in the test text
    expected_entities = {
        "PERSON": ["习近平", "马云"],
        "ORGANIZATION": ["华为技术有限公司", "清华大学", "中科院"],
        "LOCATION": ["北京", "深圳"],
        "CONCEPT": ["人工智能", "数字经济", "5G技术", "量子计算"]
    }
    
    # Get actual entities
    actual_entities = await extractor.extract_entities_enhanced(chinese_text)
    actual_by_type = {}
    for entity in actual_entities:
        if entity.entity_type not in actual_by_type:
            actual_by_type[entity.entity_type] = []
        actual_by_type[entity.entity_type].append(entity.text)
    
    # Calculate accuracy
    total_expected = sum(len(entities) for entities in expected_entities.values())
    total_found = sum(len(entities) for entities in actual_by_type.values())
    
    print(f"   Expected entities: {total_expected}")
    print(f"   Found entities: {total_found}")
    
    # Calculate accuracy by type
    for entity_type, expected_list in expected_entities.items():
        actual_list = actual_by_type.get(entity_type, [])
        found_count = len(set(expected_list) & set(actual_list))
        accuracy = found_count / len(expected_list) if expected_list else 0
        print(f"     {entity_type}: {found_count}/{len(expected_list)} ({accuracy:.1%})")
    
    print("\n🎉 Phase 6.1 Improvements Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_phase6_1_improvements())
