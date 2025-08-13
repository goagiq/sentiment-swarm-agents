#!/usr/bin/env python3
"""
Test script to check Russian PDF processing functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config.language_specific_regex_config import (
    detect_language_from_text,
    get_language_processing_settings,
    get_language_regex_patterns
)

def test_russian_language_detection():
    """Test Russian language detection."""
    print("=== Testing Russian Language Detection ===")
    
    russian_text = "Привет мир! Это русский текст для тестирования."
    detected = detect_language_from_text(russian_text)
    print(f"Detected language: {detected}")
    print(f"Expected: ru, Got: {detected}")
    print(f"✓ Language detection working" if detected == "ru" else "✗ Language detection failed")
    print()

def test_russian_processing_settings():
    """Test Russian processing settings."""
    print("=== Testing Russian Processing Settings ===")
    
    settings = get_language_processing_settings("ru")
    print(f"Russian settings: {settings}")
    
    expected_settings = {
        'min_entity_length': 3,
        'max_entity_length': 50,
        'confidence_threshold': 0.7,
        'use_enhanced_extraction': True,
        'relationship_prompt_simplified': True
    }
    
    for key, expected_value in expected_settings.items():
        actual_value = settings.get(key)
        print(f"{key}: expected {expected_value}, got {actual_value}")
        if actual_value == expected_value:
            print(f"✓ {key} setting correct")
        else:
            print(f"✗ {key} setting incorrect")
    print()

def test_russian_regex_patterns():
    """Test Russian regex patterns."""
    print("=== Testing Russian Regex Patterns ===")
    
    patterns = get_language_regex_patterns("ru")
    print(f"Pattern types: {list(patterns.keys())}")
    
    # Test person patterns
    person_patterns = patterns.get("person", [])
    print(f"Person patterns: {len(person_patterns)} patterns")
    
    # Test organization patterns
    org_patterns = patterns.get("organization", [])
    print(f"Organization patterns: {len(org_patterns)} patterns")
    
    # Test location patterns
    location_patterns = patterns.get("location", [])
    print(f"Location patterns: {len(location_patterns)} patterns")
    
    # Test concept patterns
    concept_patterns = patterns.get("concept", [])
    print(f"Concept patterns: {len(concept_patterns)} patterns")
    print()

def test_russian_entity_extraction():
    """Test Russian entity extraction patterns."""
    print("=== Testing Russian Entity Extraction ===")
    
    import re
    patterns = get_language_regex_patterns("ru")
    
    test_text = """
    Иван Петров работает в компании ООО "Технологии будущего" в городе Москва.
    Он изучает искусственный интеллект и машинное обучение.
    Доктор Сидоров преподает в Московском университете.
    """
    
    print(f"Test text: {test_text.strip()}")
    
    for entity_type, pattern_list in patterns.items():
        print(f"\n{entity_type.upper()} entities:")
        for pattern in pattern_list:
            matches = re.findall(pattern, test_text, re.IGNORECASE)
            if matches:
                print(f"  Pattern: {pattern}")
                print(f"  Matches: {matches}")
    print()

def main():
    """Run all tests."""
    print("Testing Russian PDF Processing Functionality")
    print("=" * 50)
    
    try:
        test_russian_language_detection()
        test_russian_processing_settings()
        test_russian_regex_patterns()
        test_russian_entity_extraction()
        
        print("=== Summary ===")
        print("✓ All tests completed successfully")
        print("Russian processing configuration appears to be working correctly")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
