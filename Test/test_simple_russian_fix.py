#!/usr/bin/env python3
"""
Simple test to check Russian processing and identify the issue.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_basic_russian_config():
    """Test basic Russian configuration without complex dependencies."""
    print("=== Testing Basic Russian Configuration ===")
    
    try:
        from src.config.language_specific_regex_config import (
            detect_language_from_text,
            get_language_processing_settings,
            get_language_regex_patterns
        )
        
        # Test language detection
        russian_text = "Привет мир! Это русский текст."
        detected = detect_language_from_text(russian_text)
        print(f"Language detection: {detected}")
        
        # Test processing settings
        settings = get_language_processing_settings("ru")
        print(f"Russian settings: {settings}")
        
        # Test regex patterns
        patterns = get_language_regex_patterns("ru")
        print(f"Pattern types: {list(patterns.keys())}")
        
        print("✓ Basic Russian configuration working")
        return True
        
    except Exception as e:
        print(f"✗ Basic Russian configuration failed: {e}")
        return False


def test_language_service():
    """Test the new language processing service."""
    print("\n=== Testing Language Processing Service ===")
    
    try:
        from src.core.language_processing_service import LanguageProcessingService
        
        service = LanguageProcessingService()
        
        # Test language detection
        russian_text = "Привет мир! Это русский текст."
        detected = service.detect_language(russian_text)
        print(f"Service language detection: {detected}")
        
        # Test entity extraction
        result = service.extract_entities_with_config(russian_text, "ru")
        print(f"Entity extraction result: {result}")
        
        print("✓ Language processing service working")
        return True
        
    except Exception as e:
        print(f"✗ Language processing service failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_knowledge_graph_agent_simple():
    """Test knowledge graph agent with minimal initialization."""
    print("\n=== Testing Knowledge Graph Agent (Simple) ===")
    
    try:
        from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
        
        # Initialize with minimal settings
        agent = KnowledgeGraphAgent(model_name="llama3.2")
        
        print("✓ Knowledge graph agent initialized")
        return True
        
    except Exception as e:
        print(f"✗ Knowledge graph agent initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Simple Russian Processing Test")
    print("=" * 40)
    
    results = []
    
    # Test 1: Basic configuration
    results.append(test_basic_russian_config())
    
    # Test 2: Language service
    results.append(test_language_service())
    
    # Test 3: Knowledge graph agent
    results.append(test_knowledge_graph_agent_simple())
    
    print("\n=== Summary ===")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed - Russian processing should work")
    else:
        print("✗ Some tests failed - Russian processing has issues")


if __name__ == "__main__":
    main()
