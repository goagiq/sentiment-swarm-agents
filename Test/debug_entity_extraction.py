#!/usr/bin/env python3
"""
Debug script for enhanced entity extraction.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.knowledge_graph_agent import KnowledgeGraphAgent
from config.config import config


def test_enhanced_fallback_directly():
    """Test the enhanced fallback entity extraction method directly."""
    
    print("Testing Enhanced Fallback Entity Extraction Directly")
    print("=" * 60)
    
    # Initialize the knowledge graph agent
    agent = KnowledgeGraphAgent(model_name=config.model.default_text_model)
    
    # Test text
    test_text = "Donald Trump and Joe Biden are discussing trade policies with China."
    
    print(f"Test Text: {test_text}")
    print()
    
    # Test the enhanced fallback method directly
    try:
        result = agent._enhanced_fallback_entity_extraction(test_text, "en")
        print("Enhanced Fallback Result:")
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        entities = result.get("entities", [])
        print(f"\nExtracted {len(entities)} entities:")
        
        for i, entity in enumerate(entities):
            print(f"  {i+1}. {entity}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def test_is_potential_entity():
    """Test the _is_potential_entity method."""
    
    print("\n\nTesting _is_potential_entity Method")
    print("=" * 60)
    
    agent = KnowledgeGraphAgent(model_name=config.model.default_text_model)
    
    test_words = ["Donald", "Trump", "Joe", "Biden", "China", "trade", "policies", "the", "and", "are"]
    
    for word in test_words:
        result = agent._is_potential_entity(word, "en", {})
        print(f"'{word}' -> {result}")


def test_determine_entity_type():
    """Test the _determine_entity_type_comprehensive method."""
    
    print("\n\nTesting _determine_entity_type_comprehensive Method")
    print("=" * 60)
    
    agent = KnowledgeGraphAgent(model_name=config.model.default_text_model)
    
    # Get comprehensive patterns
    from src.config.entity_extraction_config import (
        COMPREHENSIVE_PERSON_PATTERNS,
        COMPREHENSIVE_LOCATION_PATTERNS,
        COMPREHENSIVE_ORGANIZATION_PATTERNS,
        COMPREHENSIVE_CONCEPT_PATTERNS,
        COMPREHENSIVE_OBJECT_PATTERNS,
        COMPREHENSIVE_PROCESS_PATTERNS
    )
    
    comprehensive_patterns = {
        "PERSON": COMPREHENSIVE_PERSON_PATTERNS,
        "LOCATION": COMPREHENSIVE_LOCATION_PATTERNS,
        "ORGANIZATION": COMPREHENSIVE_ORGANIZATION_PATTERNS,
        "CONCEPT": COMPREHENSIVE_CONCEPT_PATTERNS,
        "OBJECT": COMPREHENSIVE_OBJECT_PATTERNS,
        "PROCESS": COMPREHENSIVE_PROCESS_PATTERNS
    }
    
    test_words = ["Donald", "Trump", "Joe", "Biden", "China", "trade", "policies", "government"]
    
    for word in test_words:
        entity_type = agent._determine_entity_type_comprehensive(
            word, "en", comprehensive_patterns, {}
        )
        print(f"'{word}' -> {entity_type}")


if __name__ == "__main__":
    test_enhanced_fallback_directly()
    test_is_potential_entity()
    test_determine_entity_type()
