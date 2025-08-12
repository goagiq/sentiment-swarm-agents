#!/usr/bin/env python3
"""
Test script for enhanced entity categorization in knowledge graph agent.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.knowledge_graph_agent import KnowledgeGraphAgent
from config.config import config


async def test_enhanced_entity_categorization():
    """Test the enhanced entity categorization functionality."""
    
    print("Testing Enhanced Entity Categorization")
    print("=" * 50)
    
    # Initialize the knowledge graph agent
    agent = KnowledgeGraphAgent(model_name=config.model.default_text_model)
    
    # Test text with various entity types
    test_text = """
    Donald Trump and Joe Biden are discussing trade policies with China. 
    The US government is considering new tariffs on Chinese imports. 
    Michigan Governor Gretchen Whitmer supports the economic policies. 
    Microsoft and Apple are leading technology companies in the United States.
    Artificial intelligence and machine learning are transforming the industry.
    """
    
    print(f"Test Text: {test_text.strip()}")
    print()
    
    # Test entity extraction
    print("Testing Entity Extraction...")
    try:
        entities_result = await agent.extract_entities(test_text, language="en")
        entities = entities_result.get("content", [{}])[0].get("json", {}).get("entities", [])
        
        print(f"Extracted {len(entities)} entities:")
        print("-" * 50)
        
        # Group entities by type
        entity_types = {}
        for entity in entities:
            entity_type = entity.get("type", "UNKNOWN")
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity)
        
        # Display entities by type
        for entity_type, type_entities in entity_types.items():
            print(f"\n{entity_type} ({len(type_entities)} entities):")
            for entity in type_entities:
                name = entity.get("text", entity.get("name", "Unknown"))
                confidence = entity.get("confidence", 0.0)
                print(f"  - {name} (confidence: {confidence:.2f})")
        
        print()
        
        # Test categorization accuracy
        expected_categorizations = {
            "Donald": "PERSON",
            "Trump": "PERSON", 
            "Joe": "PERSON",
            "Biden": "PERSON",
            "China": "LOCATION",
            "government": "ORGANIZATION",
            "tariffs": "CONCEPT",
            "Michigan": "LOCATION",
            "Gretchen": "PERSON",
            "Whitmer": "PERSON",
            "Microsoft": "ORGANIZATION",
            "Apple": "ORGANIZATION",
            "United States": "LOCATION",
            "artificial intelligence": "CONCEPT",
            "machine learning": "CONCEPT"
        }
        
        print("Categorization Accuracy Check:")
        print("-" * 50)
        
        correct_categorizations = 0
        total_expected = len(expected_categorizations)
        
        for entity in entities:
            name = entity.get("text", entity.get("name", "")).lower()
            actual_type = entity.get("type", "UNKNOWN")
            
            # Check if this entity was expected
            for expected_name, expected_type in expected_categorizations.items():
                if name in expected_name.lower() or expected_name.lower() in name:
                    if actual_type == expected_type:
                        print(f"✓ {name} correctly categorized as {actual_type}")
                        correct_categorizations += 1
                    else:
                        print(f"✗ {name} incorrectly categorized as {actual_type} (expected {expected_type})")
                    break
        
        accuracy = (correct_categorizations / total_expected) * 100 if total_expected > 0 else 0
        print(f"\nCategorization Accuracy: {accuracy:.1f}% ({correct_categorizations}/{total_expected})")
        
        # Test relationship mapping
        print("\nTesting Relationship Mapping...")
        print("-" * 50)
        
        relationships_result = await agent.map_relationships(test_text, entities)
        relationships = relationships_result.get("relationships", [])
        
        print(f"Mapped {len(relationships)} relationships:")
        for rel in relationships:
            source = rel.get("source", "Unknown")
            target = rel.get("target", "Unknown")
            rel_type = rel.get("relationship_type", "Unknown")
            confidence = rel.get("confidence", 0.0)
            print(f"  - {source} --[{rel_type}]--> {target} (confidence: {confidence:.2f})")
        
        # Test graph report generation
        print("\nTesting Graph Report Generation...")
        print("-" * 50)
        
        # Add entities and relationships to the graph
        await agent._add_to_graph(entities, relationships, "test_request", "en")
        
        # Generate HTML report
        output_path = Path("Test/enhanced_entity_categorization_report.html")
        report_result = await agent.generate_graph_report(
            output_path=str(output_path), 
            target_language="en"
        )
        
        print(f"Generated report: {output_path}")
        print(f"Report status: {report_result.get('status', 'Unknown')}")
        
        if report_result.get("status") == "success":
            print("✓ Enhanced entity categorization test completed successfully!")
            print(f"✓ HTML report generated: {output_path}")
        else:
            print("✗ Report generation failed")
            print(f"Error: {report_result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


async def test_multilingual_entity_categorization():
    """Test entity categorization with different languages."""
    
    print("\n\nTesting Multilingual Entity Categorization")
    print("=" * 50)
    
    agent = KnowledgeGraphAgent(model_name=config.model.default_text_model)
    
    # Test Chinese text
    chinese_text = "习近平主席访问美国，与特朗普总统讨论贸易政策。中国政府支持经济发展。"
    
    print(f"Chinese Test Text: {chinese_text}")
    print()
    
    try:
        entities_result = await agent.extract_entities(chinese_text, language="zh")
        entities = entities_result.get("content", [{}])[0].get("json", {}).get("entities", [])
        
        print(f"Extracted {len(entities)} Chinese entities:")
        for entity in entities:
            name = entity.get("text", entity.get("name", "Unknown"))
            entity_type = entity.get("type", "UNKNOWN")
            confidence = entity.get("confidence", 0.0)
            print(f"  - {name} ({entity_type}, confidence: {confidence:.2f})")
        
        print("✓ Chinese entity categorization test completed!")
        
    except Exception as e:
        print(f"✗ Chinese test failed: {e}")


async def main():
    """Main test function."""
    print("Enhanced Entity Categorization Test Suite")
    print("=" * 60)
    
    # Test English entity categorization
    await test_enhanced_entity_categorization()
    
    # Test multilingual entity categorization
    await test_multilingual_entity_categorization()
    
    print("\n" + "=" * 60)
    print("Test suite completed!")


if __name__ == "__main__":
    asyncio.run(main())
