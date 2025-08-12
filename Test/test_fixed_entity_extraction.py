#!/usr/bin/env python3
"""
Test script to verify the entity extraction and relationship mapping fixes.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

async def test_improved_entity_extraction():
    """Test the improved entity extraction."""
    print("🧪 Testing Improved Entity Extraction")
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
        
        # Extract entities
        result = await agent.extract_entities(test_text)
        
        print("\nExtracted entities:")
        if 'entities' in result:
            for entity in result['entities']:
                name = entity.get('name', 'N/A')
                entity_type = entity.get('type', 'N/A')
                confidence = entity.get('confidence', 'N/A')
                print(f"  - {name} ({entity_type}) - confidence: {confidence}")
                
                # Check for overly long entities
                if len(name) > 50:
                    print(f"    ⚠️  Entity too long: {len(name)} characters")
                if "." in name or "，" in name:
                    print(f"    ⚠️  Entity contains sentence markers")
        else:
            print(f"  No entities found or unexpected result format: {result}")
            
    except Exception as e:
        print(f"❌ Error testing entity extraction: {e}")
        import traceback
        traceback.print_exc()

async def test_improved_relationship_mapping():
    """Test the improved relationship mapping."""
    print("\n🔗 Testing Improved Relationship Mapping")
    print("=" * 50)
    
    try:
        from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
        
        # Create agent
        agent = KnowledgeGraphAgent()
        
        # Test entities with proper types
        test_entities = [
            {"name": "Artificial Intelligence", "type": "TECHNOLOGY", "confidence": 0.9},
            {"name": "Google", "type": "ORGANIZATION", "confidence": 0.9},
            {"name": "Machine Learning", "type": "TECHNOLOGY", "confidence": 0.9},
            {"name": "Healthcare", "type": "DOMAIN", "confidence": 0.8}
        ]
        
        test_text = """
        Artificial Intelligence (AI) is transforming the world. 
        Companies like Google are leading the development of machine learning.
        These technologies are being used in healthcare applications.
        """
        
        print(f"Test entities: {[e['name'] for e in test_entities]}")
        print(f"Test text: {test_text.strip()}")
        
        # Map relationships
        result = await agent.map_relationships(test_text, test_entities)
        
        print("\nMapped relationships:")
        if 'content' in result and result['content']:
            relationships = (result['content'][0].get('json', {})
                           .get('relationships', []))
            if relationships:
                for rel in relationships:
                    print(f"  - {rel.get('source', 'N/A')} -> "
                          f"{rel.get('target', 'N/A')} "
                          f"({rel.get('relationship_type', 'N/A')})")
            else:
                print("  No relationships found")
        else:
            print(f"  Unexpected result format: {result}")
            
    except Exception as e:
        print(f"❌ Error testing relationship mapping: {e}")
        import traceback
        traceback.print_exc()

async def test_knowledge_graph_integration():
    """Test the complete knowledge graph integration."""
    print("\n📊 Testing Knowledge Graph Integration")
    print("=" * 50)
    
    try:
        from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
        
        # Create agent
        agent = KnowledgeGraphAgent()
        
        # Test text
        test_text = """
        Artificial Intelligence (AI) is transforming the world. 
        Google and Microsoft are leading the development of machine learning.
        These technologies are being used in healthcare applications.
        """
        
        print(f"Test text: {test_text.strip()}")
        
        # Extract entities
        entities_result = await agent.extract_entities(test_text)
        entities = entities_result.get('entities', [])
        
        print(f"\nExtracted {len(entities)} entities:")
        for entity in entities:
            print(f"  - {entity.get('name', 'N/A')} ({entity.get('type', 'N/A')})")
        
        if entities:
            # Map relationships
            relationships_result = await agent.map_relationships(test_text, entities)
            
            print(f"\nRelationship mapping result:")
            if 'content' in relationships_result and relationships_result['content']:
                relationships = (relationships_result['content'][0].get('json', {})
                               .get('relationships', []))
                print(f"  Found {len(relationships)} relationships")
                for rel in relationships:
                    print(f"  - {rel.get('source', 'N/A')} -> "
                          f"{rel.get('target', 'N/A')} "
                          f"({rel.get('relationship_type', 'N/A')})")
            else:
                print("  No relationships found")
        else:
            print("  No entities found to create relationships")
            
    except Exception as e:
        print(f"❌ Error testing knowledge graph integration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run all tests
    asyncio.run(test_improved_entity_extraction())
    asyncio.run(test_improved_relationship_mapping())
    asyncio.run(test_knowledge_graph_integration())
