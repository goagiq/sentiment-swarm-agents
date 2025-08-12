#!/usr/bin/env python3
"""
Debug script to test and fix relationship mapping in KnowledgeGraphAgent.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.knowledge_graph_agent import KnowledgeGraphAgent

async def debug_relationship_mapping():
    """Debug the relationship mapping functionality."""
    
    # Test text with clear relationships
    test_text = """
    Apple Inc. is a technology company based in Cupertino, California. 
    Steve Jobs and Steve Wozniak founded Apple in 1976. 
    Tim Cook is the current CEO of Apple. 
    The company has offices in San Francisco and New York.
    """
    
    print("=== Debugging Relationship Mapping ===")
    
    try:
        # Initialize the KnowledgeGraphAgent
        kg_agent = KnowledgeGraphAgent()
        
        # Test entity extraction first
        print("\n--- Testing Entity Extraction ---")
        entities_result = await kg_agent.extract_entities(test_text, language="en")
        
        if isinstance(entities_result, dict) and 'content' in entities_result:
            content = entities_result['content']
            if content and isinstance(content[0], dict) and 'json' in content[0]:
                json_data = content[0]['json']
                entities = json_data.get('entities', [])
                
                print(f"Extracted {len(entities)} entities:")
                for i, entity in enumerate(entities, 1):
                    print(f"  {i}. {entity.get('text', 'N/A')} ({entity.get('type', 'N/A')})")
                
                # Test relationship mapping
                print("\n--- Testing Relationship Mapping ---")
                relationships_result = await kg_agent.map_relationships(test_text, entities)
                
                print(f"Relationship mapping result type: {type(relationships_result)}")
                print(f"Relationship mapping result keys: {relationships_result.keys() if isinstance(relationships_result, dict) else 'Not a dict'}")
                
                if isinstance(relationships_result, dict) and 'content' in relationships_result:
                    content = relationships_result['content']
                    if content and isinstance(content[0], dict) and 'json' in content[0]:
                        json_data = content[0]['json']
                        relationships = json_data.get('relationships', [])
                        
                        print(f"\nExtracted {len(relationships)} relationships:")
                        for i, rel in enumerate(relationships, 1):
                            print(f"  {i}. {rel.get('source', 'N/A')} --{rel.get('relationship_type', 'N/A')}--> {rel.get('target', 'N/A')}")
                        
                        if relationships:
                            print("\n✓ Relationship mapping working")
                        else:
                            print("\n⚠ No relationships extracted - this is the problem!")
                            
                            # Debug the LLM response
                            print("\n--- Debugging LLM Response ---")
                            # Let's manually test the relationship mapping prompt
                            entity_names = [e.get('text', '') for e in entities if e.get('text')]
                            print(f"Entity names for relationship mapping: {entity_names}")
                            
                            # Test the prompt manually
                            prompt = f"""
You are an expert relationship extraction system. Analyze the relationships between entities and return them in JSON format.

CRITICAL INSTRUCTIONS:
1. Extract relationships between the provided entities
2. Use the exact entity names provided
3. Return ONLY valid JSON in the exact format specified
4. If no clear relationships exist, return an empty relationships array

Entities to analyze: {entity_names}

Text to analyze:
{test_text}

Expected JSON format:
{{
    "relationships": [
        {{
            "source": "entity_name",
            "target": "entity_name", 
            "relationship_type": "relationship_type",
            "confidence": 0.95,
            "description": "clear description of the relationship"
        }}
    ]
}}

Return only the JSON object, no additional text.
"""
                            print(f"Prompt: {prompt}")
                            
                    else:
                        print("\n⚠ Unexpected content structure in relationships")
                else:
                    print("\n⚠ Unexpected result structure in relationships")
            else:
                print("\n⚠ Unexpected content structure in entities")
        else:
            print("\n⚠ Unexpected result structure in entities")
        
    except Exception as e:
        print(f"❌ Debug failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_relationship_mapping())
