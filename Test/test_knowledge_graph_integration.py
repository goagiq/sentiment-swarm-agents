#!/usr/bin/env python3
"""
Test script to verify KnowledgeGraphAgent integration with improved EntityExtractionAgent.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.knowledge_graph_agent import KnowledgeGraphAgent
from loguru import logger

async def test_knowledge_graph_integration():
    """Test the KnowledgeGraphAgent with improved entity extraction."""
    
    # Sample text for testing
    test_text = """
    Apple Inc. is a technology company based in Cupertino, California. 
    Steve Jobs and Steve Wozniak founded Apple in 1976. 
    The company is known for products like the iPhone, iPad, and MacBook. 
    Tim Cook is the current CEO of Apple. 
    The company has offices in San Francisco and New York.
    """
    
    print("=== Testing KnowledgeGraphAgent with Improved Entity Extraction ===")
    
    try:
        # Initialize the KnowledgeGraphAgent
        print("Initializing KnowledgeGraphAgent...")
        kg_agent = KnowledgeGraphAgent()
        print("✓ KnowledgeGraphAgent initialized successfully")
        
        # Test entity extraction
        print("\n--- Testing Entity Extraction ---")
        entities_result = await kg_agent.extract_entities(test_text, language="en")
        
        print(f"Entity extraction result type: {type(entities_result)}")
        print(f"Entity extraction result keys: {entities_result.keys() if isinstance(entities_result, dict) else 'Not a dict'}")
        
        if isinstance(entities_result, dict) and 'content' in entities_result:
            content = entities_result['content']
            if content and isinstance(content[0], dict) and 'json' in content[0]:
                json_data = content[0]['json']
                entities = json_data.get('entities', [])
                
                print(f"\nExtracted {len(entities)} entities:")
                for i, entity in enumerate(entities, 1):
                    print(f"  {i}. {entity.get('text', 'N/A')} ({entity.get('type', 'N/A')}) - Confidence: {entity.get('confidence', 'N/A')}")
                
                if entities:
                    print("\n✓ Entity extraction working with improved method")
                else:
                    print("\n⚠ No entities extracted")
            else:
                print("\n⚠ Unexpected content structure")
        else:
            print("\n⚠ Unexpected result structure")
        
        # Test relationship mapping
        print("\n--- Testing Relationship Mapping ---")
        if entities:
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
                        print(f"  {i}. {rel.get('source', 'N/A')} --{rel.get('relationship', 'N/A')}--> {rel.get('target', 'N/A')}")
                    
                    if relationships:
                        print("\n✓ Relationship mapping working")
                    else:
                        print("\n⚠ No relationships extracted")
                else:
                    print("\n⚠ Unexpected content structure in relationships")
            else:
                print("\n⚠ Unexpected result structure in relationships")
        else:
            print("Skipping relationship mapping test - no entities to work with")
        
        # Test complete processing
        print("\n--- Testing Complete Processing ---")
        from core.models import AnalysisRequest, DataType
        
        request = AnalysisRequest(
            content=test_text,
            data_type=DataType.TEXT,
            language="en"
        )
        
        result = await kg_agent.process(request)
        
        print(f"Processing result status: {result.status}")
        print(f"Processing result content: {len(result.content) if result.content else 0} items")
        
        if result.status == "completed":
            print("✓ Complete processing working")
        else:
            print(f"⚠ Processing failed with status: {result.status}")
        
        print("\n=== Test Complete ===")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_knowledge_graph_integration())
