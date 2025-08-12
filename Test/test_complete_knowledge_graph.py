#!/usr/bin/env python3
"""
Comprehensive test for the complete knowledge graph integration.
"""

import sys
import os
import asyncio

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

async def test_complete_knowledge_graph():
    """Test the complete knowledge graph integration."""
    print("🧪 Testing Complete Knowledge Graph Integration")
    print("=" * 60)
    
    try:
        from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
        
        # Create agent
        agent = KnowledgeGraphAgent()
        
        # Test text with clear entities
        test_text = """
        Artificial Intelligence (AI) is transforming the world. 
        Google and Microsoft are leading the development of machine learning.
        These technologies are being used in healthcare applications.
        OpenAI has created advanced language models like GPT-4.
        """
        
        print(f"Test text: {test_text.strip()}")
        
        # Extract entities using the knowledge graph agent
        entities_result = await agent.extract_entities(test_text)
        entities = entities_result.get('entities', [])
        
        print(f"\n📋 Extracted {len(entities)} entities:")
        for entity in entities:
            name = entity.get('name', 'N/A')
            entity_type = entity.get('type', 'N/A')
            confidence = entity.get('confidence', 'N/A')
            print(f"  - {name} ({entity_type}) - confidence: {confidence}")
        
        if entities:
            # Map relationships
            relationships_result = await agent.map_relationships(test_text, entities)
            
            print(f"\n🔗 Relationship mapping result:")
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
            
        # Test generating a graph report
        print(f"\n📊 Testing graph report generation...")
        try:
            report_result = await agent.generate_graph_report()
            print(f"  Report generation: {report_result.get('status', 'unknown')}")
        except Exception as e:
            print(f"  Report generation failed: {e}")
            
    except Exception as e:
        print(f"❌ Error testing knowledge graph integration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_complete_knowledge_graph())
