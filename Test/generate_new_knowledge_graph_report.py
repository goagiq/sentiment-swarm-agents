#!/usr/bin/env python3
"""
Generate a brand new knowledge graph report with enhanced entity categorization.
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.knowledge_graph_agent import KnowledgeGraphAgent
from config.config import config


async def generate_new_knowledge_graph_report():
    """Generate a brand new knowledge graph report with fresh data."""
    
    print("Generating Brand New Knowledge Graph Report")
    print("=" * 60)
    
    # Initialize the knowledge graph agent
    agent = KnowledgeGraphAgent(model_name=config.model.default_text_model)
    
    # Fresh test data with various entity types
    test_text = """
    Artificial Intelligence and Machine Learning are transforming the technology industry. 
    Elon Musk's Tesla and SpaceX are leading innovation in electric vehicles and space exploration. 
    Microsoft CEO Satya Nadella and Apple's Tim Cook are competing in the AI race. 
    The United States government is investing heavily in AI research and development. 
    China's tech companies like Alibaba and Tencent are also advancing in AI technology. 
    Stanford University and MIT are conducting groundbreaking research in deep learning. 
    The European Union is implementing new AI regulations to ensure ethical development. 
    Google's DeepMind and OpenAI are developing advanced language models. 
    Climate change and renewable energy are becoming critical global priorities. 
    The World Health Organization is using AI to improve healthcare delivery worldwide.
    """
    
    print(f"Processing Text: {test_text.strip()}")
    print()
    
    try:
        # Step 1: Extract entities with enhanced categorization
        print("Step 1: Extracting Entities with Enhanced Categorization...")
        print("-" * 60)
        
        entities_result = await agent.extract_entities(test_text, language="en")
        entities = entities_result.get("content", [{}])[0].get("json", {}).get("entities", [])
        
        print(f"Extracted {len(entities)} entities:")
        
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
        
        # Step 2: Map relationships between entities
        print("Step 2: Mapping Relationships Between Entities...")
        print("-" * 60)
        
        relationships_result = await agent.map_relationships(test_text, entities)
        relationships = relationships_result.get("relationships", [])
        
        print(f"Mapped {len(relationships)} relationships:")
        for rel in relationships:
            source = rel.get("source", "Unknown")
            target = rel.get("target", "Unknown")
            rel_type = rel.get("relationship_type", "Unknown")
            confidence = rel.get("confidence", 0.0)
            print(f"  - {source} --[{rel_type}]--> {target} (confidence: {confidence:.2f})")
        
        print()
        
        # Step 3: Add entities and relationships to the graph
        print("Step 3: Building Knowledge Graph...")
        print("-" * 60)
        
        await agent._add_to_graph(entities, relationships, f"new_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}", "en")
        
        print(f"Added {len(entities)} entities and {len(relationships)} relationships to the graph")
        
        # Step 4: Generate comprehensive HTML report
        print("\nStep 4: Generating Comprehensive HTML Report...")
        print("-" * 60)
        
        # Generate report with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"Results/reports/new_knowledge_graph_report_{timestamp}.html")
        
        report_result = await agent.generate_graph_report(
            output_path=str(output_path), 
            target_language="en"
        )
        
        print(f"Report generated: {output_path}")
        print(f"Report status: {report_result.get('status', 'Unknown')}")
        
        if report_result.get("status") == "success":
            print("✅ Brand new knowledge graph report generated successfully!")
            print(f"✅ HTML report: {output_path}")
            
            # Get graph statistics
            graph_stats = agent._get_graph_stats()
            print(f"\n📊 Graph Statistics:")
            print(f"   - Total Nodes: {graph_stats.get('total_nodes', 0)}")
            print(f"   - Total Edges: {graph_stats.get('total_edges', 0)}")
            print(f"   - Entity Types: {graph_stats.get('entity_types', 0)}")
            print(f"   - Relationship Types: {graph_stats.get('relationship_types', 0)}")
            
        else:
            print("❌ Report generation failed")
            print(f"Error: {report_result.get('error', 'Unknown error')}")
        
        # Step 5: Generate additional analysis
        print("\nStep 5: Generating Additional Analysis...")
        print("-" * 60)
        
        # Query the knowledge graph
        query_result = await agent.query_knowledge_graph("What are the main AI companies and their leaders?", "en")
        print("Knowledge Graph Query Results:")
        print(f"Query: What are the main AI companies and their leaders?")
        print(f"Results: {query_result.get('results', 'No results')}")
        
        # Find entity paths
        if len(entities) >= 2:
            source_entity = entities[0].get("text", "Unknown")
            target_entity = entities[-1].get("text", "Unknown")
            path_result = await agent.find_entity_paths(source_entity, target_entity)
            print(f"\nPath Analysis: {source_entity} -> {target_entity}")
            print(f"Path Result: {path_result.get('path', 'No path found')}")
        
        print("\n" + "=" * 60)
        print("✅ Brand new knowledge graph report generation completed!")
        print(f"📁 Report location: {output_path}")
        
        return {
            "status": "success",
            "report_path": str(output_path),
            "entities_count": len(entities),
            "relationships_count": len(relationships),
            "graph_stats": graph_stats
        }
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }


async def main():
    """Main function to generate the report."""
    print("Brand New Knowledge Graph Report Generator")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    result = await generate_new_knowledge_graph_report()
    
    print("\n" + "=" * 60)
    if result["status"] == "success":
        print("🎉 Report generation completed successfully!")
        print(f"📊 Summary:")
        print(f"   - Entities: {result['entities_count']}")
        print(f"   - Relationships: {result['relationships_count']}")
        print(f"   - Report: {result['report_path']}")
    else:
        print("❌ Report generation failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    asyncio.run(main())
