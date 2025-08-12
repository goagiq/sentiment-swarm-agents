#!/usr/bin/env python3
"""
Test script to analyze the current knowledge graph and understand the CONCEPT issue.
"""

import sys
import os
import pickle
import networkx as nx
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def analyze_knowledge_graph():
    """Analyze the current knowledge graph to understand the CONCEPT issue."""
    
    # Path to the knowledge graph file
    graph_file = Path("../Results/knowledge_graphs/knowledge_graph.pkl")
    
    if not graph_file.exists():
        print(f"❌ Knowledge graph file not found: {graph_file}")
        return
    
    try:
        # Load the graph
        with open(graph_file, 'rb') as f:
            graph = pickle.load(f)
        
        print(f"📊 Knowledge Graph Analysis")
        print(f"=" * 50)
        print(f"Total nodes: {graph.number_of_nodes()}")
        print(f"Total edges: {graph.number_of_edges()}")
        print(f"Graph density: {nx.density(graph) if graph.number_of_nodes() > 1 else 0:.4f}")
        
        # Analyze node types
        node_types = {}
        for node, attrs in graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            if node_type not in node_types:
                node_types[node_type] = []
            node_types[node_type].append(node)
        
        print(f"\n📋 Node Type Distribution:")
        for node_type, nodes in node_types.items():
            print(f"  {node_type}: {len(nodes)} nodes")
            if len(nodes) <= 5:  # Show all nodes if 5 or fewer
                for node in nodes:
                    confidence = graph.nodes[node].get('confidence', 'N/A')
                    language = graph.nodes[node].get('language', 'N/A')
                    print(f"    - {node} (confidence: {confidence}, language: {language})")
            else:
                # Show first 3 and last 2
                for node in nodes[:3]:
                    confidence = graph.nodes[node].get('confidence', 'N/A')
                    language = graph.nodes[node].get('language', 'N/A')
                    print(f"    - {node} (confidence: {confidence}, language: {language})")
                print(f"    ... and {len(nodes) - 5} more")
        
        # Analyze edges
        if graph.number_of_edges() > 0:
            print(f"\n🔗 Edge Analysis:")
            edge_types = {}
            for source, target, attrs in graph.edges(data=True):
                rel_type = attrs.get('relationship_type', 'unknown')
                if rel_type not in edge_types:
                    edge_types[rel_type] = []
                edge_types[rel_type].append((source, target))
            
            for rel_type, edges in edge_types.items():
                print(f"  {rel_type}: {len(edges)} relationships")
                if len(edges) <= 3:
                    for source, target in edges:
                        print(f"    - {source} -> {target}")
        else:
            print(f"\n❌ No relationships found in the graph!")
        
        # Analyze language distribution
        languages = {}
        for node, attrs in graph.nodes(data=True):
            lang = attrs.get('language', 'unknown')
            if lang not in languages:
                languages[lang] = 0
            languages[lang] += 1
        
        print(f"\n🌍 Language Distribution:")
        for lang, count in languages.items():
            print(f"  {lang}: {count} nodes")
        
        # Check for potential issues
        print(f"\n🔍 Potential Issues:")
        
        # Check if all entities are CONCEPT
        if 'CONCEPT' in node_types and len(node_types['CONCEPT']) == graph.number_of_nodes():
            print(f"  ❌ ALL entities are categorized as CONCEPT - this indicates a categorization issue")
        
        # Check if no relationships exist
        if graph.number_of_edges() == 0:
            print(f"  ❌ No relationships found - this indicates a relationship mapping issue")
        
        # Check confidence scores
        low_confidence_nodes = []
        for node, attrs in graph.nodes(data=True):
            confidence = attrs.get('confidence', 0)
            if confidence < 0.5:
                low_confidence_nodes.append((node, confidence))
        
        if low_confidence_nodes:
            print(f"  ⚠️  {len(low_confidence_nodes)} nodes have low confidence (< 0.5)")
            for node, conf in low_confidence_nodes[:5]:
                print(f"    - {node}: {conf}")
        
        # Check for duplicate or similar entity names
        entity_names = list(graph.nodes())
        potential_duplicates = []
        for i, name1 in enumerate(entity_names):
            for name2 in entity_names[i+1:]:
                if name1.lower() == name2.lower() or name1.lower() in name2.lower() or name2.lower() in name1.lower():
                    potential_duplicates.append((name1, name2))
        
        if potential_duplicates:
            print(f"  ⚠️  {len(potential_duplicates)} potential duplicate entities found")
            for name1, name2 in potential_duplicates[:3]:
                print(f"    - {name1} vs {name2}")
        
    except Exception as e:
        print(f"❌ Error analyzing knowledge graph: {e}")
        import traceback
        traceback.print_exc()

async def test_entity_extraction():
    """Test entity extraction to see what's happening."""
    print("\n🧪 Testing Entity Extraction")
    print("=" * 50)
    
    try:
        from src.agents.entity_extraction_agent import EntityExtractionAgent
        
        # Create agent
        agent = EntityExtractionAgent()
        
        # Test text
        test_text = """
        Artificial Intelligence (AI) is transforming the world. 
        Companies like Google, Microsoft, and OpenAI are leading the development.
        Machine learning algorithms are being used in healthcare, finance, and education.
        Deep learning models like GPT-4 and BERT have revolutionized natural language processing.
        """
        
        print(f"Test text: {test_text.strip()}")
        
        # Extract entities
        result = await agent.extract_entities_enhanced(test_text)
        
        print("\nExtracted entities:")
        if 'entities' in result:
            for entity in result['entities']:
                print(f"  - {entity.get('name', 'N/A')} "
                      f"({entity.get('type', 'N/A')}) - "
                      f"confidence: {entity.get('confidence', 'N/A')}")
        else:
            print(f"  No entities found or unexpected result format: {result}")
            
    except Exception as e:
        print(f"❌ Error testing entity extraction: {e}")
        import traceback
        traceback.print_exc()


async def test_relationship_mapping():
    """Test relationship mapping to see what's happening."""
    print("\n🔗 Testing Relationship Mapping")
    print("=" * 50)
    
    try:
        from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
        
        # Create agent
        agent = KnowledgeGraphAgent()
        
        # Test entities
        test_entities = [
            {"name": "Artificial Intelligence", "type": "TECHNOLOGY", 
             "confidence": 0.9},
            {"name": "Google", "type": "ORGANIZATION", "confidence": 0.9},
            {"name": "Machine Learning", "type": "TECHNOLOGY", 
             "confidence": 0.9},
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

if __name__ == "__main__":
    import asyncio
    
    # Analyze current graph
    analyze_knowledge_graph()
    
    # Test entity extraction
    asyncio.run(test_entity_extraction())
    
    # Test relationship mapping
    asyncio.run(test_relationship_mapping())
