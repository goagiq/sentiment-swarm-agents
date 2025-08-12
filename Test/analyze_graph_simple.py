#!/usr/bin/env python3
"""
Simple script to analyze the current knowledge graph state.
"""

import sys
import os
import pickle
import networkx as nx
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def analyze_graph():
    """Analyze the current knowledge graph."""
    
    # Path to the knowledge graph file
    graph_file = Path("Results/knowledge_graphs/knowledge_graph.pkl")
    
    if not graph_file.exists():
        print(f"❌ Knowledge graph file not found: {graph_file}")
        return
    
    try:
        # Load the graph
        with open(graph_file, 'rb') as f:
            graph = pickle.load(f)
        
        print("📊 Knowledge Graph Analysis")
        print("=" * 50)
        print(f"Total nodes: {graph.number_of_nodes()}")
        print(f"Total edges: {graph.number_of_edges()}")
        
        if graph.number_of_nodes() > 1:
            density = nx.density(graph)
            print(f"Graph density: {density:.4f}")
        else:
            print("Graph density: 0.0000")
        
        # Analyze node types
        node_types = {}
        for node, attrs in graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            if node_type not in node_types:
                node_types[node_type] = []
            node_types[node_type].append(node)
        
        print("\n📋 Node Type Distribution:")
        for node_type, nodes in node_types.items():
            print(f"  {node_type}: {len(nodes)} nodes")
            if len(nodes) <= 10:  # Show all nodes if 10 or fewer
                for node in nodes:
                    confidence = graph.nodes[node].get('confidence', 'N/A')
                    language = graph.nodes[node].get('language', 'N/A')
                    print(f"    - {node} (confidence: {confidence}, language: {language})")
            else:
                # Show first 5
                for node in nodes[:5]:
                    confidence = graph.nodes[node].get('confidence', 'N/A')
                    language = graph.nodes[node].get('language', 'N/A')
                    print(f"    - {node} (confidence: {confidence}, language: {language})")
                print(f"    ... and {len(nodes) - 5} more")
        
        # Analyze edges
        if graph.number_of_edges() > 0:
            print("\n🔗 Edge Analysis:")
            edge_types = {}
            for source, target, attrs in graph.edges(data=True):
                rel_type = attrs.get('relationship_type', 'unknown')
                if rel_type not in edge_types:
                    edge_types[rel_type] = []
                edge_types[rel_type].append((source, target))
            
            for rel_type, edges in edge_types.items():
                print(f"  {rel_type}: {len(edges)} relationships")
                if len(edges) <= 5:
                    for source, target in edges:
                        print(f"    - {source} -> {target}")
        else:
            print("\n❌ No relationships found in the graph!")
        
        # Analyze language distribution
        languages = {}
        for node, attrs in graph.nodes(data=True):
            lang = attrs.get('language', 'unknown')
            if lang not in languages:
                languages[lang] = 0
            languages[lang] += 1
        
        print("\n🌍 Language Distribution:")
        for lang, count in languages.items():
            print(f"  {lang}: {count} nodes")
        
        # Check for issues
        print("\n🔍 Issues Found:")
        
        # Check if all entities are CONCEPT
        if 'CONCEPT' in node_types and len(node_types['CONCEPT']) == graph.number_of_nodes():
            print("  ❌ ALL entities are categorized as CONCEPT")
        
        # Check if no relationships exist
        if graph.number_of_edges() == 0:
            print("  ❌ No relationships found")
        
        # Check confidence scores
        low_confidence_nodes = []
        for node, attrs in graph.nodes(data=True):
            confidence = attrs.get('confidence', 0)
            if confidence < 0.5:
                low_confidence_nodes.append((node, confidence))
        
        if low_confidence_nodes:
            print(f"  ⚠️  {len(low_confidence_nodes)} nodes have low confidence (< 0.5)")
        
    except Exception as e:
        print(f"❌ Error analyzing knowledge graph: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_graph()
