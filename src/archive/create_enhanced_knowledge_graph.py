#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def extract_classical_chinese_entities(text):
    """Extract entities from Classical Chinese text using domain-specific patterns."""
    entities = []
    
    # Extract linguistic terms
    linguistic_terms = re.findall(r'([诸盍焉耳尔叵之其者也乃是所以为])', text)
    for term in set(linguistic_terms):
        entities.append({
            "name": term,
            "type": "LINGUISTIC_TERM",
            "confidence": 0.9,
            "description": f"Classical Chinese linguistic term: {term}",
            "domain": "classical_chinese"
        })
    
    # Extract historical figures
    historical_figures = re.findall(r'(孔子|孟子|韩愈|柳宗元|欧阳修|陶渊明|吕布|刘备|曹操|马岱|启功)', text)
    for figure in set(historical_figures):
        entities.append({
            "name": figure,
            "type": "PERSON",
            "confidence": 0.9,
            "description": f"Historical figure: {figure}",
            "domain": "classical_chinese"
        })
    
    # Extract classical texts
    classical_texts = re.findall(r'《([^》]+)》', text)
    for text_name in set(classical_texts):
        entities.append({
            "name": text_name,
            "type": "WORK",
            "confidence": 0.8,
            "description": f"Classical Chinese text: {text_name}",
            "domain": "classical_chinese"
        })
    
    # Extract lesson titles
    lesson_patterns = [
        r'第([一二三四五六七八九十\d]+)课[：:]\s*([^，。\n]+)',
        r'Lesson\s*(\d+)[：:]\s*([^，。\n]+)'
    ]
    
    for pattern in lesson_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                lesson_num, lesson_title = match
            else:
                lesson_num = match
                lesson_title = f"Lesson {match}"
            
            entities.append({
                "name": f"Lesson {lesson_num}: {lesson_title}",
                "type": "LESSON",
                "confidence": 0.8,
                "description": f"Classical Chinese lesson: {lesson_title}",
                "domain": "classical_chinese"
            })
    
    # Extract authors and works
    authors = ["Kai Li", "James Erwin Dew", "李恺", "杜尔文", "启功"]
    works = ["文言章句", "Classical Chinese", "A Functional Approach"]
    
    for author in authors:
        if author in text:
            entities.append({
                "name": author,
                "type": "PERSON",
                "confidence": 0.9,
                "description": f"Author: {author}",
                "domain": "classical_chinese"
            })
    
    for work in works:
        if work in text:
            entities.append({
                "name": work,
                "type": "WORK",
                "confidence": 0.9,
                "description": f"Textbook: {work}",
                "domain": "classical_chinese"
            })
    
    return entities

def extract_relationships(text, entities):
    """Extract relationships between entities."""
    relationships = []
    entity_names = [e["name"] for e in entities]
    entity_dict = {e["name"]: e for e in entities}
    
    # Author-Work relationships
    author_work_patterns = [
        r'(Kai Li|James Erwin Dew|李恺|杜尔文).*?(文言章句|Classical Chinese)',
        r'(启功).*?(文言章句)',
    ]
    
    for pattern in author_work_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                author, work = match
            else:
                continue
            
            if author in entity_dict and work in entity_dict:
                relationships.append({
                    "source": author,
                    "target": work,
                    "relationship_type": "CREATED_BY",
                    "confidence": 0.9,
                    "description": f"{author} created {work}"
                })
    
    # Work-Lesson relationships
    work_lesson_patterns = [
        r'(文言章句|Classical Chinese).*?(第[一二三四五六七八九十\d]+课|Lesson\s*\d+)',
    ]
    
    for pattern in work_lesson_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                work, lesson = match
            else:
                continue
            
            if work in entity_dict and lesson in entity_dict:
                relationships.append({
                    "source": work,
                    "target": lesson,
                    "relationship_type": "CONTAINS",
                    "confidence": 0.8,
                    "description": f"{work} contains {lesson}"
                })
    
    # Linguistic term relationships
    term_function_patterns = [
        r'([诸盍焉耳尔叵])\s*[是|为]\s*([^，。\n]+)',
        r'([之|其|者|也|乃|是|以|所|为])\s*[的|地|得]\s*([^，。\n]+)',
    ]
    
    for pattern in term_function_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                term, function = match
            else:
                continue
            
            if term in entity_dict:
                relationships.append({
                    "source": term,
                    "target": function.strip(),
                    "relationship_type": "HAS_FUNCTION",
                    "confidence": 0.7,
                    "description": f"{term} has function: {function.strip()}"
                })
    
    # Historical figure - work relationships
    figure_work_patterns = [
        r'(孔子|孟子|韩愈|柳宗元|欧阳修|陶渊明).*?《([^》]+)》',
    ]
    
    for pattern in figure_work_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                figure, work = match
            else:
                continue
            
            if figure in entity_dict and work in entity_dict:
                relationships.append({
                    "source": figure,
                    "target": work,
                    "relationship_type": "AUTHOR_OF",
                    "confidence": 0.8,
                    "description": f"{figure} is author of {work}"
                })
    
    # Example relationships
    example_patterns = [
        r'([诸盍焉耳尔叵]).*?《([^》]+)》',
    ]
    
    for pattern in example_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                term, source = match
            else:
                continue
            
            if term in entity_dict and source in entity_dict:
                relationships.append({
                    "source": term,
                    "target": source,
                    "relationship_type": "EXAMPLE_FROM",
                    "confidence": 0.6,
                    "description": f"Example of {term} from {source}"
                })
    
    # Create hierarchical relationships
    for entity in entities:
        if entity["type"] == "LINGUISTIC_TERM":
            # Connect linguistic terms to lessons
            for lesson_entity in entities:
                if lesson_entity["type"] == "LESSON" and "兼词" in lesson_entity["name"]:
                    relationships.append({
                        "source": lesson_entity["name"],
                        "target": entity["name"],
                        "relationship_type": "TEACHES",
                        "confidence": 0.8,
                        "description": f"{lesson_entity['name']} teaches {entity['name']}"
                    })
        
        elif entity["type"] == "WORK":
            # Connect works to their authors
            for author_entity in entities:
                if author_entity["type"] == "PERSON":
                    if "Kai Li" in author_entity["name"] or "James" in author_entity["name"]:
                        relationships.append({
                            "source": author_entity["name"],
                            "target": entity["name"],
                            "relationship_type": "CREATED_BY",
                            "confidence": 0.9,
                            "description": f"{author_entity['name']} created {entity['name']}"
                        })
    
    return relationships

def create_knowledge_graph(entities, relationships):
    """Create a NetworkX graph from entities and relationships."""
    G = nx.DiGraph()
    
    # Add nodes
    for entity in entities:
        G.add_node(entity["name"], 
                  type=entity["type"],
                  confidence=entity["confidence"],
                  domain=entity["domain"],
                  description=entity["description"])
    
    # Add edges
    for rel in relationships:
        if rel["source"] in G and rel["target"] in G:
            G.add_edge(rel["source"], rel["target"],
                      relationship_type=rel["relationship_type"],
                      confidence=rel["confidence"],
                      description=rel["description"])
    
    return G

def generate_html_report(G, output_path="enhanced_classical_chinese_report.html"):
    """Generate an HTML report for the knowledge graph."""
    
    # Prepare data for visualization
    nodes_data = []
    for node, data in G.nodes(data=True):
        nodes_data.append({
            "id": node,
            "label": node,
            "type": data.get("type", "unknown"),
            "domain": data.get("domain", "general"),
            "confidence": data.get("confidence", 0.5)
        })
    
    edges_data = []
    for source, target, data in G.edges(data=True):
        edges_data.append({
            "source": source,
            "target": target,
            "type": data.get("relationship_type", "related"),
            "confidence": data.get("confidence", 0.5)
        })
    
    # Create HTML template
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Classical Chinese Knowledge Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #3498db;
        }}
        
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .stat-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        
        .graph-container {{
            border: 2px solid #ecf0f1;
            border-radius: 10px;
            padding: 20px;
            background: #f8f9fa;
            height: 600px;
            position: relative;
        }}
        
        .node {{
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .node:hover {{
            stroke-width: 3px;
        }}
        
        .link {{
            stroke: #95a5a6;
            stroke-width: 2px;
            transition: all 0.3s ease;
        }}
        
        .link:hover {{
            stroke: #e74c3c;
            stroke-width: 4px;
        }}
        
        .tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
        }}
        
        .legend {{
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }}
        
        .legend h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        
        .legend-item {{
            display: inline-block;
            margin: 5px 15px 5px 0;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Enhanced Classical Chinese Knowledge Graph</h1>
            <p>Comprehensive analysis of Classical Chinese educational content with meaningful relationships</p>
        </div>
        
        <div class="content">
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{len(nodes_data)}</div>
                    <div class="stat-label">Entities</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(edges_data)}</div>
                    <div class="stat-label">Relationships</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(set(edge['type'] for edge in edges_data))}</div>
                    <div class="stat-label">Relationship Types</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(set(node['type'] for node in nodes_data))}</div>
                    <div class="stat-label">Entity Types</div>
                </div>
            </div>
            
            <div class="graph-container" id="graph">
                <div class="tooltip" id="tooltip" style="display: none;"></div>
            </div>
            
            <div class="legend">
                <h3>Entity Types</h3>
                <div class="legend-item" style="background: #e74c3c;">PERSON</div>
                <div class="legend-item" style="background: #3498db;">WORK</div>
                <div class="legend-item" style="background: #2ecc71;">LINGUISTIC_TERM</div>
                <div class="legend-item" style="background: #f39c12;">LESSON</div>
                <div class="legend-item" style="background: #9b59b6;">CONCEPT</div>
            </div>
        </div>
    </div>

    <script>
        // Graph data
        const nodes = {json.dumps(nodes_data)};
        const edges = {json.dumps(edges_data)};
        
        // Setup
        const width = document.getElementById('graph').clientWidth;
        const height = 600;
        
        const svg = d3.select('#graph')
            .append('svg')
            .attr('width', width)
            .attr('height', height);
        
        // Create force simulation
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(edges).id(d => d.id).distance(100))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(30));
        
        // Create links
        const link = svg.append('g')
            .selectAll('line')
            .data(edges)
            .enter().append('line')
            .attr('class', 'link')
            .attr('stroke-width', d => Math.sqrt(d.confidence) * 3);
        
        // Create nodes
        const node = svg.append('g')
            .selectAll('circle')
            .data(nodes)
            .enter().append('circle')
            .attr('class', 'node')
            .attr('r', d => d.type === 'PERSON' ? 8 : d.type === 'WORK' ? 10 : 6)
            .attr('fill', d => {{
                switch(d.type) {{
                    case 'PERSON': return '#e74c3c';
                    case 'WORK': return '#3498db';
                    case 'LINGUISTIC_TERM': return '#2ecc71';
                    case 'LESSON': return '#f39c12';
                    default: return '#9b59b6';
                }}
            }})
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));
        
        // Add labels
        const label = svg.append('g')
            .selectAll('text')
            .data(nodes)
            .enter().append('text')
            .text(d => d.label)
            .attr('font-size', '10px')
            .attr('dx', 12)
            .attr('dy', 4);
        
        // Tooltip
        const tooltip = d3.select('#tooltip');
        
        node.on('mouseover', function(event, d) {{
            tooltip.style('display', 'block')
                .html(`<strong>${{d.label}}</strong><br/>
                       Type: ${{d.type}}<br/>
                       Domain: ${{d.domain}}<br/>
                       Confidence: ${{(d.confidence * 100).toFixed(1)}}%`)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 10) + 'px');
        }})
        .on('mouseout', function() {{
            tooltip.style('display', 'none');
        }});
        
        // Update positions on simulation tick
        simulation.on('tick', () => {{
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
            
            label
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        }});
        
        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    </script>
</body>
</html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    return output_path

def main():
    """Main function to process Classical Chinese content and create enhanced knowledge graph."""
    
    # Read the extracted text
    try:
        with open("extracted_text.txt", "r", encoding="utf-8") as f:
            text_content = f.read()
        print(f"Loaded text content: {len(text_content)} characters")
    except FileNotFoundError:
        print("extracted_text.txt not found. Please run extract_pdf_text.py first.")
        return
    
    print("Extracting entities from Classical Chinese content...")
    entities = extract_classical_chinese_entities(text_content)
    print(f"Extracted {len(entities)} entities")
    
    print("Extracting relationships...")
    relationships = extract_relationships(text_content, entities)
    print(f"Extracted {len(relationships)} relationships")
    
    print("Creating knowledge graph...")
    G = create_knowledge_graph(entities, relationships)
    
    # Print statistics
    print(f"\nKnowledge Graph Statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.4f}")
    print(f"  Connected Components: {nx.number_connected_components(G.to_undirected())}")
    
    # Show sample entities
    print(f"\nSample Entities:")
    for entity in entities[:10]:
        print(f"  {entity['name']} ({entity['type']}) - {entity['domain']}")
    
    # Show sample relationships
    print(f"\nSample Relationships:")
    for rel in relationships[:10]:
        print(f"  {rel['source']} --[{rel['relationship_type']}]--> {rel['target']}")
    
    # Generate HTML report
    print(f"\nGenerating HTML report...")
    output_path = generate_html_report(G)
    print(f"HTML report generated: {output_path}")
    
    # Save graph data as JSON
    graph_data = {
        "entities": entities,
        "relationships": relationships,
        "statistics": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "connected_components": nx.number_connected_components(G.to_undirected())
        }
    }
    
    with open("enhanced_knowledge_graph_data.json", "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    print(f"Graph data saved to: enhanced_knowledge_graph_data.json")

if __name__ == "__main__":
    main()
