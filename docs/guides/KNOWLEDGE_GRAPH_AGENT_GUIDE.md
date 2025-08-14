# Knowledge Graph Agent Guide

## Overview

The Knowledge Graph Agent is a sophisticated component that implements GraphRAG-inspired approaches for entity extraction, relationship mapping, and graph analysis. It provides advanced capabilities for building and querying knowledge graphs from various types of content.

## Key Features

### 🚀 GraphRAG-Inspired Architecture
- **Chunk-based Processing**: Intelligent text splitting with 1200 token chunks and 100 token overlap
- **Advanced Entity Extraction**: Sophisticated prompts with entity type categorization and confidence scoring
- **Comprehensive Relationship Mapping**: Multiple relationship types with context-aware inference
- **Community Detection**: Multiple algorithms (Louvain, Label Propagation, Girvan-Newman)
- **Robust Error Handling**: Multiple fallback strategies for JSON parsing and entity extraction

### 📊 Entity Types Supported
- **PERSON**: Individuals, characters, people
- **ORGANIZATION**: Companies, institutions, groups
- **LOCATION**: Places, countries, cities
- **EVENT**: Occurrences, happenings, incidents
- **CONCEPT**: Abstract ideas, theories, concepts
- **OBJECT**: Physical items, artifacts, objects
- **TECHNOLOGY**: Tools, systems, platforms
- **METHOD**: Processes, techniques, approaches
- **PROCESS**: Procedures, workflows, operations

### 🔗 Relationship Types
- **IS_A**: Hierarchical relationships
- **PART_OF**: Component relationships
- **LOCATED_IN**: Spatial relationships
- **WORKS_FOR**: Employment relationships
- **CREATED_BY**: Attribution relationships
- **USES**: Dependency relationships
- **IMPLEMENTS**: Implementation relationships
- **SIMILAR_TO**: Similarity relationships
- **OPPOSES**: Opposition relationships
- **SUPPORTS**: Support relationships
- **LEADS_TO**: Causal relationships
- **DEPENDS_ON**: Dependency relationships
- **RELATED_TO**: General relationships

### Supported Data Types

- Text documents
- Audio transcripts
- Video content
- Web pages
- PDF documents
- Social media content

## Architecture

### Integration with Existing System

The Knowledge Graph Agent follows the same patterns as other agents in the system:

- Inherits from `StrandsBaseAgent`
- Uses the same Ollama models for processing
- Integrates with ChromaDB for storage
- Follows the established request/response pattern

### Graph Storage

- Uses NetworkX for graph operations
- Stores graphs in pickle format for persistence
- Integrates with existing vector database for metadata storage

## Usage

### Basic Usage

```python
from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.core.models import AnalysisRequest, DataType

# Initialize the agent
agent = KnowledgeGraphAgent()

# Process content
request = AnalysisRequest(
    data_type=DataType.TEXT,
    content="Your text content here",
    language="en"
)

result = await agent.process(request)
```

### Individual Tools

#### Entity Extraction

```python
entities_result = await agent.extract_entities(text_content)
```

#### Relationship Mapping

```python
relationships_result = await agent.map_relationships(text_content, entities)
```

#### Graph Querying

```python
query_result = await agent.query_knowledge_graph("your query here")
```

#### Community Analysis

```python
communities_result = await agent.analyze_graph_communities()
```

#### Path Finding

```python
path_result = await agent.find_entity_paths("source_entity", "target_entity")
```

#### Entity Context

```python
context_result = await agent.get_entity_context("entity_name")
```

#### Graph Report Generation

The agent generates both PNG and interactive HTML visualizations:

```python
report_result = await agent.generate_graph_report()
# Returns both PNG and HTML files
print(f"PNG file: {{report_result['png_file']}}")
print(f"HTML file: {{report_result['html_file']}}")
```

The HTML report includes:
- Interactive D3.js visualization
- Draggable nodes
- Hover tooltips with entity details
- Multiple view modes (Main, Communities, Centrality, Relationships)
- Color-coded entity types
- Graph statistics and metadata

## MCP Server Integration

The Knowledge Graph Agent includes an MCP (Model Context Protocol) server for integration with external tools and applications.

### Available MCP Tools

1. `extract_entities` - Extract entities from text
2. `map_relationships` - Map relationships between entities
3. `query_knowledge_graph` - Query the knowledge graph
4. `generate_graph_report` - Generate visual graph reports
5. `analyze_graph_communities` - Analyze graph communities
6. `find_entity_paths` - Find paths between entities
7. `get_entity_context` - Get entity context
8. `process_content` - Process content and build graph

### Running the MCP Server

```bash
python src/mcp/knowledge_graph_agent_server.py
```

## Configuration

### Agent Configuration

Add to your configuration:

```python
# Knowledge Graph settings
knowledge_graph_agent_capacity: int = 5
graph_storage_path: str = "./data/knowledge_graphs"
enable_graph_visualization: bool = True
max_graph_nodes: int = 10000
max_graph_edges: int = 50000
```

### Dependencies

Required packages:
- `networkx` - Graph operations
- `matplotlib` - Graph visualization
- `chromadb` - Vector database integration

## Examples

### Demo Script

Run the demo to see the agent in action:

```bash
python examples/knowledge_graph_agent_demo.py
```

### Test Script

Run tests to verify functionality:

```bash
python Test/test_knowledge_graph_agent.py
```

## Graph Analysis Features

### Community Detection

The agent can identify communities within the knowledge graph using the Louvain method, helping to understand how entities cluster together.

### Path Finding

Find connections between any two entities in the graph, useful for understanding relationships and discovering indirect connections.

### Centrality Analysis

Calculate entity importance based on their position in the graph network.

### Graph Statistics

- Node count
- Edge count
- Graph density
- Average clustering coefficient
- Number of connected components

## Visual Reports

The agent generates comprehensive visual graph reports including:

### PNG Reports
- Static graph visualization with matplotlib
- Entity nodes with different colors by type
- Relationship edges with labels
- Graph statistics and metadata
- Community clusters
- Centrality indicators

### Interactive HTML Reports
- Dynamic D3.js visualization
- Draggable and interactive nodes
- Hover tooltips with detailed entity information
- Multiple view modes for different analysis perspectives
- Color-coded entity categorization
- Real-time graph statistics
- Responsive design for different screen sizes

Reports are saved as high-resolution PNG files with timestamps.

## Integration with Other Agents

The Knowledge Graph Agent works alongside other agents:

1. **Text Agent**: Processes extracted text content
2. **Audio Agent**: Works with transcribed audio content
3. **Video Agent**: Processes video transcripts and metadata
4. **Web Agent**: Analyzes webpage content
5. **OCR Agent**: Processes PDF and image text

## Best Practices

1. **Content Quality**: Higher quality input content leads to better entity extraction
2. **Regular Updates**: Process new content regularly to keep the graph current
3. **Graph Maintenance**: Monitor graph size and performance
4. **Query Optimization**: Use specific queries for better results
5. **Backup**: Regularly backup graph data

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Graph Size**: Monitor memory usage with large graphs
3. **Model Availability**: Ensure Ollama models are running
4. **Storage Issues**: Check disk space for graph storage

### Performance Tips

1. Use appropriate graph size limits
2. Enable graph visualization only when needed
3. Regular graph cleanup and optimization
4. Monitor processing times for large content

## Future Enhancements

Potential improvements:

1. **Advanced NLP**: Integration with specialized NER models
2. **Graph Embeddings**: Vector representations of graph entities
3. **Temporal Analysis**: Time-based relationship tracking
4. **Multi-language Support**: Enhanced language processing
5. **Real-time Updates**: Live graph updates from streaming data

## API Reference

### KnowledgeGraphAgent Class

#### Methods

- `process(request)`: Process content and build graph
- `extract_entities(text)`: Extract entities from text
- `map_relationships(text, entities)`: Map entity relationships
- `query_knowledge_graph(query)`: Query the knowledge graph
- `generate_graph_report(output_path)`: Generate visual report
- `analyze_graph_communities()`: Analyze graph communities
- `find_entity_paths(source, target)`: Find paths between entities
- `get_entity_context(entity)`: Get entity context

#### Properties

- `graph`: NetworkX graph object
- `metadata`: Agent metadata and statistics
- `graph_stats`: Current graph statistics

## Contributing

When contributing to the Knowledge Graph Agent:

1. Follow existing code patterns
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility
5. Test with various content types
