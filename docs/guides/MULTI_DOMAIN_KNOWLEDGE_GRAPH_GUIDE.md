# Multi-Domain Knowledge Graph System Guide

## Overview

The Multi-Domain Knowledge Graph System solves the "one pot" problem by implementing language-based content isolation while maintaining cross-domain relationship capabilities. This follows W3C RDF dataset best practices and provides flexible querying and visualization options.

## Architecture

### Core Components

1. **Multi-Domain Knowledge Graph Agent** (`src/agents/multi_domain_knowledge_graph_agent.py`)
   - Language-based domain separation
   - Cross-domain relationship detection
   - Topic categorization within domains
   - Flexible querying patterns

2. **Multi-Domain Visualization Agent** (`src/agents/multi_domain_visualization_agent.py`)
   - Separate domain visualizations
   - Combined view with filtering
   - Hierarchical domain relationships
   - Interactive Plotly charts

### Language Domains

The system supports 10 language domains:

| Language Code | Language Name | Color |
|---------------|---------------|-------|
| en | English | Blue |
| zh | Chinese | Orange |
| es | Spanish | Green |
| fr | French | Red |
| de | German | Purple |
| ja | Japanese | Brown |
| ko | Korean | Pink |
| ar | Arabic | Gray |
| ru | Russian | Yellow-green |
| pt | Portuguese | Cyan |

### Topic Categories

Within each language domain, content is categorized by topics:

- **economics**: trade, finance, market, economy, business
- **politics**: government, policy, election, political, diplomacy
- **social**: society, culture, social, community, people
- **science**: research, technology, science, innovation, discovery
- **war**: conflict, military, war, defense, security
- **tech**: technology, digital, software, hardware, innovation

## Key Features

### 1. Content Isolation
- Each language gets its own separate graph
- No mixing of unrelated content
- Clean, domain-specific reports and visualizations

### 2. Cross-Domain Relationships
- Automatic detection of entities appearing in multiple languages
- Cross-domain relationship mapping
- Support for international topics (e.g., "Trump tariff war")

### 3. Flexible Querying
- **Domain-specific queries**: Search within a single language domain
- **Cross-domain queries**: Find entities across multiple domains
- **Comprehensive queries**: Search all domains simultaneously

### 4. Multiple Visualization Modes
- **Separate graphs per domain**: Clean, isolated views
- **Combined view with filtering**: Select specific domains and topics
- **Hierarchical view**: Show domain relationships and structure

## Usage Examples

### Basic Usage

```python
from src.agents.multi_domain_knowledge_graph_agent import MultiDomainKnowledgeGraphAgent

# Initialize the agent
agent = MultiDomainKnowledgeGraphAgent()

# Process content (language detection is automatic)
result = await agent.process(request)

# Query within a specific domain
domain_results = await agent.query_domain("Trump", "en")

# Query across domains
cross_domain_results = await agent.query_cross_domain("Trump")

# Query all domains
all_results = await agent.query_all_domains("Trump")
```

### Visualization Usage

```python
from src.agents.multi_domain_visualization_agent import MultiDomainVisualizationAgent

# Initialize visualization agent
viz_agent = MultiDomainVisualizationAgent()

# Create separate domain visualizations
separate_results = await viz_agent.visualize_separate_domains(graphs_data)

# Create combined view with filtering
combined_results = await viz_agent.visualize_combined_view(
    graphs_data,
    options={
        "selected_domains": ["en", "zh"],
        "selected_topics": ["politics", "economics"],
        "max_nodes": 100
    }
)

# Create hierarchical view
hierarchical_results = await viz_agent.visualize_hierarchical_view(graphs_data)
```

## Benefits Over Single Graph Approach

### 1. Cleaner Reports
- **Before**: All content mixed together, cluttered visualizations
- **After**: Domain-specific reports, clean and focused

### 2. Better Performance
- **Before**: Large single graph, slow queries
- **After**: Smaller domain graphs, faster targeted queries

### 3. Improved Analysis
- **Before**: Difficult to analyze domain-specific patterns
- **After**: Easy domain comparison and topic analysis

### 4. Flexible Querying
- **Before**: All-or-nothing queries
- **After**: Domain-specific, cross-domain, or comprehensive queries

## Implementation Details

### Graph Storage
- Each domain stored separately: `knowledge_graph_{lang_code}.pkl`
- Cross-domain relationships: `cross_domain_graph.pkl`
- Automatic loading and saving

### Language Detection
- Character-based detection for major languages
- Extensible for additional languages
- Fallback to English for unknown languages

### Entity Extraction
- Domain-aware entity extraction
- Topic categorization
- Confidence scoring

### Relationship Mapping
- Domain-specific relationship types
- Cross-domain relationship detection
- Confidence scoring

## Migration from Single Graph

### Option 1: Fresh Start (Recommended)
```python
# Initialize new multi-domain system
agent = MultiDomainKnowledgeGraphAgent()

# Process existing content with new system
for content in existing_content:
    result = await agent.process(content)
```

### Option 2: Gradual Migration
```python
# Keep both systems running
old_agent = KnowledgeGraphAgent()
new_agent = MultiDomainKnowledgeGraphAgent()

# Gradually migrate content
for content in existing_content:
    # Process with new system
    new_result = await new_agent.process(content)
    
    # Keep old system for comparison
    old_result = await old_agent.process(content)
```

## Best Practices

### 1. Content Processing
- Let the system automatically detect language
- Review topic categorization for accuracy
- Monitor cross-domain relationship detection

### 2. Querying
- Use domain-specific queries for focused results
- Use cross-domain queries for international topics
- Use comprehensive queries for broad searches

### 3. Visualization
- Use separate domain views for focused analysis
- Use combined views for cross-domain comparison
- Use hierarchical views for structural understanding

### 4. Performance
- Limit node counts in visualizations for large graphs
- Use topic filtering to focus on relevant content
- Monitor graph sizes and optimize as needed

## Troubleshooting

### Common Issues

1. **Language Detection Errors**
   - Check character encoding
   - Verify language detection patterns
   - Use explicit language specification if needed

2. **Cross-Domain Relationship Issues**
   - Verify entity name consistency
   - Check for transliteration differences
   - Review relationship confidence scores

3. **Visualization Performance**
   - Reduce max_nodes parameter
   - Use topic filtering
   - Consider separate domain views

### Debugging

```python
# Get comprehensive statistics
stats = await agent.get_domain_statistics()

# Analyze specific domain
domain_report = await agent.generate_domain_report("en")

# Check cross-domain connections
cross_connections = await agent.analyze_cross_domain_connections()
```

## Future Enhancements

### Planned Features
1. **Advanced Language Detection**: Integration with language detection libraries
2. **Entity Linking**: Better cross-domain entity matching
3. **Semantic Similarity**: Enhanced relationship detection
4. **Real-time Updates**: Live graph updates
5. **API Integration**: REST API for external access

### Extensibility
- Add new language domains
- Customize topic categories
- Implement custom visualization types
- Add domain-specific processing rules

## Conclusion

The Multi-Domain Knowledge Graph System provides a robust solution to the content isolation problem while maintaining the flexibility to analyze cross-domain relationships. It follows industry best practices and provides multiple visualization options to meet different analysis needs.

The system is designed to be scalable, maintainable, and extensible, making it suitable for both current needs and future growth.
