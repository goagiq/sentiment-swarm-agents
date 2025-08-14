# Semantic Search & Knowledge Graph Integration Guide

## Overview

The semantic search system provides advanced search capabilities across all content types in your sentiment analysis system. It integrates with your existing knowledge graph and provides multilingual support, conceptual search, and real-time performance.

## Features

### 🔍 Semantic Search Types
- **Semantic Search**: Find similar content using vector embeddings
- **Conceptual Search**: Find related ideas even when exact terms don't match
- **Multilingual Search**: Cross-language semantic search
- **Cross-Content Search**: Search across different content types
- **Combined Search**: Semantic + Knowledge Graph search

### 🌐 Language Support
- **English (en)**: Full support with stemming and synonyms
- **Chinese (zh)**: Character-level analysis with cultural context
- **Russian (ru)**: Morphological analysis
- **Japanese (ja)**: Kanji-level analysis with particle consideration
- **Korean (ko)**: Morpheme-based analysis
- **Arabic (ar)**: Root-based analysis
- **Hindi (hi)**: Morphological analysis
- **Auto-detection**: Automatic language detection

### 📄 Content Type Support
- **Text**: Plain text documents
- **PDF**: PDF documents with text extraction
- **Audio**: Audio transcripts
- **Video**: Video transcripts and descriptions
- **Image**: Image descriptions and metadata
- **Web**: Web page content
- **Document**: General document types

## Architecture

### Core Components

1. **VectorDBManager** (`src/core/vector_db.py`)
   - ChromaDB-based vector storage
   - Semantic search collections
   - Multi-language indexing
   - Performance optimization

2. **SemanticSearchService** (`src/core/semantic_search_service.py`)
   - High-level search interface
   - Multilingual support
   - Translation integration
   - Performance monitoring

3. **SemanticSearchConfig** (`src/config/semantic_search_config.py`)
   - Language-specific settings
   - Search parameters
   - Performance thresholds
   - Cultural context

4. **MCP Tools** (`src/mcp_servers/unified_mcp_server.py`)
   - 5 new semantic search tools
   - Unified interface
   - Error handling
   - Performance optimization

5. **Streamlit UI** (`ui/main.py`)
   - Interactive search interface
   - Real-time results
   - Multiple search types
   - Statistics dashboard

6. **API Endpoints** (`src/api/main.py`)
   - RESTful search endpoints
   - MCP integration
   - Error handling
   - Performance monitoring

## Usage

### Streamlit UI

1. **Start the application**:
   ```bash
   .venv/Scripts/python.exe main.py
   ```

2. **Access the UI**:
   - Main UI: http://localhost:8501
   - Navigate to "Semantic Search" tab

3. **Search Options**:
   - **Semantic Search**: Find similar content
   - **Knowledge Graph**: Query knowledge graph
   - **Combined Search**: Both semantic and KG
   - **Search Statistics**: View index information

### API Usage

#### Semantic Search
```python
import requests

# Basic semantic search
response = requests.post("http://localhost:8003/search/semantic", json={
    "query": "artificial intelligence applications",
    "search_type": "semantic",
    "language": "en",
    "content_types": ["text", "pdf", "document"],
    "n_results": 10,
    "similarity_threshold": 0.7
})

results = response.json()
```

#### Knowledge Graph Search
```python
# Knowledge graph search
response = requests.post("http://localhost:8003/search/knowledge-graph", json={
    "query": "machine learning companies",
    "language": "en"
})

kg_results = response.json()
```

#### Combined Search
```python
# Combined semantic and knowledge graph search
response = requests.post("http://localhost:8003/search/combined", json={
    "query": "AI technology trends",
    "language": "en",
    "n_results": 10,
    "similarity_threshold": 0.7,
    "include_kg_results": True
})

combined_results = response.json()
```

#### Search Statistics
```python
# Get search statistics
response = requests.get("http://localhost:8003/search/statistics")
stats = response.json()
```

### MCP Tools

#### 1. Semantic Search
```python
# Call semantic search MCP tool
result = await call_unified_mcp_tool("semantic_search", {
    "query": "artificial intelligence",
    "search_type": "semantic",
    "language": "en",
    "content_types": ["text", "pdf"],
    "n_results": 10,
    "similarity_threshold": 0.7,
    "include_metadata": True
})
```

#### 2. Multilingual Search
```python
# Call multilingual search MCP tool
result = await call_unified_mcp_tool("multilingual_semantic_search", {
    "query": "technology",
    "target_languages": ["en", "zh", "ru"],
    "n_results": 5,
    "similarity_threshold": 0.7
})
```

#### 3. Conceptual Search
```python
# Call conceptual search MCP tool
result = await call_unified_mcp_tool("conceptual_search", {
    "concept": "machine learning",
    "n_results": 10,
    "similarity_threshold": 0.6
})
```

#### 4. Cross-Content Search
```python
# Call cross-content search MCP tool
result = await call_unified_mcp_tool("cross_content_search", {
    "query": "data analysis",
    "content_types": ["text", "pdf", "document"],
    "n_results": 10,
    "similarity_threshold": 0.7
})
```

#### 5. Combined Search
```python
# Call combined search MCP tool
result = await call_unified_mcp_tool("combined_search", {
    "query": "AI and machine learning",
    "language": "en",
    "n_results": 10,
    "similarity_threshold": 0.7,
    "include_kg_results": True
})
```

## Configuration

### Language-Specific Settings

Each language has specific configuration for optimal search performance:

```python
# English settings
en_settings = {
    "similarity_threshold": 0.7,
    "content_types": ["text", "pdf", "document", "web"],
    "search_strategies": ["semantic", "conceptual", "keyword"],
    "use_stemming": True,
    "use_synonyms": True,
    "case_sensitive": False
}

# Chinese settings
zh_settings = {
    "similarity_threshold": 0.65,
    "content_types": ["text", "pdf", "document"],
    "search_strategies": ["semantic", "conceptual", "character"],
    "use_stemming": False,
    "use_synonyms": True,
    "case_sensitive": False,
    "cultural_context": {
        "use_character_level": True,
        "consider_tone": False
    }
}
```

### Performance Settings

```python
# Performance configuration
performance_settings = {
    "search_timeout_seconds": 30,
    "batch_size": 50,
    "cache_results": True,
    "cache_ttl_seconds": 3600,
    "default_n_results": 10,
    "max_n_results": 100,
    "min_similarity_threshold": 0.3,
    "max_similarity_threshold": 0.95
}
```

## Testing

### Run Test Suite
```bash
# Run comprehensive semantic search tests
.venv/Scripts/python.exe Test/test_semantic_search.py
```

### Test Components
```python
# Test vector database
from src.core.vector_db import VectorDBManager
vector_db = VectorDBManager()

# Test semantic search
results = await vector_db.semantic_search(
    query="artificial intelligence",
    language="en",
    n_results=5,
    similarity_threshold=0.7
)

# Test multilingual search
multilingual_results = await vector_db.multi_language_semantic_search(
    query="technology",
    n_results=3,
    similarity_threshold=0.6
)
```

## Performance Optimization

### Search Performance Targets
- **Semantic Search**: < 2.0 seconds
- **Conceptual Search**: < 3.0 seconds
- **Multilingual Search**: < 5.0 seconds
- **Combined Search**: < 4.0 seconds

### Optimization Strategies
1. **Caching**: Results cached for 1 hour
2. **Batch Processing**: Process multiple queries efficiently
3. **Index Optimization**: Optimized ChromaDB collections
4. **Language-Specific**: Tailored settings per language
5. **Async Processing**: Non-blocking operations

## Integration with Existing System

### Knowledge Graph Integration
The semantic search system works alongside your existing knowledge graph:

1. **Complementary Search**: Semantic search finds content, KG finds relationships
2. **Combined Results**: Both search types in single query
3. **Unified Interface**: Same MCP tools and API endpoints
4. **Shared Configuration**: Common language and content settings

### MCP Framework Compliance
- All operations go through MCP tools
- No direct API access to components
- Unified error handling
- Consistent interface patterns

### File Organization
```
src/
├── core/
│   ├── vector_db.py              # Enhanced with semantic search
│   └── semantic_search_service.py # New semantic search service
├── config/
│   └── semantic_search_config.py # New search configuration
├── mcp_servers/
│   └── unified_mcp_server.py     # Enhanced with 5 new tools
└── api/
    └── main.py                   # Enhanced with search endpoints

ui/
└── main.py                       # Enhanced with search UI

Test/
└── test_semantic_search.py       # New comprehensive test suite

Results/
└── semantic_search_test_results_*.json # Test results
```

## Troubleshooting

### Common Issues

1. **No Search Results**
   - Check if content is indexed
   - Lower similarity threshold
   - Try different search types
   - Verify language settings

2. **Slow Performance**
   - Check system resources
   - Reduce batch size
   - Enable caching
   - Optimize similarity threshold

3. **Language Detection Issues**
   - Use explicit language setting
   - Check content encoding
   - Verify language support

4. **MCP Tool Errors**
   - Check MCP server status
   - Verify tool registration
   - Check parameter validation

### Debug Information

```python
# Get search statistics
stats = await semantic_search_service.get_search_statistics()
print(f"Total documents: {stats['total_documents']}")
print(f"Languages: {stats['supported_languages']}")
print(f"Content types: {stats['supported_content_types']}")

# Check vector database status
vector_db_stats = await vector_db.get_search_statistics()
print(f"Index statistics: {vector_db_stats}")
```

## Best Practices

### Search Queries
1. **Use Natural Language**: "artificial intelligence applications" vs "AI apps"
2. **Be Specific**: "machine learning algorithms" vs "ML"
3. **Consider Context**: Include relevant terms for better matching
4. **Use Multiple Types**: Try semantic, conceptual, and combined search

### Performance
1. **Optimize Thresholds**: Start with 0.7, adjust based on results
2. **Limit Results**: Use appropriate n_results for your use case
3. **Cache Results**: Enable caching for repeated queries
4. **Batch Operations**: Use batch search for multiple queries

### Multilingual Usage
1. **Language Detection**: Use "auto" for unknown languages
2. **Cultural Context**: Consider cultural differences in search
3. **Translation**: Results can be translated back to target language
4. **Mixed Content**: Search across multiple languages simultaneously

## Future Enhancements

### Planned Features
1. **Advanced Filtering**: Date ranges, sentiment filters
2. **Search Analytics**: Query patterns and insights
3. **Personalization**: User-specific search preferences
4. **Real-time Indexing**: Live content indexing
5. **Advanced Visualization**: Interactive search result visualization

### Integration Opportunities
1. **External APIs**: Connect to external search services
2. **Document Processing**: Enhanced PDF and document parsing
3. **Audio/Video**: Improved multimedia search
4. **Social Media**: Real-time social media search
5. **Business Intelligence**: Advanced analytics integration

## Support

For issues and questions:
1. Check the test results in `/Results/`
2. Review the configuration files
3. Run the test suite for diagnostics
4. Check the system logs for errors
5. Verify MCP server status

The semantic search system is designed to complement your existing knowledge graph capabilities while providing enhanced search functionality across all content types and languages.
