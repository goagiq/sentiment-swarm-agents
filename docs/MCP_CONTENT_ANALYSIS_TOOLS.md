# MCP Content Analysis Tools Documentation

## Overview

This document describes the new MCP (Model Context Protocol) content analysis tools that have been implemented to enhance the system's capabilities for text summarization, chapter analysis, content extraction, and comparison.

## New MCP Tools

### 1. Text Summarization Tools

#### `summarize_text_content`
**Description**: Summarize text content with structured analysis and optional key points and entity extraction.

**Parameters**:
- `text` (str): The text content to summarize
- `summary_type` (str): Type of summary ("brief", "comprehensive", "detailed")
- `language` (str): Language code (default: "en")
- `include_key_points` (bool): Whether to include key points (default: True)
- `include_entities` (bool): Whether to include entity extraction (default: True)

**Returns**:
```json
{
  "success": true,
  "summary_type": "comprehensive",
  "summary": "Generated summary text...",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "entities": [{"name": "Entity 1", "type": "PERSON"}],
  "processing_time": 2.5
}
```

**Usage Example**:
```python
# Summarize Chapter 1 of The Art of War
result = await mcp_client.call_tool(
    "summarize_text_content",
    {
        "text": chapter_1_content,
        "summary_type": "comprehensive",
        "include_key_points": True,
        "include_entities": True
    }
)
```

### 2. Chapter Analysis Tools

#### `analyze_chapter_content`
**Description**: Analyze chapter content with structured breakdown and insights.

**Parameters**:
- `chapter_text` (str): The chapter text to analyze
- `chapter_title` (str): Title of the chapter (default: "")
- `language` (str): Language code (default: "en")
- `analysis_type` (str): Type of analysis ("summary", "themes", "entities", "comprehensive")

**Returns**:
```json
{
  "success": true,
  "chapter_title": "Chapter 1: Laying Plans",
  "analysis_type": "comprehensive",
  "summary": "Chapter analysis summary...",
  "entities": [{"name": "Sun Tzu", "type": "PERSON"}],
  "entity_statistics": {"PERSON": 5, "CONCEPT": 3},
  "themes": ["Strategy", "Leadership", "Planning"],
  "key_concepts": ["Five Factors", "Deception", "Preparation"],
  "processing_time": 3.2
}
```

**Usage Example**:
```python
# Analyze Chapter 1
result = await mcp_client.call_tool(
    "analyze_chapter_content",
    {
        "chapter_text": chapter_1_content,
        "chapter_title": "Chapter 1: Laying Plans",
        "analysis_type": "comprehensive"
    }
)
```

### 3. Content Extraction Tools

#### `extract_content_sections`
**Description**: Extract and analyze specific content sections (chapters, paragraphs, etc.).

**Parameters**:
- `content` (str): The full content to extract sections from
- `section_type` (str): Type of sections ("chapters", "paragraphs", "sections")
- `language` (str): Language code (default: "en")
- `include_analysis` (bool): Whether to include analysis for each section (default: True)

**Returns**:
```json
{
  "success": true,
  "section_type": "chapters",
  "sections_found": 3,
  "sections": [
    {
      "title": "Chapter 1: Laying Plans",
      "content": "Chapter content...",
      "section_number": 1,
      "analysis": {
        "summary": "Analysis summary...",
        "entities": [...],
        "themes": [...]
      }
    }
  ]
}
```

**Usage Example**:
```python
# Extract chapters from The Art of War
result = await mcp_client.call_tool(
    "extract_content_sections",
    {
        "content": full_art_of_war_text,
        "section_type": "chapters",
        "include_analysis": True
    }
)
```

### 4. Knowledge Graph Query Tools

#### `query_knowledge_graph`
**Description**: Query the knowledge graph for specific content and relationships.

**Parameters**:
- `query` (str): The query string
- `query_type` (str): Type of query ("entities", "relationships", "concepts", "full_text")
- `language` (str): Language code (default: "en")
- `limit` (int): Maximum number of results (default: 50)

**Returns**:
```json
{
  "success": true,
  "query": "Sun Tzu",
  "query_type": "entities",
  "results": [
    {
      "name": "Sun Tzu",
      "type": "PERSON",
      "relationships": [...]
    }
  ],
  "total_found": 5,
  "processing_time": 1.8
}
```

**Usage Example**:
```python
# Query for Sun Tzu entities
result = await mcp_client.call_tool(
    "query_knowledge_graph",
    {
        "query": "Sun Tzu",
        "query_type": "entities",
        "limit": 20
    }
)
```

### 5. Content Comparison Tools

#### `compare_content_sections`
**Description**: Compare multiple content sections for themes, entities, and patterns.

**Parameters**:
- `content_sections` (List[str]): List of content sections to compare
- `comparison_type` (str): Type of comparison ("themes", "entities", "sentiment", "comprehensive")
- `language` (str): Language code (default: "en")

**Returns**:
```json
{
  "success": true,
  "comparison_type": "themes",
  "sections_analyzed": 3,
  "individual_analyses": [...],
  "comparison_analysis": {
    "common_entities": [
      {"name": "Sun Tzu", "frequency": 3}
    ],
    "common_themes": ["Strategy", "Leadership"],
    "sentiment_variations": [...],
    "structural_patterns": [...]
  }
}
```

**Usage Example**:
```python
# Compare multiple chapters
result = await mcp_client.call_tool(
    "compare_content_sections",
    {
        "content_sections": [chapter_1, chapter_2, chapter_3],
        "comparison_type": "comprehensive"
    }
)
```

## Tool Selection Matrix

| User Request | Recommended MCP Tool | Parameters |
|--------------|---------------------|------------|
| "summarize chapter 1" | `summarize_text_content` | text=chapter_content, summary_type="comprehensive" |
| "analyze chapter 1" | `analyze_chapter_content` | chapter_text=content, analysis_type="comprehensive" |
| "extract entities from chapter" | `extract_entities` | text=chapter_content, language="en" |
| "compare chapters" | `compare_content_sections` | content_sections=[ch1, ch2, ch3] |
| "find related concepts" | `query_knowledge_graph` | query="concept", query_type="concepts" |
| "extract all chapters" | `extract_content_sections` | content=full_text, section_type="chapters" |

## Integration Patterns

### 1. Chapter Summarization Workflow

```python
async def summarize_chapter(chapter_content: str, chapter_title: str = ""):
    """Complete workflow for chapter summarization."""
    
    # Step 1: Generate comprehensive summary
    summary_result = await mcp_client.call_tool(
        "summarize_text_content",
        {
            "text": chapter_content,
            "summary_type": "comprehensive",
            "include_key_points": True,
            "include_entities": True
        }
    )
    
    # Step 2: Analyze chapter structure
    analysis_result = await mcp_client.call_tool(
        "analyze_chapter_content",
        {
            "chapter_text": chapter_content,
            "chapter_title": chapter_title,
            "analysis_type": "comprehensive"
        }
    )
    
    # Step 3: Extract entities for knowledge graph
    entities_result = await mcp_client.call_tool(
        "extract_entities",
        {
            "text": chapter_content,
            "language": "en"
        }
    )
    
    return {
        "summary": summary_result,
        "analysis": analysis_result,
        "entities": entities_result
    }
```

### 2. Multi-Chapter Analysis Workflow

```python
async def analyze_multiple_chapters(chapters: List[Dict]):
    """Analyze multiple chapters and compare them."""
    
    # Step 1: Extract and analyze each chapter
    chapter_analyses = []
    for chapter in chapters:
        analysis = await analyze_chapter_content(
            chapter["content"],
            chapter["title"]
        )
        chapter_analyses.append(analysis)
    
    # Step 2: Compare chapters
    comparison_result = await mcp_client.call_tool(
        "compare_content_sections",
        {
            "content_sections": [ch["content"] for ch in chapters],
            "comparison_type": "comprehensive"
        }
    )
    
    return {
        "individual_analyses": chapter_analyses,
        "comparison": comparison_result
    }
```

### 3. Knowledge Graph Integration Workflow

```python
async def enrich_knowledge_graph(content: str):
    """Enrich knowledge graph with content analysis."""
    
    # Step 1: Extract entities
    entities_result = await mcp_client.call_tool(
        "extract_entities",
        {
            "text": content,
            "language": "en"
        }
    )
    
    # Step 2: Query existing knowledge graph
    for entity in entities_result.get("entities", []):
        query_result = await mcp_client.call_tool(
            "query_knowledge_graph",
            {
                "query": entity["name"],
                "query_type": "entities"
            }
        )
        
        # Process relationships and add to knowledge graph
        if query_result.get("results"):
            # Add new relationships
            pass
    
    return entities_result
```

## Error Handling

All MCP tools follow a consistent error handling pattern:

```json
{
  "success": false,
  "error": "Error description",
  "processing_time": 0.0
}
```

Common error scenarios:
- **Invalid input**: Text too short, invalid language code
- **Processing failure**: Agent unavailable, model error
- **Timeout**: Processing takes too long
- **Resource limits**: Memory or CPU constraints

## Performance Considerations

### 1. Processing Times
- **Text summarization**: 2-5 seconds for 1000 words
- **Chapter analysis**: 3-8 seconds for comprehensive analysis
- **Entity extraction**: 1-3 seconds
- **Content comparison**: 5-15 seconds for multiple sections

### 2. Resource Usage
- **Memory**: 100-500MB per tool call
- **CPU**: Moderate usage during processing
- **Network**: Minimal (local processing)

### 3. Caching
- Results are cached for 1 hour by default
- Cache keys based on content hash and parameters
- Cache can be disabled for real-time processing

## Testing

### Running Tests
```bash
# Run the comprehensive test suite
python Test/test_mcp_content_analysis.py

# Run specific test
python -m pytest Test/test_mcp_content_analysis.py::test_text_summarization
```

### Test Coverage
- ✅ Text summarization functionality
- ✅ Chapter analysis capabilities
- ✅ Content extraction patterns
- ✅ Knowledge graph querying
- ✅ Content comparison logic
- ✅ MCP tool integration patterns

## Best Practices

### 1. Tool Selection
- Use `summarize_text_content` for general text summarization
- Use `analyze_chapter_content` for structured chapter analysis
- Use `extract_content_sections` for multi-chapter documents
- Use `query_knowledge_graph` for relationship discovery
- Use `compare_content_sections` for comparative analysis

### 2. Parameter Optimization
- Choose appropriate `summary_type` based on use case
- Set `include_analysis=True` for comprehensive results
- Use `limit` parameter to control result size
- Specify `language` for better accuracy

### 3. Error Handling
- Always check `success` field in responses
- Implement retry logic for transient failures
- Log errors for debugging
- Provide fallback behavior

### 4. Performance Optimization
- Cache results when appropriate
- Use batch processing for multiple requests
- Monitor processing times
- Implement timeout handling

## Future Enhancements

### Planned Features
1. **Advanced Summarization**: Multi-level summaries with different detail levels
2. **Semantic Analysis**: Deep semantic understanding of content
3. **Cross-Language Analysis**: Compare content across different languages
4. **Temporal Analysis**: Track changes and evolution in content
5. **Interactive Analysis**: Real-time analysis with user feedback

### Integration Opportunities
1. **Document Management**: Integration with document storage systems
2. **Collaborative Analysis**: Multi-user analysis workflows
3. **API Integration**: RESTful API endpoints for external access
4. **Visualization**: Interactive charts and graphs for analysis results
5. **Export Formats**: Support for various export formats (PDF, Excel, etc.)

## Conclusion

The new MCP content analysis tools provide a comprehensive framework for text analysis, summarization, and knowledge extraction. These tools follow the MCP-first architecture principles and integrate seamlessly with the existing agent swarm system.

For questions or support, please refer to the main project documentation or contact the development team.
