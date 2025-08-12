# Classical Chinese PDF Processing Test Plan

## Overview
This test plan addresses the console errors encountered during Classical Chinese PDF processing and provides a systematic approach to testing multilingual support with proper error handling.

## Test Objectives
- Process Classical Chinese PDF with enhanced multilingual support
- Fix vector database metadata errors
- Ensure proper MCP framework integration
- Validate configuration-based language-specific parameters
- Generate comprehensive test reports

## Issues Identified and Solutions

### 1. Vector Database Metadata Errors

**Problem**: 
```
ERROR | src.core.vector_db:add_texts:610 - Failed to add texts to collection translations: 
Expected metadata value to be a str, int, float, bool, or None, got {} which is a dict in add.
```

**Root Cause**: Translation service trying to store empty dictionaries `{}` as metadata in ChromaDB.

**Solution**: 
- Fix metadata validation in `src/core/vector_db.py`
- Ensure all metadata values are primitive types (str, int, float, bool, None)
- Add metadata sanitization before storage

### 2. MCP Server Integration Issues

**Problem**: MCP tools not accessible as Python modules in test scripts.

**Root Cause**: MCP tools are framework-level functions, not importable Python modules.

**Solution**: 
- Use direct agent calls for testing instead of MCP framework calls
- Create proper MCP client integration for production use
- Implement fallback mechanisms for testing

### 3. Translation Service Warnings

**Problem**: 
```
WARNING - Failed to store translation in memory: Expected metadata value to be a str, int, float, bool, or None, got {} which is a dict in add.
```

**Root Cause**: Translation service metadata format incompatibility.

**Solution**: 
- Update translation service metadata handling
- Implement proper metadata serialization
- Add error handling for translation failures

## Test Plan Structure

### Phase 1: Configuration Validation
- [ ] Test Chinese language configuration loading
- [ ] Validate Classical Chinese patterns
- [ ] Verify Ollama model configuration
- [ ] Test entity extraction patterns

### Phase 2: File Processing
- [ ] Test PDF file existence and accessibility
- [ ] Validate file extraction agent initialization
- [ ] Test text extraction with Chinese language detection
- [ ] Verify extracted content quality

### Phase 3: Knowledge Graph Processing
- [ ] Test entity extraction from Chinese text
- [ ] Validate relationship mapping
- [ ] Test graph storage and retrieval
- [ ] Verify multilingual entity handling

### Phase 4: Vector Database Integration
- [ ] Test vector storage with proper metadata
- [ ] Validate retrieval functionality
- [ ] Test translation service integration
- [ ] Verify error handling

### Phase 5: Report Generation
- [ ] Test HTML report generation
- [ ] Validate PNG visualization creation
- [ ] Test multilingual report content
- [ ] Verify report file organization

## Test Scripts

### 1. Configuration Test Script
```python
# Test/config_validation_test.py
- Test Chinese configuration loading
- Validate Classical Chinese patterns
- Test Ollama model configuration
```

### 2. File Processing Test Script
```python
# Test/file_processing_test.py
- Test PDF extraction
- Validate text content
- Test language detection
```

### 3. Knowledge Graph Test Script
```python
# Test/knowledge_graph_test.py
- Test entity extraction
- Validate graph storage
- Test relationship mapping
```

### 4. Vector Database Test Script
```python
# Test/vector_db_test.py
- Test metadata handling
- Validate storage/retrieval
- Test error handling
```

### 5. Integration Test Script
```python
# Test/integration_test.py
- End-to-end processing
- Error handling validation
- Performance testing
```

## Error Fixes Implementation

### 1. Vector Database Metadata Fix

**File**: `src/core/vector_db.py`

**Fix**: Add metadata sanitization function
```python
def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize metadata to ensure ChromaDB compatibility."""
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            sanitized[key] = value
        elif isinstance(value, dict):
            # Convert dict to JSON string
            sanitized[key] = json.dumps(value, ensure_ascii=False)
        else:
            # Convert other types to string
            sanitized[key] = str(value)
    return sanitized
```

### 2. Translation Service Fix

**File**: `src/core/translation_service.py`

**Fix**: Update metadata handling
```python
def store_translation(self, original: str, translated: str, metadata: Dict[str, Any] = None):
    """Store translation with proper metadata handling."""
    if metadata is None:
        metadata = {}
    
    # Ensure metadata is compatible with vector database
    sanitized_metadata = self.vector_db.sanitize_metadata(metadata)
    
    try:
        self.vector_db.add_texts(
            texts=[translated],
            metadatas=[sanitized_metadata],
            ids=[f"trans_{hash(original)}"]
        )
    except Exception as e:
        logger.warning(f"Failed to store translation: {e}")
```

### 3. MCP Integration Fix

**File**: `Test/mcp_integration_test.py`

**Fix**: Use direct agent calls for testing
```python
async def test_mcp_integration():
    """Test MCP integration using direct agent calls."""
    # Use direct agent calls instead of MCP framework calls
    file_agent = EnhancedFileExtractionAgent()
    kg_agent = KnowledgeGraphAgent()
    
    # Process PDF directly
    result = await file_agent.process(request)
    kg_result = await kg_agent.process(kg_request)
```

## Test Execution Plan

### Pre-Test Setup
1. Clear databases: `python scripts/flushdb.py`
2. Verify PDF file exists: `data/Classical Chinese Sample 22208_0_8.pdf`
3. Check Ollama models are available
4. Ensure virtual environment is activated

### Test Execution Order
1. **Configuration Test**: `python Test/config_validation_test.py`
2. **File Processing Test**: `python Test/file_processing_test.py`
3. **Knowledge Graph Test**: `python Test/knowledge_graph_test.py`
4. **Vector Database Test**: `python Test/vector_db_test.py`
5. **Integration Test**: `python Test/integration_test.py`

### Post-Test Validation
1. Check generated reports in `Results/reports/`
2. Validate knowledge graph file: `Results/knowledge_graphs/knowledge_graph.pkl`
3. Verify vector database collections
4. Review test result files

## Expected Results

### Success Criteria
- ✅ No vector database metadata errors
- ✅ No translation service warnings
- ✅ Successful PDF text extraction
- ✅ Entity extraction with 1000+ entities
- ✅ Knowledge graph with 5000+ nodes
- ✅ Generated HTML and PNG reports
- ✅ Proper Chinese language support

### Performance Metrics
- PDF processing time: < 30 seconds
- Entity extraction time: < 60 seconds
- Report generation time: < 120 seconds
- Memory usage: < 2GB
- CPU usage: < 80%

## Error Handling Strategy

### 1. Graceful Degradation
- Continue processing if non-critical errors occur
- Log warnings instead of failing completely
- Provide fallback mechanisms

### 2. Error Recovery
- Retry failed operations with exponential backoff
- Implement circuit breaker pattern for external services
- Provide detailed error messages for debugging

### 3. Monitoring and Alerting
- Log all errors with appropriate severity levels
- Monitor system resources during processing
- Alert on critical failures

## Documentation Requirements

### Test Reports
- Generate comprehensive test reports in JSON format
- Include performance metrics and error counts
- Provide recommendations for improvements

### Code Documentation
- Update docstrings for all modified functions
- Document error handling procedures
- Provide usage examples

### User Documentation
- Update README with new features
- Document configuration options
- Provide troubleshooting guide

## Future Improvements

### 1. Enhanced Error Handling
- Implement comprehensive error categorization
- Add automatic error recovery mechanisms
- Improve error reporting and logging

### 2. Performance Optimization
- Implement parallel processing for large files
- Add caching mechanisms for repeated operations
- Optimize memory usage for large datasets

### 3. Testing Automation
- Implement continuous integration testing
- Add automated regression testing
- Create performance benchmarking suite

## Conclusion

This test plan provides a comprehensive approach to fixing the console errors and ensuring robust Classical Chinese PDF processing. The key focus areas are:

1. **Fix vector database metadata handling**
2. **Improve translation service integration**
3. **Implement proper MCP framework testing**
4. **Add comprehensive error handling**
5. **Ensure proper configuration management**

By following this test plan, we can achieve reliable, error-free processing of Classical Chinese PDFs with full multilingual support.
