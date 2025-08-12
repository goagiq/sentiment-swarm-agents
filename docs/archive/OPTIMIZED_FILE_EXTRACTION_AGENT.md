# Optimized File Extraction Agent

## Overview

The optimized `FileExtractionAgent` provides high-performance PDF text extraction with intelligent processing strategies, memory management, and comprehensive error handling. This agent has been significantly enhanced based on real-world usage patterns and performance analysis.

## Key Optimizations

### 1. Performance Improvements

**Memory Management:**
- Chunked processing to prevent memory overflow
- Automatic memory cleanup with configurable thresholds
- Optimized image processing for OCR
- Efficient data structures using dataclasses

**Parallel Processing:**
- Improved parallel page processing with `as_completed`
- Configurable worker pools and chunk sizes
- Better timeout handling per page
- Progress tracking with ETA calculations

**Image Optimization:**
- Configurable image size limits
- Quality-optimized JPEG compression
- Automatic RGB conversion
- Resize optimization for large images

### 2. Enhanced Data Structures

**ExtractionConfig:**
```python
@dataclass
class ExtractionConfig:
    max_workers: int = 4
    chunk_size: int = 1
    retry_attempts: int = 1
    timeout_per_page: int = 60
    max_image_size: int = 1024
    image_quality: int = 85
    memory_cleanup_threshold: int = 10
```

**PageResult:**
```python
@dataclass
class PageResult:
    page_number: int
    content: str
    content_length: int
    extraction_method: str
    confidence: float
    processing_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
```

### 3. Intelligent Processing Strategies

**PDF Type Detection:**
- Automatic detection of text-based vs image-based PDFs
- Smart fallback strategies
- Content analysis for optimal method selection

**Error Handling:**
- Page-level error tracking
- Retry mechanisms with exponential backoff
- Graceful degradation for failed pages
- Comprehensive error reporting

### 4. Enhanced Metadata and Monitoring

**Performance Metrics:**
- Memory cleanup tracking
- Average page processing time
- Success/failure rates per method
- Detailed extraction statistics

**Quality Assessment:**
- Confidence scoring per page
- Content length validation
- Extraction method tracking
- Error categorization

## Usage Examples

### Basic Usage
```python
from src.agents.file_extraction_agent import FileExtractionAgent
from src.core.models import AnalysisRequest, DataType

# Initialize optimized agent
agent = FileExtractionAgent(
    agent_id="my_agent",
    max_workers=4,
    enable_chroma_storage=False
)

# Process PDF
request = AnalysisRequest(
    data_type=DataType.PDF,
    content="path/to/document.pdf"
)

result = await agent.process(request)
```

### Advanced Configuration
```python
# Custom configuration
agent = FileExtractionAgent(
    agent_id="custom_agent",
    max_workers=8,
    chunk_size=5,
    retry_attempts=3,
    enable_chroma_storage=True
)

# Access configuration
config = agent.config
print(f"Max workers: {config.max_workers}")
print(f"Chunk size: {config.chunk_size}")
print(f"Timeout per page: {config.timeout_per_page}")
```

### Page-by-Page Analysis
```python
# Access structured page data
for page in result.pages:
    if not page.error_message:
        print(f"Page {page.page_number}: {page.content_length} chars")
        print(f"Confidence: {page.confidence}")
        print(f"Processing time: {page.processing_time:.2f}s")
    else:
        print(f"Page {page.page_number} failed: {page.error_message}")
```

### Performance Monitoring
```python
# Get detailed statistics
stats = agent.get_stats()
print(f"Total files: {stats['total_files']}")
print(f"Successful extractions: {stats['successful_extractions']}")
print(f"Memory cleanups: {stats['memory_cleanups']}")

# Access performance metrics
perf_metrics = result.metadata.get("performance_metrics", {})
print(f"Average page time: {perf_metrics.get('average_page_time', 0.0):.2f}s")
```

## Configuration Options

### Agent Configuration
- `max_workers`: Number of parallel workers (default: 4)
- `chunk_size`: Pages per processing chunk (default: 1)
- `retry_attempts`: Retry attempts per page (default: 1)
- `enable_chroma_storage`: Enable ChromaDB storage (default: True)

### Extraction Configuration
- `timeout_per_page`: Timeout per page in seconds (default: 60)
- `max_image_size`: Maximum image size for OCR (default: 1024)
- `image_quality`: JPEG quality for optimization (default: 85)
- `memory_cleanup_threshold`: Pages before memory cleanup (default: 10)

## Performance Characteristics

### Text-Based PDFs (PyPDF2)
- **Speed**: Very fast (typically < 1 second per page)
- **Memory**: Low memory usage
- **Accuracy**: High accuracy for text content
- **Limitations**: Cannot extract from image-based content

### Image-Based PDFs (Vision OCR)
- **Speed**: Moderate (typically 5-15 seconds per page)
- **Memory**: Higher memory usage with optimization
- **Accuracy**: Good accuracy with optimized prompts
- **Scalability**: Parallel processing with configurable workers

### Memory Management
- **Automatic cleanup**: After processing chunks
- **Configurable thresholds**: Based on page count
- **Optimized image handling**: Resize and compress large images
- **Efficient data structures**: Using dataclasses for better performance

## Error Handling

### Page-Level Errors
- Individual page failures don't stop entire extraction
- Detailed error messages for each failed page
- Retry mechanisms for transient failures
- Graceful degradation with partial results

### Recovery Strategies
- Automatic fallback between extraction methods
- Configurable retry attempts
- Timeout handling for long-running operations
- Memory cleanup on failures

### Error Reporting
- Comprehensive error categorization
- Page-level error tracking
- Performance impact analysis
- Detailed logging for debugging

## Integration Features

### ChromaDB Integration
- Optional storage of extraction results
- Configurable metadata storage
- Efficient indexing of extracted content
- Searchable document storage

### Structured Output
- Consistent page data structure
- Backward compatibility with existing code
- Flexible metadata access
- Standardized error reporting

### Monitoring and Analytics
- Real-time progress tracking
- Performance metrics collection
- Success rate monitoring
- Resource usage tracking

## Best Practices

### Performance Optimization
1. **Adjust worker count** based on system resources
2. **Use appropriate chunk sizes** for memory management
3. **Configure timeouts** based on document complexity
4. **Monitor memory usage** and adjust cleanup thresholds

### Error Handling
1. **Always check page error messages** for failed pages
2. **Use confidence scores** to filter low-quality extractions
3. **Implement retry logic** for transient failures
4. **Monitor performance metrics** for optimization

### Memory Management
1. **Process large documents in chunks**
2. **Enable memory cleanup** for long-running operations
3. **Monitor cleanup frequency** and adjust thresholds
4. **Use appropriate image quality settings**

## Migration from Previous Version

### Backward Compatibility
- All existing code continues to work
- Same API interface maintained
- Enhanced features are optional
- Gradual migration supported

### New Features
- Structured page data access
- Enhanced metadata and statistics
- Improved error handling
- Performance monitoring

### Recommended Updates
1. **Use structured page data** instead of text parsing
2. **Access enhanced metadata** for better insights
3. **Implement performance monitoring** for optimization
4. **Configure memory management** for large documents

## Testing

### Test Script
A comprehensive test script is provided: `Test/test_optimized_file_extraction.py`

**Test Features:**
- Performance benchmarking
- Memory usage monitoring
- Error handling validation
- Structured data verification

### Performance Testing
```python
# Run performance test
python Test/test_optimized_file_extraction.py

# Monitor output for:
# - Processing time per page
# - Memory cleanup frequency
# - Success rates
# - Error patterns
```

## Future Enhancements

### Planned Improvements
1. **Advanced OCR confidence scoring**
2. **Content type detection per page**
3. **Language identification**
4. **Layout analysis and structure detection**
5. **Compression optimization for large files**

### Integration Opportunities
1. **Direct text agent integration**
2. **Translation pipeline support**
3. **Content summarization**
4. **Quality assessment automation**

## Conclusion

The optimized `FileExtractionAgent` provides significant performance improvements while maintaining backward compatibility. Key benefits include:

- **Better Performance**: Optimized memory management and parallel processing
- **Enhanced Reliability**: Comprehensive error handling and recovery
- **Improved Monitoring**: Detailed metrics and progress tracking
- **Flexible Configuration**: Configurable parameters for different use cases
- **Structured Output**: Consistent and accessible page data

This agent is designed for production use with large-scale PDF processing requirements while providing the flexibility needed for various use cases.
