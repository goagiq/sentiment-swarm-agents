# File Extraction Agent Guide

## Overview

The File Extraction Agent is a comprehensive PDF processing solution that provides intelligent text extraction from PDF documents. It uses a conditional strategy to determine the best extraction method based on the PDF content type.

## Features

- **Conditional Extraction Strategy**: Automatically chooses between PyPDF2 (for text-based PDFs) and Vision OCR (for image-based PDFs)
- **Parallel Processing**: Multi-threaded page processing for improved performance
- **Real-time Progress Tracking**: Detailed progress updates with ETA calculations
- **ChromaDB Integration**: Automatic storage of extracted text in vector database
- **Memory Management**: Aggressive cleanup and optimization for large PDFs
- **Retry Logic**: Page-level retries with graceful degradation
- **Vision OCR**: Advanced OCR using vision models for scanned documents
- **Error Handling**: Comprehensive error reporting and recovery mechanisms

## Architecture

### Processing Pipeline

```
PDF Input → Content Analysis → Conditional Strategy Selection → Parallel Processing → Text Output → ChromaDB Storage
```

### Key Components

1. **Content Analyzer**: Determines if PDF contains embedded text or is image-based
2. **Conditional Router**: Routes to appropriate extraction method
3. **PyPDF2 Extractor**: Fast text extraction for text-based PDFs
4. **Vision OCR Processor**: Advanced OCR for image-based PDFs
5. **Parallel Processor**: Multi-threaded page processing
6. **ChromaDB Manager**: Vector database storage integration

### Conditional Strategy

The agent implements a sophisticated conditional strategy:

#### Content Analysis
- Analyzes PDF structure and content
- Determines if PDF contains embedded text or is image-based
- Uses multiple heuristics for accurate classification

#### PyPDF2 Strategy (Text-based PDFs)
- Direct text extraction using PyPDF2
- Fast processing (~10-100ms per page)
- High accuracy for well-formatted text
- Minimal memory usage

#### Vision OCR Strategy (Image-based PDFs)
- Converts PDF pages to images using PyMuPDF
- Processes images with vision models (llava:latest)
- Advanced OCR capabilities for scanned documents
- Handles complex layouts and handwritten text

#### Error Handling Strategy
- Retry logic for failed page processing
- Graceful degradation when methods fail
- Comprehensive error reporting
- Fallback to alternative extraction methods

## Configuration

### Agent Parameters

```python
FileExtractionAgent(
    agent_id="file_extraction_agent",
    max_capacity=5,
    model_name="llava:latest",
    max_workers=4,
    chunk_size=1,
    retry_attempts=1
)
```

### Environment Variables

```bash
OLLAMA_HOST=localhost:11434
DEFAULT_VISION_MODEL=llava:latest
MAX_IMAGE_SIZE=2048
VISION_TEMPERATURE=0.1
```

## Usage Examples

### Basic Usage

```python
from src.agents.file_extraction_agent import FileExtractionAgent
from src.core.models import AnalysisRequest, DataType

# Create agent
agent = FileExtractionAgent()

# Create request
request = AnalysisRequest(
    data_type=DataType.PDF,
    content="/path/to/document.pdf"
)

# Process PDF
result = await agent.process(request)

# Access results
print(f"Extracted text: {result.extracted_text}")
print(f"Method used: {result.metadata['method']}")
print(f"Pages processed: {result.metadata['pages_processed']}")
```

### Advanced Configuration

```python
# Custom configuration
agent = FileExtractionAgent(
    max_workers=8,  # More parallel workers
    chunk_size=2,   # Process 2 pages at a time
    retry_attempts=3  # More retry attempts
)

# Process with custom parameters
result = await agent.process(request)
```

### Standalone Utility

For simple PDF text extraction without the full agent framework, you can use the standalone utility:

```python
from src.agents.extract_pdf_text import extract_pdf_text, extract_pdf_text_to_file

# Extract text
text = extract_pdf_text("/path/to/document.pdf")

# Extract and save to file
success = extract_pdf_text_to_file("/path/to/document.pdf", "output.txt")
```

## Performance Optimization

### Memory Management

- **Aggressive Cleanup**: Immediate deletion of processed data
- **Image Optimization**: Resize and compress images before OCR
- **Garbage Collection**: Forced cleanup after processing
- **Streaming**: Process large files without loading entire content

### Parallel Processing

- **Configurable Workers**: Adjust `max_workers` based on system capabilities
- **Page-level Parallelism**: Process multiple pages simultaneously
- **Load Balancing**: Distribute work across available workers
- **Progress Tracking**: Real-time updates with ETA calculations

### Caching

- **ChromaDB Storage**: Automatic vector database integration
- **Result Caching**: Avoid reprocessing identical files
- **Metadata Storage**: Store processing statistics and metadata

## Error Handling

### Retry Logic

- **Page-level Retries**: Retry failed pages with exponential backoff
- **Method Fallback**: Try alternative extraction methods
- **Graceful Degradation**: Continue processing even if some pages fail
- **Error Reporting**: Detailed error messages and statistics

### Recovery Mechanisms

- **Memory Recovery**: Automatic cleanup after failures
- **File Cleanup**: Remove temporary files on errors
- **State Recovery**: Maintain processing state across retries
- **Logging**: Comprehensive error logging for debugging

## Monitoring and Statistics

### Performance Metrics

```python
stats = agent.get_stats()
print(f"Total files processed: {stats['total_files']}")
print(f"Successful extractions: {stats['successful_extractions']}")
print(f"PyPDF2 success rate: {stats['pypdf2_success']}")
print(f"Vision OCR success rate: {stats['vision_ocr_success']}")
```

### Progress Tracking

- **Real-time Updates**: Progress callbacks with current/total pages
- **ETA Calculation**: Estimated time to completion
- **Status Reporting**: Current processing status and method
- **Performance Monitoring**: Processing time and memory usage

## Integration

### Orchestrator Integration

The agent is automatically registered with the orchestrator:

```python
from src.core.orchestrator import SentimentOrchestrator

orchestrator = SentimentOrchestrator()
result = await orchestrator.analyze_pdf("/path/to/document.pdf")
```

### API Integration

Access via REST API:

```bash
curl -X POST "http://localhost:8001/analyze/pdf" \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "/path/to/document.pdf",
    "model_preference": "llava:latest"
  }'
```

### ChromaDB Integration

Automatic storage in vector database:

```python
# Results are automatically stored in ChromaDB
# Access stored documents
from src.core.vector_db import vector_db

documents = vector_db.search("your search query")
```

## Best Practices

### File Preparation

- **File Validation**: Ensure PDF files are not corrupted
- **Size Optimization**: Consider file size for memory usage
- **Format Consistency**: Use standard PDF formats for best results

### Performance Tuning

- **Worker Configuration**: Adjust `max_workers` based on CPU cores
- **Memory Monitoring**: Monitor memory usage for large files
- **Batch Processing**: Process multiple files in batches

### Error Prevention

- **Input Validation**: Validate file paths and formats
- **Resource Management**: Monitor system resources
- **Backup Strategies**: Implement fallback extraction methods

## Troubleshooting

### Common Issues

1. **PyPDF2 Import Error**: Install with `pip install PyPDF2`
2. **PyMuPDF Import Error**: Install with `pip install PyMuPDF`
3. **Ollama Connection Error**: Ensure Ollama server is running
4. **Memory Issues**: Reduce `max_workers` or file size

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

agent = FileExtractionAgent()
```

### Performance Issues

- **Slow Processing**: Increase `max_workers`
- **High Memory Usage**: Reduce `max_workers` or file size
- **OCR Failures**: Check Ollama server and vision model availability

## Future Enhancements

### Planned Features

1. **Batch Processing**: Process multiple PDFs simultaneously
2. **Advanced OCR**: Enhanced table and form recognition
3. **Multi-language Support**: Better OCR for non-English documents
4. **Cloud Integration**: Support for cloud storage providers
5. **Caching**: Intelligent result caching for repeated requests

### Performance Optimizations

1. **GPU Acceleration**: CUDA support for vision models
2. **Streaming**: Real-time processing for large files
3. **Compression**: Advanced image compression techniques
4. **Load Balancing**: Intelligent agent selection
