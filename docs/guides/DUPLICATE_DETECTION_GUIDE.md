# Duplicate Detection System Guide

## Overview

The duplicate detection system prevents redundant processing of the same files or content, improving efficiency and reducing storage bloat. It handles file-based, content-based, and similarity-based duplicate detection.

## How It Works

### 1. **File Path Duplicate Detection**
- Tracks which files have been processed by their file path
- Checks file modification time to detect if a file has been updated
- Prevents re-processing of unchanged files

### 2. **Content Hash Duplicate Detection**
- Computes SHA-256 hash of file content or text
- Detects exact content matches regardless of file path
- Handles large files by using path-based hashing for files > 100MB

### 3. **Similarity-Based Duplicate Detection**
- Uses vector database embeddings to find similar content
- Configurable similarity threshold (default: 95%)
- Detects near-duplicates that aren't exact matches

### 4. **Knowledge Graph Deduplication**
- Deduplicates entities and relationships within each analysis
- Updates existing nodes with higher confidence scores
- Prevents duplicate entities in the knowledge graph

## Components

### DuplicateDetectionService
The core service that manages duplicate detection:

```python
from src.core.duplicate_detection_service import DuplicateDetectionService

# Initialize service
service = DuplicateDetectionService()

# Check for duplicates
result = await service.detect_duplicates(
    file_path="path/to/file.pdf",
    content="text content",
    data_type="pdf",
    agent_id="agent_123"
)
```

### Vector Database Integration
Enhanced vector database with similarity search:

```python
from src.core.vector_db import VectorDBManager

# Find similar content
similar_results = await vector_db.find_similar_content(
    content="text to compare",
    threshold=0.95
)

# Check for exact duplicates
duplicate = await vector_db.check_content_duplicate(
    content="text content",
    threshold=0.98
)
```

### Orchestrator Integration
Automatic duplicate detection in the analysis pipeline:

```python
from src.core.orchestrator import SentimentOrchestrator

# Duplicate detection is automatically applied
result = await orchestrator.analyze(request)

# Check if duplicate was detected
if result.metadata.get("duplicate_detected"):
    print(f"Duplicate type: {result.metadata.get('duplicate_type')}")
    print(f"Recommendation: {result.metadata.get('recommendation')}")
```

## Configuration

### DuplicateDetectionConfig
Configure duplicate detection behavior:

```python
from src.config.duplicate_detection_config import DuplicateDetectionConfig

# Default configuration
config = DuplicateDetectionConfig()

# Custom configuration
config = DuplicateDetectionConfig(
    enabled=True,
    check_file_path=True,
    check_content_hash=True,
    similarity_threshold=0.95,
    default_action="skip"
)
```

### Configuration Presets

```python
from src.config.duplicate_detection_config import presets

# Strict mode - high threshold, always skip
strict_config = presets["strict"]

# Lenient mode - lower threshold, allow updates
lenient_config = presets["lenient"]

# Disabled mode - no duplicate detection
disabled_config = presets["disabled"]
```

### Data Type Specific Settings

```python
# Configure settings for specific data types
config.data_type_settings = {
    "pdf": {
        "check_file_path": True,
        "check_content_hash": True,
        "similarity_threshold": 0.98,
        "default_action": "skip"
    },
    "video": {
        "check_file_path": True,
        "check_content_hash": False,  # Videos are large
        "similarity_threshold": 0.85,
        "default_action": "skip"
    }
}
```

## Usage Examples

### Basic Usage

```python
import asyncio
from src.core.orchestrator import SentimentOrchestrator
from src.core.models import AnalysisRequest, DataType

async def analyze_file():
    orchestrator = SentimentOrchestrator()
    
    # First processing
    request1 = AnalysisRequest(
        data_type=DataType.PDF,
        content="data/Classical Chinese Sample 22208_0_8.pdf",
        language="zh"
    )
    result1 = await orchestrator.analyze(request1)
    print(f"First result: {result1.sentiment.label}")
    
    # Second processing (will be detected as duplicate)
    request2 = AnalysisRequest(
        data_type=DataType.PDF,
        content="data/Classical Chinese Sample 22208_0_8.pdf",
        language="zh"
    )
    result2 = await orchestrator.analyze(request2)
    
    if result2.metadata.get("duplicate_detected"):
        print("Duplicate detected and skipped!")
    else:
        print("Processing continued...")

asyncio.run(analyze_file())
```

### Force Reprocessing

```python
# Force reprocessing even if duplicate detected
request = AnalysisRequest(
    data_type=DataType.PDF,
    content="data/Classical Chinese Sample 22208_0_8.pdf",
    language="zh"
)
request.force_reprocess = True  # Bypass duplicate detection

result = await orchestrator.analyze(request)
```

### Get Statistics

```python
# Get duplicate detection statistics
stats = await orchestrator.get_duplicate_stats()

print(f"Total processed files: {stats['total_files']}")
print(f"Files by type: {stats['files_by_type']}")
print(f"Recent activity: {stats['recent_activity_24h']}")

# Most processed files
for file_info in stats['most_processed']:
    print(f"{file_info['file_path']}: {file_info['processing_count']} times")
```

## Duplicate Types

### 1. **file_path**
- Same file path processed before
- File hasn't been modified since last processing
- **Action**: Skip processing

### 2. **file_path_modified**
- Same file path processed before
- File has been modified since last processing
- **Action**: Update/reprocess

### 3. **content_hash**
- Same content hash found (exact content match)
- **Action**: Skip processing

### 4. **similar**
- Similar content found using vector similarity
- **Action**: Skip or update based on threshold

## Recommendations

### 1. **skip**
- Duplicate detected, skip processing
- Return cached result if available
- Fastest option

### 2. **update**
- Duplicate detected but content may have changed
- Continue with processing to update result
- Good for files that may be modified

### 3. **reprocess**
- Force reprocessing regardless of duplicates
- Useful for testing or when you want fresh analysis

## Database Schema

The duplicate detection service uses SQLite to track processed files:

```sql
CREATE TABLE processed_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    content_hash TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    modification_time REAL NOT NULL,
    first_processed TEXT NOT NULL,
    last_processed TEXT NOT NULL,
    processing_count INTEGER DEFAULT 1,
    data_type TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    result_id TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

## Performance Considerations

### 1. **File Size Limits**
- Files > 100MB use path-based hashing instead of content hashing
- Configurable via `max_file_size_for_hash` setting

### 2. **Database Indexes**
- Indexes on file_path, content_hash, data_type, and agent_id
- Efficient querying for duplicate detection

### 3. **Cache Integration**
- Duplicate detection results are cached
- Configurable TTL (default: 24 hours)

### 4. **Cleanup**
- Old records are automatically cleaned up
- Configurable retention period (default: 30 days)

## Testing

Run the duplicate detection test:

```bash
python Test/test_duplicate_detection.py
```

This test demonstrates:
- File path duplicate detection
- Content hash duplicate detection
- Force reprocessing
- Statistics collection
- Different content handling

## Troubleshooting

### Common Issues

1. **Duplicate not detected**
   - Check if duplicate detection is enabled
   - Verify file path is correct
   - Check database connection

2. **False positives**
   - Adjust similarity threshold
   - Review data type specific settings
   - Check content hash computation

3. **Performance issues**
   - Reduce similarity threshold
   - Disable similarity checking for large files
   - Increase cleanup frequency

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger("src.core.duplicate_detection_service").setLevel(logging.DEBUG)
```

## Best Practices

1. **Configure appropriately for your use case**
   - Use strict mode for production
   - Use lenient mode for development
   - Disable for testing

2. **Monitor statistics**
   - Track duplicate detection rates
   - Monitor storage usage
   - Review most processed files

3. **Regular maintenance**
   - Clean up old records
   - Monitor database size
   - Update configuration as needed

4. **Handle edge cases**
   - Large files
   - Network files
   - Temporary files
   - File permission issues

## Integration with Existing System

The duplicate detection system integrates seamlessly with:

- **Vector Database**: Enhanced with similarity search
- **Knowledge Graph**: Deduplicates entities and relationships
- **Orchestrator**: Automatic duplicate detection in analysis pipeline
- **Cache System**: Caches duplicate detection results
- **Configuration System**: Configurable behavior per data type

This ensures efficient processing while maintaining data integrity and preventing storage bloat from duplicate content.
