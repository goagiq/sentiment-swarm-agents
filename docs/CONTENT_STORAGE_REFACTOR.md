# Content Storage Refactor: Full Content vs Summaries

## Overview

This document outlines the refactoring changes made to ensure that full transcriptions, translations, and content are stored in the vector database and knowledge graph instead of summaries.

## Problem Identified

### Original Issue
The system was inadvertently storing summaries instead of full content in both:
1. **Vector Database** (ChromaDB)
2. **Knowledge Graph** (Entity extraction)

### Root Cause Analysis
1. **Storage Priority**: Vector database prioritized `extracted_text` over `raw_content`
2. **Content Assignment**: Agents were setting:
   - `raw_content = str(request.content)` (original input/path)
   - `extracted_text = str(response)` (processed/summarized content)
3. **Audio/Video Processing**: Full transcriptions were not being properly stored in `extracted_text`

## Solution Implemented

### 1. Vector Database Storage Logic (`src/core/vector_db.py`)

**Before:**
```python
text_content = result.extracted_text or str(result.raw_content) or ""
```

**After:**
```python
text_content = self._get_full_content_for_storage(result)
```

**New Priority Order:**
1. `full_transcription` (from metadata)
2. `full_translation` (from metadata)
3. `extracted_text` (if confirmed to be full content)
4. `full_content` (from metadata)
5. `raw_content` (fallback)

**Content Detection:**
- Added `_is_full_content()` method to detect summaries vs full content
- Uses metadata flags and heuristics to determine content type
- Checks for summary indicators in text

### 2. Audio Agent Updates (`src/agents/unified_audio_agent.py`)

**Key Changes:**
- Store full transcriptions in `extracted_text`
- Store summaries separately in metadata
- Added metadata flags to track content type

**New Processing Flow:**
```python
# Get full transcription
full_transcription = await self._perform_enhanced_transcription(audio_path)

# Store full transcription in extracted_text
extracted_text=full_transcription,

# Store summary separately in metadata
metadata={
    "summary": summary_text,
    "content_type": "full_transcription",
    "is_full_content": True,
    "has_full_transcription": True
}
```

### 3. Vision Agent Updates (`src/agents/unified_vision_agent.py`)

**Key Changes:**
- Store full video transcriptions and visual analysis in `extracted_text`
- Store summaries separately in metadata
- Added methods to extract full video and image content

**New Processing Flow:**
```python
# Get full video content (transcription + visual analysis)
full_content = await self._get_full_video_content(content)

# Store full content in extracted_text
extracted_text=full_content,

# Store summary separately in metadata
metadata={
    "summary": summary_text,
    "content_type": "full_content",
    "is_full_content": True,
    "has_full_transcription": True
}
```

### 4. Knowledge Graph Agent Updates (`src/agents/knowledge_graph_agent.py`)

**Key Changes:**
- Use full content for entity extraction
- Added metadata flags to track content type
- Improved content extraction logic for different data types

**New Processing Flow:**
```python
# Extract full text content
text_content = await self._extract_text_content(request)

# Store full content in extracted_text
extracted_text=text_content,

# Add metadata flags
metadata={
    "content_type": "full_content",
    "is_full_content": True,
    "has_full_transcription": True
}
```

### 5. Configuration Options (`src/config/config.py`)

**New Configuration Settings:**
```python
# Content storage preferences
store_full_content: bool = True
store_summaries_separately: bool = True
content_storage_priority: str = "full_content"
min_content_length: int = 50
```

## Metadata Schema

### New Metadata Fields
```python
metadata = {
    # Content type tracking
    "content_type": "full_transcription" | "full_content" | "summary",
    "is_full_content": True | False,
    "has_full_transcription": True | False,
    "has_translation": True | False,
    
    # Content length tracking
    "content_length": int,
    "expected_min_length": int,
    "transcription_length": int,
    "summary_length": int,
    
    # Processing mode
    "processing_mode": "standard" | "with_summarization" | "knowledge_graph",
    
    # Summary storage (when applicable)
    "summary": str,
}
```

## Benefits

### 1. Better Search and Retrieval
- Full content enables more accurate semantic search
- Complete transcriptions provide better context
- Entity extraction from full content is more comprehensive

### 2. Improved Knowledge Graph
- Full content leads to better entity extraction
- More relationships can be discovered
- Better graph connectivity

### 3. Flexible Configuration
- Users can choose between full content and summaries
- Backward compatibility maintained
- Configurable storage preferences

### 4. Enhanced Metadata
- Clear tracking of content type
- Length information for quality assessment
- Processing mode indicators

## Migration Strategy

### For Existing Data
1. **Vector Database**: Existing data remains accessible
2. **New Processing**: All new content uses full content by default
3. **Re-processing**: Option to re-process existing content with full extraction

### Configuration Migration
```python
# Default behavior (full content)
config.agent.store_full_content = True

# Legacy behavior (summaries)
config.agent.store_full_content = False
config.agent.content_storage_priority = "summary"
```

## Testing

### Test Cases
1. **Audio Processing**: Verify full transcriptions are stored
2. **Video Processing**: Verify full transcriptions + visual analysis are stored
3. **Knowledge Graph**: Verify entity extraction uses full content
4. **Vector Search**: Verify search results include full content
5. **Configuration**: Verify different storage modes work correctly

### Validation
- Check metadata flags are set correctly
- Verify content length meets minimum requirements
- Confirm summaries are stored separately when requested
- Test backward compatibility

## Future Enhancements

### 1. Translation Support
- Add `has_translation` flag implementation
- Store translated content in metadata
- Support multiple language versions

### 2. Content Quality Metrics
- Add content quality scoring
- Track transcription confidence
- Monitor content completeness

### 3. Advanced Content Detection
- Improve summary detection algorithms
- Add content type classification
- Support custom content types

### 4. Performance Optimization
- Implement content caching
- Add lazy loading for large content
- Optimize storage for different content types

## Conclusion

This refactor ensures that the system stores full, comprehensive content instead of summaries, leading to:
- Better search capabilities
- More accurate knowledge graphs
- Improved entity extraction
- Enhanced user experience

The changes maintain backward compatibility while providing new configuration options for different use cases.
