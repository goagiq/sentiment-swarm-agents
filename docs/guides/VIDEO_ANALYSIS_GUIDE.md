# Video Analysis Guide

## Overview

The Sentiment Analysis system now includes comprehensive video analysis capabilities that automatically detect and handle different video platforms and file types. The unified video analysis system provides a single interface for analyzing videos from YouTube, local files, and other platforms.

## Supported Video Types

### 1. YouTube Videos
- **URLs**: `https://www.youtube.com/watch?v=...`, `https://youtu.be/...`
- **Features**: Full audio/visual analysis, metadata extraction, sentiment analysis
- **Processing**: Automatic download, audio extraction, frame extraction

### 2. Local Video Files
- **Formats**: MP4, AVI, MOV, MKV, WebM, FLV, WMV
- **Features**: Video summarization, key scene detection, sentiment analysis
- **Processing**: Large file chunking, progressive analysis

### 3. Other Video Platforms
- **Platforms**: Vimeo, TikTok, Instagram, Facebook, Twitter, Twitch, Dailymotion, Bilibili, Rutube, OK.ru, VK
- **Features**: Webpage analysis, content extraction, sentiment analysis

## Usage

### Unified Video Analysis

The main entry point for all video analysis is the `unified_video_analysis` function:

```python
from src.agents.orchestrator_agent import unified_video_analysis

# Analyze any video type
result = await unified_video_analysis(video_input)
```

### Platform-Specific Analysis

For specific platform analysis, you can use dedicated functions:

```python
from src.agents.orchestrator_agent import (
    youtube_comprehensive_analysis,
    video_summarization_analysis
)

# YouTube-specific analysis
youtube_result = await youtube_comprehensive_analysis("https://youtube.com/...")

# Local video analysis
local_result = await video_summarization_analysis("path/to/video.mp4")
```

## API Endpoints

### MCP Tools

The system provides the following MCP tools for video analysis:

1. **`analyze_video_unified`**: Unified video analysis for all platforms
2. **`analyze_video_summarization`**: Local video summarization
3. **`analyze_youtube_comprehensive`**: YouTube-specific analysis

### FastAPI Endpoints

```python
# Unified video analysis
POST /analyze/video/unified
{
    "video_input": "https://youtube.com/watch?v=...",
    "language": "en"
}

# Video summarization
POST /analyze/video/summary
{
    "video_path": "data/video.mp4",
    "language": "en"
}
```

## Analysis Results

### YouTube Analysis Results

```json
{
    "status": "success",
    "content": [{
        "json": {
            "video_type": "youtube",
            "video_url": "https://youtube.com/watch?v=...",
            "sentiment": "positive",
            "confidence": 0.85,
            "method": "youtube_comprehensive_analyzer",
            "audio_sentiment": "positive",
            "visual_sentiment": "neutral",
            "video_metadata": {
                "title": "Video Title",
                "duration": "10:30",
                "views": 1000000
            },
            "processing_time": 45.2,
            "extracted_frames": 10,
            "audio_analysis": {...},
            "visual_analysis": {...}
        }
    }]
}
```

### Local Video Analysis Results

```json
{
    "status": "success",
    "content": [{
        "json": {
            "video_type": "local_video",
            "video_path": "data/video.mp4",
            "sentiment": "positive",
            "confidence": 0.8,
            "method": "video_summarization_agent",
            "summary": "Comprehensive video summary...",
            "key_scenes": ["Scene 1", "Scene 2", "Scene 3"],
            "key_moments": ["Moment 1", "Moment 2"],
            "topics": ["Topic 1", "Topic 2"],
            "executive_summary": "High-level summary for executives",
            "transcript": "Video transcript with timestamps",
            "visual_analysis": "Analysis of visual content",
            "timeline": "Timeline of scenes"
        }
    }]
}
```

## Platform Detection Logic

The system automatically detects video types using the following logic:

### YouTube Detection
- URLs containing: `youtube.com`, `youtu.be`, `youtube-nocookie.com`
- Mobile URLs: `m.youtube.com`, `www.youtube.com`

### Local Video Detection
- File extensions: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.wmv`
- File paths containing: `/` or `\`

### Other Platform Detection
- URLs containing: `vimeo.com`, `tiktok.com`, `instagram.com`, `facebook.com`, `twitter.com`, `twitch.tv`, `dailymotion.com`, `bilibili.com`, `rutube.ru`, `ok.ru`, `vk.com`
- Any URL starting with `http://` or `https://` not matching YouTube patterns

## Processing Features

### Large File Processing
- **Chunking**: Videos are split into 5-minute chunks for efficient processing
- **Progressive Analysis**: Each chunk is analyzed independently
- **Memory Management**: Temporary files are cleaned up automatically

### Audio Analysis
- **Extraction**: Audio is extracted from video files
- **Transcription**: Speech-to-text conversion
- **Sentiment Analysis**: Audio content sentiment detection
- **Quality Assessment**: Audio quality evaluation

### Visual Analysis
- **Frame Extraction**: Key frames are extracted for analysis
- **Scene Detection**: Important scenes are identified
- **Visual Sentiment**: Visual content sentiment analysis
- **Object Detection**: Objects and activities in video frames

### Metadata Analysis
- **Video Information**: Duration, resolution, format
- **Platform Data**: Views, likes, comments (for YouTube)
- **Content Analysis**: Topics, themes, key moments

## Configuration

### Video Processing Settings

```python
# In config/settings.py
VIDEO_CHUNK_DURATION = 300  # 5 minutes
MAX_VIDEO_DURATION = 7200   # 2 hours
VIDEO_CACHE_DIR = "./cache/video"
VIDEO_TEMP_DIR = "./temp/video"
```

### Supported Models

- **Vision Model**: `ollama:llava:latest` (default)
- **Audio Model**: `facebook/wav2vec2-base-960h`
- **Sentiment Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`

## Error Handling

The system includes comprehensive error handling:

- **Invalid URLs**: Returns error for unsupported platforms
- **File Not Found**: Handles missing local video files
- **Processing Errors**: Graceful degradation for analysis failures
- **Timeout Handling**: Long video processing timeout management

## Performance Considerations

### Processing Time
- **YouTube Videos**: 30-60 seconds for typical videos
- **Local Videos**: Depends on file size and duration
- **Large Files**: Progressive processing with progress updates

### Resource Usage
- **Memory**: Efficient chunking reduces memory requirements
- **CPU**: Multi-threaded processing for faster analysis
- **Storage**: Temporary files are automatically cleaned up

## Examples

### Complete Video Analysis Workflow

```python
import asyncio
from src.agents.orchestrator_agent import unified_video_analysis

async def analyze_video_workflow():
    # Analyze different video types
    videos = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "data/meeting_recording.mp4",
        "https://vimeo.com/123456789"
    ]
    
    for video in videos:
        print(f"Analyzing: {video}")
        result = await unified_video_analysis(video)
        
        if result["status"] == "success":
            content = result["content"][0]["json"]
            print(f"Video Type: {content['video_type']}")
            print(f"Sentiment: {content['sentiment']}")
            print(f"Confidence: {content['confidence']}")
        else:
            print(f"Analysis failed: {result}")

# Run the workflow
asyncio.run(analyze_video_workflow())
```

### Integration with Main Application

```python
# Using the main application
from main import UnifiedMCPServer

# Initialize the server
server = UnifiedMCPServer()

# Analyze video using MCP tool
result = await server.analyze_video_unified("https://youtube.com/watch?v=...")
```

## Troubleshooting

### Common Issues

1. **YouTube Analysis Fails**
   - Check internet connection
   - Verify YouTube URL format
   - Ensure yt-dlp is properly installed

2. **Local Video Processing Fails**
   - Verify file exists and is accessible
   - Check video format is supported
   - Ensure sufficient disk space

3. **Memory Issues**
   - Reduce chunk duration for large files
   - Increase system memory
   - Use SSD storage for better performance

### Debug Mode

Enable debug logging for detailed analysis:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- **Real-time Analysis**: Live video stream processing
- **Batch Processing**: Multiple video analysis
- **Custom Models**: User-defined analysis models
- **Advanced Features**: Object tracking, face recognition
- **Cloud Integration**: Cloud-based video processing
