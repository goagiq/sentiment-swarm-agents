# Summarization Agents Implementation

## Overview

This document describes the implementation of two new specialized summarization agents for the Sentiment Analysis system:

1. **AudioSummarizationAgent** - For comprehensive audio content summarization
2. **VideoSummarizationAgent** - For comprehensive video content summarization

## Audio Summarization Agent

### Features

The `AudioSummarizationAgent` provides comprehensive audio content analysis and summarization capabilities:

#### Core Capabilities
- **Audio Transcription**: Convert audio to text with high accuracy
- **Key Points Extraction**: Identify and extract main discussion points
- **Action Items Identification**: Detect tasks, assignments, and follow-ups
- **Topic Analysis**: Categorize and analyze discussion topics
- **Sentiment Analysis**: Analyze emotional tone and sentiment
- **Speaker Emotion Analysis**: Track emotional changes throughout the audio
- **Meeting Minutes Generation**: Create structured meeting documentation
- **Executive Summary**: Generate high-level summaries for leadership
- **Timeline Summary**: Create chronological summaries of events
- **Quote Extraction**: Identify and extract important quotes
- **Bullet Point Summary**: Generate concise bullet-point summaries

#### Supported Audio Formats
- MP3, WAV, FLAC, M4A, OGG, AAC, WMA, OPUS

#### Tools Available
1. `generate_audio_summary` - Comprehensive audio summary
2. `extract_key_points` - Extract key discussion points
3. `identify_action_items` - Identify tasks and action items
4. `analyze_topics` - Analyze and categorize topics
5. `create_executive_summary` - Executive-level summary
6. `generate_timeline_summary` - Timeline-based summary
7. `extract_quotes` - Extract important quotes
8. `analyze_speaker_emotions` - Analyze speaker emotions
9. `create_meeting_minutes` - Structured meeting minutes
10. `generate_bullet_points` - Bullet point summary

### Usage Example

```python
from src.agents.audio_summarization_agent import AudioSummarizationAgent
from src.core.models import AnalysisRequest, DataType

# Initialize agent
agent = AudioSummarizationAgent()

# Create analysis request
request = AnalysisRequest(
    data_type=DataType.AUDIO,
    content="path/to/audio.mp3",
    language="en"
)

# Process audio
result = await agent.process(request)

# Access results
print(f"Sentiment: {result.sentiment.label}")
print(f"Key Points: {result.metadata.get('key_points', [])}")
print(f"Action Items: {result.metadata.get('action_items', [])}")
```

## Video Summarization Agent

### Features

The `VideoSummarizationAgent` provides comprehensive video content analysis and summarization capabilities:

#### Core Capabilities
- **Visual Content Analysis**: Analyze visual elements, people, objects, and settings
- **Key Scene Extraction**: Identify and extract important scenes
- **Key Moments Identification**: Detect critical moments and decision points
- **Scene Timeline Creation**: Create chronological scene breakdowns
- **Video Metadata Extraction**: Extract technical video information
- **Video Sentiment Analysis**: Analyze emotional tone throughout the video
- **Executive Summary**: Generate high-level summaries for leadership
- **Video Transcript Creation**: Create timestamped transcripts
- **Topic Analysis**: Categorize and analyze video topics
- **YouTube Integration**: Specialized handling for YouTube videos

#### Supported Video Formats
- MP4, AVI, MOV, MKV, WEBM, FLV, WMV
- YouTube URLs (with yt-dlp integration)

#### Tools Available
1. `generate_video_summary` - Comprehensive video summary
2. `extract_key_scenes` - Extract key scenes from video
3. `identify_key_moments` - Identify important moments
4. `analyze_visual_content` - Analyze visual content and elements
5. `create_scene_timeline` - Create timeline of scenes
6. `extract_video_metadata` - Extract video metadata
7. `analyze_video_sentiment` - Analyze video sentiment
8. `generate_executive_summary` - Executive-level summary
9. `create_video_transcript` - Create video transcript
10. `analyze_video_topics` - Analyze video topics

### Usage Example

```python
from src.agents.video_summarization_agent import VideoSummarizationAgent
from src.core.models import AnalysisRequest, DataType

# Initialize agent
agent = VideoSummarizationAgent()

# Create analysis request for video file
request = AnalysisRequest(
    data_type=DataType.VIDEO,
    content="path/to/video.mp4",
    language="en"
)

# Process video
result = await agent.process(request)

# Access results
print(f"Sentiment: {result.sentiment.label}")
print(f"Key Scenes: {result.metadata.get('key_scenes', [])}")
print(f"Key Moments: {result.metadata.get('key_moments', [])}")

# For YouTube videos
youtube_request = AnalysisRequest(
    data_type=DataType.WEBPAGE,
    content="https://www.youtube.com/watch?v=example",
    language="en"
)
```

## Orchestrator Integration

Both summarization agents are integrated into the `OrchestratorAgent` with dedicated tools:

### New Orchestrator Tools

1. **`audio_summarization_analysis`**
   - Handles comprehensive audio summarization
   - Returns key points, action items, topics, and sentiment analysis

2. **`video_summarization_analysis`**
   - Handles comprehensive video summarization
   - Returns key scenes, moments, topics, and sentiment analysis
   - Supports both video files and YouTube URLs

### Usage with Orchestrator

```python
from src.agents.orchestrator_agent import OrchestratorAgent

# Initialize orchestrator
orchestrator = OrchestratorAgent()

# Process audio summarization
audio_result = await orchestrator.audio_summarization_analysis("audio.mp3")

# Process video summarization
video_result = await orchestrator.video_summarization_analysis("video.mp4")

# Process YouTube summarization
youtube_result = await orchestrator.video_summarization_analysis(
    "https://www.youtube.com/watch?v=example"
)
```

## MCP Server Integration

Both agents are available as tools in the unified MCP server:

### New MCP Tools

1. **`analyze_audio_summarization`**
   - Parameters: `audio_path`, `language` (optional)
   - Returns comprehensive audio analysis with summaries

2. **`analyze_video_summarization`**
   - Parameters: `video_path`, `language` (optional)
   - Returns comprehensive video analysis with summaries
   - Supports YouTube URLs automatically

### MCP Usage

```python
# Audio summarization
result = await mcp_client.call_tool(
    "analyze_audio_summarization",
    {"audio_path": "meeting_recording.mp3"}
)

# Video summarization
result = await mcp_client.call_tool(
    "analyze_video_summarization",
    {"video_path": "presentation.mp4"}
)

# YouTube summarization
result = await mcp_client.call_tool(
    "analyze_video_summarization",
    {"video_path": "https://www.youtube.com/watch?v=example"}
)
```

## Configuration

Both agents use the configuration system for model selection and settings:

### Audio Agent Configuration
- **Default Model**: `config.model.default_audio_model`
- **Max Duration**: `config.agent.max_audio_duration`
- **Supported Formats**: MP3, WAV, FLAC, M4A, OGG, AAC, WMA, OPUS

### Video Agent Configuration
- **Default Model**: `config.model.default_vision_model`
- **Max Duration**: 30 seconds (configurable)
- **Supported Formats**: MP4, AVI, MOV, MKV, WEBM, FLV, WMV
- **YouTube Integration**: Uses `config.youtube_dl.download_path`

## Testing

A comprehensive test suite is available in `Test/test_summarization_agents.py`:

```bash
cd Test
python test_summarization_agents.py
```

The test suite covers:
- Audio summarization agent functionality
- Video summarization agent functionality
- YouTube video summarization
- Individual tool testing
- Error handling and edge cases

## Output Format

Both agents return structured `AnalysisResult` objects with:

### Common Fields
- `sentiment`: SentimentResult with label and confidence
- `processing_time`: Processing duration in seconds
- `metadata`: Rich metadata including:
  - `summary_type`: Type of summary generated
  - `key_points_count`: Number of key points extracted
  - `action_items_count`: Number of action items identified
  - `topics_identified`: Number of topics analyzed
  - `summary_length`: Length of generated summary

### Audio-Specific Metadata
- `summary`: Comprehensive audio summary
- `key_points`: List of key discussion points
- `action_items`: List of identified tasks
- `topics`: List of analyzed topics
- `executive_summary`: High-level summary
- `meeting_minutes`: Structured meeting documentation
- `quotes`: Important quotes extracted
- `emotions`: Speaker emotion analysis
- `timeline`: Timeline summary

### Video-Specific Metadata
- `summary`: Comprehensive video summary
- `key_scenes`: List of key scenes
- `key_moments`: List of important moments
- `topics`: List of analyzed topics
- `executive_summary`: High-level summary
- `transcript`: Video transcript with timestamps
- `visual_analysis`: Visual content analysis
- `timeline`: Scene timeline

## Error Handling

Both agents implement comprehensive error handling:

- **File Not Found**: Graceful handling of missing files
- **Unsupported Formats**: Clear error messages for unsupported formats
- **Processing Errors**: Detailed error information in metadata
- **Model Initialization**: Fallback handling for model loading issues
- **Network Errors**: Robust handling of URL-based content

## Performance Considerations

- **Async Processing**: All operations are asynchronous for better performance
- **Resource Management**: Proper cleanup of temporary files and models
- **Memory Efficiency**: Streaming processing for large files
- **Caching**: Model caching to avoid repeated initialization
- **Timeout Handling**: Configurable timeouts for long-running operations

## Future Enhancements

Potential improvements for future versions:

1. **Real-time Processing**: Support for streaming audio/video analysis
2. **Multi-language Support**: Enhanced language detection and processing
3. **Custom Models**: Support for custom fine-tuned models
4. **Batch Processing**: Efficient batch processing of multiple files
5. **Advanced Analytics**: More sophisticated analytics and insights
6. **Integration APIs**: REST APIs for external system integration
7. **Web Interface**: Web-based interface for easy usage
8. **Export Formats**: Support for various export formats (PDF, Word, etc.)

## Conclusion

The new summarization agents provide powerful capabilities for analyzing and summarizing audio and video content. They integrate seamlessly with the existing agent ecosystem and provide comprehensive tools for content analysis, making them valuable additions to the sentiment analysis system.
