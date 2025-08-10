# AudioAgent MCP Server

The AudioAgent MCP Server provides audio sentiment analysis tools through the Model Context Protocol (MCP). It wraps the existing AudioAgent and exposes its capabilities as MCP tools for external clients to use.

## Architecture

```
MCP Client → AudioAgent MCP Server → AudioAgent → Ollama/Strands
                ↓
            MCP Tools (FastMCP)
```

## Features

- **Audio Transcription**: Convert audio to text for analysis
- **Sentiment Analysis**: Analyze audio sentiment using Ollama
- **Feature Extraction**: Extract audio features and metadata
- **Fallback Analysis**: Rule-based analysis when primary methods fail
- **Batch Processing**: Process multiple audio files efficiently
- **Multi-format Support**: MP3, WAV, FLAC, M4A, OGG formats
- **Async Processing**: Full async/await support for better performance

## Installation

### Prerequisites

1. Install FastMCP for full functionality:
```bash
pip install fastmcp
```

2. The server will automatically fall back to a mock server if FastMCP is not available.

## Quick Start

### Basic Usage

```python
from mcp.audio_agent_server import create_audio_agent_mcp_server

# Create server with default model
server = create_audio_agent_mcp_server()

# Create server with specific model
server = create_audio_agent_mcp_server(model_name="llava:latest")

# Run the server
server.run(host="0.0.0.0", port=8007, debug=True)
```

### Direct Execution

```bash
cd src/mcp
python audio_agent_server.py
```

## MCP Tools

The AudioAgent MCP Server exposes the following tools:

### 1. `transcribe_audio`

Transcribe audio content to text using AudioAgent.

**Parameters:**
- `audio_path` (str): Path or URL to audio file
- `language` (str, optional): Language code (default: "en")

**Returns:**
```json
{
  "audio_path": "path/to/audio.mp3",
  "transcription": "Transcribed text content",
  "analysis_time": 1.23,
  "status": "success",
  "metadata": {
    "method": "audio_agent_transcription",
    "language": "en",
    "agent_id": "AudioAgent_abc123"
  }
}
```

### 2. `analyze_audio_sentiment`

Analyze the sentiment of audio content using AudioAgent.

**Parameters:**
- `audio_path` (str): Path or URL to audio file
- `language` (str, optional): Language code (default: "en")
- `confidence_threshold` (float, optional): Minimum confidence threshold (default: 0.8)

**Returns:**
```json
{
  "audio_path": "path/to/audio.mp3",
  "sentiment": "positive",
  "confidence": 0.85,
  "transcription": "Transcribed text content",
  "analysis_time": 2.45,
  "status": "success",
  "metadata": {
    "method": "audio_agent_sentiment_analysis",
    "language": "en",
    "agent_id": "AudioAgent_abc123",
    "raw_response": "POSITIVE"
  }
}
```

### 3. `extract_audio_features`

Extract audio features for analysis using AudioAgent.

**Parameters:**
- `audio_path` (str): Path or URL to audio file
- `language` (str, optional): Language code (default: "en")

**Returns:**
```json
{
  "audio_path": "path/to/audio.mp3",
  "features": {
    "file_path": "path/to/audio.mp3",
    "file_type": "mp3",
    "has_transcription": true,
    "analysis_method": "ollama_audio"
  },
  "analysis_time": 0.67,
  "status": "success",
  "metadata": {
    "method": "audio_agent_feature_extraction",
    "language": "en"
  }
}
```

### 4. `comprehensive_audio_analysis`

Perform comprehensive audio analysis including transcription, sentiment, and features.

**Parameters:**
- `audio_path` (str): Path or URL to audio file
- `language` (str, optional): Language code (default: "en")
- `confidence_threshold` (float, optional): Minimum confidence threshold (default: 0.8)

**Returns:**
```json
{
  "audio_path": "path/to/audio.mp3",
  "sentiment": "positive",
  "confidence": 0.85,
  "scores": {
    "positive": 0.85,
    "negative": 0.10,
    "neutral": 0.05
  },
  "transcription": "Transcribed text content",
  "analysis_time": 3.12,
  "status": "success",
  "metadata": {
    "agent_id": "AudioAgent_abc123",
    "processing_time": 2.98,
    "status": "completed",
    "method": "audio_agent_comprehensive_analysis",
    "language": "en",
    "tools_used": ["transcribe_audio", "analyze_audio_sentiment", "extract_audio_features"]
  }
}
```

### 5. `fallback_audio_analysis`

Use fallback audio analysis when primary method fails.

**Parameters:**
- `audio_path` (str): Path or URL to audio file
- `language` (str, optional): Language code (default: "en")

**Returns:**
```json
{
  "audio_path": "path/to/audio.mp3",
  "sentiment": "neutral",
  "confidence": 0.5,
  "analysis_time": 0.34,
  "method": "fallback",
  "status": "success",
  "metadata": {
    "method": "audio_agent_fallback_analysis",
    "language": "en",
    "analysis": "Audio analysis of path/to/audio.mp3 - sentiment unclear"
  }
}
```

### 6. `batch_analyze_audio`

Analyze sentiment for multiple audio files in batch using AudioAgent.

**Parameters:**
- `audio_paths` (List[str]): List of audio file paths to analyze
- `language` (str, optional): Language code (default: "en")
- `confidence_threshold` (float, optional): Minimum confidence threshold (default: 0.8)

**Returns:**
```json
[
  {
    "audio_path": "path/to/audio1.mp3",
    "sentiment": "positive",
    "confidence": 0.85,
    "transcription": "First audio transcription",
    "analysis_time": 2.1,
    "status": "success"
  },
  {
    "audio_path": "path/to/audio2.mp3",
    "sentiment": "negative",
    "confidence": 0.78,
    "transcription": "Second audio transcription",
    "analysis_time": 1.9,
    "status": "success"
  }
]
```

### 7. `get_audio_agent_capabilities`

Get information about AudioAgent capabilities and configuration.

**Parameters:** None

**Returns:**
```json
{
  "agent_id": "AudioAgent_abc123",
  "model": "granite3.2-vision",
  "supported_formats": ["mp3", "wav", "flac", "m4a", "ogg"],
  "max_audio_duration": 300,
  "capabilities": ["audio", "transcription", "sentiment_analysis"],
  "available_tools": [
    "transcribe_audio",
    "analyze_audio_sentiment",
    "extract_audio_features",
    "comprehensive_audio_analysis",
    "fallback_audio_analysis",
    "batch_analyze_audio"
  ],
  "features": [
    "audio transcription",
    "sentiment classification",
    "confidence scoring",
    "audio feature extraction",
    "fallback analysis",
    "batch processing",
    "multi-format support"
  ]
}
```

## Integration Examples

### With Strands Agents

```python
from mcp.audio_agent_server import create_audio_agent_mcp_server

# Create MCP server
audio_server = create_audio_agent_mcp_server()

# Start the server
audio_server.run(host="localhost", port=8007)
```

### With FastMCP Clients

```python
from fastmcp import FastMCPClient

# Create MCP client
client = FastMCPClient("http://localhost:8007")

# Use client to call MCP tools
# (Tool calls would be made through the FastMCP client interface)
```

### With MCP Streamable HTTP Client

```python
from mcp.client.streamable_http import streamablehttp_client

def create_mcp_client():
    """Create MCP client for audio analysis."""
    return streamablehttp_client("http://localhost:8007/mcp/")

# Create MCP client
client = create_mcp_client()

# Create Strands agent with MCP tools
# (Integration would depend on specific MCP client implementation)
```

## Configuration

### Model Configuration

The AudioAgent MCP Server inherits model configuration from the AudioAgent:

```python
# Default audio model from config
default_model = config.model.default_audio_model

# Create server with specific model
server = create_audio_agent_mcp_server(model_name="custom-model:latest")
```

### Port Configuration

The server runs on port 8007 by default:

```python
# Custom port
server.run(host="0.0.0.0", port=9000, debug=True)

# Default port (8007)
server.run(host="localhost", debug=True)
```

## Error Handling

The server provides comprehensive error handling:

- **Tool-level errors**: Each tool catches exceptions and returns error information
- **Fallback mechanisms**: Automatic fallback to rule-based analysis
- **Graceful degradation**: Server continues operating even if individual tools fail
- **Detailed error reporting**: Error messages include context and metadata

### Error Response Format

```json
{
  "error": "Error description",
  "audio_path": "path/to/audio.mp3",
  "sentiment": "neutral",
  "confidence": 0.0,
  "transcription": "",
  "analysis_time": 0.0,
  "status": "failed"
}
```

## Testing

### Run Tests

```bash
cd Test
python test_audio_agent_mcp_server.py
```

### Run Demo

```bash
cd examples
python audio_agent_mcp_demo.py
```

## Troubleshooting

### Common Issues

1. **FastMCP Not Available**: The server will automatically use a mock server
2. **Import Errors**: Ensure the src directory is in your Python path
3. **Port Conflicts**: Change the port if 8007 is already in use
4. **Audio Format Issues**: Check that your audio files are in supported formats

### Mock Server Mode

When FastMCP is not available, the server runs in mock mode:

- Tools are registered but not accessible through standard MCP protocols
- Server provides development and testing capabilities
- Install FastMCP for full MCP protocol support

## Performance Considerations

- **Async Processing**: All tools use async/await for better performance
- **Batch Processing**: Use `batch_analyze_audio` for multiple files
- **Caching**: Consider implementing caching for repeated audio analysis
- **Resource Management**: Audio processing can be resource-intensive

## Security Considerations

- **File Path Validation**: Validate audio file paths to prevent path traversal
- **URL Validation**: Validate audio URLs before processing
- **Resource Limits**: Implement limits on audio file size and duration
- **Access Control**: Consider implementing authentication for production use

## Future Enhancements

- **Real-time Audio Streaming**: Support for live audio analysis
- **Advanced Audio Features**: Spectral analysis, tempo detection
- **Multi-language Support**: Enhanced language detection and processing
- **Cloud Integration**: Support for cloud-based audio storage and processing

## Support

For issues and questions:

1. Check the test scripts for usage examples
2. Review the demo script for integration patterns
3. Check FastMCP documentation for MCP protocol details
4. Review AudioAgent source code for underlying functionality

## License

This MCP server is part of the Sentiment Analysis Swarm project and follows the same licensing terms.
