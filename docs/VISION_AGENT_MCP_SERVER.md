# VisionAgent MCP Server

## Overview

The VisionAgent MCP Server provides vision sentiment analysis tools through the Model Context Protocol (MCP). It wraps the existing VisionAgent and exposes its capabilities as MCP tools for external clients to use.

## Architecture

```
MCP Client → VisionAgent MCP Server → VisionAgent → Ollama/Strands
                ↓
            MCP Tools (FastMCP)
```

## Features

- **Image Sentiment Analysis**: Analyze sentiment of image content
- **Video Frame Processing**: Process video frames for analysis
- **Vision Feature Extraction**: Extract visual features from images
- **Comprehensive Analysis**: Combined sentiment and feature analysis
- **Fallback Analysis**: Rule-based analysis when primary methods fail
- **Video Sentiment Analysis**: Analyze sentiment of video content
- **Batch Processing**: Process multiple images in batch
- **Capabilities Discovery**: Get server capabilities and configuration

## Installation

### Prerequisites

- Python 3.8+
- FastMCP (optional, falls back to mock server)
- VisionAgent dependencies

### Setup

1. Install FastMCP for full functionality:
```bash
pip install fastmcp
```

2. The server will automatically fall back to a mock server if FastMCP is not available.

## Usage

### Basic Server Creation

```python
from mcp.vision_agent_server import create_vision_agent_mcp_server

# Create server with default model
server = create_vision_agent_mcp_server()

# Create server with specific model
server = create_vision_agent_mcp_server(model_name="llava:latest")
```

### Running the Server

```python
# Run on default port 8003
server.run(host="0.0.0.0", port=8003, debug=True)

# Or run directly
python src/mcp/vision_agent_mcp_server.py
```

### Command Line

```bash
cd src/mcp
python vision_agent_server.py
```

## MCP Tools

### 1. analyze_image_sentiment

Analyze the sentiment of image content using VisionAgent.

**Parameters:**
- `image_path` (str): Path or URL to image file
- `language` (str, optional): Language code (default: "en")
- `confidence_threshold` (float, optional): Minimum confidence threshold (default: 0.8)

**Returns:**
```json
{
    "image_path": "path/to/image.jpg",
    "content_type": "image",
    "sentiment": "positive",
    "confidence": 0.85,
    "scores": {"positive": 0.85, "negative": 0.10, "neutral": 0.05},
    "description": "A happy family photo...",
    "analysis_time": 2.34,
    "metadata": {...}
}
```

### 2. process_video_frame

Process video frame for sentiment analysis using VisionAgent.

**Parameters:**
- `video_path` (str): Path or URL to video file
- `language` (str, optional): Language code (default: "en")

**Returns:**
```json
{
    "video_path": "path/to/video.mp4",
    "content_type": "video",
    "sentiment": "neutral",
    "confidence": 0.0,
    "scores": {"neutral": 1.0},
    "description": "Video frame analysis result...",
    "analysis_time": 1.23,
    "status": "success",
    "metadata": {...}
}
```

### 3. extract_vision_features

Extract vision features for analysis using VisionAgent.

**Parameters:**
- `image_path` (str): Path or URL to image file
- `language` (str, optional): Language code (default: "en")

**Returns:**
```json
{
    "image_path": "path/to/image.jpg",
    "content_type": "image",
    "features": {
        "dimensions": "1920x1080",
        "mode": "RGB",
        "format": "JPEG",
        "size_bytes": 245760,
        "type": "color_image"
    },
    "analysis_time": 0.45,
    "status": "success",
    "metadata": {...}
}
```

### 4. comprehensive_vision_analysis

Perform comprehensive vision analysis including sentiment and features.

**Parameters:**
- `image_path` (str): Path or URL to image file
- `language` (str, optional): Language code (default: "en")
- `confidence_threshold` (float, optional): Minimum confidence threshold (default: 0.8)

**Returns:**
```json
{
    "image_path": "path/to/image.jpg",
    "content_type": "image",
    "sentiment": "positive",
    "confidence": 0.85,
    "scores": {"positive": 0.85, "negative": 0.10, "neutral": 0.05},
    "features": {...},
    "description": "A happy family photo...",
    "analysis_time": 3.12,
    "metadata": {...}
}
```

### 5. fallback_vision_analysis

Use fallback vision analysis when primary method fails.

**Parameters:**
- `image_path` (str): Path or URL to image file
- `language` (str, optional): Language code (default: "en")

**Returns:**
```json
{
    "image_path": "path/to/image.jpg",
    "content_type": "image",
    "sentiment": "neutral",
    "confidence": 0.5,
    "scores": {"neutral": 1.0},
    "description": "Image with dimensions 1920x1080 pixels, color image",
    "analysis_time": 0.12,
    "method": "fallback_vision_analysis",
    "metadata": {...}
}
```

### 6. analyze_video_sentiment

Analyze the sentiment of video content using VisionAgent.

**Parameters:**
- `video_path` (str): Path or URL to video file
- `language` (str, optional): Language code (default: "en")
- `confidence_threshold` (float, optional): Minimum confidence threshold (default: 0.8)

**Returns:**
```json
{
    "video_path": "path/to/video.mp4",
    "content_type": "video",
    "sentiment": "positive",
    "confidence": 0.78,
    "scores": {"positive": 0.78, "negative": 0.15, "neutral": 0.07},
    "description": "Video content description...",
    "analysis_time": 4.56,
    "metadata": {...}
}
```

### 7. batch_analyze_images

Analyze sentiment for multiple images in batch using VisionAgent.

**Parameters:**
- `image_paths` (List[str]): List of image paths to analyze
- `language` (str, optional): Language code (default: "en")
- `confidence_threshold` (float, optional): Minimum confidence threshold (default: 0.8)

**Returns:**
```json
[
    {
        "image_path": "path/to/image1.jpg",
        "content_type": "image",
        "sentiment": "positive",
        "confidence": 0.85,
        "scores": {...},
        "features": {...},
        "description": "...",
        "analysis_time": 2.34,
        "metadata": {...}
    },
    {
        "image_path": "path/to/image2.jpg",
        "content_type": "image",
        "sentiment": "negative",
        "confidence": 0.72,
        "scores": {...},
        "features": {...},
        "description": "...",
        "analysis_time": 1.89,
        "metadata": {...}
    }
]
```

### 8. get_vision_agent_capabilities

Get information about VisionAgent capabilities and configuration.

**Parameters:** None

**Returns:**
```json
{
    "agent_id": "vision_agent_abc123",
    "model": "llava:latest",
    "supported_formats": ["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov"],
    "max_image_size": 1024,
    "max_video_duration": 30,
    "capabilities": ["vision", "tool_calling"],
    "available_tools": [
        "analyze_image_sentiment",
        "process_video_frame",
        "extract_vision_features",
        "comprehensive_vision_analysis",
        "fallback_vision_analysis",
        "analyze_video_sentiment",
        "batch_analyze_images"
    ],
    "features": [
        "image sentiment analysis",
        "video frame processing",
        "vision feature extraction",
        "comprehensive vision analysis",
        "fallback analysis",
        "batch processing",
        "multi-format support"
    ]
}
```

## Configuration

### Model Configuration

The server inherits model configuration from the VisionAgent:

```python
# Default model from config
server = create_vision_agent_mcp_server()

# Custom model
server = create_vision_agent_mcp_server(model_name="llava:13b")
```

### Port Configuration

The server runs on port 8003 by default:

```python
server.run(host="0.0.0.0", port=8003, debug=True)
```

## Integration Examples

### With Strands Agents

```python
from strands import Agent
from mcp.vision_agent_server import create_vision_agent_mcp_server

# Create MCP server
vision_server = create_vision_agent_mcp_server()

# Create Strands agent with vision tools
vision_agent = Agent(
    name="vision_analyzer",
    system_prompt="You are a vision analysis expert. Use the available tools to analyze images and videos.",
    tools=[
        vision_server.analyze_image_sentiment,
        vision_server.extract_vision_features
    ]
)
```

### With FastMCP Clients

```python
from fastmcp import FastMCPClient
from mcp.vision_agent_server import create_vision_agent_mcp_server

# Start server
server = create_vision_agent_mcp_server()
server.run(host="localhost", port=8003)

# Connect client
client = FastMCPClient("http://localhost:8003")

# Use tools
result = await client.analyze_image_sentiment(
    image_path="path/to/image.jpg"
)
```

## Error Handling

The server provides comprehensive error handling:

- **Tool-level errors**: Each tool catches exceptions and returns error information
- **Fallback mechanisms**: Automatic fallback to simpler analysis methods
- **Consistent error format**: All errors follow the same response structure
- **Logging**: Comprehensive logging for debugging

### Error Response Format

```json
{
    "error": "Error description",
    "image_path": "path/to/image.jpg",
    "content_type": "image",
    "sentiment": "neutral",
    "confidence": 0.0,
    "scores": {"neutral": 1.0},
    "description": "",
    "analysis_time": 0.0,
    "status": "failed"
}
```

## Testing

### Run Tests

```bash
cd Test
python test_vision_agent_mcp_server.py
```

### Run Demo

```bash
cd examples
python vision_agent_mcp_demo.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the `src` directory is in your Python path
2. **FastMCP Not Available**: The server will automatically use a mock server
3. **Port Already in Use**: Change the port number in the server configuration
4. **VisionAgent Errors**: Check that the underlying VisionAgent is properly configured

### Debug Mode

Enable debug mode for detailed logging:

```python
server.run(host="0.0.0.0", port=8003, debug=True)
```

## Performance

- **Async Processing**: All tools are async for better performance
- **Batch Processing**: Support for processing multiple images efficiently
- **Caching**: VisionAgent may cache model results for better performance
- **Resource Management**: Proper cleanup of temporary files and resources

## Security Considerations

- **File Path Validation**: Validate image/video file paths
- **Size Limits**: Respect maximum file size and duration limits
- **Input Sanitization**: Sanitize all input parameters
- **Access Control**: Consider implementing authentication for production use

## Future Enhancements

- **Real-time Video Analysis**: Support for streaming video analysis
- **Advanced Feature Extraction**: More sophisticated visual feature analysis
- **Multi-modal Analysis**: Combine vision with other data types
- **Performance Optimization**: GPU acceleration and model optimization
- **API Rate Limiting**: Implement rate limiting for production use

## Support

For issues and questions:

1. Check the test scripts for usage examples
2. Review the VisionAgent implementation
3. Check FastMCP documentation for MCP protocol details
4. Review error logs for debugging information
