# TextAgent MCP Server

The TextAgent MCP Server provides a Model Context Protocol (MCP) interface for the TextAgent, enabling external applications to access text sentiment analysis capabilities through standardized MCP tools.

## Overview

The TextAgent MCP Server wraps the existing TextAgent functionality and exposes it as MCP tools that can be consumed by MCP clients. This allows for:

- Standardized tool interfaces
- Easy integration with MCP-compatible applications
- Consistent error handling and response formats
- Scalable text analysis services

## Features

### Core Capabilities

- **Text Sentiment Analysis**: Analyze sentiment using Ollama models
- **Feature Extraction**: Extract text features for analysis
- **Comprehensive Analysis**: Combined sentiment and feature analysis
- **Fallback Analysis**: Rule-based sentiment analysis when primary methods fail
- **Batch Processing**: Process multiple texts efficiently

### MCP Tools

1. **`analyze_text_sentiment`**: Primary sentiment analysis tool
2. **`extract_text_features`**: Text feature extraction tool
3. **`comprehensive_text_analysis`**: Full analysis including sentiment and features
4. **`fallback_sentiment_analysis`**: Rule-based fallback analysis
5. **`batch_analyze_texts`**: Batch processing for multiple texts
6. **`get_text_agent_capabilities`**: Get server capabilities and configuration

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Client   │───▶│ TextAgent MCP    │───▶│   TextAgent     │
│                 │    │ Server           │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   MCP Tools      │
                       │   (FastMCP)      │
                       └──────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- Ollama running locally (for sentiment analysis)
- FastMCP (optional, falls back to mock server)

### Dependencies

```bash
pip install fastmcp loguru pydantic
```

## Usage

### Starting the Server

```python
from mcp.text_agent_server import create_text_agent_mcp_server

# Create server instance
server = create_text_agent_mcp_server()

# Run server
server.run(host="0.0.0.0", port=8002, debug=True)
```

### Using MCP Tools

```python
# Example: Analyze text sentiment
result = await analyze_text_sentiment(
    text="I love this product!",
    language="en",
    confidence_threshold=0.8
)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}")
```

### Tool Parameters

#### `analyze_text_sentiment`

- **`text`** (required): Text content to analyze
- **`language`** (optional): Language code (default: "en")
- **`confidence_threshold`** (optional): Minimum confidence (default: 0.8)

#### `extract_text_features`

- **`text`** (required): Text content to analyze
- **`language`** (optional): Language code (default: "en")

#### `comprehensive_text_analysis`

- **`text`** (required): Text content to analyze
- **`language`** (optional): Language code (default: "en")
- **`confidence_threshold`** (optional): Minimum confidence (default: 0.8)

## Response Format

All tools return consistent response formats:

```json
{
    "text": "analyzed text",
    "sentiment": "positive|negative|neutral",
    "confidence": 0.85,
    "scores": {
        "positive": 0.8,
        "negative": 0.1,
        "neutral": 0.1
    },
    "analysis_time": 0.15,
    "metadata": {
        "agent_id": "text-agent-123",
        "method": "text_agent_sentiment_analysis",
        "status": "completed"
    }
}
```

## Error Handling

The server provides comprehensive error handling:

- **Tool-level errors**: Caught and returned with error details
- **Fallback mechanisms**: Automatic fallback to rule-based analysis
- **Graceful degradation**: Returns neutral sentiment on failures
- **Detailed logging**: Comprehensive error logging for debugging

## Configuration

### Server Settings

- **Host**: Server host (default: "localhost")
- **Port**: Server port (default: 8002)
- **Debug**: Enable debug mode (default: False)
- **Transport**: MCP transport protocol (default: "streamable-http")

### Model Configuration

The server inherits model configuration from the TextAgent:

- **Default Model**: Configurable via TextAgent initialization
- **Model Metadata**: Accessible via `get_text_agent_capabilities()`
- **Language Support**: Configurable language capabilities

## Testing

### Running Tests

```bash
cd Test
python test_text_agent_mcp_server.py
```

### Demo Script

```bash
cd examples
python text_agent_mcp_demo.py
```

## Integration Examples

### With Strands Agents

```python
from strands import Agent
from mcp.client.streamable_http import streamablehttp_client

# Create MCP client
def create_transport():
    return streamablehttp_client("http://localhost:8002/mcp/")

# Create Strands agent with MCP tools
agent = Agent(tools=create_transport())
```

### With FastMCP Client

```python
from fastmcp import FastMCPClient

# Connect to server
client = FastMCPClient("http://localhost:8002")

# Use tools
result = await client.analyze_text_sentiment(
    text="Sample text",
    language="en"
)
```

## Performance Considerations

- **Async Processing**: All tools are async for better performance
- **Connection Pooling**: Efficient HTTP connection management
- **Caching**: Response caching for repeated requests
- **Batch Processing**: Optimized for multiple text analysis

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure src directory is in Python path
2. **Connection Failures**: Check if Ollama is running on localhost:11434
3. **Tool Registration**: Verify MCP server initialization
4. **Model Loading**: Check TextAgent model configuration

### Debug Mode

Enable debug mode for detailed logging:

```python
server.run(host="0.0.0.0", port=8002, debug=True)
```

## Future Enhancements

- **Model Switching**: Dynamic model selection
- **Advanced Features**: Named entity recognition, topic modeling
- **Performance Optimization**: Response caching, connection pooling
- **Security**: Authentication, rate limiting
- **Monitoring**: Metrics collection, health checks

## Contributing

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure backward compatibility

## License

This project follows the same license as the parent Sentiment analysis system.
