# Unified Agents Guide

## Overview

The Sentiment Analysis system has been consolidated to use three unified agents that provide comprehensive functionality for all content types:

- **UnifiedTextAgent**: Handles all text processing with configurable modes
- **UnifiedAudioAgent**: Handles all audio processing with advanced features
- **UnifiedVisionAgent**: Handles all vision processing with comprehensive analysis

## UnifiedTextAgent

### Features
- **Simple Mode**: Direct text processing without frameworks
- **Strands Mode**: Framework-based processing with enhanced capabilities
- **Swarm Mode**: Coordinated analysis with multiple agents
- **Configurable Models**: Support for different Ollama models
- **Multi-language Support**: English and other languages

### Configuration Options
```python
agent = UnifiedTextAgent(
    use_strands=True,      # Enable Strands framework
    use_swarm=False,       # Enable swarm coordination
    agent_count=3,         # Number of swarm agents
    model_name="llama3.2"  # Custom model name
)
```

### Processing Modes

#### Simple Mode
```python
agent = UnifiedTextAgent(use_strands=False, use_swarm=False)
# Direct processing with basic sentiment analysis
```

#### Strands Mode
```python
agent = UnifiedTextAgent(use_strands=True, use_swarm=False)
# Framework-based processing with enhanced features
```

#### Swarm Mode
```python
agent = UnifiedTextAgent(use_strands=True, use_swarm=True, agent_count=3)
# Coordinated analysis with multiple specialized agents
```

## UnifiedAudioAgent

### Features
- **Audio Transcription**: High-quality speech-to-text conversion
- **Sentiment Analysis**: Audio content sentiment detection
- **Audio Summarization**: Key points and action items extraction
- **Large File Processing**: Chunked analysis for long audio files
- **Multiple Formats**: Support for mp3, wav, flac, m4a, ogg, aac, wma, opus
- **Quality Assessment**: Audio quality and emotion analysis

### Configuration Options
```python
agent = UnifiedAudioAgent(
    enable_summarization=True,           # Enable audio summarization
    enable_large_file_processing=True,   # Enable chunked processing
    model_name="llama3.2"               # Custom model name
)
```

### Capabilities
- **Basic Audio Processing**: Transcription and sentiment analysis
- **Advanced Summarization**: Key points, action items, topics
- **Large File Support**: Automatic chunking and processing
- **Quality Analysis**: Audio quality and emotion detection
- **Stream Processing**: Real-time audio stream analysis

## UnifiedVisionAgent

### Features
- **Image Analysis**: Comprehensive visual content analysis
- **Sentiment Detection**: Visual sentiment and emotion analysis
- **Object Recognition**: Detection and classification of objects
- **Text Extraction**: OCR capabilities for text in images
- **Scene Understanding**: Context and scene analysis
- **Multiple Formats**: Support for jpg, png, gif, bmp, tiff

### Configuration Options
```python
agent = UnifiedVisionAgent(
    enable_ocr=True,        # Enable text extraction
    enable_objects=True,    # Enable object recognition
    model_name="llama3.2"   # Custom model name
)
```

### Capabilities
- **Visual Sentiment**: Emotion and mood analysis from images
- **Object Detection**: Recognition and classification of objects
- **Text Extraction**: OCR for text in images
- **Scene Analysis**: Context and environment understanding
- **Quality Assessment**: Image quality and composition analysis

## Usage Examples

### Text Processing
```python
from agents.unified_text_agent import UnifiedTextAgent
from core.models import AnalysisRequest, DataType

# Initialize agent
agent = UnifiedTextAgent(use_strands=True)

# Create request
request = AnalysisRequest(
    content="I love this product! It's amazing.",
    data_type=DataType.TEXT,
    language="en"
)

# Process
result = await agent.process(request)
print(f"Sentiment: {result.sentiment.label}")
print(f"Confidence: {result.sentiment.confidence}")
```

### Audio Processing
```python
from agents.unified_audio_agent import UnifiedAudioAgent

# Initialize agent
agent = UnifiedAudioAgent(enable_summarization=True)

# Process audio file
request = AnalysisRequest(
    content="path/to/audio.mp3",
    data_type=DataType.AUDIO,
    language="en"
)

result = await agent.process(request)
print(f"Transcription: {result.extracted_text}")
print(f"Sentiment: {result.sentiment.label}")
```

### Vision Processing
```python
from agents.unified_vision_agent import UnifiedVisionAgent

# Initialize agent
agent = UnifiedVisionAgent(enable_ocr=True)

# Process image
request = AnalysisRequest(
    content="path/to/image.jpg",
    data_type=DataType.IMAGE,
    language="en"
)

result = await agent.process(request)
print(f"Visual sentiment: {result.sentiment.label}")
print(f"Objects detected: {result.metadata.get('objects', [])}")
```

## Agent Lifecycle

### Initialization
```python
agent = UnifiedTextAgent()
await agent.start()
```

### Processing
```python
result = await agent.process(request)
```

### Cleanup
```python
await agent.stop()
await agent.cleanup()
```

## Status and Monitoring

### Get Agent Status
```python
status = agent.get_status()
print(f"Agent ID: {status['agent_id']}")
print(f"Status: {status['status']}")
print(f"Capabilities: {status['capabilities']}")
```

### Check Capabilities
```python
capabilities = agent.metadata['capabilities']
if 'text' in capabilities:
    print("Text processing supported")
if 'sentiment_analysis' in capabilities:
    print("Sentiment analysis supported")
```

## Error Handling

All unified agents include comprehensive error handling:

```python
try:
    result = await agent.process(request)
except Exception as e:
    print(f"Processing failed: {e}")
    # Agent will return neutral sentiment on error
```

## Performance Optimization

### Text Agent
- Use simple mode for basic processing
- Use strands mode for enhanced features
- Use swarm mode for complex analysis

### Audio Agent
- Enable large file processing for long audio
- Use summarization for meeting recordings
- Enable quality assessment for production use

### Vision Agent
- Enable OCR for text-heavy images
- Use object detection for complex scenes
- Enable quality assessment for production use

## Migration from Old Agents

The unified agents replace the following old implementations:

### Text Agents
- `text_agent.py` → `UnifiedTextAgent` (simple mode)
- `text_agent_simple.py` → `UnifiedTextAgent` (simple mode)
- `text_agent_strands.py` → `UnifiedTextAgent` (strands mode)
- `text_agent_swarm.py` → `UnifiedTextAgent` (swarm mode)

### Audio Agents
- `audio_agent_enhanced.py` → `UnifiedAudioAgent`
- `audio_summarization_agent.py` → `UnifiedAudioAgent` (with summarization)

### Vision Agents
- `vision_agent_enhanced.py` → `UnifiedVisionAgent`

## Best Practices

1. **Choose the Right Mode**: Use simple mode for basic needs, strands for enhanced features
2. **Configure Appropriately**: Enable only the features you need
3. **Handle Errors**: Always include error handling in your code
4. **Monitor Performance**: Use status methods to monitor agent health
5. **Clean Up Resources**: Always call cleanup methods when done

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're importing from the correct unified agent
2. **Model Not Found**: Check that the specified Ollama model is available
3. **Processing Failures**: Check file paths and formats
4. **Memory Issues**: Use large file processing for big files

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

agent = UnifiedTextAgent()
# Debug information will be logged
```
