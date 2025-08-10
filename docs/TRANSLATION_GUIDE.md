# Translation Agent Integration Guide

## Overview

The Translation Agent provides comprehensive foreign language translation capabilities integrated into the existing sentiment analysis system. It supports automatic language detection, high-quality translation using Ollama models, and translation memory using Chroma vector DB.

## Features

### 🎯 Core Capabilities
- **Automatic Language Detection**: Detects source language using pattern matching and AI
- **High-Quality Translation**: Uses mistral-small3.1:latest for optimal translation quality
- **Translation Memory**: Stores translations in Chroma vector DB for consistency
- **Multi-Content Support**: Translates text, URLs, audio, video, images, and PDFs
- **Batch Processing**: Process multiple items simultaneously
- **Fallback Models**: Automatic fallback to llama3.2:latest if primary model fails

### 🌍 Supported Languages
- **European**: Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt)
- **Asian**: Chinese (zh), Japanese (ja), Korean (ko), Thai (th)
- **Middle Eastern**: Arabic (ar)
- **South Asian**: Hindi (hi)
- **Cyrillic**: Russian (ru)
- **Default**: English (en) - no translation needed

## Models Used

### Primary Translation Model
- **Model**: `mistral-small3.1:latest`
- **Use Case**: High-quality translation with excellent multilingual capabilities
- **Temperature**: 0.3 (balanced creativity and accuracy)
- **Max Tokens**: 500 (sufficient for most translations)

### Fallback Models
- **Primary Fallback**: `llama3.2:latest`
- **Fast Translation**: `phi3:mini` (for quick translations)
- **Vision Translation**: `llava:latest` (for image content)

## Usage

### 1. Text Translation

```python
# Single text translation
result = await translate_text("Hola, ¿cómo estás?")

# Response:
{
    "success": True,
    "agent": "translation",
    "original_text": "Hola, ¿cómo estás?",
    "translated_text": "Hello, how are you?",
    "source_language": "es",
    "translation_memory_hit": False,
    "model_used": "mistral-small3.1:latest",
    "processing_time": 1.23
}
```

### 2. Webpage Translation

```python
# Translate webpage content
result = await translate_webpage("https://example.com/spanish-content")

# Response includes:
{
    "success": True,
    "agent": "translation",
    "url": "https://example.com/spanish-content",
    "original_text": "Contenido original en español...",
    "translated_text": "Original content in Spanish...",
    "source_language": "es",
    "translation_memory_hit": False,
    "model_used": "mistral-small3.1:latest"
}
```

### 3. Audio Translation

```python
# Translate audio file (transcribe + translate)
result = await translate_audio("path/to/spanish_audio.mp3")

# Response includes:
{
    "success": True,
    "agent": "translation",
    "audio_path": "path/to/spanish_audio.mp3",
    "original_text": "Transcribed Spanish audio content...",
    "translated_text": "Translated English content...",
    "source_language": "es"
}
```

### 4. Video Translation

```python
# Translate video content (audio + visual)
result = await translate_video("path/to/spanish_video.mp4")

# Response includes:
{
    "success": True,
    "agent": "translation",
    "video_path": "path/to/spanish_video.mp4",
    "original_text": "Combined audio and visual content...",
    "translated_text": "Translated English content...",
    "source_language": "es"
}
```

### 5. PDF Translation

```python
# Translate PDF content (extract text and translate)
result = await translate_pdf("path/to/spanish_document.pdf")

# Response includes:
{
    "success": True,
    "agent": "translation",
    "pdf_path": "path/to/spanish_document.pdf",
    "original_text": "Extracted text from PDF...",
    "translated_text": "Translated English content...",
    "source_language": "es",
    "translation_memory_hit": False,
    "model_used": "mistral-small3.1:latest",
    "processing_time": 2.45
}
```

### 6. Image Translation

```python
# Translate image content (extract text using vision model and translate)
result = await translate_image("path/to/spanish_image.jpg")

# Response includes:
{
    "success": True,
    "agent": "translation",
    "image_path": "path/to/spanish_image.jpg",
    "original_text": "Text extracted from image...",
    "translated_text": "Translated English content...",
    "source_language": "es"
}

# Response includes:
{
    "success": True,
    "agent": "translation",
    "video_path": "path/to/spanish_video.mp4",
    "original_text": "Combined audio and visual content...",
    "translated_text": "Translated combined content...",
    "source_language": "es"
}
```

### 5. Batch Translation

```python
# Batch translate multiple items
requests = [
    {"data_type": "text", "content": "Bonjour le monde"},
    {"data_type": "text", "content": "Hola mundo"},
    {"data_type": "webpage", "content": "https://example.com/french"}
]

result = await batch_translate(requests)

# Response includes:
{
    "success": True,
    "agent": "translation",
    "total_requests": 3,
    "completed": 3,
    "failed": 0,
    "results": [
        {
            "success": True,
            "original_text": "Bonjour le monde",
            "translated_text": "Hello world",
            "source_language": "fr"
        },
        # ... more results
    ]
}
```

## MCP Server Integration

### Available Tools

The Translation Agent is fully integrated with the MCP server and provides the following tools:

1. **translate_text** - Translate text content to English
2. **translate_webpage** - Translate webpage content to English
3. **translate_audio** - Translate audio content to English
4. **translate_video** - Translate video content to English
5. **batch_translate** - Batch translate multiple content items

### Tool Descriptions

All translation tools include:
- Automatic language detection
- Translation memory integration
- Model usage tracking
- Processing time measurement
- Error handling and suggestions

## Translation Memory

### How It Works

1. **Storage**: All translations are stored in Chroma vector DB with metadata
2. **Retrieval**: Similar text queries are matched using vector similarity
3. **Consistency**: Ensures consistent translations across similar content
4. **Performance**: Reduces processing time for repeated translations

### Memory Features

- **Similarity Threshold**: 0.8 (80% similarity for memory hits)
- **Metadata Storage**: Original text, translation, language, model used, timestamp
- **Automatic Cleanup**: Old translations are managed by Chroma DB

## Configuration

### Ollama Models

Translation models are configured in `src/config/ollama_config.py`:

```python
"translation": OllamaModelConfig(
    model_id="mistral-small3.1:latest",
    temperature=0.3,
    max_tokens=500,
    keep_alive="10m",
    capabilities=["translation", "multilingual"],
    is_shared=True,
    fallback_model="llama3.2:latest"
)
```

### Agent Mapping

The Translation Agent is mapped in the agent configuration:

```python
"TranslationAgent": "translation"
```

## Error Handling

### Common Errors

1. **File Not Found**: Audio/video files don't exist
2. **Unsupported Format**: File format not supported
3. **Network Issues**: Webpage access problems
4. **Model Failures**: Ollama model unavailable

### Error Responses

All tools return structured error responses:

```python
{
    "success": False,
    "error": "Error description",
    "suggestion": "Helpful suggestion for resolution"
}
```

## Performance Considerations

### Optimization Tips

1. **Batch Processing**: Use batch_translate for multiple items
2. **Translation Memory**: Leverage stored translations for consistency
3. **Model Selection**: Use appropriate models for different content types
4. **Concurrent Processing**: Batch operations use semaphore limiting (5 concurrent)

### Resource Usage

- **Memory**: Translation memory stored in Chroma DB
- **Processing**: Parallel processing with controlled concurrency
- **Models**: Shared model instances for efficiency

## Integration with Existing Agents

### Seamless Integration

The Translation Agent works alongside existing agents:

1. **Audio Agent**: Uses for transcription before translation
2. **Video Agent**: Uses for content extraction before translation
3. **OCR Agent**: Uses for image text extraction before translation
4. **Web Agent**: Uses for webpage content extraction before translation

### Workflow Example

```python
# 1. Audio transcription + translation
audio_result = await translate_audio("spanish_audio.mp3")

# 2. Sentiment analysis on translated content
sentiment_result = await analyze_text_sentiment(
    audio_result["translated_text"]
)
```

## Testing

### Test Files

Create test files in the `Test/` directory:

```python
# test_translation.py
async def test_text_translation():
    result = await translate_text("Bonjour le monde")
    assert result["success"] == True
    assert result["translated_text"] == "Hello world"
    assert result["source_language"] == "fr"
```

### Sample Content

Test with various languages and content types:
- Spanish text: "Hola, ¿cómo estás?"
- French audio: Record French speech
- German webpage: German news site
- Chinese video: Chinese tutorial video

## Troubleshooting

### Common Issues

1. **Ollama Not Running**: Ensure Ollama is running on localhost:11434
2. **Model Not Available**: Pull required models: `ollama pull mistral-small3.1:latest`
3. **Memory Issues**: Check Chroma DB connection and storage
4. **Performance**: Monitor model loading and response times

### Debug Information

Enable debug logging to see detailed translation process:

```python
import logging
logging.getLogger("src.agents.translation_agent").setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features

1. **Real-time Translation**: Stream translation for live content
2. **Custom Models**: Support for custom translation models
3. **Quality Assessment**: Automatic translation quality scoring
4. **Domain-specific Translation**: Specialized models for technical/medical content
5. **Multi-target Translation**: Translate to languages other than English

### API Extensions

1. **Translation Quality Metrics**: Confidence scores and quality indicators
2. **Language Detection API**: Standalone language detection service
3. **Translation Memory Management**: Manual memory management tools
4. **Model Performance Monitoring**: Translation model performance tracking

## Conclusion

The Translation Agent provides a comprehensive, high-quality translation solution that integrates seamlessly with the existing sentiment analysis system. With automatic language detection, translation memory, and support for multiple content types, it offers a robust foundation for multilingual content processing.
