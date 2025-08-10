# Configurable Models Guide

## Overview

The sentiment analysis system now supports fully configurable models through environment variables. This allows you to easily switch between different Ollama models without modifying code.

## 🎯 **Key Features**

- **Two Model Types**: Text model and Vision model (for audio/video/image)
- **Environment Variable Configuration**: No hardcoded models
- **Fallback Support**: Automatic fallback when primary models fail
- **Configurable Ollama Host**: Support for local and remote Ollama servers
- **Flexible Parameters**: Temperature, max tokens, and other settings configurable

## 📋 **Model Types**

### 1. **Text Model** (`TEXT_MODEL`)
- **Purpose**: Sentiment analysis, entity extraction, text processing
- **Used by**: Text agents, knowledge graph agents, translation agents
- **Default**: `ollama:mistral-small3.1:latest`

### 2. **Vision Model** (`VISION_MODEL`)
- **Purpose**: Audio, video, and image processing
- **Used by**: Vision agents, audio agents, video agents
- **Default**: `ollama:llava:latest`

## 🔧 **Configuration**

### Environment Variables

Create a `.env` file in your project root with these variables:

```bash
# Primary Models
TEXT_MODEL=ollama:mistral-small3.1:latest
VISION_MODEL=ollama:llava:latest

# Fallback Models
FALLBACK_TEXT_MODEL=ollama:llama3.2:latest
FALLBACK_VISION_MODEL=ollama:granite3.2-vision

# Ollama Server
OLLAMA_HOST=http://localhost:11434
OLLAMA_TIMEOUT=30

# Model Parameters
TEXT_TEMPERATURE=0.1
TEXT_MAX_TOKENS=200
VISION_TEMPERATURE=0.7
VISION_MAX_TOKENS=200

# Tool Calling
ENABLE_TOOL_CALLING=true
MAX_TOOL_CALLS=5
```

### Configuration Examples

#### **Development Setup**
```bash
TEXT_MODEL=ollama:mistral-small3.1:latest
VISION_MODEL=ollama:llava:latest
OLLAMA_HOST=http://localhost:11434
```

#### **Production Setup**
```bash
TEXT_MODEL=ollama:llama3.2:latest
VISION_MODEL=ollama:granite3.2-vision
OLLAMA_HOST=http://your-ollama-server:11434
```

#### **Testing Setup**
```bash
TEXT_MODEL=ollama:phi3:mini
VISION_MODEL=ollama:llava:latest
OLLAMA_HOST=http://localhost:11434
```

## 🚀 **Usage**

### 1. **Set Environment Variables**

Copy the example file and customize:
```bash
cp env.example .env
# Edit .env with your preferred models
```

### 2. **Run the System**

The system will automatically use your configured models:
```bash
.venv/Scripts/python.exe main.py
```

### 3. **Test Configuration**

Verify your configuration:
```bash
.venv/Scripts/python.exe Test/test_configurable_models.py
```

## 📁 **File Structure**

### Configuration Files
- `src/config/model_config.py` - Main model configuration
- `src/config/config.py` - Updated with configurable models
- `src/config/ollama_config.py` - Enhanced Ollama configuration
- `env.example` - Example environment variables

### Integration
- `src/core/ollama_integration.py` - Updated to use configurable models
- `Test/test_configurable_models.py` - Test suite for configuration

## 🔄 **Model Routing**

The system automatically routes different agent types to appropriate models:

| Agent Type | Model Used | Configuration |
|------------|------------|---------------|
| `text` | Text Model | `TEXT_MODEL` |
| `vision` | Vision Model | `VISION_MODEL` |
| `audio` | Vision Model | `VISION_MODEL` |
| `video` | Vision Model | `VISION_MODEL` |
| `image` | Vision Model | `VISION_MODEL` |

## ⚙️ **Advanced Configuration**

### Model Parameters

#### **Temperature**
- **Text Models**: Lower temperature (0.1) for consistent analysis
- **Vision Models**: Higher temperature (0.7) for creative processing

#### **Max Tokens**
- **Text Models**: 200 tokens for concise responses
- **Vision Models**: 200 tokens for detailed analysis

#### **Fallback Models**
- Automatically used when primary models fail
- Ensures system reliability

### Ollama Configuration

#### **Host Configuration**
- **Local**: `http://localhost:11434`
- **Remote**: `http://your-server:11434`
- **Custom Port**: `http://localhost:8080`

#### **Timeout Settings**
- **Default**: 30 seconds
- **Adjustable**: Based on model response times

## 🧪 **Testing**

### Run Configuration Tests
```bash
.venv/Scripts/python.exe Test/test_configurable_models.py
```

### Expected Output
```
📋 Testing Text Model Configuration:
  - Model ID: ollama:your-text-model:latest
  - Fallback Model: ollama:your-fallback-text:latest
  - Host: http://your-ollama-host:11434

📋 Testing Vision Model Configuration:
  - Model ID: ollama:your-vision-model:latest
  - Fallback Model: ollama:your-fallback-vision:latest
  - Host: http://your-ollama-host:11434

✅ SUCCESS: All model configurations working correctly
```

## 🔧 **Troubleshooting**

### Common Issues

#### **1. Models Not Loading**
- Check if Ollama is running
- Verify model names are correct
- Check network connectivity for remote hosts

#### **2. Environment Variables Not Working**
- Ensure `.env` file is in project root
- Check variable names match exactly
- Restart the application after changes

#### **3. Fallback Models Not Working**
- Verify fallback models are installed in Ollama
- Check model names match Ollama's installed models

### Debug Commands

#### **Check Ollama Models**
```bash
ollama list
```

#### **Test Ollama Connection**
```bash
curl http://localhost:11434/api/tags
```

#### **Verify Environment Variables**
```bash
python -c "from src.config.model_config import model_config; print(model_config.get_text_model_config())"
```

## 📈 **Performance Optimization**

### Model Selection Tips

#### **For Speed**
- Use smaller models: `phi3:mini`, `mistral-small3.1:latest`
- Lower temperature values
- Reduce max tokens

#### **For Quality**
- Use larger models: `llama3.2:latest`, `granite3.2-vision`
- Higher temperature for creativity
- Increase max tokens for detailed responses

#### **For Reliability**
- Always configure fallback models
- Use models with good availability
- Test models before deployment

## 🔄 **Migration Guide**

### From Hardcoded Models

#### **Before (Hardcoded)**
```python
# Old way - hardcoded in code
model_id = "ollama:mistral-small3.1:latest"
host = "http://localhost:11434"
```

#### **After (Configurable)**
```bash
# New way - environment variables
TEXT_MODEL=ollama:mistral-small3.1:latest
OLLAMA_HOST=http://localhost:11434
```

### Code Changes

The system automatically uses the new configuration:
- No code changes required
- Environment variables override defaults
- Backward compatibility maintained

## 🎯 **Best Practices**

### 1. **Environment Management**
- Use different `.env` files for different environments
- Never commit `.env` files to version control
- Document required environment variables

### 2. **Model Selection**
- Test models before production use
- Monitor model performance and reliability
- Keep fallback models updated

### 3. **Configuration Validation**
- Run tests after configuration changes
- Validate model availability
- Check system performance

### 4. **Security**
- Use secure connections for remote Ollama servers
- Validate model sources
- Monitor model usage

## 📞 **Support**

For issues with configurable models:

1. **Check the test suite**: `Test/test_configurable_models.py`
2. **Review configuration**: `src/config/model_config.py`
3. **Verify environment variables**: Check `.env` file
4. **Test Ollama connection**: Ensure Ollama is running
5. **Check model availability**: Verify models are installed

---

**Status**: ✅ **IMPLEMENTED AND TESTED**  
**Last Updated**: 2025-08-10  
**Version**: 1.0.0
