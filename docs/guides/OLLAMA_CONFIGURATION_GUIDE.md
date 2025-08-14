# Ollama Configuration Guide

This guide explains how Ollama models are configurable for all 7 agents in the Sentiment project.

## Overview

The Ollama integration provides a comprehensive, configurable system that allows you to:
- Configure models for specific agent types
- Tune performance parameters
- Set up connection pooling
- Configure model sharing and fallbacks
- Validate configurations automatically
- Update settings dynamically at runtime

## Configuration Structure

### 1. Model Configuration (`OllamaModelConfig`)

Each model type has its own configuration:

```python
from config.ollama_config import OllamaModelConfig

# Example configuration
text_model = OllamaModelConfig(
    model_id="llama3.2:latest",
    temperature=0.1,
    max_tokens=100,
    keep_alive="5m",
    capabilities=["text", "sentiment_analysis"],
    is_shared=True,
    fallback_model="phi3:mini",
    performance_threshold=0.8
)
```

**Configurable Parameters:**
- `model_id`: The Ollama model identifier
- `temperature`: Creativity level (0.0 - 2.0)
- `max_tokens`: Maximum output length (1 - 4096)
- `keep_alive`: How long to keep model in memory
- `capabilities`: List of model capabilities
- `is_shared`: Whether model can be shared between agents
- `fallback_model`: Backup model if primary fails
- `performance_threshold`: Minimum performance requirement

### 2. Connection Configuration (`OllamaConnectionConfig`)

Network and connection settings:

```python
from config.ollama_config import OllamaConnectionConfig

connection_config = OllamaConnectionConfig(
    host="http://localhost:11434",
    max_connections=10,
    max_keepalive=30,
    connection_timeout=30,
    retry_attempts=3,
    health_check_interval=60
)
```

**Configurable Parameters:**
- `host`: Ollama server URL
- `max_connections`: Connection pool size
- `max_keepalive`: Keepalive timeout
- `connection_timeout`: Request timeout
- `retry_attempts`: Retry count on failure
- `health_check_interval`: Health check frequency

### 3. Performance Configuration (`OllamaPerformanceConfig`)

Performance monitoring and optimization:

```python
from config.ollama_config import OllamaPerformanceConfig

perf_config = OllamaPerformanceConfig(
    enable_metrics=True,
    cleanup_interval=300,
    max_idle_time=300,
    performance_window=3600,
    enable_auto_scaling=True,
    min_models=1,
    max_models=10
)
```

**Configurable Parameters:**
- `enable_metrics`: Enable performance tracking
- `cleanup_interval`: Model cleanup frequency
- `max_idle_time`: Max time before cleanup
- `performance_window`: Metrics window size
- `enable_auto_scaling`: Automatic model scaling
- `min_models`: Minimum models to keep
- `max_models`: Maximum models to load

## Agent-Specific Configurations

Each of the 7 agents has its own model configuration:

### Text Agents
- **TextAgent**: Uses `llama3.2:latest` with low temperature for consistent analysis
- **SimpleTextAgent**: Similar to TextAgent but with simplified processing
- **TextAgentSwarm**: Uses `llama3.2:latest` with medium temperature for coordination

### Vision and Audio
- **VisionAgent**: Uses `llava:latest` for image analysis
- **AudioAgent**: Shares `llava:latest` with VisionAgent for efficiency

### Specialized Agents
- **WebAgent**: Uses `llama3.2:latest` for web content analysis
- **OrchestratorAgent**: Uses `llama3.2:latest` for coordination tasks

## Configuration Methods

### 1. Static Configuration

Define configurations in code:

```python
from config.ollama_config import get_ollama_config

config = get_ollama_config()
config.models["custom"] = OllamaModelConfig(
    model_id="llama3.2:latest",
    temperature=0.5,
    max_tokens=200
)
```

### 2. Dynamic Updates

Update configurations at runtime:

```python
from config.ollama_config import update_ollama_config

# Update connection settings
update_ollama_config(
    connection=OllamaConnectionConfig(
        max_connections=20,
        connection_timeout=45
    )
)

# Update specific model
config = get_ollama_config()
config.models["text"].temperature = 0.2
config.models["text"].max_tokens = 150
```

### 3. Agent Mapping

Configure which model each agent uses:

```python
config = get_ollama_config()

# Change TextAgent to use a different model
config.agent_model_mapping["TextAgent"] = "custom"

# Add new agent type
config.agent_model_mapping["NewAgent"] = "text"
```

## Model Sharing Configuration

Configure models to be shared between agents:

```python
# Vision and Audio agents share the same llava model
vision_config = OllamaModelConfig(
    model_id="llava:latest",
    is_shared=True,
    capabilities=["vision", "image_analysis"]
)

audio_config = OllamaModelConfig(
    model_id="llava:latest",  # Same model
    is_shared=True,
    capabilities=["audio", "transcription"]
)
```

## Fallback Configuration

Set up fallback chains for reliability:

```python
# Primary model with fallback
primary_config = OllamaModelConfig(
    model_id="llama3.2:latest",
    fallback_model="phi3:mini"
)

# Fallback model
fallback_config = OllamaModelConfig(
    model_id="phi3:mini",
    fallback_model="llama3.2:latest"
)
```

## Validation

All configurations are automatically validated:

```python
try:
    # This will fail validation
    invalid_config = OllamaModelConfig(
        model_id="test",
        temperature=3.0,  # Invalid: > 2.0
        max_tokens=10000  # Invalid: > 4096
    )
except Exception as e:
    print(f"Validation error: {e}")
```

## Performance Tuning

### Connection Pooling
```python
# High-performance connection settings
high_perf_connection = OllamaConnectionConfig(
    max_connections=50,
    max_keepalive=60,
    connection_timeout=15,
    retry_attempts=2
)
```

### Model Management
```python
# Aggressive cleanup for memory-constrained environments
aggressive_perf = OllamaPerformanceConfig(
    cleanup_interval=180,  # 3 minutes
    max_idle_time=300,     # 5 minutes
    max_models=5           # Limit total models
)
```

## Configuration Persistence

Save and restore configurations:

```python
import json
from config.ollama_config import get_ollama_config

# Save configuration
config = get_ollama_config()
config_dict = config.model_dump()

with open("ollama_config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

# Restore configuration
with open("ollama_config.json", "r") as f:
    saved_config = json.load(f)

restored_config = OptimizedOllamaConfig(**saved_config)
```

## Environment-Specific Configurations

### Development
```python
dev_config = OllamaConnectionConfig(
    host="http://localhost:11434",
    max_connections=5,
    connection_timeout=60
)
```

### Production
```python
prod_config = OllamaConnectionConfig(
    host="http://ollama-server:11434",
    max_connections=50,
    connection_timeout=15,
    retry_attempts=5
)
```

### Testing
```python
test_config = OllamaPerformanceConfig(
    enable_metrics=False,
    cleanup_interval=60,
    max_models=3
)
```

## Best Practices

1. **Use Appropriate Temperatures**
   - Low (0.0-0.3): Consistent, factual responses
   - Medium (0.4-0.7): Balanced creativity
   - High (0.8-1.0): Creative, varied responses

2. **Configure Model Sharing**
   - Share compatible models between agents
   - Reduce memory usage and startup time

3. **Set Up Fallbacks**
   - Ensure reliability with backup models
   - Consider performance vs. availability trade-offs

4. **Monitor Performance**
   - Enable metrics for optimization
   - Adjust settings based on usage patterns

5. **Validate Configurations**
   - Let Pydantic handle validation
   - Test configurations before deployment

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure `src` directory is in Python path
   - Check that all dependencies are installed

2. **Validation Errors**
   - Check parameter ranges (temperature 0.0-2.0, max_tokens 1-4096)
   - Verify required fields are provided

3. **Configuration Not Applied**
   - Use `update_ollama_config()` for global updates
   - Modify config objects directly for specific changes

### Debug Configuration

```python
from config.ollama_config import get_ollama_config

config = get_ollama_config()
print(f"Current configuration: {config.model_dump_json(indent=2)}")
```

## Summary

The Ollama configuration system provides:

✅ **Full Model Configuration**: Every aspect of Ollama models is configurable
✅ **Agent-Specific Settings**: Each agent can have optimized configurations
✅ **Dynamic Updates**: Change settings at runtime without restart
✅ **Automatic Validation**: Pydantic ensures configuration correctness
✅ **Performance Tuning**: Optimize for your specific use case
✅ **Model Sharing**: Efficient resource utilization
✅ **Fallback Support**: Reliable operation with backup models
✅ **Persistence**: Save and restore configurations

This system ensures that Ollama models are fully configurable for all 7 agents while maintaining performance, reliability, and ease of use.
