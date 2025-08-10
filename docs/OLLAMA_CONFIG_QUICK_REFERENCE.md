# Ollama Configuration Quick Reference

## 🚀 Quick Start

```python
from config.ollama_config import get_ollama_config, update_ollama_config

# Get current config
config = get_ollama_config()

# Update connection
update_ollama_config(connection=OllamaConnectionConfig(max_connections=20))

# Update model
config.models["text"].temperature = 0.2
```

## 📊 Model Configuration

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `model_id` | string | - | Ollama model identifier |
| `temperature` | 0.0 - 2.0 | 0.7 | Creativity level |
| `max_tokens` | 1 - 4096 | 100 | Max output length |
| `keep_alive` | string | "5m" | Memory retention |
| `is_shared` | bool | False | Share between agents |
| `fallback_model` | string | None | Backup model |

## 🔧 Connection Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `host` | localhost:11434 | Ollama server URL |
| `max_connections` | 10 | Connection pool size |
| `connection_timeout` | 30s | Request timeout |
| `retry_attempts` | 3 | Retry on failure |
| `health_check_interval` | 60s | Health check frequency |

## ⚡ Performance Tuning

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_metrics` | True | Performance tracking |
| `cleanup_interval` | 300s | Model cleanup frequency |
| `max_idle_time` | 300s | Max idle before cleanup |
| `enable_auto_scaling` | True | Automatic scaling |
| `min_models` | 1 | Minimum models to keep |
| `max_models` | 10 | Maximum models to load |

## 🤖 Agent Model Mapping

| Agent | Model | Temperature | Use Case |
|-------|-------|-------------|----------|
| TextAgent | llama3.2:latest | 0.1 | Consistent analysis |
| VisionAgent | llava:latest | 0.7 | Image analysis |
| AudioAgent | llava:latest | 0.7 | Audio transcription |
| WebAgent | llama3.2:latest | 0.1 | Web content analysis |
| OrchestratorAgent | llama3.2:latest | 0.2 | Coordination |
| TextAgentSwarm | llama3.2:latest | 0.3 | Multi-agent planning |
| SimpleTextAgent | llama3.2:latest | 0.1 | Basic text processing |

## 🔄 Model Sharing

```python
# Audio and Vision share llava model
config.models["audio"].is_shared = True
config.models["vision"].is_shared = True
```

## 🆘 Fallback Chains

```python
# Primary → Fallback → Backup
config.models["text"].fallback_model = "phi3:mini"
config.models["phi3:mini"].fallback_model = "llama3.2:latest"
```

## 📝 Common Configurations

### High Performance
```python
update_ollama_config(
    connection=OllamaConnectionConfig(
        max_connections=50,
        connection_timeout=15,
        retry_attempts=2
    ),
    performance=OllamaPerformanceConfig(
        cleanup_interval=180,
        max_models=15
    )
)
```

### Memory Constrained
```python
update_ollama_config(
    performance=OllamaPerformanceConfig(
        max_models=3,
        cleanup_interval=120,
        max_idle_time=180
    )
)
```

### Development
```python
update_ollama_config(
    connection=OllamaConnectionConfig(
        max_connections=5,
        connection_timeout=60
    ),
    performance=OllamaPerformanceConfig(
        enable_metrics=False,
        cleanup_interval=600
    )
)
```

## ✅ Validation Rules

- **Temperature**: 0.0 ≤ temp ≤ 2.0
- **Max Tokens**: 1 ≤ tokens ≤ 4096
- **Model ID**: Required string
- **Capabilities**: List of strings
- **Fallback**: Must reference existing model

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import error | Check `src` in Python path |
| Validation error | Verify parameter ranges |
| Config not applied | Use `update_ollama_config()` |
| Performance issues | Adjust connection pool size |

## 📚 Full Documentation

See `docs/OLLAMA_CONFIGURATION_GUIDE.md` for complete details.
