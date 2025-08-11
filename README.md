# Unified Sentiment Analysis System

A consolidated and optimized sentiment analysis system with unified agents for text, audio, video, and image analysis. The system has been streamlined to use three comprehensive unified agents that replace multiple specialized implementations.

## 🚀 Key Features

- **Unified Agents**: Three comprehensive agents replace 10+ specialized implementations
- **Optimized MCP Server**: Streamlined tool interface with unified access
- **Multi-Modal Analysis**: Text, audio, video, image, webpage, and PDF sentiment analysis
- **Configurable Processing Modes**: Simple, Strands, and Swarm modes for text processing
- **Advanced Audio Processing**: Transcription, summarization, and large file support
- **Comprehensive Vision Analysis**: OCR, object detection, and scene understanding
- **YouTube Integration**: Comprehensive video analysis with parallel processing
- **Knowledge Graph**: Entity extraction and relationship mapping with enhanced categorization
- **Translation Services**: Multi-language content translation and analysis
- **PDF Processing**: Advanced PDF text extraction with PyPDF2 and vision OCR
- **Enhanced Ollama Integration**: Optimized local LLM processing with configurable models
- **Large File Processing**: Advanced chunking and progressive processing
- **Vector Database**: ChromaDB for efficient storage and retrieval
- **Local Deployment**: Run everything locally with CPU optimization
- **Interactive Visualizations**: D3.js-based knowledge graph visualization with zoom and pan
- **Production Ready**: Docker support, monitoring, logging, and security features

### 🤖 Enhanced Ollama Integration

The system features robust Ollama integration for local LLM processing:

- **Optimized Ollama Connection**: Enhanced timeout handling and error recovery
- **Improved Prompt Engineering**: Better sentiment analysis prompts for accurate results
- **Fallback Mechanisms**: Rule-based sentiment analysis when Ollama is unavailable
- **Multiple Model Support**: Compatible with phi3:mini, llama3, and other Ollama models
- **Error Logging**: Comprehensive error tracking and debugging capabilities
- **Connection Stability**: 10-second timeout with retry mechanisms
- **Response Parsing**: Intelligent parsing of Ollama responses for sentiment classification
- **Configurable Models**: Dynamic model selection and configuration through settings

### 🎬 Unified Video Analysis Features

The system includes comprehensive video analysis capabilities that automatically detect and handle different video platforms:

- **Multi-Platform Support**: YouTube, Vimeo, TikTok, Instagram, Facebook, Twitter, Twitch, and more
- **Local Video Processing**: Support for MP4, AVI, MOV, MKV, WebM, FLV, WMV files
- **Automatic Platform Detection**: Intelligent routing based on URL or file path
- **YouTube Comprehensive Analysis**: Full audio/visual sentiment analysis with yt-dlp integration
- **Video Download & Processing**: Automatic video downloading using yt-dlp
- **Audio Extraction**: Extract and analyze audio content for sentiment
- **Frame Extraction**: Extract video frames for visual sentiment analysis
- **Metadata Analysis**: Comprehensive video metadata including title, description, views, likes
- **Multi-Modal Sentiment**: Combined audio and visual sentiment analysis
- **Large File Processing**: Advanced chunking and progressive processing for large video files
- **Unified API**: Single endpoint for all video analysis needs
- **Intelligent Weighting**: Audio (60%) and visual (40%) sentiment combination
- **Batch Processing**: Analyze multiple YouTube videos efficiently
- **Error Handling**: Graceful handling of download restrictions and errors
- **Resource Management**: Automatic cleanup of temporary files

### 🧠 GraphRAG-Inspired Knowledge Graph Features

Advanced knowledge graph functionality with GraphRAG-inspired improvements:

- **Enhanced Entity Categorization**: 100% accurate entity type classification with comprehensive pattern matching
- **Chunk-Based Processing**: Intelligent text splitting with 1200 token chunks and 100 token overlap
- **Advanced Entity Extraction**: Sophisticated prompts with 9 entity types and confidence scoring
- **Comprehensive Relationship Mapping**: 13 relationship types with context-aware inference
- **Multiple Community Detection**: Louvain, Label Propagation, and Girvan-Newman algorithms
- **Robust Error Handling**: Multiple fallback strategies for JSON parsing and entity extraction
- **Interactive D3.js Visualization**: Full zoom and pan capabilities with color-coded nodes and relationship lines
- **Graph Analysis**: Community detection, path finding, and centrality analysis
- **Scalable Processing**: Efficient handling of large documents through chunk-based processing
- **Enhanced Reasoning**: Multi-hop reasoning through graph traversal and relationship analysis
- **Confidence Scoring**: Confidence levels for entities and relationships based on extraction method
- **Duplicate Removal**: Automatic deduplication of entities and relationships across multiple articles
- **Comprehensive Reporting**: Detailed summary reports with entity/relationship breakdowns and statistics
- **Error Handling**: Graceful degradation when original agent fails
- **MCP Server Integration**: Full integration with MCP server for remote access

#### 🎯 Enhanced Entity Categorization System

The knowledge graph now features a sophisticated entity categorization system with:

- **6 Entity Types**: PERSON, ORGANIZATION, LOCATION, CONCEPT, OBJECT, PROCESS
- **250+ Pattern Matches**: Comprehensive pattern matching for accurate categorization
- **100% Test Accuracy**: Verified accuracy on standard entity types
- **Color-Coded Visualization**: Proper entity type mapping to visual groups
- **Enhanced Fallback Logic**: Robust categorization even when AI models fail
- **Confidence Scoring**: Reliable confidence scores (0.7) for pattern-based categorization
- **Extensible Patterns**: Easy to add new patterns for different domains or languages

### 🤖 Unified Agents System

The system has been consolidated to use three comprehensive unified agents:

#### UnifiedTextAgent
- **Three Processing Modes**: Simple, Strands, and Swarm modes
- **Configurable Models**: Support for different Ollama models
- **Multi-language Support**: English and other languages
- **Comprehensive Analysis**: Sentiment, features, and advanced processing

#### UnifiedAudioAgent
- **Audio Transcription**: High-quality speech-to-text conversion
- **Audio Summarization**: Key points and action items extraction
- **Large File Processing**: Chunked analysis for long audio files
- **Multiple Formats**: Support for mp3, wav, flac, m4a, ogg, aac, wma, opus
- **Quality Assessment**: Audio quality and emotion analysis

#### UnifiedVisionAgent
- **Image Analysis**: Comprehensive visual content analysis
- **Object Recognition**: Detection and classification of objects
- **Text Extraction**: OCR capabilities for text in images
- **Scene Understanding**: Context and scene analysis
- **Multiple Formats**: Support for jpg, png, gif, bmp, tiff

### 🔧 Configurable Models System

The system now supports dynamic model configuration:

- **Multiple Model Support**: phi3:mini, llama3, and other Ollama models
- **Dynamic Configuration**: Runtime model selection and configuration
- **Fallback Mechanisms**: Automatic fallback to alternative models
- **Performance Optimization**: Model-specific optimizations and settings
- **Easy Integration**: Simple configuration through settings files

## 📊 Optimized Tool Structure

### Core Management (3 tools)
- `get_all_agents_status` - Get status of all available agents
- `start_all_agents` - Start all agents
- `stop_all_agents` - Stop all agents

### Unified Analysis (4 tools)
- `analyze_text` - Text analysis with agent selection (standard, simple, strands, swarm)
- `analyze_media` - Media analysis (audio, image, webpage, video)
- `analyze_youtube` - YouTube video analysis with parallel processing
- `analyze_content` - Automatic content type detection and analysis

### Summarization (2 tools)
- `summarize_audio` - Comprehensive audio summary generation
- `analyze_video_summarization` - Video summary with key scenes and analysis

### OCR Operations (5 tools)
- `analyze_ocr_text_extraction` - Extract text from images using OCR
- `analyze_ocr_document` - Analyze document structure and extract information
- `analyze_ocr_batch` - Process multiple images for OCR in batch
- `analyze_ocr_report` - Generate comprehensive OCR report for an image
- `analyze_ocr_optimize` - Optimize image specifically for OCR processing

### Translation (7 tools)
- `translate_text` - Translate text content to English
- `translate_webpage` - Translate webpage content to English
- `translate_audio` - Translate audio content to English
- `translate_video` - Translate video content to English
- `translate_pdf` - Translate PDF content to English
- `batch_translate` - Batch translate multiple content items
- `translate_text_comprehensive` - Translate text with comprehensive analysis

### Orchestration (2 tools)
- `process_query_orchestrator` - Process query using OrchestratorAgent
- `get_orchestrator_tools` - Get available tools from OrchestratorAgent

### Knowledge Graph (8 tools)
- `extract_entities` - Entity extraction from text with enhanced categorization
- `map_relationships` - Relationship mapping between entities
- `query_knowledge_graph` - Knowledge graph queries
- `generate_graph_report` - Visual graph report generation
- `analyze_graph_communities` - Community analysis in knowledge graph
- `find_entity_paths` - Find paths between two entities in the graph
- `get_entity_context` - Get context and connections for a specific entity
- `process_content_knowledge_graph` - Content processing and graph building

### Specialized Analysis (4 tools)
- `analyze_chinese_news_comprehensive` - Chinese news analysis with translation
- `process_articles_improved_knowledge_graph` - Process articles with improved knowledge graph utility
- `process_articles_knowledge_graph_integration` - Process articles with knowledge graph integration
- `validate_knowledge_graph_integration` - Knowledge graph integration validation

### 📄 PDF Processing Features

Advanced PDF text extraction capabilities with intelligent strategy selection:

- **Conditional Extraction Strategy**: Uses PyPDF2 for text-based PDFs, PyMuPDF + vision OCR for image-based PDFs
- **Content Analysis**: Automatically determines if PDF contains embedded text or is image-based
- **Parallel Processing**: Multi-threaded page processing for improved performance
- **Real-time Progress Tracking**: Detailed progress updates with ETA calculations
- **ChromaDB Integration**: Automatic storage of extracted text in vector database
- **Memory Management**: Aggressive cleanup and optimization for large PDFs
- **Retry Logic**: Page-level retries with graceful degradation
- **Vision OCR**: Advanced OCR using vision models for scanned documents
- **Error Handling**: Comprehensive error reporting and recovery mechanisms

### 📁 Large File Processing Features

Advanced processing capabilities for large audio and video files:

- **Intelligent Chunking**: Automatic splitting of large files into manageable segments (5-minute chunks)
- **Progressive Processing**: Stage-by-stage analysis with real-time progress reporting
- **Memory-Efficient Streaming**: Process files without loading entire content into memory
- **Progress Tracking**: Real-time status updates with ETA calculations
- **Caching Support**: Framework for result caching to avoid reprocessing
- **Parallel Processing**: Multi-worker support for concurrent chunk processing
- **FFmpeg Integration**: Professional-grade media manipulation using ffmpeg/ffprobe
- **Error Recovery**: Graceful handling of processing failures with cleanup
- **Resource Management**: Automatic cleanup of temporary files and chunks

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Git
- Virtual environment (recommended)
- Ollama (for local LLM processing)
- FFmpeg (for video processing)

### Development Setup
```bash
# Clone the repository
git clone <repository-url>
cd Sentiment

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install MCP dependencies (optional)
pip install fastmcp

# Install Ollama (if not already installed)
# Visit https://ollama.ai for installation instructions

# Pull required models
ollama pull phi3:mini
ollama pull llama3
```

### Production Setup

#### Docker Deployment (Recommended)
```bash
# Build the production image
docker build -t sentiment-analysis:latest .

# Run with environment variables
docker run -d \
  --name sentiment-analysis \
  -p 8000:8000 \
  -p 8002:8002 \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -e TEXT_MODEL=ollama:mistral-small3.1:latest \
  -e VISION_MODEL=ollama:llava:latest \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/cache:/app/cache \
  sentiment-analysis:latest
```

#### Docker Compose (Multi-Service)
```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop services
docker-compose -f docker-compose.prod.yml down
```

#### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n sentiment-analysis

# Access the service
kubectl port-forward svc/sentiment-analysis-api 8002:8002
```

## 🚀 Quick Start

### Development Mode
```bash
# Using Python directly
python main.py

# Or using the virtual environment
.venv/Scripts/python.exe main.py
```

### Production Mode
```bash
# Using Docker
docker run -d --name sentiment-analysis -p 8000:8000 -p 8002:8002 sentiment-analysis:latest

# Using Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Using Kubernetes
kubectl apply -f k8s/
```

### Access Points
- **FastAPI Server**: http://0.0.0.0:8002
- **API Documentation**: http://0.0.0.0:8002/docs
- **Health Check**: http://0.0.0.0:8002/health
- **MCP Server**: http://localhost:8000/mcp
- **Metrics**: http://0.0.0.0:8002/metrics (Prometheus format)

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `env.example`:

```bash
# Copy example configuration
cp env.example .env

# Edit for your environment
nano .env
```

#### Production Configuration
```bash
# Model Configuration
TEXT_MODEL=ollama:mistral-small3.1:latest
VISION_MODEL=ollama:llava:latest
FALLBACK_TEXT_MODEL=ollama:llama3.2:latest
FALLBACK_VISION_MODEL=ollama:granite3.2-vision

# Ollama Configuration
OLLAMA_HOST=http://your-ollama-server:11434
OLLAMA_TIMEOUT=30

# Performance Settings
MAX_WORKERS=4
CHUNK_SIZE=1200
OVERLAP_SIZE=100

# Security Settings
API_KEY=your-secure-api-key
CORS_ORIGINS=https://your-domain.com
RATE_LIMIT=100

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO
SENTRY_DSN=your-sentry-dsn

# Storage
CHROMA_PERSIST_DIRECTORY=/app/data/chroma
CACHE_DIRECTORY=/app/cache
```

### Model Configuration

The system supports multiple Ollama models with automatic fallback:

```yaml
# config/models.yaml
models:
  text:
    primary: "ollama:mistral-small3.1:latest"
    fallback: "ollama:llama3.2:latest"
    parameters:
      temperature: 0.1
      max_tokens: 200
  vision:
    primary: "ollama:llava:latest"
    fallback: "ollama:granite3.2-vision"
    parameters:
      temperature: 0.7
      max_tokens: 200
```

## 📖 Usage Examples

### Text Analysis
```python
# Analyze text with different agents
result = await analyze_text(
    text="I love this product!",
    agent_type="swarm",  # Options: standard, simple, strands, swarm
    language="en"
)
```

### Media Analysis
```python
# Analyze different media types
result = await analyze_media(
    content_path="path/to/file.jpg",
    media_type="image",  # Options: audio, image, webpage, video
    language="en"
)
```

### PDF Analysis
```python
# Analyze PDF content and extract text
result = await analyze_pdf(
    pdf_path="path/to/document.pdf",
    model_preference="llava:latest",  # Vision model for OCR
    reflection_enabled=True,
    max_iterations=3,
    confidence_threshold=0.8
)

# Access extracted text
extracted_text = result.extracted_text
processing_method = result.metadata["method"]  # "pypdf2" or "vision_ocr"
```

### Simple PDF Text Extraction
```python
# For basic PDF text extraction without the full agent framework
from src.agents.extract_pdf_text import extract_pdf_text, extract_pdf_text_to_file

# Extract text
text = extract_pdf_text("path/to/document.pdf")

# Extract and save to file
success = extract_pdf_text_to_file("path/to/document.pdf", "output.txt")
```

### YouTube Analysis
```python
# Analyze YouTube video with parallel processing
result = await analyze_youtube(
    youtube_url="https://www.youtube.com/watch?v=example",
    use_parallel=True,
    num_frames=5
)
```

### Knowledge Graph Analysis
```python
# Extract entities from text
entities = await extract_entities(
    text="Apple Inc. was founded by Steve Jobs in Cupertino, California."
)

# Map relationships between entities
relationships = await map_relationships(
    text="Apple Inc. was founded by Steve Jobs in Cupertino, California.",
    entities=entities
)

# Generate visual graph report
await generate_graph_report(output_path="graph_report.html")
```

### Translation Services
```python
# Translate text to English
translated = await translate_text(
    text="Bonjour le monde!",
    language="fr"
)

# Translate webpage
webpage_translated = await translate_webpage(
    url="https://example.com/french-page"
)
```

## 🚀 Production Deployment

### Docker Deployment

#### Single Container
```bash
# Build production image
docker build -t sentiment-analysis:latest .

# Run with production settings
docker run -d \
  --name sentiment-analysis \
  --restart unless-stopped \
  -p 8000:8000 \
  -p 8002:8002 \
  -e NODE_ENV=production \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -v /path/to/data:/app/data \
  -v /path/to/cache:/app/cache \
  sentiment-analysis:latest
```

#### Multi-Container with Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  sentiment-analysis:
    build: .
    restart: unless-stopped
    ports:
      - "8000:8000"
      - "8002:8002"
    environment:
      - NODE_ENV=production
      - OLLAMA_HOST=http://ollama:11434
    volumes:
      - ./data:/app/data
      - ./cache:/app/cache
    depends_on:
      - ollama
      - redis

  ollama:
    image: ollama/ollama:latest
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  ollama_data:
  redis_data:
```

### Kubernetes Deployment

#### Namespace
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: sentiment-analysis
```

#### ConfigMap
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sentiment-config
  namespace: sentiment-analysis
data:
  TEXT_MODEL: "ollama:mistral-small3.1:latest"
  VISION_MODEL: "ollama:llava:latest"
  OLLAMA_HOST: "http://ollama-service:11434"
  LOG_LEVEL: "INFO"
```

#### Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-analysis
  namespace: sentiment-analysis
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentiment-analysis
  template:
    metadata:
      labels:
        app: sentiment-analysis
    spec:
      containers:
      - name: sentiment-analysis
        image: sentiment-analysis:latest
        ports:
        - containerPort: 8000
        - containerPort: 8002
        envFrom:
        - configMapRef:
            name: sentiment-config
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Service
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: sentiment-analysis-service
  namespace: sentiment-analysis
spec:
  selector:
    app: sentiment-analysis
  ports:
  - name: mcp
    port: 8000
    targetPort: 8000
  - name: api
    port: 8002
    targetPort: 8002
  type: ClusterIP
```

#### Ingress
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sentiment-analysis-ingress
  namespace: sentiment-analysis
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: sentiment.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: sentiment-analysis-service
            port:
              number: 8002
```

### Monitoring and Observability

#### Prometheus Metrics
The system exposes Prometheus-compatible metrics at `/metrics`:

```bash
# Example metrics
sentiment_analysis_requests_total{endpoint="/analyze_text",status="200"} 1234
sentiment_analysis_duration_seconds{endpoint="/analyze_text"} 0.5
sentiment_analysis_model_usage{model="mistral-small3.1"} 567
```

#### Grafana Dashboard
Import the provided Grafana dashboard configuration:

```bash
# Import dashboard
curl -X POST http://grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -d @monitoring/grafana-dashboard.json
```

#### Logging Configuration
```yaml
# config/logging.yaml
version: 1
formatters:
  detailed:
    format: '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    formatter: detailed
    level: INFO
  file:
    class: logging.handlers.RotatingFileHandler
    filename: logs/sentiment.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    formatter: detailed
    level: INFO
root:
  level: INFO
  handlers: [console, file]
```

### Security Considerations

#### API Security
```python
# Enable API key authentication
API_KEY_HEADER = "X-API-Key"
API_KEY = os.getenv("API_KEY")

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if request.url.path in ["/health", "/metrics", "/docs"]:
        return await call_next(request)
    
    api_key = request.headers.get(API_KEY_HEADER)
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return await call_next(request)
```

#### CORS Configuration
```python
# Configure CORS for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

#### Rate Limiting
```python
# Implement rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/analyze_text")
@limiter.limit("100/minute")
async def analyze_text(request: Request, text: str):
    # Implementation
    pass
```

### Performance Optimization

#### Caching Strategy
```python
# Redis caching for expensive operations
import redis
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expire_time=3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            cached_result = redis_client.get(cache_key)
            
            if cached_result:
                return json.loads(cached_result)
            
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expire_time, json.dumps(result))
            return result
        return wrapper
    return decorator
```

#### Connection Pooling
```python
# Optimize database connections
import aiohttp

# Create connection pool
conn = aiohttp.TCPConnector(
    limit=100,
    limit_per_host=30,
    ttl_dns_cache=300,
    use_dns_cache=True
)

session = aiohttp.ClientSession(connector=conn)
```

#### Memory Management
```python
# Implement memory-efficient processing
import gc
import psutil

def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    
    if memory_info.rss > 1024 * 1024 * 1024:  # 1GB
        gc.collect()
        return True
    return False
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **Configuration Guides**: Model configuration, Ollama setup, and system settings
- **Feature Guides**: Knowledge graph, entity categorization, visualization
- **Integration Guides**: MCP server integration and API usage
- **Implementation Summaries**: Detailed technical implementation notes
- **Production Guides**: Deployment, monitoring, and security best practices

## 🧪 Testing

### Development Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest Test/test_knowledge_graph_agent.py
pytest Test/test_configurable_models.py
pytest Test/test_main_integration.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### Production Testing
```bash
# Run production tests
pytest Test/test_production.py

# Load testing
locust -f Test/load_test.py --host=http://localhost:8002

# Security testing
bandit -r src/
safety check
```

### Integration Testing
```bash
# Run integration tests
pytest Test/test_integration.py

# Test Docker deployment
docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# Test Kubernetes deployment
kubectl apply -f k8s/test/
kubectl wait --for=condition=ready pod -l app=sentiment-analysis -n test
```

## 🔧 Maintenance

### Backup Strategy
```bash
# Backup ChromaDB data
tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz cache/chroma_db/

# Backup configuration
cp .env backup_env_$(date +%Y%m%d_%H%M%S)

# Restore from backup
tar -xzf backup_20231201_120000.tar.gz
```

### Log Rotation
```bash
# Configure logrotate
sudo tee /etc/logrotate.d/sentiment-analysis << EOF
/path/to/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 www-data www-data
    postrotate
        systemctl reload sentiment-analysis
    endscript
}
EOF
```

### Health Checks
```bash
# Automated health check script
#!/bin/bash
HEALTH_URL="http://localhost:8002/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)

if [ $RESPONSE -eq 200 ]; then
    echo "Service is healthy"
    exit 0
else
    echo "Service is unhealthy (HTTP $RESPONSE)"
    exit 1
fi
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the documentation in the `docs/` directory
- Review the test files for usage examples
- Open an issue on GitHub
- Check the troubleshooting guide in `docs/TROUBLESHOOTING.md`

## 🔄 Recent Updates

- **Production Deployment**: Added Docker, Kubernetes, and monitoring support
- **Security Features**: API key authentication, CORS, and rate limiting
- **Performance Optimization**: Caching, connection pooling, and memory management
- **Monitoring**: Prometheus metrics, Grafana dashboards, and structured logging
- **Enhanced Entity Categorization**: 100% accurate entity type classification
- **Configurable Models**: Dynamic model selection and configuration
- **Improved Knowledge Graph**: Enhanced visualization and analysis capabilities
- **Optimized MCP Server**: Reduced tool count with unified interfaces
- **Comprehensive Testing**: Extensive test coverage for all features
- **Enhanced Documentation**: Detailed guides and implementation summaries
