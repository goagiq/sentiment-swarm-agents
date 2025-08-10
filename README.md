# Sentiment Analysis Swarm with Optimized MCP Integration

A comprehensive sentiment analysis system with optimized MCP (Model Context Protocol) server integration, providing unified access to multiple AI agents for text, audio, video, and image analysis.

## 🚀 Key Features

- **Optimized MCP Server**: Reduced from 46 to 20 tools with unified interfaces
- **Multi-Modal Analysis**: Text, audio, video, image, and webpage sentiment analysis
- **YouTube Integration**: Comprehensive video analysis with parallel processing
- **Knowledge Graph**: Entity extraction and relationship mapping with enhanced categorization
- **Translation Services**: Multi-language content translation and analysis
- **OCR Capabilities**: Text extraction from images and documents
- **Summarization**: Audio and video content summarization
- **Unified API**: Single interface for all analysis types
- **Enhanced Ollama Integration**: Optimized local LLM processing with configurable models
- **Large File Processing**: Advanced chunking and progressive processing
- **Vector Database**: ChromaDB for efficient storage and retrieval
- **Local Deployment**: Run everything locally with CPU optimization
- **Configurable Models**: Dynamic model selection and configuration
- **Enhanced Entity Categorization**: 100% accurate entity type classification
- **Interactive Visualizations**: D3.js-based knowledge graph visualization with zoom and pan

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

### Setup
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

## 🚀 Quick Start

### Start the Optimized MCP Server
```bash
# Using Python directly
python main.py

# Or using the virtual environment
.venv/Scripts/python.exe main.py
```

### Access Points
- **FastAPI Server**: http://0.0.0.0:8001
- **API Documentation**: http://0.0.0.0:8001/docs
- **Health Check**: http://0.0.0.0:8001/health
- **MCP Server**: http://localhost:8000/mcp

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

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **Configuration Guides**: Model configuration, Ollama setup, and system settings
- **Feature Guides**: Knowledge graph, entity categorization, visualization
- **Integration Guides**: MCP server integration and API usage
- **Implementation Summaries**: Detailed technical implementation notes

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Run all tests
pytest

# Run specific test categories
pytest Test/test_knowledge_graph_agent.py
pytest Test/test_configurable_models.py
pytest Test/test_main_integration.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the documentation in the `docs/` directory
- Review the test files for usage examples
- Open an issue on GitHub

## 🔄 Recent Updates

- **Enhanced Entity Categorization**: 100% accurate entity type classification
- **Configurable Models**: Dynamic model selection and configuration
- **Improved Knowledge Graph**: Enhanced visualization and analysis capabilities
- **Optimized MCP Server**: Reduced tool count with unified interfaces
- **Comprehensive Testing**: Extensive test coverage for all features
- **Enhanced Documentation**: Detailed guides and implementation summaries
