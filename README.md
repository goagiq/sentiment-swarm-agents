# Unified Sentiment Analysis System

A comprehensive, enterprise-grade sentiment analysis system with advanced AI capabilities, multi-agent architecture, and real-time processing. The system provides unified agents for text, audio, video, and image analysis with advanced analytics, knowledge graph generation, and multilingual support.

## 🏗️ **System Architecture**

The Unified Sentiment Analysis System follows a sophisticated multi-layer architecture designed for scalability, extensibility, and high performance:

### **Architecture Overview**

```mermaid
graph TB
    %% Frontend Layer
    subgraph "Frontend Layer"
        UI[Streamlit UI<br/>Interactive Dashboard]
        API_CLIENTS[API Clients<br/>REST/MCP]
    end

    %% API Gateway Layer
    subgraph "API Gateway Layer"
        FASTAPI[FastAPI Server<br/>REST API Endpoints]
        MCP_SERVER[Unified MCP Server<br/>25+ Tools]
        CORS[CORS Middleware<br/>Security]
    end

    %% Orchestration Layer
    subgraph "Orchestration Layer"
        ORCHESTRATOR[SentimentOrchestrator<br/>Agent Swarm Manager]
        ROUTER[Request Router<br/>Load Balancer]
        CACHE[Request Cache<br/>TTL Management]
    end

    %% Agent Swarm Layer
    subgraph "Agent Swarm Layer"
        subgraph "Core Agents"
            TEXT_AGENT[UnifiedTextAgent<br/>Strands + Swarm]
            VISION_AGENT[UnifiedVisionAgent<br/>Image/Video]
            AUDIO_AGENT[UnifiedAudioAgent<br/>Audio Processing]
            FILE_AGENT[EnhancedFileExtractionAgent<br/>PDF/Documents]
            WEB_AGENT[EnhancedWebAgent<br/>Web Content]
            KG_AGENT[KnowledgeGraphAgent<br/>Entity/Relationships]
        end
        
        subgraph "Advanced Analytics Agents"
            FORECAST_AGENT[AdvancedForecastingAgent<br/>Time Series]
            CAUSAL_AGENT[CausalAnalysisAgent<br/>Causal Inference]
            ANOMALY_AGENT[AnomalyDetectionAgent<br/>Outlier Detection]
            ML_AGENT[AdvancedMLAgent<br/>Deep Learning]
            SCENARIO_AGENT[ScenarioAnalysisAgent<br/>What-if Analysis]
        end
        
        subgraph "Support Agents"
            SEARCH_AGENT[SemanticSearchAgent<br/>Vector Search]
            REFLECTION_AGENT[ReflectionCoordinatorAgent<br/>Quality Control]
            REPORT_AGENT[ReportGenerationAgent<br/>Documentation]
            EXPORT_AGENT[DataExportAgent<br/>Data Export]
        end
    end

    %% Core Services Layer
    subgraph "Core Services Layer"
        subgraph "Data Management"
            VECTOR_DB[VectorDBManager<br/>Embeddings Storage]
            KG_UTILITY[KnowledgeGraphUtility<br/>Graph Operations]
            DUPLICATE_DETECT[DuplicateDetectionService<br/>Content Deduplication]
        end
        
        subgraph "Processing Services"
            TRANSLATION[TranslationService<br/>Multilingual Support]
            PERFORMANCE_MON[PerformanceMonitor<br/>System Metrics]
            SEMANTIC_SEARCH[SemanticSearchService<br/>Content Search]
            IMAGE_PROC[ImageProcessingService<br/>Image Analysis]
            VIDEO_PROC[VideoProcessingService<br/>Video Analysis]
            YOUTUBE_DL[YouTubeDLService<br/>Video Download]
        end
        
        subgraph "System Services"
            MODEL_MGR[ModelManager<br/>Model Lifecycle]
            ERROR_HANDLER[ErrorHandlingService<br/>Exception Management]
            CACHING_SVC[CachingService<br/>Multi-level Cache]
            MEMORY_MGR[MemoryManager<br/>Resource Management]
        end
    end

    %% Configuration Layer
    subgraph "Configuration Layer"
        CONFIG[Centralized Config<br/>Settings Management]
        LANG_CONFIG[Language Configs<br/>Multilingual Support]
        MODEL_CONFIG[Model Configs<br/>Ollama Integration]
        MCP_CONFIG[MCP Config<br/>Tool Management]
    end

    %% Storage Layer
    subgraph "Storage Layer"
        VECTOR_STORE[(Vector Database<br/>Chroma/FAISS)]
        FILE_SYSTEM[(File System<br/>Processed Content)]
        CACHE_STORE[(Cache Store<br/>Redis/Memory)]
        RESULTS_DIR[(Results Directory<br/>Analysis Output)]
    end

    %% External Integrations
    subgraph "External Integrations"
        OLLAMA[Ollama Server<br/>Local Model Hosting]
        EXTERNAL_APIS[External APIs<br/>Market Data, Social Media]
        STRANDS[Strands Framework<br/>Agent Coordination]
    end

    %% Infrastructure Layer
    subgraph "Infrastructure Layer"
        DOCKER[Docker Containers<br/>Application Packaging]
        NGINX[Nginx<br/>Reverse Proxy]
        MONITORING[Monitoring Stack<br/>Performance Tracking]
        K8S[Kubernetes<br/>Container Orchestration]
    end

    %% Data Flow Connections
    UI --> FASTAPI
    API_CLIENTS --> FASTAPI
    API_CLIENTS --> MCP_SERVER
    
    FASTAPI --> ORCHESTRATOR
    MCP_SERVER --> ORCHESTRATOR
    CORS --> FASTAPI
    
    ORCHESTRATOR --> ROUTER
    ROUTER --> CACHE
    CACHE --> TEXT_AGENT
    CACHE --> VISION_AGENT
    CACHE --> AUDIO_AGENT
    CACHE --> FILE_AGENT
    CACHE --> WEB_AGENT
    CACHE --> KG_AGENT
    CACHE --> FORECAST_AGENT
    CACHE --> CAUSAL_AGENT
    CACHE --> ANOMALY_AGENT
    CACHE --> ML_AGENT
    CACHE --> SCENARIO_AGENT
    CACHE --> SEARCH_AGENT
    CACHE --> REFLECTION_AGENT
    CACHE --> REPORT_AGENT
    CACHE --> EXPORT_AGENT
    
    %% Agent to Service Connections
    TEXT_AGENT --> TRANSLATION
    VISION_AGENT --> IMAGE_PROC
    AUDIO_AGENT --> VIDEO_PROC
    FILE_AGENT --> YOUTUBE_DL
    KG_AGENT --> KG_UTILITY
    SEARCH_AGENT --> SEMANTIC_SEARCH
    
    %% Service to Storage Connections
    VECTOR_DB --> VECTOR_STORE
    KG_UTILITY --> FILE_SYSTEM
    DUPLICATE_DETECT --> CACHE_STORE
    CACHING_SVC --> CACHE_STORE
    REPORT_AGENT --> RESULTS_DIR
    
    %% External Integration Connections
    TEXT_AGENT --> OLLAMA
    VISION_AGENT --> OLLAMA
    AUDIO_AGENT --> OLLAMA
    WEB_AGENT --> EXTERNAL_APIS
    MODEL_MGR --> OLLAMA
    TEXT_AGENT --> STRANDS
    
    %% Configuration Connections
    CONFIG --> ORCHESTRATOR
    LANG_CONFIG --> TRANSLATION
    MODEL_CONFIG --> MODEL_MGR
    MCP_CONFIG --> MCP_SERVER
    
    %% Infrastructure Connections
    FASTAPI --> DOCKER
    MCP_SERVER --> DOCKER
    DOCKER --> NGINX
    NGINX --> MONITORING
    DOCKER --> K8S
    
    %% Styling
    classDef frontend fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef api fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef orchestration fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef agent fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef service fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef config fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    classDef storage fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    classDef external fill:#fafafa,stroke:#424242,stroke-width:2px
    classDef infrastructure fill:#f5f5f5,stroke:#212121,stroke-width:2px
    
    class UI,API_CLIENTS frontend
    class FASTAPI,MCP_SERVER,CORS api
    class ORCHESTRATOR,ROUTER,CACHE orchestration
    class TEXT_AGENT,VISION_AGENT,AUDIO_AGENT,FILE_AGENT,WEB_AGENT,KG_AGENT,FORECAST_AGENT,CAUSAL_AGENT,ANOMALY_AGENT,ML_AGENT,SCENARIO_AGENT,SEARCH_AGENT,REFLECTION_AGENT,REPORT_AGENT,EXPORT_AGENT agent
    class VECTOR_DB,KG_UTILITY,DUPLICATE_DETECT,TRANSLATION,PERFORMANCE_MON,SEMANTIC_SEARCH,IMAGE_PROC,VIDEO_PROC,YOUTUBE_DL,MODEL_MGR,ERROR_HANDLER,CACHING_SVC,MEMORY_MGR service
    class CONFIG,LANG_CONFIG,MODEL_CONFIG,MCP_CONFIG config
    class VECTOR_STORE,FILE_SYSTEM,CACHE_STORE,RESULTS_DIR storage
    class OLLAMA,EXTERNAL_APIS,STRANDS external
    class DOCKER,NGINX,MONITORING,K8S infrastructure
```

### **Architecture Layers**

#### **1. Frontend Layer**
- **Streamlit UI**: Interactive dashboard for data analysis and visualization
- **API Clients**: REST and MCP protocol clients for system integration

#### **2. API Gateway Layer**
- **FastAPI Server**: Comprehensive REST API with 50+ endpoints
- **Unified MCP Server**: 25+ consolidated tools for external integration
- **CORS Middleware**: Security and cross-origin request handling

#### **3. Orchestration Layer**
- **SentimentOrchestrator**: Central agent swarm manager and request router
- **Request Router**: Intelligent load balancing and request distribution
- **Request Cache**: TTL-based caching for performance optimization

#### **4. Agent Swarm Layer**
- **Core Agents**: Specialized agents for different data types (text, vision, audio, files, web, knowledge graphs)
- **Advanced Analytics Agents**: Forecasting, causal analysis, anomaly detection, machine learning
- **Support Agents**: Search, reflection, reporting, and data export capabilities

#### **5. Core Services Layer**
- **Data Management**: Vector database, knowledge graph utilities, duplicate detection
- **Processing Services**: Translation, performance monitoring, semantic search, media processing
- **System Services**: Model management, error handling, caching, memory management

#### **6. Configuration Layer**
- **Centralized Config**: Unified settings management
- **Language Configs**: Multilingual support and language-specific processing
- **Model Configs**: Ollama integration and model lifecycle management
- **MCP Config**: Tool management and external integration configuration

#### **7. Storage Layer**
- **Vector Database**: Embeddings and semantic search storage
- **File System**: Processed content and analysis results
- **Cache Store**: Multi-level caching for performance
- **Results Directory**: Analysis output and generated reports

#### **8. External Integrations**
- **Ollama Server**: Local model hosting and inference
- **External APIs**: Market data, social media, and third-party services
- **Strands Framework**: Agent coordination and tool calling

#### **9. Infrastructure Layer**
- **Docker Containers**: Application packaging and deployment
- **Nginx**: Reverse proxy and load balancing
- **Monitoring Stack**: Performance tracking and system health
- **Kubernetes**: Container orchestration and scaling

### **Key Architectural Features**

- **🔄 Multi-Agent Architecture**: Swarm-based processing with specialized agents
- **🌐 API-First Design**: REST API and MCP protocol support
- **📊 Advanced Analytics**: Machine learning, forecasting, and causal analysis
- **🔍 Knowledge Graphs**: Entity extraction and relationship mapping
- **🌍 Multilingual Support**: Language-specific processing and configurations
- **⚡ Real-time Processing**: Streaming and live analytics capabilities
- **🔧 Extensible Design**: Plugin-based architecture with MCP integration
- **📈 Scalable Infrastructure**: Docker, Kubernetes, and monitoring support
- **🛡️ Production Ready**: Error handling, caching, and performance optimization

### **Component Breakdown**

#### **Core Agents (src/agents/)**
- **UnifiedTextAgent**: Text processing with Strands framework and swarm coordination
- **UnifiedVisionAgent**: Image and video analysis with OCR and visual understanding
- **UnifiedAudioAgent**: Audio processing, transcription, and audio content analysis
- **EnhancedFileExtractionAgent**: PDF and document processing with OCR
- **EnhancedWebAgent**: Web content scraping and analysis
- **KnowledgeGraphAgent**: Entity extraction, relationship mapping, and graph analysis

#### **Advanced Analytics Agents**
- **AdvancedForecastingAgent**: Time series forecasting and prediction
- **CausalAnalysisAgent**: Causal relationship discovery and analysis
- **AnomalyDetectionAgent**: Statistical outlier detection and anomaly identification
- **AdvancedMLAgent**: Deep learning and machine learning automation
- **ScenarioAnalysisAgent**: What-if scenario modeling and impact analysis

#### **Support Agents**
- **SemanticSearchAgent**: Vector-based semantic search across all content
- **ReflectionCoordinatorAgent**: Quality control and result validation
- **ReportGenerationAgent**: Automated report generation and documentation
- **DataExportAgent**: Data export and integration capabilities

#### **Core Services (src/core/)**
- **VectorDBManager**: Vector database operations and embeddings management
- **KnowledgeGraphUtility**: Graph operations and relationship analysis
- **TranslationService**: Multilingual translation and language processing
- **PerformanceMonitor**: System performance tracking and optimization
- **ModelManager**: Model lifecycle management and Ollama integration
- **ErrorHandlingService**: Comprehensive error handling and recovery
- **CachingService**: Multi-level caching for performance optimization

#### **Configuration System (src/config/)**
- **Centralized Config**: Unified settings management across all components
- **Language Configs**: Language-specific processing rules and configurations
- **Model Configs**: Ollama model configurations and parameters
- **MCP Config**: Tool management and external integration settings

#### **API Layer (src/api/)**
- **FastAPI Application**: RESTful API with comprehensive endpoints
- **Advanced Analytics Routes**: Machine learning and statistical analysis endpoints
- **MCP Integration**: Model Context Protocol server for external tools

#### **Testing Framework (Test/)**
- **Integration Tests**: End-to-end system testing
- **Performance Tests**: Load testing and performance validation
- **Unit Tests**: Individual component testing
- **Comprehensive Test Suite**: Automated testing with 100% success rate

#### **Infrastructure (/)**
- **Docker Configuration**: Containerization and deployment
- **Nginx Setup**: Reverse proxy and load balancing
- **Monitoring Stack**: Performance tracking and alerting
- **Kubernetes Manifests**: Container orchestration and scaling

### **Data Flow Architecture**

#### **1. Input Processing**
```
User Request → API Gateway → Orchestrator → Agent Selection → Processing
```

- **REST API**: Direct HTTP requests to FastAPI endpoints
- **MCP Tools**: External tool calls through MCP server
- **File Uploads**: PDF, images, audio, video processing
- **Web Content**: URL-based content extraction

#### **2. Content Processing Pipeline**
```
Raw Content → Content Extraction → Analysis → Knowledge Graph → Results
```

- **Text Processing**: Sentiment analysis, entity extraction, language detection
- **Media Processing**: OCR, transcription, visual analysis, audio processing
- **Knowledge Graph**: Entity relationships, graph construction, pattern recognition
- **Advanced Analytics**: Forecasting, causal analysis, anomaly detection

#### **3. Storage and Retrieval**
```
Processed Data → Vector Database → Knowledge Graph → Cache → Results
```

- **Vector Storage**: Embeddings for semantic search
- **Graph Storage**: Entity relationships and knowledge graphs
- **File Storage**: Processed content and analysis results
- **Cache Layer**: Performance optimization and quick retrieval

#### **4. Output Generation**
```
Analysis Results → Report Generation → Visualization → Export → Delivery
```

- **Reports**: Automated report generation with insights
- **Visualizations**: Interactive graphs and charts
- **API Responses**: Structured data for external consumption
- **MCP Tools**: Tool responses for external integrations

### **Integration Patterns**

#### **MCP Integration**
- **25+ Unified Tools**: Consolidated functionality for external access
- **Tool Categories**: Content processing, analysis, management, reporting
- **Protocol Support**: Standard MCP protocol for tool calling
- **Error Handling**: Robust error handling and fallback mechanisms

#### **Ollama Integration**
- **Local Model Hosting**: Self-hosted AI models for privacy and performance
- **Model Management**: Dynamic model loading and configuration
- **Fallback Support**: Multiple model options for reliability
- **Performance Optimization**: Optimized inference and caching

#### **External API Integration**
- **Market Data**: Real-time financial and market information
- **Social Media**: Social media content analysis and monitoring
- **Web Services**: External data sources and APIs
- **Third-party Tools**: Integration with specialized analysis tools

## 🆕 **Latest Updates - Phase 7.5 Advanced Analytics System Integration & Testing**

### ✨ **New Features Added:**

#### **🎯 Phase 7.5: Advanced Analytics System Integration & Testing** ✅ **COMPLETED**
- **Advanced Analytics API Integration**: 20+ advanced analytics endpoints fully operational
- **MCP Tool Extension**: Extended from 25 to 30 unified MCP tools with advanced analytics capabilities
- **Comprehensive Test Suite**: 18/18 tests passing (100% success rate) with full system validation
- **Health Monitoring**: Advanced analytics health endpoint with 19/20 components available (95% success rate)
- **Error Resolution**: Fixed all console errors, URL patterns, and import issues
- **Conditional Initialization**: Robust error handling for missing dependencies
- **Documentation**: Complete Phase 7.5 completion report and system status

#### **🔧 Advanced Analytics Components:**
- **Multivariate Forecasting Engine**: Ensemble prediction methods with confidence intervals
- **Causal Inference Engine**: Granger causality, correlation analysis, and conditional independence
- **Scenario Analysis Engine**: What-if scenario modeling and impact analysis
- **Advanced Anomaly Detection**: Isolation forest, statistical outlier detection
- **Model Optimization**: Hyperparameter tuning and automated model selection
- **Feature Engineering**: Automated feature creation and selection
- **Deep Learning Integration**: Neural networks, CNNs, LSTMs, and transformers
- **Performance Monitoring**: Real-time system performance tracking

#### **🤖 Advanced Analytics Agents:**
- **Advanced Forecasting Agent**: Multivariate time series forecasting
- **Causal Analysis Agent**: Causal relationship discovery and analysis
- **Anomaly Detection Agent**: Real-time anomaly detection and alerting
- **Advanced ML Agent**: Deep learning and machine learning automation

#### **🎯 Phase 1: Enhanced Pattern Recognition Foundation** ✅ **COMPLETED**

### ✨ **New Features Added:**

#### **🎯 Phase 1: Enhanced Pattern Recognition Foundation** ✅ **COMPLETED**
- **Temporal Pattern Recognition Engine**: Historical pattern detection and time series analysis
- **Seasonal Pattern Detector**: Cyclical behavior identification and seasonal analysis
- **Trend Analysis Engine**: Direction, strength, and pattern analysis with forecasting
- **Pattern Storage Service**: Persistent pattern database with versioning and metadata
- **Vector Pattern Engine**: Advanced vector clustering and similarity pattern detection
- **Anomaly Detection**: Statistical outlier identification and isolation forest detection
- **Pattern Classification**: Automated pattern categorization and labeling
- **Cross-Modal Matcher**: Text, audio, video pattern correlation
- **Pattern Recognition Agent**: Unified agent for comprehensive pattern analysis
- **Configuration System**: Flexible configuration management for all components

#### **🎯 Interactive Knowledge Graph Visualization**
- **Full-width interactive HTML graphs** with zoom, pan, and click functionality
- **Query-specific graph filtering** based on user input
- **Real-time graph statistics** (nodes, edges, density)
- **Working download functionality** for HTML files
- **Scrollable interface** for large graphs

#### **🔍 Enhanced Semantic Search System**
- **Vector-based semantic search** across all content types
- **Multilingual semantic search** with automatic translation
- **Conceptual search** for finding related content
- **Cross-content type search** (PDFs, web pages, audio, video)
- **Combined search** integrating semantic and knowledge graph results

#### **📊 Streamlit UI Improvements**
- **Separated text search and visual graph generation**
- **Clear button labeling** and user instructions
- **Automatic graph generation** with smart filtering
- **Full-width graph display** with scroll bars
- **Working download buttons** for all generated content

### 🚀 **Quick Start - Advanced Analytics Features:**

#### **📊 Advanced Analytics API Endpoints:**
- **Health Check**: `GET /advanced-analytics/health` - System health and component status
- **Multivariate Forecasting**: `POST /advanced-analytics/forecasting/multivariate` - Time series forecasting
- **Causal Analysis**: `POST /advanced-analytics/causal/analysis` - Causal relationship discovery
- **Anomaly Detection**: `POST /advanced-analytics/anomaly/detection` - Outlier detection
- **Model Optimization**: `POST /advanced-analytics/optimization/model` - Hyperparameter tuning
- **Feature Engineering**: `POST /advanced-analytics/features/engineering` - Automated feature creation
- **Deep Learning**: `POST /advanced-analytics/ml/deep-learning` - Neural network training
- **Clustering**: `POST /advanced-analytics/ml/clustering` - Data clustering analysis
- **Performance Monitoring**: `POST /advanced-analytics/monitoring/performance` - System performance tracking

#### **🤖 Advanced Analytics Agents:**
- **Forecasting Agent**: `POST /advanced-analytics/forecasting/agent` - Agent-based forecasting
- **Causal Agent**: `POST /advanced-analytics/causal/agent` - Agent-based causal analysis
- **Anomaly Agent**: `POST /advanced-analytics/anomaly/agent` - Agent-based anomaly detection

#### **🌐 Web Interface:**
1. **Navigate to**: `http://localhost:8501` → **"Advanced Analytics"**

2. **Forecasting**: Use multivariate forecasting for time series prediction

3. **Causal Analysis**: Discover causal relationships in your data

4. **Anomaly Detection**: Identify outliers and anomalies

5. **Model Optimization**: Automatically optimize machine learning models

#### **🔍 Knowledge Graph Features:**
1. **Navigate to**: `http://localhost:8501` → **"Semantic Search"** → **"🌐 Knowledge Graph"**

2. **Text Search**: Use "🔍 Search Knowledge Graph" for text results

3. **Visual Graphs**: Use "🎯 Generate Filtered Graph" for interactive visualizations

4. **Full Graph**: Use "📊 Generate Full Graph" for complete knowledge base visualization

## 🏗️ Design Document & Architecture Framework

### System Purpose
The Sentiment Analysis Swarm is a comprehensive AI-powered system for processing and analyzing content across multiple languages and modalities (text, audio, video, images, PDFs) using an agent swarm architecture with MCP (Model Context Protocol) framework integration.

### Core Architecture Pattern
- **Agent Swarm Architecture**: Multiple specialized agents working together
- **MCP Framework Integration**: All operations go through MCP tools
- **Multilingual Processing**: Language-specific configurations and processing
- **Microservices**: Modular, scalable component design
- **Event-Driven**: Asynchronous processing with proper error handling

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Client Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Web UI    │  │   API       │  │   MCP       │         │
│  │             │  │   Client    │  │   Client    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ FastAPI     │  │ MCP Server  │  │ Orchestrator│         │
│  │ Endpoints   │  │ Tools (30)  │  │             │         │
│  │ + Advanced  │  │ + Advanced  │  │ + Advanced  │         │
│  │   Analytics │  │   Analytics │  │   Analytics │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Agent Swarm Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Text Agent  │  │ Vision Agent│  │ Audio Agent │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ File Agent  │  │ Knowledge   │  │ Web Agent   │         │
│  │             │  │ Graph Agent │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Forecasting │  │ Causal      │  │ Anomaly     │         │
│  │ Agent       │  │ Analysis    │  │ Detection   │         │
│  │             │  │ Agent       │  │ Agent       │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Advanced ML │  │ Pattern     │  │ Predictive  │         │
│  │ Agent       │  │ Recognition │  │ Analytics   │         │
│  │             │  │ Agent       │  │ Agent       │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Core Services Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Vector DB   │  │ Translation │  │ Model       │         │
│  │ Manager     │  │ Service     │  │ Manager     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Advanced    │  │ Performance │  │ Deep        │         │
│  │ Analytics   │  │ Monitoring  │  │ Learning    │         │
│  │ Engine      │  │ System      │  │ Engine      │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Configuration Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Language    │  │ Model       │  │ System      │         │
│  │ Configs     │  │ Configs     │  │ Settings    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Core Design Principles

#### 1. MCP-First Architecture
- **All operations must go through MCP tools**
- **No direct API access to agents or services**
- **Unified interface for all functionality**
- **Consistent tool definitions and error handling**
- **Single unified MCP server with consolidated tools**
- **FastAPI + MCP integration pattern**

#### 2. Multilingual Support
- **Language-agnostic processing pipeline**
- **Language-specific configurations stored in config files**
- **Automatic language detection**
- **Cultural and linguistic context awareness**

#### 3. Agent Swarm Coordination
- **Specialized agents for specific tasks**
- **Orchestrator for request routing and coordination**
- **Load balancing and failover capabilities**
- **Agent health monitoring and recovery**

#### 4. Configuration-Driven Development
- **All language-specific parameters in config files**
- **Dynamic configuration loading**
- **Environment-specific settings**
- **Hot-reload capability for non-critical changes**

#### 5. Error Resilience
- **Graceful degradation on failures**
- **Comprehensive error handling and logging**
- **Retry mechanisms with exponential backoff**
- **Circuit breaker patterns for external dependencies**

#### 6. Performance Optimization
- **Asynchronous processing throughout**
- **Caching at multiple levels**
- **Resource pooling and connection reuse**
- **Monitoring and metrics collection**
- **Tool consolidation for reduced overhead**
- **Unified server architecture for better resource utilization**

### Technology Stack

#### Core Technologies
- **Python 3.11+**: Primary development language
- **FastAPI**: Web framework for API endpoints
- **FastMCP**: MCP framework implementation
- **Ollama**: Local LLM inference engine
- **ChromaDB**: Vector database for embeddings
- **NetworkX**: Knowledge graph management
- **PyPDF2**: PDF text extraction
- **TensorFlow/PyTorch**: Deep learning frameworks
- **Scikit-learn**: Machine learning algorithms
- **Statsmodels**: Statistical analysis and forecasting
- **Pandas/NumPy**: Data manipulation and numerical computing

#### Development Tools
- **Poetry/Virtual Environment**: Dependency management
- **Pytest**: Testing framework
- **Loguru**: Structured logging
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server

### MCP Tools Overview

#### **30 Unified MCP Tools Available:**

##### **📊 Advanced Analytics Tools (5 new):**
- **`scenario_analysis`**: Business scenario planning and what-if analysis
- **`model_optimization`**: Hyperparameter tuning and model optimization
- **`feature_engineering`**: Automated feature creation and selection
- **`deep_learning_training`**: Neural network training and deployment
- **`automl_pipeline`**: Automated machine learning pipeline

##### **🔍 Content Analysis Tools:**
- **`process_content`**: Unified content processing for all types
- **`extract_text_from_content`**: Text extraction from various formats
- **`summarize_content`**: Content summarization with configurable length
- **`translate_content`**: Multi-language translation support
- **`convert_content_format`**: Format conversion between types

##### **📈 Analytics & Intelligence Tools:**
- **`analyze_sentiment`**: Sentiment analysis with detailed metrics
- **`extract_entities`**: Named entity recognition and extraction
- **`generate_knowledge_graph`**: Knowledge graph generation from content
- **`analyze_business_intelligence`**: Business intelligence analysis
- **`create_visualizations`**: Data visualization generation

##### **🗄️ Data Management Tools:**
- **`store_in_vector_db`**: Vector database storage and indexing
- **`query_knowledge_graph`**: Knowledge graph querying and search
- **`export_data`**: Data export in multiple formats
- **`manage_data_sources`**: Data source configuration and management

##### **📋 Reporting & Monitoring Tools:**
- **`generate_report`**: Automated report generation
- **`create_dashboard`**: Interactive dashboard creation
- **`export_results`**: Results export and sharing
- **`schedule_reports`**: Automated report scheduling
- **`get_agent_status`**: Agent status monitoring
- **`start_agents`**: Agent lifecycle management
- **`stop_agents`**: Agent shutdown and cleanup

##### **⚙️ System Management Tools:**
- **`get_system_status`**: System health and status monitoring
- **`configure_system`**: System configuration management
- **`monitor_performance`**: Performance monitoring and metrics
- **`manage_configurations`**: Configuration file management

#### Infrastructure
- **Docker**: Containerization (when needed)
- **Git**: Version control
- **GitHub Actions**: CI/CD (when implemented)

### Component Architecture

#### 1. Main Entry Point (`main.py`)
- **Unified MCP server initialization and management**
- **FastAPI + MCP integration**
- **Agent swarm coordination**
- **Tool registration and routing**
- **System health monitoring**
- **Process management for Windows environment**

#### 2. Orchestrator (`src/core/orchestrator.py`)
- **Request routing to appropriate agents**
- **Load balancing and failover**
- **Result aggregation and caching**
- **Agent health monitoring**

#### 3. Agent Base Class (`src/agents/base_agent.py`)
- **Common agent functionality**
- **MCP tool integration**
- **Error handling and logging**
- **Performance monitoring**

#### 4. Language Configuration System (`src/config/language_config/`)
- **Language-specific parameter management**
- **Entity pattern definitions**
- **Processing settings**
- **Model configurations**

### Dual MCP Server Architecture

The system implements a dual MCP server architecture to support both FastAPI integration and standalone Strands integration:

1. **Unified MCP Server** (Port 8003): Integrated with FastAPI for web access
2. **Standalone MCP Server** (Port 8000): Dedicated server for Strands integration with Streamable HTTP transport

#### Server Endpoints
- **FastAPI Server**: http://localhost:8003
- **MCP Server (FastAPI Integrated)**: http://localhost:8003/mcp
- **Standalone MCP Server**: http://localhost:8000 (Strands integration)
- **API Documentation**: http://localhost:8003/docs

#### Strands Integration
For Strands integration, use the standalone MCP server:
```python
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.tools.mcp.mcp_client import MCPClient

# Create MCP client with Streamable HTTP transport
streamable_http_mcp_client = MCPClient(
    lambda: streamablehttp_client("http://localhost:8000/mcp")
)

# Create an agent with MCP tools
with streamable_http_mcp_client:
    tools = streamable_http_mcp_client.list_tools_sync()
    agent = Agent(tools=tools)
```

### MCP Implementation Patterns

#### Tool Consolidation Strategy
1. **Identify Duplicate Functionality**: Map all existing tools and identify overlapping capabilities
2. **Create Unified Interfaces**: Standardize parameter patterns and return formats across all tools
3. **Implement Content Type Detection**: Auto-detect content type for intelligent routing
4. **Maintain Backward Compatibility**: Ensure existing functionality is preserved during consolidation
5. **Achieve Target Reduction**: Aim for 70%+ reduction in tool count while maintaining all features

#### FastMCP Integration Best Practices
1. **Correct Constructor Usage**: FastMCP(name, version) - no 'description' parameter
2. **HTTP App Integration**: Use `mcp.http_app(path="/mcp")` for FastAPI integration
3. **Streamable HTTP Transport**: Use proper headers (`text/event-stream, application/json`) for Strands integration
4. **Proper Error Handling**: Implement comprehensive try-catch blocks with logging
5. **Process Management**: Use Windows-specific process termination commands

#### Strands Integration Best Practices
1. **Standalone Server**: Dedicated MCP server on port 8000 for Strands integration
2. **Streamable HTTP Transport**: Use `text/event-stream, application/json` headers
3. **Proper Client Pattern**: Use `streamablehttp_client("http://localhost:8000/mcp")` for Strands
4. **Tool Availability**: All 25 consolidated tools available through Strands integration
5. **Error Handling**: Proper handling of 400/406 errors for non-Strands clients

### Multilingual Processing Framework

#### Language Configuration Factory
```python
class LanguageConfigFactory:
    @classmethod
    def get_config(cls, language_code: str) -> BaseLanguageConfig:
        # Return language-specific configuration
    
    @classmethod
    def detect_language_from_text(cls, text: str) -> str:
        # Automatic language detection
```

#### Supported Languages
- **Chinese (zh)**: Modern and Classical Chinese
- **Russian (ru)**: Cyrillic text processing
- **English (en)**: Standard English processing
- **Japanese (ja)**: Japanese with Kanji support
- **Korean (ko)**: Korean text processing
- **Arabic (ar)**: Arabic with RTL support
- **Hindi (hi)**: Hindi text processing

#### Language-Specific Features
Each language configuration must include:
- **Entity Patterns**: Regex patterns for entity extraction
- **Processing Settings**: Language-specific parameters
- **Ollama Models**: Appropriate model configurations
- **Grammar Patterns**: Language-specific structures
- **Detection Patterns**: Language identification patterns

#### Processing Pipeline
1. **Language Detection**: Automatic detection from content
2. **Configuration Loading**: Load language-specific settings
3. **Entity Extraction**: Use language-appropriate patterns
4. **Knowledge Graph Processing**: Apply language-specific rules
5. **Result Generation**: Format results appropriately

### 🎯 **Interactive Knowledge Graph & Semantic Search System**

#### **Knowledge Graph Features**
- **Interactive HTML Visualizations**: Full-width, scrollable graphs with zoom/pan
- **Query-Specific Filtering**: Automatically filter graphs based on user queries
- **Real-time Statistics**: Live node/edge counts and graph density metrics
- **Download Functionality**: Working download buttons for all generated content
- **Multilingual Support**: Graph generation in 7 supported languages

#### **Semantic Search Capabilities**
- **Vector-Based Search**: ChromaDB-powered semantic similarity search
- **Cross-Content Search**: Search across PDFs, web pages, audio, video, and text
- **Conceptual Matching**: Find related content even without exact term matches
- **Multilingual Search**: Automatic translation and search across languages
- **Combined Results**: Integrate semantic search with knowledge graph queries

#### **Streamlit UI Components**
```
🌐 Knowledge Graph Tab:
├── 🔍 Text Search Section
│   ├── Search Query Input
│   ├── Language Selector
│   └── "Search Knowledge Graph" Button
├── 🎯 Visual Graph Section
│   ├── Graph Query Input (optional)
│   ├── "Generate Filtered Graph" Button
│   └── "Generate Full Graph" Button
└── 📊 Results Display
    ├── Graph Statistics
    ├── Interactive HTML Visualization
    └── Download Button
```

#### **API Endpoints**
- `POST /search/semantic` - Semantic search across all content
- `POST /search/knowledge-graph` - Knowledge graph text search
- `POST /search/combined` - Combined semantic + knowledge graph search
- `POST /search/generate-graph-report` - Generate interactive visualizations
- `GET /search/statistics` - Get search system statistics

### Testing Framework

#### Test Organization
```
Test/
├── unit/                    # Unit tests
├── integration/             # Integration tests
├── performance/             # Performance tests
├── multilingual/            # Multilingual-specific tests
└── semantic_search/         # Semantic search tests
```

#### Test Categories
1. **Configuration Tests**: Language config validation
2. **Component Tests**: Individual component testing
3. **Integration Tests**: End-to-end workflow testing
4. **MCP Tests**: MCP tool functionality testing
5. **Multilingual Tests**: Language-specific processing testing
6. **Semantic Search Tests**: Vector search and graph generation testing

#### Test Execution Pattern
```python
# Use .venv/Scripts/python.exe for all test execution
.venv/Scripts/python.exe Test/test_name.py

# Test results should be stored in Results/ directory
# Test reports should include performance metrics

# MCP-specific testing
# Test unified MCP server functionality
.venv/Scripts/python.exe Test/test_unified_mcp_server.py

# Test FastAPI + MCP integration
curl http://localhost:8003/mcp

# Test individual MCP tools
.venv/Scripts/python.exe Test/test_mcp_tools.py
```

#### Test Requirements
- [ ] All components must have unit tests
- [ ] Integration tests for all workflows
- [ ] Multilingual processing tests
- [ ] MCP tool functionality tests (25 unified tools)
- [ ] Performance benchmarks
- [ ] Error handling validation
- [ ] FastAPI + MCP integration tests
- [ ] Unified MCP server functionality tests
- [ ] Windows process management tests
- [ ] Asyncio thread conflict resolution tests

### File Organization Standards

#### Directory Structure
```
project_root/
├── main.py                  # Main entry point
├── src/                     # Source code
│   ├── agents/             # Agent implementations
│   ├── api/                # API endpoints
│   ├── core/               # Core services
│   ├── config/             # Configuration files
│   └── mcp_servers/        # MCP server implementations
├── Test/                   # Test scripts
├── Results/                # Test results and reports
├── data/                   # Test data and samples
├── .venv/                  # Virtual environment
└── docs/                   # Documentation
```

#### Naming Conventions
- **Files**: snake_case.py
- **Classes**: PascalCase
- **Functions**: snake_case
- **Constants**: UPPER_SNAKE_CASE
- **Variables**: snake_case

#### Import Organization
```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
from fastapi import FastAPI
from loguru import logger

# Local imports
from src.core.models import AnalysisRequest
from src.agents.base_agent import StrandsBaseAgent
```

### Coding Standards

#### Python Style Guide
- **PEP 8**: Primary style guide
- **Type Hints**: Required for all function signatures
- **Docstrings**: Required for all classes and functions
- **Error Handling**: Comprehensive exception handling

#### Code Quality Requirements
```python
# Required imports for all files
from typing import Dict, List, Optional, Any
from loguru import logger

# Required error handling pattern
@with_error_handling("component_name")
async def function_name(param: str) -> Dict[str, Any]:
    """Function documentation."""
    try:
        # Implementation
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Error in function_name: {e}")
        return {"success": False, "error": str(e)}
```

#### Async/Await Usage
- **All I/O operations must be async**
- **Use asyncio.gather() for parallel operations**
- **Proper exception handling in async contexts**
- **Avoid blocking operations in async functions**

#### Logging Standards
```python
# Use structured logging with loguru
logger.info("Operation started", extra={"component": "agent_name"})
logger.error("Operation failed", extra={"error": str(e), "component": "agent_name"})
logger.debug("Debug information", extra={"data": debug_data})
```

### Integration Patterns

#### Agent Integration Pattern
```python
class SpecializedAgent(StrandsBaseAgent):
    def __init__(self):
        super().__init__()
        self.agent_id = f"{self.__class__.__name__}_{self._generate_id()}"
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        # Validate request
        # Process with language-specific configuration
        # Return standardized result
```

#### MCP Tool Integration Pattern (Unified Interface)
```python
@self.mcp.tool(description="Unified tool description")
async def tool_name(
    content: str,
    content_type: str = "auto",
    language: str = "en",
    options: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Unified tool implementation with consistent interface."""
    try:
        # Auto-detect content type if not specified
        if content_type == "auto":
            content_type = self._detect_content_type(content)
        
        # Route to appropriate agent based on content type
        if content_type in ["text", "pdf"]:
            result = await self.text_agent.process_content(content, language, options)
        elif content_type in ["audio", "video"]:
            result = await self.audio_agent.process_content(content, language, options)
        elif content_type in ["image", "vision"]:
            result = await self.vision_agent.process_content(content, language, options)
        else:
            result = await self.text_agent.process_content(content, language, options)
        
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Error in tool_name: {e}")
        return {"success": False, "error": str(e)}
```

#### **New Semantic Search MCP Tools**
```python
# Semantic Search Tools
@self.mcp.tool(description="Perform semantic search across all content types")
async def semantic_search(
    query: str,
    language: str = "en",
    n_results: int = 10,
    similarity_threshold: float = 0.7
) -> Dict[str, Any]:
    """Semantic search using vector embeddings."""
    return await self.semantic_search_service.search(query, language, n_results, similarity_threshold)

@self.mcp.tool(description="Multilingual semantic search with automatic translation")
async def multilingual_semantic_search(
    query: str,
    source_language: str = "en",
    target_languages: List[str] = None,
    n_results: int = 10
) -> Dict[str, Any]:
    """Search across multiple languages."""
    return await self.semantic_search_service.multi_language_search(query, source_language, target_languages, n_results)

@self.mcp.tool(description="Conceptual search for finding related content")
async def conceptual_search(
    concept: str,
    language: str = "en",
    n_results: int = 10
) -> Dict[str, Any]:
    """Search for conceptually related content."""
    return await self.semantic_search_service.search_by_concept(concept, language, n_results)

@self.mcp.tool(description="Cross-content type semantic search")
async def cross_content_search(
    query: str,
    content_types: List[str] = None,
    language: str = "en",
    n_results: int = 10
) -> Dict[str, Any]:
    """Search across specific content types."""
    return await self.semantic_search_service.search_across_content_types(query, content_types, language, n_results)

@self.mcp.tool(description="Combined semantic and knowledge graph search")
async def combined_search(
    query: str,
    language: str = "en",
    n_results: int = 10
) -> Dict[str, Any]:
    """Combine semantic search with knowledge graph results."""
    return await self.semantic_search_service.search_with_knowledge_graph(query, language, n_results)
```

#### Configuration Integration Pattern
```python
# Always use factory pattern for configuration
config = LanguageConfigFactory.get_config(language_code)

# Access configuration through methods
patterns = config.get_entity_patterns()
settings = config.get_processing_settings()
```

## MCP Tools Consolidation

This project has been successfully consolidated to use a unified MCP server with 30 tools:

### Tool Categories:
1. **Content Processing** (5 tools): process_content, extract_text_from_content, summarize_content, translate_content, convert_content_format
2. **Analysis & Intelligence** (5 tools): analyze_sentiment, extract_entities, generate_knowledge_graph, analyze_business_intelligence, create_visualizations
3. **Semantic Search** (5 tools): semantic_search, multilingual_semantic_search, conceptual_search, cross_content_search, combined_search
4. **Agent Management** (3 tools): get_agent_status, start_agents, stop_agents
5. **Data Management** (4 tools): store_in_vector_db, query_knowledge_graph, export_data, manage_data_sources
6. **Reporting & Export** (4 tools): generate_report, create_dashboard, export_results, schedule_reports
7. **System Management** (4 tools): get_system_status, configure_system, monitor_performance, manage_configurations

### Benefits:
- **65% reduction** in tool count (from 85+ to 30 tools)
- **Unified interface** for all tools
- **Consistent error handling** and logging
- **Improved performance** and maintainability
- **Single MCP server** instance

### Access:
- **API Server**: http://localhost:8003
- **MCP Endpoint**: http://localhost:8003/mcp
- **API Documentation**: http://localhost:8003/docs

## 🚀 Key Features

- **Unified Agents**: Three comprehensive agents replace 10+ specialized implementations
- **Consolidated MCP Server**: Optimized from 44 to 4 category servers (PDF, Audio, Video, Website)
- **Optimized MCP Server**: Streamlined tool interface with unified access
- **Multi-Modal Analysis**: Text, audio, video, image, webpage, and PDF sentiment analysis
- **Configurable Processing Modes**: Simple, Strands, and Swarm modes for text processing
- **Advanced Audio Processing**: Transcription, summarization, and large file support
- **Comprehensive Vision Analysis**: OCR, object detection, and scene understanding
- **YouTube Integration**: Comprehensive video analysis with parallel processing
- **Interactive Knowledge Graph**: Full-width HTML visualizations with zoom/pan and query-specific filtering
- **Semantic Search System**: Vector-based search across all content types with multilingual support
- **Knowledge Graph**: Entity extraction and relationship mapping with enhanced categorization
- **Translation Services**: Multi-language content translation and analysis
- **PDF Processing**: Advanced PDF text extraction with PyPDF2 and vision OCR
- **Enhanced Ollama Integration**: Optimized local LLM processing with configurable models
- **Large File Processing**: Advanced chunking and progressive processing
- **Vector Database**: ChromaDB for efficient storage and retrieval
- **Local Deployment**: Run everything locally with CPU optimization
- **Interactive Visualizations**: D3.js-based knowledge graph visualization with zoom and pan
- **Production Ready**: Docker support, monitoring, logging, and security features

## 📁 Project Structure

```
Sentiment/
├── src/                    # Main source code
│   ├── agents/            # Unified agents implementation
│   ├── api/               # API endpoints and FastAPI app
│   ├── config/            # Configuration management
│   ├── core/              # Core services and utilities
│   └── mcp/               # MCP server implementation
├── Test/                  # All test files organized by functionality
├── docs/                  # Comprehensive documentation
├── Results/               # Analysis results and reports
│   └── reports/          # Generated report files
├── scripts/              # Utility scripts and automation
├── data/                 # Data files and extracted content
├── examples/             # Example usage and demos
├── cache/                # Cache and temporary files
├── models/               # Model files and configurations
├── ui/                   # User interface components
├── monitoring/           # Monitoring and logging
├── k8s/                  # Kubernetes configurations
├── nginx/                # Web server configurations
└── main.py              # Main application entry point
```

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

### 🔧 Consolidated MCP Server Architecture

The system features a highly optimized MCP (Model Context Protocol) server architecture that consolidates 44 individual servers into 4 category-specific servers:

- **PDF Processing Server**: Handles all PDF-related operations including text extraction, image conversion, summarization, translation, vector storage, and knowledge graph creation
- **Audio Processing Server**: Manages audio transcription, spectrogram generation, summarization, translation, vector storage, and knowledge graph creation
- **Video Processing Server**: Processes video content with OCR, frame extraction, summarization, translation, vector storage, and knowledge graph creation
- **Website Processing Server**: Handles web scraping, screenshot generation, summarization, translation, vector storage, and knowledge graph creation

Each server provides 6 core functions:
1. **Text Extraction**: Extract text from content (OCR for images/videos, transcription for audio)
2. **Content Conversion**: Convert content to images (PDF pages, video frames, spectrograms, screenshots)
3. **Summarization**: Generate comprehensive summaries of content
4. **Translation**: Translate content from foreign languages to English
5. **Vector Storage**: Store processed content in ChromaDB for semantic search
6. **Knowledge Graph**: Create structured knowledge graphs from content

**Benefits:**
- **90.9% Server Reduction**: From 44 to 4 servers
- **Unified Interface**: Consistent API across all content types
- **Resource Optimization**: Reduced memory and CPU usage
- **Simplified Configuration**: Centralized configuration management
- **Enhanced Scalability**: Better resource allocation and load balancing

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
- **Codebase Organization**: Cleanup and organization guidelines

### Documentation Structure
```
docs/
├── Implementation Guides/     # Feature implementation details
├── Configuration Guides/      # Setup and configuration
├── Production Guides/         # Deployment and operations
├── Implementation Summaries/  # Technical implementation notes
└── CODEBASE_CLEANUP_AND_ORGANIZATION.md  # Organization guidelines
```

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

### Test Organization
The `/Test/` directory contains all test files organized by functionality:
- **Core Functionality Tests**: Main system integration and agent tests
- **Feature Tests**: YouTube, vision, audio, and translation analysis tests
- **Configuration Tests**: Model configuration and settings tests
- **Integration Tests**: End-to-end integration and verification tests

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

### Compliance Checklist

#### Before Any Implementation
- [ ] Review this design framework
- [ ] Ensure MCP compliance (unified server architecture)
- [ ] Follow multilingual patterns
- [ ] Implement proper error handling
- [ ] Add comprehensive tests
- [ ] Update documentation
- [ ] Verify FastMCP integration patterns
- [ ] Test Windows process management
- [ ] Validate asyncio thread handling

#### Before Deployment
- [ ] All tests passing
- [ ] Performance benchmarks met
- [ ] Security review completed
- [ ] Documentation updated
- [ ] Configuration validated
- [ ] Monitoring in place
- [ ] MCP server integration verified
- [ ] 25 unified tools functionality confirmed
- [ ] FastAPI + MCP integration tested
- [ ] Windows process management validated

## 🔄 Recent Updates

### **🎯 Phase 7.5: Advanced Analytics System Integration & Testing** ✅ **COMPLETED**
- **Advanced Analytics API Integration**: 20+ advanced analytics endpoints fully operational
- **MCP Tool Extension**: Extended from 25 to 30 unified MCP tools with advanced analytics capabilities
- **Comprehensive Test Suite**: 18/18 tests passing (100% success rate) with full system validation
- **Health Monitoring**: Advanced analytics health endpoint with 19/20 components available (95% success rate)
- **Error Resolution**: Fixed all console errors, URL patterns, and import issues
- **Conditional Initialization**: Robust error handling for missing dependencies
- **Documentation**: Complete Phase 7.5 completion report and system status

### **🔧 Advanced Analytics Components:**
- **Multivariate Forecasting Engine**: Ensemble prediction methods with confidence intervals
- **Causal Inference Engine**: Granger causality, correlation analysis, and conditional independence
- **Scenario Analysis Engine**: What-if scenario modeling and impact analysis
- **Advanced Anomaly Detection**: Isolation forest, statistical outlier detection
- **Model Optimization**: Hyperparameter tuning and automated model selection
- **Feature Engineering**: Automated feature creation and selection
- **Deep Learning Integration**: Neural networks, CNNs, LSTMs, and transformers
- **Performance Monitoring**: Real-time system performance tracking

### **🤖 Advanced Analytics Agents:**
- **Advanced Forecasting Agent**: Multivariate time series forecasting
- **Causal Analysis Agent**: Causal relationship discovery and analysis
- **Anomaly Detection Agent**: Real-time anomaly detection and alerting
- **Advanced ML Agent**: Deep learning and machine learning automation

### **📊 Previous Major Updates:**
- **Design Framework Integration**: Added comprehensive Design Document and Architecture Framework to README
- **Dual MCP Server Architecture**: Implemented both unified and standalone servers for maximum compatibility
- **Strands Integration**: Complete support for Strands integration with Streamable HTTP transport
- **Codebase Cleanup**: Complete reorganization of file structure for improved maintainability
- **Enhanced Knowledge Graph Integration**: Fully integrated enhanced knowledge graph functionality
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
