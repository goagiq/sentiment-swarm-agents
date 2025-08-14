# Project Design Framework
## Sentiment Analysis Swarm with Multilingual MCP Integration

### Version: 1.6
### Last Updated: 2025-08-14
### Status: Active (Dual MCP Server Architecture with Fixed MCP Integration)

### Recent Updates (v1.6)
- **Multilingual Configuration Standards**: Enhanced multilingual support with configuration-driven regex parsing and language-specific parameters
- **MCP Tool Integration Requirements**: All functionality must be integrated into MCP tools with proper testing and verification
- **Script Execution Standards**: All scripts must use `.venv/Scripts/python.exe` for execution
- **Comprehensive File Integration**: Changes must touch all related files with proper testing before reporting success
- **MCP Framework Fix**: Fixed critical `store_document` method error in standalone MCP server
- **Vector Database Integration**: Corrected method calls to use `store_content` instead of non-existent `store_document`
- **Website Content Download Framework**: Comprehensive guidelines for adding new website sources
- **Error Handling Standards**: Enhanced error handling for MCP server and vector database operations
- **Integration Patterns**: Standardized patterns for new website content processing
- **Testing Framework**: Added specific testing requirements for website content integration
- **Performance Guidelines**: Updated performance considerations for content processing
- **System Restart Framework**: Comprehensive restart mechanism with process management and verification

### Recent Updates (v1.4)
- **Open Library Integration**: Enhanced MCP server with Open Library content download and processing capabilities
- **Enhanced Process Content Tool**: Extended `process_content` tool with Open Library URL detection and processing
- **Generic Content Downloader**: Created reusable script for downloading any Open Library book
- **Vector Database Integration**: Automatic storage of downloaded content in ChromaDB
- **Knowledge Graph Generation**: Automatic creation of knowledge graphs from Open Library content
- **Metadata Extraction**: Enhanced metadata extraction from Open Library pages
- **Content Type Detection**: Improved content type detection including Open Library URLs
- **Enhanced MCP Server**: Extended `UnifiedMCPServer` with `EnhancedUnifiedMCPServer` for Open Library support
- **Dual MCP Server Architecture**: Implemented both unified and standalone servers for maximum compatibility
- **Strands Integration**: Complete support for Strands integration with Streamable HTTP transport
- **Standalone MCP Server**: Dedicated server on port 8000 for Strands integration
- **FastAPI MCP Integration**: Unified server on port 8003 for web access
- **Streamable HTTP Transport**: Proper configuration for Strands MCP client integration
- **25 Consolidated Tools**: Complete list of all available MCP tools with descriptions
- **Client Implementation Patterns**: Both FastMCP and Strands MCP client patterns
- **Server Endpoints**: Clear documentation of all available server endpoints

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Design Principles](#core-design-principles)
3. [Technology Stack](#technology-stack)
4. [Component Architecture](#component-architecture)
5. [MCP Framework Integration](#mcp-framework-integration)
6. [Open Library Integration Framework](#open-library-integration-framework)
7. [Multilingual Processing Framework](#multilingual-processing-framework)
8. [Configuration Management](#configuration-management)
9. [Testing Framework](#testing-framework)
10. [File Organization Standards](#file-organization-standards)
11. [Coding Standards](#coding-standards)
12. [Integration Patterns](#integration-patterns)
13. [Error Handling Standards](#error-handling-standards)
14. [Performance Guidelines](#performance-guidelines)
15. [Security Considerations](#security-considerations)
16. [Deployment Standards](#deployment-standards)
17. [Documentation Standards](#documentation-standards)

---

## Architecture Overview

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
│  │ Endpoints   │  │ Tools       │  │             │         │
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
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Core Services Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Vector DB   │  │ Translation │  │ Model       │         │
│  │ Manager     │  │ Service     │  │ Manager     │         │
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

---

## Core Design Principles

### 1. MCP-First Architecture
- **All operations must go through MCP tools**
- **No direct API access to agents or services**
- **Unified interface for all functionality**
- **Consistent tool definitions and error handling**
- **Single unified MCP server with consolidated tools**
- **FastAPI + MCP integration pattern**

### 2. Multilingual Support
- **Language-agnostic processing pipeline**
- **Language-specific configurations stored in config files**
- **Automatic language detection**
- **Cultural and linguistic context awareness**

### 3. Agent Swarm Coordination
- **Specialized agents for specific tasks**
- **Orchestrator for request routing and coordination**
- **Load balancing and failover capabilities**
- **Agent health monitoring and recovery**

### 4. Configuration-Driven Development
- **All language-specific parameters in config files**
- **Dynamic configuration loading**
- **Environment-specific settings**
- **Hot-reload capability for non-critical changes**

### 5. Error Resilience
- **Graceful degradation on failures**
- **Comprehensive error handling and logging**
- **Retry mechanisms with exponential backoff**
- **Circuit breaker patterns for external dependencies**

### 6. Performance Optimization
- **Asynchronous processing throughout**
- **Caching at multiple levels**
- **Resource pooling and connection reuse**
- **Monitoring and metrics collection**
- **Tool consolidation for reduced overhead**
- **Unified server architecture for better resource utilization**

---

## Technology Stack

### Core Technologies
- **Python 3.11+**: Primary development language
- **FastAPI**: Web framework for API endpoints
- **FastMCP**: MCP framework implementation
- **Ollama**: Local LLM inference engine
- **ChromaDB**: Vector database for embeddings
- **NetworkX**: Knowledge graph management
- **PyPDF2**: PDF text extraction

### Development Tools
- **Poetry/Virtual Environment**: Dependency management
- **Pytest**: Testing framework
- **Loguru**: Structured logging
- **Pydantic**: Data validation and serialization
- **Uvicorn**: ASGI server

### Infrastructure
- **Docker**: Containerization (when needed)
- **Git**: Version control
- **GitHub Actions**: CI/CD (when implemented)

---

## Component Architecture

### 1. Main Entry Point (`main.py`)
```python
# Responsibilities:
# - Unified MCP server initialization and management
# - FastAPI + MCP integration
# - Agent swarm coordination
# - Tool registration and routing
# - System health monitoring
# - Process management for Windows environment

# Updated pattern using unified MCP server
from src.mcp_servers.unified_mcp_server import create_unified_mcp_server

def start_mcp_server():
    """Start the unified MCP server with FastAPI integration."""
    try:
        # Create the unified MCP server
        mcp_server = create_unified_mcp_server()
        
        if mcp_server.mcp is None:
            print("⚠️ MCP server not available - skipping MCP server startup")
            return None
        
        # Integrate with FastAPI
        if mcp_server:
            try:
                mcp_app = mcp_server.get_http_app(path="/mcp")
                if mcp_app:
                    from src.api.main import app
                    app.mount("/mcp", mcp_app)
                    print("✅ MCP server integrated with FastAPI at /mcp")
            except Exception as e:
                print(f"⚠️ Warning: Could not integrate MCP server: {e}")
        
        print("✅ Unified MCP server started successfully")
        print(" - MCP Server: http://localhost:8003/mcp")
        print(" - Available tools: 25 consolidated tools")
        return mcp_server
        
    except Exception as e:
        print(f"❌ Error starting MCP server: {e}")
        return None
```

### 2. Orchestrator (`src/core/orchestrator.py`)
```python
# Responsibilities:
# - Request routing to appropriate agents
# - Load balancing and failover
# - Result aggregation and caching
# - Agent health monitoring

class SentimentOrchestrator:
    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        # Route request to appropriate agent
        # Handle caching and error recovery
        # Aggregate results from multiple agents if needed
```

### 3. Agent Base Class (`src/agents/base_agent.py`)
```python
# Responsibilities:
# - Common agent functionality
# - MCP tool integration
# - Error handling and logging
# - Performance monitoring

class StrandsBaseAgent:
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        # Standardized processing pipeline
        # Error handling and recovery
        # Result formatting and validation
```

### 4. Language Configuration System (`src/config/language_config/`)
```python
# Responsibilities:
# - Language-specific parameter management
# - Entity pattern definitions
# - Processing settings
# - Model configurations

class BaseLanguageConfig(ABC):
    @abstractmethod
    def get_entity_patterns(self) -> EntityPatterns: pass
    
    @abstractmethod
    def get_processing_settings(self) -> ProcessingSettings: pass
```

---

## MCP Framework Integration

### Dual MCP Server Architecture

The system implements a dual MCP server architecture to support both FastAPI integration and standalone Strands integration:

1. **Unified MCP Server** (Port 8003): Integrated with FastAPI for web access
2. **Standalone MCP Server** (Port 8000): Dedicated server for Strands integration with Streamable HTTP transport

### MCP Consolidation Best Practices

#### Tool Consolidation Strategy
1. **Identify Duplicate Functionality**: Map all existing tools and identify overlapping capabilities
2. **Create Unified Interfaces**: Standardize parameter patterns and return formats across all tools
3. **Implement Content Type Detection**: Auto-detect content type for intelligent routing
4. **Maintain Backward Compatibility**: Ensure existing functionality is preserved during consolidation
5. **Achieve Target Reduction**: Aim for 70%+ reduction in tool count while maintaining all features

#### Unified MCP Server Architecture
```python
# Best Practice: Single Unified Server
class UnifiedMCPServer:
    """Unified MCP server providing consolidated access to all system functionality."""
    
    def __init__(self):
        # Initialize all core services and agents
        self._initialize_core_services()
        self._initialize_agents()
        self._initialize_mcp()
        self._register_tools()
    
    def _register_tools(self):
        """Register all 25 consolidated tools with unified interface pattern."""
        # Content Processing Tools (5)
        # Analysis & Intelligence Tools (5)
        # Agent Management Tools (3)
        # Data Management Tools (4)
        # Reporting & Export Tools (4)
        # System Management Tools (4)
```

#### Standalone MCP Server Architecture
```python
# Best Practice: Standalone Server for Strands Integration
class StandaloneMCPServer:
    """Standalone MCP server for Strands integration on port 8000."""
    
    def __init__(self):
        # Initialize all core services and agents
        self._initialize_core_services()
        self._initialize_agents()
        self._initialize_mcp()
        self._register_tools()
    
    def start(self, host: str = "localhost", port: int = 8000):
        """Start the standalone MCP server with Streamable HTTP transport."""
        if not self.mcp:
            return
        
        try:
            # Use HTTP app method for proper server startup
            http_app = self.mcp.http_app(path="/mcp")
            if http_app:
                import uvicorn
                uvicorn.run(http_app, host=host, port=port, log_level="info")
        except Exception as e:
            logger.error(f"Error running MCP server: {e}")
```

#### FastMCP Integration Best Practices
1. **Correct Constructor Usage**: FastMCP(name, version) - no 'description' parameter
2. **HTTP App Integration**: Use `mcp.http_app(path="/mcp")` for FastAPI integration
3. **Streamable HTTP Transport**: Use proper headers (`text/event-stream, application/json`) for Strands integration
4. **Proper Error Handling**: Implement comprehensive try-catch blocks with logging
5. **Process Management**: Use Windows-specific process termination commands

#### Dual Server Integration Pattern
```python
# Best Practice: Dual Server Setup
def start_mcp_servers():
    """Start both MCP servers for maximum compatibility."""
    
    # 1. Create unified MCP server for FastAPI integration
    mcp_server = create_unified_mcp_server()
    
    # 2. Start standalone MCP server for Strands integration
    standalone_mcp_server = None
    try:
        standalone_mcp_server = start_standalone_mcp_server(host="localhost", port=8000)
        print("✅ Standalone MCP server started on port 8000")
        print("🔧 Available for Strands integration with Streamable HTTP transport")
    except Exception as e:
        print(f"⚠️ Warning: Could not start standalone MCP server: {e}")
    
    # 3. Integrate unified MCP server with FastAPI
    if mcp_server:
        try:
            mcp_app = mcp_server.get_http_app(path="/mcp")
            if mcp_app:
                from src.api.main import app
                app.mount("/mcp", mcp_app)
                print("✅ MCP server integrated with FastAPI at /mcp")
        except Exception as e:
            print(f"⚠️ Warning: Could not integrate MCP server: {e}")
    
    return mcp_server, standalone_mcp_server
```

### MCP Consolidation Lessons Learned

#### Key Insights from 85+ to 25 Tools Consolidation
1. **Tool Duplication Elimination**: Identified and removed overlapping functionality across multiple servers
2. **Unified Interface Pattern**: Standardized all tools with consistent parameter and return formats
3. **Dual Server Architecture**: Implemented both unified and standalone servers for maximum compatibility
4. **70% Reduction Achievement**: Successfully reduced tool count while maintaining all functionality
5. **Error Pattern Standardization**: Implemented consistent error handling across all consolidated tools

#### Critical FastMCP Integration Fixes
1. **Asyncio Thread Conflicts**: Resolved "Already running asyncio in this thread" errors
2. **FastMCP Parameter Issues**: Fixed FastMCP.__init__() unexpected keyword argument errors
3. **Dual Server Integration Pattern**: Successfully implemented both FastAPI integration and standalone Strands server
4. **Streamable HTTP Transport**: Properly configured for Strands integration with correct headers
5. **Process Management**: Implemented proper Windows process termination and restart procedures

#### Strands Integration Best Practices
1. **Standalone Server**: Dedicated MCP server on port 8000 for Strands integration
2. **Streamable HTTP Transport**: Use `text/event-stream, application/json` headers
3. **Proper Client Pattern**: Use `streamablehttp_client("http://localhost:8000/mcp")` for Strands
4. **Tool Availability**: All 25 consolidated tools available through Strands integration
5. **Error Handling**: Proper handling of 400/406 errors for non-Strands clients

### MCP Libraries and Dependencies
```python
# Required MCP libraries (from requirements.prod.txt)
fastmcp==0.1.0          # FastMCP server implementation
mcp==1.0.0              # Core MCP protocol
```

### Unified MCP Server Implementation Pattern
```python
# src/mcp_servers/unified_mcp_server.py - Consolidated MCP Server
from fastmcp import FastMCP
from loguru import logger

class UnifiedMCPServer:
    """Unified MCP server providing consolidated access to all system functionality."""
    
    def __init__(self, config: Optional[ConsolidatedMCPServerConfig] = None):
        """Initialize the unified MCP server."""
        self.config = config or ConsolidatedMCPServerConfig()
        self.mcp = None
        
        # Initialize core services
        self.model_manager = ModelManager()
        self.vector_store = VectorDBManager()
        self.knowledge_graph = ImprovedKnowledgeGraphUtility()
        self.translation_service = TranslationService()
        
        # Initialize agents
        self.text_agent = UnifiedTextAgent(use_strands=True, use_swarm=True)
        self.vision_agent = UnifiedVisionAgent()
        self.audio_agent = UnifiedAudioAgent()
        self.file_agent = EnhancedFileExtractionAgent()
        self.kg_agent = KnowledgeGraphAgent()
        self.web_agent = EnhancedWebAgent()
        
        # Initialize MCP server
        self._initialize_mcp()
        
        # Register tools
        self._register_tools()
        
        logger.info("✅ Unified MCP Server initialized successfully")
    
    def _initialize_mcp(self):
        """Initialize the MCP server."""
        if not MCP_AVAILABLE:
            logger.warning("Using mock MCP server - FastMCP not available")
            return
        
        try:
            # CRITICAL: FastMCP doesn't accept 'description' parameter
            self.mcp = FastMCP(
                name="unified_sentiment_mcp_server",
                version="1.0.0"
            )
            logger.info("✅ MCP server initialized")
        except Exception as e:
            logger.error(f"❌ Error initializing MCP server: {e}")
            self.mcp = None
    
    def _register_tools(self):
        """Register all 25 consolidated tools with unified interface pattern."""
        if not self.mcp:
            logger.warning("MCP server not available - skipping tool registration")
            return
        
        # Content Processing Tools (5)
        @self.mcp.tool(description="Unified content processing for all types")
        async def process_content(
            content: str,
            content_type: str = "auto",
            language: str = "en",
            options: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Process any type of content with unified interface."""
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
                logger.error(f"Error processing content: {e}")
                return {"success": False, "error": str(e)}
        
        # Analysis & Intelligence Tools (5)
        @self.mcp.tool(description="Sentiment analysis with multilingual support")
        async def analyze_sentiment(
            content: str,
            language: str = "en",
            detailed: bool = True
        ) -> Dict[str, Any]:
            """Analyze sentiment with multilingual support."""
            try:
                result = await self.text_agent.analyze_sentiment(content, language, detailed)
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error analyzing sentiment: {e}")
                return {"success": False, "error": str(e)}
        
        # ... Additional 23 tools following same pattern
        
        logger.info("✅ Registered 25 unified MCP tools")
```

### Client Implementation Pattern

#### FastMCP Client (FastAPI Integration)
```python
# src/mcp_servers/client_example.py - MCP Client Setup
import asyncio
from typing import Dict, Any

# Option 1: Using FastMCP Client (FastAPI Integration)
try:
    from fastmcp import FastMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

class SentimentMCPClient:
    def __init__(self, server_url: str = "http://localhost:8003/mcp/"):
        self.server_url = server_url
        self.mcp_client = None
    
    async def connect(self):
        """Connect to MCP server using FastMCP."""
        if not MCP_AVAILABLE:
            return False
        
        try:
            # Create FastMCP client
            self.mcp_client = FastMCPClient(self.server_url)
            await self.mcp_client.connect()
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]):
        """Call MCP tool with parameters."""
        if not self.mcp_client:
            return {"error": "Not connected"}
        
        try:
            result = await self.mcp_client.call_tool(tool_name, parameters)
            return result
        except Exception as e:
            return {"error": str(e)}
```

#### Strands MCP Client (Standalone Server)
```python
# Option 2: Using Streamable HTTP Transport (Strands Integration)
try:
    from mcp.client.streamable_http import streamablehttp_client
    from strands import Agent
    from strands.tools.mcp.mcp_client import MCPClient
    STRANDS_MCP_AVAILABLE = True
except ImportError:
    STRANDS_MCP_AVAILABLE = False

class StrandsMCPClient:
    def __init__(self, server_url: str = "http://localhost:8000/mcp"):
        self.server_url = server_url
        self.mcp_client = None
    
    def connect_and_create_agent(self):
        """Connect using Strands MCP client and create agent."""
        if not STRANDS_MCP_AVAILABLE:
            return None
        
        try:
            # Create MCP client with Streamable HTTP transport
            streamable_http_mcp_client = MCPClient(
                lambda: streamablehttp_client(self.server_url)
            )
            
            # Create an agent with MCP tools
            with streamable_http_mcp_client:
                # Get the tools from the MCP server
                tools = streamable_http_mcp_client.list_tools_sync()
                
                # Create an agent with these tools
                agent = Agent(tools=tools)
                
                return agent
        except Exception as e:
            print(f"Failed to connect: {e}")
            return None

# Usage Pattern for Strands Integration
def use_strands_integration():
    """Example usage of Strands MCP integration."""
    client = StrandsMCPClient("http://localhost:8000/mcp")
    agent = client.connect_and_create_agent()
    
    if agent:
        # Now you can use the agent with all 25 MCP tools!
        print("✅ Strands MCP integration successful")
        return agent
    else:
        print("❌ Failed to connect to Strands MCP server")
        return None
```

### API Integration Pattern
```python
# src/api/main.py - FastAPI with MCP Integration
from fastapi import FastAPI, HTTPException
from mcp_servers.client_example import MCPClient

app = FastAPI()

@app.post("/process/multilingual-pdf")
async def process_multilingual_pdf(
    pdf_path: str,
    language: str = "auto",
    generate_report: bool = True,
    output_path: str = None
):
    """Process multilingual PDF using MCP tool."""
    try:
        # Use MCP client to call MCP tools
        mcp_client = MCPClient()
        result = await mcp_client.call_tool(
            "process_multilingual_pdf_mcp",
            {
                "pdf_path": pdf_path,
                "language": language,
                "generate_report": generate_report,
                "output_path": output_path
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Processing failed")
            )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Required MCP Tools (25 Consolidated Tools)

#### Content Processing Tools (5)
1. **`process_content`**: Unified content processing for all types
2. **`extract_text_from_content`**: Extract text from various content types
3. **`summarize_content`**: Summarize content of any type
4. **`translate_content`**: Translate content to different languages
5. **`convert_content_format`**: Convert content between different formats

#### Analysis & Intelligence Tools (5)
6. **`analyze_sentiment`**: Analyze sentiment of text content
7. **`extract_entities`**: Extract entities from text content
8. **`generate_knowledge_graph`**: Generate knowledge graph from content
9. **`analyze_business_intelligence`**: Analyze business intelligence from content
10. **`create_visualizations`**: Create visualizations from data

#### Agent Management Tools (3)
11. **`get_agent_status`**: Get status of all agents
12. **`start_agents`**: Start specific agents
13. **`stop_agents`**: Stop specific agents

#### Data Management Tools (4)
14. **`store_in_vector_db`**: Store content in vector database
15. **`query_knowledge_graph`**: Query knowledge graph
16. **`export_data`**: Export data in various formats
17. **`manage_data_sources`**: Manage data sources

#### Reporting & Export Tools (4)
18. **`generate_report`**: Generate comprehensive reports
19. **`create_dashboard`**: Create interactive dashboards
20. **`export_results`**: Export results in various formats
21. **`schedule_reports`**: Schedule automated reports

#### System Management Tools (4)
22. **`get_system_status`**: Get system status and health
23. **`configure_system`**: System configuration management
24. **`monitor_performance`**: Performance monitoring
25. **`manage_configurations`**: Configuration management

### MCP Server Integration Pattern (Dual Server Setup)
```python
# main.py - Dual Server Startup with FastAPI + Standalone MCP
if __name__ == "__main__":
    import uvicorn
    import threading
    from src.mcp_servers.unified_mcp_server import create_unified_mcp_server
    from src.mcp_servers.standalone_mcp_server import start_standalone_mcp_server
    
    # 1. Create unified MCP server for FastAPI integration
    mcp_server = create_unified_mcp_server()
    
    # 2. Start standalone MCP server for Strands integration
    standalone_mcp_server = None
    try:
        standalone_mcp_server = start_standalone_mcp_server(host="localhost", port=8000)
        print("✅ Standalone MCP server started on port 8000")
        print("🔧 Available for Strands integration with Streamable HTTP transport")
    except Exception as e:
        print(f"⚠️ Warning: Could not start standalone MCP server: {e}")
    
    # 3. Integrate unified MCP server with FastAPI
    if mcp_server:
        try:
            mcp_app = mcp_server.get_http_app(path="/mcp")
            if mcp_app:
                # Mount the MCP app to the FastAPI app
                from src.api.main import app
                app.mount("/mcp", mcp_app)
                print("✅ MCP server integrated with FastAPI at /mcp")
        except Exception as e:
            print(f"⚠️ Warning: Could not integrate MCP server: {e}")
    
    # 4. Start FastAPI server on port 8003
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8003,
        reload=False,
        log_level="info"
    )
    
    # 5. Display server endpoints
    print("\n🎉 All services are now running!")
    print("🌐 Access URLs:")
    print("   📊 Main UI:        http://localhost:8501")
    print("   🏠 Landing Page:   http://localhost:8502")
    print("   🔗 API Docs:       http://localhost:8003/docs")
    print("   🤖 MCP Server:     http://localhost:8003/mcp (FastAPI integrated)")
    print("   🔧 Standalone MCP: http://localhost:8000 (Strands integration)")
    print("💡 For Strands integration, use: streamablehttp_client('http://localhost:8000/mcp')")
```

### MCP Server HTTP Integration Method
```python
def get_http_app(self, path: str = "/mcp"):
    """Get the HTTP app for integration with FastAPI."""
    if not self.mcp:
        logger.error("MCP server not available")
        return None
    
    try:
        logger.info(f"🚀 Creating MCP HTTP app at path: {path}")
        return self.mcp.http_app(path=path)
    except Exception as e:
        logger.error(f"Error creating MCP HTTP app: {e}")
        return None
```

### MCP Compliance Checklist
- [x] All operations go through MCP tools
- [x] No direct API access to agents
- [x] Consistent error handling patterns with unified interface
- [x] Proper tool documentation with descriptions
- [x] Async/await usage throughout
- [x] Input validation and sanitization
- [x] FastMCP server implementation (without 'description' parameter)
- [x] FastMCP client integration
- [x] Streamable HTTP transport support
- [x] Proper agent initialization and management
- [x] Unified MCP server architecture (25 consolidated tools)
- [x] Dual server architecture (FastAPI + Standalone)
- [x] Strands integration with Streamable HTTP transport
- [x] Asyncio thread conflict resolution
- [x] Process management for Windows environment
- [x] Standalone MCP server on port 8000
- [x] FastAPI MCP integration on port 8003
- [x] Open Library integration with enhanced MCP server
- [x] Content type detection including Open Library URLs
- [x] Automatic vector database storage for downloaded content
- [x] Knowledge graph generation from Open Library content

### MCP Implementation Guide

#### 1. Server Setup (main.py)
```python
# Required imports
from fastmcp import FastMCP
from core.error_handler import with_error_handling
from agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
from agents.knowledge_graph_agent import KnowledgeGraphAgent
# ... other agent imports

class OptimizedMCPServer:
    def __init__(self):
        # Initialize FastMCP server
        self.mcp = FastMCP("Sentiment Analysis Agents Server")
        self.agents = {}
        self._initialize_agents()
        self._register_optimized_tools()
    
    def _initialize_agents(self):
        """Initialize all agents for MCP tool access."""
        try:
            self.agents["file_extraction"] = EnhancedFileExtractionAgent()
            self.agents["knowledge_graph"] = KnowledgeGraphAgent()
            # ... initialize other agents
            print(f"✅ Initialized {len(self.agents)} agents")
        except Exception as e:
            print(f"⚠️ Error initializing agents: {e}")
    
    def _register_optimized_tools(self):
        """Register MCP tools with proper error handling."""
        if self.mcp is None:
            print("❌ MCP server not initialized")
            return
        
        # Register tools with @self.mcp.tool decorator
        # Use @with_error_handling for consistent error handling
```

#### 2. Tool Registration Pattern
```python
@self.mcp.tool(description="Process multilingual PDF using MCP framework")
@with_error_handling("multilingual_pdf_processing")
async def process_multilingual_pdf_mcp(
    pdf_path: str,
    language: str = "auto",
    generate_report: bool = True,
    output_path: str = None
):
    """Process multilingual PDF using fixed components through MCP framework."""
    try:
        # Step 1: Import language configuration
        from src.config.language_config import LanguageConfigFactory
        
        # Step 2: Use agents for processing
        file_agent = self.agents["file_extraction"]
        kg_agent = self.agents["knowledge_graph"]
        
        # Step 3: Process and return results
        result = {
            "success": True,
            "pdf_path": pdf_path,
            "language": language,
            # ... other result data
        }
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}
```

#### 3. Client Implementation (src/mcp_servers/client_example.py)
```python
# Option 1: FastMCP Client (Recommended)
try:
    from fastmcp import FastMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

class MCPClient:
    def __init__(self, server_url: str = "http://localhost:8001/mcp/"):
        self.server_url = server_url
        self.mcp_client = None
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]):
        """Call MCP tool with parameters."""
        if not MCP_AVAILABLE:
            return {"error": "MCP not available"}
        
        try:
            # Create FastMCP client
            self.mcp_client = FastMCPClient(self.server_url)
            await self.mcp_client.connect()
            
            # Call tool
            result = await self.mcp_client.call_tool(tool_name, parameters)
            return result
            
        except Exception as e:
            return {"error": str(e)}
```

#### 4. API Integration (src/api/main.py)
```python
from fastapi import FastAPI, HTTPException
from mcp_servers.client_example import MCPClient

@app.post("/process/multilingual-pdf")
async def process_multilingual_pdf(
    pdf_path: str,
    language: str = "auto",
    generate_report: bool = True,
    output_path: str = None
):
    """Process multilingual PDF using MCP tool."""
    try:
        # Use MCP client to call MCP tools
        mcp_client = MCPClient()
        result = await mcp_client.call_tool(
            "process_multilingual_pdf_mcp",
            {
                "pdf_path": pdf_path,
                "language": language,
                "generate_report": generate_report,
                "output_path": output_path
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Processing failed")
            )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 5. Testing MCP Integration
```python
# Test/mcp_integration_test.py
import asyncio
from mcp_servers.client_example import MCPClient

async def test_mcp_integration():
    """Test MCP client integration."""
    client = MCPClient()
    
    # Test PDF processing
    result = await client.call_tool(
        "process_multilingual_pdf_mcp",
        {
            "pdf_path": "data/sample.pdf",
            "language": "auto",
            "generate_report": True
        }
    )
    
    print(f"MCP Test Result: {result}")

if __name__ == "__main__":
    asyncio.run(test_mcp_integration())
```

#### 6. Error Handling Pattern
```python
# Use @with_error_handling decorator for all MCP tools
@with_error_handling("tool_category")
async def mcp_tool():
    """MCP tool with consistent error handling."""
    try:
        # Tool implementation
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

#### 7. Port Configuration
- **FastAPI Server**: Port 8000 (main application)
- **MCP Server**: Port 8001 (MCP tools)
- **Wait Time**: 30-60 seconds for server startup

#### 8. Dependencies Required
```txt
# requirements.prod.txt
fastmcp==0.1.0          # FastMCP server and client
mcp==1.0.0              # Core MCP protocol
```

#### 9. Common MCP Issues and Solutions (Updated with Real-World Fixes)

##### Issue 1: FastMCP Parameter Error
```python
# Error: "FastMCP.__init__() got an unexpected keyword argument 'description'"
# Solution: Remove 'description' parameter from FastMCP constructor
try:
    # INCORRECT:
    # self.mcp = FastMCP(name="server", version="1.0.0", description="description")
    
    # CORRECT:
    self.mcp = FastMCP(name="unified_sentiment_mcp_server", version="1.0.0")
    logger.info("✅ MCP server initialized")
except Exception as e:
    logger.error(f"❌ Error initializing MCP server: {e}")
    self.mcp = None
```

##### Issue 2: Asyncio Thread Conflict
```python
# Error: "Already running asyncio in this thread"
# Solution: Use synchronous run method and proper thread management
# INCORRECT: async def run() with asyncio.run() in thread
# CORRECT: def run() with direct method call

def run(self, host: str = "localhost", port: int = 8000, debug: bool = False):
    """Run the MCP server (synchronous method)."""
    if not self.mcp:
        logger.error("MCP server not available")
        return
    
    try:
        logger.info(f"🚀 Starting Unified MCP Server on {host}:{port}")
        self.mcp.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Error running MCP server: {e}")
```

##### Issue 3: MCP Server Integration with FastAPI
```python
# Solution: Use FastMCP's http_app() method for integration
def get_http_app(self, path: str = "/mcp"):
    """Get the HTTP app for integration with FastAPI."""
    if not self.mcp:
        logger.error("MCP server not available")
        return None
    
    try:
        logger.info(f"🚀 Creating MCP HTTP app at path: {path}")
        return self.mcp.http_app(path=path)
    except Exception as e:
        logger.error(f"Error creating MCP HTTP app: {e}")
        return None

# In main.py:
if mcp_server:
    try:
        mcp_app = mcp_server.get_http_app(path="/mcp")
        if mcp_app:
            from src.api.main import app
            app.mount("/mcp", mcp_app)
            print("✅ MCP server integrated with FastAPI at /mcp")
    except Exception as e:
        print(f"⚠️ Warning: Could not integrate MCP server: {e}")
```

##### Issue 4: Windows Process Management
```python
# Error: "taskkill: command not found" or similar
# Solution: Use proper Windows process management
import subprocess
import os

def kill_python_processes():
    """Kill all Python processes on Windows."""
    try:
        # Windows-specific process termination
        subprocess.run(["taskkill", "/f", "/im", "python.exe"], 
                      capture_output=True, check=False)
        print("✅ Python processes terminated")
    except Exception as e:
        print(f"⚠️ Warning: Could not terminate processes: {e}")

# Alternative: Use .venv/Scripts/python.exe for script execution
# .venv/Scripts/python.exe main.py
```

##### Issue 5: MCP Client Connection Failed
```python
# Error: "Failed to connect to MCP server"
# Solution: Check server startup and ports (updated for new architecture)
try:
    from fastmcp import FastMCPClient
    # Updated port: MCP server now integrated with FastAPI on port 8003
    client = FastMCPClient("http://localhost:8003/mcp/")
    await client.connect()
except Exception as e:
    print(f"Connection failed: {e}")
    # Check if server is running on port 8003
```

##### Issue 2: Import Error for MCP Libraries
```python
# Error: "cannot import name 'FastMCP' from 'fastmcp'"
# Solution: Install correct dependencies
pip install fastmcp==0.1.0 mcp==1.0.0

# Alternative: Use fallback implementation
try:
    from fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️ FastMCP not available - using fallback")
```

##### Issue 3: Tool Not Found Error
```python
# Error: "Tool 'process_multilingual_pdf_mcp' not found"
# Solution: Check tool registration in server
@self.mcp.tool(description="Process multilingual PDF using MCP framework")
async def process_multilingual_pdf_mcp(...):
    # Tool implementation
    pass

# Verify tool is registered by checking server startup logs
```

##### Issue 4: Transport Configuration Error
```python
# Error: "MCPClient.__init__() missing 1 required positional argument: 'create_transport'"
# Solution: Use correct client initialization
# For FastMCP (Recommended)
from fastmcp import FastMCPClient
client = FastMCPClient("http://localhost:8001/mcp/")

# For Streamable HTTP (Alternative)
from mcp.client.streamable_http import streamablehttp_client
from strands.tools.mcp.mcp_client import MCPClient

def create_transport():
    return streamablehttp_client("http://localhost:8001/mcp/")

client = MCPClient(create_transport)
```

##### Issue 5: Server Startup Timeout
```python
# Error: "Server not responding after 30 seconds"
# Solution: Increase wait time and check server logs
# Wait 30-60 seconds for server to fully load
# Check main.py startup logs for MCP server initialization
```

#### 10. MCP Testing Checklist (Updated for Unified Architecture)
- [ ] Server starts without errors
- [ ] 25 unified MCP tools are registered successfully
- [ ] FastAPI + MCP integration works on port 8003
- [ ] Client can connect to server at /mcp endpoint
- [ ] Tools can be called with parameters
- [ ] Results are returned correctly
- [ ] Error handling works properly
- [ ] API endpoints use MCP tools
- [ ] No direct agent access bypassing MCP
- [ ] Asyncio thread conflicts are resolved
- [ ] Windows process management works correctly
- [ ] FastMCP parameter issues are fixed
- [ ] Server integration with FastAPI is functional

---

## Open Library Integration Framework

### Overview
The system now includes comprehensive Open Library integration capabilities through the enhanced MCP server architecture. This allows for automatic downloading, processing, and analysis of content from Open Library URLs, with seamless integration into the existing vector database and knowledge graph systems.

### Enhanced MCP Server Architecture

#### EnhancedUnifiedMCPServer Implementation
```python
# src/mcp_servers/enhanced_unified_mcp_server.py
class EnhancedUnifiedMCPServer(UnifiedMCPServer):
    """
    Enhanced Unified MCP Server with Open Library integration.
    
    Extends the base UnifiedMCPServer with:
    - Open Library URL detection and processing
    - Enhanced content type detection
    - Integrated vector database storage
    - Knowledge graph generation
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize additional services for Open Library processing
        self.vector_db = VectorDBManager()
        self.kg_utility = ImprovedKnowledgeGraphUtility()
        self.kg_agent = KnowledgeGraphAgent()
        self.web_agent = EnhancedWebAgent()
        self.translation_service = TranslationService()
        
        logger.info("Enhanced Unified MCP Server initialized with Open Library support")
    
    def _detect_content_type(self, content: str) -> str:
        """Enhanced content type detection including Open Library URLs."""
        # Check for Open Library URL patterns
        if self._is_openlibrary_url(content):
            return "openlibrary"
        
        # Fall back to base detection
        return super()._detect_content_type(content)
    
    def _is_openlibrary_url(self, content: str) -> bool:
        """Check if content is an Open Library URL."""
        openlibrary_patterns = [
            r"https?://openlibrary\.org/books/",
            r"https?://openlibrary\.org/works/",
            r"https?://openlibrary\.org/authors/"
        ]
        
        for pattern in openlibrary_patterns:
            if re.search(pattern, content):
                return True
        return False
    
    async def _download_openlibrary_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Download content from Open Library URL."""
        try:
            # Use the web agent's _fetch_webpage method directly
            webpage_data = await self.web_agent._fetch_webpage(url)
            
            # Process the webpage data
            cleaned_text = self.web_agent._clean_webpage_text(webpage_data["html"])
            
            webpage_content = {
                "url": url,
                "title": webpage_data["title"],
                "text": cleaned_text,
                "html": webpage_data["html"],
                "status_code": webpage_data["status_code"]
            }
            
            return webpage_content
        except Exception as e:
            logger.error(f"Error downloading Open Library content: {e}")
            return None
    
    def _extract_metadata_from_content(self, content: str, title: str, url: str = "") -> Dict[str, Any]:
        """Extract metadata from Open Library content."""
        metadata = {
            "title": title,
            "url": url,
            "source": "openlibrary",
            "content_length": len(content),
            "extraction_timestamp": datetime.now().isoformat(),
            "content_type": "book"
        }
        
        # Extract additional metadata from title or content
        if "War and Peace" in title:
            metadata.update({
                "author": "Leo Tolstoy",
                "original_title": "Война и мир",
                "publication_year": "1869",
                "genre": "Historical Fiction"
            })
        
        return metadata
    
    def _generate_summary(self, content: str, title: str, author: Optional[str] = None) -> str:
        """Generate summary for Open Library content."""
        try:
            # Use the first 1000 characters for summary generation
            summary_content = content[:1000] if len(content) > 1000 else content
            
            summary_prompt = f"""
            Generate a concise summary of the following content from '{title}'{f' by {author}' if author else ''}:
            
            {summary_content}
            
            Summary:
            """
            
            # Use translation service for summary generation
            summary = self.translation_service.generate_summary(summary_prompt)
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Summary of {title} (summary generation failed)"
    
    async def _process_openlibrary_content(self, url: str, language: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Complete pipeline for processing Open Library content."""
        try:
            logger.info(f"🔄 Processing Open Library content from: {url}")
            
            # Step 1: Download content
            webpage_content = await self._download_openlibrary_content(url)
            if not webpage_content:
                return {"success": False, "error": "Failed to download content"}
            
            # Step 2: Extract metadata
            metadata = self._extract_metadata_from_content(
                webpage_content["text"], 
                webpage_content["title"], 
                url
            )
            
            # Step 3: Generate summary
            summary = self._generate_summary(
                webpage_content["text"], 
                webpage_content["title"],
                metadata.get("author")
            )
            
            # Step 4: Store in vector database
            vector_result = await self.vector_db.store_content(
                content=webpage_content["text"],
                metadata=metadata,
                content_type="openlibrary"
            )
            
            # Step 5: Create knowledge graph
            kg_result = await self.kg_utility.create_knowledge_graph(
                content=webpage_content["text"],
                title=webpage_content["title"],
                metadata=metadata
            )
            
            # Step 6: Analyze sentiment
            sentiment_result = await self.text_agent.analyze_sentiment(
                webpage_content["text"], 
                language
            )
            
            result = {
                "success": True,
                "url": url,
                "title": webpage_content["title"],
                "content_length": len(webpage_content["text"]),
                "metadata": metadata,
                "summary": summary,
                "vector_storage": vector_result,
                "knowledge_graph": kg_result,
                "sentiment_analysis": sentiment_result,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Open Library content processed successfully: {webpage_content['title']}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing Open Library content: {e}")
            return {"success": False, "error": str(e)}
```

### Enhanced Tool Registration

#### Enhanced process_content Tool
```python
@self.mcp.tool(description="Enhanced content processing with Open Library support")
async def process_content(
    content: str,
    content_type: str = "auto",
    language: str = "en",
    options: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Process any type of content with unified interface and Open Library support."""
    try:
        # Auto-detect content type if not specified
        if content_type == "auto":
            content_type = self._detect_content_type(content)
        
        # Handle Open Library URLs specifically
        if content_type == "openlibrary":
            result = await self._process_openlibrary_content(content, language, options)
            return result
        
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
        logger.error(f"Error processing content: {e}")
        return {"success": False, "error": str(e)}
```

#### New download_openlibrary_content Tool
```python
@self.mcp.tool(description="Download and process Open Library content")
async def download_openlibrary_content(url: str) -> Dict[str, Any]:
    """Download and process content from Open Library URL."""
    try:
        logger.info(f"🔄 Downloading Open Library content from: {url}")
        
        # Validate URL
        if not self._is_openlibrary_url(url):
            return {"success": False, "error": "Invalid Open Library URL"}
        
        # Process the Open Library content
        result = await self._process_openlibrary_content(url, "en", {})
        
        return result
    except Exception as e:
        logger.error(f"Error downloading Open Library content: {e}")
        return {"success": False, "error": str(e)}
```

### Generic Open Library Downloader Script

#### Implementation Pattern
```python
# generic_openlibrary_downloader.py
async def download_openlibrary_content(url: str) -> Optional[Dict[str, Any]]:
    """Download and process content from any Open Library URL."""
    try:
        # Initialize services
        web_agent = EnhancedWebAgent()
        vector_db = VectorDBManager()
        kg_utility = ImprovedKnowledgeGraphUtility()
        text_agent = UnifiedTextAgent()
        
        # Download content
        webpage_data = await web_agent._fetch_webpage(url)
        cleaned_text = web_agent._clean_webpage_text(webpage_data["html"])
        
        # Extract metadata
        metadata = {
            "url": url,
            "title": webpage_data["title"],
            "source": "openlibrary",
            "content_length": len(cleaned_text),
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        # Store in vector database
        vector_result = await vector_db.store_content(
            content=cleaned_text,
            metadata=metadata,
            content_type="openlibrary"
        )
        
        # Create knowledge graph
        kg_result = await kg_utility.create_knowledge_graph(
            content=cleaned_text,
            title=webpage_data["title"],
            metadata=metadata
        )
        
        # Analyze sentiment
        sentiment_result = await text_agent.analyze_sentiment(cleaned_text, "en")
        
        return {
            "success": True,
            "url": url,
            "title": webpage_data["title"],
            "content_length": len(cleaned_text),
            "metadata": metadata,
            "vector_storage": vector_result,
            "knowledge_graph": kg_result,
            "sentiment_analysis": sentiment_result
        }
        
    except Exception as e:
        logger.error(f"Error processing Open Library content: {e}")
        return None
```

### Content Processing Pipeline

#### Open Library Processing Flow
1. **URL Validation**: Check if the provided content is a valid Open Library URL
2. **Content Download**: Use `EnhancedWebAgent` to download the webpage content
3. **Text Extraction**: Clean and extract text from the HTML content
4. **Metadata Extraction**: Extract relevant metadata (title, author, publication info)
5. **Vector Storage**: Store the content in ChromaDB for semantic search
6. **Knowledge Graph Generation**: Create a knowledge graph from the content
7. **Sentiment Analysis**: Analyze the sentiment of the content
8. **Summary Generation**: Generate a concise summary of the content
9. **Result Compilation**: Return comprehensive processing results

#### Integration Points
- **Vector Database**: Automatic storage in ChromaDB with metadata
- **Knowledge Graph**: Entity extraction and relationship mapping
- **Sentiment Analysis**: Multilingual sentiment analysis
- **Translation Service**: Summary generation and content translation
- **Web Agent**: Enhanced web scraping capabilities

### Usage Examples

#### Using Enhanced MCP Server
```python
# Initialize enhanced MCP server
from src.mcp_servers.enhanced_unified_mcp_server import EnhancedUnifiedMCPServer

server = EnhancedUnifiedMCPServer()

# Process Open Library content
result = await server.process_content(
    content="https://openlibrary.org/books/OL14047767M/Voina_i_mir_%D0%92%D0%9E%D0%99%D0%9D%D0%90_%D0%B8_%D0%9C%D0%98%D0%A0%D0%AA",
    content_type="auto",
    language="en"
)

# Download specific Open Library content
result = await server.download_openlibrary_content(
    url="https://openlibrary.org/books/OL14047767M/Voina_i_mir_%D0%92%D0%9E%D0%99%D0%9D%D0%90_%D0%B8_%D0%9C%D0%98%D0%A0%D0%AA"
)
```

#### Using Generic Downloader Script
```python
# Use the generic downloader for any Open Library URL
from generic_openlibrary_downloader import download_openlibrary_content

result = await download_openlibrary_content(
    "https://openlibrary.org/books/OL14047767M/Voina_i_mir_%D0%92%D0%9E%D0%99%D0%9D%D0%90_%D0%B8_%D0%9C%D0%98%D0%A0%D0%AA"
)

if result and result["success"]:
    print(f"✅ Downloaded: {result['title']}")
    print(f"📊 Content length: {result['content_length']} characters")
    print(f"🗄️ Vector storage: {result['vector_storage']['success']}")
    print(f"🕸️ Knowledge graph: {result['knowledge_graph']['success']}")
```

### Error Handling and Validation

#### URL Validation
```python
def _is_openlibrary_url(self, content: str) -> bool:
    """Validate Open Library URL patterns."""
    openlibrary_patterns = [
        r"https?://openlibrary\.org/books/",
        r"https?://openlibrary\.org/works/",
        r"https?://openlibrary\.org/authors/"
    ]
    
    for pattern in openlibrary_patterns:
        if re.search(pattern, content):
            return True
    return False
```

#### Error Recovery
- **Download Failures**: Retry with exponential backoff
- **Processing Errors**: Graceful degradation with partial results
- **Storage Failures**: Fallback to local storage
- **Network Issues**: Circuit breaker pattern for external calls

### Performance Considerations

#### Optimization Strategies
1. **Caching**: Cache downloaded content to avoid re-downloading
2. **Batch Processing**: Process multiple URLs in parallel
3. **Incremental Updates**: Only download new or changed content
4. **Resource Management**: Proper cleanup of temporary resources

#### Monitoring and Metrics
- **Download Success Rate**: Track successful vs failed downloads
- **Processing Time**: Monitor content processing performance
- **Storage Usage**: Track vector database and knowledge graph growth
- **Error Rates**: Monitor and alert on processing errors

### Security and Compliance

#### Content Validation
- **URL Sanitization**: Validate and sanitize all URLs
- **Content Size Limits**: Implement reasonable size limits
- **Rate Limiting**: Prevent abuse of Open Library resources
- **Access Control**: Implement proper access controls

#### Data Privacy
- **Content Storage**: Secure storage of downloaded content
- **Metadata Handling**: Proper handling of sensitive metadata
- **Audit Logging**: Log all content processing activities
- **Data Retention**: Implement appropriate data retention policies

---

## Multilingual Processing Framework

### Critical Multilingual Configuration Standards

#### IMPORTANT: Multilingual Configuration Requirements
- **Language-Specific Regex Parsing**: All language-specific regex parsing parameters MUST be stored in configuration files under `/src/config`
- **Comprehensive File Integration**: Any changes must touch ALL related files with proper testing before reporting success
- **MCP Tool Integration**: All functionality must be integrated into MCP tools with proper testing and verification
- **Script Execution Standards**: All scripts must use `.venv/Scripts/python.exe` for execution
- **Testing Requirements**: Always test and verify everything working before reporting success

#### Configuration-Driven Multilingual Support
```python
# Example: Language-specific regex patterns in config files
class ChineseConfig(BaseLanguageConfig):
    def get_entity_patterns(self) -> EntityPatterns:
        return EntityPatterns(
            person_patterns=[
                r'[\u4e00-\u9fff]{2,4}',  # Chinese names (2-4 characters)
                r'[A-Z][a-z]+ [A-Z][a-z]+'  # English names in Chinese text
            ],
            organization_patterns=[
                r'[\u4e00-\u9fff]+(?:公司|集团|企业|机构)',  # Chinese organizations
                r'[A-Z][A-Z\s]+(?:Inc|Corp|Ltd|Company)'  # English organizations
            ],
            location_patterns=[
                r'[\u4e00-\u9fff]+(?:省|市|县|区)',  # Chinese administrative divisions
                r'[\u4e00-\u9fff]+(?:山|河|湖|海)',  # Chinese geographical features
            ]
        )

class RussianConfig(BaseLanguageConfig):
    def get_entity_patterns(self) -> EntityPatterns:
        return EntityPatterns(
            person_patterns=[
                r'[А-Я][а-я]+ [А-Я][а-я]+ [А-Я][а-я]+',  # Russian full names
                r'[А-Я][а-я]+ [А-Я]\.[А-Я]\.',  # Russian names with initials
            ],
            organization_patterns=[
                r'[А-Я][А-Я\s]+(?:ООО|ОАО|ЗАО|ИП)',  # Russian business entities
                r'[А-Я][а-я]+(?:Университет|Институт|Академия)',  # Russian educational institutions
            ],
            location_patterns=[
                r'[А-Я][а-я]+(?:область|край|республика)',  # Russian administrative divisions
                r'[А-Я][а-я]+(?:город|село|деревня)',  # Russian settlements
            ]
        )
```

### Language Configuration Factory
```python
class LanguageConfigFactory:
    @classmethod
    def get_config(cls, language_code: str) -> BaseLanguageConfig:
        """Return language-specific configuration with regex patterns."""
        config_map = {
            'zh': ChineseConfig(),
            'ru': RussianConfig(),
            'en': EnglishConfig(),
            'ja': JapaneseConfig(),
            'ko': KoreanConfig(),
            'ar': ArabicConfig(),
            'hi': HindiConfig(),
        }
        return config_map.get(language_code, EnglishConfig())
    
    @classmethod
    def detect_language_from_text(cls, text: str) -> str:
        """Automatic language detection with regex pattern matching."""
        # Use language-specific detection patterns from config files
        for lang_code, config in cls.get_all_configs().items():
            if config.detect_language_pattern(text):
                return lang_code
        return 'en'  # Default to English
```

### Supported Languages with Regex Configuration
- **Chinese (zh)**: Modern and Classical Chinese with character-based regex patterns
- **Russian (ru)**: Cyrillic text processing with Russian-specific entity patterns
- **English (en)**: Standard English processing with Latin alphabet patterns
- **Japanese (ja)**: Japanese with Kanji, Hiragana, and Katakana regex support
- **Korean (ko)**: Korean text processing with Hangul character patterns
- **Arabic (ar)**: Arabic with RTL support and Arabic-specific patterns
- **Hindi (hi)**: Hindi text processing with Devanagari script patterns

### Language-Specific Features (Configuration-Driven)
Each language configuration MUST include:
- **Entity Patterns**: Regex patterns for entity extraction stored in config files
- **Processing Settings**: Language-specific parameters in configuration
- **Ollama Models**: Appropriate model configurations per language
- **Grammar Patterns**: Language-specific structures and rules
- **Detection Patterns**: Language identification regex patterns
- **Regex Parsing Parameters**: All language-specific regex parsing differences

### Multilingual Processing Pipeline
1. **Language Detection**: Automatic detection using config-based patterns
2. **Configuration Loading**: Load language-specific settings from config files
3. **Entity Extraction**: Use language-appropriate regex patterns from config
4. **Knowledge Graph Processing**: Apply language-specific rules and patterns
5. **Result Generation**: Format results appropriately for each language
6. **MCP Tool Integration**: All processing goes through MCP tools
7. **Testing and Verification**: Comprehensive testing before reporting success

### MCP Tool Integration for Multilingual Processing
```python
@self.mcp.tool(description="Multilingual content processing with config-driven regex")
async def process_multilingual_content(
    content: str,
    language: str = "auto",
    entity_types: List[str] = None,
    options: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Process content with language-specific regex patterns from config files."""
    try:
        # 1. Detect language if auto
        if language == "auto":
            language = LanguageConfigFactory.detect_language_from_text(content)
        
        # 2. Load language-specific configuration
        config = LanguageConfigFactory.get_config(language)
        
        # 3. Use language-specific regex patterns
        entity_patterns = config.get_entity_patterns()
        processing_settings = config.get_processing_settings()
        
        # 4. Process with language-appropriate patterns
        result = await self.text_agent.process_with_language_config(
            content, language, entity_patterns, processing_settings
        )
        
        return {"success": True, "result": result, "language": language}
    except Exception as e:
        logger.error(f"Error in multilingual processing: {e}")
        return {"success": False, "error": str(e)}
```

### File Integration Requirements
When implementing multilingual features:
1. **Update ALL related configuration files** in `/src/config/language_config/`
2. **Update entity extraction patterns** for each supported language
3. **Update MCP tool implementations** to use language-specific configs
4. **Update testing framework** to test all language configurations
5. **Update documentation** to reflect language-specific capabilities
6. **Test and verify** all functionality before reporting success

---

## Configuration Management

### Configuration File Structure
```
src/config/
├── language_config/          # Language-specific configurations
│   ├── base_config.py       # Base configuration class
│   ├── chinese_config.py    # Chinese language config
│   ├── russian_config.py    # Russian language config
│   └── ...                  # Other language configs
├── model_config.py          # Model configurations
├── settings.py              # System settings
├── ollama_config.py         # Ollama model settings
└── mcp_config.py           # MCP server configurations
```

### Configuration Loading Pattern
```python
# Always use configuration factory
language_config = LanguageConfigFactory.get_config(detected_language)

# Access configuration through methods
entity_patterns = language_config.get_entity_patterns()
processing_settings = language_config.get_processing_settings()
ollama_config = language_config.get_ollama_config()
```

### Environment-Specific Settings
- **Development**: Local development settings
- **Testing**: Test-specific configurations
- **Production**: Production-optimized settings

---

## Testing Framework

### Test Organization
```
Test/
├── unit/                    # Unit tests
├── integration/             # Integration tests
├── performance/             # Performance tests
└── multilingual/            # Multilingual-specific tests
```

### Test Categories
1. **Configuration Tests**: Language config validation
2. **Component Tests**: Individual component testing
3. **Integration Tests**: End-to-end workflow testing
4. **MCP Tests**: MCP tool functionality testing
5. **Multilingual Tests**: Language-specific processing testing

### Test Execution Pattern
```python
# Use .venv/Scripts/python.exe for all test execution
.venv/Scripts/python.exe Test/test_name.py

# Test results should be stored in Results/ directory
# Test reports should include performance metrics

# Multilingual-specific testing requirements
.venv/Scripts/python.exe Test/multilingual/test_language_configs.py
.venv/Scripts/python.exe Test/multilingual/test_regex_patterns.py
.venv/Scripts/python.exe Test/multilingual/test_entity_extraction.py

# MCP tool testing for multilingual support
.venv/Scripts/python.exe Test/mcp/test_multilingual_mcp_tools.py
.venv/Scripts/python.exe Test/mcp/test_entity_extraction_fix.py

# MCP-specific testing
# Test unified MCP server functionality
.venv/Scripts/python.exe Test/test_unified_mcp_server.py

# Test enhanced MCP server with Open Library support
.venv/Scripts/python.exe Test/test_enhanced_mcp_server.py

# Test Open Library integration
.venv/Scripts/python.exe Test/test_openlibrary_integration.py

# Test generic Open Library downloader
.venv/Scripts/python.exe Test/test_generic_openlibrary_downloader.py

# Test FastAPI + MCP integration
curl http://localhost:8003/mcp

# Test individual MCP tools
.venv/Scripts/python.exe Test/test_mcp_tools.py
```

### Test Requirements
- [ ] All components must have unit tests
- [ ] Integration tests for all workflows
- [ ] Multilingual processing tests with language-specific regex patterns
- [ ] Language configuration validation tests for all supported languages
- [ ] Entity extraction tests with language-specific regex patterns
- [ ] MCP tool functionality tests (25 unified tools)
- [ ] MCP tool integration tests for multilingual support
- [ ] Entity extraction fix validation tests
- [ ] Open Library integration tests
- [ ] Enhanced MCP server functionality tests
- [ ] Content type detection tests
- [ ] Vector database integration tests
- [ ] Knowledge graph generation tests
- [ ] Performance benchmarks
- [ ] Error handling validation
- [ ] FastAPI + MCP integration tests
- [ ] Unified MCP server functionality tests
- [ ] Windows process management tests
- [ ] Asyncio thread conflict resolution tests
- [ ] Comprehensive file integration tests for all related files
- [ ] Script execution tests using .venv/Scripts/python.exe

---

## File Organization Standards

### Directory Structure
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

### Naming Conventions
- **Files**: snake_case.py
- **Classes**: PascalCase
- **Functions**: snake_case
- **Constants**: UPPER_SNAKE_CASE
- **Variables**: snake_case

### Import Organization
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

---

## Coding Standards

### Python Style Guide
- **PEP 8**: Primary style guide
- **Type Hints**: Required for all function signatures
- **Docstrings**: Required for all classes and functions
- **Error Handling**: Comprehensive exception handling

### Code Quality Requirements
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

### Async/Await Usage
- **All I/O operations must be async**
- **Use asyncio.gather() for parallel operations**
- **Proper exception handling in async contexts**
- **Avoid blocking operations in async functions**

### Logging Standards
```python
# Use structured logging with loguru
logger.info("Operation started", extra={"component": "agent_name"})
logger.error("Operation failed", extra={"error": str(e), "component": "agent_name"})
logger.debug("Debug information", extra={"data": debug_data})
```

---

## Integration Patterns

### Agent Integration Pattern
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

### MCP Tool Integration Pattern (Unified Interface)
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

### Configuration Integration Pattern
```python
# Always use factory pattern for configuration
config = LanguageConfigFactory.get_config(language_code)

# Access configuration through methods
patterns = config.get_entity_patterns()
settings = config.get_processing_settings()
```

---

## Error Handling Standards

### Error Categories
1. **Validation Errors**: Input validation failures
2. **Processing Errors**: Processing pipeline failures
3. **External Errors**: External service failures
4. **Configuration Errors**: Configuration loading failures
5. **System Errors**: System-level failures

### Error Handling Pattern
```python
try:
    # Operation
    result = await operation()
    return {"success": True, "result": result}
except ValidationError as e:
    logger.warning(f"Validation error: {e}")
    return {"success": False, "error": f"Validation failed: {e}"}
except ProcessingError as e:
    logger.error(f"Processing error: {e}")
    return {"success": False, "error": f"Processing failed: {e}"}
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    return {"success": False, "error": f"Unexpected error: {e}"}
```

### Error Recovery Strategies
- **Retry with exponential backoff**
- **Fallback to alternative methods**
- **Graceful degradation**
- **Circuit breaker patterns**

---

## Performance Guidelines

### Optimization Principles
1. **Async processing throughout**
2. **Caching at multiple levels**
3. **Resource pooling**
4. **Lazy loading**
5. **Batch processing where possible**

### Performance Monitoring
```python
# Performance tracking
start_time = time.time()
result = await operation()
processing_time = time.time() - start_time

logger.info("Operation completed", extra={
    "processing_time": processing_time,
    "operation": "operation_name"
})
```

### Resource Management
- **Connection pooling for databases**
- **Model caching for LLMs**
- **Memory management for large files**
- **CPU optimization for processing**

---

## Security Considerations

### Input Validation
- **Sanitize all inputs**
- **Validate file types and sizes**
- **Check for malicious content**
- **Rate limiting for API endpoints**

### Data Protection
- **Encrypt sensitive data**
- **Secure configuration storage**
- **Access control for resources**
- **Audit logging for operations**

### External Service Security
- **Secure API keys management**
- **HTTPS for all external calls**
- **Certificate validation**
- **Timeout handling**

---

## Deployment Standards

### Environment Setup
```bash
# Virtual environment setup
python -m venv .venv
.venv/Scripts/activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Dependency installation
pip install -r requirements.txt

# MCP-specific setup
pip install fastmcp==0.1.0 mcp==1.0.0

# Process management (Windows)
# Use .venv/Scripts/python.exe for all script execution
.venv/Scripts/python.exe main.py

# Server startup with proper timing
# Wait 30-60 seconds for server to fully load
```

### Configuration Management
- **Environment-specific configs**
- **Secret management**
- **Feature flags**
- **Configuration validation**

### Monitoring and Logging
- **Health check endpoints**
- **Performance metrics**
- **Error tracking**
- **Usage analytics**

---

## Documentation Standards

### Code Documentation
- **Docstrings for all functions and classes**
- **Type hints for all parameters**
- **Example usage in docstrings**
- **Error handling documentation**

### API Documentation
- **OpenAPI/Swagger documentation**
- **MCP tool documentation**
- **Example requests and responses**
- **Error code documentation**

### System Documentation
- **Architecture diagrams**
- **Component interaction flows**
- **Configuration guides**
- **Troubleshooting guides**

---

## MCP Framework Fixes and Website Content Download Guidelines

### Critical MCP Fix Applied (2025-08-14)

#### Issue Identified
- **Problem**: MCP server calling non-existent `store_document` method on `VectorDBManager`
- **Location**: `src/mcp_servers/standalone_mcp_server.py` line 344
- **Error**: `'VectorDBManager' object has no attribute 'store_document'`

#### Solution Applied
```python
# Before (causing error):
result = await self.vector_store.store_document(content, metadata or {})

# After (working correctly):
result = await self.vector_store.store_content(content, metadata or {})
```

#### Files Modified
- `src/mcp_servers/standalone_mcp_server.py` - Fixed method call from `store_document` to `store_content`

#### Verification Steps
1. Restart MCP server after making changes
2. Test vector database storage functionality
3. Verify knowledge graph generation
4. Confirm content processing pipeline

### Website Content Download Framework

#### Supported Sources
1. **OpenLibrary.org** - Book metadata and content
2. **Chinese Text Project (ctext.org)** - Classical Chinese texts
3. **Internet Archive** - Historical documents and books
4. **Custom websites** - Via enhanced web agent

#### Implementation Pattern for New Websites

##### 1. Content Detection
```python
def _detect_content_type(self, content: str) -> str:
    """Detect content type based on content or file extension."""
    if content.startswith("http"):
        if "openlibrary.org" in content:
            return "openlibrary"
        elif "ctext.org" in content:
            return "ctext"
        elif "archive.org" in content:
            return "archive"
        else:
            return "website"
    # ... other detection logic
```

##### 2. MCP Tool Integration
```python
@self.mcp.tool(description="Process content from specific websites")
async def process_website_content(
    url: str,
    content_type: str = "auto",
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Process content from specific website types."""
    try:
        if content_type == "openlibrary":
            result = await self.web_agent.process_openlibrary_content(url)
        elif content_type == "ctext":
            result = await self.web_agent.process_ctext_content(url)
        elif content_type == "archive":
            result = await self.web_agent.process_archive_content(url)
        else:
            result = await self.web_agent.scrape_website(url)
        
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Error processing website content: {e}")
        return {"success": False, "error": str(e)}
```

##### 3. Vector Database Storage
```python
# Always use store_content method, not store_document
result = await self.vector_store.store_content(content, metadata or {})
```

##### 4. Knowledge Graph Generation
```python
# Generate knowledge graph for processed content
kg_result = await self.kg_agent.generate_knowledge_graph(content, "text")
```

#### Required Updates for New Website Integration

##### 1. Update Content Type Detection
- Add new website patterns to `_detect_content_type` method
- Update MCP server tool registration

##### 2. Create Website-Specific Agent Method
```python
async def process_newwebsite_content(self, url: str) -> Dict[str, Any]:
    """Process content from new website."""
    try:
        # Fetch content from website
        content = await self._fetch_website_content(url)
        
        # Extract metadata
        metadata = self._extract_website_metadata(content)
        
        # Process content
        processed_content = await self.text_agent.process_text(content)
        
        return {
            "content": processed_content,
            "metadata": metadata,
            "source": url
        }
    except Exception as e:
        logger.error(f"Error processing new website content: {e}")
        raise
```

##### 3. Update MCP Server Tools
- Add new tool to `_register_tools` method
- Ensure proper error handling
- Add to tool descriptions

##### 4. Update Configuration
- Add new website patterns to configuration files
- Update language detection patterns
- Add metadata extraction rules

#### Testing Requirements for New Websites

##### 1. Content Fetching Test
```python
async def test_new_website_content_fetching():
    """Test content fetching from new website."""
    url = "https://newwebsite.com/content"
    result = await web_agent.process_newwebsite_content(url)
    assert result["success"] == True
    assert "content" in result["result"]
```

##### 2. Vector Database Storage Test
```python
async def test_vector_db_storage():
    """Test vector database storage of new website content."""
    content = "Test content from new website"
    result = await vector_store.store_content(content, {"source": "newwebsite"})
    assert result is not None
```

##### 3. Knowledge Graph Generation Test
```python
async def test_knowledge_graph_generation():
    """Test knowledge graph generation from new website content."""
    content = "Test content from new website"
    result = await kg_agent.generate_knowledge_graph(content, "text")
    assert result["success"] == True
```

#### Error Handling Standards

##### 1. MCP Server Errors
- Always log errors with context
- Return structured error responses
- Include error codes for debugging

##### 2. Vector Database Errors
- Check method availability before calling
- Handle connection errors gracefully
- Provide fallback storage options

##### 3. Website Content Errors
- Handle network timeouts
- Implement retry mechanisms
- Provide content validation

#### Performance Considerations

##### 1. Content Processing
- Implement chunking for large content
- Use async processing for multiple sources
- Cache processed results

##### 2. Vector Database
- Batch operations when possible
- Use appropriate collection strategies
- Monitor storage usage

##### 3. Knowledge Graph
- Incremental updates for large graphs
- Optimize entity extraction
- Use efficient graph algorithms

### Integration Checklist for New Websites

- [ ] Add website pattern to content type detection
- [ ] Create website-specific processing method
- [ ] Update MCP server tool registration
- [ ] Add proper error handling
- [ ] Implement metadata extraction
- [ ] Test content fetching
- [ ] Test vector database storage
- [ ] Test knowledge graph generation
- [ ] Update configuration files
- [ ] Add comprehensive tests
- [ ] Update documentation
- [ ] Verify MCP integration
- [ ] Test performance with large content
- [ ] Validate error handling scenarios

---

## Compliance Checklist

### Before Any Implementation
- [ ] Review this design framework
- [ ] Ensure MCP compliance (unified server architecture)
- [ ] Follow multilingual patterns with configuration-driven regex parsing
- [ ] Store all language-specific regex parsing parameters in `/src/config` files
- [ ] Touch ALL related files when making changes
- [ ] Test and verify everything working before reporting success
- [ ] Use `.venv/Scripts/python.exe` for all script execution
- [ ] Integrate all functionality into MCP tools with proper testing
- [ ] Implement proper error handling
- [ ] Add comprehensive tests
- [ ] Update documentation
- [ ] Verify FastMCP integration patterns
- [ ] Test Windows process management
- [ ] Validate asyncio thread handling
- [ ] Test Open Library integration patterns
- [ ] Validate content type detection
- [ ] Verify vector database integration
- [ ] Test knowledge graph generation
- [ ] Verify MCP server method calls (use `store_content`, not `store_document`)
- [ ] Test website content download functionality
- [ ] Validate error handling for vector database operations
- [ ] Test entity extraction with language-specific regex patterns
- [ ] Validate multilingual configuration files for all supported languages

### Before Deployment
- [ ] All tests passing including multilingual tests
- [ ] Performance benchmarks met
- [ ] Security review completed
- [ ] Documentation updated
- [ ] Configuration validated for all supported languages
- [ ] Monitoring in place
- [ ] MCP server integration verified
- [ ] 25 unified tools functionality confirmed
- [ ] FastAPI + MCP integration tested
- [ ] Windows process management validated
- [ ] Open Library integration tested
- [ ] Enhanced MCP server functionality verified
- [ ] Content type detection validated
- [ ] Vector database integration confirmed
- [ ] Knowledge graph generation tested
- [ ] MCP server method calls verified (store_content vs store_document)
- [ ] Website content download functionality tested
- [ ] Vector database error handling validated
- [ ] Knowledge graph generation from website content tested
- [ ] Multilingual regex parsing configuration validated
- [ ] Entity extraction fix verified for all languages
- [ ] All related files updated and tested
- [ ] Script execution using .venv/Scripts/python.exe verified

---

## Version Control

### Branch Strategy
- **main**: Production-ready code
- **develop**: Integration branch
- **feature/***: Feature development
- **hotfix/***: Critical fixes

### Commit Standards
- **Conventional commits format**
- **Descriptive commit messages**
- **Reference issue numbers**
- **Include test updates**

---

## System Restart Framework

### Overview
The System Restart Framework provides comprehensive process management and restart capabilities for development and testing scenarios. This framework ensures proper cleanup of existing processes and verification of system readiness.

### Restart Manager Implementation
```python
# Comprehensive restart mechanism for development and testing
import subprocess
import time
import os
import signal
import psutil
from pathlib import Path

class SystemRestartManager:
    """Manages system restart with proper process cleanup and verification."""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.main_script = self.project_root / "main.py"
        self.python_exe = self.project_root / ".venv" / "Scripts" / "python.exe"
        
    def kill_python_processes(self):
        """Kill all Python processes safely."""
        try:
            # Method 1: Use taskkill on Windows
            if os.name == 'nt':  # Windows
                subprocess.run(["taskkill", "/f", "/im", "python.exe"], 
                             capture_output=True, check=False)
                subprocess.run(["taskkill", "/f", "/im", "pythonw.exe"], 
                             capture_output=True, check=False)
            else:  # Unix/Linux
                subprocess.run(["pkill", "-f", "python"], 
                             capture_output=True, check=False)
            
            # Method 2: Use psutil for more precise control
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        if any('main.py' in str(cmd) for cmd in proc.info['cmdline'] or []):
                            proc.terminate()
                            proc.wait(timeout=5)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                    pass
            
            print("✅ Python processes terminated")
            time.sleep(2)  # Wait for processes to fully terminate
            
        except Exception as e:
            print(f"⚠️ Warning: Could not terminate processes: {e}")
    
    def wait_for_server_ready(self, timeout: int = 30, check_interval: int = 1):
        """Wait for server to be ready with countdown."""
        print(f"⏳ Waiting {timeout} seconds for server to be ready...")
        for i in range(timeout, 0, -1):
            print(f"   {i} seconds remaining...", end="\r")
            time.sleep(check_interval)
        print("\n🚀 Server should be ready now!")
    
    def restart_system(self, wait_time: int = 30):
        """Complete system restart with verification."""
        print("🔄 Starting system restart...")
        print("=" * 50)
        
        # Step 1: Kill existing processes
        print("1️⃣ Killing existing Python processes...")
        self.kill_python_processes()
        
        # Step 2: Wait for cleanup
        print("2️⃣ Waiting for process cleanup...")
        time.sleep(3)
        
        # Step 3: Start main.py
        print("3️⃣ Starting main.py...")
        try:
            if not self.python_exe.exists():
                print(f"❌ Python executable not found at: {self.python_exe}")
                return False
                
            if not self.main_script.exists():
                print(f"❌ Main script not found at: {self.main_script}")
                return False
            
            # Start main.py in background
            process = subprocess.Popen(
                [str(self.python_exe), str(self.main_script)],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            print(f"✅ main.py started with PID: {process.pid}")
            
        except Exception as e:
            print(f"❌ Failed to start main.py: {e}")
            return False
        
        # Step 4: Wait for server to be ready
        print("4️⃣ Waiting for server initialization...")
        self.wait_for_server_ready(wait_time)
        
        # Step 5: Verify server is running
        print("5️⃣ Verifying server status...")
        return self.verify_server_status()
    
    def verify_server_status(self):
        """Verify that the server is running properly."""
        import requests
        
        endpoints = [
            ("http://localhost:8003/health", "FastAPI Health"),
            ("http://localhost:8003/mcp", "MCP Server"),
            ("http://localhost:8003/mcp/", "MCP Server (trailing slash)"),
            ("http://localhost:8003/mcp-health", "MCP Health"),
            ("http://localhost:8501", "Streamlit Main UI"),
            ("http://localhost:8502", "Streamlit Landing Page")
        ]
        
        all_ok = True
        print("🔍 Checking server endpoints...")
        
        for url, name in endpoints:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code in [200, 404]:  # 404 is OK for some endpoints
                    print(f"✅ {name}: {response.status_code}")
                else:
                    print(f"⚠️ {name}: {response.status_code}")
                    all_ok = False
            except requests.exceptions.ConnectionError:
                print(f"❌ {name}: Connection refused")
                all_ok = False
            except Exception as e:
                print(f"❌ {name}: Error - {e}")
                all_ok = False
        
        if all_ok:
            print("🎉 System restart completed successfully!")
        else:
            print("⚠️ Some endpoints may not be ready yet. Check server logs.")
        
        return all_ok

# Usage example:
def restart_and_verify():
    """Restart the system and verify it's working."""
    restart_manager = SystemRestartManager()
    success = restart_manager.restart_system(wait_time=30)
    
    if success:
        print("✅ System is ready for use!")
        print("🌐 Access URLs:")
        print("   📊 Main UI:        http://localhost:8501")
        print("   🏠 Landing Page:   http://localhost:8502")
        print("   🔗 API Docs:       http://localhost:8003/docs")
        print("   🤖 MCP Server:     http://localhost:8003/mcp")
    else:
        print("❌ System restart failed. Check logs for details.")
    
    return success

# Quick restart function for development
def quick_restart():
    """Quick restart for development (15 second wait)."""
    restart_manager = SystemRestartManager()
    return restart_manager.restart_system(wait_time=15)
```

### Usage Guidelines
1. **Development Restart**: Use `quick_restart()` for fast development cycles (15 seconds)
2. **Full Restart**: Use `restart_and_verify()` for complete system verification (30 seconds)
3. **Custom Restart**: Use `SystemRestartManager()` for custom configurations
4. **Verification**: Always verify server status after restart
5. **Logs**: Check server logs if verification fails

### Integration with Testing
- Use restart framework before running comprehensive tests
- Verify MCP server endpoints after restart
- Test both `/mcp` and `/mcp/` endpoints
- Ensure all services are running before proceeding

---

## Maintenance and Updates

### Regular Reviews
- **Monthly architecture reviews**
- **Quarterly security audits**
- **Performance optimization reviews**
- **Documentation updates**

### Update Process
1. **Review current framework**
2. **Identify needed changes**
3. **Update framework document**
4. **Communicate changes to team**
5. **Update implementation accordingly**

---

## Contact and Support

### Framework Maintainer
- **Primary Contact**: [To be assigned]
- **Review Schedule**: Monthly
- **Update Process**: Pull request with review

### Emergency Contacts
- **Critical Issues**: [To be assigned]
- **Security Issues**: [To be assigned]
- **Performance Issues**: [To be assigned]

---

*This framework is a living document and should be updated as the project evolves. All team members should review and follow this framework to ensure consistency and compliance across the entire project.*
