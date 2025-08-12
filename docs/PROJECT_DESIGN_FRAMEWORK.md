# Project Design Framework
## Sentiment Analysis Swarm with Multilingual MCP Integration

### Version: 1.1
### Last Updated: 2025-08-12
### Status: Active (MCP Implementation Enhanced)

### Recent Updates (v1.1)
- **Enhanced MCP Implementation Guide**: Comprehensive patterns for server, client, and API integration
- **FastMCP Integration**: Detailed implementation using FastMCP library
- **Client Implementation Patterns**: Both FastMCP and Streamable HTTP transport options
- **Troubleshooting Guide**: Common MCP issues and solutions
- **Testing Checklist**: Complete MCP testing validation
- **Port Configuration**: Clear server port assignments and startup timing

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Design Principles](#core-design-principles)
3. [Technology Stack](#technology-stack)
4. [Component Architecture](#component-architecture)
5. [MCP Framework Integration](#mcp-framework-integration)
6. [Multilingual Processing Framework](#multilingual-processing-framework)
7. [Configuration Management](#configuration-management)
8. [Testing Framework](#testing-framework)
9. [File Organization Standards](#file-organization-standards)
10. [Coding Standards](#coding-standards)
11. [Integration Patterns](#integration-patterns)
12. [Error Handling Standards](#error-handling-standards)
13. [Performance Guidelines](#performance-guidelines)
14. [Security Considerations](#security-considerations)
15. [Deployment Standards](#deployment-standards)
16. [Documentation Standards](#documentation-standards)

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
# - MCP server initialization and management
# - Agent swarm coordination
# - Tool registration and routing
# - System health monitoring

class OptimizedMCPServer:
    def __init__(self):
        self.mcp = FastMCP("Sentiment Analysis Agents Server")
        self.agents = {}
        self._initialize_agents()
        self._register_optimized_tools()
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

### MCP Libraries and Dependencies
```python
# Required MCP libraries (from requirements.prod.txt)
fastmcp==0.1.0          # FastMCP server implementation
mcp==1.0.0              # Core MCP protocol
```

### Server Implementation Pattern
```python
# main.py - MCP Server Setup
from fastmcp import FastMCP
from core.error_handler import with_error_handling

class OptimizedMCPServer:
    def __init__(self):
        # Initialize FastMCP server
        self.mcp = FastMCP("Sentiment Analysis Agents Server")
        self.agents = {}
        self._initialize_agents()
        self._register_optimized_tools()
    
    def _initialize_agents(self):
        """Initialize all agents for MCP tool access."""
        self.agents["text"] = UnifiedTextAgent()
        self.agents["audio"] = UnifiedAudioAgent()
        self.agents["vision"] = UnifiedVisionAgent()
        self.agents["knowledge_graph"] = KnowledgeGraphAgent()
        self.agents["file_extraction"] = EnhancedFileExtractionAgent()
        # ... other agents
    
    def _register_optimized_tools(self):
        """Register MCP tools with proper error handling."""
        if self.mcp is None:
            return
        
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
                # Implementation using agents
                file_agent = self.agents["file_extraction"]
                kg_agent = self.agents["knowledge_graph"]
                
                # Process PDF and return results
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}
```

### Client Implementation Pattern
```python
# src/mcp_servers/client_example.py - MCP Client Setup
import asyncio
from typing import Dict, Any

# Option 1: Using FastMCP Client (Recommended)
try:
    from fastmcp import FastMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

class SentimentMCPClient:
    def __init__(self, server_url: str = "http://localhost:8001/mcp/"):
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

# Option 2: Using Streamable HTTP Transport (Alternative)
try:
    from mcp.client.streamable_http import streamablehttp_client
    from strands.tools.mcp.mcp_client import MCPClient
    STRANDS_MCP_AVAILABLE = True
except ImportError:
    STRANDS_MCP_AVAILABLE = False

class StrandsMCPClient:
    def __init__(self, server_url: str = "http://localhost:8001/mcp/"):
        self.server_url = server_url
        self.mcp_client = None
    
    async def connect(self):
        """Connect using Strands MCP client."""
        if not STRANDS_MCP_AVAILABLE:
            return False
        
        try:
            def create_transport():
                return streamablehttp_client(self.server_url)
            
            self.mcp_client = MCPClient(create_transport)
            async with self.mcp_client:
                tools = await self.mcp_client.list_tools_async()
                print(f"Connected. Available tools: {len(tools)}")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
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

### Required MCP Tools (Current Implementation)
1. **`process_multilingual_pdf_mcp`**: Generic multilingual PDF processing
2. **`process_pdf_enhanced_multilingual`**: Enhanced multilingual PDF processing  
3. **`get_all_agents_status`**: Agent health monitoring
4. **`start_all_agents`**: Agent lifecycle management
5. **`stop_all_agents`**: Agent lifecycle management

### MCP Server Startup Pattern
```python
# main.py - Server Startup
if __name__ == "__main__":
    import uvicorn
    
    # Start MCP server on port 8001
    mcp_server = OptimizedMCPServer()
    
    # Start FastAPI server on port 8000
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
```

### MCP Compliance Checklist
- [x] All operations go through MCP tools
- [x] No direct API access to agents
- [x] Consistent error handling patterns with `@with_error_handling`
- [x] Proper tool documentation with descriptions
- [x] Async/await usage throughout
- [x] Input validation and sanitization
- [x] FastMCP server implementation
- [x] FastMCP client integration
- [x] Streamable HTTP transport support
- [x] Proper agent initialization and management

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

#### 9. Common MCP Issues and Solutions

##### Issue 1: MCP Client Connection Failed
```python
# Error: "Failed to connect to MCP server"
# Solution: Check server startup and ports
try:
    from fastmcp import FastMCPClient
    client = FastMCPClient("http://localhost:8001/mcp/")
    await client.connect()
except Exception as e:
    print(f"Connection failed: {e}")
    # Check if server is running on port 8001
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

#### 10. MCP Testing Checklist
- [ ] Server starts without errors
- [ ] MCP tools are registered successfully
- [ ] Client can connect to server
- [ ] Tools can be called with parameters
- [ ] Results are returned correctly
- [ ] Error handling works properly
- [ ] API endpoints use MCP tools
- [ ] No direct agent access bypassing MCP

---

## Multilingual Processing Framework

### Language Configuration Factory
```python
class LanguageConfigFactory:
    @classmethod
    def get_config(cls, language_code: str) -> BaseLanguageConfig:
        # Return language-specific configuration
    
    @classmethod
    def detect_language_from_text(cls, text: str) -> str:
        # Automatic language detection
```

### Supported Languages
- **Chinese (zh)**: Modern and Classical Chinese
- **Russian (ru)**: Cyrillic text processing
- **English (en)**: Standard English processing
- **Japanese (ja)**: Japanese with Kanji support
- **Korean (ko)**: Korean text processing
- **Arabic (ar)**: Arabic with RTL support
- **Hindi (hi)**: Hindi text processing

### Language-Specific Features
Each language configuration must include:
- **Entity Patterns**: Regex patterns for entity extraction
- **Processing Settings**: Language-specific parameters
- **Ollama Models**: Appropriate model configurations
- **Grammar Patterns**: Language-specific structures
- **Detection Patterns**: Language identification patterns

### Processing Pipeline
1. **Language Detection**: Automatic detection from content
2. **Configuration Loading**: Load language-specific settings
3. **Entity Extraction**: Use language-appropriate patterns
4. **Knowledge Graph Processing**: Apply language-specific rules
5. **Result Generation**: Format results appropriately

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
```

### Test Requirements
- [ ] All components must have unit tests
- [ ] Integration tests for all workflows
- [ ] Multilingual processing tests
- [ ] MCP tool functionality tests
- [ ] Performance benchmarks
- [ ] Error handling validation

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

### MCP Tool Integration Pattern
```python
@self.mcp.tool(description="Tool description")
@with_error_handling("tool_category")
async def tool_name(param1: str, param2: str = "default"):
    """Tool documentation with parameter descriptions."""
    # Validate inputs
    # Process request
    # Return standardized response
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

## Compliance Checklist

### Before Any Implementation
- [ ] Review this design framework
- [ ] Ensure MCP compliance
- [ ] Follow multilingual patterns
- [ ] Implement proper error handling
- [ ] Add comprehensive tests
- [ ] Update documentation

### Before Deployment
- [ ] All tests passing
- [ ] Performance benchmarks met
- [ ] Security review completed
- [ ] Documentation updated
- [ ] Configuration validated
- [ ] Monitoring in place

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
