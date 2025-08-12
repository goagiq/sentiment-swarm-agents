"""
MCP Server implementation following the sentiment-swarm-agents pattern.
This provides proper MCP tool integration with FastMCP and streamable HTTP support.
Enhanced with Ollama integration and language-specific configurations.
"""

import os
import sys
import threading
import warnings
from typing import List, Dict, Any, Optional
from loguru import logger

# Suppress websockets deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets.legacy")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets.server")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="uvicorn.protocols.websockets")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="uvicorn")

# Import MCP server
try:
    from fastmcp import FastMCP
    MCP_AVAILABLE = True
    logger.info("✅ FastMCP available")
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("⚠️ FastMCP not available")

# Import project modules
from src.config.settings import settings
from src.config.config import config
from src.core.strands_mock import Agent, Swarm, tool

# Import Ollama integration
try:
    from src.core.strands_ollama_integration import strands_ollama_integration
    from src.core.strands_ollama_integration import STRANDS_AVAILABLE
    OLLAMA_AVAILABLE = True
    logger.info("✅ Strands Ollama integration available")
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("⚠️ Strands Ollama integration not available")

# Import language configurations
try:
    from src.config.language_config import LanguageConfigFactory
    LANGUAGE_CONFIG_AVAILABLE = True
    logger.info("✅ Language configurations available")
except ImportError:
    LANGUAGE_CONFIG_AVAILABLE = False
    logger.warning("⚠️ Language configurations not available")

# Import agents
from src.agents.file_extraction_agent import FileExtractionAgent
from src.agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.core.models import AnalysisRequest, DataType


class OptimizedMCPServer:
    """Optimized MCP server providing unified access to all agents with reduced tool count.
    Enhanced with Ollama integration and language-specific configurations."""
    
    def __init__(self):
        # Initialize the MCP server with proper streamable HTTP support
        self.mcp = None
        self._initialize_mcp()
        
        # Initialize all agents
        self.agents = {}
        self._initialize_agents()
        
        # Initialize Ollama integration
        self.ollama_integration = None
        self._initialize_ollama()
        
        # Register optimized tools
        self._register_optimized_tools()
        
        logger.info("✅ Optimized MCP Server initialized with unified tools and Ollama integration")
    
    def _initialize_mcp(self):
        """Initialize the MCP server using FastMCP with streamable HTTP support."""
        if MCP_AVAILABLE:
            self.mcp = FastMCP("Sentiment Analysis Agents Server")
            logger.info("✅ FastMCP Server with streamable HTTP support initialized successfully")
        else:
            logger.warning("⚠️ FastMCP not available - skipping MCP server initialization")
            self.mcp = None
    
    def _initialize_ollama(self):
        """Initialize Ollama integration with language-specific configurations."""
        if OLLAMA_AVAILABLE:
            self.ollama_integration = strands_ollama_integration
            logger.info("✅ Ollama integration initialized with Strands framework")
            
            # Initialize language-specific models if available
            if LANGUAGE_CONFIG_AVAILABLE:
                self._initialize_language_specific_models()
        else:
            logger.warning("⚠️ Ollama integration not available")
    
    def _initialize_language_specific_models(self):
        """Initialize language-specific Ollama models."""
        try:
            # Get available languages
            available_languages = LanguageConfigFactory.get_available_languages()
            logger.info(f"✅ Language-specific models: {len(available_languages)} languages available")
            
            # Initialize models for each language
            for lang_code in available_languages:
                try:
                    config = LanguageConfigFactory.get_config(lang_code)
                    ollama_config = config.get_ollama_config()
                    
                    if ollama_config:
                        # Create language-specific models
                        for model_type, model_config in ollama_config.items():
                            model_name = f"{lang_code}_{model_type}"
                            self.ollama_integration.create_custom_model(
                                model_id=model_config["model_id"],
                                model_type=model_name,
                                temperature=model_config.get("temperature", 0.7),
                                max_tokens=model_config.get("max_tokens", 1000),
                                system_prompt=model_config.get("system_prompt", ""),
                                keep_alive=model_config.get("keep_alive", "10m")
                            )
                            logger.info(f"✅ Created {model_name} model: {model_config['model_id']}")
                            
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize {lang_code} models: {e}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize language-specific models: {e}")
    
    def _initialize_agents(self):
        """Initialize all agents with Ollama integration."""
        try:
            # Get default text model for system prompts
            default_text_model = None
            if OLLAMA_AVAILABLE:
                default_text_model = self.ollama_integration.get_text_model()
            
            # Initialize specialized agents
            self.agents["sentiment"] = Agent(
                name="sentiment_analyzer",
                system_prompt="You are a sentiment analysis expert. Analyze text sentiment and extract entities.",
                tools=[]
            )
            
            self.agents["vision"] = Agent(
                name="vision_analyzer", 
                system_prompt="You are a computer vision expert. Analyze images and extract visual features.",
                tools=[]
            )
            
            self.agents["audio"] = Agent(
                name="audio_analyzer",
                system_prompt="You are an audio analysis expert. Analyze audio content and extract features.",
                tools=[]
            )
            
            self.agents["web"] = Agent(
                name="web_analyzer",
                system_prompt="You are a web content analysis expert. Analyze web pages and extract information.",
                tools=[]
            )
            
            self.agents["orchestrator"] = Agent(
                name="orchestrator",
                system_prompt="You are an orchestration expert. Coordinate multiple agents for comprehensive analysis.",
                tools=[]
            )
            
            logger.info(f"✅ Initialized {len(self.agents)} agents with Ollama integration")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize agents: {e}")
    
    def _register_optimized_tools(self):
        """Register optimized tools with unified interfaces."""
        if self.mcp is None:
            logger.error("❌ MCP server not initialized")
            return
        
        try:
            # Core Management Tools (3)
            @self.mcp.tool(description="Get status of all available agents")
            async def get_all_agents_status():
                """Get status of all available agents."""
                try:
                    status = {}
                    for agent_name, agent in self.agents.items():
                        status[agent_name] = {
                            "agent_id": getattr(agent, 'name', f"{agent_name}_agent"),
                            "status": "active",
                            "type": agent.__class__.__name__,
                            "tools_count": len(getattr(agent, 'tools', []))
                        }
                    
                    return {
                        "success": True,
                        "total_agents": len(self.agents),
                        "agents": status
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            @self.mcp.tool(description="Start all agents")
            async def start_all_agents():
                """Start all agents."""
                try:
                    results = {}
                    for agent_name, agent in self.agents.items():
                        try:
                            if hasattr(agent, 'start'):
                                await agent.start()
                                results[agent_name] = {"success": True, "message": "Started"}
                            else:
                                results[agent_name] = {"success": True, "message": "No start method needed"}
                        except Exception as e:
                            results[agent_name] = {"success": False, "error": str(e)}
                    
                    return {
                        "success": True,
                        "results": results
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            @self.mcp.tool(description="Stop all agents")
            async def stop_all_agents():
                """Stop all agents."""
                try:
                    results = {}
                    for agent_name, agent in self.agents.items():
                        try:
                            if hasattr(agent, 'stop'):
                                await agent.stop()
                                results[agent_name] = {"success": True, "message": "Stopped"}
                            else:
                                results[agent_name] = {"success": True, "message": "No stop method needed"}
                        except Exception as e:
                            results[agent_name] = {"success": False, "error": str(e)}
                    
                    return {
                        "success": True,
                        "results": results
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            # Unified Analysis Tools (4)
            @self.mcp.tool(description="Analyze text content with unified interface")
            async def analyze_text_sentiment(text: str, language: str = "en"):
                """Analyze text content using sentiment agent."""
                try:
                    agent = self.agents["sentiment"]
                    prompt = f"Analyze the sentiment of this text: {text}"
                    result = await agent.run(prompt)
                    
                    return {
                        "success": True,
                        "agent_type": "sentiment",
                        "agent_used": "sentiment_analyzer",
                        "text": text,
                        "language": language,
                        "result": result
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            @self.mcp.tool(description="Analyze image content with unified interface")
            async def analyze_image_sentiment(image_path: str):
                """Analyze image content using vision agent."""
                try:
                    agent = self.agents["vision"]
                    prompt = f"Analyze this image: {image_path}"
                    result = await agent.run(prompt)
                    
                    return {
                        "success": True,
                        "agent_type": "vision",
                        "agent_used": "vision_analyzer",
                        "image_path": image_path,
                        "result": result
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            @self.mcp.tool(description="Analyze audio content with unified interface")
            async def analyze_audio_sentiment(audio_path: str):
                """Analyze audio content using audio agent."""
                try:
                    agent = self.agents["audio"]
                    prompt = f"Analyze this audio: {audio_path}"
                    result = await agent.run(prompt)
                    
                    return {
                        "success": True,
                        "agent_type": "audio",
                        "agent_used": "audio_analyzer",
                        "audio_path": audio_path,
                        "result": result
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            @self.mcp.tool(description="Analyze webpage content with unified interface")
            async def analyze_webpage_sentiment(url: str):
                """Analyze webpage content using web agent."""
                try:
                    agent = self.agents["web"]
                    prompt = f"Analyze this webpage: {url}"
                    result = await agent.run(prompt)
                    
                    return {
                        "success": True,
                        "agent_type": "web",
                        "agent_used": "web_analyzer",
                        "url": url,
                        "result": result
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            # Orchestrator Tools (2)
            @self.mcp.tool(description="Process query using orchestrator agent")
            async def process_query_orchestrator(query: str):
                """Process query using orchestrator agent."""
                try:
                    agent = self.agents["orchestrator"]
                    prompt = f"Process this query: {query}"
                    result = await agent.run(prompt)
                    
                    return {
                        "success": True,
                        "agent_type": "orchestrator",
                        "agent_used": "orchestrator",
                        "query": query,
                        "result": result
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            @self.mcp.tool(description="Get orchestrator tools and capabilities")
            async def get_orchestrator_tools():
                """Get orchestrator tools and capabilities."""
                try:
                    return {
                        "success": True,
                        "orchestrator_tools": [
                            "process_query_orchestrator",
                            "get_orchestrator_tools",
                            "coordinate_agents",
                            "comprehensive_analysis"
                        ],
                        "available_agents": list(self.agents.keys()),
                        "total_agents": len(self.agents)
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            # Entity Extraction Tools (2)
            @self.mcp.tool(description="Extract entities from text content")
            async def extract_entities(text: str):
                """Extract entities from text content."""
                try:
                    agent = self.agents["sentiment"]
                    prompt = f"Extract entities from this text: {text}"
                    result = await agent.run(prompt)
                    
                    return {
                        "success": True,
                        "text": text,
                        "entities": result,
                        "total_entities": len(result) if isinstance(result, list) else 0
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            @self.mcp.tool(description="Map relationships between entities")
            async def map_relationships(entities: List[str]):
                """Map relationships between entities."""
                try:
                    agent = self.agents["orchestrator"]
                    prompt = f"Map relationships between these entities: {entities}"
                    result = await agent.run(prompt)
                    
                    return {
                        "success": True,
                        "entities": entities,
                        "relationships": result,
                        "total_relationships": len(result) if isinstance(result, list) else 0
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e)
                    }
            
            # PDF Processing Tools
            @self.mcp.tool(description="Process PDF with enhanced multilingual entity extraction and knowledge graph generation")
            async def process_pdf_enhanced_multilingual(
                pdf_path: str,
                language: str = "auto",
                generate_report: bool = True,
                output_path: str = None
            ):
                """Process PDF file with enhanced multilingual entity extraction and knowledge graph generation.
                
                This tool specifically supports Russian, Chinese, and English PDFs with enhanced entity extraction
                using language-specific patterns, dictionaries, and LLM-based extraction methods.
                """
                try:
                    # Import required modules
                    from src.agents.file_extraction_agent import FileExtractionAgent
                    from src.agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
                    from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
                    from src.core.models import AnalysisRequest, DataType
                    from src.config.language_specific_config import detect_primary_language
                    import os
                    from datetime import datetime
                    
                    # Validate PDF file exists
                    if not os.path.exists(pdf_path):
                        return {
                            "success": False,
                            "error": f"PDF file not found: {pdf_path}",
                            "suggestion": "Please provide a valid path to a PDF file"
                        }
                    
                    # Step 1: Extract text from PDF using enhanced extraction
                    print(f"📄 Extracting text from PDF with enhanced multilingual processing: {pdf_path}")
                    file_agent = EnhancedFileExtractionAgent()
                    
                    pdf_request = AnalysisRequest(
                        data_type=DataType.PDF,
                        content=pdf_path,
                        language=language
                    )
                    
                    extraction_result = await file_agent.process(pdf_request)
                    
                    if extraction_result.status != "completed":
                        return {
                            "success": False,
                            "error": f"PDF extraction failed: {extraction_result.metadata.get('error', 'Unknown error')}",
                            "extraction_status": extraction_result.status
                        }
                    
                    text_content = extraction_result.extracted_text
                    
                    # Step 2: Detect language if auto is specified
                    detected_language = language
                    if language == "auto":
                        detected_language = detect_primary_language(text_content)
                        print(f"🌍 Detected language: {detected_language}")
                    
                    # Step 3: Process with knowledge graph agent using enhanced multilingual support
                    print(f"🧠 Processing with enhanced multilingual entity extraction for language: {detected_language}")
                    kg_agent = KnowledgeGraphAgent()
                    
                    kg_request = AnalysisRequest(
                        data_type=DataType.TEXT,
                        content=text_content,
                        language=detected_language
                    )
                    
                    kg_result = await kg_agent.process(kg_request)
                    
                    # Step 4: Generate report if requested
                    report_files = {}
                    if generate_report:
                        print(f"📊 Generating knowledge graph report...")
                        if not output_path:
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            output_path = f"Results/reports/enhanced_multilingual_pdf_{detected_language}_{timestamp}"
                        
                        report_result = await kg_agent.generate_graph_report(
                            output_path=output_path,
                            target_language=detected_language
                        )
                        
                        if hasattr(report_result, 'success') and report_result.success:
                            report_files = {
                                "html": report_result.metadata.get('html_path', 'Unknown'),
                                "png": report_result.metadata.get('png_path', 'Unknown')
                            }
                    
                    # Step 5: Compile results
                    result = {
                        "success": True,
                        "pdf_path": pdf_path,
                        "detected_language": detected_language,
                        "text_extraction": {
                            "success": True,
                            "content_length": len(text_content),
                            "pages_processed": len(extraction_result.pages) if extraction_result.pages else 'Unknown',
                            "extraction_method": "PyPDF2"
                        },
                        "entity_extraction": {
                            "entities_found": kg_result.metadata.get("statistics", {}).get("entities_found", 0) if kg_result.metadata else 0,
                            "entity_types": kg_result.metadata.get("statistics", {}).get("entity_types", {}) if kg_result.metadata else {},
                            "language_stats": kg_result.metadata.get("statistics", {}).get("language_stats", {}) if kg_result.metadata else {},
                            "extraction_method": "enhanced_multilingual"
                        },
                        "knowledge_graph": {
                            "nodes": kg_result.metadata.get("statistics", {}).get("nodes", 0) if kg_result.metadata else 0,
                            "edges": kg_result.metadata.get("statistics", {}).get("edges", 0) if kg_result.metadata else 0,
                            "communities": kg_result.metadata.get("statistics", {}).get("communities", 0) if kg_result.metadata else 0,
                            "processing_time": kg_result.processing_time
                        },
                        "report_files": report_files,
                        "enhanced_features": {
                            "language_specific_patterns": True,
                            "dictionary_lookup": True,
                            "llm_based_extraction": True,
                            "multilingual_support": ["en", "ru", "zh"]
                        }
                    }
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"❌ Error in process_pdf_enhanced_multilingual: {e}")
                    return {
                        "success": False,
                        "error": str(e),
                        "pdf_path": pdf_path,
                        "language": language
                    }
            
            logger.info("✅ Registered 12 tools with streamable HTTP support")
            
        except Exception as e:
            logger.error(f"❌ Error registering tools: {e}")
    
    def run(self, host: str = "localhost", port: int = 8000, debug: bool = False):
        """Run the MCP server with streamable HTTP support."""
        if self.mcp:
            logger.info(f"🚀 Starting MCP server with streamable HTTP on {host}:{port}")
            return self.mcp.run(transport="streamable-http")
        else:
            logger.error("❌ MCP server not initialized")


# Global MCP server instance
mcp_server = None


def start_mcp_server():
    """Start the unified MCP server with streamable HTTP support."""
    global mcp_server
    
    try:
        # Create the unified MCP server
        mcp_server = OptimizedMCPServer()
        
        if mcp_server.mcp is None:
            logger.warning("⚠️ MCP server not available - skipping MCP server startup")
            return None
        
        # Start the server in a separate thread
        def run_mcp_server():
            try:
                mcp_server.run(host="localhost", port=8000, debug=False)
            except Exception as e:
                logger.error(f"❌ Error starting MCP server: {e}")
        
        mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
        mcp_thread.start()
        
        logger.success("✅ Unified MCP server with streamable HTTP started successfully")
        logger.info(" - MCP Server: http://localhost:8000/mcp")
        logger.info(" - Available agents: sentiment, vision, audio, web, orchestrator")
        
        return mcp_server
        
    except Exception as e:
        logger.warning(f"⚠️ Warning: Could not start MCP server: {e}")
        logger.info(" The application will run without MCP server integration")
        return None


def get_mcp_tools_info():
    """Get information about available MCP tools."""
    try:
        if mcp_server is None or mcp_server.mcp is None:
            logger.warning("⚠️ MCP server not available")
            return []
        
        # Get available tools from the server
        tools = []
        
        # Try different ways to access tools from FastMCP
        if hasattr(mcp_server.mcp, 'tools'):
            tools = list(mcp_server.mcp.tools.keys())
        elif hasattr(mcp_server.mcp, '_tools'):
            tools = list(mcp_server.mcp._tools.keys())
        elif hasattr(mcp_server.mcp, 'app') and hasattr(mcp_server.mcp.app, 'state') and hasattr(mcp_server.mcp.app.state, 'tools'):
            tools = list(mcp_server.mcp.app.state.tools.keys())
        else:
            # If we can't access tools directly, provide a list of known tools
            tools = [
                "get_all_agents_status",
                "start_all_agents", 
                "stop_all_agents",
                "analyze_text_sentiment",
                "analyze_image_sentiment",
                "analyze_audio_sentiment",
                "analyze_webpage_sentiment",
                "process_query_orchestrator",
                "get_orchestrator_tools",
                "extract_entities",
                "map_relationships",
                "process_pdf_enhanced_multilingual"
            ]
        
        logger.info(f"🔧 Available MCP tools: {len(tools)} tools")
        return tools
        
    except Exception as e:
        logger.error(f"⚠️ Could not get MCP tools info: {e}")
        # Return comprehensive tool list as fallback
        return [
            "get_all_agents_status",
            "start_all_agents",
            "stop_all_agents", 
            "analyze_text_sentiment",
            "analyze_image_sentiment",
            "analyze_audio_sentiment",
            "analyze_webpage_sentiment",
            "process_query_orchestrator",
            "get_orchestrator_tools",
            "extract_entities",
            "map_relationships",
            "process_pdf_enhanced_multilingual"
        ]
