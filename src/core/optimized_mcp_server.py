"""
Optimized MCP server with lazy loading for improved startup performance.
This version defers heavy initializations until they're actually needed.
"""

import asyncio
import threading
from typing import Dict, Any, Optional
from loguru import logger

# Import MCP framework
try:
    from fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("⚠️ FastMCP not available")

# Import lazy loading system
from src.core.lazy_loader import service_manager


class OptimizedMCPServer:
    """Optimized MCP server with lazy loading for improved startup performance."""

    def __init__(self):
        # Initialize MCP server immediately (lightweight)
        self.mcp = None
        self._initialize_mcp()
        
        # Register lazy-loaded services
        self._register_lazy_services()
        
        # Start background initialization
        self._start_background_init()
        
        # Register optimized tools
        self._register_optimized_tools()
        
        logger.info("✅ Optimized MCP Server initialized with lazy loading")

    def _initialize_mcp(self):
        """Initialize the MCP server (lightweight operation)."""
        if MCP_AVAILABLE:
            self.mcp = FastMCP("Sentiment Analysis Agents Server")
            logger.info("✅ FastMCP Server initialized successfully")
        else:
            logger.warning("⚠️ FastMCP not available - skipping MCP server initialization")
            self.mcp = None

    def _register_lazy_services(self):
        """Register services for lazy loading."""
        # Register agent factories
        service_manager.register_sync_service("text_agent", self._create_text_agent)
        service_manager.register_sync_service("audio_agent", self._create_audio_agent)
        service_manager.register_sync_service("vision_agent", self._create_vision_agent)
        service_manager.register_sync_service("web_agent", self._create_web_agent)
        service_manager.register_sync_service("ocr_agent", self._create_ocr_agent)
        service_manager.register_sync_service("orchestrator_agent", self._create_orchestrator_agent)
        service_manager.register_sync_service("knowledge_graph_agent", self._create_knowledge_graph_agent)
        service_manager.register_sync_service("file_extraction_agent", self._create_file_extraction_agent)
        
        # Register service factories
        service_manager.register_sync_service("vector_db", self._create_vector_db)
        service_manager.register_sync_service("translation_service", self._create_translation_service)
        service_manager.register_sync_service("improved_knowledge_graph_utility", self._create_improved_knowledge_graph_utility)
        service_manager.register_sync_service("knowledge_graph_integration", self._create_knowledge_graph_integration)
        
        logger.info("✅ Registered lazy-loaded services")

    def _start_background_init(self):
        """Start background initialization of heavy services."""
        # Start background initialization of commonly used services
        heavy_services = [
            "vector_db",
            "translation_service", 
            "knowledge_graph_agent"
        ]
        service_manager.start_background_init(heavy_services)
        logger.info("✅ Started background initialization")

    def _create_text_agent(self):
        """Create text agent (lazy loaded)."""
        from src.agents.unified_text_agent import UnifiedTextAgent
        return UnifiedTextAgent()

    def _create_audio_agent(self):
        """Create audio agent (lazy loaded)."""
        from src.agents.unified_audio_agent import UnifiedAudioAgent
        return UnifiedAudioAgent()

    def _create_vision_agent(self):
        """Create vision agent (lazy loaded)."""
        from src.agents.unified_vision_agent import UnifiedVisionAgent
        return UnifiedVisionAgent()

    def _create_web_agent(self):
        """Create web agent (lazy loaded)."""
        from src.agents.web_agent_enhanced import EnhancedWebAgent
        return EnhancedWebAgent()

    def _create_ocr_agent(self):
        """Create OCR agent (lazy loaded)."""
        from src.agents.ocr_agent import OCRAgent
        return OCRAgent()

    def _create_orchestrator_agent(self):
        """Create orchestrator agent (lazy loaded)."""
        from src.agents.orchestrator_agent import OrchestratorAgent
        return OrchestratorAgent()

    def _create_knowledge_graph_agent(self):
        """Create knowledge graph agent (lazy loaded)."""
        from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
        from src.config.settings import settings
        return KnowledgeGraphAgent(
            graph_storage_path=str(settings.paths.knowledge_graphs_dir)
        )

    def _create_file_extraction_agent(self):
        """Create file extraction agent (lazy loaded)."""
        from src.agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
        return EnhancedFileExtractionAgent()

    def _create_vector_db(self):
        """Create vector database (lazy loaded)."""
        from src.core.vector_db import VectorDBManager
        return VectorDBManager()

    def _create_translation_service(self):
        """Create translation service (lazy loaded)."""
        from src.core.translation_service import TranslationService
        return TranslationService()

    def _create_improved_knowledge_graph_utility(self):
        """Create improved knowledge graph utility (lazy loaded)."""
        from src.core.improved_knowledge_graph_utility import ImprovedKnowledgeGraphUtility
        return ImprovedKnowledgeGraphUtility()

    def _create_knowledge_graph_integration(self):
        """Create knowledge graph integration (lazy loaded)."""
        from src.core.knowledge_graph_integration import KnowledgeGraphIntegration
        return KnowledgeGraphIntegration()

    def get_agent(self, agent_type: str):
        """Get an agent (lazy loaded)."""
        return service_manager.get_sync_service(f"{agent_type}_agent")

    def get_service(self, service_name: str):
        """Get a service (lazy loaded)."""
        return service_manager.get_sync_service(service_name)

    async def get_async_service(self, service_name: str):
        """Get an async service (lazy loaded)."""
        return await service_manager.get_async_service(service_name)

    def _register_optimized_tools(self):
        """Register optimized tools with lazy loading support."""
        if self.mcp is None:
            logger.error("❌ MCP server not initialized")
            return

        try:
            # Core Management Tools
            @self.mcp.tool(description="Get status of all available agents")
            async def get_all_agents_status():
                """Get status of all available agents."""
                try:
                    status = {}
                    agent_types = [
                        "text", "audio", "vision", "web", "ocr", 
                        "orchestrator", "knowledge_graph", "file_extraction"
                    ]
                    
                    for agent_type in agent_types:
                        try:
                            agent = self.get_agent(agent_type)
                            if hasattr(agent, 'get_status'):
                                status[agent_type] = agent.get_status()
                            else:
                                status[agent_type] = {
                                    "agent_id": getattr(agent, 'agent_id', f"{agent_type}_agent"),
                                    "status": "active",
                                    "type": agent.__class__.__name__
                                }
                        except Exception as e:
                            status[agent_type] = {
                                "status": "error",
                                "error": str(e)
                            }
                    
                    # Add service initialization status
                    service_status = service_manager.get_initialization_status()
                    
                    return {
                        "success": True,
                        "total_agents": len(status),
                        "agents": status,
                        "services": service_status
                    }
                except Exception as e:
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Start all agents")
            async def start_all_agents():
                """Start all agents."""
                try:
                    results = {}
                    agent_types = [
                        "text", "audio", "vision", "web", "ocr", 
                        "orchestrator", "knowledge_graph", "file_extraction"
                    ]
                    
                    for agent_type in agent_types:
                        try:
                            agent = self.get_agent(agent_type)
                            if hasattr(agent, 'start'):
                                await agent.start()
                                results[agent_type] = {"success": True, "message": "Started"}
                            else:
                                results[agent_type] = {"success": True, "message": "No start method needed"}
                        except Exception as e:
                            results[agent_type] = {"success": False, "error": str(e)}
                    
                    return {
                        "success": True,
                        "results": results
                    }
                except Exception as e:
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Stop all agents")
            async def stop_all_agents():
                """Stop all agents."""
                try:
                    results = {}
                    agent_types = [
                        "text", "audio", "vision", "web", "ocr", 
                        "orchestrator", "knowledge_graph", "file_extraction"
                    ]
                    
                    for agent_type in agent_types:
                        try:
                            agent = self.get_agent(agent_type)
                            if hasattr(agent, 'stop'):
                                await agent.stop()
                                results[agent_type] = {"success": True, "message": "Stopped"}
                            else:
                                results[agent_type] = {"success": True, "message": "No stop method needed"}
                        except Exception as e:
                            results[agent_type] = {"success": False, "error": str(e)}
                    
                    return {
                        "success": True,
                        "results": results
                    }
                except Exception as e:
                    return {"success": False, "error": str(e)}

            # PDF Processing Tools
            @self.mcp.tool(description="Process PDF with enhanced multilingual support")
            async def process_pdf_enhanced_multilingual(
                pdf_path: str,
                language: str = "auto",
                generate_report: bool = True,
                output_path: str = None
            ):
                """Process PDF with enhanced multilingual support."""
                try:
                    # Get lazy-loaded agents
                    file_agent = self.get_agent("file_extraction")
                    kg_agent = self.get_agent("knowledge_graph")
                    
                    # Process the PDF
                    from src.core.models import AnalysisRequest, DataType
                    
                    # Step 1: Extract text
                    pdf_request = AnalysisRequest(
                        data_type=DataType.PDF,
                        content=pdf_path,
                        language=language
                    )
                    
                    extraction_result = await file_agent.process(pdf_request)
                    
                    if extraction_result.status != "completed":
                        return {
                            "success": False,
                            "error": f"PDF extraction failed: {extraction_result.metadata.get('error', 'Unknown error')}"
                        }
                    
                    # Step 2: Process with knowledge graph
                    kg_request = AnalysisRequest(
                        data_type=DataType.TEXT,
                        content=extraction_result.extracted_text,
                        language=language
                    )
                    
                    kg_result = await kg_agent.process(kg_request)
                    
                    return {
                        "success": True,
                        "extraction_result": extraction_result.metadata,
                        "knowledge_graph_result": kg_result.metadata,
                        "processing_time": kg_result.processing_time
                    }
                    
                except Exception as e:
                    return {"success": False, "error": str(e)}

            # Text Analysis Tools
            @self.mcp.tool(description="Analyze text sentiment")
            async def analyze_text_sentiment(
                text: str,
                language: str = "auto"
            ):
                """Analyze text sentiment."""
                try:
                    text_agent = self.get_agent("text")
                    
                    from src.core.models import AnalysisRequest, DataType
                    request = AnalysisRequest(
                        data_type=DataType.TEXT,
                        content=text,
                        language=language
                    )
                    
                    result = await text_agent.process(request)
                    
                    return {
                        "success": True,
                        "sentiment": result.sentiment,
                        "confidence": result.confidence,
                        "processing_time": result.processing_time
                    }
                    
                except Exception as e:
                    return {"success": False, "error": str(e)}

            # Entity Extraction Tools
            @self.mcp.tool(description="Extract entities from text")
            async def extract_entities(
                text: str,
                language: str = "auto"
            ):
                """Extract entities from text."""
                try:
                    kg_agent = self.get_agent("knowledge_graph")
                    
                    from src.core.models import AnalysisRequest, DataType
                    request = AnalysisRequest(
                        data_type=DataType.TEXT,
                        content=text,
                        language=language
                    )
                    
                    result = await kg_agent.process(request)
                    
                    return {
                        "success": True,
                        "entities": result.metadata.get("entities", []),
                        "entity_count": len(result.metadata.get("entities", [])),
                        "processing_time": result.processing_time
                    }
                    
                except Exception as e:
                    return {"success": False, "error": str(e)}

            # Report Generation Tools
            @self.mcp.tool(description="Generate knowledge graph report")
            async def generate_graph_report(
                output_path: str = None,
                target_language: str = "en"
            ):
                """Generate knowledge graph report."""
                try:
                    kg_utility = self.get_service("improved_knowledge_graph_utility")
                    
                    if not output_path:
                        from datetime import datetime
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        output_path = f"Results/reports/knowledge_graph_report_{target_language}_{timestamp}"
                    
                    report_files = await kg_utility.generate_report(
                        output_path=output_path,
                        target_language=target_language
                    )
                    
                    return {
                        "success": True,
                        "report_files": report_files,
                        "output_path": output_path
                    }
                    
                except Exception as e:
                    return {"success": False, "error": str(e)}

            logger.info("✅ Registered optimized tools with lazy loading")

        except Exception as e:
            logger.error(f"Failed to register tools: {e}")

    def run(self, host: str = "localhost", port: int = 8000, debug: bool = False):
        """Run the MCP server."""
        if self.mcp is None:
            logger.error("❌ MCP server not initialized")
            return
        
        try:
            logger.info(f"🚀 Starting MCP server on {host}:{port}")
            # FastMCP uses different parameters - try the correct method
            if hasattr(self.mcp, 'run_stdio_async'):
                # For stdio mode
                import asyncio
                asyncio.run(self.mcp.run_stdio_async())
            elif hasattr(self.mcp, 'run_http'):
                # For HTTP mode
                self.mcp.run_http(host=host, port=port)
            else:
                # Fallback to default run method
                self.mcp.run()
        except Exception as e:
            logger.error(f"Failed to run MCP server: {e}")

    def get_initialization_status(self) -> Dict[str, str]:
        """Get the initialization status of all services."""
        return service_manager.get_initialization_status()

    async def wait_for_services(self, service_names: list = None, timeout: float = 30.0):
        """Wait for services to be initialized."""
        if service_names is None:
            service_names = [
                "vector_db",
                "translation_service",
                "knowledge_graph_agent"
            ]
        await service_manager.wait_for_services(service_names, timeout)
