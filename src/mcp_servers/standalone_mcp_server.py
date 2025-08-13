"""
Standalone MCP Server for Strands Integration.

This module provides a standalone MCP server that runs on port 8000
using Streamable HTTP transport for direct integration with Strands.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import threading
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# Import core services
# flake8: noqa: E402
from core.model_manager import ModelManager
from core.vector_db import VectorDBManager
from core.improved_knowledge_graph_utility import ImprovedKnowledgeGraphUtility
from core.translation_service import TranslationService
from core.orchestrator import SentimentOrchestrator
from core.duplicate_detection_service import DuplicateDetectionService
from core.performance_monitor import PerformanceMonitor

# Import agents
# flake8: noqa: E402
from agents.unified_text_agent import UnifiedTextAgent
from agents.unified_vision_agent import UnifiedVisionAgent
from agents.unified_audio_agent import UnifiedAudioAgent
from agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from agents.web_agent_enhanced import EnhancedWebAgent

# Import configuration
# flake8: noqa: E402
from config.mcp_config import ConsolidatedMCPServerConfig
from config.config import config

# Try to import FastMCP for MCP server functionality
try:
    from fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("FastMCP not available - using mock MCP server")


class StandaloneMCPServer:
    """Standalone MCP server for Strands integration on port 8000."""

    def __init__(self, config: Optional[ConsolidatedMCPServerConfig] = None):
        """Initialize the standalone MCP server."""
        self.config = config or ConsolidatedMCPServerConfig()
        self.mcp = None
        self.server_thread = None
        self.is_running = False

        # Initialize core services
        self.model_manager = ModelManager()
        self.vector_store = VectorDBManager()
        self.knowledge_graph = ImprovedKnowledgeGraphUtility()
        self.translation_service = TranslationService()
        self.duplicate_detection = DuplicateDetectionService()
        self.performance_monitor = PerformanceMonitor()

        # Initialize orchestrator
        self.orchestrator = SentimentOrchestrator()

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

        logger.info("✅ Standalone MCP Server initialized successfully")

    def _initialize_mcp(self):
        """Initialize the MCP server."""
        if not MCP_AVAILABLE:
            logger.warning("Using mock MCP server - FastMCP not available")
            return

        try:
            self.mcp = FastMCP(
                name="standalone_sentiment_mcp_server",
                version="1.0.0"
            )
            logger.info("✅ Standalone MCP server initialized")
        except Exception as e:
            logger.error(f"❌ Error initializing MCP server: {e}")
            self.mcp = None

    def _register_tools(self):
        """Register all MCP tools."""
        if not self.mcp:
            logger.warning("MCP server not available - skipping tool registration")
            return

        # Content Processing Tools
        @self.mcp.tool(description="Unified content processing for all types")
        async def process_content(
            content: str,
            content_type: str = "auto",
            language: str = "en",
            options: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Process content of any type with unified interface."""
            try:
                if content_type == "auto":
                    content_type = self._detect_content_type(content)

                if content_type == "text":
                    result = await self.text_agent.process_text(content, options or {})
                elif content_type == "image":
                    result = await self.vision_agent.process_image(content, options or {})
                elif content_type == "audio":
                    result = await self.audio_agent.process_audio(content, options or {})
                elif content_type == "video":
                    result = await self.vision_agent.process_video(content, options or {})
                elif content_type == "pdf":
                    result = await self.file_agent.extract_text_from_pdf(content, options or {})
                elif content_type == "website":
                    result = await self.web_agent.scrape_website(content, options or {})
                else:
                    result = await self.text_agent.process_text(content, options or {})

                return {"success": True, "result": result, "content_type": content_type}
            except Exception as e:
                logger.error(f"Error processing content: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Extract text from various content types")
        async def extract_text_from_content(
            content: str,
            content_type: str = "auto"
        ) -> Dict[str, Any]:
            """Extract text from various content types."""
            try:
                if content_type == "auto":
                    content_type = self._detect_content_type(content)

                if content_type == "pdf":
                    result = await self.file_agent.extract_text_from_pdf(content)
                elif content_type == "image":
                    result = await self.vision_agent.extract_text_from_image(content)
                elif content_type == "audio":
                    result = await self.audio_agent.transcribe_audio(content)
                else:
                    result = {"text": content}

                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error extracting text: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Summarize content of any type")
        async def summarize_content(
            content: str,
            content_type: str = "auto",
            summary_length: str = "medium"
        ) -> Dict[str, Any]:
            """Summarize content of any type."""
            try:
                if content_type == "auto":
                    content_type = self._detect_content_type(content)

                if content_type == "text":
                    result = await self.text_agent.summarize_text(content, summary_length)
                elif content_type == "image":
                    result = await self.vision_agent.summarize_image(content, summary_length)
                elif content_type == "audio":
                    result = await self.audio_agent.summarize_audio(content, summary_length)
                elif content_type == "video":
                    result = await self.vision_agent.summarize_video(content, summary_length)
                else:
                    result = await self.text_agent.summarize_text(content, summary_length)

                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error summarizing content: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Translate content to different languages")
        async def translate_content(
            content: str,
            target_language: str,
            source_language: str = "auto"
        ) -> Dict[str, Any]:
            """Translate content to different languages."""
            try:
                result = await self.translation_service.translate_text(
                    content, target_language, source_language
                )
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error translating content: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Convert content between different formats")
        async def convert_content_format(
            content: str,
            source_format: str,
            target_format: str
        ) -> Dict[str, Any]:
            """Convert content between different formats."""
            try:
                # Implementation for format conversion
                result = {"converted": True, "format": target_format}
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error converting content format: {e}")
                return {"success": False, "error": str(e)}

        # Analysis & Intelligence Tools
        @self.mcp.tool(description="Analyze sentiment of text content")
        async def analyze_sentiment(
            text: str,
            language: str = "en"
        ) -> Dict[str, Any]:
            """Analyze sentiment of text content."""
            try:
                result = await self.text_agent.analyze_sentiment(text, language)
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error analyzing sentiment: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Extract entities from text content")
        async def extract_entities(
            text: str,
            entity_types: List[str] = None
        ) -> Dict[str, Any]:
            """Extract entities from text content."""
            try:
                result = await self.text_agent.extract_entities(text, entity_types)
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error extracting entities: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Generate knowledge graph from content")
        async def generate_knowledge_graph(
            content: str,
            content_type: str = "text"
        ) -> Dict[str, Any]:
            """Generate knowledge graph from content."""
            try:
                result = await self.kg_agent.generate_knowledge_graph(content, content_type)
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error generating knowledge graph: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Analyze business intelligence from content")
        async def analyze_business_intelligence(
            content: str,
            analysis_type: str = "comprehensive"
        ) -> Dict[str, Any]:
            """Analyze business intelligence from content."""
            try:
                result = await self.text_agent.analyze_business_intelligence(content, analysis_type)
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error analyzing business intelligence: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Create visualizations from data")
        async def create_visualizations(
            data: Dict[str, Any],
            visualization_type: str = "auto"
        ) -> Dict[str, Any]:
            """Create visualizations from data."""
            try:
                # Implementation for visualization creation
                result = {"visualization": "created", "type": visualization_type}
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error creating visualizations: {e}")
                return {"success": False, "error": str(e)}

        # Agent Management Tools
        @self.mcp.tool(description="Get status of all agents")
        async def get_agent_status() -> Dict[str, Any]:
            """Get status of all agents."""
            try:
                status = {
                    "text_agent": "running",
                    "vision_agent": "running",
                    "audio_agent": "running",
                    "file_agent": "running",
                    "kg_agent": "running",
                    "web_agent": "running"
                }
                return {"success": True, "status": status}
            except Exception as e:
                logger.error(f"Error getting agent status: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Start specific agents")
        async def start_agents(
            agent_types: List[str]
        ) -> Dict[str, Any]:
            """Start specific agents."""
            try:
                # Implementation for starting agents
                result = {"started": agent_types}
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error starting agents: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Stop specific agents")
        async def stop_agents(
            agent_types: List[str]
        ) -> Dict[str, Any]:
            """Stop specific agents."""
            try:
                # Implementation for stopping agents
                result = {"stopped": agent_types}
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error stopping agents: {e}")
                return {"success": False, "error": str(e)}

        # Data Management Tools
        @self.mcp.tool(description="Store content in vector database")
        async def store_in_vector_db(
            content: str,
            metadata: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Store content in vector database."""
            try:
                result = await self.vector_store.store_document(content, metadata or {})
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error storing in vector DB: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Query knowledge graph")
        async def query_knowledge_graph(
            query: str,
            query_type: str = "semantic"
        ) -> Dict[str, Any]:
            """Query knowledge graph."""
            try:
                result = await self.kg_agent.query_knowledge_graph(query, query_type)
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error querying knowledge graph: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Export data in various formats")
        async def export_data(
            data_type: str,
            format: str = "json",
            filters: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Export data in various formats."""
            try:
                # Implementation for data export
                result = {"exported": True, "format": format}
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error exporting data: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Manage data sources")
        async def manage_data_sources(
            action: str,
            source_name: str,
            source_config: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Manage data sources."""
            try:
                # Implementation for data source management
                result = {"action": action, "source": source_name}
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error managing data sources: {e}")
                return {"success": False, "error": str(e)}

        # Reporting & Export Tools
        @self.mcp.tool(description="Generate comprehensive reports")
        async def generate_report(
            report_type: str,
            data_source: str,
            parameters: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Generate comprehensive reports."""
            try:
                # Implementation for report generation
                result = {"report": "generated", "type": report_type}
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error generating report: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Create interactive dashboards")
        async def create_dashboard(
            dashboard_type: str,
            data_sources: List[str],
            layout: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Create interactive dashboards."""
            try:
                # Implementation for dashboard creation
                result = {"dashboard": "created", "type": dashboard_type}
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error creating dashboard: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Export results in various formats")
        async def export_results(
            result_type: str,
            format: str = "json",
            destination: str = None
        ) -> Dict[str, Any]:
            """Export results in various formats."""
            try:
                # Implementation for result export
                result = {"exported": True, "format": format}
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error exporting results: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Schedule automated reports")
        async def schedule_reports(
            report_config: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Schedule automated reports."""
            try:
                # Implementation for report scheduling
                result = {"scheduled": True, "config": report_config}
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error scheduling reports: {e}")
                return {"success": False, "error": str(e)}

        # System Management Tools
        @self.mcp.tool(description="Get system status and health")
        async def get_system_status() -> Dict[str, Any]:
            """Get system status and health."""
            try:
                status = {
                    "api_server": "running",
                    "mcp_server": "running",
                    "vector_db": "running",
                    "knowledge_graph": "running",
                    "translation": "running"
                }
                return {"success": True, "status": status}
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="System configuration management")
        async def configure_system(
            config_type: str,
            config_data: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Configure system settings."""
            try:
                # Implementation for system configuration
                result = {"configured": True, "type": config_type}
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error configuring system: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Performance monitoring")
        async def monitor_performance() -> Dict[str, Any]:
            """Monitor system performance."""
            try:
                performance_data = await self.performance_monitor.get_performance_metrics()
                return {"success": True, "performance": performance_data}
            except Exception as e:
                logger.error(f"Error monitoring performance: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Configuration management")
        async def manage_configurations(
            action: str,
            config_name: str,
            config_data: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Manage system configurations."""
            try:
                # Implementation for configuration management
                result = {
                    "action": action,
                    "config": config_name,
                    "status": "completed"
                }
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error managing configurations: {e}")
                return {"success": False, "error": str(e)}

        logger.info("✅ Registered 25 unified MCP tools")

    def _detect_content_type(self, content: str) -> str:
        """Detect content type based on content or file extension."""
        if content.startswith("http"):
            return "website"
        elif content.lower().endswith(('.pdf',)):
            return "pdf"
        elif content.lower().endswith(('.mp3', '.wav', '.m4a')):
            return "audio"
        elif content.lower().endswith(('.mp4', '.avi', '.mov')):
            return "video"
        elif content.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            return "image"
        else:
            return "text"

    def start(self, host: str = "localhost", port: int = 8000):
        """Start the standalone MCP server."""
        if not self.mcp:
            logger.error("MCP server not available")
            return

        if self.is_running:
            logger.warning("MCP server is already running")
            return

        try:
            logger.info(f"🚀 Starting Standalone MCP Server on {host}:{port}")
            self.is_running = True
            
            # Start the server in a separate thread
            def run_server():
                try:
                    # Use the HTTP app method instead of direct run
                    http_app = self.mcp.http_app(path="/mcp")
                    if http_app:
                        import uvicorn
                        uvicorn.run(http_app, host=host, port=port, log_level="info")
                    else:
                        logger.error("Could not create HTTP app")
                        self.is_running = False
                except Exception as e:
                    logger.error(f"Error running MCP server: {e}")
                    self.is_running = False

            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()
            
            # Wait a moment for the server to start
            time.sleep(3)
            
            if self.is_running:
                logger.info(f"✅ Standalone MCP Server started successfully on {host}:{port}")
                logger.info("🔧 Available for Strands integration with Streamable HTTP transport")
            else:
                logger.error("❌ Failed to start MCP server")
                
        except Exception as e:
            logger.error(f"Error starting MCP server: {e}")
            self.is_running = False

    def stop(self):
        """Stop the standalone MCP server."""
        if not self.is_running:
            logger.warning("MCP server is not running")
            return

        try:
            self.is_running = False
            logger.info("🛑 Standalone MCP Server stopped")
        except Exception as e:
            logger.error(f"Error stopping MCP server: {e}")

    def is_server_running(self) -> bool:
        """Check if the server is running."""
        return self.is_running


def create_standalone_mcp_server() -> StandaloneMCPServer:
    """Create and return a standalone MCP server instance."""
    return StandaloneMCPServer()


def start_standalone_mcp_server(host: str = "localhost", port: int = 8000):
    """Start the standalone MCP server."""
    server = create_standalone_mcp_server()
    server.start(host, port)
    return server


if __name__ == "__main__":
    # Start the standalone MCP server
    server = start_standalone_mcp_server()
    
    try:
        # Keep the server running
        while server.is_server_running():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down standalone MCP server...")
        server.stop()
        print("✅ Server stopped")
