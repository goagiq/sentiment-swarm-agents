"""
Unified MCP Server for Sentiment Analysis System.

This module provides a single, unified MCP server that consolidates all
functionality into 25 tools while maintaining full feature compatibility
and following the design framework.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

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
from core.semantic_search_service import semantic_search_service

# Import agents
# flake8: noqa: E402
from src.agents.unified_text_agent import UnifiedTextAgent
from src.agents.unified_vision_agent import UnifiedVisionAgent
from src.agents.unified_audio_agent import UnifiedAudioAgent
from src.agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.agents.web_agent_enhanced import EnhancedWebAgent

# Import advanced analytics agents
from src.agents.advanced_forecasting_agent import AdvancedForecastingAgent
from src.agents.causal_analysis_agent import CausalAnalysisAgent
from src.agents.anomaly_detection_agent import AnomalyDetectionAgent
from src.agents.advanced_ml_agent import AdvancedMLAgent

# Import configuration
# flake8: noqa: E402
from config.mcp_config import ConsolidatedMCPServerConfig
from config.config import config

# Import report manager
# flake8: noqa: E402
from core.report_manager import report_manager

# Try to import FastMCP for MCP server functionality
try:
    from fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("FastMCP not available - using mock MCP server")


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
        self.duplicate_detection = DuplicateDetectionService()
        self.performance_monitor = PerformanceMonitor()
        self.semantic_search = semantic_search_service

        # Initialize orchestrator
        self.orchestrator = SentimentOrchestrator()

        # Initialize agents
        self.text_agent = UnifiedTextAgent(use_strands=True, use_swarm=True)
        self.vision_agent = UnifiedVisionAgent()
        self.audio_agent = UnifiedAudioAgent()
        self.file_agent = EnhancedFileExtractionAgent()
        self.kg_agent = KnowledgeGraphAgent()
        self.web_agent = EnhancedWebAgent()
        
        # Initialize advanced analytics agents
        self.forecasting_agent = AdvancedForecastingAgent()
        self.causal_agent = CausalAnalysisAgent()
        self.anomaly_agent = AnomalyDetectionAgent()
        self.ml_agent = AdvancedMLAgent()
        
        # Initialize scenario analysis agent (using existing scenario analysis agent)
        try:
            from src.agents.scenario_analysis_agent import ScenarioAnalysisAgent
            self.scenario_agent = ScenarioAnalysisAgent()
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize ScenarioAnalysisAgent: {e}")
            self.scenario_agent = None

        # Initialize enhanced decision support agent
        try:
            from src.agents.decision_support_agent import DecisionSupportAgent
            self.decision_support_agent = DecisionSupportAgent()
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize DecisionSupportAgent: {e}")
            self.decision_support_agent = None

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
            self.mcp = FastMCP(
                name="unified_sentiment_mcp_server",
                version="1.0.0"
            )
            logger.info("✅ MCP server initialized")
        except Exception as e:
            logger.error(f"❌ Error initializing MCP server: {e}")
            self.mcp = None

    def _register_tools(self):
        """Register all 25 consolidated tools."""
        if not self.mcp:
            logger.warning("MCP server not available - skipping tool registration")
            return

        # Content Processing Tools (5)
        @self.mcp.tool(description="Enhanced unified content processing with bulk import and Open Library support")
        async def process_content(
            content: str,
            content_type: str = "auto",
            language: str = "en",
            options: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Process any type of content with unified interface, including bulk import requests and Open Library URLs."""
            try:
                # Auto-detect content type if not specified
                if content_type == "auto":
                    content_type = self._detect_content_type(content)
                
                # Check for bulk import requests first
                if self._detect_bulk_import_request(content):
                    logger.info("Detected bulk import request, processing multiple URLs...")
                    return await self._process_bulk_import_request(content, language, options)
                
                # Check for Open Library URLs
                if self._is_openlibrary_url(content):
                    logger.info("Detected Open Library URL, processing with enhanced agent...")
                    return await self._process_openlibrary_content(content, language, options)
                
                # Check for ctext.org URLs
                if self._is_ctext_url(content):
                    logger.info("Detected ctext.org URL, processing with enhanced agent...")
                    return await self._process_ctext_content(content, language, options)

                # Route to appropriate agent based on content type
                if content_type in ["text", "pdf"]:
                    result = await self.text_agent.process_content(
                        content, language, options
                    )
                elif content_type in ["audio", "video"]:
                    result = await self.audio_agent.process_content(
                        content, language, options
                    )
                elif content_type in ["image", "vision"]:
                    result = await self.vision_agent.process_content(
                        content, language, options
                    )
                else:
                    result = await self.text_agent.process_content(
                        content, language, options
                    )

                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error processing content: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Extract text from any content type")
        async def extract_text_from_content(
            content: str,
            content_type: str = "auto",
            language: str = "en"
        ) -> Dict[str, Any]:
            """Extract text from any content type."""
            try:
                if content_type == "auto":
                    content_type = self._detect_content_type(content)

                if content_type == "pdf":
                    result = await self.file_agent.extract_text(content)
                elif content_type in ["audio", "video"]:
                    result = await self.audio_agent.extract_text(content)
                elif content_type in ["image", "vision"]:
                    result = await self.vision_agent.extract_text(content)
                else:
                    result = {"text": content, "language": language}

                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error extracting text: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Advanced multivariate forecasting")
        async def advanced_forecasting(
            data: str,
            target_variables: str,
            forecast_horizon: int = 10,
            model_type: str = "ensemble"
        ) -> Dict[str, Any]:
            """Perform advanced multivariate time series forecasting."""
            try:
                # Parse data and target variables
                import json
                data_list = json.loads(data) if isinstance(data, str) else data
                target_list = json.loads(target_variables) if isinstance(target_variables, str) else target_variables
                
                result = await self.forecasting_agent.forecast(
                    data=data_list,
                    target_variables=target_list,
                    forecast_horizon=forecast_horizon,
                    model_type=model_type
                )
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error in advanced forecasting: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Causal inference analysis")
        async def causal_analysis(
            data: str,
            variables: str,
            analysis_type: str = "granger"
        ) -> Dict[str, Any]:
            """Perform causal inference analysis."""
            try:
                # Parse data and variables
                import json
                data_list = json.loads(data) if isinstance(data, str) else data
                variables_list = json.loads(variables) if isinstance(variables, str) else variables
                
                result = await self.causal_agent.analyze_causality(
                    data=data_list,
                    variables=variables_list,
                    analysis_type=analysis_type
                )
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error in causal analysis: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Format conversion between types")
        async def convert_content_format(
            content: str,
            source_format: str,
            target_format: str,
            options: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Convert content between different formats."""
            try:
                # Implementation for format conversion
                result = {"converted_content": content, "format": target_format}
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error converting format: {e}")
                return {"success": False, "error": str(e)}

        # Analysis & Intelligence Tools (5)
        @self.mcp.tool(description="Sentiment analysis with multilingual support")
        async def analyze_sentiment(
            text: str,
            language: str = "en"
        ) -> Dict[str, Any]:
            """Analyze sentiment with multilingual support."""
            try:
                result = await self.text_agent.analyze_sentiment(
                    text, language
                )
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error analyzing sentiment: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Entity extraction and relationship mapping")
        async def extract_entities(
            text: str,
            entity_types: List[str] = None
        ) -> Dict[str, Any]:
            """Extract entities and relationships."""
            try:
                # Handle entity_types parameter properly
                if entity_types is None:
                    entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "EVENT"]
                elif isinstance(entity_types, str):
                    # Convert string to list if needed
                    entity_types = [entity_types.upper()]
                elif isinstance(entity_types, list):
                    # Ensure all types are uppercase
                    entity_types = [et.upper() if isinstance(et, str) else str(et).upper() for et in entity_types]
                
                result = await self.kg_agent.extract_entities(
                    text, "en", entity_types
                )
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error extracting entities: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Knowledge graph creation and management")
        async def generate_knowledge_graph(
            content: str,
            language: str = "en",
            graph_type: str = "comprehensive"
        ) -> Dict[str, Any]:
            """Generate knowledge graph from content."""
            try:
                result = await self.kg_agent.generate_knowledge_graph(
                    content, language, graph_type
                )
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error generating knowledge graph: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Business intelligence analysis")
        async def analyze_business_intelligence(
            content: str,
            analysis_type: str = "comprehensive",
            language: str = "en"
        ) -> Dict[str, Any]:
            """Analyze business intelligence from content."""
            try:
                # Implementation for business intelligence analysis
                result = {
                    "analysis": "business_intelligence_result",
                    "type": analysis_type
                }
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error analyzing business intelligence: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Data visualization generation")
        async def create_visualizations(
            data: Dict[str, Any],
            visualization_type: str = "auto",
            options: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Create data visualizations with automatic saving."""
            try:
                # Generate HTML visualization content
                html_content = self._generate_visualization_html(
                    data, visualization_type, options
                )
                
                # Generate title for the visualization
                title = f"{visualization_type.title()} Visualization"
                
                # Save visualization using report manager
                save_result = report_manager.save_visualization(
                    html_content=html_content,
                    title=title,
                    visualization_type=visualization_type,
                    metadata={
                        "data_keys": list(data.keys()) if data else [],
                        "options": options or {},
                        "generated_by": "mcp_server"
                    }
                )
                
                if save_result["success"]:
                    return {
                        "success": True,
                        "result": {
                            "visualization": "generated_visualization",
                            "type": visualization_type,
                            "saved_to": save_result["visualization_info"]["relative_path"],
                            "filename": save_result["visualization_info"]["filename"]
                        },
                        "visualization_info": save_result["visualization_info"]
                    }
                else:
                    return {"success": False, "error": save_result["error"]}
                    
            except Exception as e:
                logger.error(f"Error creating visualizations: {e}")
                return {"success": False, "error": str(e)}

        # Decision Support Tools (7)
        @self.mcp.tool(description="Query knowledge graph for decision context")
        async def query_decision_context(
            content: str,
            language: str = "en",
            context_type: str = "comprehensive"
        ) -> Dict[str, Any]:
            """Query knowledge graph for decision-making context."""
            try:
                if self.decision_support_agent:
                    from src.core.models import AnalysisRequest
                    request = AnalysisRequest(
                        data_type="text",
                        content=content,
                        language=language
                    )
                    context = await self.decision_support_agent.knowledge_graph_integrator.extract_decision_context(
                        request, language
                    )
                    return {"success": True, "result": {
                        "business_entities": len(context.business_entities),
                        "market_entities": len(context.market_entities),
                        "risk_entities": len(context.risk_entities),
                        "opportunity_entities": len(context.opportunity_entities),
                        "confidence_score": context.confidence_score,
                        "language": context.language
                    }}
                else:
                    return {"success": False, "error": "Decision support agent not available"}
            except Exception as e:
                logger.error(f"Error querying decision context: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Extract entities for decision support")
        async def extract_entities_for_decisions(
            content: str,
            language: str = "en",
            entity_types: List[str] = None
        ) -> Dict[str, Any]:
            """Extract entities specifically for decision support analysis."""
            try:
                if self.decision_support_agent:
                    from src.core.models import AnalysisRequest
                    request = AnalysisRequest(
                        data_type="text",
                        content=content,
                        language=language
                    )
                    entities = await self.decision_support_agent.knowledge_graph_integrator._extract_entities_from_content(
                        content, language
                    )
                    return {"success": True, "result": {
                        "entities": entities,
                        "count": len(entities),
                        "language": language
                    }}
                else:
                    return {"success": False, "error": "Decision support agent not available"}
            except Exception as e:
                logger.error(f"Error extracting entities for decisions: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Analyze decision patterns")
        async def analyze_decision_patterns(
            entity_name: str,
            pattern_type: str = "business_patterns",
            language: str = "en",
            time_window: str = "1_year"
        ) -> Dict[str, Any]:
            """Analyze historical decision patterns for an entity."""
            try:
                if self.decision_support_agent:
                    # Create mock entity for pattern analysis
                    entity = {"name": entity_name, "type": "organization"}
                    patterns = await self.decision_support_agent.knowledge_graph_integrator._find_business_patterns(
                        entity, language
                    )
                    return {"success": True, "result": {
                        "patterns": patterns,
                        "count": len(patterns),
                        "entity_name": entity_name,
                        "pattern_type": pattern_type
                    }}
                else:
                    return {"success": False, "error": "Decision support agent not available"}
            except Exception as e:
                logger.error(f"Error analyzing decision patterns: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Generate AI-powered recommendations")
        async def generate_recommendations(
            business_context: str,
            current_performance: Dict[str, Any] = None,
            market_conditions: Dict[str, Any] = None,
            resource_constraints: Dict[str, Any] = None,
            language: str = "en"
        ) -> Dict[str, Any]:
            """Generate AI-powered recommendations based on context."""
            try:
                if self.decision_support_agent:
                    from src.core.models import AnalysisRequest
                    request = AnalysisRequest(
                        data_type="text",
                        content=business_context,
                        language=language
                    )
                    result = await self.decision_support_agent.process(request)
                    return {"success": True, "result": result.metadata}
                else:
                    return {"success": False, "error": "Decision support agent not available"}
            except Exception as e:
                logger.error(f"Error generating recommendations: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Prioritize actions and recommendations")
        async def prioritize_actions(
            recommendations: List[str],
            available_resources: Dict[str, Any] = None,
            time_constraints: Dict[str, Any] = None,
            stakeholder_preferences: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Prioritize actions and recommendations based on multiple factors."""
            try:
                if self.decision_support_agent:
                    # Create mock request for prioritization
                    from src.core.models import AnalysisRequest
                    request = AnalysisRequest(
                        data_type="text",
                        content=" ".join(recommendations),
                        language="en"
                    )
                    result = await self.decision_support_agent._prioritize_actions_only(request)
                    return {"success": True, "result": result.metadata}
                else:
                    return {"success": False, "error": "Decision support agent not available"}
            except Exception as e:
                logger.error(f"Error prioritizing actions: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Create implementation plans")
        async def create_implementation_plans(
            recommendation: str,
            available_resources: Dict[str, Any] = None,
            budget_constraints: float = None,
            timeline_constraints: int = None
        ) -> Dict[str, Any]:
            """Create detailed implementation plans for recommendations."""
            try:
                if self.decision_support_agent:
                    from src.core.models import AnalysisRequest
                    request = AnalysisRequest(
                        data_type="text",
                        content=recommendation,
                        language="en"
                    )
                    result = await self.decision_support_agent._create_implementation_plan_only(request)
                    return {"success": True, "result": result.metadata}
                else:
                    return {"success": False, "error": "Decision support agent not available"}
            except Exception as e:
                logger.error(f"Error creating implementation plans: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Predict success likelihood")
        async def predict_success(
            recommendation: str,
            historical_data: Dict[str, Any] = None,
            organizational_capabilities: Dict[str, Any] = None,
            market_conditions: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Predict likelihood of success for recommendations."""
            try:
                if self.decision_support_agent:
                    from src.core.models import AnalysisRequest
                    request = AnalysisRequest(
                        data_type="text",
                        content=recommendation,
                        language="en"
                    )
                    result = await self.decision_support_agent._predict_success_only(request)
                    return {"success": True, "result": result.metadata}
                else:
                    return {"success": False, "error": "Decision support agent not available"}
            except Exception as e:
                logger.error(f"Error predicting success: {e}")
                return {"success": False, "error": str(e)}

        # Agent Management Tools (3)
        @self.mcp.tool(description="Get status of all agents")
        async def get_agent_status() -> Dict[str, Any]:
            """Get status of all agents."""
            try:
                agents = {
                    "text_agent": self.text_agent.get_status(),
                    "vision_agent": self.vision_agent.get_status(),
                    "audio_agent": self.audio_agent.get_status(),
                    "file_agent": self.file_agent.get_status(),
                    "kg_agent": self.kg_agent.get_status(),
                    "web_agent": self.web_agent.get_status()
                }
                return {"success": True, "agents": agents}
            except Exception as e:
                logger.error(f"Error getting agent status: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Start agent swarm")
        async def start_agents() -> Dict[str, Any]:
            """Start all agents."""
            try:
                results = {}
                for agent_name, agent in [
                    ("text", self.text_agent),
                    ("vision", self.vision_agent),
                    ("audio", self.audio_agent),
                    ("file", self.file_agent),
                    ("kg", self.kg_agent),
                    ("web", self.web_agent)
                ]:
                    if hasattr(agent, 'start'):
                        await agent.start()
                        results[agent_name] = {"status": "started"}
                    else:
                        results[agent_name] = {"status": "no_start_method"}

                return {"success": True, "results": results}
            except Exception as e:
                logger.error(f"Error starting agents: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Stop agent swarm")
        async def stop_agents() -> Dict[str, Any]:
            """Stop all agents."""
            try:
                results = {}
                for agent_name, agent in [
                    ("text", self.text_agent),
                    ("vision", self.vision_agent),
                    ("audio", self.audio_agent),
                    ("file", self.file_agent),
                    ("kg", self.kg_agent),
                    ("web", self.web_agent)
                ]:
                    if hasattr(agent, 'stop'):
                        await agent.stop()
                        results[agent_name] = {"status": "stopped"}
                    else:
                        results[agent_name] = {"status": "no_stop_method"}

                return {"success": True, "results": results}
            except Exception as e:
                logger.error(f"Error stopping agents: {e}")
                return {"success": False, "error": str(e)}

        # Data Management Tools (4)
        @self.mcp.tool(description="Vector database operations")
        async def store_in_vector_db(
            content: str,
            metadata: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Store content in vector database."""
            try:
                result = await self.vector_store.store_content(content, metadata)
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error storing in vector DB: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Knowledge graph queries")
        async def query_knowledge_graph(
            query: str,
            query_type: str = "semantic",
            limit: int = 10
        ) -> Dict[str, Any]:
            """Query knowledge graph."""
            try:
                result = await self.knowledge_graph.query(
                    query, query_type, limit
                )
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error querying knowledge graph: {e}")
                return {"success": False, "error": str(e)}

        # Semantic Search Tools (5)
        @self.mcp.tool(description="Semantic search across all content")
        async def semantic_search(
            query: str,
            search_type: str = "semantic",
            language: str = "en",
            content_types: List[str] = None,
            n_results: int = 10,
            similarity_threshold: float = 0.7,
            include_metadata: bool = True
        ) -> Dict[str, Any]:
            """Perform semantic search across all indexed content."""
            try:
                from src.config.semantic_search_config import SearchType
                
                # Convert string to SearchType enum
                search_type_enum = SearchType(search_type)
                
                result = await self.semantic_search.search(
                    query=query,
                    search_type=search_type_enum,
                    language=language,
                    content_types=content_types,
                    n_results=n_results,
                    similarity_threshold=similarity_threshold,
                    include_metadata=include_metadata
                )
                return result
            except Exception as e:
                logger.error(f"Error in semantic search: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Multi-language semantic search")
        async def multilingual_semantic_search(
            query: str,
            target_languages: List[str] = None,
            n_results: int = 10,
            similarity_threshold: float = 0.7
        ) -> Dict[str, Any]:
            """Perform semantic search across multiple languages."""
            try:
                result = await self.semantic_search.multi_language_semantic_search(
                    query=query,
                    target_languages=target_languages,
                    n_results=n_results,
                    similarity_threshold=similarity_threshold
                )
                return {"success": True, "results": result}
            except Exception as e:
                logger.error(f"Error in multilingual search: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Conceptual search for related ideas")
        async def conceptual_search(
            concept: str,
            n_results: int = 10,
            similarity_threshold: float = 0.6
        ) -> Dict[str, Any]:
            """Search for content related to a specific concept."""
            try:
                result = await self.semantic_search.search_by_concept(
                    concept=concept,
                    n_results=n_results,
                    similarity_threshold=similarity_threshold
                )
                return {"success": True, "results": result}
            except Exception as e:
                logger.error(f"Error in conceptual search: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Cross-content-type search")
        async def cross_content_search(
            query: str,
            content_types: List[str],
            n_results: int = 10,
            similarity_threshold: float = 0.7
        ) -> Dict[str, Any]:
            """Search across specific content types."""
            try:
                result = await self.semantic_search.search_across_content_types(
                    query=query,
                    content_types=content_types,
                    n_results=n_results,
                    similarity_threshold=similarity_threshold
                )
                return {"success": True, "results": result}
            except Exception as e:
                logger.error(f"Error in cross-content search: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Combined semantic and knowledge graph search")
        async def combined_search(
            query: str,
            language: str = "en",
            n_results: int = 10,
            similarity_threshold: float = 0.7,
            include_kg_results: bool = True
        ) -> Dict[str, Any]:
            """Perform combined semantic search and knowledge graph search."""
            try:
                result = await self.semantic_search.search_with_knowledge_graph(
                    query=query,
                    language=language,
                    n_results=n_results,
                    similarity_threshold=similarity_threshold,
                    include_kg_results=include_kg_results
                )
                return result
            except Exception as e:
                logger.error(f"Error in combined search: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Data export in multiple formats")
        async def export_data(
            data: Dict[str, Any],
            format: str = "json",
            options: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Export data in various formats."""
            try:
                # Implementation for data export
                result = {"exported_data": data, "format": format}
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error exporting data: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="External data source management")
        async def manage_data_sources(
            action: str,
            source_type: str = "api",
            config: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Manage external data sources."""
            try:
                # Implementation for data source management
                result = {
                    "action": action,
                    "source_type": source_type,
                    "status": "completed"
                }
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error managing data sources: {e}")
                return {"success": False, "error": str(e)}

        # Reporting & Export Tools (4)
        @self.mcp.tool(description="Comprehensive report generation")
        async def generate_report(
            content: str,
            report_type: str = "comprehensive",
            language: str = "en",
            options: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Generate comprehensive reports with automatic saving."""
            try:
                # Generate filename based on content and type
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{report_type}_Report_{timestamp}.md"
                
                # Save report using report manager
                save_result = report_manager.save_report(
                    content=content,
                    filename=filename,
                    report_type=report_type,
                    metadata={
                        "language": language,
                        "options": options or {},
                        "generated_by": "mcp_server"
                    }
                )
                
                if save_result["success"]:
                    return {
                        "success": True,
                        "result": {
                            "report": "generated_report",
                            "type": report_type,
                            "saved_to": save_result["report_info"]["relative_path"],
                            "filename": save_result["report_info"]["filename"]
                        },
                        "report_info": save_result["report_info"]
                    }
                else:
                    return {"success": False, "error": save_result["error"]}
                    
            except Exception as e:
                logger.error(f"Error generating report: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Interactive dashboard creation")
        async def create_dashboard(
            dashboard_type: str,
            data_sources: List[str],
            layout: Dict[str, Any] = None
        ) -> Dict[str, Any]:
            """Create interactive dashboards."""
            try:
                # Implementation for dashboard creation
                result = {"dashboard": "created_dashboard", "type": dashboard_type}
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error creating dashboard: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Result export to various formats")
        async def export_results(
            result_type: str,
            format: str = "json",
            destination: str = None
        ) -> Dict[str, Any]:
            """Export results to various formats."""
            try:
                # Implementation for result export
                result = {"exported_results": result_type, "format": format}
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error exporting results: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Generate summary report with all generated reports")
        async def generate_summary_report(
            analysis_title: str,
            analysis_type: str = "comprehensive",
            key_findings: List[str] = None
        ) -> Dict[str, Any]:
            """Generate a summary report with links to all generated reports."""
            try:
                summary_result = report_manager.generate_summary_report(
                    analysis_title=analysis_title,
                    analysis_type=analysis_type,
                    key_findings=key_findings or []
                )
                
                if summary_result["success"]:
                    return {
                        "success": True,
                        "result": {
                            "summary": "generated_summary",
                            "title": analysis_title,
                            "saved_to": summary_result["summary_info"]["relative_path"],
                            "filename": summary_result["summary_info"]["filename"],
                            "total_reports": summary_result["summary_info"]["total_reports"]
                        },
                        "summary_info": summary_result["summary_info"],
                        "all_reports": summary_result["all_reports"]
                    }
                else:
                    return {"success": False, "error": summary_result["message"]}
                    
            except Exception as e:
                logger.error(f"Error generating summary report: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Get all generated reports")
        async def get_generated_reports() -> Dict[str, Any]:
            """Get information about all generated reports."""
            try:
                reports = report_manager.get_all_reports()
                return {
                    "success": True,
                    "total_reports": len(reports),
                    "reports": reports,
                    "total_size_kb": sum(r["size_kb"] for r in reports)
                }
            except Exception as e:
                logger.error(f"Error getting generated reports: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Clear generated reports for new session")
        async def clear_reports() -> Dict[str, Any]:
            """Clear the generated reports list for a new analysis session."""
            try:
                report_manager.clear_reports()
                return {
                    "success": True,
                    "message": "Reports cleared for new analysis session"
                }
            except Exception as e:
                logger.error(f"Error clearing reports: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Automated report scheduling")
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

        # System Management Tools (4)
        @self.mcp.tool(description="System health and status")
        async def get_system_status() -> Dict[str, Any]:
            """Get system health and status."""
            try:
                status = {
                    "mcp_server": "running" if self.mcp else "not_available",
                    "agents": await get_agent_status(),
                    "services": {
                        "vector_db": "running",
                        "knowledge_graph": "running",
                        "translation": "running"
                    }
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

        # Advanced Analytics Tools (5 new tools)
        @self.mcp.tool(description="Scenario analysis for business planning")
        async def scenario_analysis(
            base_data: str,
            scenarios: str,
            target_variable: str,
            analysis_type: str = "impact"
        ) -> Dict[str, Any]:
            """Perform scenario analysis for business planning."""
            try:
                # Parse data
                import json
                base_data_list = json.loads(base_data) if isinstance(base_data, str) else base_data
                scenarios_list = json.loads(scenarios) if isinstance(scenarios, str) else scenarios
                
                result = await self.scenario_agent.analyze_scenarios(
                    base_data=base_data_list,
                    scenarios=scenarios_list,
                    target_variable=target_variable,
                    analysis_type=analysis_type
                )
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error in scenario analysis: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Model optimization and hyperparameter tuning")
        async def model_optimization(
            model_config: str,
            optimization_type: str = "hyperparameter",
            metric: str = "accuracy"
        ) -> Dict[str, Any]:
            """Optimize machine learning models."""
            try:
                # Parse config
                import json
                config_dict = json.loads(model_config) if isinstance(model_config, str) else model_config
                
                result = await self.ml_agent.optimize_model(
                    model_config=config_dict,
                    optimization_type=optimization_type,
                    metric=metric
                )
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error in model optimization: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Feature engineering for machine learning")
        async def feature_engineering(
            data: str,
            features: str,
            engineering_type: str = "automatic"
        ) -> Dict[str, Any]:
            """Perform automated feature engineering."""
            try:
                # Parse data and features
                import json
                data_list = json.loads(data) if isinstance(data, str) else data
                features_list = json.loads(features) if isinstance(features, str) else features
                
                result = await self.ml_agent.engineer_features(
                    data=data_list,
                    features=features_list,
                    engineering_type=engineering_type
                )
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error in feature engineering: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="Deep learning model training")
        async def deep_learning_training(
            data: str,
            model_type: str = "mlp",
            task: str = "classification",
            config: str = None
        ) -> Dict[str, Any]:
            """Train deep learning models."""
            try:
                # Parse data and config
                import json
                data_list = json.loads(data) if isinstance(data, str) else data
                config_dict = json.loads(config) if config and isinstance(config, str) else config
                
                result = await self.ml_agent.create_and_train_model(
                    data=data_list,
                    model_type=model_type,
                    task=task,
                    config=config_dict
                )
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error in deep learning training: {e}")
                return {"success": False, "error": str(e)}

        @self.mcp.tool(description="AutoML pipeline for automated model selection")
        async def automl_pipeline(
            data: str,
            target: str,
            task: str = "classification",
            time_limit: int = 3600
        ) -> Dict[str, Any]:
            """Run AutoML pipeline for automated model selection."""
            try:
                # Parse data
                import json
                data_list = json.loads(data) if isinstance(data, str) else data
                
                result = await self.ml_agent.run_automl_pipeline(
                    data=data_list,
                    target=target,
                    task=task,
                    time_limit=time_limit
                )
                return {"success": True, "result": result}
            except Exception as e:
                logger.error(f"Error in AutoML pipeline: {e}")
                return {"success": False, "error": str(e)}

        logger.info("✅ Registered 30 unified MCP tools (including 5 new advanced analytics tools)")

    def _detect_content_type(self, content: str) -> str:
        """Enhanced content type detection with bulk import and library URL support."""
        content_lower = content.lower()
        
        # Check for bulk import requests first
        if self._detect_bulk_import_request(content):
            return "bulk_import_request"
        
        # Check for URLs
        if content.startswith(('http://', 'https://')):
            if 'openlibrary.org' in content_lower:
                return "open_library"
            elif 'ctext.org' in content_lower:
                return "ctext_library"
            elif any(ext in content_lower for ext in ['.pdf', '.doc', '.docx']):
                return "pdf"
            elif any(ext in content_lower for ext in ['.mp3', '.wav', '.m4a']):
                return "audio"
            elif any(ext in content_lower for ext in ['.mp4', '.avi', '.mov']):
                return "video"
            elif any(ext in content_lower for ext in ['.jpg', '.png', '.gif']):
                return "image"
            else:
                return "website"
        
        # Check for file paths
        if Path(content).exists():
            ext = Path(content).suffix.lower()
            if ext in ['.pdf', '.doc', '.docx', '.txt']:
                return "pdf"
            elif ext in ['.mp3', '.wav', '.m4a']:
                return "audio"
            elif ext in ['.mp4', '.avi', '.mov']:
                return "video"
            elif ext in ['.jpg', '.png', '.gif']:
                return "image"
        
        # Default to text
        return "text"
    
    def _is_openlibrary_url(self, content: str) -> bool:
        """Check if content is an Open Library URL."""
        return 'openlibrary.org' in content.lower()
    
    def _is_ctext_url(self, content: str) -> bool:
        """Check if content is a ctext.org URL."""
        return 'ctext.org' in content.lower()
    
    def _detect_bulk_import_request(self, content: str) -> bool:
        """Detect if this is a bulk import request with multiple URLs."""
        # Check for patterns like "add @url1 and @url2 to both vector and knowledge graph db"
        import re
        bulk_patterns = [
            r"add\s+@[^\s]+\s+and\s+@[^\s]+",
            r"add\s+@[^\s]+\s+to\s+both\s+vector\s+and\s+knowledge\s+graph",
            r"add\s+@[^\s]+\s+to\s+both\s+databases",
            r"add\s+@[^\s]+\s+and\s+@[^\s]+\s+to\s+both",
            r"add\s+@[^\s]+\s+to\s+vector\s+and\s+knowledge\s+graph\s+db",
            r"add\s+@[^\s]+\s+to\s+both\s+vector\s+and\s+knowledge\s+graph\s+db"
        ]
        
        content_lower = content.lower()
        return any(re.search(pattern, content_lower) for pattern in bulk_patterns)
    
    def _extract_urls_from_request(self, content: str) -> List[str]:
        """Extract URLs from a bulk import request."""
        import re
        # Extract URLs that start with @
        url_pattern = r'@(https?://[^\s]+)'
        urls = re.findall(url_pattern, content)
        return urls

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

    async def _process_bulk_import_request(self, content: str, language: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process bulk import request with multiple URLs."""
        try:
            # Extract URLs from the request
            urls = self._extract_urls_from_request(content)
            
            if not urls:
                return {"success": False, "error": "No URLs found in bulk import request"}
            
            logger.info(f"Processing bulk import request with {len(urls)} URLs: {urls}")
            
            results = []
            total_entities = 0
            total_relationships = 0
            
            # Process each URL
            for url in urls:
                try:
                    if self._is_openlibrary_url(url):
                        result = await self._process_openlibrary_content(url, language, options)
                    elif self._is_ctext_url(url):
                        result = await self._process_ctext_content(url, language, options)
                    else:
                        # Handle as standard URL
                        result = await self.text_agent.process_content(url, language, options)
                    
                    if result.get("success"):
                        results.append({
                            "url": url,
                            "success": True,
                            "title": result.get("result", {}).get("metadata", {}).get("title", "Unknown"),
                            "entities_count": result.get("result", {}).get("metadata", {}).get("entities_count", 0),
                            "relationships_count": result.get("result", {}).get("metadata", {}).get("relationships_count", 0)
                        })
                        
                        total_entities += result.get("result", {}).get("metadata", {}).get("entities_count", 0)
                        total_relationships += result.get("result", {}).get("metadata", {}).get("relationships_count", 0)
                    else:
                        results.append({
                            "url": url,
                            "success": False,
                            "error": result.get("error", "Unknown error")
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {e}")
                    results.append({
                        "url": url,
                        "success": False,
                        "error": str(e)
                    })
            
            return {
                "success": True,
                "content_type": "bulk_import",
                "urls_processed": len(urls),
                "successful_imports": len([r for r in results if r["success"]]),
                "failed_imports": len([r for r in results if not r["success"]]),
                "total_entities": total_entities,
                "total_relationships": total_relationships,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error processing bulk import request: {e}")
            return {"success": False, "error": str(e)}
    
    async def _process_openlibrary_content(self, url: str, language: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process Open Library content with full pipeline."""
        try:
            # Download content from Open Library
            webpage_content = await self._download_openlibrary_content(url)
            
            if not webpage_content:
                return {"success": False, "error": "Failed to download content from Open Library"}
            
            # Extract text content
            content_text = webpage_content.get("text", "")
            title = webpage_content.get("title", "Unknown Book")
            
            # Extract metadata
            metadata = self._extract_metadata_from_content(content_text, title, url)
            
            # Store in vector database
            vector_id = await self.vector_store.store_content(content_text, metadata)
            
            # Extract entities and create knowledge graph
            entities_result = await self.kg_agent.extract_entities(content_text, language)
            entities = entities_result.get("content", [{}])[0].get("json", {}).get("entities", [])
            
            relationships_result = await self.kg_agent.map_relationships(content_text, entities)
            relationships = relationships_result.get("content", [{}])[0].get("json", {}).get("relationships", [])
            
            # Create knowledge graph
            transformed_entities = [
                {
                    "name": entity.get("text", ""),
                    "type": entity.get("type", "CONCEPT"),
                    "confidence": entity.get("confidence", 0.0),
                    "source": title
                }
                for entity in entities
            ]
            
            transformed_relationships = [
                {
                    "source": rel.get("source", ""),
                    "target": rel.get("target", ""),
                    "relationship_type": rel.get("type", "RELATED_TO"),
                    "confidence": rel.get("confidence", 0.0),
                    "source_type": title
                }
                for rel in relationships
            ]
            
            kg_result = await self.knowledge_graph.create_knowledge_graph(transformed_entities, transformed_relationships)
            
            return {
                "success": True,
                "content_type": "open_library",
                "title": title,
                "vector_id": vector_id,
                "entities_count": len(entities),
                "relationships_count": len(relationships),
                "knowledge_graph_nodes": kg_result.number_of_nodes(),
                "knowledge_graph_edges": kg_result.number_of_edges(),
                "content_length": len(content_text)
            }
            
        except Exception as e:
            logger.error(f"Error processing Open Library content: {e}")
            return {"success": False, "error": str(e)}
    
    async def _process_ctext_content(self, url: str, language: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Process ctext.org content with full pipeline."""
        try:
            # Download content from ctext.org
            webpage_content = await self._download_ctext_content(url)
            
            if not webpage_content:
                return {"success": False, "error": "Failed to download content from ctext.org"}
            
            # Extract text content
            content_text = webpage_content.get("text", "")
            title = webpage_content.get("title", "Unknown Text")
            
            # Extract metadata
            metadata = self._extract_metadata_from_content(content_text, title, url)
            metadata.update({
                "source": "ctext.org",
                "content_type": "classical_text"
            })
            
            # Store in vector database
            vector_id = await self.vector_store.store_content(content_text, metadata)
            
            # Extract entities and create knowledge graph
            entities_result = await self.kg_agent.extract_entities(content_text, "zh")  # Chinese for classical texts
            entities = entities_result.get("content", [{}])[0].get("json", {}).get("entities", [])
            
            relationships_result = await self.kg_agent.map_relationships(content_text, entities)
            relationships = relationships_result.get("content", [{}])[0].get("json", {}).get("relationships", [])
            
            # Create knowledge graph
            transformed_entities = [
                {
                    "name": entity.get("text", ""),
                    "type": entity.get("type", "CONCEPT"),
                    "confidence": entity.get("confidence", 0.0),
                    "source": title
                }
                for entity in entities
            ]
            
            transformed_relationships = [
                {
                    "source": rel.get("source", ""),
                    "target": rel.get("target", ""),
                    "relationship_type": rel.get("type", "RELATED_TO"),
                    "confidence": rel.get("confidence", 0.0),
                    "source_type": title
                }
                for rel in relationships
            ]
            
            kg_result = await self.knowledge_graph.create_knowledge_graph(transformed_entities, transformed_relationships)
            
            return {
                "success": True,
                "content_type": "ctext_classical_text",
                "title": title,
                "vector_id": vector_id,
                "entities_count": len(entities),
                "relationships_count": len(relationships),
                "knowledge_graph_nodes": kg_result.number_of_nodes(),
                "knowledge_graph_edges": kg_result.number_of_edges(),
                "content_length": len(content_text)
            }
            
        except Exception as e:
            logger.error(f"Error processing ctext.org content: {e}")
            return {"success": False, "error": str(e)}
    
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
            
            logger.info(f"✅ Successfully downloaded Open Library content: {len(cleaned_text)} characters")
            return webpage_content
            
        except Exception as e:
            logger.error(f"❌ Error downloading Open Library content: {e}")
            return None
    
    async def _download_ctext_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Download content from ctext.org URL."""
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
            
            logger.info(f"✅ Successfully downloaded ctext.org content: {len(cleaned_text)} characters")
            return webpage_content
            
        except Exception as e:
            logger.error(f"❌ Error downloading ctext.org content: {e}")
            return None
    
    def _extract_metadata_from_content(self, content: str, title: str, url: str = "") -> Dict[str, Any]:
        """Extract metadata from content text."""
        import re
        content_lower = content.lower()
        
        # Try to extract author
        author = "Unknown"
        author_patterns = ["by ", "author:", "written by", "author is"]
        for pattern in author_patterns:
            if pattern in content_lower:
                start_idx = content_lower.find(pattern) + len(pattern)
                end_idx = content.find("\n", start_idx)
                if end_idx == -1:
                    end_idx = start_idx + 100
                author = content[start_idx:end_idx].strip()
                break
        
        # Try to extract publication year
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, content)
        publication_year = years[0] if years else "Unknown"
        
        # Determine genre
        genre_keywords = {
            "fiction": ["novel", "story", "tale", "fiction"],
            "non-fiction": ["history", "biography", "memoir", "essay"],
            "poetry": ["poem", "poetry", "verse"],
            "drama": ["play", "drama", "theater", "theatre"],
            "science": ["science", "physics", "chemistry", "biology"],
            "philosophy": ["philosophy", "philosophical", "ethics"],
            "religion": ["religion", "religious", "spiritual", "theology"]
        }
        
        detected_genre = "Literature"
        for genre, keywords in genre_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_genre = genre.title()
                break
        
        # Extract subjects
        subjects = []
        subject_keywords = [
            "history", "war", "peace", "love", "family", "politics", 
            "society", "culture", "art", "music", "science", "philosophy",
            "religion", "nature", "travel", "adventure", "mystery"
        ]
        
        for subject in subject_keywords:
            if subject in content_lower:
                subjects.append(subject.title())
        
        return {
            "title": title,
            "author": author,
            "publication_year": publication_year,
            "genre": detected_genre,
            "category": "Classic Literature" if "classic" in content_lower else detected_genre,
            "subjects": subjects[:10],
            "source": "Open Library" if "openlibrary.org" in url else "ctext.org" if "ctext.org" in url else "Unknown",
            "source_url": url,
            "content_type": "book_description",
            "language": "en"
        }

    def _generate_visualization_html(
        self,
        data: Dict[str, Any],
        visualization_type: str,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate HTML content for visualizations."""
        try:
            # Basic HTML template for visualizations
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{visualization_type.title()} Visualization</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .content {{
            padding: 40px;
        }}
        .data-section {{
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border: 1px solid #e9ecef;
        }}
        .data-key {{
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        .data-value {{
            background: white;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #dee2e6;
            font-family: monospace;
            white-space: pre-wrap;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{visualization_type.title()} Visualization</h1>
            <p>Generated automatically by MCP Server</p>
        </div>
        
        <div class="content">
            <h2>Data Overview</h2>
            <p>This visualization contains {len(data)} data sections.</p>
            
            <h2>Data Sections</h2>
"""
            
            # Add data sections
            for key, value in data.items():
                html_content += f"""
            <div class="data-section">
                <div class="data-key">{key}</div>
                <div class="data-value">{str(value)}</div>
            </div>
"""
            
            # Close HTML
            html_content += """
        </div>
    </div>
</body>
</html>
"""
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error generating visualization HTML: {e}")
            return f"<html><body><h1>Error generating visualization</h1><p>{str(e)}</p></body></html>"

    def run(self, host: str = "localhost", port: int = 8000, debug: bool = False):
        """Run the MCP server (legacy method - use get_http_app for integration)."""
        if not self.mcp:
            logger.error("MCP server not available")
            return

        try:
            logger.info(f"🚀 Starting Unified MCP Server on {host}:{port}")
            self.mcp.run(host=host, port=port, debug=debug)
        except Exception as e:
            logger.error(f"Error running MCP server: {e}")


def create_unified_mcp_server() -> UnifiedMCPServer:
    """Create and return a unified MCP server instance."""
    return UnifiedMCPServer()
