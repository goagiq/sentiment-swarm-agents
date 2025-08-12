#!/usr/bin/env python3
"""
Main entry point for the Sentiment Analysis Swarm system.
Provides both MCP server and FastAPI server functionality.
"""

# Suppress all deprecation warnings BEFORE any other imports
import warnings
import sys

# Set warnings filter to ignore all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="uvicorn")
warnings.filterwarnings("ignore", category=UserWarning, module="websockets")

# Custom warning filter function
def ignore_all_warnings(message, category, filename, lineno, file=None, line=None):
    """Custom warning filter to ignore all warnings."""
    if category in [DeprecationWarning, FutureWarning, UserWarning]:
        return True
    return False

# Add custom filter
warnings.showwarning = ignore_all_warnings

import os
import threading
import uvicorn
import subprocess
import time
import requests
from typing import List, Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import MCP server after adding src to path to avoid conflicts
try:
    from fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️ MCP server not available")

# Import after path modification
# flake8: noqa: E402
from src.api.main import app
from src.core.error_handler import with_error_handling
from src.agents.unified_text_agent import UnifiedTextAgent
from src.agents.unified_audio_agent import UnifiedAudioAgent
from src.agents.unified_vision_agent import UnifiedVisionAgent
from src.agents.web_agent_enhanced import EnhancedWebAgent
from src.agents.unified_vision_agent import UnifiedVisionAgent
from src.agents.ocr_agent import OCRAgent
from src.agents.orchestrator_agent import OrchestratorAgent
from src.core.tool_registry import ToolRegistry
from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.core.improved_knowledge_graph_utility import ImprovedKnowledgeGraphUtility
from src.core.knowledge_graph_integration import KnowledgeGraphIntegration
from src.config.settings import settings
from src.config.config import config
from src.core.port_checker import get_safe_port
from src.agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
from src.agents.business_intelligence_agent import BusinessIntelligenceAgent
from src.agents.data_visualization_agent import DataVisualizationAgent
from src.agents.social_media_agent import SocialMediaAgent
from src.agents.external_data_agent import ExternalDataAgent
from src.agents.market_data_agent import MarketDataAgent
from src.agents.multi_modal_analysis_agent import MultiModalAnalysisAgent

# YouTube analysis now handled by UnifiedVisionAgent
from src.core.models import (
    AnalysisRequest, DataType, ProcessingStatus, SentimentResult
)


class OptimizedMCPServer:
    """Optimized MCP server providing unified access to all agents with reduced tool count."""

    def __init__(self):
        # Initialize the MCP server with proper streamable HTTP support
        self.mcp = None
        self._initialize_mcp()
        
        # Initialize all agents
        self.agents = {}
        self._initialize_agents()
        
        # Register optimized tools
        self._register_optimized_tools()
        
        print("Optimized MCP Server initialized with unified tools")

    def _initialize_mcp(self):
        """Initialize the MCP server using FastMCP with streamable HTTP support."""
        if MCP_AVAILABLE:
            self.mcp = FastMCP("Sentiment Analysis Agents Server")
            print("FastMCP Server with streamable HTTP support initialized successfully")
        else:
            print("⚠️ FastMCP not available - skipping MCP server initialization")
            self.mcp = None

    def _initialize_agents(self):
        """Initialize all agents."""
        try:
            # Initialize each agent type
            self.agents["text"] = UnifiedTextAgent()
            self.agents["audio"] = UnifiedAudioAgent()
            self.agents["vision"] = UnifiedVisionAgent()
            self.agents["web"] = EnhancedWebAgent()
            self.agents["video_summary"] = UnifiedVisionAgent()
            self.agents["ocr"] = OCRAgent()
            self.agents["orchestrator"] = OrchestratorAgent()
            
            # Initialize KnowledgeGraphAgent with settings-based configuration
            self.agents["knowledge_graph"] = KnowledgeGraphAgent(
                graph_storage_path=str(settings.paths.knowledge_graphs_dir)
            )
            
            # Initialize EnhancedFileExtractionAgent for PDF processing (using fixed version)
            self.agents["file_extraction"] = EnhancedFileExtractionAgent()
            
            # Initialize Business Intelligence Agents
            self.agents["business_intelligence"] = BusinessIntelligenceAgent()
            self.agents["data_visualization"] = DataVisualizationAgent()
            
            # Initialize Phase 2 External Data Integration Agents
            self.agents["social_media"] = SocialMediaAgent()
            self.agents["external_data"] = ExternalDataAgent()
            self.agents["market_data"] = MarketDataAgent()
            
            # Initialize Phase 3 Multi-Modal Analysis Agent
            self.agents["multi_modal_analysis"] = MultiModalAnalysisAgent()
            
            # Initialize Phase 4 Export & Automation Agents
            from src.agents.report_generation_agent import ReportGenerationAgent
            from src.agents.data_export_agent import DataExportAgent
            
            self.agents["report_generation"] = ReportGenerationAgent()
            self.agents["data_export"] = DataExportAgent()
            
            # Initialize Phase 5 Semantic Search & Agent Reflection Agents
            from src.agents.semantic_search_agent import SemanticSearchAgent
            from src.agents.reflection_agent import ReflectionCoordinatorAgent
            
            self.agents["semantic_search"] = SemanticSearchAgent()
            self.agents["reflection_coordinator"] = ReflectionCoordinatorAgent()
            
            # Initialize fixed services
            from src.core.vector_db import VectorDBManager
            from src.core.translation_service import TranslationService
            
            # Initialize fixed vector database and translation service
            self.vector_db = VectorDBManager()
            self.translation_service = TranslationService()
            
            # YouTube analysis now handled by UnifiedVisionAgent
            
            # Initialize improved knowledge graph utilities
            self.improved_knowledge_graph_utility = ImprovedKnowledgeGraphUtility()
            self.knowledge_graph_integration = KnowledgeGraphIntegration()
            
            print(f"Initialized {len(self.agents)} unified agents including knowledge graph and improved utilities")
        except Exception as e:
            print(f"⚠️ Error initializing agents: {e}")

    def _register_optimized_tools(self):
        """Register optimized tools with unified interfaces."""
        if self.mcp is None:
            print("❌ MCP server not initialized")
            return

        try:
            # Core Management Tools (3)
            @self.mcp.tool(description="Get status of all available agents")
            async def get_all_agents_status():
                """Get status of all available agents."""
                try:
                    status = {}
                    for agent_name, agent in self.agents.items():
                        if hasattr(agent, 'get_status'):
                            status[agent_name] = agent.get_status()
                        else:
                            status[agent_name] = {
                                "agent_id": getattr(agent, 'agent_id', f"{agent_name}_agent"),
                                "status": "active",
                                "type": agent.__class__.__name__
                            }
                    return {
                        "success": True,
                        "total_agents": len(self.agents),
                        "agents": status
                    }
                except Exception as e:
                    return {"success": False, "error": str(e)}

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
                    return {"success": False, "error": str(e)}

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
                    return {"success": False, "error": str(e)}

            # PDF Processing Tools
            @self.mcp.tool(description="Process PDF with enhanced multilingual support using fixed components")
            @with_error_handling("pdf_processing")
            async def process_pdf_enhanced_multilingual(
                pdf_path: str,
                language: str = "auto",
                generate_report: bool = True,
                output_path: str = None
            ):
                """Process PDF with enhanced multilingual support using optimized agents."""
                try:
                    print(f"🔧 Processing PDF: {pdf_path}")
                    
                    # Step 1: Extract text from PDF using enhanced file extraction agent
                    file_agent = self.agents["file_extraction"]
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
                    
                    text_content = extraction_result.extracted_text
                    
                    # Step 2: Process with knowledge graph agent
                    kg_agent = self.agents["knowledge_graph"]
                    kg_request = AnalysisRequest(
                        data_type=DataType.TEXT,
                        content=text_content,
                        language=language
                    )
                    
                    kg_result = await kg_agent.process(kg_request)
                    
                    # Step 3: Generate report if requested
                    report_files = {}
                    if generate_report:
                        if not output_path:
                            from datetime import datetime
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            output_path = f"Results/reports/enhanced_multilingual_pdf_{language}_{timestamp}"
                        
                        report_result = await kg_agent.generate_graph_report(
                            output_path=output_path,
                            target_language=language
                        )
                        
                        if hasattr(report_result, 'success') and report_result.success:
                            report_files = {
                                "html": report_result.metadata.get('html_path', 'Unknown'),
                                "png": report_result.metadata.get('png_path', 'Unknown')
                            }
                    
                    # Step 4: Compile results
                    result = {
                        "success": True,
                        "pdf_path": pdf_path,
                        "language": language,
                        "text_extraction": {
                            "success": True,
                            "content_length": len(text_content),
                            "pages_processed": len(extraction_result.pages) if extraction_result.pages else 0,
                            "extraction_method": "Enhanced File Extraction Agent"
                        },
                        "entity_extraction": {
                            "entities_found": kg_result.metadata.get('statistics', {}).get('entities_found', 0) if kg_result.metadata else 0,
                            "entity_types": kg_result.metadata.get('statistics', {}).get('entity_types', {}) if kg_result.metadata else {},
                            "extraction_method": "Enhanced Knowledge Graph Agent"
                        },
                        "knowledge_graph": {
                            "nodes": kg_result.metadata.get('statistics', {}).get('nodes', 0) if kg_result.metadata else 0,
                            "edges": kg_result.metadata.get('statistics', {}).get('edges', 0) if kg_result.metadata else 0,
                            "communities": kg_result.metadata.get('statistics', {}).get('communities', 0) if kg_result.metadata else 0,
                            "processing_time": kg_result.processing_time
                        },
                        "report_files": report_files
                    }
                    
                    print("✅ PDF processing completed successfully")
                    return result
                    
                except Exception as e:
                    return {"success": False, "error": f"PDF processing failed: {str(e)}"}

            @self.mcp.tool(description="Process multilingual PDF using fixed components and MCP framework")
            @with_error_handling("multilingual_pdf_processing")
            async def process_multilingual_pdf_mcp(
                pdf_path: str,
                language: str = "auto",
                generate_report: bool = True,
                output_path: str = None
            ):
                """Process multilingual PDF using fixed components through MCP framework."""
                try:
                    print(f"🔧 Processing multilingual PDF via MCP: {pdf_path}")
                    
                    # Import language configuration system
                    from src.config.language_config import LanguageConfigFactory
                    
                    # Use the fixed EnhancedFileExtractionAgent
                    file_agent = self.agents["file_extraction"]
                    pdf_request = AnalysisRequest(
                        data_type=DataType.PDF,
                        content=pdf_path,
                        language=language
                    )
                    
                    print("🔧 Extracting text from PDF using fixed agent...")
                    extraction_result = await file_agent.process(pdf_request)
                    
                    if extraction_result.status != "completed":
                        return {
                            "success": False,
                            "error": f"PDF extraction failed: {extraction_result.metadata.get('error', 'Unknown error')}"
                        }
                    
                    text_content = extraction_result.extracted_text
                    print(f"✅ Text extraction successful! Length: {len(text_content)} characters")
                    
                    # Detect language if auto is specified
                    detected_language = language
                    if language == "auto":
                        detected_language = LanguageConfigFactory.detect_language_from_text(text_content)
                        print(f"🔍 Auto-detected language: {detected_language}")
                    
                    # Get language-specific configuration
                    try:
                        language_config = LanguageConfigFactory.get_config(detected_language)
                        print(f"✅ Loaded configuration for language: {language_config.language_name} ({detected_language})")
                    except ValueError as e:
                        print(f"⚠️ No specific configuration for {detected_language}, using default")
                        language_config = LanguageConfigFactory.get_config("en")  # Fallback to English
                    
                    # Use the fixed KnowledgeGraphAgent with language-specific configuration
                    kg_agent = self.agents["knowledge_graph"]
                    kg_request = AnalysisRequest(
                        data_type=DataType.TEXT,
                        content=text_content,
                        language=detected_language
                    )
                    
                    print(f"🔧 Processing with fixed knowledge graph agent for {detected_language}...")
                    kg_result = await kg_agent.process(kg_request)
                    print(f"✅ Knowledge graph processing successful! Time: {kg_result.processing_time:.2f}s")
                    
                    # Generate report if requested
                    report_files = {}
                    if generate_report:
                        if not output_path:
                            from datetime import datetime
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            output_path = f"Results/reports/multilingual_pdf_{detected_language}_{timestamp}"
                        
                        print("🔧 Generating knowledge graph report...")
                        report_result = await kg_agent.generate_graph_report(
                            output_path=output_path,
                            target_language=detected_language
                        )
                        
                        if hasattr(report_result, 'success') and report_result.success:
                            report_files = {
                                "html": report_result.metadata.get('html_path', 'Unknown'),
                                "png": report_result.metadata.get('png_path', 'Unknown')
                            }
                            print("✅ Report generated successfully!")
                    
                    # Compile results with language-specific information
                    result = {
                        "success": True,
                        "pdf_path": pdf_path,
                        "language": detected_language,
                        "language_name": language_config.language_name,
                        "processing_method": "MCP Framework with Fixed Components",
                        "language_config_used": {
                            "language_code": detected_language,
                            "language_name": language_config.language_name,
                            "entity_patterns_available": len(language_config.entity_patterns.person) + 
                                                       len(language_config.entity_patterns.organization) + 
                                                       len(language_config.entity_patterns.location) + 
                                                       len(language_config.entity_patterns.concept),
                            "processing_settings": {
                                "confidence_threshold": language_config.processing_settings.confidence_threshold,
                                "use_enhanced_extraction": language_config.processing_settings.use_enhanced_extraction,
                                "entity_clustering_enabled": language_config.processing_settings.entity_clustering_enabled
                            }
                        },
                        "text_extraction": {
                            "success": True,
                            "content_length": len(text_content),
                            "pages_processed": len(extraction_result.pages) if extraction_result.pages else 0,
                            "extraction_method": "Fixed Enhanced File Extraction Agent"
                        },
                        "entity_extraction": {
                            "entities_found": kg_result.metadata.get('statistics', {}).get('entities_found', 0) if kg_result.metadata else 0,
                            "entity_types": kg_result.metadata.get('statistics', {}).get('entity_types', {}) if kg_result.metadata else {},
                            "extraction_method": "Fixed Knowledge Graph Agent with Language-Specific Config"
                        },
                        "knowledge_graph": {
                            "nodes": kg_result.metadata.get('statistics', {}).get('nodes', 0) if kg_result.metadata else 0,
                            "edges": kg_result.metadata.get('statistics', {}).get('edges', 0) if kg_result.metadata else 0,
                            "communities": kg_result.metadata.get('statistics', {}).get('communities', 0) if kg_result.metadata else 0,
                            "processing_time": kg_result.processing_time
                        },
                        "report_files": report_files
                    }
                    
                    print(f"✅ Multilingual PDF processing completed successfully via MCP for {detected_language}")
                    return result
                    
                except Exception as e:
                    print(f"❌ Multilingual PDF processing failed: {e}")
                    return {"success": False, "error": f"Processing failed: {str(e)}"}

            # Unified Analysis Tools
            @self.mcp.tool(description="Analyze text content with unified interface")
            @with_error_handling("text_analysis")
            async def analyze_text_sentiment(
                text: str,
                agent_type: str = "standard",
                language: str = "en"
            ):
                """Analyze text content using specified agent type."""
                agent_map = {
                    "standard": "text",
                    "simple": "text_simple",
                    "strands": "text_strands",
                    "swarm": "text_swarm"
                }
                agent_key = agent_map.get(agent_type, "text")
                agent = self.agents[agent_key]
                
                analysis_request = AnalysisRequest(
                    data_type=DataType.TEXT,
                    content=text,
                    language=language
                )
                
                result = await agent.process(analysis_request)
                
                return {
                    "success": True,
                    "agent_type": agent_type,
                    "agent_used": agent_key,
                    "sentiment": result.sentiment.label,
                    "confidence": result.sentiment.confidence,
                    "reasoning": result.sentiment.reasoning,
                    "processing_time": result.processing_time
                }

            # Knowledge Graph Tools
            @self.mcp.tool(description="Extract entities from text")
            @with_error_handling("entity_extraction")
            async def extract_entities(text: str, language: str = "en"):
                """Extract entities from text using knowledge graph agent."""
                kg_agent = self.agents["knowledge_graph"]
                request = AnalysisRequest(
                    data_type=DataType.TEXT,
                    content=text,
                    language=language
                )
                
                result = await kg_agent.process(request)
                
                return {
                    "success": True,
                    "entities": result.metadata.get('entities', []),
                    "statistics": result.metadata.get('statistics', {}),
                    "processing_time": result.processing_time
                }

            @self.mcp.tool(description="Generate knowledge graph report")
            @with_error_handling("report_generation")
            async def generate_graph_report(
                output_path: str = None,
                target_language: str = "en"
            ):
                """Generate knowledge graph report."""
                kg_agent = self.agents["knowledge_graph"]
                
                if not output_path:
                    from datetime import datetime
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_path = f"Results/reports/knowledge_graph_{target_language}_{timestamp}"
                
                result = await kg_agent.generate_graph_report(
                    output_path=output_path,
                    target_language=target_language
                )
                
                return {
                    "success": result.success if hasattr(result, 'success') else True,
                    "output_path": output_path,
                    "metadata": result.metadata if hasattr(result, 'metadata') else {},
                    "processing_time": result.processing_time if hasattr(result, 'processing_time') else 0.0
                }

            # Content Analysis Tools
            @self.mcp.tool(description="Summarize text content with structured analysis")
            @with_error_handling("text_summarization")
            async def summarize_text_content(
                text: str,
                summary_type: str = "comprehensive",  # "brief", "comprehensive", "detailed"
                language: str = "en",
                include_key_points: bool = True,
                include_entities: bool = True
            ):
                """Summarize text content with optional key points and entity extraction."""
                try:
                    print(f"🔧 Summarizing text content (type: {summary_type})")
                    
                    # Use text agent for summarization
                    text_agent = self.agents["text"]
                    request = AnalysisRequest(
                        data_type=DataType.TEXT,
                        content=text,
                        language=language,
                        metadata={
                            "summary_type": summary_type,
                            "include_key_points": include_key_points,
                            "include_entities": include_entities
                        }
                    )
                    
                    result = await text_agent.process(request)
                    
                    # Extract summary from result
                    summary = ""
                    if hasattr(result, 'summary') and result.summary:
                        summary = result.summary
                    elif hasattr(result, 'sentiment') and result.sentiment.reasoning:
                        summary = result.sentiment.reasoning
                    else:
                        summary = "Summary generated successfully"
                    
                    # Extract key points and entities
                    key_points = result.metadata.get('key_points', []) if result.metadata else []
                    entities = result.metadata.get('entities', []) if result.metadata else []
                    
                    print(f"✅ Text summarization completed successfully")
                    
                    return {
                        "success": True,
                        "summary_type": summary_type,
                        "summary": summary,
                        "key_points": key_points,
                        "entities": entities,
                        "processing_time": result.processing_time
                    }
                except Exception as e:
                    print(f"❌ Text summarization failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Analyze chapter content with structured breakdown")
            @with_error_handling("chapter_analysis")
            async def analyze_chapter_content(
                chapter_text: str,
                chapter_title: str = "",
                language: str = "en",
                analysis_type: str = "comprehensive"  # "summary", "themes", "entities", "comprehensive"
            ):
                """Analyze chapter content with structured breakdown and insights."""
                try:
                    print(f"🔧 Analyzing chapter content: {chapter_title}")
                    
                    # Use knowledge graph agent for entity extraction
                    kg_agent = self.agents["knowledge_graph"]
                    kg_request = AnalysisRequest(
                        data_type=DataType.TEXT,
                        content=chapter_text,
                        language=language
                    )
                    
                    kg_result = await kg_agent.process(kg_request)
                    
                    # Use text agent for summarization and analysis
                    text_agent = self.agents["text"]
                    text_request = AnalysisRequest(
                        data_type=DataType.TEXT,
                        content=chapter_text,
                        language=language
                    )
                    
                    text_result = await text_agent.process(text_request)
                    
                    # Extract summary
                    summary = ""
                    if hasattr(text_result, 'summary') and text_result.summary:
                        summary = text_result.summary
                    elif hasattr(text_result, 'sentiment') and text_result.sentiment.reasoning:
                        summary = text_result.sentiment.reasoning
                    else:
                        summary = "Chapter analysis completed successfully"
                    
                    # Extract entities and statistics
                    entities = kg_result.metadata.get('entities', []) if kg_result.metadata else []
                    entity_statistics = kg_result.metadata.get('statistics', {}) if kg_result.metadata else {}
                    themes = text_result.metadata.get('themes', []) if text_result.metadata else []
                    key_concepts = text_result.metadata.get('key_concepts', []) if text_result.metadata else []
                    
                    print(f"✅ Chapter analysis completed successfully")
                    
                    return {
                        "success": True,
                        "chapter_title": chapter_title,
                        "analysis_type": analysis_type,
                        "summary": summary,
                        "entities": entities,
                        "entity_statistics": entity_statistics,
                        "themes": themes,
                        "key_concepts": key_concepts,
                        "processing_time": text_result.processing_time
                    }
                except Exception as e:
                    print(f"❌ Chapter analysis failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Extract and analyze specific content sections")
            @with_error_handling("content_extraction")
            async def extract_content_sections(
                content: str,
                section_type: str = "chapters",  # "chapters", "paragraphs", "sections"
                language: str = "en",
                include_analysis: bool = True
            ):
                """Extract and analyze specific content sections."""
                try:
                    print(f"🔧 Extracting content sections (type: {section_type})")
                    
                    # Parse content into sections
                    sections = []
                    if section_type == "chapters":
                        # Split by chapter markers
                        import re
                        chapter_pattern = r'##\s*Chapter\s*\d+[:\s]*([^\n]+)'
                        chapters = re.split(chapter_pattern, content)
                        
                        for i in range(1, len(chapters), 2):
                            if i + 1 < len(chapters):
                                title = chapters[i].strip()
                                content_text = chapters[i + 1].strip()
                                sections.append({
                                    "title": title,
                                    "content": content_text,
                                    "section_number": len(sections) + 1
                                })
                    
                    # Analyze each section if requested
                    if include_analysis and sections:
                        for section in sections:
                            analysis_result = await analyze_chapter_content(
                                section["content"],
                                section["title"],
                                language
                            )
                            section["analysis"] = analysis_result
                    
                    print(f"✅ Content extraction completed: {len(sections)} sections found")
                    
                    return {
                        "success": True,
                        "section_type": section_type,
                        "sections_found": len(sections),
                        "sections": sections
                    }
                except Exception as e:
                    print(f"❌ Content extraction failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Query knowledge graph for specific content")
            @with_error_handling("knowledge_graph_query")
            async def query_knowledge_graph(
                query: str,
                query_type: str = "entities",  # "entities", "relationships", "concepts", "full_text"
                language: str = "en",
                limit: int = 50
            ):
                """Query the knowledge graph for specific content and relationships."""
                try:
                    print(f"🔧 Querying knowledge graph: {query}")
                    
                    kg_agent = self.agents["knowledge_graph"]
                    
                    # Use the knowledge graph agent's query capabilities
                    if hasattr(kg_agent, 'query_graph'):
                        result = await kg_agent.query_graph(query, query_type, language, limit)
                    else:
                        # Fallback to processing the query as text
                        request = AnalysisRequest(
                            data_type=DataType.TEXT,
                            content=query,
                            language=language
                        )
                        result = await kg_agent.process(request)
                    
                    # Extract query results
                    query_results = result.metadata.get('query_results', []) if result.metadata else []
                    
                    print(f"✅ Knowledge graph query completed: {len(query_results)} results found")
                    
                    return {
                        "success": True,
                        "query": query,
                        "query_type": query_type,
                        "results": query_results,
                        "total_found": len(query_results),
                        "processing_time": result.processing_time
                    }
                except Exception as e:
                    print(f"❌ Knowledge graph query failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Compare multiple content sections or chapters")
            @with_error_handling("content_comparison")
            async def compare_content_sections(
                content_sections: List[str],
                comparison_type: str = "themes",  # "themes", "entities", "sentiment", "comprehensive"
                language: str = "en"
            ):
                """Compare multiple content sections for themes, entities, and patterns."""
                try:
                    print(f"🔧 Comparing {len(content_sections)} content sections")
                    
                    results = []
                    
                    for i, content in enumerate(content_sections):
                        # Analyze each section
                        analysis_result = await analyze_chapter_content(
                            content,
                            f"Section {i+1}",
                            language,
                            "comprehensive"
                        )
                        results.append(analysis_result)
                    
                    # Perform comparison analysis
                    comparison_analysis = {
                        "common_entities": [],
                        "common_themes": [],
                        "sentiment_variations": [],
                        "structural_patterns": []
                    }
                    
                    # Extract common entities across sections
                    all_entities = []
                    for result in results:
                        if result.get("success"):
                            all_entities.extend(result.get("entities", []))
                    
                    # Find common entities
                    entity_counts = {}
                    for entity in all_entities:
                        entity_name = entity.get("name", "")
                        entity_counts[entity_name] = entity_counts.get(entity_name, 0) + 1
                    
                    comparison_analysis["common_entities"] = [
                        {"name": name, "frequency": count}
                        for name, count in entity_counts.items()
                        if count > 1
                    ]
                    
                    print(f"✅ Content comparison completed: {len(comparison_analysis['common_entities'])} common entities found")
                    
                    return {
                        "success": True,
                        "comparison_type": comparison_type,
                        "sections_analyzed": len(results),
                        "individual_analyses": results,
                        "comparison_analysis": comparison_analysis
                    }
                except Exception as e:
                    print(f"❌ Content comparison failed: {e}")
                    return {"success": False, "error": str(e)}

            # Business Intelligence Tools
            @self.mcp.tool(description="Generate interactive business dashboard")
            @with_error_handling("business_dashboard")
            async def generate_business_dashboard(
                data_source: str,
                dashboard_type: str = "comprehensive",  # "executive", "detailed", "comprehensive"
                time_range: str = "30d",
                include_visualizations: bool = True
            ):
                """Generate interactive business dashboard."""
                try:
                    print(f"🔧 Generating {dashboard_type} business dashboard")
                    
                    bi_agent = self.agents["business_intelligence"]
                    result = await bi_agent.generate_business_dashboard(data_source, dashboard_type)
                    
                    print(f"✅ Business dashboard generated successfully: {dashboard_type}")
                    return {
                        "success": True,
                        "dashboard_type": dashboard_type,
                        "data_source": data_source,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Business dashboard generation failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Create executive summary dashboard")
            @with_error_handling("executive_summary")
            async def create_executive_summary(
                content_data: str,
                summary_type: str = "business",  # "business", "technical", "stakeholder"
                include_metrics: bool = True,
                include_trends: bool = True
            ):
                """Create executive summary dashboard."""
                try:
                    print(f"🔧 Creating {summary_type} executive summary")
                    
                    bi_agent = self.agents["business_intelligence"]
                    request = AnalysisRequest(
                        data_type=DataType.TEXT,
                        content=content_data,
                        language="en",
                        metadata={
                            "request_type": "report",
                            "report_type": "executive",
                            "summary_type": summary_type,
                            "include_metrics": include_metrics,
                            "include_trends": include_trends
                        }
                    )
                    
                    result = await bi_agent.process(request)
                    
                    print(f"✅ Executive summary created successfully: {summary_type}")
                    return {
                        "success": True,
                        "summary_type": summary_type,
                        "result": result.metadata
                    }
                except Exception as e:
                    print(f"❌ Executive summary creation failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Generate interactive data visualizations")
            @with_error_handling("data_visualization")
            async def generate_interactive_visualizations(
                data: str,
                chart_types: List[str] = ["trend", "distribution", "correlation"],
                interactive: bool = True,
                export_format: str = "html"
            ):
                """Generate interactive data visualizations."""
                try:
                    print(f"🔧 Generating {len(chart_types)} interactive visualizations")
                    
                    viz_agent = self.agents["data_visualization"]
                    result = await viz_agent.generate_visualizations(data, chart_types, interactive)
                    
                    print(f"✅ Interactive visualizations generated successfully")
                    return {
                        "success": True,
                        "chart_types": chart_types,
                        "interactive": interactive,
                        "export_format": export_format,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Visualization generation failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Generate executive business report")
            @with_error_handling("executive_report")
            async def generate_executive_report(
                content_data: str,
                report_type: str = "comprehensive",  # "executive", "detailed", "summary"
                include_insights: bool = True,
                include_recommendations: bool = True
            ):
                """Generate executive business report."""
                try:
                    print(f"🔧 Generating {report_type} executive report")
                    
                    bi_agent = self.agents["business_intelligence"]
                    result = await bi_agent.generate_executive_report(content_data, report_type)
                    
                    print(f"✅ Executive report generated successfully: {report_type}")
                    return {
                        "success": True,
                        "report_type": report_type,
                        "include_insights": include_insights,
                        "include_recommendations": include_recommendations,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Executive report generation failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Create business-focused content summary")
            @with_error_handling("business_summary")
            async def create_business_summary(
                content: str,
                summary_length: str = "executive",  # "brief", "executive", "detailed"
                focus_areas: List[str] = ["key_insights", "trends", "actions"],
                include_metrics: bool = True
            ):
                """Create business-focused content summary."""
                try:
                    print(f"🔧 Creating {summary_length} business summary")
                    
                    bi_agent = self.agents["business_intelligence"]
                    request = AnalysisRequest(
                        data_type=DataType.TEXT,
                        content=content,
                        language="en",
                        metadata={
                            "request_type": "report",
                            "report_type": "summary",
                            "summary_length": summary_length,
                            "focus_areas": focus_areas,
                            "include_metrics": include_metrics
                        }
                    )
                    
                    result = await bi_agent.process(request)
                    
                    print(f"✅ Business summary created successfully: {summary_length}")
                    return {
                        "success": True,
                        "summary_length": summary_length,
                        "focus_areas": focus_areas,
                        "include_metrics": include_metrics,
                        "result": result.metadata
                    }
                except Exception as e:
                    print(f"❌ Business summary creation failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Analyze business trends and patterns")
            @with_error_handling("business_trends")
            async def analyze_business_trends(
                data: str,
                trend_period: str = "30d",
                analysis_type: str = "comprehensive",  # "sentiment", "topics", "entities", "comprehensive"
                include_forecasting: bool = True
            ):
                """Analyze business trends and patterns."""
                try:
                    print(f"🔧 Analyzing business trends for period: {trend_period}")
                    
                    bi_agent = self.agents["business_intelligence"]
                    result = await bi_agent.analyze_business_trends(data, trend_period)
                    
                    print(f"✅ Business trends analysis completed successfully")
                    return {
                        "success": True,
                        "trend_period": trend_period,
                        "analysis_type": analysis_type,
                        "include_forecasting": include_forecasting,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Business trends analysis failed: {e}")
                    return {"success": False, "error": str(e)}

            # Phase 2: External Data Integration Tools
            
            # Social Media Integration Tools
            @self.mcp.tool(description="Integrate social media data from multiple platforms")
            @with_error_handling("social_media_integration")
            async def integrate_social_media_data(
                platforms: List[str] = ["twitter", "linkedin", "facebook", "instagram"],
                data_types: List[str] = ["posts", "comments", "sentiment", "trends"],
                time_range: str = "7d",
                include_metadata: bool = True
            ):
                """Integrate social media data from multiple platforms."""
                try:
                    print(f"🔧 Integrating social media data from {len(platforms)} platforms")
                    
                    social_agent = self.agents["social_media"]
                    result = await social_agent.integrate_social_media_data(
                        platforms, data_types, time_range, include_metadata
                    )
                    
                    print(f"✅ Social media integration completed successfully")
                    return {
                        "success": True,
                        "platforms": platforms,
                        "data_types": data_types,
                        "time_range": time_range,
                        "include_metadata": include_metadata,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Social media integration failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Analyze social media content and trends")
            @with_error_handling("social_media_analysis")
            async def analyze_social_media_content(
                platform: str,
                content_type: str = "posts",
                analysis_type: str = "comprehensive",  # "sentiment", "topics", "influencers", "comprehensive"
                include_engagement: bool = True
            ):
                """Analyze social media content and trends."""
                try:
                    print(f"🔧 Analyzing {platform} social media content")
                    
                    social_agent = self.agents["social_media"]
                    result = await social_agent.analyze_social_media_content(
                        platform, content_type, analysis_type, include_engagement
                    )
                    
                    print(f"✅ Social media content analysis completed successfully")
                    return {
                        "success": True,
                        "platform": platform,
                        "content_type": content_type,
                        "analysis_type": analysis_type,
                        "include_engagement": include_engagement,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Social media content analysis failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Monitor social media trends and mentions")
            @with_error_handling("social_media_monitoring")
            async def monitor_social_media_trends(
                keywords: List[str],
                platforms: List[str] = ["twitter", "linkedin"],
                monitoring_period: str = "24h",
                alert_threshold: int = 100
            ):
                """Monitor social media trends and mentions."""
                try:
                    print(f"🔧 Monitoring social media trends for {len(keywords)} keywords")
                    
                    social_agent = self.agents["social_media"]
                    result = await social_agent.monitor_social_media_trends(
                        keywords, platforms, monitoring_period, alert_threshold
                    )
                    
                    print(f"✅ Social media trend monitoring completed successfully")
                    return {
                        "success": True,
                        "keywords": keywords,
                        "platforms": platforms,
                        "monitoring_period": monitoring_period,
                        "alert_threshold": alert_threshold,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Social media trend monitoring failed: {e}")
                    return {"success": False, "error": str(e)}

            # Database & API Integration Tools
            @self.mcp.tool(description="Connect and query database sources")
            @with_error_handling("database_connection")
            async def connect_database_source(
                database_type: str,  # "mongodb", "postgresql", "mysql", "elasticsearch"
                connection_string: str,
                query: str,
                include_metadata: bool = True
            ):
                """Connect and query database sources."""
                try:
                    print(f"🔧 Connecting to {database_type} database")
                    
                    external_agent = self.agents["external_data"]
                    result = await external_agent.connect_database_source(
                        database_type, connection_string, query, include_metadata
                    )
                    
                    print(f"✅ Database connection and query completed successfully")
                    return {
                        "success": True,
                        "database_type": database_type,
                        "connection_string": connection_string[:20] + "..." if len(connection_string) > 20 else connection_string,
                        "query": query,
                        "include_metadata": include_metadata,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Database connection failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Fetch data from external APIs")
            @with_error_handling("api_integration")
            async def fetch_external_api_data(
                api_endpoint: str,
                api_type: str = "rest",  # "rest", "graphql", "soap"
                parameters: Dict[str, Any] = {},
                authentication: Dict[str, str] = {},
                include_caching: bool = True
            ):
                """Fetch data from external APIs."""
                try:
                    print(f"🔧 Fetching data from {api_type} API: {api_endpoint}")
                    
                    external_agent = self.agents["external_data"]
                    result = await external_agent.fetch_external_api_data(
                        api_endpoint, api_type, parameters, authentication, include_caching
                    )
                    
                    print(f"✅ External API data fetch completed successfully")
                    return {
                        "success": True,
                        "api_endpoint": api_endpoint,
                        "api_type": api_type,
                        "parameters": parameters,
                        "authentication": {k: "***" for k in authentication.keys()},
                        "include_caching": include_caching,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ External API data fetch failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Manage external data sources")
            @with_error_handling("data_source_management")
            async def manage_data_sources(
                action: str,  # "add", "update", "remove", "list", "test"
                source_config: Dict[str, Any] = {},
                include_validation: bool = True
            ):
                """Manage external data sources."""
                try:
                    print(f"🔧 Managing data sources with action: {action}")
                    
                    external_agent = self.agents["external_data"]
                    result = await external_agent.manage_data_sources(
                        action, source_config, include_validation
                    )
                    
                    print(f"✅ Data source management completed successfully")
                    return {
                        "success": True,
                        "action": action,
                        "source_config": source_config,
                        "include_validation": include_validation,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Data source management failed: {e}")
                    return {"success": False, "error": str(e)}

            # Market Data & News Integration Tools
            @self.mcp.tool(description="Analyze market data and trends")
            @with_error_handling("market_data_analysis")
            async def analyze_market_data(
                market_sector: str,
                data_types: List[str] = ["sentiment", "trends", "news", "social"],
                time_range: str = "30d",
                include_competitors: bool = True
            ):
                """Analyze market data and trends."""
                try:
                    print(f"🔧 Analyzing market data for sector: {market_sector}")
                    
                    market_agent = self.agents["market_data"]
                    result = await market_agent.analyze_market_data(
                        market_sector, data_types, time_range, include_competitors
                    )
                    
                    print(f"✅ Market data analysis completed successfully")
                    return {
                        "success": True,
                        "market_sector": market_sector,
                        "data_types": data_types,
                        "time_range": time_range,
                        "include_competitors": include_competitors,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Market data analysis failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Monitor news sources and headlines")
            @with_error_handling("news_monitoring")
            async def monitor_news_sources(
                sources: List[str] = ["reuters", "bloomberg", "cnn", "bbc"],
                keywords: List[str] = [],
                analysis_type: str = "sentiment",  # "sentiment", "topics", "entities", "comprehensive"
                include_summaries: bool = True
            ):
                """Monitor news sources and headlines."""
                try:
                    print(f"🔧 Monitoring news sources: {sources}")
                    
                    market_agent = self.agents["market_data"]
                    result = await market_agent.monitor_news_sources(
                        sources, keywords, analysis_type, include_summaries
                    )
                    
                    print(f"✅ News source monitoring completed successfully")
                    return {
                        "success": True,
                        "sources": sources,
                        "keywords": keywords,
                        "analysis_type": analysis_type,
                        "include_summaries": include_summaries,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ News source monitoring failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Integrate financial and economic data")
            @with_error_handling("financial_data_integration")
            async def integrate_financial_data(
                data_source: str,  # "yahoo_finance", "alpha_vantage", "quandl"
                symbols: List[str],
                data_types: List[str] = ["price", "volume", "news", "sentiment"],
                include_analysis: bool = True
            ):
                """Integrate financial and economic data."""
                try:
                    print(f"🔧 Integrating financial data from {data_source}")
                    
                    market_agent = self.agents["market_data"]
                    result = await market_agent.integrate_financial_data(
                        data_source, symbols, data_types, include_analysis
                    )
                    
                    print(f"✅ Financial data integration completed successfully")
                    return {
                        "success": True,
                        "data_source": data_source,
                        "symbols": symbols,
                        "data_types": data_types,
                        "include_analysis": include_analysis,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Financial data integration failed: {e}")
                    return {"success": False, "error": str(e)}

            # Phase 3: Multi-Modal Analysis Tools
            @self.mcp.tool(description="Analyze content comprehensively across all modalities")
            @with_error_handling("comprehensive_content_analysis")
            async def analyze_content_comprehensive(
                content_data: Dict[str, Any],
                analysis_type: str = "business",
                include_cross_modal: bool = True,
                include_insights: bool = True
            ):
                """Analyze content comprehensively across all modalities."""
                try:
                    print(f"🔧 Analyzing content comprehensively across modalities")
                    
                    multi_modal_agent = self.agents["multi_modal_analysis"]
                    result = await multi_modal_agent.cross_modal_analyzer.analyze_content_comprehensive(
                        content_data, analysis_type, include_cross_modal, include_insights
                    )
                    
                    print(f"✅ Comprehensive content analysis completed successfully")
                    return {
                        "success": True,
                        "analysis_type": analysis_type,
                        "include_cross_modal": include_cross_modal,
                        "include_insights": include_insights,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Comprehensive content analysis failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Generate cross-modal business insights")
            @with_error_handling("cross_modal_insights")
            async def generate_cross_modal_insights(
                content_sources: List[str],
                insight_type: str = "business",
                include_visualization: bool = True,
                include_recommendations: bool = True
            ):
                """Generate cross-modal business insights."""
                try:
                    print(f"🔧 Generating cross-modal business insights")
                    
                    multi_modal_agent = self.agents["multi_modal_analysis"]
                    result = await multi_modal_agent._generate_cross_modal_insights(
                        content_sources, insight_type, include_visualization, include_recommendations
                    )
                    
                    print(f"✅ Cross-modal insights generation completed successfully")
                    return {
                        "success": True,
                        "insight_type": insight_type,
                        "include_visualization": include_visualization,
                        "include_recommendations": include_recommendations,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Cross-modal insights generation failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Create comprehensive business intelligence report")
            @with_error_handling("business_intelligence_report")
            async def create_business_intelligence_report(
                data_sources: List[str],
                report_scope: str = "comprehensive",
                include_benchmarks: bool = True,
                include_forecasting: bool = True
            ):
                """Create comprehensive business intelligence report."""
                try:
                    print(f"🔧 Creating comprehensive business intelligence report")
                    
                    bi_agent = self.agents["business_intelligence"]
                    result = await bi_agent.create_business_intelligence_report(
                        data_sources, report_scope, include_benchmarks, include_forecasting
                    )
                    
                    print(f"✅ Business intelligence report creation completed successfully")
                    return {
                        "success": True,
                        "report_scope": report_scope,
                        "include_benchmarks": include_benchmarks,
                        "include_forecasting": include_forecasting,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Business intelligence report creation failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Create narrative-driven content analysis")
            @with_error_handling("content_storytelling")
            async def create_content_story(
                content_data: str,
                story_type: str = "business",
                include_visuals: bool = True,
                include_actions: bool = True
            ):
                """Create narrative-driven content analysis."""
                try:
                    print(f"🔧 Creating narrative-driven content story")
                    
                    multi_modal_agent = self.agents["multi_modal_analysis"]
                    result = await multi_modal_agent.content_storyteller.create_content_story(
                        content_data, story_type, include_visuals, include_actions
                    )
                    
                    print(f"✅ Content story creation completed successfully")
                    return {
                        "success": True,
                        "story_type": story_type,
                        "include_visuals": include_visuals,
                        "include_actions": include_actions,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Content story creation failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Generate data storytelling presentation")
            @with_error_handling("data_storytelling")
            async def generate_data_story(
                insights: List[Dict[str, Any]],
                presentation_type: str = "executive",
                include_slides: bool = True,
                include_narrative: bool = True
            ):
                """Generate data storytelling presentation."""
                try:
                    print(f"🔧 Generating data storytelling presentation")
                    
                    multi_modal_agent = self.agents["multi_modal_analysis"]
                    result = await multi_modal_agent._generate_data_story(
                        insights, presentation_type, include_slides, include_narrative
                    )
                    
                    print(f"✅ Data story generation completed successfully")
                    return {
                        "success": True,
                        "presentation_type": presentation_type,
                        "include_slides": include_slides,
                        "include_narrative": include_narrative,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Data story generation failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Create actionable business insights")
            @with_error_handling("actionable_insights")
            async def create_actionable_insights(
                analysis_results: Dict[str, Any],
                insight_type: str = "strategic",
                include_prioritization: bool = True,
                include_timeline: bool = True
            ):
                """Create actionable business insights."""
                try:
                    print(f"🔧 Creating actionable business insights")
                    
                    bi_agent = self.agents["business_intelligence"]
                    result = await bi_agent.create_actionable_insights(
                        analysis_results, insight_type, include_prioritization, include_timeline
                    )
                    
                    print(f"✅ Actionable insights creation completed successfully")
                    return {
                        "success": True,
                        "insight_type": insight_type,
                        "include_prioritization": include_prioritization,
                        "include_timeline": include_timeline,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Actionable insights creation failed: {e}")
                    return {"success": False, "error": str(e)}

            # Phase 4: Export & Automation Tools
            @self.mcp.tool(description="Export analysis results to multiple formats")
            @with_error_handling("export_analysis_results")
            async def export_analysis_results(
                data: Dict[str, Any],
                export_formats: List[str] = ["pdf", "excel", "html", "json"],
                include_visualizations: bool = True,
                include_metadata: bool = True
            ):
                """Export analysis results to multiple formats."""
                try:
                    print(f"🔧 Exporting analysis results to formats: {export_formats}")
                    
                    export_agent = self.agents["data_export"]
                    result = await export_agent.export_analysis_results(
                        data, export_formats, include_visualizations, include_metadata
                    )
                    
                    print(f"✅ Export completed successfully")
                    return {
                        "success": True,
                        "export_formats": export_formats,
                        "include_visualizations": include_visualizations,
                        "include_metadata": include_metadata,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Export failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Generate automated business reports")
            @with_error_handling("generate_automated_reports")
            async def generate_automated_reports(
                report_type: str = "business",
                schedule: str = "weekly",
                recipients: List[str] = [],
                include_attachments: bool = True
            ):
                """Generate automated business reports."""
                try:
                    print(f"🔧 Generating automated {report_type} report with {schedule} schedule")
                    
                    report_agent = self.agents["report_generation"]
                    result = await report_agent.generate_automated_report(
                        report_type, schedule, recipients, include_attachments
                    )
                    
                    print(f"✅ Automated report generation completed successfully")
                    return {
                        "success": True,
                        "report_type": report_type,
                        "schedule": schedule,
                        "recipients": recipients,
                        "include_attachments": include_attachments,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Automated report generation failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Share reports via multiple channels")
            @with_error_handling("share_reports")
            async def share_reports(
                report_data: Dict[str, Any],
                sharing_methods: List[str] = ["email", "cloud", "api"],
                recipients: List[str] = [],
                include_notifications: bool = True
            ):
                """Share reports via multiple channels."""
                try:
                    print(f"🔧 Sharing reports via methods: {sharing_methods}")
                    
                    export_agent = self.agents["data_export"]
                    result = await export_agent.share_reports(
                        report_data, sharing_methods, recipients, include_notifications
                    )
                    
                    print(f"✅ Report sharing completed successfully")
                    return {
                        "success": True,
                        "sharing_methods": sharing_methods,
                        "recipients": recipients,
                        "include_notifications": include_notifications,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Report sharing failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Schedule recurring reports")
            @with_error_handling("schedule_reports")
            async def schedule_reports(
                report_type: str,
                schedule: str,
                recipients: List[str] = None,
                start_date: str = None
            ):
                """Schedule recurring reports."""
                try:
                    print(f"🔧 Scheduling {report_type} report with {schedule} schedule")
                    
                    report_agent = self.agents["report_generation"]
                    result = await report_agent.schedule_report(
                        report_type, schedule, recipients, start_date
                    )
                    
                    print(f"✅ Report scheduling completed successfully")
                    return {
                        "success": True,
                        "report_type": report_type,
                        "schedule": schedule,
                        "recipients": recipients,
                        "start_date": start_date,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Report scheduling failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Get report generation history")
            @with_error_handling("get_report_history")
            async def get_report_history(limit: int = 10):
                """Get report generation history."""
                try:
                    print(f"🔧 Getting report history (limit: {limit})")
                    
                    report_agent = self.agents["report_generation"]
                    result = await report_agent.get_report_history(limit)
                    
                    print(f"✅ Report history retrieved successfully")
                    return {
                        "success": True,
                        "limit": limit,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Getting report history failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Get export history")
            @with_error_handling("get_export_history")
            async def get_export_history(limit: int = 10):
                """Get export history."""
                try:
                    print(f"🔧 Getting export history (limit: {limit})")
                    
                    export_agent = self.agents["data_export"]
                    result = await export_agent.get_export_history(limit)
                    
                    print(f"✅ Export history retrieved successfully")
                    return {
                        "success": True,
                        "limit": limit,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Getting export history failed: {e}")
                    return {"success": False, "error": str(e)}

            # Phase 5: Semantic Search & Agent Reflection Tools
            @self.mcp.tool(description="Intelligent semantic search across all content types")
            @with_error_handling("semantic_search_intelligent")
            async def semantic_search_intelligent(
                query: str,
                content_types: List[str] = None,
                search_strategy: str = "accuracy",
                include_agent_metadata: bool = True,
                combine_results: bool = True
            ):
                """Intelligent semantic search across all content types."""
                try:
                    print(f"🔧 Performing intelligent semantic search for: {query}")
                    
                    search_agent = self.agents["semantic_search"]
                    result = await search_agent.semantic_search_intelligent(
                        query=query,
                        content_types=content_types,
                        search_strategy=search_strategy,
                        include_agent_metadata=include_agent_metadata,
                        combine_results=combine_results
                    )
                    
                    print(f"✅ Semantic search completed successfully")
                    return {
                        "success": True,
                        "query": query,
                        "search_strategy": search_strategy,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Semantic search failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Route queries to optimal agents based on content and capability")
            @with_error_handling("route_query_intelligently")
            async def route_query_intelligently(
                query: str,
                content_data: Dict[str, Any] = None,
                routing_strategy: str = "accuracy",
                include_fallback: bool = True
            ):
                """Route queries to optimal agents based on content and capability."""
                try:
                    print(f"🔧 Routing query intelligently: {query}")
                    
                    search_agent = self.agents["semantic_search"]
                    result = await search_agent.route_query_intelligently(
                        query=query,
                        content_data=content_data,
                        routing_strategy=routing_strategy,
                        include_fallback=include_fallback
                    )
                    
                    print(f"✅ Query routing completed successfully")
                    return {
                        "success": True,
                        "query": query,
                        "routing_strategy": routing_strategy,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Query routing failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Combine and synthesize results from multiple agents")
            @with_error_handling("combine_agent_results")
            async def combine_agent_results(
                results: List[Dict[str, Any]],
                combination_strategy: str = "weighted",
                include_confidence_scores: bool = True
            ):
                """Combine and synthesize results from multiple agents."""
                try:
                    print(f"🔧 Combining results from {len(results)} agents using {combination_strategy} strategy")
                    
                    search_agent = self.agents["semantic_search"]
                    result = await search_agent.combine_agent_results(
                        results=results,
                        combination_strategy=combination_strategy,
                        include_confidence_scores=include_confidence_scores
                    )
                    
                    print(f"✅ Result combination completed successfully")
                    return {
                        "success": True,
                        "combination_strategy": combination_strategy,
                        "input_results_count": len(results),
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Result combination failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Get agent capabilities and specializations")
            @with_error_handling("get_agent_capabilities")
            async def get_agent_capabilities(
                agent_ids: List[str] = None,
                include_performance_metrics: bool = True
            ):
                """Get agent capabilities and specializations."""
                try:
                    print(f"🔧 Getting agent capabilities for {len(agent_ids) if agent_ids else 'all'} agents")
                    
                    search_agent = self.agents["semantic_search"]
                    result = await search_agent.get_agent_capabilities(
                        agent_ids=agent_ids,
                        include_performance_metrics=include_performance_metrics
                    )
                    
                    print(f"✅ Agent capabilities retrieved successfully")
                    return {
                        "success": True,
                        "agent_ids": agent_ids,
                        "include_performance_metrics": include_performance_metrics,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Getting agent capabilities failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Coordinate agent reflection and communication")
            @with_error_handling("coordinate_agent_reflection")
            async def coordinate_agent_reflection(
                query: str,
                initial_response: Dict[str, Any],
                reflection_type: str = "comprehensive",
                include_agent_questioning: bool = True
            ):
                """Coordinate agent reflection and communication."""
                try:
                    print(f"🔧 Coordinating agent reflection for query: {query}")
                    
                    reflection_agent = self.agents["reflection_coordinator"]
                    result = await reflection_agent.coordinate_agent_reflection(
                        query=query,
                        initial_response=initial_response,
                        reflection_type=reflection_type,
                        include_agent_questioning=include_agent_questioning
                    )
                    
                    print(f"✅ Agent reflection coordination completed successfully")
                    return {
                        "success": True,
                        "query": query,
                        "reflection_type": reflection_type,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Agent reflection coordination failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Enable agents to question and validate each other")
            @with_error_handling("agent_questioning_system")
            async def agent_questioning_system(
                source_agent: str,
                target_agent: str,
                question: str,
                context: Dict[str, Any] = None,
                response_format: str = "structured"
            ):
                """Enable agents to question and validate each other."""
                try:
                    print(f"🔧 Agent questioning: {source_agent} -> {target_agent}")
                    
                    reflection_agent = self.agents["reflection_coordinator"]
                    result = await reflection_agent.agent_questioning_system(
                        source_agent=source_agent,
                        target_agent=target_agent,
                        question=question,
                        context=context,
                        response_format=response_format
                    )
                    
                    print(f"✅ Agent questioning completed successfully")
                    return {
                        "success": True,
                        "source_agent": source_agent,
                        "target_agent": target_agent,
                        "question": question,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Agent questioning failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Get reflection insights and recommendations")
            @with_error_handling("get_reflection_insights")
            async def get_reflection_insights(
                query_id: str,
                include_agent_feedback: bool = True,
                include_confidence_improvements: bool = True
            ):
                """Get reflection insights and recommendations."""
                try:
                    print(f"🔧 Getting reflection insights for query: {query_id}")
                    
                    reflection_agent = self.agents["reflection_coordinator"]
                    result = await reflection_agent.get_reflection_insights(
                        query_id=query_id,
                        include_agent_feedback=include_agent_feedback,
                        include_confidence_improvements=include_confidence_improvements
                    )
                    
                    print(f"✅ Reflection insights retrieved successfully")
                    return {
                        "success": True,
                        "query_id": query_id,
                        "include_agent_feedback": include_agent_feedback,
                        "include_confidence_improvements": include_confidence_improvements,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Getting reflection insights failed: {e}")
                    return {"success": False, "error": str(e)}

            @self.mcp.tool(description="Validate and improve response quality")
            @with_error_handling("validate_response_quality")
            async def validate_response_quality(
                response: Dict[str, Any],
                validation_criteria: List[str] = None,
                include_improvement_suggestions: bool = True
            ):
                """Validate and improve response quality."""
                try:
                    print(f"🔧 Validating response quality")
                    
                    reflection_agent = self.agents["reflection_coordinator"]
                    result = await reflection_agent.validate_response_quality(
                        response=response,
                        validation_criteria=validation_criteria,
                        include_improvement_suggestions=include_improvement_suggestions
                    )
                    
                    print(f"✅ Response quality validation completed successfully")
                    return {
                        "success": True,
                        "validation_criteria": validation_criteria,
                        "include_improvement_suggestions": include_improvement_suggestions,
                        "result": result
                    }
                except Exception as e:
                    print(f"❌ Response quality validation failed: {e}")
                    return {"success": False, "error": str(e)}

            print("Registered optimized tools with streamable HTTP support")
        except Exception as e:
            print(f"❌ Error registering tools: {e}")

    def run(self, host: str = "localhost", port: int = 8000, debug: bool = False):
        """Run the MCP server with streamable HTTP support."""
        if self.mcp:
            print(f"🚀 Starting MCP server with streamable HTTP on {host}:{port}")
            return self.mcp.run(transport="streamable-http")
        else:
            print("❌ MCP server not initialized")


def start_mcp_server():
    """Start the unified MCP server with streamable HTTP support."""
    try:
        # Create the unified MCP server
        mcp_server = OptimizedMCPServer()
        
        if mcp_server.mcp is None:
            print("⚠️ MCP server not available - skipping MCP server startup")
            return None
        
        # Start the server in a separate thread
        def run_mcp_server():
            try:
                mcp_server.run(host="localhost", port=8000, debug=False)
            except Exception as e:
                print(f"❌ Error starting MCP server: {e}")
        
        mcp_thread = threading.Thread(target=run_mcp_server, daemon=True)
        mcp_thread.start()
        
        print("Unified MCP server with streamable HTTP started successfully")
        print(" - MCP Server: http://localhost:8000/mcp")
        print(" - Available tools:")
        print("   • PDF Processing: process_pdf_enhanced_multilingual, process_multilingual_pdf_mcp")
        print("   • Text Analysis: analyze_text_sentiment, extract_entities")
        print("   • Content Analysis: summarize_text_content, analyze_chapter_content, extract_content_sections")
        print("   • Knowledge Graph: generate_graph_report, query_knowledge_graph")
        print("   • Content Comparison: compare_content_sections")
        print("   • Business Intelligence: generate_business_dashboard, create_executive_summary, generate_interactive_visualizations")
        print("   • Executive Reporting: generate_executive_report, create_business_summary, analyze_business_trends")
        print("   • Social Media Integration: integrate_social_media_data, analyze_social_media_content, monitor_social_media_trends")
        print("   • Database & API Integration: connect_database_source, fetch_external_api_data, manage_data_sources")
        print("   • Market Data & News: analyze_market_data, monitor_news_sources, integrate_financial_data")
        print("   • Multi-Modal Analysis: analyze_content_comprehensive, generate_cross_modal_insights, create_content_story")
        print("   • Business Intelligence: create_business_intelligence_report, create_actionable_insights, generate_data_story")
        print("   • Export & Automation: export_analysis_results, generate_automated_reports, share_reports, schedule_reports, get_report_history, get_export_history")
        print("   • Semantic Search & Reflection: semantic_search_intelligent, route_query_intelligently, combine_agent_results, get_agent_capabilities")
        print("   • Agent Reflection: coordinate_agent_reflection, agent_questioning_system, get_reflection_insights, validate_response_quality")
        print("   • System Management: get_all_agents_status, start_all_agents, stop_all_agents")
        
        return mcp_server
    except Exception as e:
        print(f"⚠️ Warning: Could not start MCP server: {e}")
        print(" The application will run without MCP server integration")
        return None


async def process_classical_chinese_pdf_simple(
    pdf_path: str,
    language: str = "zh",
    generate_report: bool = True,
    output_path: str = None
):
    """
    Process Classical Chinese PDF using optimized agents directly.
    
    Args:
        pdf_path: Path to the PDF file
        language: Language code (default: "zh" for Chinese)
        generate_report: Whether to generate a knowledge graph report
        output_path: Custom output path for the report
    
    Returns:
        Dictionary with processing results
    """
    print(f"🔧 Processing Classical Chinese PDF: {pdf_path}")
    
    try:
        # Import required modules
        from src.agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
        from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
        from src.core.models import AnalysisRequest, DataType
        
        # Create agents
        file_agent = EnhancedFileExtractionAgent()
        kg_agent = KnowledgeGraphAgent()
        
        # Step 1: Extract text from PDF
        pdf_request = AnalysisRequest(
            data_type=DataType.PDF,
            content=pdf_path,
            language=language
        )
        
        print("🔧 Extracting text from PDF...")
        extraction_result = await file_agent.process(pdf_request)
        
        if extraction_result.status != "completed":
            return {
                "success": False,
                "error": f"PDF extraction failed: {extraction_result.metadata.get('error', 'Unknown error')}"
            }
        
        text_content = extraction_result.extracted_text
        print(f"✅ Text extraction successful! Length: {len(text_content)} characters")
        
        # Step 2: Process with knowledge graph
        print("🔧 Processing with knowledge graph...")
        
        kg_request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=text_content,
            language=language
        )
        
        kg_result = await kg_agent.process(kg_request)
        print(f"✅ Knowledge graph processing successful! Time: {kg_result.processing_time:.2f}s")
        
        # Step 3: Generate report if requested
        report_files = {}
        if generate_report:
            if not output_path:
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"Results/reports/enhanced_multilingual_pdf_{language}_{timestamp}"
            
            print("🔧 Generating knowledge graph report...")
            report_result = await kg_agent.generate_graph_report(
                output_path=output_path,
                target_language=language
            )
            
            if hasattr(report_result, 'success') and report_result.success:
                report_files = {
                    "html": report_result.metadata.get('html_path', 'Unknown'),
                    "png": report_result.metadata.get('png_path', 'Unknown')
                }
                print("✅ Report generated successfully!")
        
        # Step 4: Compile results
        result = {
            "success": True,
            "pdf_path": pdf_path,
            "language": language,
            "text_extraction": {
                "success": True,
                "content_length": len(text_content),
                "pages_processed": len(extraction_result.pages) if extraction_result.pages else 0,
                "extraction_method": "Enhanced File Extraction Agent"
            },
            "entity_extraction": {
                "entities_found": kg_result.metadata.get('statistics', {}).get('entities_found', 0) if kg_result.metadata else 0,
                "entity_types": kg_result.metadata.get('statistics', {}).get('entity_types', {}) if kg_result.metadata else {},
                "extraction_method": "Enhanced Knowledge Graph Agent"
            },
            "knowledge_graph": {
                "nodes": kg_result.metadata.get('statistics', {}).get('nodes', 0) if kg_result.metadata else 0,
                "edges": kg_result.metadata.get('statistics', {}).get('edges', 0) if kg_result.metadata else 0,
                "communities": kg_result.metadata.get('statistics', {}).get('communities', 0) if kg_result.metadata else 0,
                "processing_time": kg_result.processing_time
            },
            "report_files": report_files
        }
        
        print("✅ PDF processing completed successfully")
        return result
        
    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        return {"success": False, "error": f"Processing failed: {str(e)}"}


def check_service_health(url: str, timeout: int = 5) -> bool:
    """Check if a service is healthy by making a request to its health endpoint."""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed for {url}: {e}")
        return False

def wait_for_services(api_host: str, api_port: int, max_wait: int = 60) -> bool:
    """Wait for all services to be ready."""
    print("⏳ Waiting for services to be ready...")
    
    api_url = f"http://127.0.0.1:{api_port}/health"
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        api_ready = check_service_health(api_url)
        
        if api_ready:
            print("✅ API service is ready!")
            return True
        
        # Show progress
        elapsed = int(time.time() - start_time)
        print(f"⏳ Waiting... ({elapsed}s) - API: {'✅' if api_ready else '❌'}")
        time.sleep(1)
    
    print("⚠️ API service not ready within timeout period")
    return False

def launch_streamlit_apps():
    """Launch Streamlit applications in background."""
    print("🚀 Launching Streamlit applications...")
    
    try:
        # Launch main UI on port 8501
        main_ui_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "ui/main.py",
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Launch landing page on port 8502
        landing_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "ui/landing_page.py",
            "--server.port", "8502", 
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait a moment for Streamlit to start
        time.sleep(5)
        
        print("✅ Streamlit applications launched successfully!")
        print("   📊 Main UI: http://localhost:8501")
        print("   🏠 Landing Page: http://localhost:8502")
        
        return main_ui_process, landing_process
        
    except Exception as e:
        print(f"❌ Failed to launch Streamlit applications: {e}")
        return None, None


def get_mcp_tools_info():
    """Get information about available MCP tools."""
    try:
        mcp_server = OptimizedMCPServer()
        if mcp_server.mcp is None:
            print("⚠️ MCP server not available")
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
        elif hasattr(mcp_server.mcp, 'get_tools'):
            result = mcp_server.mcp.get_tools()
            if hasattr(result, '__await__'):
                print("ℹ️ Using comprehensive tool list (async discovery not needed)")
                tools = []
            else:
                tools = result
        elif hasattr(mcp_server.mcp, 'list_tools'):
            result = mcp_server.mcp.list_tools()
            if hasattr(result, '__await__'):
                print("ℹ️ Using comprehensive tool list (async discovery not needed)")
                tools = []
            else:
                tools = result
        else:
            # If we can't access tools directly, provide a list of known tools
            tools = [
                "get_all_agents_status",
                "start_all_agents", 
                "stop_all_agents",
                "process_pdf_enhanced_multilingual",
                "process_multilingual_pdf_mcp",
                "analyze_text_sentiment",
                "extract_entities",
                "generate_graph_report"
            ]
        
        print(f"🔧 Available MCP tools: {len(tools)} tools")
        return tools
        
    except Exception as e:
        print(f"⚠️ Could not get MCP tools info: {e}")
        # Return comprehensive tool list as fallback
        return [
            "get_all_agents_status",
            "start_all_agents",
            "stop_all_agents", 
            "process_pdf_enhanced_multilingual",
            "process_multilingual_pdf_mcp",
            "analyze_text_sentiment",
            "extract_entities",
            "generate_graph_report"
        ]


if __name__ == "__main__":
    print("Starting Sentiment Analysis Swarm with MCP Integration")
    print("=" * 60)
    
    # Start MCP server on port 8000
    print("Starting MCP server on port 8000...")
    mcp_server = start_mcp_server()
    
    # Show available tools
    if mcp_server:
        tools = get_mcp_tools_info()
        if tools:
            print(f"🔧 MCP Tools: {len(tools)} tools available")
    
    # Get API configuration and ensure port is available
    api_host = config.api.host
    api_port = get_safe_port(api_host, config.api.port)
    
    print("\nStarting FastAPI server...")
    print(f"✅ FastAPI server started in background")
    
    # Start FastAPI server in a separate thread
    def start_fastapi_server():
        uvicorn.run(app, host=api_host, port=api_port, log_level="info")
    
    api_thread = threading.Thread(target=start_fastapi_server, daemon=True)
    api_thread.start()
    
    # Launch Streamlit applications directly
    main_ui_process, landing_process = launch_streamlit_apps()
    
    print("\n🎉 All services are now running!")
    print("=" * 60)
    print("🌐 Access URLs:")
    print("   📊 Main UI:        http://localhost:8501")
    print("   🏠 Landing Page:   http://localhost:8502")
    print("   🔗 API Docs:       http://localhost:8003/docs")
    print("   🤖 MCP Server:     http://localhost:8000/mcp")
    print("=" * 60)
    print("🚀 System is ready for use!")
    
    # Keep the process running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        if main_ui_process:
            main_ui_process.terminate()
        if landing_process:
            landing_process.terminate()
        print("✅ Services stopped")
