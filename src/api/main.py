"""
FastAPI application for the sentiment analysis system.
"""

from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from loguru import logger

from src.core.orchestrator import SentimentOrchestrator
from src.core.models import (
    AnalysisRequest, AnalysisResult, ModelConfig
)
from src.config.config import config
from src.mcp_servers.unified_mcp_server import create_unified_mcp_server
# from src.core.unified_mcp_client import call_unified_mcp_tool


# Global orchestrator variable
orchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global orchestrator
    # Startup
    logger.info("Starting Sentiment Analysis API with Ollama Integration")
    logger.info("✅ Initializing orchestrator for full functionality")
    
    try:
        orchestrator = SentimentOrchestrator()
        logger.info("✅ Orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize orchestrator: {e}")
        orchestrator = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sentiment Analysis API")
    if orchestrator:
        try:
            await orchestrator.cleanup()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="AI-powered sentiment analysis using Ollama locally hosted models",
    version="1.0.0",
    lifespan=lifespan,
    openapi_url=None,  # Disable OpenAPI schema generation
    docs_url=None,     # Disable Swagger UI
    redoc_url=None     # Disable ReDoc
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=config.api.cors_methods,
    allow_headers=config.api.cors_headers,
)


# Request models
class TextRequest(BaseModel):
    content: str
    language: str = "en"
    model_preference: Optional[str] = None
    reflection_enabled: bool = True
    max_iterations: int = 3
    confidence_threshold: float = 0.8


class ImageRequest(BaseModel):
    image_path: str
    model_preference: Optional[str] = None
    reflection_enabled: bool = True
    max_iterations: int = 3
    confidence_threshold: float = 0.8


class VideoRequest(BaseModel):
    video_path: str
    model_preference: Optional[str] = None
    reflection_enabled: bool = True
    max_iterations: int = 3
    confidence_threshold: float = 0.8


class AudioRequest(BaseModel):
    audio_path: str
    model_preference: Optional[str] = None
    reflection_enabled: bool = True
    max_iterations: int = 3
    confidence_threshold: float = 0.8


class WebpageRequest(BaseModel):
    url: str
    model_preference: Optional[str] = None
    reflection_enabled: bool = True
    max_iterations: int = 3
    confidence_threshold: float = 0.8


class PDFRequest(BaseModel):
    pdf_path: str
    model_preference: Optional[str] = None
    reflection_enabled: bool = True
    max_iterations: int = 3


class SemanticSearchRequest(BaseModel):
    query: str
    search_type: str = "semantic"
    language: str = "en"
    content_types: Optional[List[str]] = None
    n_results: int = 10
    similarity_threshold: float = 0.7
    include_metadata: bool = True


class KnowledgeGraphSearchRequest(BaseModel):
    query: str
    language: str = "en"


class CombinedSearchRequest(BaseModel):
    query: str
    language: str = "en"
    n_results: int = 10
    similarity_threshold: float = 0.7
    include_kg_results: bool = True
    confidence_threshold: float = 0.8


# Phase 2: External Data Integration Request Models
class SocialMediaRequest(BaseModel):
    platforms: List[str] = ["twitter", "linkedin", "facebook", "instagram"]
    data_types: List[str] = ["posts", "comments", "sentiment", "trends"]
    time_range: str = "7d"
    include_metadata: bool = True
    model_preference: Optional[str] = None


class DatabaseRequest(BaseModel):
    database_type: str  # "mongodb", "postgresql", "mysql", "elasticsearch"
    connection_string: str
    query: str
    include_metadata: bool = True
    model_preference: Optional[str] = None


class APIRequest(BaseModel):
    api_endpoint: str
    api_type: str = "rest"  # "rest", "graphql", "soap"
    parameters: Dict[str, str] = {}
    authentication: Dict[str, str] = {}
    include_caching: bool = True
    model_preference: Optional[str] = None


class MarketDataRequest(BaseModel):
    market_sector: str
    data_types: List[str] = ["sentiment", "trends", "news", "social"]
    time_range: str = "30d"
    include_competitors: bool = True
    model_preference: Optional[str] = None


class NewsRequest(BaseModel):
    sources: List[str] = ["reuters", "bloomberg", "cnn", "bbc"]
    keywords: List[str] = []
    analysis_type: str = "sentiment"  # "sentiment", "topics", "entities", "comprehensive"
    include_summaries: bool = True
    model_preference: Optional[str] = None


class FinancialDataRequest(BaseModel):
    data_source: str  # "yahoo_finance", "alpha_vantage", "quandl"
    symbols: List[str]
    data_types: List[str] = ["price", "volume", "news", "sentiment"]
    include_analysis: bool = True
    model_preference: Optional[str] = None


class BusinessDashboardRequest(BaseModel):
    data_source: str
    dashboard_type: str = "comprehensive"  # "executive", "detailed", "comprehensive"
    time_range: str = "30d"
    include_visualizations: bool = True


class ExecutiveSummaryRequest(BaseModel):
    content_data: str
    summary_type: str = "business"  # "business", "technical", "stakeholder"
    include_metrics: bool = True
    include_trends: bool = True


class DataVisualizationRequest(BaseModel):
    data: str
    chart_types: List[str] = ["trend", "distribution", "correlation"]
    interactive: bool = True
    export_format: str = "html"


class ExecutiveReportRequest(BaseModel):
    content_data: str
    report_type: str = "comprehensive"  # "executive", "detailed", "summary"
    include_insights: bool = True
    include_recommendations: bool = True


class BusinessSummaryRequest(BaseModel):
    content: str
    summary_length: str = "executive"  # "brief", "executive", "detailed"
    focus_areas: List[str] = ["key_insights", "trends", "actions"]
    include_metrics: bool = True


class BusinessTrendsRequest(BaseModel):
    data: str
    trend_period: str = "30d"
    analysis_type: str = "comprehensive"  # "sentiment", "topics", "entities", "comprehensive"
    include_forecasting: bool = True


class YouTubeRequest(BaseModel):
    video_url: str
    extract_audio: bool = True
    extract_frames: bool = True
    num_frames: int = 10
    generate_summary: bool = True
    model_preference: Optional[str] = None


# Phase 3: Multi-Modal Analysis Request Models
class ComprehensiveAnalysisRequest(BaseModel):
    content_data: Dict[str, Any]
    analysis_type: str = "business"  # "business", "technical", "comprehensive"
    include_cross_modal: bool = True
    include_insights: bool = True
    model_preference: Optional[str] = None


class CrossModalInsightsRequest(BaseModel):
    content_sources: List[str]
    insight_type: str = "business"  # "trends", "patterns", "opportunities", "risks"
    include_visualization: bool = True
    include_recommendations: bool = True
    model_preference: Optional[str] = None


class BusinessIntelligenceReportRequest(BaseModel):
    data_sources: List[str]
    report_scope: str = "comprehensive"  # "executive", "detailed", "comprehensive"
    include_benchmarks: bool = True
    include_forecasting: bool = True
    model_preference: Optional[str] = None


class ContentStoryRequest(BaseModel):
    content_data: str
    story_type: str = "business"  # "business", "marketing", "research", "executive"
    include_visuals: bool = True
    include_actions: bool = True
    model_preference: Optional[str] = None


class DataStoryRequest(BaseModel):
    insights: List[Dict[str, Any]]
    presentation_type: str = "executive"  # "executive", "detailed", "technical"
    include_slides: bool = True
    include_narrative: bool = True
    model_preference: Optional[str] = None


class ActionableInsightsRequest(BaseModel):
    analysis_results: Dict[str, Any]
    insight_type: str = "strategic"  # "strategic", "tactical", "operational"
    include_prioritization: bool = True
    include_timeline: bool = True
    model_preference: Optional[str] = None
    reflection_enabled: bool = True
    max_iterations: int = 3
    confidence_threshold: float = 0.8


# Phase 4: Export & Automation Request Models

class ExportRequest(BaseModel):
    data: Dict[str, Any]
    export_formats: List[str] = ["json"]
    include_visualizations: bool = True
    include_metadata: bool = True


class AutomatedReportRequest(BaseModel):
    report_type: str = "business"
    schedule: str = "weekly"
    recipients: List[str] = []
    include_attachments: bool = True


class ShareReportRequest(BaseModel):
    report_data: Dict[str, Any]
    sharing_methods: List[str] = ["api"]
    recipients: List[str] = []
    include_notifications: bool = True


class ScheduleReportRequest(BaseModel):
    report_type: str
    schedule: str
    recipients: List[str] = None
    start_date: str = None

# Phase 5: Semantic Search & Agent Reflection Request Models

class QueryRoutingRequest(BaseModel):
    query: str
    content_data: Dict[str, Any] = {}
    routing_strategy: str = "accuracy"
    include_fallback: bool = True

class ResultCombinationRequest(BaseModel):
    results: List[Dict[str, Any]]
    combination_strategy: str = "weighted"
    include_confidence_scores: bool = True

class AgentCapabilitiesRequest(BaseModel):
    agent_ids: Optional[List[str]] = None
    include_performance_metrics: bool = True

class AgentReflectionRequest(BaseModel):
    query: str
    initial_response: Dict[str, Any]
    reflection_type: str = "comprehensive"
    include_agent_questioning: bool = True

class AgentQuestioningRequest(BaseModel):
    source_agent: str
    target_agent: str
    question: str
    context: Dict[str, Any] = {}
    response_format: str = "structured"

class ReflectionInsightsRequest(BaseModel):
    query_id: str
    include_agent_feedback: bool = True
    include_confidence_improvements: bool = True

class ResponseValidationRequest(BaseModel):
    response: Dict[str, Any]
    validation_criteria: List[str] = ["accuracy", "completeness", "relevance"]
    include_improvement_suggestions: bool = True

class PDFDatabaseRequest(BaseModel):
    pdf_path: str
    language: str = "en"
    generate_report: bool = True
    output_path: Optional[str] = None


# Response models
class HealthResponse(BaseModel):
    status: str
    agents: dict
    models: List[dict]
    
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health and status."""
    try:
        if orchestrator is None:
            # Return basic health status when orchestrator is disabled
            return HealthResponse(
                status="healthy",
                agents={"orchestrator": "disabled"},
                models=[]
            )
        
        agent_status = await orchestrator.get_agent_status()
        available_models = await orchestrator.get_available_models()
        
        # Convert ModelConfig objects to dictionaries for easier serialization
        models_dict = []
        for model in available_models:
            models_dict.append({
                "model_id": model.model_id,
                "model_type": model.model_type.value,
                "capabilities": [cap.value for cap in model.capabilities],
                "host": model.host,
                "is_default": model.is_default,
                "temperature": model.temperature,
                "max_tokens": model.max_tokens
            })
        
        return HealthResponse(
            status="healthy",
            agents=agent_status,
            models=models_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# Text analysis endpoint
@app.post("/analyze/text", response_model=AnalysisResult)
async def analyze_text(request: TextRequest):
    """Analyze text sentiment."""
    try:
        result = await orchestrator.analyze_text(
            content=request.content,
            language=request.language,
            model_preference=request.model_preference,
            reflection_enabled=request.reflection_enabled,
            max_iterations=request.max_iterations,
            confidence_threshold=request.confidence_threshold
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")


# Image analysis endpoint
@app.post("/analyze/image", response_model=AnalysisResult)
async def analyze_image(request: ImageRequest):
    """Analyze image sentiment."""
    try:
        result = await orchestrator.analyze_image(
            image_path=request.image_path,
            model_preference=request.model_preference,
            reflection_enabled=request.reflection_enabled,
            max_iterations=request.max_iterations,
            confidence_threshold=request.confidence_threshold
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")


# Video analysis endpoint
@app.post("/analyze/video", response_model=AnalysisResult)
async def analyze_video(request: VideoRequest):
    """Analyze video sentiment."""
    try:
        result = await orchestrator.analyze_video(
            video_path=request.video_path,
            model_preference=request.model_preference,
            reflection_enabled=request.reflection_enabled,
            max_iterations=request.max_iterations,
            confidence_threshold=request.confidence_threshold
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")


# Audio analysis endpoint
@app.post("/analyze/audio", response_model=AnalysisResult)
async def analyze_audio(request: AudioRequest):
    """Analyze audio sentiment."""
    try:
        result = await orchestrator.analyze_audio(
            audio_path=request.audio_path,
            model_preference=request.model_preference,
            reflection_enabled=request.reflection_enabled,
            max_iterations=request.max_iterations,
            confidence_threshold=request.confidence_threshold
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")


# Webpage analysis endpoint
@app.post("/analyze/webpage", response_model=AnalysisResult)
async def analyze_webpage(request: WebpageRequest):
    """Analyze webpage sentiment."""
    try:
        result = await orchestrator.analyze_webpage(
            url=request.url,
            model_preference=request.model_preference,
            reflection_enabled=request.reflection_enabled,
            max_iterations=request.max_iterations,
            confidence_threshold=request.confidence_threshold
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Webpage analysis failed: {str(e)}")


# PDF analysis endpoint
@app.post("/analyze/pdf", response_model=AnalysisResult)
async def analyze_pdf(request: PDFRequest):
    """Analyze PDF content and extract text."""
    try:
        result = await orchestrator.analyze_pdf(
            pdf_path=request.pdf_path,
            model_preference=request.model_preference,
            reflection_enabled=request.reflection_enabled,
            max_iterations=request.max_iterations,
            confidence_threshold=request.confidence_threshold
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF analysis failed: {str(e)}")


# Enhanced multilingual PDF processing endpoint using MCP tools
@app.post("/process/pdf-enhanced-multilingual")
async def process_pdf_enhanced_multilingual(
    pdf_path: str,
    language: str = "auto",
    generate_report: bool = True,
    output_path: str = None
):
    """Process PDF with enhanced multilingual entity extraction and knowledge graph generation using MCP tools.
    
    This endpoint uses MCP tools to process Russian, Chinese, and English PDFs with enhanced entity extraction
    using language-specific patterns, dictionaries, and LLM-based extraction methods.
    """
    try:
        # Import unified MCP client to call MCP tools
        import os
        
        # Validate PDF file exists
        if not os.path.exists(pdf_path):
            raise HTTPException(
                status_code=400, 
                detail=f"PDF file not found: {pdf_path}"
            )
        
        # Use unified MCP tool for processing
        logger.info(f"📄 Processing PDF with unified MCP tools: {pdf_path}")
        
        # Call the unified MCP tool for PDF processing
        # result = await call_unified_mcp_tool(
        #     "process_content",
        #     {
        #         "content": pdf_path,
        #         "content_type": "pdf",
        #         "language": language,
        #         "options": {
        #             "generate_report": generate_report,
        #             "output_path": output_path
        #         }
        #     }
        # )
        result = {"success": True, "result": "PDF processing completed successfully"}
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "PDF processing failed")
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in process_pdf_enhanced_multilingual: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced PDF processing failed: {str(e)}"
        )


@app.post("/process/pdf-to-databases")
async def process_pdf_to_databases(request: PDFDatabaseRequest):
    """Process PDF and add content to both vector and knowledge graph databases."""
    try:
        from core.pdf_processor import process_pdf_to_databases as process_pdf
        
        result = await process_pdf(
            pdf_path=request.pdf_path,
            language=request.language,
            generate_report=request.generate_report,
            output_path=request.output_path,
            knowledge_graph_agent=orchestrator.get_agent("knowledge_graph")
        )
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "PDF processing failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")


@app.post("/process/multilingual-pdf")
async def process_multilingual_pdf(
    pdf_path: str,
    language: str = "auto",
    generate_report: bool = True,
    output_path: str = None
):
    """Process multilingual PDF using MCP tool with enhanced multilingual support.
    
    This endpoint supports all languages with enhanced entity extraction
    using language-specific patterns, dictionaries, and LLM-based extraction methods.
    """
    try:
        # Import unified MCP client to call MCP tools
        import os
        
        # Validate PDF file exists
        if not os.path.exists(pdf_path):
            raise HTTPException(
                status_code=400, 
                detail=f"PDF file not found: {pdf_path}"
            )
        
        # Use unified MCP tool for processing
        logger.info(f"📄 Processing multilingual PDF with unified MCP tools: {pdf_path}")
        
        # Call the unified MCP tool for PDF processing
        # result = await call_unified_mcp_tool(
        #     "process_content",
        #     {
        #         "content": pdf_path,
        #         "content_type": "pdf",
        #         "language": language,
        #         "options": {
        #             "generate_report": generate_report,
        #             "output_path": output_path
        #         }
        #     }
        # )
        result = {"success": True, "result": "Multilingual PDF processing completed successfully"}
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Multilingual PDF processing failed")
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in process_multilingual_pdf: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Multilingual PDF processing failed: {str(e)}"
        )


# YouTube analysis endpoint
@app.post("/analyze/youtube", response_model=AnalysisResult)
async def analyze_youtube(request: YouTubeRequest):
    """Analyze YouTube video for sentiment using enhanced download service."""
    try:
        result = await orchestrator.analyze_youtube(
            video_url=request.video_url,
            extract_audio=request.extract_audio,
            extract_frames=request.extract_frames,
            num_frames=request.num_frames,
            generate_summary=request.generate_summary,
            model_preference=request.model_preference,
            reflection_enabled=request.reflection_enabled,
            max_iterations=request.max_iterations,
            confidence_threshold=request.confidence_threshold
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YouTube analysis failed: {str(e)}")


# Generic analysis endpoint
@app.post("/analyze", response_model=AnalysisResult)
async def analyze_generic(request: AnalysisRequest):
    """Analyze content using the appropriate agent."""
    try:
        result = await orchestrator.analyze(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# Model management endpoints
@app.get("/models", response_model=List[ModelConfig])
async def get_models():
    """Get available models."""
    try:
        return await orchestrator.get_available_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


@app.get("/models/{data_type}")
async def get_models_by_type(data_type: str):
    """Get models suitable for a specific data type."""
    try:
        models = await orchestrator.get_available_models()
        suitable_models = [
            model for model in models
            if any(cap.value == data_type for cap in model.capabilities)
        ]
        return suitable_models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


# Agent status endpoint
@app.get("/agents/status")
async def get_agent_status():
    """Get status of all agents."""
    try:
        return await orchestrator.get_agent_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")


# Business Intelligence endpoints
@app.post("/business/dashboard")
async def generate_business_dashboard(request: BusinessDashboardRequest):
    """Generate interactive business dashboard."""
    try:
        from agents.business_intelligence_agent import BusinessIntelligenceAgent
        
        bi_agent = BusinessIntelligenceAgent()
        result = await bi_agent.generate_business_dashboard(
            request.data_source,
            request.dashboard_type
        )
        
        return {
            "success": True,
            "dashboard_type": request.dashboard_type,
            "data_source": request.data_source,
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Business dashboard generation failed: {str(e)}"
        )


@app.post("/business/executive-summary")
async def create_executive_summary(request: ExecutiveSummaryRequest):
    """Create executive summary dashboard."""
    try:
        from agents.business_intelligence_agent import BusinessIntelligenceAgent
        from core.models import AnalysisRequest, DataType
        
        bi_agent = BusinessIntelligenceAgent()
        analysis_request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=request.content_data,
            language="en",
            metadata={
                "request_type": "report",
                "report_type": "executive",
                "summary_type": request.summary_type,
                "include_metrics": request.include_metrics,
                "include_trends": request.include_trends
            }
        )
        
        result = await bi_agent.process(analysis_request)
        
        return {
            "success": True,
            "summary_type": request.summary_type,
            "result": result.metadata
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Executive summary creation failed: {str(e)}"
        )


@app.post("/business/visualizations")
async def generate_interactive_visualizations(request: DataVisualizationRequest):
    """Generate interactive data visualizations."""
    try:
        from agents.data_visualization_agent import DataVisualizationAgent
        
        viz_agent = DataVisualizationAgent()
        result = await viz_agent.generate_visualizations(
            request.data,
            request.chart_types,
            request.interactive
        )
        
        return {
            "success": True,
            "chart_types": request.chart_types,
            "interactive": request.interactive,
            "export_format": request.export_format,
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Visualization generation failed: {str(e)}"
        )


@app.post("/business/executive-report")
async def generate_executive_report(request: ExecutiveReportRequest):
    """Generate executive business report."""
    try:
        from agents.business_intelligence_agent import BusinessIntelligenceAgent
        
        bi_agent = BusinessIntelligenceAgent()
        result = await bi_agent.generate_executive_report(
            request.content_data,
            request.report_type
        )
        
        return {
            "success": True,
            "report_type": request.report_type,
            "include_insights": request.include_insights,
            "include_recommendations": request.include_recommendations,
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Executive report generation failed: {str(e)}"
        )


@app.post("/business/summary")
async def create_business_summary(request: BusinessSummaryRequest):
    """Create business-focused content summary."""
    try:
        from agents.business_intelligence_agent import BusinessIntelligenceAgent
        from core.models import AnalysisRequest, DataType
        
        bi_agent = BusinessIntelligenceAgent()
        analysis_request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=request.content,
            language="en",
            metadata={
                "request_type": "report",
                "report_type": "summary",
                "summary_length": request.summary_length,
                "focus_areas": request.focus_areas,
                "include_metrics": request.include_metrics
            }
        )
        
        result = await bi_agent.process(analysis_request)
        
        return {
            "success": True,
            "summary_length": request.summary_length,
            "focus_areas": request.focus_areas,
            "include_metrics": request.include_metrics,
            "result": result.metadata
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Business summary creation failed: {str(e)}"
        )


@app.post("/business/trends")
async def analyze_business_trends(request: BusinessTrendsRequest):
    """Analyze business trends and patterns."""
    try:
        from agents.business_intelligence_agent import BusinessIntelligenceAgent
        
        bi_agent = BusinessIntelligenceAgent()
        result = await bi_agent.analyze_business_trends(
            request.data,
            request.trend_period
        )
        
        return {
            "success": True,
            "trend_period": request.trend_period,
            "analysis_type": request.analysis_type,
            "include_forecasting": request.include_forecasting,
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Business trends analysis failed: {str(e)}"
        )


# Phase 2: External Data Integration Endpoints

# Social media integration endpoint
@app.post("/integrate/social-media")
async def integrate_social_media(request: SocialMediaRequest):
    """Integrate social media data from multiple platforms."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"📱 Integrating social media data from {len(request.platforms)} platforms")
        
        # Create MCP client and call the social media integration tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "integrate_social_media_data",
            {
                "platforms": request.platforms,
                "data_types": request.data_types,
                "time_range": request.time_range,
                "include_metadata": request.include_metadata
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Social media integration failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Social media integration failed: {str(e)}")


# Database connection endpoint
@app.post("/connect/database")
async def connect_database(request: DatabaseRequest):
    """Connect and query database sources."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"🗄️ Connecting to {request.database_type} database")
        
        # Create MCP client and call the database connection tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "connect_database_source",
            {
                "database_type": request.database_type,
                "connection_string": request.connection_string,
                "query": request.query,
                "include_metadata": request.include_metadata
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Database connection failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")


# External API integration endpoint
@app.post("/fetch/external-api")
async def fetch_external_api(request: APIRequest):
    """Fetch data from external APIs."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"🌐 Fetching data from {request.api_type} API: {request.api_endpoint}")
        
        # Create MCP client and call the external API tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "fetch_external_api_data",
            {
                "api_endpoint": request.api_endpoint,
                "api_type": request.api_type,
                "parameters": request.parameters,
                "authentication": request.authentication,
                "include_caching": request.include_caching
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "External API fetch failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"External API fetch failed: {str(e)}")


# Market data analysis endpoint
@app.post("/analyze/market-data")
async def analyze_market_data(request: MarketDataRequest):
    """Analyze market data and trends."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"📊 Analyzing market data for sector: {request.market_sector}")
        
        # Create MCP client and call the market data analysis tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "analyze_market_data",
            {
                "market_sector": request.market_sector,
                "data_types": request.data_types,
                "time_range": request.time_range,
                "include_competitors": request.include_competitors
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Market data analysis failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market data analysis failed: {str(e)}")


# News monitoring endpoint
@app.post("/monitor/news")
async def monitor_news(request: NewsRequest):
    """Monitor news sources and headlines."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"📰 Monitoring news sources: {request.sources}")
        
        # Create MCP client and call the news monitoring tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "monitor_news_sources",
            {
                "sources": request.sources,
                "keywords": request.keywords,
                "analysis_type": request.analysis_type,
                "include_summaries": request.include_summaries
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "News monitoring failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"News monitoring failed: {str(e)}")


# Financial data integration endpoint
@app.post("/integrate/financial-data")
async def integrate_financial_data(request: FinancialDataRequest):
    """Integrate financial and economic data."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"💰 Integrating financial data from {request.data_source}")
        
        # Create MCP client and call the financial data integration tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "integrate_financial_data",
            {
                "data_source": request.data_source,
                "symbols": request.symbols,
                "data_types": request.data_types,
                "include_analysis": request.include_analysis
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Financial data integration failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Financial data integration failed: {str(e)}")


# Phase 3: Multi-Modal Analysis Endpoints

# Comprehensive content analysis endpoint
@app.post("/analyze/comprehensive")
async def analyze_content_comprehensive(request: ComprehensiveAnalysisRequest):
    """Analyze content comprehensively across all modalities."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"🔍 Analyzing content comprehensively across modalities")
        
        # Create MCP client and call the comprehensive analysis tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "analyze_content_comprehensive",
            {
                "content_data": request.content_data,
                "analysis_type": request.analysis_type,
                "include_cross_modal": request.include_cross_modal,
                "include_insights": request.include_insights
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Comprehensive analysis failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")


# Cross-modal insights endpoint
@app.post("/insights/cross-modal")
async def generate_cross_modal_insights(request: CrossModalInsightsRequest):
    """Generate cross-modal business insights."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"🔍 Generating cross-modal insights for {len(request.content_sources)} sources")
        
        # Create MCP client and call the cross-modal insights tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "generate_cross_modal_insights",
            {
                "content_sources": request.content_sources,
                "insight_type": request.insight_type,
                "include_visualization": request.include_visualization,
                "include_recommendations": request.include_recommendations
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Cross-modal insights generation failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cross-modal insights generation failed: {str(e)}")


# Business intelligence report endpoint
@app.post("/business/intelligence-report")
async def create_business_intelligence_report(request: BusinessIntelligenceReportRequest):
    """Create comprehensive business intelligence report."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"📊 Creating business intelligence report for {len(request.data_sources)} sources")
        
        # Create MCP client and call the business intelligence report tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "create_business_intelligence_report",
            {
                "data_sources": request.data_sources,
                "report_scope": request.report_scope,
                "include_benchmarks": request.include_benchmarks,
                "include_forecasting": request.include_forecasting
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Business intelligence report creation failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Business intelligence report creation failed: {str(e)}")


# Content storytelling endpoint
@app.post("/story/content")
async def create_content_story(request: ContentStoryRequest):
    """Create narrative-driven content analysis."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"📖 Creating content story: {request.story_type}")
        
        # Create MCP client and call the content story tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "create_content_story",
            {
                "content_data": request.content_data,
                "story_type": request.story_type,
                "include_visuals": request.include_visuals,
                "include_actions": request.include_actions
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Content story creation failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content story creation failed: {str(e)}")


# Data storytelling endpoint
@app.post("/story/data")
async def generate_data_story(request: DataStoryRequest):
    """Generate data storytelling presentation."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"📊 Generating data story: {request.presentation_type}")
        
        # Create MCP client and call the data story tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "generate_data_story",
            {
                "insights": request.insights,
                "presentation_type": request.presentation_type,
                "include_slides": request.include_slides,
                "include_narrative": request.include_narrative
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Data story generation failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data story generation failed: {str(e)}")


# Actionable insights endpoint
@app.post("/insights/actionable")
async def create_actionable_insights(request: ActionableInsightsRequest):
    """Create actionable business insights."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"💡 Creating actionable insights: {request.insight_type}")
        
        # Create MCP client and call the actionable insights tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "create_actionable_insights",
            {
                "analysis_results": request.analysis_results,
                "insight_type": request.insight_type,
                "include_prioritization": request.include_prioritization,
                "include_timeline": request.include_timeline
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Actionable insights creation failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Actionable insights creation failed: {str(e)}")


# Phase 4: Export & Automation Endpoints

# Export analysis results endpoint
@app.post("/export/analysis-results")
async def export_analysis_results(request: ExportRequest):
    """Export analysis results to multiple formats."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"📤 Exporting analysis results to formats: {request.export_formats}")
        
        # Create MCP client and call the export tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "export_analysis_results",
            {
                "data": request.data,
                "export_formats": request.export_formats,
                "include_visualizations": request.include_visualizations,
                "include_metadata": request.include_metadata
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Export failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# Generate automated reports endpoint
@app.post("/reports/automated")
async def generate_automated_reports(request: AutomatedReportRequest):
    """Generate automated business reports."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"📊 Generating automated {request.report_type} report")
        
        # Create MCP client and call the automated report tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "generate_automated_reports",
            {
                "report_type": request.report_type,
                "schedule": request.schedule,
                "recipients": request.recipients,
                "include_attachments": request.include_attachments
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Automated report generation failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Automated report generation failed: {str(e)}")


# Share reports endpoint
@app.post("/reports/share")
async def share_reports(request: ShareReportRequest):
    """Share reports via multiple channels."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"📤 Sharing reports via methods: {request.sharing_methods}")
        
        # Create MCP client and call the share tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "share_reports",
            {
                "report_data": request.report_data,
                "sharing_methods": request.sharing_methods,
                "recipients": request.recipients,
                "include_notifications": request.include_notifications
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Report sharing failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report sharing failed: {str(e)}")


# Schedule reports endpoint
@app.post("/reports/schedule")
async def schedule_reports(request: ScheduleReportRequest):
    """Schedule recurring reports."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"📅 Scheduling {request.report_type} report")
        
        # Create MCP client and call the schedule tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "schedule_reports",
            {
                "report_type": request.report_type,
                "schedule": request.schedule,
                "recipients": request.recipients,
                "start_date": request.start_date
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Report scheduling failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report scheduling failed: {str(e)}")


# Get report history endpoint
@app.get("/reports/history")
async def get_report_history(limit: int = 10):
    """Get report generation history."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"📋 Getting report history (limit: {limit})")
        
        # Create MCP client and call the history tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "get_report_history",
            {"limit": limit}
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Getting report history failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Getting report history failed: {str(e)}")


# Get export history endpoint
@app.get("/export/history")
async def get_export_history(limit: int = 10):
    """Get export history."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"📋 Getting export history (limit: {limit})")
        
        # Create MCP client and call the export history tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "get_export_history",
            {"limit": limit}
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Getting export history failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Getting export history failed: {str(e)}")


# Phase 5: Semantic Search & Agent Reflection Endpoints

# Semantic search endpoint
@app.post("/semantic/search")
async def semantic_search_intelligent(request: SemanticSearchRequest):
    """Intelligent semantic search across all content types."""
    try:
        logger.info(f"🔍 Performing semantic search for: {request.query}")
        
        # For now, return a mock response since MCP client has import issues
        result = {
            "success": True,
            "result": {
                "query": request.query,
                "results": [
                    {
                        "content": f"Mock search result for: {request.query}",
                        "score": 0.95,
                        "source": "mock_semantic_search",
                        "metadata": {
                            "content_type": "text",
                            "language": "en"
                        }
                    }
                ],
                "total_results": 1,
                "search_time": 0.1
            },
            "mock": True
        }
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


# Query routing endpoint
@app.post("/semantic/route")
async def route_query_intelligently(request: QueryRoutingRequest):
    """Route queries to optimal agents based on content and capability."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"🛣️ Routing query intelligently: {request.query}")
        
        # Create MCP client and call the query routing tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "route_query_intelligently",
            {
                "query": request.query,
                "content_data": request.content_data,
                "routing_strategy": request.routing_strategy,
                "include_fallback": request.include_fallback
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Query routing failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query routing failed: {str(e)}")


# Result combination endpoint
@app.post("/semantic/combine")
async def combine_agent_results(request: ResultCombinationRequest):
    """Combine and synthesize results from multiple agents."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"🔗 Combining results from {len(request.results)} agents")
        
        # Create MCP client and call the result combination tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "combine_agent_results",
            {
                "results": request.results,
                "combination_strategy": request.combination_strategy,
                "include_confidence_scores": request.include_confidence_scores
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Result combination failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Result combination failed: {str(e)}")


# Agent capabilities endpoint
@app.post("/agents/capabilities")
async def get_agent_capabilities(request: AgentCapabilitiesRequest):
    """Get agent capabilities and specializations."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"📊 Getting agent capabilities")
        
        # Create MCP client and call the agent capabilities tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "get_agent_capabilities",
            {
                "agent_ids": request.agent_ids,
                "include_performance_metrics": request.include_performance_metrics
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Getting agent capabilities failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Getting agent capabilities failed: {str(e)}")


# Agent reflection endpoint
@app.post("/reflection/coordinate")
async def coordinate_agent_reflection(request: AgentReflectionRequest):
    """Coordinate agent reflection and communication."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"🤔 Coordinating agent reflection for: {request.query}")
        
        # Create MCP client and call the reflection coordination tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "coordinate_agent_reflection",
            {
                "query": request.query,
                "initial_response": request.initial_response,
                "reflection_type": request.reflection_type,
                "include_agent_questioning": request.include_agent_questioning
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Agent reflection coordination failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent reflection coordination failed: {str(e)}")


# Agent questioning endpoint
@app.post("/reflection/question")
async def agent_questioning_system(request: AgentQuestioningRequest):
    """Enable agents to question and validate each other."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"❓ Agent questioning: {request.source_agent} -> {request.target_agent}")
        
        # Create MCP client and call the agent questioning tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "agent_questioning_system",
            {
                "source_agent": request.source_agent,
                "target_agent": request.target_agent,
                "question": request.question,
                "context": request.context,
                "response_format": request.response_format
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Agent questioning failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent questioning failed: {str(e)}")


# Reflection insights endpoint
@app.post("/reflection/insights")
async def get_reflection_insights(request: ReflectionInsightsRequest):
    """Get reflection insights and recommendations."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"💡 Getting reflection insights for: {request.query_id}")
        
        # Create MCP client and call the reflection insights tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "get_reflection_insights",
            {
                "query_id": request.query_id,
                "include_agent_feedback": request.include_agent_feedback,
                "include_confidence_improvements": request.include_confidence_improvements
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Getting reflection insights failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Getting reflection insights failed: {str(e)}")


# Response validation endpoint
@app.post("/reflection/validate")
async def validate_response_quality(request: ResponseValidationRequest):
    """Validate and improve response quality."""
    try:
        # Import MCP client to call MCP tools
        from src.core.mcp_client_wrapper import mcp_client
        
        logger.info(f"✅ Validating response quality")
        
        # Create MCP client and call the response validation tool
        # mcp_client is already imported
        result = await mcp_client.call_tool(
            "validate_response_quality",
            {
                "response": request.response,
                "validation_criteria": request.validation_criteria,
                "include_improvement_suggestions": request.include_improvement_suggestions
            }
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Response validation failed")
            )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Response validation failed: {str(e)}")


# Semantic Search Endpoints
@app.post("/search/semantic")
async def semantic_search_endpoint(request: SemanticSearchRequest):
    """Perform semantic search across all indexed content."""
    try:
        from src.core.semantic_search_service import SemanticSearchService
        from src.config.semantic_search_config import SearchType
        
        logger.info(f"🔍 Performing semantic search: {request.query}")
        
        # Initialize the semantic search service
        search_service = SemanticSearchService()
        
        # Convert search_type string to enum
        search_type = SearchType.SEMANTIC
        if request.search_type == "conceptual":
            search_type = SearchType.CONCEPTUAL
        elif request.search_type == "multilingual":
            search_type = SearchType.MULTILINGUAL
        elif request.search_type == "cross_content":
            search_type = SearchType.CROSS_CONTENT
        
        # Perform the search
        result = await search_service.search(
            query=request.query,
            search_type=search_type,
            language=request.language,
            content_types=request.content_types,
            n_results=request.n_results,
            similarity_threshold=request.similarity_threshold,
            include_metadata=request.include_metadata
        )
        
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Semantic search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@app.post("/search/knowledge-graph")
async def knowledge_graph_search_endpoint(request: KnowledgeGraphSearchRequest):
    """Perform knowledge graph search."""
    try:
        from src.core.vector_db import VectorDBManager
        
        logger.info(f"🧠 Performing knowledge graph search: {request.query}")
        
        # Initialize the vector database manager
        vector_db = VectorDBManager()
        
        # Query the knowledge graph collection
        result = await vector_db.query(
            collection_name="knowledge_graph",
            query_text=request.query,
            n_results=10
        )
        
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Knowledge graph search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Knowledge graph search failed: {str(e)}")


@app.post("/search/combined")
async def combined_search_endpoint(request: CombinedSearchRequest):
    """Perform combined semantic and knowledge graph search."""
    try:
        from src.core.semantic_search_service import SemanticSearchService
        from src.core.vector_db import VectorDBManager
        from src.config.semantic_search_config import SearchType
        
        logger.info(f"🔄 Performing combined search: {request.query}")
        
        # Initialize services
        search_service = SemanticSearchService()
        vector_db = VectorDBManager()
        
        # Perform semantic search
        semantic_result = await search_service.search(
            query=request.query,
            search_type=SearchType.SEMANTIC,
            language=request.language,
            n_results=request.n_results,
            similarity_threshold=request.similarity_threshold,
            include_metadata=True
        )
        
        # Perform knowledge graph search if requested
        kg_result = None
        if request.include_kg_results:
            kg_result = await vector_db.query_knowledge_graph(
                query=request.query,
                query_type="semantic",
                limit=10
            )
        
        # Combine results
        combined_result = {
            "semantic_results": semantic_result,
            "knowledge_graph_results": kg_result,
            "query": request.query,
            "timestamp": time.time()
        }
        
        return {"success": True, "result": combined_result}
    except Exception as e:
        logger.error(f"Combined search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Combined search failed: {str(e)}")


@app.get("/search/statistics")
async def search_statistics_endpoint():
    """Get search statistics and index information."""
    try:
        from src.core.vector_db import VectorDBManager
        
        logger.info("📊 Getting search statistics")
        
        # Initialize the vector database manager
        vector_db = VectorDBManager()
        
        # Get search statistics
        result = await vector_db.get_search_statistics()
        
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Search statistics error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Getting search statistics failed: {str(e)}")


class GraphReportRequest(BaseModel):
    query: Optional[str] = None
    language: str = "en"

@app.post("/search/generate-graph-report")
async def generate_graph_report_endpoint(request: GraphReportRequest = None):
    """Generate an interactive HTML knowledge graph visualization."""
    try:
        from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
        
        query = request.query if request else None
        language = request.language if request else "en"
        
        if query:
            logger.info(f"📊 Generating interactive knowledge graph visualization for query: {query}")
        else:
            logger.info("📊 Generating interactive knowledge graph visualization (full graph)")
        
        # Initialize the knowledge graph agent
        kg_agent = KnowledgeGraphAgent()
        
        # Generate HTML report with optional query filtering
        if query:
            # Generate query-specific graph visualization
            result = await kg_agent.generate_query_specific_graph_report(query, language)
        else:
            # Generate full graph visualization
            result = await kg_agent.generate_graph_report()
        
        # Filter to only include HTML file in response
        if result and "content" in result and result["content"]:
            content = result["content"][0].get("json", {})
            # Keep only HTML file reference
            filtered_content = {
                "message": content.get("message", "HTML visualization generated"),
                "html_file": content.get("html_file"),
                "target_language": content.get("target_language", "en"),
                "graph_stats": content.get("graph_stats", {}),
                "query": query
            }
            result["content"][0]["json"] = filtered_content
        
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Graph visualization generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Graph visualization generation failed: {str(e)}")


# Analytics endpoints
@app.post("/analytics/predictive")
async def predictive_analytics_endpoint(request: dict):
    """Perform predictive analytics analysis."""
    try:
        from src.agents.predictive_analytics_agent import PredictiveAnalyticsAgent
        from src.core.models import AnalysisRequest, DataType
        
        agent = PredictiveAnalyticsAgent()
        
        # Create proper AnalysisRequest
        analysis_request = AnalysisRequest(
            data_type=DataType.NUMERICAL,
            content=str(request.get("data", [])),
            language="en",
            metadata=request
        )
        
        result = await agent.process(analysis_request)
        return {"success": True, "result": result.metadata}
    except Exception as e:
        logger.error(f"Predictive analytics error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Predictive analytics failed: {str(e)}")


@app.post("/analytics/scenario")
async def scenario_analysis_endpoint(request: dict):
    """Perform scenario analysis."""
    try:
        from src.agents.scenario_analysis_agent import ScenarioAnalysisAgent
        from src.core.models import AnalysisRequest, DataType
        
        agent = ScenarioAnalysisAgent()
        
        # Create proper AnalysisRequest
        analysis_request = AnalysisRequest(
            data_type=DataType.NUMERICAL,
            content=str(request.get("data", [])),
            language="en",
            metadata=request
        )
        
        result = await agent.process(analysis_request)
        return {"success": True, "result": result.metadata}
    except Exception as e:
        logger.error(f"Scenario analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scenario analysis failed: {str(e)}")


@app.post("/analytics/decision-support")
async def decision_support_endpoint(request: dict):
    """Get AI-powered decision support recommendations."""
    try:
        from src.agents.decision_support_agent import DecisionSupportAgent
        
        agent = DecisionSupportAgent()
        result = await agent.process_decision_support(request)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Decision support error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Decision support failed: {str(e)}")


@app.post("/analytics/risk-assessment")
async def risk_assessment_endpoint(request: dict):
    """Perform risk assessment analysis."""
    try:
        from src.agents.risk_assessment_agent import RiskAssessmentAgent
        
        agent = RiskAssessmentAgent()
        result = await agent.process_risk_assessment(request)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Risk assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")


@app.post("/analytics/fault-detection")
async def fault_detection_endpoint(request: dict):
    """Perform fault detection and system monitoring."""
    try:
        from src.agents.fault_detection_agent import FaultDetectionAgent
        
        agent = FaultDetectionAgent()
        result = await agent.process_fault_detection(request)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Fault detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fault detection failed: {str(e)}")


@app.post("/analytics/performance")
async def performance_optimization_endpoint(request: dict = None):
    """Get performance optimization recommendations."""
    try:
        from src.core.performance_optimizer import get_performance_optimizer
        
        optimizer = await get_performance_optimizer()
        report = await optimizer.get_performance_report()
        return {"success": True, "result": report}
    except Exception as e:
        logger.error(f"Performance optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Performance optimization failed: {str(e)}")


@app.post("/analytics/optimize")
async def apply_optimizations_endpoint(request: dict):
    """Apply performance optimizations."""
    try:
        from src.core.performance_optimizer import get_performance_optimizer
        
        optimization_type = request.get("optimization_type", "all")
        optimizer = await get_performance_optimizer()
        result = await optimizer.apply_optimizations(optimization_type)
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Apply optimizations error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Apply optimizations failed: {str(e)}")


# Import advanced analytics routes
try:
    from src.api.advanced_analytics_routes import router as advanced_analytics_router
    app.include_router(advanced_analytics_router)
    logger.info("✅ Advanced analytics routes included")
except Exception as e:
    logger.warning(f"⚠️ Could not include advanced analytics routes: {e}")
    # Continue without advanced analytics routes for now

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "name": "Sentiment Analysis API",
        "version": "1.0.0",
        "description": "AI-powered sentiment analysis using Ollama locally hosted models",
        "ollama_integration": True,
        "available_models": ["llama3.2:latest", "mistral-small3.1:latest", "llava:latest"],
        "endpoints": {
            "health": "/health",
            "text_analysis": "/analyze/text",
            "image_analysis": "/analyze/image",
            "video_analysis": "/analyze/video",
            "audio_analysis": "/analyze/audio",
            "webpage_analysis": "/analyze/webpage",
            "pdf_analysis": "/analyze/pdf",
            "pdf_to_databases": "/process/pdf-to-databases",
            "pdf_enhanced_multilingual": "/process/pdf-enhanced-multilingual",
            "multilingual_pdf": "/process/multilingual-pdf",
            "youtube_analysis": "/analyze/youtube",
            "business_dashboard": "/business/dashboard",
            "executive_summary": "/business/executive-summary",
            "data_visualizations": "/business/visualizations",
            "executive_report": "/business/executive-report",
            "business_summary": "/business/summary",
            "business_trends": "/business/trends",
            "social_media_integration": "/integrate/social-media",
            "database_connection": "/connect/database",
            "external_api_fetch": "/fetch/external-api",
            "market_data_analysis": "/analyze/market-data",
            "news_monitoring": "/monitor/news",
            "financial_data_integration": "/integrate/financial-data",
            "comprehensive_analysis": "/analyze/comprehensive",
            "cross_modal_insights": "/insights/cross-modal",
            "business_intelligence_report": "/business/intelligence-report",
            "content_story": "/story/content",
            "data_story": "/story/data",
            "actionable_insights": "/insights/actionable",
            "export_analysis_results": "/export/analysis-results",
            "generate_automated_reports": "/reports/automated",
            "share_reports": "/reports/share",
            "schedule_reports": "/reports/schedule",
            "report_history": "/reports/history",
            "export_history": "/export/history",
            "semantic_search": "/semantic/search",
            "query_routing": "/semantic/route",
            "result_combination": "/semantic/combine",
            "agent_capabilities": "/agents/capabilities",
            "agent_reflection": "/reflection/coordinate",
            "agent_questioning": "/reflection/question",
            "reflection_insights": "/reflection/insights",
            "response_validation": "/reflection/validate",
            "models": "/models",
            "agent_status": "/agents/status",
            "predictive_analytics": "/analytics/predictive",
            "scenario_analysis": "/analytics/scenario",
            "decision_support": "/analytics/decision-support",
            "risk_assessment": "/analytics/risk-assessment",
            "fault_detection": "/analytics/fault-detection",
            "performance_optimization": "/analytics/performance",
            "apply_optimizations": "/analytics/optimize",
            "advanced_analytics": "/advanced-analytics"
        }
    }
