"""
FastAPI application for the sentiment analysis system.
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from core.orchestrator import SentimentOrchestrator
from core.models import (
    AnalysisRequest, AnalysisResult, DataType, SentimentResult,
    ModelConfig
)
from config.config import config


# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis Swarm API",
    description="AI-powered sentiment analysis using agent swarm architecture",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=config.api.cors_methods,
    allow_headers=config.api.cors_headers,
)

# Initialize orchestrator
orchestrator = SentimentOrchestrator()


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


# Response models
class HealthResponse(BaseModel):
    status: str
    agents: dict
    models: List[ModelConfig]
    
    class Config:
        from_attributes = True


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health and status."""
    try:
        agent_status = await orchestrator.get_agent_status()
        available_models = await orchestrator.get_available_models()
        
        return HealthResponse(
            status="healthy",
            agents=agent_status,
            models=available_models
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


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    logger.info("Starting Sentiment Analysis Swarm API")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    logger.info("Shutting down Sentiment Analysis Swarm API")
    await orchestrator.cleanup()


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "name": "Sentiment Analysis Swarm API",
        "version": "1.0.0",
        "description": "AI-powered sentiment analysis using agent swarm architecture",
        "endpoints": {
            "health": "/health",
            "text_analysis": "/analyze/text",
            "image_analysis": "/analyze/image",
            "video_analysis": "/analyze/video",
            "audio_analysis": "/analyze/audio",
            "webpage_analysis": "/analyze/webpage",
            "models": "/models",
            "agent_status": "/agents/status"
        }
    }
