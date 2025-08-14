"""
Simplified FastAPI application for testing OpenAPI generation.
"""

from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Simplified Sentiment Analysis API")
    yield
    # Shutdown
    logger.info("Shutting down Simplified Sentiment Analysis API")


# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="AI-powered sentiment analysis using Ollama",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request models
class TextRequest(BaseModel):
    content: str
    language: str = "en"
    model_preference: Optional[str] = None


class BusinessDashboardRequest(BaseModel):
    data_source: str
    dashboard_type: str = "comprehensive"
    time_range: str = "30d"
    include_visualizations: bool = True


class ExecutiveSummaryRequest(BaseModel):
    content_data: str
    summary_type: str = "business"
    include_metrics: bool = True
    include_trends: bool = True


# Health endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "status": "running",
        "ollama_integration": True
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "ollama_models": ["llama3.2:latest", "mistral-small3.1:latest"],
        "endpoints": [
            "/analyze/text",
            "/business/dashboard",
            "/business/executive-summary"
        ]
    }


# Analysis endpoints
@app.post("/analyze/text")
async def analyze_text(request: TextRequest):
    """Analyze text content using Ollama."""
    try:
        # Simulate analysis with Ollama
        result = {
            "content": request.content,
            "language": request.language,
            "model_used": request.model_preference or "llama3.2:latest",
            "sentiment": "positive",
            "confidence": 0.85,
            "entities": ["sample", "entity"],
            "summary": f"Analysis of {len(request.content)} characters"
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Business endpoints
@app.post("/business/dashboard")
async def generate_business_dashboard(request: BusinessDashboardRequest):
    """Generate business dashboard using Ollama."""
    try:
        result = {
            "dashboard_type": request.dashboard_type,
            "data_source": request.data_source,
            "ollama_model": "mistral-small3.1:latest",
            "charts": [
                {
                    "type": "pie",
                    "title": "Sentiment Overview",
                    "data": {"labels": ["Positive", "Neutral", "Negative"], "values": [60, 25, 15]}
                }
            ],
            "summary": "Business dashboard generated with Ollama",
            "metrics": {"total_analyses": 150, "average_sentiment": 0.75}
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/business/executive-summary")
async def create_executive_summary(request: ExecutiveSummaryRequest):
    """Create executive summary using Ollama."""
    try:
        result = {
            "summary_type": request.summary_type,
            "ollama_model": "mistral-small3.1:latest",
            "executive_summary": f"Executive summary of {len(request.content_data)} characters",
            "key_insights": ["Positive trend", "Growth opportunity", "Market strength"],
            "recommendations": ["Continue strategy", "Monitor trends", "Enhance engagement"],
            "metrics": {"sentiment_score": 0.75, "engagement_rate": 0.85}
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)

