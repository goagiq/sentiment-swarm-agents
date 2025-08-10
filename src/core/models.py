"""
Core data models for the sentiment analysis system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    UNCERTAIN = "uncertain"


class ProcessingStatus(str, Enum):
    """Processing status for analysis requests."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataType(str, Enum):
    """Supported data types for analysis."""
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    WEBPAGE = "webpage"
    PDF = "pdf"
    API_RESPONSE = "api_response"
    SOCIAL_MEDIA = "social_media"


class ModelType(str, Enum):
    """Supported model types."""
    OLLAMA = "ollama"
    # Removed unused model types: HUGGINGFACE, OPENAI, ANTHROPIC


class ModelCapability(str, Enum):
    """Model capabilities."""
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    TOOL_CALLING = "tool_calling"
    MULTIMODAL = "multimodal"


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    model_id: str
    model_type: ModelType
    capabilities: List[ModelCapability]
    host: Optional[str] = "http://localhost:11434"  # Default Ollama host
    api_key: Optional[str] = None  # Not needed for Ollama
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    vision_temperature: Optional[float] = None
    vision_max_tokens: Optional[int] = None
    keep_alive: Optional[str] = "5m"  # Ollama keep_alive parameter
    top_p: Optional[float] = None  # Ollama top_p parameter
    stop_sequences: Optional[List[str]] = None  # Ollama stop sequences
    options: Optional[Dict[str, Any]] = None  # Additional Ollama options
    is_default: bool = False


class AnalysisRequest(BaseModel):
    """Request for sentiment analysis."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    data_type: DataType
    content: Union[str, bytes, Dict[str, Any]]
    language: str = "en"
    model_preference: Optional[str] = None  # Specific model to use
    reflection_enabled: bool = True  # Enable reflection for quality improvement
    max_iterations: int = 3  # Maximum reflection iterations
    confidence_threshold: float = 0.8  # Minimum confidence threshold


class SentimentResult(BaseModel):
    """Result of sentiment analysis."""
    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    scores: Dict[str, float] = Field(default_factory=dict)
    reasoning: Optional[str] = None
    context_notes: Optional[str] = None
    uncertainty_factors: Optional[List[str]] = None
    reflection_notes: Optional[List[str]] = None
    iteration_count: int = 1


class AnalysisResult(BaseModel):
    """Complete analysis result."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str
    data_type: DataType
    sentiment: SentimentResult
    processing_time: float
    status: Optional[str] = None
    raw_content: Optional[str] = None
    extracted_text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_used: Optional[str] = None
    reflection_enabled: bool = True
    quality_score: Optional[float] = None


class ModelRegistry(BaseModel):
    """Registry of available models."""
    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    default_models: Dict[DataType, str] = Field(default_factory=dict)
    
    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get a model by ID."""
        return self.models.get(model_id)
    
    def get_default_model(self, data_type: DataType) -> Optional[ModelConfig]:
        """Get the default model for a data type."""
        model_id = self.default_models.get(data_type)
        if model_id:
            return self.models.get(model_id)
        return None
    
    def register_model(self, model: ModelConfig):
        """Register a new model."""
        self.models[model.model_id] = model
        if model.is_default:
            # Set as default for all supported capabilities
            for capability in model.capabilities:
                if capability in [ModelCapability.VISION, ModelCapability.AUDIO]:
                    if capability == ModelCapability.VISION:
                        self.default_models[DataType.IMAGE] = model.model_id
                        self.default_models[DataType.VIDEO] = model.model_id
                    elif capability == ModelCapability.AUDIO:
                        self.default_models[DataType.AUDIO] = model.model_id


class ReflectionResult(BaseModel):
    """Result of agent reflection."""
    iteration: int
    confidence: float
    reasoning: str
    alternative_hypotheses: List[str] = Field(default_factory=list)
    validation_checks: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
    final_assessment: bool = False
