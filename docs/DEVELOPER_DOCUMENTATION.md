# Developer Documentation - Sentiment Analysis & Decision Support System

**Version:** 1.0.0  
**Last Updated:** 2025-08-14

## Table of Contents

1. [Development Setup Guide](#development-setup-guide)
2. [Architecture Documentation](#architecture-documentation)
3. [Code Documentation](#code-documentation)
4. [Contribution Guidelines](#contribution-guidelines)
5. [Testing Procedures](#testing-procedures)
6. [API Development](#api-development)
7. [Database Schema](#database-schema)
8. [Deployment Development](#deployment-development)

## Development Setup Guide

### Prerequisites

- **Python**: 3.8+ (recommended: 3.11)
- **Node.js**: 16+ (for frontend development)
- **Docker**: 20.10+ (for containerization)
- **Git**: 2.30+
- **IDE**: VS Code, PyCharm, or similar

### Initial Setup

```bash
# Clone repository
git clone https://github.com/your-org/sentiment-analysis-system.git
cd sentiment-analysis-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements.dev.txt

# Install pre-commit hooks
pre-commit install

# Set up environment variables
cp env.example .env
# Edit .env with your configuration
```

### Development Environment Configuration

```bash
# Development environment variables
DEBUG=True
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://user:password@localhost:5432/sentiment_dev
REDIS_URL=redis://localhost:6379
OLLAMA_URL=http://localhost:11434
```

### Database Setup

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib  # Ubuntu
# or
brew install postgresql  # macOS

# Create development database
sudo -u postgres createdb sentiment_dev

# Run migrations
python -m src.core.database.migrations

# Seed development data
python -m src.core.database.seed --env=development
```

### Running the Application

```bash
# Start development server
python main.py

# Or use uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8003

# Run with debug mode
DEBUG=True python main.py
```

### Development Tools

#### Code Quality Tools

```bash
# Run linting
flake8 src/
black src/
isort src/

# Run type checking
mypy src/

# Run security checks
bandit -r src/

# Run all quality checks
pre-commit run --all-files
```

#### Testing Tools

```bash
# Run unit tests
pytest Test/unit/

# Run integration tests
pytest Test/integration/

# Run all tests with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest Test/test_specific_feature.py -v
```

## Architecture Documentation

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Web UI    │  │  Mobile App │  │   API Docs  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway Layer                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   FastAPI   │  │   MCP Tools │  │   Auth      │        │
│  │  Server     │  │   Server    │  │   Service   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Business Logic Layer                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Sentiment   │  │ Decision    │  │ Business    │        │
│  │ Analysis    │  │ Support     │  │ Intelligence│        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Multi-Modal │  │ Advanced    │  │ Knowledge   │        │
│  │ Processing  │  │ Analytics   │  │ Graph       │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ PostgreSQL  │  │   Redis     │  │   ChromaDB  │        │
│  │  Database   │  │   Cache     │  │ Vector DB   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    AI/ML Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Ollama    │  │   Custom    │  │   External  │        │
│  │   Models    │  │   Models    │  │   APIs      │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

#### 1. API Layer (`src/api/`)

```python
# FastAPI Application Structure
src/api/
├── main.py                 # Main FastAPI application
├── main_simple.py          # Simplified API for testing
├── routes/
│   ├── sentiment.py        # Sentiment analysis endpoints
│   ├── decision.py         # Decision support endpoints
│   ├── analytics.py        # Analytics endpoints
│   └── health.py           # Health check endpoints
└── middleware/
    ├── auth.py             # Authentication middleware
    ├── logging.py          # Logging middleware
    └── cors.py             # CORS middleware
```

#### 2. Core Business Logic (`src/core/`)

```python
# Core Components Structure
src/core/
├── agents/                 # AI Agent implementations
│   ├── sentiment_agent.py
│   ├── decision_support_agent.py
│   ├── business_intelligence_agent.py
│   └── multi_modal_agent.py
├── analytics/              # Analytics engines
│   ├── predictive_analytics.py
│   ├── causal_analysis.py
│   └── scenario_analysis.py
├── data_processing/        # Data processing utilities
│   ├── text_processor.py
│   ├── audio_processor.py
│   └── video_processor.py
└── storage/                # Data storage management
    ├── database_manager.py
    ├── cache_manager.py
    └── vector_store.py
```

#### 3. Configuration (`src/config/`)

```python
# Configuration Structure
src/config/
├── settings.py             # Main application settings
├── database.py             # Database configuration
├── ollama.py               # Ollama model configuration
├── security.py             # Security settings
└── language_config/        # Language-specific configurations
    ├── english_config.py
    ├── chinese_config.py
    └── russian_config.py
```

### Data Flow Architecture

#### 1. Request Processing Flow

```
Client Request
    │
    ▼
┌─────────────┐
│   FastAPI   │ ← API Gateway
│   Router    │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Middleware  │ ← Authentication, Logging, CORS
│   Layer     │
└─────────────┘
    │
    ▼
┌─────────────┐
│   Agent     │ ← Business Logic Processing
│   Layer     │
└─────────────┘
    │
    ▼
┌─────────────┐
│   Data      │ ← Database/Cache Operations
│   Layer     │
└─────────────┘
    │
    ▼
┌─────────────┐
│   AI/ML     │ ← Model Inference
│   Layer     │
└─────────────┘
    │
    ▼
Client Response
```

#### 2. Multi-Modal Processing Flow

```
Input Data (Text/Audio/Video/Image)
    │
    ▼
┌─────────────┐
│ Preprocessor│ ← Data validation and preprocessing
└─────────────┘
    │
    ▼
┌─────────────┐
│ Feature     │ ← Feature extraction
│ Extractor   │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Model       │ ← AI model inference
│ Inference   │
└─────────────┘
    │
    ▼
┌─────────────┐
│ Post-       │ ← Result processing and formatting
│ Processor   │
└─────────────┘
    │
    ▼
Processed Results
```

## Code Documentation

### Code Style Guidelines

#### 1. Python Code Style

```python
# Follow PEP 8 guidelines
# Use type hints for all functions
# Document all public functions and classes

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Represents the result of a sentiment analysis."""
    
    sentiment: str
    confidence: float
    entities: List[str]
    metadata: Dict[str, Any]


class SentimentAnalyzer:
    """Handles sentiment analysis operations."""
    
    def __init__(self, model_name: str = "llama3.2:latest"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def analyze_text(
        self, 
        text: str, 
        language: str = "en"
    ) -> AnalysisResult:
        """
        Analyze sentiment of the given text.
        
        Args:
            text: Text to analyze
            language: Language code (default: "en")
            
        Returns:
            AnalysisResult containing sentiment analysis
            
        Raises:
            ValueError: If text is empty or invalid
            ModelError: If model inference fails
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            # Implementation here
            result = await self._process_text(text, language)
            return result
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise ModelError(f"Model inference failed: {e}")
```

#### 2. API Documentation

```python
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


class TextAnalysisRequest(BaseModel):
    """Request model for text analysis."""
    
    content: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    language: str = Field(default="en", description="Language code")
    model_preference: Optional[str] = Field(None, description="Preferred model")


class TextAnalysisResponse(BaseModel):
    """Response model for text analysis."""
    
    sentiment: str = Field(..., description="Sentiment classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    processing_time: float = Field(..., description="Processing time in seconds")


@router.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(
    request: TextAnalysisRequest,
    analyzer: SentimentAnalyzer = Depends(get_analyzer)
) -> TextAnalysisResponse:
    """
    Analyze sentiment of the provided text.
    
    This endpoint performs comprehensive sentiment analysis including:
    - Sentiment classification (positive/negative/neutral)
    - Confidence scoring
    - Entity extraction
    - Processing time measurement
    
    Args:
        request: Text analysis request
        analyzer: Injected sentiment analyzer
        
    Returns:
        TextAnalysisResponse with analysis results
        
    Raises:
        HTTPException: If analysis fails or input is invalid
    """
    try:
        result = await analyzer.analyze_text(
            text=request.content,
            language=request.language
        )
        
        return TextAnalysisResponse(
            sentiment=result.sentiment,
            confidence=result.confidence,
            entities=result.entities,
            processing_time=result.processing_time
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### Error Handling

#### 1. Custom Exceptions

```python
class SentimentAnalysisError(Exception):
    """Base exception for sentiment analysis errors."""
    pass


class ModelError(SentimentAnalysisError):
    """Raised when model inference fails."""
    pass


class ValidationError(SentimentAnalysisError):
    """Raised when input validation fails."""
    pass


class ConfigurationError(SentimentAnalysisError):
    """Raised when configuration is invalid."""
    pass
```

#### 2. Error Handling Patterns

```python
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def handle_errors(func):
    """Decorator for consistent error handling."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValidationError as e:
            logger.warning(f"Validation error in {func.__name__}: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        except ModelError as e:
            logger.error(f"Model error in {func.__name__}: {e}")
            raise HTTPException(status_code=503, detail="Service temporarily unavailable")
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    return wrapper
```

## Contribution Guidelines

### Development Workflow

#### 1. Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and commit
git add .
git commit -m "feat: add new sentiment analysis feature"

# Push branch
git push origin feature/new-feature

# Create pull request
# Follow PR template and guidelines
```

#### 2. Commit Message Convention

```bash
# Commit message format
<type>(<scope>): <description>

# Types
feat:     New feature
fix:      Bug fix
docs:     Documentation changes
style:    Code style changes (formatting, etc.)
refactor: Code refactoring
test:     Adding or updating tests
chore:    Maintenance tasks

# Examples
feat(sentiment): add multi-language support
fix(api): resolve authentication issue
docs(readme): update installation instructions
test(analytics): add unit tests for forecasting
```

#### 3. Pull Request Guidelines

```markdown
## Pull Request Template

### Description
Brief description of changes

### Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

### Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

### Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes
```

### Code Review Process

#### 1. Review Checklist

- [ ] Code follows style guidelines
- [ ] Type hints are used
- [ ] Error handling is appropriate
- [ ] Tests are included
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance considerations addressed

#### 2. Review Comments

```python
# Good review comment
# Consider using a more descriptive variable name here
# Current: data = process_input(input)
# Suggested: processed_text = process_input(raw_text)

# Bad review comment
# This is wrong
```

## Testing Procedures

### Test Structure

```python
# Test file structure
Test/
├── unit/                    # Unit tests
│   ├── test_sentiment_agent.py
│   ├── test_decision_support.py
│   └── test_analytics.py
├── integration/             # Integration tests
│   ├── test_api_endpoints.py
│   ├── test_database.py
│   └── test_external_apis.py
├── performance/             # Performance tests
│   ├── test_load.py
│   └── test_stress.py
└── fixtures/                # Test data and fixtures
    ├── sample_data.json
    └── mock_responses.py
```

### Unit Testing

```python
import pytest
from unittest.mock import Mock, patch
from src.agents.sentiment_agent import SentimentAnalyzer


class TestSentimentAnalyzer:
    """Test cases for SentimentAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing."""
        return SentimentAnalyzer(model_name="test-model")
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return "This is a great product!"
    
    async def test_analyze_text_positive(self, analyzer, sample_text):
        """Test positive sentiment analysis."""
        with patch.object(analyzer, '_call_model') as mock_model:
            mock_model.return_value = {
                'sentiment': 'positive',
                'confidence': 0.85,
                'entities': ['product']
            }
            
            result = await analyzer.analyze_text(sample_text)
            
            assert result.sentiment == 'positive'
            assert result.confidence == 0.85
            assert 'product' in result.entities
    
    async def test_analyze_text_empty_input(self, analyzer):
        """Test handling of empty input."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await analyzer.analyze_text("")
    
    async def test_analyze_text_model_error(self, analyzer, sample_text):
        """Test handling of model errors."""
        with patch.object(analyzer, '_call_model') as mock_model:
            mock_model.side_effect = Exception("Model error")
            
            with pytest.raises(ModelError):
                await analyzer.analyze_text(sample_text)
```

### Integration Testing

```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


class TestAPIEndpoints:
    """Integration tests for API endpoints."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis endpoint."""
        data = {
            "content": "I love this product!",
            "language": "en"
        }
        
        response = client.post("/sentiment/analyze", json=data)
        assert response.status_code == 200
        
        result = response.json()
        assert "sentiment" in result
        assert "confidence" in result
        assert "entities" in result
    
    def test_sentiment_analysis_invalid_input(self):
        """Test sentiment analysis with invalid input."""
        data = {
            "content": "",  # Empty content
            "language": "en"
        }
        
        response = client.post("/sentiment/analyze", json=data)
        assert response.status_code == 400
        assert "Text cannot be empty" in response.json()["detail"]
```

### Performance Testing

```python
import asyncio
import time
import pytest
from src.agents.sentiment_agent import SentimentAnalyzer


class TestPerformance:
    """Performance tests."""
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self):
        """Test concurrent sentiment analysis."""
        analyzer = SentimentAnalyzer()
        texts = [f"Sample text {i}" for i in range(100)]
        
        start_time = time.time()
        
        # Run concurrent analysis
        tasks = [analyzer.analyze_text(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Assertions
        assert len(results) == 100
        assert processing_time < 30  # Should complete within 30 seconds
        assert all(hasattr(result, 'sentiment') for result in results)
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage during processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        analyzer = SentimentAnalyzer()
        
        # Process large batch
        large_texts = [f"Large text content {i}" * 100 for i in range(1000)]
        
        for text in large_texts:
            await analyzer.analyze_text(text)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
```

## API Development

### API Design Principles

#### 1. RESTful Design

```python
# Resource-based URL structure
GET    /sentiment/analyses          # List analyses
POST   /sentiment/analyses          # Create analysis
GET    /sentiment/analyses/{id}     # Get specific analysis
PUT    /sentiment/analyses/{id}     # Update analysis
DELETE /sentiment/analyses/{id}     # Delete analysis
```

#### 2. Response Format

```python
# Standard response format
{
    "success": true,
    "data": {
        // Response data
    },
    "metadata": {
        "timestamp": "2025-08-14T10:30:00Z",
        "processing_time": 0.5,
        "model_used": "llama3.2:latest"
    },
    "errors": null
}

# Error response format
{
    "success": false,
    "data": null,
    "metadata": {
        "timestamp": "2025-08-14T10:30:00Z"
    },
    "errors": [
        {
            "code": "VALIDATION_ERROR",
            "message": "Invalid input",
            "details": {
                "field": "content",
                "issue": "Text cannot be empty"
            }
        }
    ]
}
```

### API Versioning

```python
# Versioned API structure
from fastapi import APIRouter

# v1 API
v1_router = APIRouter(prefix="/api/v1")

@v1_router.post("/sentiment/analyze")
async def analyze_sentiment_v1():
    """Version 1 sentiment analysis endpoint."""
    pass

# v2 API
v2_router = APIRouter(prefix="/api/v2")

@v2_router.post("/sentiment/analyze")
async def analyze_sentiment_v2():
    """Version 2 sentiment analysis endpoint with improvements."""
    pass

# Include routers in main app
app.include_router(v1_router, tags=["v1"])
app.include_router(v2_router, tags=["v2"])
```

## Database Schema

### Core Tables

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Analyses table
CREATE TABLE analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    content TEXT NOT NULL,
    language VARCHAR(10) NOT NULL,
    sentiment VARCHAR(20) NOT NULL,
    confidence DECIMAL(3,2) NOT NULL,
    entities JSONB,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge graph entities
CREATE TABLE knowledge_graph_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    properties JSONB,
    confidence DECIMAL(3,2),
    source VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge graph relationships
CREATE TABLE knowledge_graph_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_entity_id UUID REFERENCES knowledge_graph_entities(id),
    target_entity_id UUID REFERENCES knowledge_graph_entities(id),
    relationship_type VARCHAR(50) NOT NULL,
    properties JSONB,
    confidence DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Indexes

```sql
-- Performance indexes
CREATE INDEX idx_analyses_user_id ON analyses(user_id);
CREATE INDEX idx_analyses_created_at ON analyses(created_at);
CREATE INDEX idx_analyses_sentiment ON analyses(sentiment);
CREATE INDEX idx_entities_type ON knowledge_graph_entities(entity_type);
CREATE INDEX idx_entities_name ON knowledge_graph_entities(name);
CREATE INDEX idx_relationships_type ON knowledge_graph_relationships(relationship_type);
```

## Deployment Development

### Docker Development

```dockerfile
# Development Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements.dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -r requirements.dev.txt

# Copy source code
COPY . .

# Expose port
EXPOSE 8003

# Run development server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8003", "--reload"]
```

### Kubernetes Development

```yaml
# Development deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-analysis-dev
spec:
  replicas: 1
  selector:
    matchLabels:
      app: sentiment-analysis-dev
  template:
    metadata:
      labels:
        app: sentiment-analysis-dev
    spec:
      containers:
      - name: app
        image: sentiment-analysis:dev
        ports:
        - containerPort: 8003
        env:
        - name: DEBUG
          value: "true"
        - name: LOG_LEVEL
          value: "DEBUG"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
```

---

**Developer Documentation Version:** 1.0.0  
**Last Updated:** 2025-08-14  
**For Questions:** Check contribution guidelines or contact development team
