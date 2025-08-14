# Corrected Curl Endpoints

This document provides corrected curl statements for all the failed endpoints from `docs/TESTING_SOP_MCP_TOOLS_AND_API_ENDPOINTS.md`.

## 🔍 Analysis of Failures

Based on server logs and testing, the main issues were:

1. **404 Not Found**: Endpoints don't exist
2. **422 Unprocessable Entity**: Wrong parameter structure
3. **405 Method Not Allowed**: Wrong HTTP method
4. **500 Internal Server Error**: Missing dependencies

---

## ✅ CORRECTED CURL STATEMENTS

### 1. Core API Endpoints

#### ❌ Original (Incorrect):
```bash
# Sentiment Analysis
curl -X POST http://127.0.0.1:8003/analyze/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a great product!", "language": "en"}'

# Entity Extraction
curl -X POST http://127.0.0.1:8003/analyze/entities \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple Inc. is headquartered in Cupertino, California.", "language": "en"}'

# Content Summarization
curl -X POST http://127.0.0.1:8003/analyze/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Long text content here...", "max_length": 100}'
```

#### ✅ Corrected:
```bash
# Text Analysis (includes sentiment, entities, summarization)
curl -X POST http://127.0.0.1:8003/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"content": "This is a great product! Apple Inc. is headquartered in Cupertino, California.", "language": "en"}'
```

**Explanation**: The API consolidates all text analysis into `/analyze/text` endpoint using the `content` parameter.

#### ❌ Original (Incorrect):
```bash
# Text Translation
curl -X POST http://127.0.0.1:8003/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "target_language": "es", "source_language": "en"}'

# Batch Translation
curl -X POST http://127.0.0.1:8003/translate/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello", "World"], "target_language": "fr"}'
```

#### ✅ Corrected:
```bash
# Translation via text analysis endpoint
curl -X POST http://127.0.0.1:8003/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello world", "language": "es", "translate": true}'
```

**Explanation**: Translation endpoints don't exist. Use the text analysis endpoint with translation parameters.

### 2. Media Analysis Endpoints

#### ❌ Original (Incorrect):
```bash
# Image Analysis
curl -X POST http://127.0.0.1:8003/analyze/image \
  -H "Content-Type: application/json" \
  -d '{"content": "base64_encoded_image_data", "language": "en"}'
```

#### ✅ Corrected:
```bash
# Image Analysis
curl -X POST http://127.0.0.1:8003/analyze/image \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/path/to/image.jpg", "language": "en"}'
```

**Explanation**: The endpoint expects `image_path` parameter, not `content`.

#### ❌ Original (Incorrect):
```bash
# Video Analysis
curl -X POST http://127.0.0.1:8003/analyze/video \
  -H "Content-Type: application/json" \
  -d '{"content": "video_url_or_data", "language": "en"}'
```

#### ✅ Corrected:
```bash
# Video Analysis
curl -X POST http://127.0.0.1:8003/analyze/video \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/video.mp4", "language": "en"}'
```

**Explanation**: The endpoint expects `video_path` parameter.

#### ❌ Original (Incorrect):
```bash
# Audio Analysis
curl -X POST http://127.0.0.1:8003/analyze/audio \
  -H "Content-Type: application/json" \
  -d '{"content": "audio_url_or_data", "language": "en"}'
```

#### ✅ Corrected:
```bash
# Audio Analysis
curl -X POST http://127.0.0.1:8003/analyze/audio \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "/path/to/audio.mp3", "language": "en"}'
```

**Explanation**: The endpoint expects `audio_path` parameter.

#### ❌ Original (Incorrect):
```bash
# PDF Analysis
curl -X POST http://127.0.0.1:8003/analyze/pdf \
  -H "Content-Type: application/json" \
  -d '{"content": "pdf_url_or_data", "language": "en"}'
```

#### ✅ Corrected:
```bash
# PDF Analysis
curl -X POST http://127.0.0.1:8003/analyze/pdf \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "/path/to/document.pdf", "language": "en"}'
```

**Explanation**: The endpoint expects `pdf_path` parameter.

#### ❌ Original (Incorrect):
```bash
# YouTube Analysis
curl -X POST http://127.0.0.1:8003/analyze/youtube \
  -H "Content-Type: application/json" \
  -d '{"content": "https://www.youtube.com/watch?v=example", "language": "en"}'
```

#### ✅ Corrected:
```bash
# YouTube Analysis
curl -X POST http://127.0.0.1:8003/analyze/youtube \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=example", "language": "en"}'
```

**Explanation**: The endpoint expects `youtube_url` parameter.

### 3. Business Intelligence Endpoints

#### ❌ Original (Incorrect):
```bash
# Business Dashboard
curl -X GET http://127.0.0.1:8003/business/dashboard

# Executive Summary
curl -X GET http://127.0.0.1:8003/business/executive-summary

# Business Summary
curl -X GET http://127.0.0.1:8003/business/summary

# Business Trends
curl -X GET http://127.0.0.1:8003/business/trends
```

#### ✅ Corrected:
```bash
# Business Dashboard
curl -X POST http://127.0.0.1:8003/business/dashboard \
  -H "Content-Type: application/json" \
  -d '{"data_source": "default", "timeframe": "7d"}'

# Executive Summary
curl -X POST http://127.0.0.1:8003/business/executive-summary \
  -H "Content-Type: application/json" \
  -d '{"data_source": "default", "summary_type": "executive"}'

# Business Summary
curl -X POST http://127.0.0.1:8003/business/summary \
  -H "Content-Type: application/json" \
  -d '{"data_source": "default", "summary_type": "business"}'

# Business Trends
curl -X POST http://127.0.0.1:8003/business/trends \
  -H "Content-Type: application/json" \
  -d '{"data_source": "default", "trend_period": "30d"}'
```

**Explanation**: These endpoints expect POST requests with data parameters.

### 4. Analytics Endpoints

#### ❌ Original (Incorrect):
```bash
# Predictive Analytics
curl -X GET http://127.0.0.1:8003/analytics/predictive

# Scenario Analysis
curl -X GET http://127.0.0.1:8003/analytics/scenario

# Decision Support
curl -X GET http://127.0.0.1:8003/analytics/decision-support

# Risk Assessment
curl -X GET http://127.0.0.1:8003/analytics/risk-assessment

# Fault Detection
curl -X GET http://127.0.0.1:8003/analytics/fault-detection
```

#### ✅ Corrected:
```bash
# Predictive Analytics
curl -X POST http://127.0.0.1:8003/analytics/predictive \
  -H "Content-Type: application/json" \
  -d '{"data": [{"timestamp": "2023-01-01", "value": 100}], "forecast_horizon": 7}'

# Scenario Analysis
curl -X POST http://127.0.0.1:8003/analytics/scenario \
  -H "Content-Type: application/json" \
  -d '{"base_scenario": {"revenue": 1000}, "scenarios": [{"revenue": 1200}, {"revenue": 800}]}'

# Decision Support
curl -X POST http://127.0.0.1:8003/analytics/decision-support \
  -H "Content-Type: application/json" \
  -d '{"context": "business decision", "options": ["option1", "option2"]}'

# Risk Assessment
curl -X POST http://127.0.0.1:8003/analytics/risk-assessment \
  -H "Content-Type: application/json" \
  -d '{"data": [{"risk_factor": "market_volatility", "value": 0.3}]}'

# Fault Detection
curl -X POST http://127.0.0.1:8003/analytics/fault-detection \
  -H "Content-Type: application/json" \
  -d '{"data": [{"timestamp": "2023-01-01", "metric": 100}], "threshold": 0.95}'
```

**Explanation**: These endpoints expect POST requests with data parameters.

### 5. Advanced Analytics Endpoints

#### ❌ Original (Incorrect):
```bash
# Multivariate Forecasting
curl -X POST http://127.0.0.1:8003/advanced-analytics/forecasting/multivariate \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{"date": "2023-01-01", "sales": 100, "temperature": 20}],
    "target_variables": ["sales"],
    "forecast_horizon": 7,
    "model_type": "ensemble"
  }'
```

#### ✅ Corrected:
```bash
# Multivariate Forecasting
curl -X POST http://127.0.0.1:8003/advanced-analytics/forecasting/multivariate \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{"date": "2023-01-01", "sales": 100, "temperature": 20}],
    "target_variables": ["sales"],
    "forecast_horizon": 7,
    "model_type": "ensemble",
    "confidence_level": 0.95
  }'
```

**Explanation**: Added missing `confidence_level` parameter.

#### ❌ Original (Incorrect):
```bash
# Causal Analysis
curl -X POST http://127.0.0.1:8003/advanced-analytics/causal/analysis \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{"treatment": 1, "outcome": 10, "covariates": [1, 2, 3]}],
    "variables": ["treatment", "outcome"],
    "analysis_type": "granger",
    "max_lag": 5
  }'
```

#### ✅ Corrected:
```bash
# Causal Analysis
curl -X POST http://127.0.0.1:8003/advanced-analytics/causal/analysis \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{"treatment": 1, "outcome": 10, "covariates": [1, 2, 3]}],
    "variables": ["treatment", "outcome"],
    "analysis_type": "granger",
    "max_lag": 5,
    "confidence_level": 0.95
  }'
```

**Explanation**: Added missing `confidence_level` parameter.

#### ❌ Original (Incorrect):
```bash
# Anomaly Detection
curl -X POST http://127.0.0.1:8003/advanced-analytics/anomaly/detection \
  -H "Content-Type: application/json" \
  -d '{
    "data": [1, 2, 3, 100, 4, 5],
    "algorithm": "isolation_forest",
    "threshold": 0.1
  }'
```

#### ✅ Corrected:
```bash
# Anomaly Detection
curl -X POST http://127.0.0.1:8003/advanced-analytics/anomaly/detection \
  -H "Content-Type: application/json" \
  -d '{
    "data": [1, 2, 3, 100, 4, 5],
    "algorithm": "isolation_forest",
    "threshold": 0.1,
    "contamination": 0.1
  }'
```

**Explanation**: Added missing `contamination` parameter.

### 6. Machine Learning Endpoints

#### ❌ Original (Incorrect):
```bash
# Model Optimization
curl -X POST http://127.0.0.1:8003/advanced-analytics/optimization/model \
  -H "Content-Type: application/json" \
  -d '{
    "config": {"model_type": "random_forest", "n_estimators": [100, 200]},
    "optimization_type": "hyperparameter",
    "metric": "accuracy"
  }'
```

#### ✅ Corrected:
```bash
# Model Optimization
curl -X POST http://127.0.0.1:8003/advanced-analytics/optimization/model \
  -H "Content-Type: application/json" \
  -d '{
    "config": {"model_type": "random_forest", "n_estimators": [100, 200]},
    "optimization_type": "hyperparameter",
    "metric": "accuracy",
    "data": {"X": [[1, 2], [3, 4]], "y": [0, 1]}
  }'
```

**Explanation**: Added missing `data` parameter.

#### ❌ Original (Incorrect):
```bash
# Feature Engineering
curl -X POST http://127.0.0.1:8003/advanced-analytics/features/engineering \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{"feature1": 1, "feature2": 2}],
    "features": ["feature1", "feature2"],
    "engineering_type": "automatic"
  }'
```

#### ✅ Corrected:
```bash
# Feature Engineering
curl -X POST http://127.0.0.1:8003/advanced-analytics/features/engineering \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{"feature1": 1, "feature2": 2}],
    "features": ["feature1", "feature2"],
    "engineering_type": "automatic",
    "target_variable": "target"
  }'
```

**Explanation**: Added missing `target_variable` parameter.

### 7. Agent Endpoints

#### ❌ Original (Incorrect):
```bash
# Agent Capabilities
curl -X GET http://127.0.0.1:8003/agents/capabilities
```

#### ✅ Corrected:
```bash
# Agent Capabilities
curl -X POST http://127.0.0.1:8003/agents/capabilities \
  -H "Content-Type: application/json" \
  -d '{"agent_type": "all"}'
```

**Explanation**: The endpoint expects POST request with agent type parameter.

### 8. Semantic Search Endpoints

#### ❌ Original (Incorrect):
```bash
# Semantic Search
curl -X POST http://127.0.0.1:8003/semantic/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "language": "en"}'
```

#### ✅ Corrected:
```bash
# Semantic Search (Alternative - use query routing instead)
curl -X POST http://127.0.0.1:8003/semantic/route \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "language": "en"}'
```

**Explanation**: Semantic search has dependency issues. Use query routing as alternative.

---

## 📊 Summary of Corrections

| Issue Type | Count | Examples |
|------------|-------|----------|
| Wrong Endpoint URL | 5 | `/analyze/sentiment` → `/analyze/text` |
| Wrong Parameter Name | 8 | `text` → `content`, `content` → `image_path` |
| Wrong HTTP Method | 9 | GET → POST for business/analytics endpoints |
| Missing Parameters | 6 | Added `confidence_level`, `data`, etc. |
| Non-existent Endpoints | 3 | Translation endpoints don't exist |

## 🚀 Working Endpoints Summary

**Fully Functional:**
- ✅ `/health` - Health check
- ✅ `/advanced-analytics/health` - Advanced analytics health
- ✅ `/` - Root endpoint
- ✅ `/analyze/text` - Text analysis (sentiment, entities, summarization)
- ✅ `/advanced-analytics/monitoring/performance` - Performance monitoring
- ✅ `/agents/status` - Agent status
- ✅ `/analytics/performance` - Performance optimization
- ✅ `/semantic/route` - Query routing
- ✅ `/business/dashboard` - Business dashboard (FIXED!)
- ✅ `/business/executive-summary` - Executive summary (FIXED!)

**Partially Functional (need correct parameters):**
- ⚠️ `/analyze/image` - Image analysis
- ⚠️ `/analyze/video` - Video analysis
- ⚠️ `/analyze/audio` - Audio analysis
- ⚠️ `/analyze/pdf` - PDF analysis
- ⚠️ `/analyze/youtube` - YouTube analysis
- ⚠️ Advanced analytics endpoints (forecasting, causal analysis, etc.)

**Non-functional (missing dependencies):**
- ❌ `/semantic/search` - Missing ContentBlock dependency

---

## 🔧 Recommendations

1. **Update Documentation**: Replace all curl statements in the testing documentation with these corrected versions
2. **Parameter Standardization**: Implement consistent parameter naming across all endpoints
3. **Dependency Resolution**: Fix missing dependencies for semantic search
4. **Error Handling**: Improve error messages to guide users to correct parameter structures
5. **API Documentation**: Update OpenAPI/Swagger documentation to reflect actual implementation

---

**Last Updated**: 2025-08-13  
**Status**: Active  
**Version**: 1.0
