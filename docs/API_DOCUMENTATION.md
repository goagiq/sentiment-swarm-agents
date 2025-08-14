# API Documentation - Sentiment Analysis & Decision Support System

**Version:** 1.0.0  
**Base URL:** `http://localhost:8003`  
**Last Updated:** 2025-08-14

## Table of Contents

1. [Authentication](#authentication)
2. [Core Endpoints](#core-endpoints)
3. [Advanced Analytics Endpoints](#advanced-analytics-endpoints)
4. [Business Intelligence Endpoints](#business-intelligence-endpoints)
5. [Search and Semantic Endpoints](#search-and-semantic-endpoints)
6. [Analytics Endpoints](#analytics-endpoints)
7. [Error Codes](#error-codes)
8. [Rate Limiting](#rate-limiting)
9. [Examples](#examples)

## Authentication

The API supports multiple authentication methods:

### API Key Authentication
```http
Authorization: ApiKey your-api-key-here
```

### Bearer Token Authentication
```http
Authorization: Bearer your-bearer-token-here
```

### Basic Authentication
```http
Authorization: Basic base64(username:password)
```

## Core Endpoints

### Health Check

#### GET `/health`
Returns system health status and available models.

**Response:**
```json
{
  "status": "healthy",
  "ollama_models": ["llama3.2:latest", "mistral-small3.1:latest"],
  "endpoints": [
    "/analyze/text",
    "/business/summary",
    "/business/executive-summary"
  ]
}
```

### Text Analysis

#### POST `/analyze/text`
Analyzes text content using Ollama models.

**Request Body:**
```json
{
  "content": "The new product launch was extremely successful and customers are very satisfied.",
  "language": "en",
  "model_preference": "llama3.2:latest"
}
```

**Response:**
```json
{
  "content": "The new product launch was extremely successful and customers are very satisfied.",
  "language": "en",
  "model_used": "llama3.2:latest",
  "sentiment": "positive",
  "confidence": 0.85,
  "entities": ["product launch", "customers"],
  "analysis": {
    "sentiment_score": 0.85,
    "key_phrases": ["extremely successful", "very satisfied"],
    "emotions": ["satisfaction", "excitement"]
  }
}
```

### Get Available Models

#### GET `/models`
Returns list of available Ollama models.

**Response:**
```json
{
  "models": [
    {
      "name": "llama3.2:latest",
      "type": "text",
      "max_tokens": 4096,
      "temperature": 0.7
    },
    {
      "name": "mistral-small3.1:latest",
      "type": "text",
      "max_tokens": 4096,
      "temperature": 0.7
    }
  ]
}
```

## Advanced Analytics Endpoints

### Advanced Analytics Health

#### GET `/advanced-analytics/health`
Returns advanced analytics system health status.

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "forecasting_engine": "operational",
    "causal_analysis": "operational",
    "anomaly_detection": "operational",
    "pattern_recognition": "operational"
  },
  "models_loaded": 4,
  "last_update": "2025-08-14T10:30:00Z"
}
```

### Multivariate Forecasting

#### POST `/advanced-analytics/forecasting-test`
Performs multivariate forecasting analysis.

**Request Body:**
```json
{
  "data_source": "sales_data",
  "forecast_horizon": 30,
  "variables": ["revenue", "customers", "conversion_rate"],
  "confidence_level": 0.95
}
```

**Response:**
```json
{
  "forecast_id": "fc_12345",
  "forecast_horizon": 30,
  "predictions": {
    "revenue": {
      "forecast": [100000, 105000, 110000],
      "confidence_intervals": [[95000, 105000], [100000, 110000], [105000, 115000]]
    },
    "customers": {
      "forecast": [1000, 1050, 1100],
      "confidence_intervals": [[950, 1050], [1000, 1100], [1050, 1150]]
    }
  },
  "model_performance": {
    "mae": 0.05,
    "rmse": 0.08,
    "r2_score": 0.92
  }
}
```

## Business Intelligence Endpoints

### Business Summary

#### POST `/business/summary`
Generates comprehensive business summary.

**Request Body:**
```json
{
  "data_source": "quarterly_reports",
  "time_period": "Q3 2025",
  "include_metrics": true,
  "include_trends": true,
  "include_recommendations": true
}
```

**Response:**
```json
{
  "summary_id": "sum_12345",
  "time_period": "Q3 2025",
  "executive_summary": "Q3 2025 showed strong growth with 15% revenue increase...",
  "key_metrics": {
    "revenue": "$2.5M",
    "growth_rate": "15%",
    "customer_satisfaction": "4.2/5",
    "market_share": "12%"
  },
  "trends": [
    "Increasing customer acquisition",
    "Improving product satisfaction",
    "Growing market presence"
  ],
  "recommendations": [
    "Invest in customer retention programs",
    "Expand product portfolio",
    "Enhance digital marketing efforts"
  ]
}
```

### Executive Summary

#### POST `/business/executive-summary`
Creates executive-level business summary.

**Request Body:**
```json
{
  "content_data": "Q3 financial results and market analysis...",
  "summary_type": "business",
  "include_metrics": true,
  "include_trends": true
}
```

**Response:**
```json
{
  "executive_summary": "Q3 2025 demonstrates continued strong performance...",
  "key_highlights": [
    "Revenue growth exceeded expectations",
    "Market expansion successful",
    "Customer satisfaction improved"
  ],
  "strategic_insights": [
    "Market opportunity in emerging regions",
    "Product innovation driving growth",
    "Operational efficiency gains"
  ],
  "next_quarter_focus": [
    "Scale successful initiatives",
    "Enter new markets",
    "Enhance digital capabilities"
  ]
}
```

## Search and Semantic Endpoints

### Semantic Search

#### POST `/semantic/search`
Performs semantic search across knowledge base.

**Request Body:**
```json
{
  "query": "customer satisfaction improvement strategies",
  "max_results": 10,
  "similarity_threshold": 0.7,
  "include_metadata": true
}
```

**Response:**
```json
{
  "query": "customer satisfaction improvement strategies",
  "results": [
    {
      "id": "doc_123",
      "content": "Implementing customer feedback loops...",
      "similarity_score": 0.92,
      "metadata": {
        "source": "customer_service_manual",
        "date": "2025-01-15",
        "category": "customer_experience"
      }
    }
  ],
  "total_results": 15,
  "search_time": 0.045
}
```

### Knowledge Graph Search

#### POST `/search/knowledge-graph`
Searches the knowledge graph for entities and relationships.

**Request Body:**
```json
{
  "query": "product development process",
  "entity_types": ["PROCESS", "PERSON", "ORGANIZATION"],
  "max_depth": 3,
  "include_relationships": true
}
```

**Response:**
```json
{
  "query": "product development process",
  "entities": [
    {
      "id": "ent_123",
      "name": "Product Development",
      "type": "PROCESS",
      "confidence": 0.95,
      "properties": {
        "description": "End-to-end product development lifecycle",
        "duration": "6-12 months"
      }
    }
  ],
  "relationships": [
    {
      "source": "ent_123",
      "target": "ent_456",
      "type": "INCLUDES",
      "confidence": 0.88
    }
  ],
  "graph_data": {
    "nodes": [...],
    "edges": [...]
  }
}
```

## Analytics Endpoints

### Predictive Analytics

#### POST `/analytics/predictive`
Performs predictive analytics on business data.

**Request Body:**
```json
{
  "data_source": "customer_behavior",
  "prediction_type": "churn_prediction",
  "time_horizon": 90,
  "features": ["usage_frequency", "support_tickets", "payment_history"]
}
```

**Response:**
```json
{
  "prediction_id": "pred_12345",
  "prediction_type": "churn_prediction",
  "results": {
    "high_risk_customers": 150,
    "medium_risk_customers": 300,
    "low_risk_customers": 1200,
    "overall_churn_probability": 0.12
  },
  "model_metrics": {
    "accuracy": 0.89,
    "precision": 0.85,
    "recall": 0.82,
    "f1_score": 0.83
  },
  "recommendations": [
    "Implement retention campaigns for high-risk customers",
    "Enhance customer support for medium-risk segment",
    "Develop loyalty programs for low-risk customers"
  ]
}
```

### Scenario Analysis

#### POST `/analytics/scenario`
Performs scenario analysis for business planning.

**Request Body:**
```json
{
  "scenario_name": "Market Expansion",
  "variables": {
    "investment_amount": 1000000,
    "market_size": 50000000,
    "competition_level": "medium"
  },
  "time_period": 24,
  "confidence_level": 0.95
}
```

**Response:**
```json
{
  "scenario_id": "scen_12345",
  "scenario_name": "Market Expansion",
  "results": {
    "roi": 0.25,
    "payback_period": 18,
    "risk_level": "medium",
    "success_probability": 0.75
  },
  "sensitivity_analysis": {
    "investment_amount": {
      "impact": "high",
      "optimal_range": [800000, 1200000]
    },
    "market_size": {
      "impact": "medium",
      "optimal_range": [40000000, 60000000]
    }
  },
  "recommendations": [
    "Proceed with expansion plan",
    "Monitor market conditions closely",
    "Prepare contingency plans"
  ]
}
```

### Performance Optimization

#### POST `/analytics/performance`
Analyzes system performance and provides optimization recommendations.

**Request Body:**
```json
{
  "performance_metrics": {
    "response_time": 2.5,
    "throughput": 100,
    "error_rate": 0.02,
    "resource_utilization": 0.75
  },
  "optimization_target": "response_time"
}
```

**Response:**
```json
{
  "analysis_id": "perf_12345",
  "current_performance": {
    "response_time": 2.5,
    "throughput": 100,
    "error_rate": 0.02,
    "resource_utilization": 0.75
  },
  "bottlenecks": [
    "Database query optimization needed",
    "Caching strategy can be improved",
    "Load balancing configuration"
  ],
  "optimization_recommendations": [
    "Implement database indexing",
    "Add Redis caching layer",
    "Optimize API endpoints",
    "Scale horizontally"
  ],
  "expected_improvements": {
    "response_time": "40% reduction",
    "throughput": "50% increase",
    "error_rate": "60% reduction"
  }
}
```

## Error Codes

### HTTP Status Codes

| Code | Description | Example |
|------|-------------|---------|
| 200 | Success | Request completed successfully |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 422 | Validation Error | Request validation failed |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error occurred |
| 503 | Service Unavailable | Service temporarily unavailable |

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "content",
      "issue": "Content cannot be empty"
    },
    "timestamp": "2025-08-14T10:30:00Z",
    "request_id": "req_12345"
  }
}
```

### Common Error Codes

| Error Code | Description | HTTP Status |
|------------|-------------|-------------|
| `INVALID_INPUT` | Invalid input parameters | 400 |
| `AUTHENTICATION_FAILED` | Authentication failed | 401 |
| `INSUFFICIENT_PERMISSIONS` | Insufficient permissions | 403 |
| `RESOURCE_NOT_FOUND` | Requested resource not found | 404 |
| `VALIDATION_ERROR` | Request validation failed | 422 |
| `RATE_LIMIT_EXCEEDED` | Rate limit exceeded | 429 |
| `INTERNAL_ERROR` | Internal server error | 500 |
| `SERVICE_UNAVAILABLE` | Service temporarily unavailable | 503 |

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Standard Plan**: 100 requests per minute
- **Premium Plan**: 1000 requests per minute
- **Enterprise Plan**: 10000 requests per minute

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## Examples

### Python Example

```python
import requests
import json

# Base URL
base_url = "http://localhost:8003"

# Headers
headers = {
    "Content-Type": "application/json",
    "Authorization": "ApiKey your-api-key-here"
}

# Text Analysis
def analyze_text(content, language="en"):
    url = f"{base_url}/analyze/text"
    data = {
        "content": content,
        "language": language
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Business Summary
def get_business_summary(data_source, time_period):
    url = f"{base_url}/business/summary"
    data = {
        "data_source": data_source,
        "time_period": time_period,
        "include_metrics": True,
        "include_trends": True
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Usage
result = analyze_text("The product launch was successful!")
print(result)
```

### JavaScript Example

```javascript
const baseUrl = 'http://localhost:8003';
const apiKey = 'your-api-key-here';

// Text Analysis
async function analyzeText(content, language = 'en') {
    const response = await fetch(`${baseUrl}/analyze/text`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `ApiKey ${apiKey}`
        },
        body: JSON.stringify({
            content: content,
            language: language
        })
    });
    
    return await response.json();
}

// Business Summary
async function getBusinessSummary(dataSource, timePeriod) {
    const response = await fetch(`${baseUrl}/business/summary`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `ApiKey ${apiKey}`
        },
        body: JSON.stringify({
            data_source: dataSource,
            time_period: timePeriod,
            include_metrics: true,
            include_trends: true
        })
    });
    
    return await response.json();
}

// Usage
analyzeText("The product launch was successful!")
    .then(result => console.log(result))
    .catch(error => console.error(error));
```

### cURL Examples

```bash
# Health Check
curl -X GET "http://localhost:8003/health"

# Text Analysis
curl -X POST "http://localhost:8003/analyze/text" \
  -H "Content-Type: application/json" \
  -H "Authorization: ApiKey your-api-key-here" \
  -d '{
    "content": "The product launch was successful!",
    "language": "en"
  }'

# Business Summary
curl -X POST "http://localhost:8003/business/summary" \
  -H "Content-Type: application/json" \
  -H "Authorization: ApiKey your-api-key-here" \
  -d '{
    "data_source": "quarterly_reports",
    "time_period": "Q3 2025",
    "include_metrics": true,
    "include_trends": true
  }'
```

## Support

For API support and questions:

- **Documentation**: Check this documentation first
- **Error Handling**: Review error codes and responses
- **Rate Limiting**: Monitor rate limit headers
- **Authentication**: Ensure proper API key or token usage

---

**API Version:** 1.0.0  
**Last Updated:** 2025-08-14  
**Status:** Production Ready
