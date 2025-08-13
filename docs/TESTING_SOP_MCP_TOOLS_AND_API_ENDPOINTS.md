# Testing SOP: MCP Tools and API Endpoints

## Overview
This document provides a comprehensive step-by-step Standard Operating Procedure (SOP) for testing each MCP (Model Context Protocol) tool and API endpoint in the Advanced Analytics System.

## Prerequisites
- Python virtual environment activated (`.venv`)
- All dependencies installed (`pip install -r requirements.txt`)
- Main server running (`python main.py`)
- Test data available in `data/` directory

## Table of Contents
1. [System Setup and Health Checks](#system-setup-and-health-checks)
2. [API Endpoint Testing](#api-endpoint-testing)
3. [MCP Tools Testing](#mcp-tools-testing)
4. [Advanced Analytics Features Testing](#advanced-analytics-features-testing)
5. [Integration Testing](#integration-testing)
6. [Performance Testing](#performance-testing)
7. [Troubleshooting Guide](#troubleshooting-guide)

---

## 1. System Setup and Health Checks

### 1.1 Start the Main Server
```bash
# Navigate to project root
cd /d/AI/Sentiment

# Activate virtual environment
source .venv/Scripts/activate  # On Windows
# OR
source .venv/bin/activate      # On Linux/Mac

# Start the main server
.venv/Scripts/python.exe main.py
```

**Expected Output:**
```
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 1.2 Verify Basic Health Endpoints
```bash
# Test main API health
curl http://127.0.0.1:8000/health

# Test advanced analytics health
curl http://127.0.0.1:8000/advanced-analytics/health

# Test root endpoint
curl http://127.0.0.1:8000/
```

**Expected Responses:**
- Main health: `{"status": "healthy", "timestamp": "..."}`
- Advanced analytics health: `{"status": "healthy", "components": {...}}`
- Root: `{"message": "Advanced Analytics System", "endpoints": {...}}`

---

## 2. API Endpoint Testing

### 2.1 Core API Endpoints

#### 2.1.1 Text Analysis Endpoints
```bash
# Test sentiment analysis
curl -X POST http://127.0.0.1:8000/analyze/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a great product!", "language": "en"}'

# Test entity extraction
curl -X POST http://127.0.0.1:8000/analyze/entities \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple Inc. is headquartered in Cupertino, California.", "language": "en"}'

# Test content summarization
curl -X POST http://127.0.0.1:8000/analyze/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Long text content here...", "max_length": 100}'
```

#### 2.1.2 Translation Endpoints
```bash
# Test text translation
curl -X POST http://127.0.0.1:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "target_language": "es", "source_language": "en"}'

# Test batch translation
curl -X POST http://127.0.0.1:8000/translate/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello", "World"], "target_language": "fr"}'
```

#### 2.1.3 Content Processing Endpoints
```bash
# Test content extraction
curl -X POST http://127.0.0.1:8000/extract/content \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "content_type": "web"}'

# Test format conversion
curl -X POST http://127.0.0.1:8000/convert/format \
  -H "Content-Type: application/json" \
  -d '{"content": "Raw content", "source_format": "text", "target_format": "markdown"}'
```

### 2.2 Advanced Analytics API Endpoints

#### 2.2.1 Forecasting Endpoints
```bash
# Test multivariate forecasting
curl -X POST http://127.0.0.1:8000/advanced-analytics/forecasting/multivariate \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{"date": "2023-01-01", "sales": 100, "temperature": 20}],
    "target_variables": ["sales"],
    "forecast_horizon": 7,
    "model_type": "ensemble"
  }'

# Test time series forecasting
curl -X POST http://127.0.0.1:8000/advanced-analytics/forecasting/timeseries \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{"timestamp": "2023-01-01", "value": 100}],
    "forecast_horizon": 10,
    "seasonality": "daily"
  }'
```

#### 2.2.2 Causal Analysis Endpoints
```bash
# Test causal inference
curl -X POST http://127.0.0.1:8000/advanced-analytics/causal/analysis \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{"treatment": 1, "outcome": 10, "covariates": [1, 2, 3]}],
    "treatment_variable": "treatment",
    "outcome_variable": "outcome",
    "method": "propensity_score"
  }'

# Test causal discovery
curl -X POST http://127.0.0.1:8000/advanced-analytics/causal/discovery \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{"var1": 1, "var2": 2, "var3": 3}],
    "variables": ["var1", "var2", "var3"],
    "method": "pc_algorithm"
  }'
```

#### 2.2.3 Anomaly Detection Endpoints
```bash
# Test anomaly detection
curl -X POST http://127.0.0.1:8000/advanced-analytics/anomaly/detection \
  -H "Content-Type: application/json" \
  -d '{
    "data": [1, 2, 3, 100, 4, 5],
    "method": "isolation_forest",
    "threshold": 0.95
  }'

# Test real-time anomaly monitoring
curl -X POST http://127.0.0.1:8000/advanced-analytics/anomaly/monitoring \
  -H "Content-Type: application/json" \
  -d '{
    "data_stream": [1, 2, 3, 4, 5],
    "window_size": 10,
    "alert_threshold": 0.9
  }'
```

#### 2.2.4 Machine Learning Endpoints
```bash
# Test model optimization
curl -X POST http://127.0.0.1:8000/advanced-analytics/optimization/model \
  -H "Content-Type: application/json" \
  -d '{
    "config": {"model_type": "random_forest", "n_estimators": [100, 200]},
    "optimization_type": "hyperparameter",
    "metric": "accuracy"
  }'

# Test feature engineering
curl -X POST http://127.0.0.1:8000/advanced-analytics/features/engineering \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{"feature1": 1, "feature2": 2}],
    "techniques": ["polynomial", "interaction"],
    "target_variable": "target"
  }'

# Test deep learning training
curl -X POST http://127.0.0.1:8000/advanced-analytics/ml/deep-learning \
  -H "Content-Type: application/json" \
  -d '{
    "data": {"X": [[1, 2], [3, 4]], "y": [0, 1]},
    "model_type": "mlp",
    "architecture": {"layers": [64, 32, 1]},
    "training_config": {"epochs": 10, "batch_size": 32}
  }'
```

#### 2.2.5 Clustering and Dimensionality Reduction
```bash
# Test clustering
curl -X POST http://127.0.0.1:8000/advanced-analytics/ml/clustering \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[1, 2], [3, 4], [5, 6]],
    "method": "kmeans",
    "n_clusters": 2
  }'

# Test dimensionality reduction
curl -X POST http://127.0.0.1:8000/advanced-analytics/ml/dimensionality \
  -H "Content-Type: application/json" \
  -d '{
    "data": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    "method": "pca",
    "n_components": 2
  }'
```

#### 2.2.6 Performance Monitoring
```bash
# Test performance monitoring
curl -X POST http://127.0.0.1:8000/advanced-analytics/monitoring/performance \
  -H "Content-Type: application/json" \
  -d '{
    "metrics": ["cpu_usage", "memory_usage", "response_time"],
    "timeframe": "1h"
  }'

# Test model performance
curl -X POST http://127.0.0.1:8000/advanced-analytics/ml/performance \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "model_001",
    "test_data": {"X": [[1, 2], [3, 4]], "y": [0, 1]},
    "metrics": ["accuracy", "precision", "recall"]
  }'
```

---

## 3. MCP Tools Testing

### 3.1 Start MCP Server
```bash
# In a new terminal, start the MCP server
.venv/Scripts/python.exe -m src.mcp_servers.unified_mcp_server
```

### 3.2 Test Core MCP Tools

#### 3.2.1 Content Processing Tools
```python
# Test content processing
result = await mcp.process_content(
    content="Sample text content",
    content_type="text",
    options={"language": "en"}
)

# Test text extraction
result = await mcp.extract_text_from_content(
    content="<html><body>Text content</body></html>",
    content_type="html"
)

# Test content summarization
result = await mcp.summarize_content(
    content="Long text content...",
    content_type="text",
    summary_length="medium"
)
```

#### 3.2.2 Translation Tools
```python
# Test content translation
result = await mcp.translate_content(
    content="Hello world",
    target_language="es",
    source_language="en"
)
```

#### 3.2.3 Format Conversion Tools
```python
# Test format conversion
result = await mcp.convert_content_format(
    content="Raw content",
    source_format="text",
    target_format="markdown"
)
```

#### 3.2.4 Analysis Tools
```python
# Test sentiment analysis
result = await mcp.analyze_sentiment(
    text="This is a great product!",
    language="en"
)

# Test entity extraction
result = await mcp.extract_entities(
    text="Apple Inc. is headquartered in Cupertino.",
    entity_types=["ORGANIZATION", "LOCATION"]
)
```

### 3.3 Test Advanced Analytics MCP Tools

#### 3.3.1 Forecasting Tools
```python
# Test advanced forecasting
result = await mcp.advanced_forecasting(
    data='[{"date": "2023-01-01", "sales": 100}]',
    target_variables='["sales"]',
    forecast_horizon=7,
    model_type="ensemble"
)
```

#### 3.3.2 Causal Analysis Tools
```python
# Test causal analysis
result = await mcp.causal_analysis(
    data='[{"treatment": 1, "outcome": 10}]',
    treatment_variable="treatment",
    outcome_variable="outcome",
    method="propensity_score"
)
```

#### 3.3.3 Anomaly Detection Tools
```python
# Test anomaly detection
result = await mcp.anomaly_detection(
    data='[1, 2, 3, 100, 4, 5]',
    method="isolation_forest",
    threshold=0.95
)
```

#### 3.3.4 Scenario Analysis Tools
```python
# Test scenario analysis
result = await mcp.scenario_analysis(
    data='[{"scenario": "base", "revenue": 1000}]',
    scenarios='["optimistic", "pessimistic"]',
    variables='["revenue", "costs"]'
)
```

#### 3.3.5 Machine Learning Tools
```python
# Test model optimization
result = await mcp.model_optimization(
    config='{"model_type": "random_forest"}',
    optimization_type="hyperparameter",
    metric="accuracy"
)

# Test feature engineering
result = await mcp.feature_engineering(
    data='[{"feature1": 1, "feature2": 2}]',
    techniques='["polynomial", "interaction"]',
    target_variable="target"
)

# Test deep learning training
result = await mcp.deep_learning_training(
    data='{"X": [[1, 2], [3, 4]], "y": [0, 1]}',
    model_type="mlp",
    architecture='{"layers": [64, 32, 1]}'
)

# Test AutoML pipeline
result = await mcp.automl_pipeline(
    data='{"X": [[1, 2], [3, 4]], "y": [0, 1]}',
    task_type="classification",
    time_limit=300
)
```

---

## 4. Advanced Analytics Features Testing

### 4.1 Agent Testing

#### 4.1.1 Test Advanced Forecasting Agent
```python
# Test forecasting agent
from src.agents.advanced_forecasting_agent import AdvancedForecastingAgent

agent = AdvancedForecastingAgent()
result = await agent.forecast(
    data=[{"date": "2023-01-01", "sales": 100}],
    target_variables=["sales"],
    forecast_horizon=7
)
```

#### 4.1.2 Test Causal Analysis Agent
```python
# Test causal analysis agent
from src.agents.causal_analysis_agent import CausalAnalysisAgent

agent = CausalAnalysisAgent()
result = await agent.analyze_causality(
    data=[{"treatment": 1, "outcome": 10}],
    treatment_variable="treatment",
    outcome_variable="outcome"
)
```

#### 4.1.3 Test Anomaly Detection Agent
```python
# Test anomaly detection agent
from src.agents.anomaly_detection_agent import AnomalyDetectionAgent

agent = AnomalyDetectionAgent()
result = await agent.detect_anomalies(
    data=[1, 2, 3, 100, 4, 5],
    method="isolation_forest"
)
```

#### 4.1.4 Test Advanced ML Agent
```python
# Test advanced ML agent
from src.agents.advanced_ml_agent import AdvancedMLAgent

agent = AdvancedMLAgent()
result = await agent.optimize_model(
    config={"model_type": "random_forest"},
    optimization_type="hyperparameter"
)
```

### 4.2 Engine Testing

#### 4.2.1 Test Forecasting Engine
```python
# Test multivariate forecasting engine
from src.core.advanced_analytics.multivariate_forecasting import MultivariateForecastingEngine

engine = MultivariateForecastingEngine()
result = engine.forecast(
    data=[{"date": "2023-01-01", "sales": 100}],
    target_variables=["sales"],
    forecast_horizon=7
)
```

#### 4.2.2 Test Causal Inference Engine
```python
# Test causal inference engine
from src.core.advanced_analytics.causal_inference_engine import CausalInferenceEngine

engine = CausalInferenceEngine()
result = engine.analyze_causality(
    data=[{"treatment": 1, "outcome": 10}],
    treatment_variable="treatment",
    outcome_variable="outcome"
)
```

---

## 5. Integration Testing

### 5.1 Run Comprehensive Test Suite
```bash
# Run all integration tests
.venv/Scripts/python.exe -m pytest Test/test_phase7_5_integration.py -v

# Run specific test categories
.venv/Scripts/python.exe -m pytest Test/test_phase7_5_integration.py::TestPhase75Integration::test_api_endpoints -v
.venv/Scripts/python.exe -m pytest Test/test_phase7_5_integration.py::TestPhase75Integration::test_agents -v
.venv/Scripts/python.exe -m pytest Test/test_phase7_5_integration.py::TestPhase75Integration::test_engines -v
```

### 5.2 Test End-to-End Workflows
```bash
# Test complete analytics pipeline
curl -X POST http://127.0.0.1:8000/advanced-analytics/workflow/complete \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{"date": "2023-01-01", "sales": 100, "temperature": 20}],
    "workflow": ["forecasting", "causal_analysis", "anomaly_detection"],
    "config": {"forecast_horizon": 7, "anomaly_threshold": 0.95}
  }'
```

---

## 6. Performance Testing

### 6.1 Load Testing
```bash
# Test API performance under load
.venv/Scripts/python.exe scripts/performance_monitoring_dashboard.py

# Test concurrent requests
for i in {1..10}; do
  curl -X POST http://127.0.0.1:8000/analyze/sentiment \
    -H "Content-Type: application/json" \
    -d '{"text": "Test message $i", "language": "en"}' &
done
wait
```

### 6.2 Memory and CPU Monitoring
```bash
# Monitor system resources
.venv/Scripts/python.exe scripts/health_check.sh

# Check specific metrics
curl http://127.0.0.1:8000/advanced-analytics/monitoring/performance \
  -H "Content-Type: application/json" \
  -d '{"metrics": ["cpu_usage", "memory_usage", "response_time"]}'
```

---

## 7. Troubleshooting Guide

### 7.1 Common Issues and Solutions

#### 7.1.1 Server Won't Start
**Symptoms:** `ImportError`, `ModuleNotFoundError`
**Solutions:**
```bash
# Check virtual environment
which python
# Should show: /path/to/.venv/Scripts/python.exe

# Reinstall dependencies
pip install -r requirements.txt

# Check for missing imports
.venv/Scripts/python.exe -c "import src.api.advanced_analytics_routes"
```

#### 7.1.2 404 Errors on Endpoints
**Symptoms:** `404 Not Found` responses
**Solutions:**
```bash
# Check if routes are registered
curl http://127.0.0.1:8000/

# Check server logs for route registration
# Look for: "✅ Advanced analytics routes included"

# Verify endpoint URLs match implementation
grep -r "router.add_api_route" src/api/advanced_analytics_routes.py
```

#### 7.1.3 MCP Tools Not Available
**Symptoms:** Tools not listed in MCP server
**Solutions:**
```bash
# Check MCP server startup
.venv/Scripts/python.exe -m src.mcp_servers.unified_mcp_server

# Look for: "Registered 30 unified MCP tools"

# Check agent initialization
grep -r "AdvancedForecastingAgent" src/mcp_servers/unified_mcp_server.py
```

#### 7.1.4 Memory Issues
**Symptoms:** `MemoryError`, slow performance
**Solutions:**
```bash
# Monitor memory usage
.venv/Scripts/python.exe scripts/performance_monitoring_dashboard.py

# Check for memory leaks
curl http://127.0.0.1:8000/advanced-analytics/monitoring/performance

# Restart server if needed
pkill -f "python main.py"
.venv/Scripts/python.exe main.py
```

### 7.2 Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
.venv/Scripts/python.exe main.py

# Check detailed logs
tail -f logs/app.log
```

### 7.3 Clean Restart Procedure
```bash
# Stop all Python processes
taskkill //f //im python.exe  # Windows
# OR
pkill -f python               # Linux/Mac

# Clear cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Restart server
.venv/Scripts/python.exe main.py
```

---

## 8. Validation Checklist

### 8.1 Pre-Testing Checklist
- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] Main server running on port 8000
- [ ] MCP server accessible
- [ ] Test data available
- [ ] Log files writable

### 8.2 API Testing Checklist
- [ ] All health endpoints responding
- [ ] Core API endpoints functional
- [ ] Advanced analytics endpoints accessible
- [ ] Error handling working correctly
- [ ] Response formats consistent
- [ ] Authentication (if configured) working

### 8.3 MCP Tools Checklist
- [ ] All 30 tools registered
- [ ] Tool descriptions accurate
- [ ] Parameter validation working
- [ ] Error handling implemented
- [ ] Response formats consistent
- [ ] Performance acceptable

### 8.4 Integration Checklist
- [ ] End-to-end workflows functional
- [ ] Data flow between components working
- [ ] Error propagation handled
- [ ] Performance metrics collected
- [ ] Logging comprehensive
- [ ] Monitoring alerts configured

### 8.5 Post-Testing Checklist
- [ ] All tests passing
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Issues logged and tracked
- [ ] Cleanup completed
- [ ] Results documented

---

## 9. Test Data Examples

### 9.1 Sample Data Files
```bash
# Text data
echo "This is a sample text for testing sentiment analysis and entity extraction." > data/test_text.txt

# JSON data for analytics
cat > data/test_analytics.json << EOF
{
  "forecasting_data": [
    {"date": "2023-01-01", "sales": 100, "temperature": 20},
    {"date": "2023-01-02", "sales": 120, "temperature": 22}
  ],
  "causal_data": [
    {"treatment": 1, "outcome": 10, "covariates": [1, 2, 3]},
    {"treatment": 0, "outcome": 8, "covariates": [1, 2, 3]}
  ]
}
EOF
```

### 9.2 Test Scripts
```python
# test_quick_validation.py
import requests
import json

def test_basic_endpoints():
    base_url = "http://127.0.0.1:8000"
    
    # Test health
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200
    
    # Test sentiment analysis
    response = requests.post(
        f"{base_url}/analyze/sentiment",
        json={"text": "Great product!", "language": "en"}
    )
    assert response.status_code == 200
    
    print("✅ Basic endpoints working")

if __name__ == "__main__":
    test_basic_endpoints()
```

---

## 10. Reporting and Documentation

### 10.1 Test Results Template
```markdown
# Test Results Report

## Test Date: [DATE]
## Tester: [NAME]
## Environment: [DETAILS]

## Summary
- Total Tests: [NUMBER]
- Passed: [NUMBER]
- Failed: [NUMBER]
- Skipped: [NUMBER]

## API Endpoints
- [ ] Health endpoints
- [ ] Core analysis endpoints
- [ ] Advanced analytics endpoints
- [ ] Error handling

## MCP Tools
- [ ] Core tools (25)
- [ ] Advanced analytics tools (5)
- [ ] Parameter validation
- [ ] Error handling

## Performance
- [ ] Response times acceptable
- [ ] Memory usage stable
- [ ] CPU usage reasonable
- [ ] No memory leaks

## Issues Found
1. [ISSUE 1]
2. [ISSUE 2]

## Recommendations
1. [RECOMMENDATION 1]
2. [RECOMMENDATION 2]
```

### 10.2 Automated Test Reports
```bash
# Generate test report
.venv/Scripts/python.exe -m pytest Test/test_phase7_5_integration.py --html=reports/test_report.html --self-contained-html

# View report
open reports/test_report.html
```

---

## Conclusion

This SOP provides a comprehensive framework for testing all MCP tools and API endpoints in the Advanced Analytics System. Follow these steps systematically to ensure thorough validation of all system components.

For additional support or questions, refer to:
- `docs/PHASE7_5_FINAL_INTEGRATION_SUMMARY.md`
- `docs/PREDICTIVE_ANALYTICS_IMPLEMENTATION_PLAN.md`
- `README.md`

**Last Updated:** [Current Date]
**Version:** 1.0
**Status:** Active
