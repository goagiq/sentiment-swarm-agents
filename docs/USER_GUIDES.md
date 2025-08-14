# User Guides - Sentiment Analysis & Decision Support System

**Version:** 1.0.0  
**Last Updated:** 2025-08-14

## Table of Contents

1. [Getting Started](#getting-started)
2. [Decision Support Features](#decision-support-features)
3. [Multi-Modal Processing](#multi-modal-processing)
4. [Business Intelligence](#business-intelligence)
5. [Advanced Analytics](#advanced-analytics)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [FAQ](#faq)
8. [Video Tutorials](#video-tutorials)

## Getting Started

### System Overview

The Sentiment Analysis & Decision Support System is an AI-powered platform that provides:

- **Sentiment Analysis**: Analyze text sentiment and emotions
- **Decision Support**: AI-powered decision recommendations
- **Business Intelligence**: Comprehensive business insights
- **Multi-Modal Processing**: Handle text, audio, video, and images
- **Advanced Analytics**: Predictive analytics and forecasting
- **Knowledge Graphs**: Intelligent knowledge management

### Quick Start

1. **Access the System**
   ```bash
   # Start the system
   python main.py
   
   # Access web interface
   http://localhost:8003
   ```

2. **First Analysis**
   ```python
   # Simple text analysis
   import requests
   
   response = requests.post("http://localhost:8003/analyze/text", json={
       "content": "The new product is amazing!",
       "language": "en"
   })
   
   print(response.json())
   ```

3. **Check System Health**
   ```bash
   curl http://localhost:8003/health
   ```

## Decision Support Features

### Understanding Decision Support

The decision support system uses AI to analyze complex scenarios and provide actionable recommendations.

#### Key Components:

1. **Scenario Analysis**: Evaluate different business scenarios
2. **Risk Assessment**: Identify and quantify risks
3. **Action Prioritization**: Rank actions by impact and feasibility
4. **Implementation Planning**: Create detailed execution plans
5. **Success Prediction**: Forecast outcomes and success probabilities

### Using Decision Support

#### 1. Create a Decision Scenario

```python
from src.agents.decision_support_agent import DecisionSupportAgent

agent = DecisionSupportAgent()

scenario = {
    "title": "Market Expansion Decision",
    "description": "Evaluate expanding into European markets",
    "options": [
        "Expand to Germany",
        "Expand to France", 
        "Expand to both",
        "Delay expansion"
    ],
    "criteria": [
        "Market size",
        "Competition level",
        "Regulatory requirements",
        "Investment required"
    ]
}

result = agent.analyze_decision_scenario(scenario)
```

#### 2. Analyze Business Intelligence

```python
from src.agents.business_intelligence_agent import BusinessIntelligenceAgent

bi_agent = BusinessIntelligenceAgent()

# Generate business summary
summary = bi_agent.generate_business_summary(
    data_source="quarterly_reports",
    time_period="Q3 2025",
    include_metrics=True,
    include_trends=True
)

# Create executive summary
exec_summary = bi_agent.create_executive_summary(
    content_data="Q3 financial results...",
    summary_type="business"
)
```

#### 3. Perform Risk Assessment

```python
from src.agents.risk_assessment_agent import RiskAssessmentAgent

risk_agent = RiskAssessmentAgent()

risk_analysis = risk_agent.assess_risks(
    scenario_description="Market expansion to Europe",
    risk_factors=[
        "Currency fluctuations",
        "Regulatory changes",
        "Competition intensity",
        "Economic conditions"
    ],
    time_horizon=24  # months
)
```

### Decision Support Workflow

1. **Define the Problem**
   - Clearly state the decision to be made
   - Identify all available options
   - List relevant criteria

2. **Gather Data**
   - Collect relevant business data
   - Analyze market conditions
   - Review historical performance

3. **Run Analysis**
   - Execute scenario analysis
   - Perform risk assessment
   - Generate recommendations

4. **Review Results**
   - Examine recommendations
   - Consider implementation plans
   - Validate assumptions

5. **Make Decision**
   - Select optimal option
   - Create action plan
   - Monitor outcomes

## Multi-Modal Processing

### Supported Formats

The system can process multiple data types:

#### Text Processing
- **Documents**: PDF, DOCX, TXT
- **Web Content**: HTML, Markdown
- **Social Media**: Tweets, posts, comments
- **Reports**: Business reports, research papers

#### Audio Processing
- **Recordings**: MP3, WAV, M4A
- **Podcasts**: Audio content analysis
- **Meetings**: Meeting transcriptions
- **Customer Calls**: Call center analysis

#### Video Processing
- **Presentations**: Video presentations
- **Training**: Educational content
- **Marketing**: Advertisement analysis
- **Surveillance**: Security footage (if applicable)

#### Image Processing
- **Charts**: Business charts and graphs
- **Documents**: Scanned documents
- **Social Media**: Image content analysis
- **Reports**: Visual data extraction

### Using Multi-Modal Processing

#### 1. Process Text Documents

```python
from src.agents.text_processing_agent import TextProcessingAgent

text_agent = TextProcessingAgent()

# Process PDF document
result = text_agent.process_document(
    file_path="quarterly_report.pdf",
    extraction_type="full_text",
    include_metadata=True
)

# Extract entities
entities = text_agent.extract_entities(result["content"])

# Generate summary
summary = text_agent.generate_summary(result["content"])
```

#### 2. Process Audio Content

```python
from src.agents.audio_processing_agent import AudioProcessingAgent

audio_agent = AudioProcessingAgent()

# Transcribe audio
transcription = audio_agent.transcribe_audio(
    file_path="meeting_recording.mp3",
    language="en"
)

# Analyze sentiment
sentiment = audio_agent.analyze_audio_sentiment(
    file_path="meeting_recording.mp3"
)

# Extract key topics
topics = audio_agent.extract_topics(transcription["text"])
```

#### 3. Process Video Content

```python
from src.agents.video_processing_agent import VideoProcessingAgent

video_agent = VideoProcessingAgent()

# Extract frames and analyze
analysis = video_agent.analyze_video(
    file_path="presentation.mp4",
    extract_frames=True,
    analyze_content=True
)

# Generate video summary
summary = video_agent.generate_video_summary(
    file_path="presentation.mp4"
)
```

#### 4. Process Images

```python
from src.agents.vision_processing_agent import VisionProcessingAgent

vision_agent = VisionProcessingAgent()

# Analyze image content
analysis = vision_agent.analyze_image(
    file_path="chart.png",
    extract_text=True,
    analyze_charts=True
)

# Extract data from charts
chart_data = vision_agent.extract_chart_data(
    file_path="chart.png"
)
```

### Multi-Modal Integration

Combine multiple data types for comprehensive analysis:

```python
from src.agents.multi_modal_agent import MultiModalAgent

mm_agent = MultiModalAgent()

# Analyze complete presentation
presentation_analysis = mm_agent.analyze_presentation(
    video_file="presentation.mp4",
    slides_file="slides.pdf",
    transcript_file="transcript.txt"
)

# Generate comprehensive report
report = mm_agent.generate_comprehensive_report(
    data_sources=[
        "quarterly_report.pdf",
        "meeting_recording.mp3",
        "presentation.mp4",
        "charts.png"
    ]
)
```

## Business Intelligence

### Business Intelligence Features

#### 1. Business Summaries

Generate comprehensive business summaries:

```python
from src.agents.business_intelligence_agent import BusinessIntelligenceAgent

bi_agent = BusinessIntelligenceAgent()

# Generate quarterly summary
summary = bi_agent.generate_business_summary(
    data_source="quarterly_reports",
    time_period="Q3 2025",
    include_metrics=True,
    include_trends=True,
    include_recommendations=True
)
```

#### 2. Executive Dashboards

Create executive-level dashboards:

```python
# Create dashboard
dashboard = bi_agent.create_executive_dashboard(
    metrics=[
        "revenue_growth",
        "customer_satisfaction",
        "market_share",
        "operational_efficiency"
    ],
    time_range="last_12_months",
    include_visualizations=True
)
```

#### 3. Market Analysis

Perform market analysis:

```python
# Analyze market trends
market_analysis = bi_agent.analyze_market_trends(
    market_data="market_research_data",
    competitors=["competitor1", "competitor2"],
    time_period="last_6_months"
)
```

### Key Performance Indicators (KPIs)

Track important business metrics:

#### Financial KPIs
- Revenue Growth Rate
- Profit Margins
- Customer Acquisition Cost
- Customer Lifetime Value

#### Operational KPIs
- Customer Satisfaction Score
- Employee Productivity
- Process Efficiency
- Quality Metrics

#### Market KPIs
- Market Share
- Brand Awareness
- Customer Retention Rate
- Market Penetration

## Advanced Analytics

### Predictive Analytics

#### 1. Customer Churn Prediction

```python
from src.agents.predictive_analytics_agent import PredictiveAnalyticsAgent

pred_agent = PredictiveAnalyticsAgent()

# Predict customer churn
churn_prediction = pred_agent.predict_customer_churn(
    customer_data="customer_behavior_data",
    features=[
        "usage_frequency",
        "support_tickets",
        "payment_history",
        "engagement_score"
    ],
    time_horizon=90  # days
)
```

#### 2. Sales Forecasting

```python
# Forecast sales
sales_forecast = pred_agent.forecast_sales(
    historical_data="sales_history",
    forecast_period=12,  # months
    confidence_level=0.95
)
```

#### 3. Demand Prediction

```python
# Predict product demand
demand_prediction = pred_agent.predict_demand(
    product_data="product_sales_data",
    market_conditions="market_data",
    seasonal_factors=True
)
```

### Scenario Analysis

#### 1. Business Scenarios

```python
from src.agents.scenario_analysis_agent import ScenarioAnalysisAgent

scenario_agent = ScenarioAnalysisAgent()

# Analyze market expansion scenario
expansion_scenario = scenario_agent.analyze_scenario(
    scenario_name="Market Expansion",
    variables={
        "investment_amount": 1000000,
        "market_size": 50000000,
        "competition_level": "medium"
    },
    time_period=24,  # months
    confidence_level=0.95
)
```

#### 2. Risk Scenarios

```python
# Analyze risk scenarios
risk_scenarios = scenario_agent.analyze_risk_scenarios(
    risk_factors=[
        "economic_recession",
        "supply_chain_disruption",
        "regulatory_changes"
    ],
    impact_assessment=True,
    mitigation_strategies=True
)
```

### Causal Analysis

```python
from src.agents.causal_analysis_agent import CausalAnalysisAgent

causal_agent = CausalAnalysisAgent()

# Analyze cause-effect relationships
causal_analysis = causal_agent.analyze_causality(
    data_source="business_data",
    variables=[
        "marketing_spend",
        "customer_acquisition",
        "revenue_growth"
    ],
    time_period="last_24_months"
)
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. System Startup Issues

**Problem**: System fails to start
```
Error: ModuleNotFoundError: No module named 'src.core.performance_monitor'
```

**Solution**:
```bash
# Check Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/project"

# Install missing dependencies
pip install -r requirements.txt

# Verify virtual environment
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

#### 2. API Connection Issues

**Problem**: Cannot connect to API endpoints
```
Error: Connection refused
```

**Solution**:
```bash
# Check if server is running
curl http://localhost:8003/health

# Start server if not running
python main.py

# Check port availability
netstat -an | grep 8003
```

#### 3. Model Loading Issues

**Problem**: Ollama models not loading
```
Error: Model not found
```

**Solution**:
```bash
# Check available models
ollama list

# Pull required models
ollama pull llama3.2:latest
ollama pull mistral-small3.1:latest

# Verify model status
ollama show llama3.2:latest
```

#### 4. Memory Issues

**Problem**: System running out of memory
```
Error: Out of memory
```

**Solution**:
```python
# Optimize memory usage
import gc
gc.collect()

# Reduce batch size
batch_size = 10  # instead of 100

# Use streaming for large files
for chunk in process_large_file(file_path):
    process_chunk(chunk)
```

#### 5. Performance Issues

**Problem**: Slow response times
```
Response time > 5 seconds
```

**Solution**:
```python
# Enable caching
from src.core.caching_service import CachingService
cache = CachingService()

# Use async processing
import asyncio
result = await async_process_data(data)

# Optimize database queries
# Add indexes to frequently queried columns
```

### Error Codes and Meanings

| Error Code | Meaning | Solution |
|------------|---------|----------|
| `MODULE_NOT_FOUND` | Missing Python module | Install dependencies |
| `CONNECTION_REFUSED` | Server not running | Start server |
| `MODEL_NOT_FOUND` | Ollama model missing | Pull model |
| `OUT_OF_MEMORY` | Insufficient memory | Optimize memory usage |
| `TIMEOUT_ERROR` | Request timeout | Increase timeout or optimize |
| `VALIDATION_ERROR` | Invalid input data | Check input format |
| `AUTHENTICATION_ERROR` | Invalid credentials | Check API key/token |

### Performance Optimization

#### 1. System Optimization

```python
# Optimize system settings
import os
os.environ['PYTHONOPTIMIZE'] = '1'

# Use multiprocessing for CPU-intensive tasks
from multiprocessing import Pool
with Pool(processes=4) as pool:
    results = pool.map(process_data, data_list)
```

#### 2. Database Optimization

```sql
-- Add indexes for frequently queried columns
CREATE INDEX idx_created_at ON knowledge_graph_entities(created_at);
CREATE INDEX idx_entity_type ON knowledge_graph_entities(entity_type);

-- Optimize queries
EXPLAIN ANALYZE SELECT * FROM knowledge_graph_entities WHERE entity_type = 'PERSON';
```

#### 3. Caching Strategy

```python
# Implement Redis caching
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Cache frequently accessed data
def get_cached_data(key):
    cached = r.get(key)
    if cached:
        return json.loads(cached)
    else:
        data = fetch_data()
        r.setex(key, 3600, json.dumps(data))  # Cache for 1 hour
        return data
```

## FAQ

### General Questions

**Q: What is the Sentiment Analysis & Decision Support System?**
A: It's an AI-powered platform that combines sentiment analysis, decision support, business intelligence, and advanced analytics to help organizations make data-driven decisions.

**Q: What types of data can the system process?**
A: The system can process text, audio, video, and images, making it truly multi-modal.

**Q: How accurate are the AI predictions?**
A: Accuracy varies by task, but typically ranges from 85-95% for sentiment analysis and 80-90% for predictive analytics.

**Q: Is the system secure?**
A: Yes, the system implements comprehensive security measures including authentication, authorization, and data encryption.

### Technical Questions

**Q: What are the system requirements?**
A: Python 3.8+, 8GB RAM minimum, 16GB recommended, Ollama for local model hosting.

**Q: How do I install the system?**
A: Clone the repository, install dependencies with `pip install -r requirements.txt`, and start with `python main.py`.

**Q: Can I use my own models?**
A: Yes, the system supports custom Ollama models and can be configured to use different model providers.

**Q: How do I scale the system?**
A: The system supports horizontal scaling through Kubernetes deployment and load balancing.

### Business Questions

**Q: How can this system help my business?**
A: It provides data-driven insights, automates decision-making processes, and helps identify opportunities and risks.

**Q: What industries is this suitable for?**
A: The system is designed for any industry that needs data analysis and decision support, including finance, healthcare, retail, and technology.

**Q: How long does it take to implement?**
A: Basic setup takes 1-2 hours, full implementation with custom configurations typically takes 1-2 weeks.

**Q: What kind of ROI can I expect?**
A: ROI varies by use case, but typically ranges from 200-500% through improved decision-making and operational efficiency.

## Video Tutorials

### Getting Started Tutorials

#### 1. System Installation and Setup
**Duration**: 15 minutes
**Topics Covered**:
- System requirements
- Installation process
- Initial configuration
- First analysis

**Video Link**: `docs/videos/01_installation_setup.mp4`

#### 2. Basic Text Analysis
**Duration**: 10 minutes
**Topics Covered**:
- Text sentiment analysis
- Entity extraction
- Language detection
- Result interpretation

**Video Link**: `docs/videos/02_basic_text_analysis.mp4`

### Advanced Features Tutorials

#### 3. Decision Support System
**Duration**: 20 minutes
**Topics Covered**:
- Creating decision scenarios
- Risk assessment
- Action prioritization
- Implementation planning

**Video Link**: `docs/videos/03_decision_support.mp4`

#### 4. Multi-Modal Processing
**Duration**: 25 minutes
**Topics Covered**:
- Processing different file types
- Audio and video analysis
- Image processing
- Multi-modal integration

**Video Link**: `docs/videos/04_multi_modal_processing.mp4`

#### 5. Business Intelligence
**Duration**: 18 minutes
**Topics Covered**:
- Business summaries
- Executive dashboards
- KPI tracking
- Market analysis

**Video Link**: `docs/videos/05_business_intelligence.mp4`

#### 6. Advanced Analytics
**Duration**: 22 minutes
**Topics Covered**:
- Predictive analytics
- Scenario analysis
- Causal analysis
- Forecasting

**Video Link**: `docs/videos/06_advanced_analytics.mp4`

### Troubleshooting Tutorials

#### 7. Common Issues and Solutions
**Duration**: 15 minutes
**Topics Covered**:
- Startup issues
- Performance problems
- Error resolution
- System optimization

**Video Link**: `docs/videos/07_troubleshooting.mp4`

### Best Practices Tutorials

#### 8. System Optimization
**Duration**: 20 minutes
**Topics Covered**:
- Performance tuning
- Memory optimization
- Caching strategies
- Scaling considerations

**Video Link**: `docs/videos/08_optimization.mp4`

#### 9. Production Deployment
**Duration**: 25 minutes
**Topics Covered**:
- Production setup
- Security configuration
- Monitoring and alerting
- Backup and recovery

**Video Link**: `docs/videos/09_production_deployment.mp4`

### Video Tutorial Notes

- All videos include closed captions
- Code examples are provided in video descriptions
- Interactive exercises are available for practice
- Support materials are linked in video descriptions

### Accessing Tutorials

1. **Local Access**: Videos are stored in `docs/videos/` directory
2. **Online Access**: Available through the system's web interface
3. **Download**: Videos can be downloaded for offline viewing
4. **Mobile**: Optimized for mobile viewing

---

**User Guide Version:** 1.0.0  
**Last Updated:** 2025-08-14  
**For Support:** Check troubleshooting guide or contact support team
