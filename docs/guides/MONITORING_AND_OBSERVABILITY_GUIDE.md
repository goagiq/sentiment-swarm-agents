# Monitoring and Observability Guide

## Overview

This guide documents the comprehensive monitoring and observability system implemented for the Sentiment Analysis & Decision Support System. The system provides real-time monitoring across three key areas: Application Performance, Infrastructure, and Business Metrics.

## 🏗️ Architecture

The monitoring system consists of four main components:

1. **Application Performance Monitor** - Tracks application metrics, errors, and user analytics
2. **Infrastructure Monitor** - Monitors servers, databases, networks, and logs
3. **Business Metrics Monitor** - Tracks decision accuracy, user engagement, and feature usage
4. **Alert System** - Provides notification and alerting capabilities

## 📊 Application Performance Monitoring

### Features

- **Performance Metrics Tracking**: CPU, memory, disk, and network usage
- **Error Tracking**: Comprehensive error logging and analysis
- **User Analytics**: User behavior and session tracking
- **Custom Metrics**: Support for application-specific metrics
- **Alerting Rules**: Configurable thresholds and notifications

### Key Components

#### PerformanceMetric
```python
@dataclass
class PerformanceMetric:
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    category: str  # cpu, memory, disk, network, custom
    tags: Dict[str, str]
    metadata: Dict[str, Any]
```

#### ErrorRecord
```python
@dataclass
class ErrorRecord:
    error_id: str
    error_type: str
    error_message: str
    stack_trace: str
    timestamp: datetime
    severity: str  # low, medium, high, critical
    user_id: Optional[str]
    session_id: Optional[str]
    request_id: Optional[str]
```

### Usage Examples

#### Starting Application Monitoring
```python
from src.core.monitoring.application_monitor import application_monitor

# Start monitoring
await application_monitor.start_monitoring()

# Record an error
await application_monitor.record_error(
    error_type="api_error",
    error_message="Failed to process request",
    stack_trace="...",
    severity="high"
)

# Record user action
await application_monitor.record_user_action(
    user_id="user123",
    session_id="session456",
    action="text_analysis",
    duration=2.5
)

# Get performance summary
summary = await application_monitor.get_performance_summary()
```

## 🏗️ Infrastructure Monitoring

### Features

- **Server Monitoring**: CPU, memory, disk, and network metrics
- **Database Monitoring**: Connection counts, query performance, response times
- **Network Monitoring**: Latency, connectivity, and port status
- **Log Aggregation**: Centralized log collection and analysis
- **Infrastructure Alerts**: Automated alerting for infrastructure issues

### Key Components

#### ServerMetric
```python
@dataclass
class ServerMetric:
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    server_id: str
    category: str  # cpu, memory, disk, network
```

#### DatabaseMetric
```python
@dataclass
class DatabaseMetric:
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    database_name: str
    connection_id: Optional[str]
    query_type: Optional[str]
```

### Usage Examples

#### Starting Infrastructure Monitoring
```python
from src.core.monitoring.infrastructure_monitor import infrastructure_monitor

# Start monitoring
await infrastructure_monitor.start_monitoring()

# Get infrastructure summary
summary = await infrastructure_monitor.get_infrastructure_summary()

# Get server status
server_status = await infrastructure_monitor.get_server_status("localhost")

# Get database status
db_status = await infrastructure_monitor.get_database_status("default")

# Get network status
network_status = await infrastructure_monitor.get_network_status()
```

## 📈 Business Metrics Monitoring

### Features

- **Decision Accuracy Tracking**: Monitor prediction accuracy and confidence
- **User Engagement Monitoring**: Track user sessions and interactions
- **Feature Usage Analytics**: Monitor feature adoption and usage patterns
- **System Performance Metrics**: Track business-relevant performance indicators
- **Business Intelligence Dashboards**: Comprehensive reporting and analytics

### Key Components

#### DecisionAccuracyMetric
```python
@dataclass
class DecisionAccuracyMetric:
    decision_id: str
    decision_type: str
    predicted_outcome: Dict[str, Any]
    actual_outcome: Dict[str, Any]
    accuracy_score: float
    confidence_score: float
    timestamp: datetime
    user_id: Optional[str]
```

#### UserEngagementMetric
```python
@dataclass
class UserEngagementMetric:
    user_id: str
    session_id: str
    action_type: str
    duration: float
    timestamp: datetime
    success: bool
    features_used: List[str]
```

### Usage Examples

#### Starting Business Metrics Monitoring
```python
from src.core.monitoring.business_metrics import business_metrics_monitor

# Start monitoring
await business_metrics_monitor.start_monitoring()

# Record decision accuracy
await business_metrics_monitor.record_decision_accuracy(
    decision_id="decision123",
    decision_type="sentiment_analysis",
    predicted_outcome={"sentiment": "positive", "confidence": 0.8},
    actual_outcome={"sentiment": "positive", "confidence": 0.9},
    accuracy_score=0.9,
    confidence_score=0.85
)

# Record user engagement
await business_metrics_monitor.record_user_engagement(
    user_id="user123",
    session_id="session456",
    action_type="text_analysis",
    duration=2.5,
    features_used=["sentiment_analysis", "entity_extraction"]
)

# Get business summary
summary = await business_metrics_monitor.get_business_summary()
```

## 🚨 Alert System

### Features

- **Multi-channel Notifications**: Email, Slack, webhook, SMS support
- **Configurable Alert Rules**: Flexible threshold and condition settings
- **Alert Management**: Acknowledge, resolve, and track alert status
- **Severity Levels**: Low, medium, high, critical alert classification

### Usage Examples

#### Creating Alert Rules
```python
from src.core.monitoring.alert_system import AlertSystem

alert_system = AlertSystem()

# Create an alert rule
rule_id = await alert_system.create_alert_rule(
    rule_name="High CPU Usage",
    condition="cpu_usage > 80",
    metric_name="cpu_usage",
    threshold_value=80.0,
    severity="high",
    notification_channels=["email", "slack"]
)

# Generate an alert
await alert_system.generate_alert(
    alert_id="alert123",
    alert_type="performance",
    severity="high",
    message="High CPU usage detected",
    decision_id="decision123",
    metric_name="cpu_usage",
    current_value=85.0,
    threshold_value=80.0
)
```

## 🔗 API Endpoints

The monitoring system provides comprehensive REST API endpoints:

### Monitoring Status
- `GET /monitoring/status` - Get overall monitoring system status
- `POST /monitoring/start` - Start all monitoring systems
- `POST /monitoring/stop` - Stop all monitoring systems

### Application Monitoring
- `GET /monitoring/application/summary` - Get application performance summary
- `GET /monitoring/application/errors` - Get error analysis
- `GET /monitoring/application/analytics` - Get user analytics
- `POST /monitoring/application/record-error` - Record an application error
- `POST /monitoring/application/record-action` - Record a user action

### Infrastructure Monitoring
- `GET /monitoring/infrastructure/summary` - Get infrastructure summary
- `GET /monitoring/infrastructure/server/{server_id}` - Get server status
- `GET /monitoring/infrastructure/database/{database_name}` - Get database status
- `GET /monitoring/infrastructure/network` - Get network status

### Business Metrics
- `GET /monitoring/business/summary` - Get business metrics summary
- `GET /monitoring/business/accuracy` - Get decision accuracy report
- `GET /monitoring/business/engagement` - Get user engagement report
- `POST /monitoring/business/record-decision` - Record decision accuracy
- `POST /monitoring/business/record-engagement` - Record user engagement
- `POST /monitoring/business/record-feature-usage` - Record feature usage

### Dashboard
- `GET /monitoring/dashboard/overview` - Get comprehensive dashboard overview
- `GET /monitoring/dashboard/alerts` - Get all active alerts
- `GET /monitoring/dashboard/metrics` - Get key metrics from all systems

### Alert Management
- `POST /monitoring/alerts/{alert_id}/acknowledge` - Acknowledge an alert
- `POST /monitoring/alerts/{alert_id}/resolve` - Resolve an alert

### Health Check
- `GET /monitoring/health` - Health check for monitoring systems

## 🧪 Testing

The monitoring system includes comprehensive testing capabilities:

### Running Tests
```bash
# Run the monitoring system test suite
python Test/test_monitoring_system.py
```

### Test Coverage
The test suite covers:
- Application performance monitoring functionality
- Infrastructure monitoring capabilities
- Business metrics tracking
- Alert system operations
- Decision monitoring features
- Integration between monitoring systems

### Test Report
Tests generate detailed reports including:
- Test success/failure status
- Performance metrics
- Error analysis
- Recommendations for improvement

## 📋 Configuration

### Default Thresholds

#### Application Monitoring
```python
default_thresholds = {
    "cpu_usage": 80.0,
    "memory_usage": 85.0,
    "disk_usage": 90.0,
    "error_rate": 5.0,  # percentage
    "response_time": 2.0,  # seconds
    "request_rate": 1000.0  # requests per minute
}
```

#### Infrastructure Monitoring
```python
thresholds = {
    "cpu_usage": 80.0,
    "memory_usage": 85.0,
    "disk_usage": 90.0,
    "network_latency": 100.0,  # ms
    "database_connections": 80.0,
    "error_log_rate": 10.0  # errors per minute
}
```

#### Business Metrics
```python
thresholds = {
    "decision_accuracy": 0.8,  # 80% accuracy
    "user_engagement_rate": 0.6,  # 60% engagement
    "feature_adoption_rate": 0.3,  # 30% adoption
    "system_availability": 0.99,  # 99% availability
    "response_time": 2.0,  # 2 seconds
    "error_rate": 0.05  # 5% error rate
}
```

## 🔧 Integration

### FastAPI Integration
The monitoring system integrates seamlessly with FastAPI:

```python
from fastapi import FastAPI
from src.api.monitoring_routes import router

app = FastAPI()
app.include_router(router, prefix="/monitoring")
```

### Existing System Integration
The monitoring system integrates with existing components:
- **Alert System**: Leverages existing alert infrastructure
- **Decision Monitor**: Integrates with decision monitoring capabilities
- **Error Handler**: Uses existing error handling patterns
- **Logging**: Integrates with existing logging system

## 📊 Dashboards and Visualization

### Grafana Integration
The system includes Grafana dashboard configurations:
- `monitoring/grafana-dashboard.json` - Dashboard configuration
- `monitoring/grafana-datasource.yml` - Data source configuration

### Prometheus Integration
Prometheus configuration for metrics collection:
- `monitoring/prometheus.yml` - Prometheus configuration

## 🚀 Deployment

### Production Deployment
The monitoring system is designed for production deployment:

1. **Start Monitoring Systems**:
   ```bash
   curl -X POST http://localhost:8003/monitoring/start
   ```

2. **Check System Status**:
   ```bash
   curl http://localhost:8003/monitoring/status
   ```

3. **View Dashboard**:
   ```bash
   curl http://localhost:8003/monitoring/dashboard/overview
   ```

### Docker Integration
The monitoring system works with the existing Docker setup and includes:
- Health checks for monitoring services
- Proper logging configuration
- Resource monitoring integration

## 🔍 Troubleshooting

### Common Issues

1. **Monitoring Not Starting**
   - Check if all dependencies are installed
   - Verify system permissions for metrics collection
   - Check log files for error messages

2. **No Metrics Being Collected**
   - Verify monitoring systems are started
   - Check network connectivity for external metrics
   - Review configuration settings

3. **Alerts Not Triggering**
   - Verify alert rules are properly configured
   - Check notification channel settings
   - Review threshold values

### Debug Mode
Enable debug logging for troubleshooting:
```python
import logging
logging.getLogger("src.core.monitoring").setLevel(logging.DEBUG)
```

## 📈 Performance Considerations

### Optimization Tips
1. **Data Retention**: Configure appropriate data retention periods
2. **Sampling**: Use sampling for high-frequency metrics
3. **Caching**: Implement caching for frequently accessed data
4. **Batch Processing**: Use batch processing for bulk operations

### Resource Usage
- **Memory**: Monitoring systems use minimal memory (~50MB per component)
- **CPU**: Low CPU overhead (~1-2% per monitoring component)
- **Storage**: Configurable data retention with automatic cleanup

## 🔐 Security

### Security Features
- **Input Validation**: All inputs are validated and sanitized
- **Access Control**: API endpoints can be secured with authentication
- **Data Privacy**: User data is anonymized where appropriate
- **Audit Logging**: All monitoring activities are logged

## 📚 Additional Resources

### Documentation
- [API Documentation](./API_DOCUMENTATION.md)
- [Deployment Guide](./DEPLOYMENT_GUIDES.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)

### Examples
- [Monitoring Examples](../examples/)
- [Test Scripts](../Test/)

### Configuration Files
- [Monitoring Configuration](../monitoring/)
- [Grafana Dashboards](../monitoring/grafana-dashboard.json)
- [Prometheus Config](../monitoring/prometheus.yml)

---

**Last Updated**: December 2024
**Version**: 1.0
**Status**: Production Ready
