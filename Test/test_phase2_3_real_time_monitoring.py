#!/usr/bin/env python3
"""
Test script for Phase 2.3: Real-Time Monitoring System

This script tests the real-time monitoring components including:
- Pattern monitoring
- Alert system
- Performance dashboard
- Data stream processing
"""

import sys
import os
import asyncio
import numpy as np
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_pattern_monitor():
    """Test the real-time pattern monitor"""
    print("Testing Real-Time Pattern Monitor...")
    
    from src.core.real_time.pattern_monitor import RealTimePatternMonitor, MonitoringConfig
    
    # Create pattern monitor
    config = MonitoringConfig(
        window_size=50,
        update_interval=0.5,
        enable_anomaly_detection=True,
        enable_trend_detection=True,
        enable_seasonal_detection=True
    )
    
    monitor = RealTimePatternMonitor(config)
    
    # Add callback to track detected patterns
    detected_patterns = []
    def pattern_callback(pattern_event):
        detected_patterns.append(pattern_event)
        print(f"Pattern detected: {pattern_event.pattern_type} (confidence: {pattern_event.confidence:.2f})")
    
    monitor.add_callback(pattern_callback)
    
    # Generate test data with known patterns
    print("Generating test data...")
    
    # Normal data
    for i in range(30):
        monitor.add_data_point(value=100 + np.random.normal(0, 5))
    
    # Anomaly
    monitor.add_data_point(value=200)  # Outlier
    
    # Trend data
    for i in range(20):
        monitor.add_data_point(value=100 + i * 2)  # Upward trend
    
    # Seasonal pattern
    for i in range(40):
        monitor.add_data_point(value=100 + 20 * np.sin(i * 0.5))  # Sine wave
    
    # Spike
    monitor.add_data_point(value=300)  # Sudden spike
    
    # Get statistics
    stats = monitor.get_statistics()
    print(f"Pattern Monitor Statistics: {stats}")
    
    print(f"Total patterns detected: {len(detected_patterns)}")
    return len(detected_patterns) > 0

async def test_alert_system():
    """Test the alert system"""
    print("\nTesting Alert System...")
    
    from src.core.real_time.alert_system import AlertSystem, AlertConfig, AlertRule
    
    # Create alert system
    config = AlertConfig(
        log_enabled=True,
        alert_history_size=100
    )
    
    alert_system = AlertSystem(config)
    
    # Add alert rules
    anomaly_rule = AlertRule(
        rule_id="anomaly_rule",
        pattern_type="anomaly",
        severity_threshold="warning",
        conditions={"confidence_min": 0.5},
        actions=["log"]
    )
    
    trend_rule = AlertRule(
        rule_id="trend_rule",
        pattern_type="trend",
        severity_threshold="info",
        conditions={"confidence_min": 0.3},
        actions=["log"]
    )
    
    alert_system.add_rule(anomaly_rule)
    alert_system.add_rule(trend_rule)
    
    # Create mock pattern events
    from src.core.real_time.pattern_monitor import PatternEvent
    
    anomaly_event = PatternEvent(
        pattern_id="test_anomaly",
        pattern_type="anomaly",
        confidence=0.8,
        timestamp=datetime.now(),
        data_points=[200],
        severity="warning"
    )
    
    trend_event = PatternEvent(
        pattern_id="test_trend",
        pattern_type="trend",
        confidence=0.6,
        timestamp=datetime.now(),
        data_points=list(range(100, 120)),
        severity="info"
    )
    
    # Process events
    await alert_system.process_pattern_event(anomaly_event)
    await alert_system.process_pattern_event(trend_event)
    
    # Get alerts
    alerts = alert_system.get_alerts()
    print(f"Generated alerts: {len(alerts)}")
    
    # Get statistics
    stats = alert_system.get_statistics()
    print(f"Alert System Statistics: {stats}")
    
    return len(alerts) > 0

def test_performance_dashboard():
    """Test the performance dashboard"""
    print("\nTesting Performance Dashboard...")
    
    from src.core.real_time.performance_dashboard import PerformanceDashboard, DashboardConfig
    
    # Create dashboard
    config = DashboardConfig(
        update_interval=1.0,
        max_metrics_history=500
    )
    
    dashboard = PerformanceDashboard(config)
    
    # Add custom metrics
    dashboard.add_metric("custom_metric_1", 75.5, "%", "custom")
    dashboard.add_metric("custom_metric_2", 42.3, "count", "custom")
    dashboard.add_metric("response_time", 125.7, "ms", "performance")
    
    # Set alert thresholds
    dashboard.set_alert_threshold("custom_metric_1", "max", 80.0)
    dashboard.set_alert_threshold("response_time", "max", 200.0)
    
    # Get dashboard summary
    summary = dashboard.get_dashboard_summary()
    print(f"Dashboard Summary: {summary}")
    
    # Get performance report
    report = dashboard.get_performance_report()
    print(f"Performance Report Keys: {list(report.keys())}")
    
    # Get metrics by category
    custom_metrics = dashboard.get_metrics_by_category("custom")
    print(f"Custom Metrics: {len(custom_metrics)}")
    
    return len(custom_metrics) > 0

async def test_stream_processor():
    """Test the data stream processor"""
    print("\nTesting Data Stream Processor...")
    
    from src.core.real_time.stream_processor import DataStreamProcessor, StreamConfig
    
    # Create stream processor
    config = StreamConfig(
        buffer_size=100,
        batch_size=10,
        processing_interval=0.1
    )
    
    processor = DataStreamProcessor(config)
    
    # Add processors and filters
    processor.add_numeric_processor("scale_processor", "scale")
    processor.add_threshold_filter("range_filter", min_value=0, max_value=1000)
    processor.add_average_aggregator("avg_aggregator")
    
    # Track processed data
    processed_data = []
    def data_consumer(data):
        processed_data.append(data)
        print(f"Processed data: {data}")
    
    processor.register_consumer("default", data_consumer)
    
    # Add data points
    for i in range(20):
        processor.add_data_point(
            value=50 + np.random.normal(0, 10),
            source="test_source",
            metadata={"scale_factor": 1.5}
        )
    
    # Wait for processing
    await asyncio.sleep(1.0)
    
    # Get processing statistics
    stats = processor.get_processing_stats()
    print(f"Stream Processing Statistics: {stats}")
    
    # Get stream data
    stream_data = processor.get_stream_data("default", limit=10)
    print(f"Stream Data Count: {len(stream_data)}")
    
    return len(processed_data) > 0

async def test_real_time_monitoring_agent():
    """Test the real-time monitoring agent"""
    print("\nTesting Real-Time Monitoring Agent...")
    
    from src.agents.real_time_monitoring_agent import RealTimeMonitoringAgent
    from src.core.models import AnalysisRequest, DataType
    
    # Create agent
    agent = RealTimeMonitoringAgent()
    
    # Test pattern monitoring
    pattern_request = AnalysisRequest(
        data_type=DataType.NUMERICAL,
        content={"data": [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]},
        metadata={"monitoring_type": "pattern"}
    )
    
    # Test performance monitoring
    performance_request = AnalysisRequest(
        data_type=DataType.NUMERICAL,
        content={"data": {"cpu_usage": 75.5, "memory_usage": 60.2}},
        metadata={"monitoring_type": "performance"}
    )
    
    # Test stream processing
    stream_request = AnalysisRequest(
        data_type=DataType.NUMERICAL,
        content={"data": [50, 55, 60, 65, 70]},
        metadata={"monitoring_type": "stream", "source": "test_stream"}
    )
    
    # Test alert management
    alert_request = AnalysisRequest(
        data_type=DataType.TEXT,
        content={"data": "test"},
        metadata={
            "monitoring_type": "alert",
            "operation": "add_rule",
            "pattern_type": "anomaly",
            "severity_threshold": "warning",
            "actions": ["log"]
        }
    )
    
    # Process requests
    results = []
    
    # Pattern monitoring
    result = await agent.process(pattern_request)
    results.append(("pattern", result.status == "completed"))
    print(f"Pattern monitoring result: {result.status}")
    
    # Performance monitoring
    result = await agent.process(performance_request)
    results.append(("performance", result.status == "completed"))
    print(f"Performance monitoring result: {result.status}")
    
    # Stream processing
    result = await agent.process(stream_request)
    results.append(("stream", result.status == "completed"))
    print(f"Stream processing result: {result.status}")
    
    # Alert management
    result = await agent.process(alert_request)
    results.append(("alert", result.status == "completed"))
    print(f"Alert management result: {result.status}")
    
    # Get monitoring status
    status = agent.get_monitoring_status()
    print(f"Agent Status Keys: {list(status.keys())}")
    
    # Get capabilities
    capabilities = agent.get_capabilities()
    print(f"Agent Capabilities: {capabilities['monitoring_types']}")
    
    return all(success for _, success in results)

async def main():
    """Run all tests"""
    print("=" * 60)
    print("Phase 2.3: Real-Time Monitoring System Test")
    print("=" * 60)
    
    test_results = []
    
    # Test individual components
    test_results.append(("Pattern Monitor", test_pattern_monitor()))
    test_results.append(("Alert System", await test_alert_system()))
    test_results.append(("Performance Dashboard", test_performance_dashboard()))
    test_results.append(("Stream Processor", await test_stream_processor()))
    test_results.append(("Real-Time Monitoring Agent", await test_real_time_monitoring_agent()))
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All Phase 2.3 tests PASSED!")
        print("Real-Time Monitoring System is working correctly.")
    else:
        print("⚠️  Some tests FAILED. Please check the implementation.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
