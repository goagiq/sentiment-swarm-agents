#!/usr/bin/env python3
"""
Comprehensive Test Suite for Real-Time Monitoring - Phase 6.2

This script provides comprehensive testing for all real-time monitoring components including:
- Full system testing with edge cases
- Performance testing with load and stress testing
- Integration testing across all components
- User acceptance testing and validation
- Error handling and recovery testing
"""

import sys
import os
import asyncio
import time
import json
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from loguru import logger

# Import all real-time monitoring components
from src.core.real_time.pattern_monitor import PatternMonitor
from src.core.real_time.alert_system import AlertSystem
from src.core.real_time.performance_dashboard import PerformanceDashboard
from src.core.real_time.stream_processor import StreamProcessor
from src.agents.real_time_monitoring_agent import RealTimeMonitoringAgent
from src.core.models import DataType, AnalysisRequest, AnalysisResult
from src.core.orchestrator import SentimentOrchestrator


class ComprehensiveRealTimeMonitoringTester:
    """Comprehensive test suite for real-time monitoring components."""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        self.start_time = None
        self.orchestrator = None
        
    def log_test_result(self, test_name: str, status: str, message: str = "", 
                       duration: float = 0, metrics: Dict[str, Any] = None):
        """Log test result with comprehensive metrics."""
        result = {
            "test_name": test_name,
            "status": status,
            "message": message,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {}
        }
        self.test_results.append(result)
        logger.info(f"{'✅' if status == 'PASSED' else '❌'} {test_name}: {status} ({duration:.2f}s)")
        if message:
            logger.info(f"   Message: {message}")
    
    def generate_test_data_stream(self, num_points: int = 100) -> list:
        """Generate test data stream for monitoring."""
        data_stream = []
        for i in range(num_points):
            data_point = {
                "timestamp": datetime.now().isoformat(),
                "metric_name": f"test_metric_{i % 5}",
                "value": 50 + (i * 0.1) + (i % 10),
                "source": "test_system",
                "tags": {"environment": "test", "service": "monitoring"}
            }
            data_stream.append(data_point)
        return data_stream
    
    async def setup(self):
        """Setup test environment."""
        logger.info("Setting up comprehensive real-time monitoring test environment...")
        self.start_time = time.time()
        
        # Initialize orchestrator
        self.orchestrator = SentimentOrchestrator()
        
        # Initialize components
        self.pattern_monitor = PatternMonitor()
        self.alert_system = AlertSystem()
        self.performance_dashboard = PerformanceDashboard()
        self.stream_processor = StreamProcessor()
        self.monitoring_agent = RealTimeMonitoringAgent()
        
        logger.info("✅ Test environment setup complete")
    
    async def test_pattern_monitor_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of pattern monitor."""
        logger.info("🧪 Testing Pattern Monitor (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_monitoring": False,
                "pattern_detection": False,
                "anomaly_detection": False,
                "real_time_processing": False,
                "performance": False
            }
            
            # Test 1: Basic monitoring
            logger.info("   Testing basic monitoring...")
            data_stream = self.generate_test_data_stream(50)
            
            monitoring_result = self.pattern_monitor.start_monitoring(
                data_source="test_stream",
                metrics=["test_metric_0", "test_metric_1"],
                thresholds={"test_metric_0": 60, "test_metric_1": 55}
            )
            
            assert monitoring_result is not None, "Should start monitoring"
            assert monitoring_result.status == "active", "Monitoring should be active"
            test_results["basic_monitoring"] = True
            
            # Test 2: Pattern detection
            logger.info("   Testing pattern detection...")
            # Create data with patterns
            pattern_data = []
            for i in range(100):
                # Create a repeating pattern
                value = 50 + 10 * (i % 10)  # Pattern every 10 points
                pattern_data.append({
                    "timestamp": datetime.now().isoformat(),
                    "metric_name": "pattern_metric",
                    "value": value,
                    "source": "test_system"
                })
            
            patterns = self.pattern_monitor.detect_patterns(pattern_data)
            
            assert patterns is not None, "Should detect patterns"
            assert hasattr(patterns, 'detected_patterns'), "Should have detected patterns"
            test_results["pattern_detection"] = True
            
            # Test 3: Anomaly detection
            logger.info("   Testing anomaly detection...")
            # Create data with anomalies
            anomaly_data = []
            for i in range(50):
                value = 50 + (i * 0.1)
                if i == 25:  # Insert anomaly
                    value = 200
                anomaly_data.append({
                    "timestamp": datetime.now().isoformat(),
                    "metric_name": "anomaly_metric",
                    "value": value,
                    "source": "test_system"
                })
            
            anomalies = self.pattern_monitor.detect_anomalies(anomaly_data)
            
            assert anomalies is not None, "Should detect anomalies"
            assert len(anomalies.detected_anomalies) > 0, "Should detect at least one anomaly"
            test_results["anomaly_detection"] = True
            
            # Test 4: Real-time processing
            logger.info("   Testing real-time processing...")
            real_time_result = self.pattern_monitor.process_real_time_data(
                data_stream=data_stream,
                processing_mode="continuous"
            )
            
            assert real_time_result is not None, "Should process real-time data"
            assert hasattr(real_time_result, 'processed_metrics'), "Should have processed metrics"
            test_results["real_time_processing"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            large_data_stream = self.generate_test_data_stream(1000)
            
            perf_start = time.time()
            self.pattern_monitor.process_real_time_data(large_data_stream)
            perf_time = time.time() - perf_start
            
            assert perf_time < 5.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Pattern Monitor Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Pattern Monitor Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_alert_system_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of alert system."""
        logger.info("🧪 Testing Alert System (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_alerting": False,
                "alert_categorization": False,
                "notification_delivery": False,
                "alert_escalation": False,
                "performance": False
            }
            
            # Test 1: Basic alerting
            logger.info("   Testing basic alerting...")
            alert_config = {
                "metric_name": "cpu_usage",
                "threshold": 80.0,
                "condition": "greater_than",
                "severity": "warning"
            }
            
            alert_result = self.alert_system.create_alert(alert_config)
            
            assert alert_result is not None, "Should create alert"
            assert alert_result.status == "active", "Alert should be active"
            test_results["basic_alerting"] = True
            
            # Test 2: Alert categorization
            logger.info("   Testing alert categorization...")
            categories = ["performance", "security", "availability", "business"]
            
            for category in categories:
                category_alert = self.alert_system.create_alert({
                    **alert_config,
                    "category": category,
                    "metric_name": f"{category}_metric"
                })
                assert category_alert is not None, f"Should create {category} alert"
            
            test_results["alert_categorization"] = True
            
            # Test 3: Notification delivery
            logger.info("   Testing notification delivery...")
            notification_config = {
                "channels": ["email", "slack", "webhook"],
                "recipients": ["admin@test.com", "ops@test.com"],
                "template": "standard_alert"
            }
            
            notification_result = self.alert_system.send_notification(
                alert=alert_result,
                config=notification_config
            )
            
            assert notification_result is not None, "Should send notification"
            assert hasattr(notification_result, 'delivery_status'), "Should have delivery status"
            test_results["notification_delivery"] = True
            
            # Test 4: Alert escalation
            logger.info("   Testing alert escalation...")
            escalation_config = {
                "escalation_rules": [
                    {"delay_minutes": 5, "severity": "critical"},
                    {"delay_minutes": 15, "severity": "emergency"}
                ],
                "escalation_contacts": ["manager@test.com", "cto@test.com"]
            }
            
            escalation_result = self.alert_system.setup_escalation(
                alert=alert_result,
                config=escalation_config
            )
            
            assert escalation_result is not None, "Should setup escalation"
            assert hasattr(escalation_result, 'escalation_rules'), "Should have escalation rules"
            test_results["alert_escalation"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            many_alerts = []
            for i in range(50):
                alert = self.alert_system.create_alert({
                    "metric_name": f"metric_{i}",
                    "threshold": 50 + i,
                    "condition": "greater_than",
                    "severity": "warning"
                })
                many_alerts.append(alert)
            
            perf_start = time.time()
            for alert in many_alerts:
                self.alert_system.send_notification(alert, notification_config)
            perf_time = time.time() - perf_start
            
            assert perf_time < 10.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Alert System Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Alert System Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_performance_dashboard_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of performance dashboard."""
        logger.info("🧪 Testing Performance Dashboard (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_dashboard": False,
                "metric_visualization": False,
                "real_time_updates": False,
                "dashboard_customization": False,
                "performance": False
            }
            
            # Test 1: Basic dashboard
            logger.info("   Testing basic dashboard...")
            dashboard_config = {
                "name": "Test Dashboard",
                "description": "Comprehensive test dashboard",
                "metrics": ["cpu_usage", "memory_usage", "response_time"],
                "layout": "grid"
            }
            
            dashboard = self.performance_dashboard.create_dashboard(dashboard_config)
            
            assert dashboard is not None, "Should create dashboard"
            assert dashboard.name == "Test Dashboard", "Dashboard name should match"
            assert len(dashboard.metrics) == 3, "Should have 3 metrics"
            test_results["basic_dashboard"] = True
            
            # Test 2: Metric visualization
            logger.info("   Testing metric visualization...")
            visualization_config = {
                "chart_type": "line",
                "time_range": "1h",
                "refresh_interval": 30
            }
            
            visualization = self.performance_dashboard.create_visualization(
                dashboard=dashboard,
                config=visualization_config
            )
            
            assert visualization is not None, "Should create visualization"
            assert hasattr(visualization, 'chart_data'), "Should have chart data"
            test_results["metric_visualization"] = True
            
            # Test 3: Real-time updates
            logger.info("   Testing real-time updates...")
            update_config = {
                "update_frequency": 5,  # seconds
                "data_source": "real_time_stream",
                "auto_refresh": True
            }
            
            update_result = self.performance_dashboard.enable_real_time_updates(
                dashboard=dashboard,
                config=update_config
            )
            
            assert update_result is not None, "Should enable real-time updates"
            assert update_result.status == "active", "Updates should be active"
            test_results["real_time_updates"] = True
            
            # Test 4: Dashboard customization
            logger.info("   Testing dashboard customization...")
            customization_config = {
                "theme": "dark",
                "layout": "custom",
                "widgets": ["metric_card", "chart", "alert_panel"],
                "filters": ["time_range", "service", "environment"]
            }
            
            customized_dashboard = self.performance_dashboard.customize_dashboard(
                dashboard=dashboard,
                config=customization_config
            )
            
            assert customized_dashboard is not None, "Should customize dashboard"
            assert hasattr(customized_dashboard, 'theme'), "Should have theme"
            assert hasattr(customized_dashboard, 'widgets'), "Should have widgets"
            test_results["dashboard_customization"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            complex_dashboard_config = {
                "name": "Complex Dashboard",
                "metrics": [f"metric_{i}" for i in range(20)],
                "visualizations": ["line", "bar", "gauge", "table"],
                "real_time": True
            }
            
            perf_start = time.time()
            complex_dashboard = self.performance_dashboard.create_dashboard(complex_dashboard_config)
            perf_time = time.time() - perf_start
            
            assert perf_time < 8.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Performance Dashboard Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Performance Dashboard Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_stream_processor_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of stream processor."""
        logger.info("🧪 Testing Stream Processor (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_processing": False,
                "data_transformation": False,
                "stream_aggregation": False,
                "error_handling": False,
                "performance": False
            }
            
            # Test 1: Basic processing
            logger.info("   Testing basic processing...")
            data_stream = self.generate_test_data_stream(100)
            
            processing_config = {
                "batch_size": 10,
                "processing_mode": "real_time",
                "output_format": "json"
            }
            
            processed_data = self.stream_processor.process_stream(
                data_stream=data_stream,
                config=processing_config
            )
            
            assert processed_data is not None, "Should process data stream"
            assert len(processed_data) > 0, "Should have processed data"
            test_results["basic_processing"] = True
            
            # Test 2: Data transformation
            logger.info("   Testing data transformation...")
            transform_config = {
                "transformations": [
                    {"type": "filter", "condition": "value > 50"},
                    {"type": "aggregate", "function": "average", "window": "5m"},
                    {"type": "enrich", "fields": ["timestamp", "source"]}
                ]
            }
            
            transformed_data = self.stream_processor.transform_data(
                data_stream=data_stream,
                config=transform_config
            )
            
            assert transformed_data is not None, "Should transform data"
            assert hasattr(transformed_data, 'transformed_records'), "Should have transformed records"
            test_results["data_transformation"] = True
            
            # Test 3: Stream aggregation
            logger.info("   Testing stream aggregation...")
            aggregation_config = {
                "aggregation_rules": [
                    {"metric": "cpu_usage", "function": "average", "window": "1m"},
                    {"metric": "memory_usage", "function": "max", "window": "5m"},
                    {"metric": "response_time", "function": "percentile", "window": "10m"}
                ]
            }
            
            aggregated_data = self.stream_processor.aggregate_stream(
                data_stream=data_stream,
                config=aggregation_config
            )
            
            assert aggregated_data is not None, "Should aggregate data"
            assert hasattr(aggregated_data, 'aggregated_metrics'), "Should have aggregated metrics"
            test_results["stream_aggregation"] = True
            
            # Test 4: Error handling
            logger.info("   Testing error handling...")
            # Create data with errors
            error_data = [
                {"timestamp": "invalid_timestamp", "value": "not_a_number"},
                {"timestamp": datetime.now().isoformat(), "value": 50.0},  # Valid
                {"timestamp": datetime.now().isoformat(), "value": None}   # Invalid
            ]
            
            error_result = self.stream_processor.process_stream(
                data_stream=error_data,
                config={"error_handling": "continue"}
            )
            
            assert error_result is not None, "Should handle errors gracefully"
            assert hasattr(error_result, 'error_count'), "Should track error count"
            test_results["error_handling"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            large_stream = self.generate_test_data_stream(5000)
            
            perf_start = time.time()
            self.stream_processor.process_stream(large_stream, processing_config)
            perf_time = time.time() - perf_start
            
            assert perf_time < 15.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Stream Processor Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Stream Processor Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_monitoring_agent_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of real-time monitoring agent."""
        logger.info("🧪 Testing Real-Time Monitoring Agent (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_monitoring": False,
                "alert_management": False,
                "dashboard_operations": False,
                "stream_processing": False,
                "performance": False
            }
            
            # Test 1: Basic monitoring
            logger.info("   Testing basic monitoring...")
            request = AnalysisRequest(
                content="Monitor system performance in real-time",
                data_type=DataType.TEXT,
                analysis_type="real_time_monitoring",
                language="en"
            )
            
            result = await self.monitoring_agent.analyze(request)
            
            assert isinstance(result, AnalysisResult), "Should return AnalysisResult"
            assert result.success, "Analysis should be successful"
            test_results["basic_monitoring"] = True
            
            # Test 2: Alert management
            logger.info("   Testing alert management...")
            alert_request = AnalysisRequest(
                content="Set up alerts for high CPU usage and memory consumption",
                data_type=DataType.TEXT,
                analysis_type="real_time_monitoring",
                language="en"
            )
            
            result = await self.monitoring_agent.analyze(alert_request)
            assert result.success, "Alert management should be successful"
            test_results["alert_management"] = True
            
            # Test 3: Dashboard operations
            logger.info("   Testing dashboard operations...")
            dashboard_request = AnalysisRequest(
                content="Create performance dashboard with CPU, memory, and network metrics",
                data_type=DataType.TEXT,
                analysis_type="real_time_monitoring",
                language="en"
            )
            
            result = await self.monitoring_agent.analyze(dashboard_request)
            assert result.success, "Dashboard operations should be successful"
            test_results["dashboard_operations"] = True
            
            # Test 4: Stream processing
            logger.info("   Testing stream processing...")
            stream_request = AnalysisRequest(
                content="Process real-time data stream for anomaly detection",
                data_type=DataType.TEXT,
                analysis_type="real_time_monitoring",
                language="en"
            )
            
            result = await self.monitoring_agent.analyze(stream_request)
            assert result.success, "Stream processing should be successful"
            test_results["stream_processing"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            large_request = AnalysisRequest(
                content="Comprehensive real-time monitoring with multiple data sources and complex alerting rules " * 30,
                data_type=DataType.TEXT,
                analysis_type="real_time_monitoring",
                language="en"
            )
            
            perf_start = time.time()
            result = await self.monitoring_agent.analyze(large_request)
            perf_time = time.time() - perf_start
            
            assert perf_time < 40.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Real-Time Monitoring Agent Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Real-Time Monitoring Agent Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_integration_end_to_end(self) -> Dict[str, Any]:
        """End-to-end integration testing."""
        logger.info("🧪 Testing End-to-End Integration...")
        start_time = time.time()
        
        try:
            test_results = {
                "orchestrator_integration": False,
                "workflow_completion": False,
                "data_flow": False,
                "real_time_processing": False,
                "system_stability": False
            }
            
            # Test 1: Orchestrator integration
            logger.info("   Testing orchestrator integration...")
            request = AnalysisRequest(
                content="Comprehensive real-time monitoring setup",
                data_type=DataType.TEXT,
                analysis_type="real_time_monitoring",
                language="en"
            )
            
            result = await self.orchestrator.analyze(request)
            assert result.success, "Orchestrator should handle real-time monitoring"
            test_results["orchestrator_integration"] = True
            
            # Test 2: Workflow completion
            logger.info("   Testing workflow completion...")
            # Test complete workflow: Data stream -> Processing -> Pattern detection -> Alerting -> Dashboard
            data_stream = self.generate_test_data_stream(100)
            
            # Process stream
            processed_data = self.stream_processor.process_stream(data_stream)
            
            # Detect patterns
            patterns = self.pattern_monitor.detect_patterns(processed_data)
            
            # Create alerts
            alert = self.alert_system.create_alert({
                "metric_name": "test_metric",
                "threshold": 60,
                "condition": "greater_than",
                "severity": "warning"
            })
            
            # Create dashboard
            dashboard = self.performance_dashboard.create_dashboard({
                "name": "Integration Test Dashboard",
                "metrics": ["test_metric"],
                "layout": "simple"
            })
            
            assert all([processed_data, patterns, alert, dashboard]), "Complete workflow should work"
            test_results["workflow_completion"] = True
            
            # Test 3: Data flow
            logger.info("   Testing data flow...")
            # Verify data flows correctly through all components
            test_data = [{"metric_name": "flow_test", "value": 75, "timestamp": datetime.now().isoformat()}]
            
            # Flow: Stream -> Process -> Monitor -> Alert
            processed = self.stream_processor.process_stream(test_data)
            monitored = self.pattern_monitor.process_real_time_data(processed)
            alerted = self.alert_system.create_alert({
                "metric_name": "flow_test",
                "threshold": 70,
                "condition": "greater_than"
            })
            
            assert all([processed, monitored, alerted]), "Data flow should work end-to-end"
            test_results["data_flow"] = True
            
            # Test 4: Real-time processing
            logger.info("   Testing real-time processing...")
            # Test continuous real-time processing
            real_time_config = {
                "update_frequency": 1,
                "processing_mode": "continuous",
                "auto_refresh": True
            }
            
            real_time_result = self.performance_dashboard.enable_real_time_updates(
                dashboard=dashboard,
                config=real_time_config
            )
            
            assert real_time_result.status == "active", "Real-time processing should be active"
            test_results["real_time_processing"] = True
            
            # Test 5: System stability
            logger.info("   Testing system stability...")
            # Run multiple operations to test stability
            operations = []
            for i in range(10):
                op_request = AnalysisRequest(
                    content=f"Stability test real-time monitoring {i}",
                    data_type=DataType.TEXT,
                    analysis_type="real_time_monitoring",
                    language="en"
                )
                operations.append(self.monitoring_agent.analyze(op_request))
            
            # Execute all operations
            op_results = await asyncio.gather(*operations, return_exceptions=True)
            successful_ops = sum(1 for r in op_results if isinstance(r, AnalysisResult) and r.success)
            
            assert successful_ops >= 8, f"System stability test failed: {successful_ops}/10 successful"
            test_results["system_stability"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "End-to-End Integration Testing",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "successful_operations": successful_ops}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "End-to-End Integration Testing",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        logger.info("🚀 Starting Comprehensive Real-Time Monitoring Testing...")
        
        await self.setup()
        
        # Run all test categories
        test_categories = [
            ("Pattern Monitor", self.test_pattern_monitor_comprehensive),
            ("Alert System", self.test_alert_system_comprehensive),
            ("Performance Dashboard", self.test_performance_dashboard_comprehensive),
            ("Stream Processor", self.test_stream_processor_comprehensive),
            ("Real-Time Monitoring Agent", self.test_monitoring_agent_comprehensive),
            ("End-to-End Integration", self.test_integration_end_to_end)
        ]
        
        category_results = {}
        
        for category_name, test_func in test_categories:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing Category: {category_name}")
            logger.info(f"{'='*60}")
            
            try:
                result = await test_func()
                category_results[category_name] = result
            except Exception as e:
                logger.error(f"Error in {category_name}: {str(e)}")
                category_results[category_name] = {"status": "ERROR", "error": str(e)}
        
        # Calculate overall results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["status"] == "PASSED")
        failed_tests = sum(1 for r in self.test_results if r["status"] == "FAILED")
        
        overall_duration = time.time() - self.start_time
        
        # Generate comprehensive report
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_duration": overall_duration
            },
            "category_results": category_results,
            "detailed_results": self.test_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        with open("Test/comprehensive_real_time_monitoring_results.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("COMPREHENSIVE REAL-TIME MONITORING TESTING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        logger.info(f"Total Duration: {overall_duration:.2f}s")
        logger.info(f"Results saved to: Test/comprehensive_real_time_monitoring_results.json")
        
        return report


async def main():
    """Main test execution function."""
    tester = ComprehensiveRealTimeMonitoringTester()
    results = await tester.run_all_tests()
    return results


if __name__ == "__main__":
    # Run the comprehensive test suite
    results = asyncio.run(main())
    print(f"\nTest execution completed. Success rate: {results['test_summary']['success_rate']*100:.1f}%")
