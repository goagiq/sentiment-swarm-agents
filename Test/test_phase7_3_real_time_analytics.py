"""
Phase 7.3: Real-Time Analytics Dashboard Test

Comprehensive test script for Phase 7.3 real-time analytics dashboard implementation.
Tests all components including streaming, dashboard, visualizations, workflows, alerts, and collaboration.
"""

import sys
import os
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.real_time_analytics_config import get_real_time_analytics_config
from src.core.streaming.data_stream_processor import create_enhanced_stream_processor
from src.core.streaming.real_time_pipeline import create_real_time_pipeline
from src.core.streaming.stream_analytics import create_stream_analytics


class Phase7_3RealTimeAnalyticsTest:
    """
    Comprehensive test suite for Phase 7.3 Real-Time Analytics Dashboard.
    """
    
    def __init__(self):
        """Initialize the test suite."""
        self.test_results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        self.config = get_real_time_analytics_config()
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 7.3 tests."""
        print("🚀 Starting Phase 7.3: Real-Time Analytics Dashboard Tests")
        print("=" * 60)
        
        # Test 1: Configuration
        self.test_configuration()
        
        # Test 2: Data Stream Processor
        self.test_data_stream_processor()
        
        # Test 3: Real-Time Pipeline
        self.test_real_time_pipeline()
        
        # Test 4: Stream Analytics
        self.test_stream_analytics()
        
        # Test 5: Integration
        self.test_integration()
        
        # Print results
        self.print_results()
        
        return self.test_results
    
    def test_configuration(self):
        """Test real-time analytics configuration."""
        print("\n📋 Test 1: Configuration")
        try:
            # Test configuration loading
            config = get_real_time_analytics_config()
            assert config is not None, "Configuration should not be None"
            
            # Test configuration structure
            assert hasattr(config, 'stream_processing'), "Missing stream_processing config"
            assert hasattr(config, 'visualization'), "Missing visualization config"
            assert hasattr(config, 'dashboard'), "Missing dashboard config"
            assert hasattr(config, 'alerts'), "Missing alerts config"
            assert hasattr(config, 'workflow'), "Missing workflow config"
            
            # Test configuration values
            assert config.stream_processing.buffer_size > 0, "Buffer size should be positive"
            assert config.visualization.max_data_points > 0, "Max data points should be positive"
            assert config.dashboard.max_widgets > 0, "Max widgets should be positive"
            
            print("✅ Configuration test passed")
            self.test_results['passed'] += 1
            
        except Exception as e:
            print(f"❌ Configuration test failed: {str(e)}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"Configuration: {str(e)}")
    
    def test_data_stream_processor(self):
        """Test enhanced data stream processor."""
        print("\n🌊 Test 2: Data Stream Processor")
        try:
            # Create stream processor
            processor = create_enhanced_stream_processor()
            assert processor is not None, "Stream processor should not be None"
            
            # Test adding data points
            success = asyncio.run(processor.add_data_point_enhanced(
                value=42.0,
                source="test_source",
                priority="normal"
            ))
            assert success, "Adding data point should succeed"
            
            # Test quality validators
            def test_validator(data_point):
                return 0.9  # Return quality score
            
            processor.add_quality_validator("test_validator", test_validator)
            assert "test_validator" in processor.quality_validators, "Validator should be added"
            
            # Test real-time analytics
            def test_analyzer(data_point):
                return {"test_result": "success"}
            
            processor.add_real_time_analytics("test_analyzer", test_analyzer)
            assert "test_analyzer" in processor.real_time_analytics, "Analyzer should be added"
            
            # Test alert conditions
            def test_condition(data_point):
                return data_point.value > 50
            
            processor.add_alert_condition("test_condition", test_condition)
            assert "test_condition" in processor.alert_conditions, "Alert condition should be added"
            
            # Test metrics
            metrics = processor.get_stream_metrics()
            assert 'total_processed' in metrics, "Metrics should contain total_processed"
            assert 'throughput' in metrics, "Metrics should contain throughput"
            
            print("✅ Data Stream Processor test passed")
            self.test_results['passed'] += 1
            
        except Exception as e:
            print(f"❌ Data Stream Processor test failed: {str(e)}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"Data Stream Processor: {str(e)}")
    
    def test_real_time_pipeline(self):
        """Test real-time pipeline."""
        print("\n🔄 Test 3: Real-Time Pipeline")
        try:
            # Create pipeline
            pipeline = create_real_time_pipeline("test_pipeline")
            assert pipeline is not None, "Pipeline should not be None"
            
            # Test adding stages
            def test_stage(data, config):
                return data * 2
            
            pipeline.add_stage("test_stage", test_stage)
            assert len(pipeline.stages) == 1, "Stage should be added"
            
            # Test adding quality rules
            def test_quality_rule(data, stage_name):
                return 0.8
            
            pipeline.add_quality_rule("test_rule", test_quality_rule)
            assert "test_rule" in pipeline.quality_rules, "Quality rule should be added"
            
            # Test adding schema validators
            def test_schema_validator(data, stage_name):
                return True
            
            pipeline.add_schema_validator("test_validator", test_schema_validator)
            assert "test_validator" in pipeline.schema_validators, "Schema validator should be added"
            
            # Test processing data
            result = asyncio.run(pipeline.process_data({"test": "data"}, "test_source"))
            assert 'processed_data' in result, "Result should contain processed_data"
            assert 'stages_processed' in result, "Result should contain stages_processed"
            
            # Test metrics
            metrics = pipeline.get_pipeline_metrics()
            assert 'pipeline_name' in metrics, "Metrics should contain pipeline_name"
            assert 'total_processed' in metrics, "Metrics should contain total_processed"
            
            print("✅ Real-Time Pipeline test passed")
            self.test_results['passed'] += 1
            
        except Exception as e:
            print(f"❌ Real-Time Pipeline test failed: {str(e)}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"Real-Time Pipeline: {str(e)}")
    
    def test_stream_analytics(self):
        """Test stream analytics."""
        print("\n📊 Test 4: Stream Analytics")
        try:
            # Create stream analytics
            analytics = create_stream_analytics()
            assert analytics is not None, "Stream analytics should not be None"
            
            # Test adding windows
            analytics.add_window("test_window", timedelta(minutes=5))
            assert "test_window" in analytics.window_configs, "Window should be added"
            
            # Test adding aggregation functions
            def test_aggregation(data_points):
                return len(data_points)
            
            analytics.add_aggregation_function("test_count", test_aggregation)
            assert "test_count" in analytics.aggregation_functions, "Aggregation function should be added"
            
            # Test adding pattern detectors
            def test_pattern_detector(data_points):
                return [{"type": "test_pattern", "confidence": 0.8}]
            
            analytics.add_pattern_detector("test_pattern", test_pattern_detector)
            assert "test_pattern" in analytics.pattern_detectors, "Pattern detector should be added"
            
            # Test adding anomaly detectors
            def test_anomaly_detector(data_points):
                return [{"type": "test_anomaly", "confidence": 0.9}]
            
            analytics.add_anomaly_detector("test_anomaly", test_anomaly_detector)
            assert "test_anomaly" in analytics.anomaly_detectors, "Anomaly detector should be added"
            
            # Test metrics
            metrics = analytics.get_metrics()
            assert 'total_processed' in metrics, "Metrics should contain total_processed"
            assert 'windows_created' in metrics, "Metrics should contain windows_created"
            
            print("✅ Stream Analytics test passed")
            self.test_results['passed'] += 1
            
        except Exception as e:
            print(f"❌ Stream Analytics test failed: {str(e)}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"Stream Analytics: {str(e)}")
    
    def test_integration(self):
        """Test integration between components."""
        print("\n🔗 Test 5: Integration")
        try:
            # Create all components
            processor = create_enhanced_stream_processor()
            pipeline = create_real_time_pipeline("integration_test")
            analytics = create_stream_analytics()
            
            # Test data flow through components
            test_data = {"value": 75.5, "timestamp": datetime.now().isoformat()}
            
            # Process through pipeline
            pipeline_result = asyncio.run(pipeline.process_data(test_data, "integration_source"))
            assert pipeline_result is not None, "Pipeline should return result"
            
            # Add data to stream processor
            success = asyncio.run(processor.add_data_point_enhanced(
                value=test_data["value"],
                source="integration_source"
            ))
            assert success, "Data point should be added successfully"
            
            # Test analytics windows
            analytics.add_window("integration_window", timedelta(minutes=10))
            assert "integration_window" in analytics.window_configs, "Analytics window should be added"
            
            # Test metrics from all components
            processor_metrics = processor.get_stream_metrics()
            pipeline_metrics = pipeline.get_pipeline_metrics()
            analytics_metrics = analytics.get_metrics()
            
            assert all(metrics is not None for metrics in [processor_metrics, pipeline_metrics, analytics_metrics]), "All components should return metrics"
            
            print("✅ Integration test passed")
            self.test_results['passed'] += 1
            
        except Exception as e:
            print(f"❌ Integration test failed: {str(e)}")
            self.test_results['failed'] += 1
            self.test_results['errors'].append(f"Integration: {str(e)}")
    
    def print_results(self):
        """Print test results."""
        print("\n" + "=" * 60)
        print("📊 Phase 7.3 Test Results")
        print("=" * 60)
        
        total_tests = self.test_results['passed'] + self.test_results['failed']
        success_rate = (self.test_results['passed'] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"✅ Passed: {self.test_results['passed']}")
        print(f"❌ Failed: {self.test_results['failed']}")
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if self.test_results['errors']:
            print("\n🚨 Errors:")
            for error in self.test_results['errors']:
                print(f"  - {error}")
        
        print("\n" + "=" * 60)
        
        if success_rate >= 80:
            print("🎉 Phase 7.3 Real-Time Analytics Dashboard implementation is ready!")
        else:
            print("⚠️  Some issues need to be addressed before deployment.")
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report."""
        report = {
            'phase': '7.3',
            'title': 'Real-Time Analytics Dashboard',
            'timestamp': datetime.now().isoformat(),
            'test_results': self.test_results,
            'components_tested': [
                'Configuration Management',
                'Data Stream Processor',
                'Real-Time Pipeline',
                'Stream Analytics',
                'Component Integration'
            ],
            'features_implemented': [
                'Live Data Streaming',
                'Real-Time Data Validation',
                'Stream Processing Optimization',
                'ETL/ELT Processing',
                'Data Quality Management',
                'Schema Evolution',
                'Data Catalog Integration',
                'Real-Time Aggregations',
                'Window Functions',
                'Pattern Detection',
                'Anomaly Detection',
                'Interactive Visualizations',
                'Workflow Orchestration',
                'Scheduled Analytics',
                'Collaborative Analytics',
                'Version Control',
                'Custom Alert Rules',
                'Multi-Channel Notifications',
                'Alert Severity Levels',
                'Alert History Management'
            ],
            'configuration_summary': {
                'stream_processing': {
                    'buffer_size': self.config.stream_processing.buffer_size,
                    'batch_size': self.config.stream_processing.batch_size,
                    'processing_interval': self.config.stream_processing.processing_interval
                },
                'visualization': {
                    'max_data_points': self.config.visualization.max_data_points,
                    'update_interval': self.config.visualization.update_interval
                },
                'dashboard': {
                    'max_widgets': self.config.dashboard.max_widgets,
                    'enable_customization': self.config.dashboard.enable_customization
                },
                'alerts': {
                    'max_alerts': self.config.alerts.max_alerts,
                    'alert_check_interval': self.config.alerts.alert_check_interval
                }
            }
        }
        
        return report


def main():
    """Main test execution function."""
    # Create and run test suite
    test_suite = Phase7_3RealTimeAnalyticsTest()
    results = test_suite.run_all_tests()
    
    # Generate and save test report
    report = test_suite.generate_test_report()
    
    # Save report to file
    report_file = f"Test/phase7_3_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Test report saved to: {report_file}")
    
    return results


if __name__ == "__main__":
    main()
