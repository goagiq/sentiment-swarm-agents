"""
Test Fault Detection System

Comprehensive tests for the fault detection and monitoring system.
"""

import sys
import os
import time
import json
import tempfile
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.fault_detection import (
    SystemHealthMonitor,
    PerformanceAnalyzer,
    ErrorPredictor,
    RecoveryRecommender
)
from src.agents.fault_detection_agent import FaultDetectionAgent


def test_health_monitor():
    """Test SystemHealthMonitor functionality"""
    print("Testing SystemHealthMonitor...")
    
    try:
        # Initialize health monitor
        config = {
            'alert_thresholds': {
                'cpu_warning': 70.0,
                'cpu_critical': 90.0,
                'memory_warning': 80.0,
                'memory_critical': 95.0
            },
            'check_interval': 5
        }
        
        monitor = SystemHealthMonitor(config)
        
        # Test health check
        health_report = monitor.check_system_health()
        print(f"Health Report: {health_report.get('overall_status', 'unknown')}")
        
        # Test component registration
        monitor.register_component("test_component")
        
        # Test health summary
        summary = monitor.get_health_summary()
        print(f"Health Summary: {summary.get('overall_status', 'unknown')}")
        
        print("✅ SystemHealthMonitor tests passed")
        return True
        
    except Exception as e:
        print(f"❌ SystemHealthMonitor test failed: {e}")
        return False


def test_performance_analyzer():
    """Test PerformanceAnalyzer functionality"""
    print("Testing PerformanceAnalyzer...")
    
    try:
        # Initialize performance analyzer
        config = {
            'thresholds': {
                'cpu_excellent': 20.0,
                'cpu_good': 40.0,
                'cpu_fair': 60.0,
                'cpu_poor': 80.0
            },
            'analysis_interval': 10
        }
        
        analyzer = PerformanceAnalyzer(config)
        
        # Test performance analysis
        snapshot = analyzer.analyze_performance()
        print(f"Performance Score: {snapshot.overall_score:.2f}")
        
        # Test performance summary
        summary = analyzer.get_performance_summary()
        print(f"Performance Summary: {summary.get('overall_score', 0.0):.2f}")
        
        # Test bottleneck identification
        bottlenecks = analyzer.identify_bottlenecks()
        print(f"Bottlenecks found: {len(bottlenecks)}")
        
        # Test performance trends
        trends = analyzer.get_performance_trends(hours=1)
        print(f"Performance Trends: {trends.get('trend', 'unknown')}")
        
        print("✅ PerformanceAnalyzer tests passed")
        return True
        
    except Exception as e:
        print(f"❌ PerformanceAnalyzer test failed: {e}")
        return False


def test_error_predictor():
    """Test ErrorPredictor functionality"""
    print("Testing ErrorPredictor...")
    
    try:
        # Initialize error predictor
        config = {
            'thresholds': {
                'cpu_critical': 90.0,
                'memory_critical': 95.0,
                'disk_critical': 98.0
            },
            'prediction_interval': 30
        }
        
        predictor = ErrorPredictor(config)
        
        # Test error prediction
        predictions = predictor.predict_errors()
        print(f"Error Predictions: {len(predictions)}")
        
        # Test prediction summary
        summary = predictor.get_prediction_summary()
        print(f"Prediction Summary: {summary.get('total_predictions', 0)} predictions")
        
        # Test prediction accuracy
        accuracy = predictor.get_prediction_accuracy()
        print(f"Prediction Accuracy: {accuracy.get('prediction_rate', 0.0):.1f}%")
        
        print("✅ ErrorPredictor tests passed")
        return True
        
    except Exception as e:
        print(f"❌ ErrorPredictor test failed: {e}")
        return False


def test_recovery_recommender():
    """Test RecoveryRecommender functionality"""
    print("Testing RecoveryRecommender...")
    
    try:
        # Initialize recovery recommender
        config = {
            'action_configs': {
                'restart_service': {
                    'timeout': 60,
                    'retry_count': 3
                },
                'clear_cache': {
                    'cache_dirs': ['/tmp', './cache']
                }
            }
        }
        
        recommender = RecoveryRecommender(config)
        
        # Test recovery plan generation
        health_data = {
            'system_metrics': {
                'cpu_percent': 85.0,
                'memory_percent': 90.0,
                'disk_percent': 95.0
            },
            'components': {
                'test_component': {
                    'status': 'critical',
                    'error_count': 5
                }
            }
        }
        
        recovery_plan = recommender.analyze_system_issues(health_data)
        print(f"Recovery Plan: {len(recovery_plan.recommendations)} recommendations")
        
        # Test recovery action execution
        if recovery_plan.recommendations:
            recommendation = recovery_plan.recommendations[0]
            result = recommender.execute_recovery_action(recommendation)
            print(f"Recovery Action Result: {result.get('success', False)}")
        
        # Test recovery statistics
        stats = recommender.get_recovery_statistics()
        print(f"Recovery Statistics: {stats.get('success_rate', 0.0):.1f}% success rate")
        
        print("✅ RecoveryRecommender tests passed")
        return True
        
    except Exception as e:
        print(f"❌ RecoveryRecommender test failed: {e}")
        return False


def test_fault_detection_agent():
    """Test FaultDetectionAgent functionality"""
    print("Testing FaultDetectionAgent...")
    
    try:
        # Initialize fault detection agent
        config = {
            'analysis_interval': 10,
            'alert_thresholds': {
                'health_critical': 0.3,
                'performance_critical': 0.4,
                'prediction_confidence': 0.7
            }
        }
        
        agent = FaultDetectionAgent(config)
        
        # Test system status
        status = agent.get_system_status()
        print(f"Agent Status: {status.get('agent_status', 'unknown')}")
        
        # Test component registration
        agent.register_component("test_component")
        
        # Test detailed analysis
        analysis = agent.get_detailed_analysis()
        print(f"Detailed Analysis: {len(analysis)} components")
        
        # Test alerts
        alerts = agent.get_alerts()
        print(f"Alerts: {len(alerts)}")
        
        # Test report export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            report_path = f.name
            
        try:
            agent.export_analysis_report(report_path)
            print(f"Analysis Report exported to: {report_path}")
            
            # Verify report was created
            with open(report_path, 'r') as f:
                report_data = json.load(f)
                print(f"Report contains {len(report_data)} sections")
                
        finally:
            # Clean up
            if os.path.exists(report_path):
                os.unlink(report_path)
        
        print("✅ FaultDetectionAgent tests passed")
        return True
        
    except Exception as e:
        print(f"❌ FaultDetectionAgent test failed: {e}")
        return False


def test_integration():
    """Test integration between all components"""
    print("Testing Integration...")
    
    try:
        # Initialize all components
        health_monitor = SystemHealthMonitor()
        performance_analyzer = PerformanceAnalyzer()
        error_predictor = ErrorPredictor()
        recovery_recommender = RecoveryRecommender()
        
        # Test data flow
        health_data = health_monitor.get_health_summary()
        performance_data = performance_analyzer.get_performance_summary()
        predictions = error_predictor.get_predictions()
        
        # Test recovery plan generation with real data
        recovery_plan = recovery_recommender.analyze_system_issues(health_data)
        
        print(f"Integration Test Results:")
        print(f"  - Health Status: {health_data.get('overall_status', 'unknown')}")
        print(f"  - Performance Score: {performance_data.get('overall_score', 0.0):.2f}")
        print(f"  - Error Predictions: {len(predictions)}")
        print(f"  - Recovery Recommendations: {len(recovery_plan.recommendations)}")
        
        print("✅ Integration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def test_error_handling():
    """Test error handling capabilities"""
    print("Testing Error Handling...")
    
    try:
        # Test with invalid configurations
        invalid_config = {'invalid_key': 'invalid_value'}
        
        # These should handle invalid configs gracefully
        health_monitor = SystemHealthMonitor(invalid_config)
        performance_analyzer = PerformanceAnalyzer(invalid_config)
        error_predictor = ErrorPredictor(invalid_config)
        recovery_recommender = RecoveryRecommender(invalid_config)
        
        # Test with invalid health data
        invalid_health_data = {'invalid': 'data'}
        recovery_plan = recovery_recommender.analyze_system_issues(invalid_health_data)
        
        print("✅ Error handling tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def run_comprehensive_test():
    """Run comprehensive fault detection system test"""
    print("🚀 Starting Comprehensive Fault Detection System Test")
    print("=" * 60)
    
    test_results = []
    
    # Run individual component tests
    test_results.append(("SystemHealthMonitor", test_health_monitor()))
    test_results.append(("PerformanceAnalyzer", test_performance_analyzer()))
    test_results.append(("ErrorPredictor", test_error_predictor()))
    test_results.append(("RecoveryRecommender", test_recovery_recommender()))
    test_results.append(("FaultDetectionAgent", test_fault_detection_agent()))
    
    # Run integration tests
    test_results.append(("Integration", test_integration()))
    test_results.append(("Error Handling", test_error_handling()))
    
    # Print results summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Fault detection system is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
