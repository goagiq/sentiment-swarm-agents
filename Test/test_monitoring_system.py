"""
Test Monitoring and Observability System

This script tests the comprehensive monitoring system implementation for Task 7.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# Import monitoring systems
from src.core.monitoring.application_monitor import application_monitor
from src.core.monitoring.infrastructure_monitor import infrastructure_monitor
from src.core.monitoring.business_metrics import business_metrics_monitor
from src.core.monitoring.alert_system import AlertSystem
from src.core.monitoring.decision_monitor import DecisionMonitor


class MonitoringSystemTester:
    """Test class for the monitoring and observability system."""

    def __init__(self):
        self.test_results = []
        self.alert_system = AlertSystem()
        self.decision_monitor = DecisionMonitor()

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all monitoring system tests."""
        print("Starting comprehensive monitoring system tests...")
        
        # Test individual monitoring systems
        await self.test_application_monitoring()
        await self.test_infrastructure_monitoring()
        await self.test_business_metrics_monitoring()
        await self.test_alert_system()
        await self.test_decision_monitoring()
        await self.test_monitoring_integration()
        
        # Generate test report
        report = self.generate_test_report()
        recommendations = self.generate_recommendations()
        
        return {
            "test_results": self.test_results,
            "summary": report,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }

    async def test_application_monitoring(self):
        """Test application performance monitoring."""
        print("Testing Application Performance Monitoring...")
        
        try:
            # Test monitoring start/stop
            application_monitor.start_monitoring()
            self.record_test_result("app_monitor_start", True, "Application monitoring started successfully")
            
            # Test error recording
            application_monitor.record_error(
                error_type="test_error",
                error_message="Test error message",
                stack_trace="Test stack trace",
                user_id="test_user",
                severity="error"
            )
            self.record_test_result("app_error_recording", True, "Error recording works correctly")
            
            # Test user action recording
            application_monitor.record_user_action(
                user_id="test_user",
                action="test_action",
                session_id="test_session",
                page_url="/test",
                user_agent="test_agent",
                ip_address="127.0.0.1"
            )
            self.record_test_result("app_user_action_recording", True, "User action recording works correctly")
            
            # Test alert rule creation
            from src.core.monitoring.application_monitor import AlertRule
            test_rule = AlertRule(
                rule_id="test_rule",
                rule_name="Test CPU Alert",
                metric_name="cpu_usage",
                condition=">",
                threshold=90.0,
                duration=60,
                severity="high",
                notification_channels=["log"]
            )
            application_monitor.add_alert_rule(test_rule)
            self.record_test_result("app_alert_rule_creation", True, "Alert rule creation works correctly")
            
            # Test performance summary
            summary = await application_monitor.get_performance_summary()
            if "error" not in summary:
                self.record_test_result("app_performance_summary", True, "Performance summary generated correctly")
            else:
                self.record_test_result("app_performance_summary", False, f"Performance summary failed: {summary['error']}")
            
            # Test error analysis
            error_analysis = await application_monitor.get_error_analysis()
            if "error" not in error_analysis:
                self.record_test_result("app_error_analysis", True, "Error analysis generated correctly")
            else:
                self.record_test_result("app_error_analysis", False, f"Error analysis failed: {error_analysis['error']}")
            
            # Test user analytics
            analytics = await application_monitor.get_user_analytics()
            if "error" not in analytics:
                self.record_test_result("app_user_analytics", True, "User analytics generated correctly")
            else:
                self.record_test_result("app_user_analytics", False, f"User analytics failed: {analytics['error']}")
            
            application_monitor.stop_monitoring()
            self.record_test_result("app_monitor_stop", True, "Application monitoring stopped successfully")
            
        except Exception as e:
            self.record_test_result("app_monitoring_overall", False, f"Application monitoring test failed: {str(e)}")

    async def test_infrastructure_monitoring(self):
        """Test infrastructure monitoring."""
        print("Testing Infrastructure Monitoring...")
        
        try:
            # Test monitoring start/stop
            infrastructure_monitor.start_monitoring()
            self.record_test_result("infra_monitor_start", True, "Infrastructure monitoring started successfully")
            
            # Test server monitoring
            server_status = await infrastructure_monitor.get_server_status("localhost")
            if "error" not in server_status:
                self.record_test_result("infra_server_monitoring", True, "Server monitoring works correctly")
            else:
                self.record_test_result("infra_server_monitoring", False, f"Server monitoring failed: {server_status['error']}")
            
            # Test database monitoring
            db_status = await infrastructure_monitor.get_database_status("default")
            if "error" not in db_status:
                self.record_test_result("infra_db_monitoring", True, "Database monitoring works correctly")
            else:
                self.record_test_result("infra_db_monitoring", False, f"Database monitoring failed: {db_status['error']}")
            
            # Test network monitoring
            network_status = await infrastructure_monitor.get_network_status()
            if "targets" in network_status:
                self.record_test_result("infra_network_monitoring", True, "Network monitoring works correctly")
            else:
                self.record_test_result("infra_network_monitoring", False, "Network monitoring failed")
            
            # Test infrastructure summary
            summary = await infrastructure_monitor.get_infrastructure_summary()
            if "servers" in summary and "databases" in summary and "network" in summary:
                self.record_test_result("infra_summary", True, "Infrastructure summary generated correctly")
            else:
                self.record_test_result("infra_summary", False, "Infrastructure summary failed")
            
            infrastructure_monitor.stop_monitoring()
            self.record_test_result("infra_monitor_stop", True, "Infrastructure monitoring stopped successfully")
            
        except Exception as e:
            self.record_test_result("infra_monitoring_overall", False, f"Infrastructure monitoring test failed: {str(e)}")

    async def test_business_metrics_monitoring(self):
        """Test business metrics monitoring."""
        print("Testing Business Metrics Monitoring...")
        
        try:
            # Test monitoring start/stop
            business_metrics_monitor.start_monitoring()
            self.record_test_result("business_monitor_start", True, "Business metrics monitoring started successfully")
            
            # Test decision accuracy recording
            business_metrics_monitor.record_decision_accuracy(
                decision_id="test_decision_1",
                decision_type="sentiment_analysis",
                predicted_outcome={"sentiment": "positive", "confidence": 0.85},
                actual_outcome={"sentiment": "positive", "confidence": 0.90},
                accuracy_score=0.95,
                confidence_score=0.85,
                user_id="test_user"
            )
            self.record_test_result("business_decision_recording", True, "Decision accuracy recording works correctly")
            
            # Test user engagement recording
            business_metrics_monitor.record_user_engagement(
                user_id="test_user",
                session_id="test_session",
                action_type="analysis",
                duration=120.5,
                page_url="/analysis",
                feature_used="sentiment_analysis",
                success=True
            )
            self.record_test_result("business_engagement_recording", True, "User engagement recording works correctly")
            
            # Test feature usage recording
            business_metrics_monitor.record_feature_usage(
                feature_name="sentiment_analysis",
                user_id="test_user",
                usage_count=5,
                session_id="test_session",
                success_rate=0.9,
                average_duration=45.2
            )
            self.record_test_result("business_feature_usage_recording", True, "Feature usage recording works correctly")
            
            # Test business summary
            summary = await business_metrics_monitor.get_business_summary()
            if "decision_accuracy" in summary and "user_engagement" in summary:
                self.record_test_result("business_summary", True, "Business summary generated correctly")
            else:
                self.record_test_result("business_summary", False, "Business summary failed")
            
            # Test decision accuracy report
            decision_report = await business_metrics_monitor.get_decision_accuracy_report(hours=24)
            if "error" not in decision_report:
                self.record_test_result("business_decision_report", True, "Decision accuracy report generated correctly")
            else:
                self.record_test_result("business_decision_report", False, f"Decision report failed: {decision_report['error']}")
            
            # Test user engagement report
            engagement_report = await business_metrics_monitor.get_user_engagement_report(hours=24)
            if "error" not in engagement_report:
                self.record_test_result("business_engagement_report", True, "User engagement report generated correctly")
            else:
                self.record_test_result("business_engagement_report", False, f"Engagement report failed: {engagement_report['error']}")
            
            business_metrics_monitor.stop_monitoring()
            self.record_test_result("business_monitor_stop", True, "Business metrics monitoring stopped successfully")
            
        except Exception as e:
            self.record_test_result("business_monitoring_overall", False, f"Business metrics monitoring test failed: {str(e)}")

    async def test_alert_system(self):
        """Test alert system functionality."""
        print("Testing Alert System...")
        
        try:
            # Test alert creation
            alert_id = self.alert_system.create_alert(
                alert_type="test_alert",
                severity="high",
                message="Test alert message",
                source="test_source"
            )
            self.record_test_result("alert_creation", True, "Alert creation works correctly")
            
            # Test alert retrieval
            alerts = self.alert_system.get_active_alerts()
            if alerts:
                self.record_test_result("alert_retrieval", True, "Alert retrieval works correctly")
            else:
                self.record_test_result("alert_retrieval", False, "Alert retrieval failed")
            
            # Test alert acknowledgment
            self.alert_system.acknowledge_alert(alert_id)
            self.record_test_result("alert_acknowledgment", True, "Alert acknowledgment works correctly")
            
            # Test alert resolution
            self.alert_system.resolve_alert(alert_id)
            self.record_test_result("alert_resolution", True, "Alert resolution works correctly")
            
        except Exception as e:
            self.record_test_result("alert_system_overall", False, f"Alert system test failed: {str(e)}")

    async def test_decision_monitoring(self):
        """Test decision monitoring functionality."""
        print("Testing Decision Monitoring...")
        
        try:
            # Test decision recording
            decision_id = self.decision_monitor.record_decision(
                decision_type="sentiment_analysis",
                input_data={"text": "Test text"},
                decision_output={"sentiment": "positive", "confidence": 0.85},
                user_id="test_user",
                metadata={"source": "test"}
            )
            self.record_test_result("decision_recording", True, "Decision recording works correctly")
            
            # Test decision outcome recording
            self.decision_monitor.record_decision_outcome(
                decision_id=decision_id,
                actual_outcome={"sentiment": "positive"},
                success=True,
                feedback={"rating": 5}
            )
            self.record_test_result("decision_outcome_recording", True, "Decision outcome recording works correctly")
            
            # Test decision analysis
            analysis = self.decision_monitor.analyze_decisions(hours=24)
            if analysis:
                self.record_test_result("decision_analysis", True, "Decision analysis works correctly")
            else:
                self.record_test_result("decision_analysis", False, "Decision analysis failed")
            
        except Exception as e:
            self.record_test_result("decision_monitoring_overall", False, f"Decision monitoring test failed: {str(e)}")

    async def test_monitoring_integration(self):
        """Test integration between monitoring systems."""
        print("Testing Monitoring System Integration...")
        
        try:
            # Test concurrent monitoring
            application_monitor.start_monitoring()
            infrastructure_monitor.start_monitoring()
            business_metrics_monitor.start_monitoring()
            
            # Wait for some data collection
            await asyncio.sleep(2)
            
            # Test that all systems are collecting data
            app_metrics_count = len(application_monitor.performance_metrics)
            infra_metrics_count = len(infrastructure_monitor.server_metrics)
            business_metrics_count = len(business_metrics_monitor.decision_accuracy_metrics)
            
            if app_metrics_count > 0:
                self.record_test_result("integration_app_data_collection", True, "Application data collection working")
            else:
                self.record_test_result("integration_app_data_collection", False, "Application data collection failed")
            
            if infra_metrics_count > 0:
                self.record_test_result("integration_infra_data_collection", True, "Infrastructure data collection working")
            else:
                self.record_test_result("integration_infra_data_collection", False, "Infrastructure data collection failed")
            
            # Test alert integration
            # Create a test alert in each system
            application_monitor.record_error("integration_test", "Integration test error")
            business_metrics_monitor.record_decision_accuracy(
                "test_decision", "test_type", {}, {}, 0.5, 0.5
            )
            
            # Check that alerts are generated
            app_alerts = len(application_monitor.active_alerts)
            business_alerts = len(business_metrics_monitor.business_alerts)
            
            if app_alerts > 0 or business_alerts > 0:
                self.record_test_result("integration_alert_generation", True, "Alert integration working")
            else:
                self.record_test_result("integration_alert_generation", False, "Alert integration failed")
            
            # Stop all monitoring
            application_monitor.stop_monitoring()
            infrastructure_monitor.stop_monitoring()
            business_metrics_monitor.stop_monitoring()
            
            self.record_test_result("integration_concurrent_monitoring", True, "Concurrent monitoring works correctly")
            
        except Exception as e:
            self.record_test_result("integration_overall", False, f"Integration test failed: {str(e)}")

    def record_test_result(self, test_name: str, success: bool, message: str):
        """Record a test result."""
        result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "PASS" if success else "FAIL"
        print(f"  {status}: {test_name} - {message}")

    def generate_test_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report."""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["success"]])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Group results by category
        categories = {
            "application_monitoring": [r for r in self.test_results if r["test_name"].startswith("app")],
            "infrastructure_monitoring": [r for r in self.test_results if r["test_name"].startswith("infra")],
            "business_metrics": [r for r in self.test_results if r["test_name"].startswith("business")],
            "alert_system": [r for r in self.test_results if r["test_name"].startswith("alert")],
            "decision_monitoring": [r for r in self.test_results if r["test_name"].startswith("decision")],
            "integration": [r for r in self.test_results if r["test_name"].startswith("integration")]
        }
        
        category_summary = {}
        for category, results in categories.items():
            if results:
                category_passed = len([r for r in results if r["success"]])
                category_total = len(results)
                category_success_rate = (category_passed / category_total * 100) if category_total > 0 else 0
                category_summary[category] = {
                    "passed": category_passed,
                    "total": category_total,
                    "success_rate": category_success_rate
                }
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "category_summary": category_summary,
            "overall_status": "PASS" if success_rate >= 95 else "FAIL"
        }

    def generate_recommendations(self) -> list:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check success rate
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["success"]])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        if success_rate < 95:
            recommendations.append({
                "priority": "high",
                "category": "testing",
                "recommendation": f"Improve test success rate from {success_rate:.1f}% to 95%+",
                "details": "Some monitoring components are not working as expected"
            })
        
        # Check for specific failures
        failed_tests = [r for r in self.test_results if not r["success"]]
        for test in failed_tests:
            recommendations.append({
                "priority": "medium",
                "category": "functionality",
                "recommendation": f"Fix {test['test_name']}",
                "details": test['message']
            })
        
        # General recommendations
        recommendations.extend([
            {
                "priority": "low",
                "category": "performance",
                "recommendation": "Consider implementing metrics aggregation for better performance",
                "details": "Current implementation stores all metrics in memory"
            },
            {
                "priority": "low",
                "category": "scalability",
                "recommendation": "Implement data persistence for long-term monitoring",
                "details": "Current implementation only keeps recent data in memory"
            },
            {
                "priority": "low",
                "category": "security",
                "recommendation": "Add authentication and authorization to monitoring endpoints",
                "details": "Monitoring data may contain sensitive information"
            }
        ])
        
        return recommendations


async def main():
    """Main test execution function."""
    print("=" * 60)
    print("MONITORING AND OBSERVABILITY SYSTEM TEST")
    print("=" * 60)
    
    tester = MonitoringSystemTester()
    results = await tester.run_all_tests()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    summary = results["summary"]
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Overall Status: {summary['overall_status']}")
    
    # Print category summary
    print("\nCategory Summary:")
    for category, stats in summary["category_summary"].items():
        print(f"  {category}: {stats['passed']}/{stats['total']} ({stats['success_rate']:.1f}%)")
    
    # Print recommendations
    print("\nRecommendations:")
    for rec in results["recommendations"]:
        print(f"  [{rec['priority'].upper()}] {rec['category']}: {rec['recommendation']}")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"monitoring_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {filename}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
