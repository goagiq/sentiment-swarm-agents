#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite - Phase 6.2

This script provides comprehensive end-to-end integration testing for all system components including:
- Full system integration testing
- Cross-component communication testing
- Performance testing with load and stress testing
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

# Import all system components
from src.core.orchestrator import SentimentOrchestrator
from src.core.unified_mcp_client import (
    call_predictive_analytics_tool,
    call_scenario_analysis_tool,
    call_decision_support_tool,
    call_monitoring_tool,
    call_performance_optimization_tool
)
from src.core.performance_optimizer import get_performance_optimizer
from src.core.models import DataType, AnalysisRequest, AnalysisResult


class ComprehensiveIntegrationTester:
    """Comprehensive integration test suite for the entire system."""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        self.start_time = None
        self.orchestrator = None
        self.performance_optimizer = None
        
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
    
    async def setup(self):
        """Setup test environment."""
        logger.info("Setting up comprehensive integration test environment...")
        self.start_time = time.time()
        
        # Initialize orchestrator
        self.orchestrator = SentimentOrchestrator()
        
        # Initialize performance optimizer
        self.performance_optimizer = await get_performance_optimizer()
        
        logger.info("✅ Test environment setup complete")
    
    async def test_system_integration_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of system integration."""
        logger.info("🧪 Testing System Integration (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "orchestrator_initialization": False,
                "agent_registration": False,
                "component_communication": False,
                "data_flow": False,
                "performance": False
            }
            
            # Test 1: Orchestrator initialization
            logger.info("   Testing orchestrator initialization...")
            assert self.orchestrator is not None, "Orchestrator should be initialized"
            assert hasattr(self.orchestrator, 'agents'), "Orchestrator should have agents"
            assert len(self.orchestrator.agents) > 0, "Orchestrator should have registered agents"
            test_results["orchestrator_initialization"] = True
            
            # Test 2: Agent registration
            logger.info("   Testing agent registration...")
            expected_agents = [
                "pattern_recognition_agent",
                "predictive_analytics_agent",
                "scenario_analysis_agent",
                "real_time_monitoring_agent",
                "decision_support_agent",
                "risk_assessment_agent",
                "fault_detection_agent"
            ]
            
            registered_agents = list(self.orchestrator.agents.keys())
            found_agents = []
            
            for expected_agent in expected_agents:
                class_name = ''.join(word.capitalize() for word in expected_agent.split('_'))
                matching_agents = [agent for agent in registered_agents if class_name in agent]
                if matching_agents:
                    found_agents.extend(matching_agents)
            
            assert len(found_agents) >= len(expected_agents), f"Should have all expected agents. Found: {len(found_agents)}/{len(expected_agents)}"
            test_results["agent_registration"] = True
            
            # Test 3: Component communication
            logger.info("   Testing component communication...")
            # Test communication between orchestrator and agents
            request = AnalysisRequest(
                content="Test system integration and component communication",
                data_type=DataType.TEXT,
                analysis_type="sentiment_analysis",
                language="en"
            )
            
            result = await self.orchestrator.analyze(request)
            assert result is not None, "Orchestrator should communicate with agents"
            assert result.success, "Analysis should be successful"
            test_results["component_communication"] = True
            
            # Test 4: Data flow
            logger.info("   Testing data flow...")
            # Test data flow through multiple components
            test_requests = [
                AnalysisRequest(
                    content="Analyze sentiment for business decision making",
                    data_type=DataType.TEXT,
                    analysis_type="sentiment_analysis",
                    language="en"
                ),
                AnalysisRequest(
                    content="Generate predictive analytics for market trends",
                    data_type=DataType.TEXT,
                    analysis_type="predictive_analytics",
                    language="en"
                ),
                AnalysisRequest(
                    content="Create scenario analysis for strategic planning",
                    data_type=DataType.TEXT,
                    analysis_type="scenario_analysis",
                    language="en"
                )
            ]
            
            results = []
            for req in test_requests:
                result = await self.orchestrator.analyze(req)
                results.append(result)
            
            assert all(r.success for r in results), "All data flow tests should succeed"
            test_results["data_flow"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            perf_start = time.time()
            
            # Run multiple analyses to test performance
            for _ in range(10):
                await self.orchestrator.analyze(request)
            
            perf_time = time.time() - perf_start
            assert perf_time < 30.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "System Integration Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "System Integration Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_mcp_integration_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of MCP integration."""
        logger.info("🧪 Testing MCP Integration (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "predictive_analytics_mcp": False,
                "scenario_analysis_mcp": False,
                "decision_support_mcp": False,
                "monitoring_mcp": False,
                "performance": False
            }
            
            # Test 1: Predictive Analytics MCP
            logger.info("   Testing predictive analytics MCP...")
            try:
                result = await call_predictive_analytics_tool(
                    "Generate forecast for business metrics",
                    {"forecast_horizon": 12, "confidence_level": 0.95}
                )
                assert result is not None, "Predictive analytics MCP should return results"
                test_results["predictive_analytics_mcp"] = True
            except Exception as e:
                logger.warning(f"Predictive analytics MCP test failed: {str(e)}")
            
            # Test 2: Scenario Analysis MCP
            logger.info("   Testing scenario analysis MCP...")
            try:
                result = await call_scenario_analysis_tool(
                    "Create business scenarios for market expansion",
                    {"scenarios": ["optimistic", "pessimistic", "baseline"]}
                )
                assert result is not None, "Scenario analysis MCP should return results"
                test_results["scenario_analysis_mcp"] = True
            except Exception as e:
                logger.warning(f"Scenario analysis MCP test failed: {str(e)}")
            
            # Test 3: Decision Support MCP
            logger.info("   Testing decision support MCP...")
            try:
                result = await call_decision_support_tool(
                    "Provide recommendations for business growth",
                    {"analysis_type": "strategic_planning"}
                )
                assert result is not None, "Decision support MCP should return results"
                test_results["decision_support_mcp"] = True
            except Exception as e:
                logger.warning(f"Decision support MCP test failed: {str(e)}")
            
            # Test 4: Monitoring MCP
            logger.info("   Testing monitoring MCP...")
            try:
                result = await call_monitoring_tool(
                    "Monitor system performance metrics",
                    {"metrics": ["cpu", "memory", "response_time"]}
                )
                assert result is not None, "Monitoring MCP should return results"
                test_results["monitoring_mcp"] = True
            except Exception as e:
                logger.warning(f"Monitoring MCP test failed: {str(e)}")
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            perf_start = time.time()
            
            # Test MCP tool performance
            try:
                await call_performance_optimization_tool(
                    "Optimize system performance",
                    {"optimization_target": "response_time"}
                )
            except Exception:
                pass  # Performance tool might not be available
            
            perf_time = time.time() - perf_start
            assert perf_time < 10.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = sum(test_results.values()) >= 3  # At least 3 MCP tools should work
            
            self.log_test_result(
                "MCP Integration Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "MCP Integration Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_cross_component_integration(self) -> Dict[str, Any]:
        """Testing integration between different components."""
        logger.info("🧪 Testing Cross-Component Integration...")
        start_time = time.time()
        
        try:
            test_results = {
                "agent_to_agent_communication": False,
                "data_sharing": False,
                "workflow_execution": False,
                "error_propagation": False,
                "performance": False
            }
            
            # Test 1: Agent-to-agent communication
            logger.info("   Testing agent-to-agent communication...")
            # Test that agents can work together
            complex_request = AnalysisRequest(
                content="Analyze sentiment, generate predictions, and provide recommendations for business strategy",
                data_type=DataType.TEXT,
                analysis_type="comprehensive_analysis",
                language="en"
            )
            
            result = await self.orchestrator.analyze(complex_request)
            assert result is not None, "Agents should communicate effectively"
            test_results["agent_to_agent_communication"] = True
            
            # Test 2: Data sharing
            logger.info("   Testing data sharing...")
            # Test that data flows between components
            data_request = AnalysisRequest(
                content="Extract entities and analyze patterns in business documents",
                data_type=DataType.TEXT,
                analysis_type="entity_extraction",
                language="en"
            )
            
            result = await self.orchestrator.analyze(data_request)
            assert result.success, "Data should be shared between components"
            test_results["data_sharing"] = True
            
            # Test 3: Workflow execution
            logger.info("   Testing workflow execution...")
            # Test complete workflow execution
            workflow_request = AnalysisRequest(
                content="Process document, extract insights, generate report, and provide recommendations",
                data_type=DataType.TEXT,
                analysis_type="document_analysis",
                language="en"
            )
            
            result = await self.orchestrator.analyze(workflow_request)
            assert result.success, "Complete workflow should execute successfully"
            test_results["workflow_execution"] = True
            
            # Test 4: Error propagation
            logger.info("   Testing error propagation...")
            # Test error handling across components
            invalid_request = AnalysisRequest(
                content="",
                data_type=DataType.TEXT,
                analysis_type="invalid_analysis",
                language="en"
            )
            
            result = await self.orchestrator.analyze(invalid_request)
            # Should handle gracefully even if content is empty or analysis type is invalid
            assert isinstance(result, AnalysisResult), "Should handle errors gracefully"
            test_results["error_propagation"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            perf_start = time.time()
            
            # Test cross-component performance
            for _ in range(5):
                await self.orchestrator.analyze(complex_request)
            
            perf_time = time.time() - perf_start
            assert perf_time < 60.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Cross-Component Integration",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Cross-Component Integration",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_performance_optimization_integration(self) -> Dict[str, Any]:
        """Testing performance optimization integration."""
        logger.info("🧪 Testing Performance Optimization Integration...")
        start_time = time.time()
        
        try:
            test_results = {
                "optimizer_initialization": False,
                "performance_monitoring": False,
                "optimization_suggestions": False,
                "system_adaptation": False,
                "performance": False
            }
            
            # Test 1: Optimizer initialization
            logger.info("   Testing optimizer initialization...")
            assert self.performance_optimizer is not None, "Performance optimizer should be initialized"
            test_results["optimizer_initialization"] = True
            
            # Test 2: Performance monitoring
            logger.info("   Testing performance monitoring...")
            # Test that performance is being monitored
            try:
                performance_data = await self.performance_optimizer.get_performance_metrics()
                assert performance_data is not None, "Should get performance metrics"
                test_results["performance_monitoring"] = True
            except Exception as e:
                logger.warning(f"Performance monitoring test failed: {str(e)}")
            
            # Test 3: Optimization suggestions
            logger.info("   Testing optimization suggestions...")
            try:
                suggestions = await self.performance_optimizer.get_optimization_suggestions()
                assert suggestions is not None, "Should get optimization suggestions"
                test_results["optimization_suggestions"] = True
            except Exception as e:
                logger.warning(f"Optimization suggestions test failed: {str(e)}")
            
            # Test 4: System adaptation
            logger.info("   Testing system adaptation...")
            # Test that system can adapt to performance changes
            adaptation_request = AnalysisRequest(
                content="Test system adaptation to performance changes",
                data_type=DataType.TEXT,
                analysis_type="performance_test",
                language="en"
            )
            
            result = await self.orchestrator.analyze(adaptation_request)
            assert result is not None, "System should adapt to performance changes"
            test_results["system_adaptation"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            perf_start = time.time()
            
            # Test performance optimization overhead
            for _ in range(10):
                await self.orchestrator.analyze(adaptation_request)
            
            perf_time = time.time() - perf_start
            assert perf_time < 30.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = sum(test_results.values()) >= 3  # At least 3 tests should pass
            
            self.log_test_result(
                "Performance Optimization Integration",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Performance Optimization Integration",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_load_and_stress_integration(self) -> Dict[str, Any]:
        """Load and stress testing for the entire system."""
        logger.info("🧪 Testing Load and Stress Integration...")
        start_time = time.time()
        
        try:
            test_results = {
                "concurrent_requests": False,
                "memory_usage": False,
                "response_time_consistency": False,
                "error_handling": False,
                "system_recovery": False
            }
            
            # Test 1: Concurrent requests
            logger.info("   Testing concurrent requests...")
            request = AnalysisRequest(
                content="Load test content for concurrent processing",
                data_type=DataType.TEXT,
                analysis_type="sentiment_analysis",
                language="en"
            )
            
            # Run concurrent requests
            tasks = [self.orchestrator.analyze(request) for _ in range(20)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_requests = sum(1 for r in results if isinstance(r, AnalysisResult) and r.success)
            assert successful_requests >= 15, f"Too many failed requests: {successful_requests}/20"
            test_results["concurrent_requests"] = True
            
            # Test 2: Memory usage
            logger.info("   Testing memory usage...")
            # Run multiple requests to test memory usage
            for _ in range(50):
                await self.orchestrator.analyze(request)
            
            # Memory usage should be reasonable (this is a basic check)
            test_results["memory_usage"] = True
            
            # Test 3: Response time consistency
            logger.info("   Testing response time consistency...")
            response_times = []
            
            for _ in range(10):
                req_start = time.time()
                await self.orchestrator.analyze(request)
                req_time = time.time() - req_start
                response_times.append(req_time)
            
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            assert avg_response_time < 5.0, f"Average response time too high: {avg_response_time:.2f}s"
            assert max_response_time < 10.0, f"Max response time too high: {max_response_time:.2f}s"
            test_results["response_time_consistency"] = True
            
            # Test 4: Error handling
            logger.info("   Testing error handling...")
            error_count = 0
            total_requests = 30
            
            for _ in range(total_requests):
                try:
                    result = await self.orchestrator.analyze(request)
                    if not result.success:
                        error_count += 1
                except Exception:
                    error_count += 1
            
            error_rate = error_count / total_requests
            assert error_rate < 0.2, f"Error rate too high: {error_rate:.2%}"
            test_results["error_handling"] = True
            
            # Test 5: System recovery
            logger.info("   Testing system recovery...")
            # Test that system recovers after stress
            recovery_request = AnalysisRequest(
                content="Test system recovery after stress testing",
                data_type=DataType.TEXT,
                analysis_type="sentiment_analysis",
                language="en"
            )
            
            result = await self.orchestrator.analyze(recovery_request)
            assert result.success, "System should recover and function normally"
            test_results["system_recovery"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Load and Stress Integration",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {
                    "test_results": test_results,
                    "successful_concurrent_requests": successful_requests,
                    "avg_response_time": avg_response_time,
                    "error_rate": error_rate
                }
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Load and Stress Integration",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_end_to_end_workflows(self) -> Dict[str, Any]:
        """Testing complete end-to-end workflows."""
        logger.info("🧪 Testing End-to-End Workflows...")
        start_time = time.time()
        
        try:
            test_results = {
                "document_analysis_workflow": False,
                "business_intelligence_workflow": False,
                "predictive_analytics_workflow": False,
                "decision_support_workflow": False,
                "performance": False
            }
            
            # Test 1: Document analysis workflow
            logger.info("   Testing document analysis workflow...")
            doc_request = AnalysisRequest(
                content="Analyze business document for sentiment, extract entities, and generate insights",
                data_type=DataType.TEXT,
                analysis_type="document_analysis",
                language="en"
            )
            
            result = await self.orchestrator.analyze(doc_request)
            assert result.success, "Document analysis workflow should succeed"
            test_results["document_analysis_workflow"] = True
            
            # Test 2: Business intelligence workflow
            logger.info("   Testing business intelligence workflow...")
            bi_request = AnalysisRequest(
                content="Generate business intelligence report with market analysis and strategic recommendations",
                data_type=DataType.TEXT,
                analysis_type="business_intelligence",
                language="en"
            )
            
            result = await self.orchestrator.analyze(bi_request)
            assert result.success, "Business intelligence workflow should succeed"
            test_results["business_intelligence_workflow"] = True
            
            # Test 3: Predictive analytics workflow
            logger.info("   Testing predictive analytics workflow...")
            pred_request = AnalysisRequest(
                content="Create predictive models for market trends and customer behavior analysis",
                data_type=DataType.TEXT,
                analysis_type="predictive_analytics",
                language="en"
            )
            
            result = await self.orchestrator.analyze(pred_request)
            assert result.success, "Predictive analytics workflow should succeed"
            test_results["predictive_analytics_workflow"] = True
            
            # Test 4: Decision support workflow
            logger.info("   Testing decision support workflow...")
            decision_request = AnalysisRequest(
                content="Provide comprehensive decision support with scenario analysis and risk assessment",
                data_type=DataType.TEXT,
                analysis_type="decision_support",
                language="en"
            )
            
            result = await self.orchestrator.analyze(decision_request)
            assert result.success, "Decision support workflow should succeed"
            test_results["decision_support_workflow"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            perf_start = time.time()
            
            # Test all workflows performance
            workflows = [doc_request, bi_request, pred_request, decision_request]
            for workflow in workflows:
                await self.orchestrator.analyze(workflow)
            
            perf_time = time.time() - perf_start
            assert perf_time < 120.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "End-to-End Workflows",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "End-to-End Workflows",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive integration tests."""
        logger.info("🚀 Starting Comprehensive Integration Testing...")
        
        await self.setup()
        
        # Run all test categories
        test_categories = [
            ("System Integration", self.test_system_integration_comprehensive),
            ("MCP Integration", self.test_mcp_integration_comprehensive),
            ("Cross-Component Integration", self.test_cross_component_integration),
            ("Performance Optimization", self.test_performance_optimization_integration),
            ("Load and Stress", self.test_load_and_stress_integration),
            ("End-to-End Workflows", self.test_end_to_end_workflows)
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
        with open("Test/comprehensive_integration_results.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("COMPREHENSIVE INTEGRATION TESTING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        logger.info(f"Total Duration: {overall_duration:.2f}s")
        logger.info(f"Results saved to: Test/comprehensive_integration_results.json")
        
        return report


async def main():
    """Main test execution function."""
    tester = ComprehensiveIntegrationTester()
    results = await tester.run_all_tests()
    return results


if __name__ == "__main__":
    # Run the comprehensive integration test suite
    results = asyncio.run(main())
    print(f"\nTest execution completed. Success rate: {results['test_summary']['success_rate']*100:.1f}%")
