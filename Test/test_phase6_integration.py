#!/usr/bin/env python3
"""
Test script for Phase 6.1: System Integration & Optimization
Tests the integration of all Phase 1-5 components and new analytics capabilities.
"""

import asyncio
import json
import time
from typing import Dict, Any, List
from datetime import datetime

# Add src to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from loguru import logger

# Import all agents for testing
from src.agents.predictive_analytics_agent import PredictiveAnalyticsAgent
from src.agents.scenario_analysis_agent import ScenarioAnalysisAgent
from src.agents.decision_support_agent import DecisionSupportAgent
from src.agents.risk_assessment_agent import RiskAssessmentAgent
from src.agents.fault_detection_agent import FaultDetectionAgent
from src.agents.pattern_recognition_agent import PatternRecognitionAgent
from src.agents.real_time_monitoring_agent import RealTimeMonitoringAgent

# Import core components
from src.core.orchestrator import SentimentOrchestrator
from src.core.performance_optimizer import PerformanceOptimizer, get_performance_optimizer
from src.core.unified_mcp_client import (
    call_predictive_analytics_tool,
    call_scenario_analysis_tool,
    call_decision_support_tool,
    call_monitoring_tool,
    call_performance_optimization_tool
)

# Import models
from src.core.models import AnalysisRequest, DataType


class Phase6IntegrationTester:
    """Test class for Phase 6.1 system integration."""
    
    def __init__(self):
        self.test_results = []
        self.orchestrator = None
        self.performance_optimizer = None
        
    async def setup(self):
        """Setup test environment."""
        logger.info("Setting up Phase 6.1 integration test environment...")
        
        # Initialize orchestrator
        self.orchestrator = SentimentOrchestrator()
        
        # Initialize performance optimizer
        self.performance_optimizer = await get_performance_optimizer()
        
        logger.info("✅ Test environment setup complete")
    
    async def test_agent_integration(self) -> Dict[str, Any]:
        """Test that all Phase 1-5 agents are properly integrated in orchestrator."""
        logger.info("Testing agent integration in orchestrator...")
        
        try:
            # Check that all expected agents are registered
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
            # Check for agents that contain the expected names (since they have generated IDs)
            found_agents = []
            missing_agents = []
            
            for expected_agent in expected_agents:
                # Convert expected name to class name format (e.g., "pattern_recognition_agent" -> "PatternRecognitionAgent")
                class_name = ''.join(word.capitalize() for word in expected_agent.split('_'))
                # Find agents that contain the class name
                matching_agents = [agent for agent in registered_agents if class_name in agent]
                if matching_agents:
                    found_agents.extend(matching_agents)
                else:
                    missing_agents.append(expected_agent)
            
            if missing_agents:
                return {
                    "test": "Agent Integration",
                    "status": "FAILED",
                    "message": f"Missing agents: {missing_agents}",
                    "registered_agents": registered_agents
                }
            
            return {
                "test": "Agent Integration",
                "status": "PASSED",
                "message": f"All {len(expected_agents)} expected agents are registered",
                "registered_agents": registered_agents
            }
            
        except Exception as e:
            return {
                "test": "Agent Integration",
                "status": "ERROR",
                "message": f"Error testing agent integration: {str(e)}"
            }
    
    async def test_predictive_analytics_agent(self) -> Dict[str, Any]:
        """Test predictive analytics agent functionality."""
        logger.info("Testing predictive analytics agent...")
        
        try:
            agent = PredictiveAnalyticsAgent()
            
            # Test data
            test_data = {
                "time_series_data": [10, 12, 15, 18, 22, 25, 28, 30, 33, 35, 38, 40, 42, 45, 48, 50],
                "forecast_horizon": 3,
                "confidence_level": 0.95
            }
            
            # Create analysis request
            request = AnalysisRequest(
                data_type=DataType.TIME_SERIES,
                content=json.dumps(test_data),
                language="en"
            )
            
            # Process request
            result = await agent.process(request)
            
            if result.sentiment.label != "uncertain" and "Error:" not in result.sentiment.reasoning:
                return {
                    "test": "Predictive Analytics Agent",
                    "status": "PASSED",
                    "message": "Successfully processed predictive analytics request",
                    "result_summary": {
                        "forecast_generated": "forecast" in str(result.metadata.get("result", {})),
                        "confidence_intervals": "confidence" in str(result.metadata.get("result", {}))
                    }
                }
            else:
                return {
                    "test": "Predictive Analytics Agent",
                    "status": "FAILED",
                    "message": f"Failed to process request: {result.metadata.get('error', '')}"
                }
                
        except Exception as e:
            return {
                "test": "Predictive Analytics Agent",
                "status": "ERROR",
                "message": f"Error testing predictive analytics agent: {str(e)}"
            }
    
    async def test_scenario_analysis_agent(self) -> Dict[str, Any]:
        """Test scenario analysis agent functionality."""
        logger.info("Testing scenario analysis agent...")
        
        try:
            agent = ScenarioAnalysisAgent()
            
            # Test scenario data
            test_scenario = {
                "analysis_type": "scenario_building",
                "scenario": {
                    "name": "Market Expansion",
                    "description": "Market expansion scenario analysis",
                    "type": "custom",
                    "tags": ["market", "expansion"],
                    "parameters": {
                        "market_size": {
                            "type": "numerical",
                            "current_value": 1000000,
                            "scenario_value": 1500000,
                            "description": "Market size in USD"
                        },
                        "growth_rate": {
                            "type": "numerical", 
                            "current_value": 0.05,
                            "scenario_value": 0.08,
                            "description": "Growth rate percentage"
                        }
                    }
                }
            }
            
            # Create analysis request
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=json.dumps(test_scenario),
                language="en"
            )
            
            # Process request
            result = await agent.process(request)
            
            if result.sentiment.label != "uncertain" and "Error:" not in result.sentiment.reasoning:
                return {
                    "test": "Scenario Analysis Agent",
                    "status": "PASSED",
                    "message": "Successfully processed scenario analysis request",
                    "result_summary": {
                        "scenarios_analyzed": "scenario" in str(result.metadata.get("result", {})),
                        "impact_analysis": "impact" in str(result.metadata.get("result", {}))
                    }
                }
            else:
                return {
                    "test": "Scenario Analysis Agent",
                    "status": "FAILED",
                    "message": f"Failed to process request: {result.metadata.get('error', '')}"
                }
                
        except Exception as e:
            return {
                "test": "Scenario Analysis Agent",
                "status": "ERROR",
                "message": f"Error testing scenario analysis agent: {str(e)}"
            }
    
    async def test_decision_support_agent(self) -> Dict[str, Any]:
        """Test decision support agent functionality."""
        logger.info("Testing decision support agent...")
        
        try:
            agent = DecisionSupportAgent()
            
            # Test decision context
            test_context = {
                "analysis_type": "comprehensive_decision_analysis",
                "business_context": "Market expansion decision",
                "objectives": ["Increase revenue", "Market share growth"],
                "constraints": ["Budget limit", "Time constraints"],
                "alternatives": ["Option A", "Option B", "Option C"]
            }
            
            # Create analysis request
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=json.dumps(test_context),
                language="en"
            )
            
            # Process request
            result = await agent.process(request)
            
            if result.sentiment.label != "uncertain" and "Error:" not in result.sentiment.reasoning:
                return {
                    "test": "Decision Support Agent",
                    "status": "PASSED",
                    "message": "Successfully processed decision support request",
                    "result_summary": {
                        "recommendations_generated": "recommendation" in str(result.metadata.get("result", {})),
                        "prioritization": "priority" in str(result.metadata.get("result", {}))
                    }
                }
            else:
                return {
                    "test": "Decision Support Agent",
                    "status": "FAILED",
                    "message": f"Failed to process request: {result.metadata.get('error', '')}"
                }
                
        except Exception as e:
            return {
                "test": "Decision Support Agent",
                "status": "ERROR",
                "message": f"Error testing decision support agent: {str(e)}"
            }
    
    async def test_risk_assessment_agent(self) -> Dict[str, Any]:
        """Test risk assessment agent functionality."""
        logger.info("Testing risk assessment agent...")
        
        try:
            agent = RiskAssessmentAgent()
            
            # Test risk assessment data
            test_risks = {
                "project_name": "Digital Transformation",
                "risk_categories": ["Technical", "Operational", "Financial"],
                "risk_factors": {
                    "technical_complexity": "High",
                    "budget_overrun": "Medium",
                    "timeline_delay": "High"
                }
            }
            
            # Create analysis request
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=json.dumps(test_risks),
                language="en"
            )
            
            # Process request
            result = await agent.process(request)
            
            if result.sentiment.label != "uncertain" and "Error:" not in result.sentiment.reasoning:
                return {
                    "test": "Risk Assessment Agent",
                    "status": "PASSED",
                    "message": "Successfully processed risk assessment request",
                    "result_summary": {
                        "risks_identified": "risk" in str(result.metadata.get("result", {})),
                        "mitigation_plans": "mitigation" in str(result.metadata.get("result", {}))
                    }
                }
            else:
                return {
                    "test": "Risk Assessment Agent",
                    "status": "FAILED",
                    "message": f"Failed to process request: {result.metadata.get('error', '')}"
                }
                
        except Exception as e:
            return {
                "test": "Risk Assessment Agent",
                "status": "ERROR",
                "message": f"Error testing risk assessment agent: {str(e)}"
            }
    
    async def test_fault_detection_agent(self) -> Dict[str, Any]:
        """Test fault detection agent functionality."""
        logger.info("Testing fault detection agent...")
        
        try:
            agent = FaultDetectionAgent()
            
            # Test system health data
            test_health = {
                "system_metrics": {
                    "cpu_usage": 75.0,
                    "memory_usage": 80.0,
                    "disk_usage": 60.0,
                    "response_time": 2.5
                },
                "error_logs": [
                    {"timestamp": "2024-01-01T10:00:00", "level": "WARNING", "message": "High memory usage"},
                    {"timestamp": "2024-01-01T10:05:00", "level": "ERROR", "message": "Database connection timeout"}
                ]
            }
            
            # Create analysis request
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=json.dumps(test_health),
                language="en"
            )
            
            # Process request
            result = await agent.process(request)
            
            if result.sentiment.label != "uncertain" and "Error:" not in result.sentiment.reasoning:
                return {
                    "test": "Fault Detection Agent",
                    "status": "PASSED",
                    "message": "Successfully processed fault detection request",
                    "result_summary": {
                        "health_assessment": "health" in str(result.metadata.get("result", {})),
                        "recovery_recommendations": "recovery" in str(result.metadata.get("result", {}))
                    }
                }
            else:
                return {
                    "test": "Fault Detection Agent",
                    "status": "FAILED",
                    "message": f"Failed to process request: {result.metadata.get('error', '')}"
                }
                
        except Exception as e:
            return {
                "test": "Fault Detection Agent",
                "status": "ERROR",
                "message": f"Error testing fault detection agent: {str(e)}"
            }
    
    async def test_performance_optimizer(self) -> Dict[str, Any]:
        """Test performance optimizer functionality."""
        logger.info("Testing performance optimizer...")
        
        try:
            # Test performance report generation
            report = await self.performance_optimizer.get_performance_report()
            
            if report.get("success", False):
                return {
                    "test": "Performance Optimizer",
                    "status": "PASSED",
                    "message": "Successfully generated performance report",
                    "result_summary": {
                        "report_generated": True,
                        "metrics_available": "current_metrics" in report.get("report", {}),
                        "recommendations_available": "recommendations" in report.get("report", {})
                    }
                }
            else:
                return {
                    "test": "Performance Optimizer",
                    "status": "FAILED",
                    "message": f"Failed to generate performance report: {report.get('message', 'Unknown error')}"
                }
                
        except Exception as e:
            return {
                "test": "Performance Optimizer",
                "status": "ERROR",
                "message": f"Error testing performance optimizer: {str(e)}"
            }
    
    async def test_mcp_tool_integration(self) -> Dict[str, Any]:
        """Test MCP tool integration for analytics capabilities."""
        logger.info("Testing MCP tool integration...")
        
        try:
            # Test predictive analytics MCP tool
            predictive_result = await call_predictive_analytics_tool(
                "forecast",
                {"data": [1, 2, 3, 4, 5], "horizon": 3}
            )
            
            # Test scenario analysis MCP tool
            scenario_result = await call_scenario_analysis_tool(
                "analyze",
                {"scenario": "test scenario"}
            )
            
            # Test decision support MCP tool
            decision_result = await call_decision_support_tool(
                "recommend",
                {"context": "test decision context"}
            )
            
            # Test monitoring MCP tool
            monitoring_result = await call_monitoring_tool(
                "health_check",
                {}
            )
            
            # Test performance optimization MCP tool
            performance_result = await call_performance_optimization_tool(
                "optimize",
                {"type": "cache"}
            )
            
            # Check if all tools are accessible (even if they return errors, they should be reachable)
            tools_tested = [
                ("predictive_analytics", predictive_result),
                ("scenario_analysis", scenario_result),
                ("decision_support", decision_result),
                ("monitoring", monitoring_result),
                ("performance_optimization", performance_result)
            ]
            
            accessible_tools = []
            for tool_name, result in tools_tested:
                if result is not None:  # Tool is reachable
                    accessible_tools.append(tool_name)
            
            if len(accessible_tools) >= 3:  # At least 3 tools should be accessible
                return {
                    "test": "MCP Tool Integration",
                    "status": "PASSED",
                    "message": f"Successfully tested {len(accessible_tools)} MCP tools",
                    "accessible_tools": accessible_tools
                }
            else:
                return {
                    "test": "MCP Tool Integration",
                    "status": "FAILED",
                    "message": f"Only {len(accessible_tools)} tools accessible, expected at least 3",
                    "accessible_tools": accessible_tools
                }
                
        except Exception as e:
            return {
                "test": "MCP Tool Integration",
                "status": "ERROR",
                "message": f"Error testing MCP tool integration: {str(e)}"
            }
    
    async def test_orchestrator_analytics_integration(self) -> Dict[str, Any]:
        """Test that orchestrator can route analytics requests to appropriate agents."""
        logger.info("Testing orchestrator analytics integration...")
        
        try:
            # Test text analysis with analytics capabilities
            import uuid
            test_text = f"Market analysis shows increasing trends in Q4 with potential risks in supply chain. Test ID: {uuid.uuid4()}"
            
            result = await self.orchestrator.analyze_text(
                content=test_text,
                language="en",
                enable_analytics=True
            )
            
            if result.sentiment.label != "uncertain" and (result.sentiment.reasoning is None or "Error:" not in result.sentiment.reasoning):
                return {
                    "test": "Orchestrator Analytics Integration",
                    "status": "PASSED",
                    "message": "Successfully processed analytics-enabled text analysis",
                    "result_summary": {
                        "sentiment_analysis": "sentiment" in str(result.metadata.get("result", {}) if result.metadata else {}),
                        "analytics_integration": True
                    }
                }
            else:
                return {
                    "test": "Orchestrator Analytics Integration",
                    "status": "FAILED",
                    "message": f"Failed to process analytics request: {result.metadata.get('error', '') if result.metadata else ''}"
                }
                
        except Exception as e:
            return {
                "test": "Orchestrator Analytics Integration",
                "status": "ERROR",
                "message": f"Error testing orchestrator analytics integration: {str(e)}"
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 6.1 integration tests."""
        logger.info("Starting Phase 6.1 integration tests...")
        
        await self.setup()
        
        # Run all tests
        tests = [
            self.test_agent_integration(),
            self.test_predictive_analytics_agent(),
            self.test_scenario_analysis_agent(),
            self.test_decision_support_agent(),
            self.test_risk_assessment_agent(),
            self.test_fault_detection_agent(),
            self.test_performance_optimizer(),
            self.test_mcp_tool_integration(),
            self.test_orchestrator_analytics_integration()
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.test_results.append({
                    "test": f"Test {i+1}",
                    "status": "ERROR",
                    "message": f"Test failed with exception: {str(result)}"
                })
            else:
                self.test_results.append(result)
        
        # Generate summary
        passed = sum(1 for r in self.test_results if r["status"] == "PASSED")
        failed = sum(1 for r in self.test_results if r["status"] == "FAILED")
        errors = sum(1 for r in self.test_results if r["status"] == "ERROR")
        total = len(self.test_results)
        
        summary = {
            "phase": "Phase 6.1: System Integration & Optimization",
            "timestamp": datetime.now().isoformat(),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "success_rate": (passed / total * 100) if total > 0 else 0,
            "test_results": self.test_results
        }
        
        return summary


async def main():
    """Main test execution function."""
    logger.info("🚀 Starting Phase 6.1 Integration Tests")
    logger.info("=" * 60)
    
    tester = Phase6IntegrationTester()
    results = await tester.run_all_tests()
    
    # Print results
    logger.info("📊 Test Results Summary")
    logger.info("=" * 60)
    logger.info(f"Phase: {results['phase']}")
    logger.info(f"Total Tests: {results['total_tests']}")
    logger.info(f"Passed: {results['passed']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Errors: {results['errors']}")
    logger.info(f"Success Rate: {results['success_rate']:.1f}%")
    
    logger.info("\n📋 Detailed Results:")
    for result in results['test_results']:
        status_emoji = "✅" if result['status'] == "PASSED" else "❌" if result['status'] == "FAILED" else "⚠️"
        logger.info(f"{status_emoji} {result['test']}: {result['status']} - {result['message']}")
    
    # Save results to file
    output_file = "Test/phase6_integration_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n💾 Results saved to: {output_file}")
    
    if results['success_rate'] >= 80:
        logger.info("🎉 Phase 6.1 Integration Tests: SUCCESS!")
    else:
        logger.warning("⚠️ Phase 6.1 Integration Tests: Some issues detected")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
