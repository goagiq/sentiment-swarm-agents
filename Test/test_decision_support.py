#!/usr/bin/env python3
"""
Comprehensive Test Suite for Decision Support - Phase 6.2

This script provides comprehensive testing for all decision support components including:
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

# Import all decision support components
from src.core.decision_support.recommendation_engine import RecommendationEngine
from src.core.decision_support.action_prioritizer import ActionPrioritizer
from src.core.decision_support.implementation_planner import ImplementationPlanner
from src.core.decision_support.success_predictor import SuccessPredictor
from src.agents.decision_support_agent import DecisionSupportAgent
from src.core.models import DataType, AnalysisRequest, AnalysisResult
from src.core.orchestrator import SentimentOrchestrator


class ComprehensiveDecisionSupportTester:
    """Comprehensive test suite for decision support components."""
    
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
    
    def generate_test_business_context(self) -> Dict[str, Any]:
        """Generate test business context for decision support."""
        return {
            "business_objectives": [
                "Increase market share by 15%",
                "Reduce operational costs by 10%",
                "Improve customer satisfaction scores"
            ],
            "current_performance": {
                "revenue": 1000000,
                "market_share": 0.12,
                "customer_satisfaction": 7.5,
                "operational_costs": 800000
            },
            "constraints": {
                "budget_limit": 500000,
                "timeline": "6 months",
                "resource_availability": "medium"
            },
            "stakeholders": [
                "executive_team",
                "operations_team",
                "marketing_team"
            ]
        }
    
    async def setup(self):
        """Setup test environment."""
        logger.info("Setting up comprehensive decision support test environment...")
        self.start_time = time.time()
        
        # Initialize orchestrator
        self.orchestrator = SentimentOrchestrator()
        
        # Initialize components
        self.recommendation_engine = RecommendationEngine()
        self.action_prioritizer = ActionPrioritizer()
        self.implementation_planner = ImplementationPlanner()
        self.success_predictor = SuccessPredictor()
        self.decision_agent = DecisionSupportAgent()
        
        logger.info("✅ Test environment setup complete")
    
    async def test_recommendation_engine_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of recommendation engine."""
        logger.info("🧪 Testing Recommendation Engine (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_recommendations": False,
                "context_aware_recommendations": False,
                "recommendation_categorization": False,
                "confidence_scoring": False,
                "performance": False
            }
            
            # Test 1: Basic recommendations
            logger.info("   Testing basic recommendations...")
            business_context = self.generate_test_business_context()
            
            recommendations = self.recommendation_engine.generate_recommendations(
                business_context=business_context,
                analysis_type="strategic_planning"
            )
            
            assert recommendations is not None, "Should generate recommendations"
            assert hasattr(recommendations, 'recommendations'), "Should have recommendations list"
            assert len(recommendations.recommendations) > 0, "Should have at least one recommendation"
            test_results["basic_recommendations"] = True
            
            # Test 2: Context-aware recommendations
            logger.info("   Testing context-aware recommendations...")
            specific_context = {
                **business_context,
                "industry": "technology",
                "company_size": "medium",
                "growth_stage": "expansion"
            }
            
            context_recommendations = self.recommendation_engine.generate_recommendations(
                business_context=specific_context,
                analysis_type="market_expansion"
            )
            
            assert context_recommendations is not None, "Should generate context-aware recommendations"
            assert len(context_recommendations.recommendations) > 0, "Should have context-specific recommendations"
            test_results["context_aware_recommendations"] = True
            
            # Test 3: Recommendation categorization
            logger.info("   Testing recommendation categorization...")
            categories = ["strategic", "operational", "tactical", "financial"]
            
            for category in categories:
                cat_recommendations = self.recommendation_engine.generate_recommendations(
                    business_context=business_context,
                    analysis_type=category
                )
                assert cat_recommendations is not None, f"Should generate {category} recommendations"
            
            test_results["recommendation_categorization"] = True
            
            # Test 4: Confidence scoring
            logger.info("   Testing confidence scoring...")
            scored_recommendations = self.recommendation_engine.generate_recommendations(
                business_context=business_context,
                analysis_type="strategic_planning",
                include_confidence=True
            )
            
            assert hasattr(scored_recommendations, 'confidence_scores'), "Should include confidence scores"
            for score in scored_recommendations.confidence_scores:
                assert 0 <= score <= 1, "Confidence scores should be between 0 and 1"
            test_results["confidence_scoring"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            complex_context = {
                **business_context,
                "historical_data": {"revenue": [1000000, 1100000, 1200000]},
                "market_conditions": {"competition": "high", "growth_rate": 0.08},
                "internal_capabilities": {"technology": "advanced", "talent": "skilled"}
            }
            
            perf_start = time.time()
            self.recommendation_engine.generate_recommendations(
                business_context=complex_context,
                analysis_type="comprehensive_strategy"
            )
            perf_time = time.time() - perf_start
            
            assert perf_time < 8.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Recommendation Engine Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Recommendation Engine Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_action_prioritizer_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of action prioritizer."""
        logger.info("🧪 Testing Action Prioritizer (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_prioritization": False,
                "multi_criteria_prioritization": False,
                "priority_scoring": False,
                "resource_allocation": False,
                "performance": False
            }
            
            # Test 1: Basic prioritization
            logger.info("   Testing basic prioritization...")
            actions = [
                {"name": "Market Research", "impact": 8, "effort": 6, "urgency": 7},
                {"name": "Product Development", "impact": 9, "effort": 8, "urgency": 5},
                {"name": "Cost Reduction", "impact": 7, "effort": 4, "urgency": 8}
            ]
            
            prioritized_actions = self.action_prioritizer.prioritize_actions(actions)
            
            assert prioritized_actions is not None, "Should prioritize actions"
            assert len(prioritized_actions) == len(actions), "Should prioritize all actions"
            assert prioritized_actions[0]["priority_score"] >= prioritized_actions[-1]["priority_score"], "Should be sorted by priority"
            test_results["basic_prioritization"] = True
            
            # Test 2: Multi-criteria prioritization
            logger.info("   Testing multi-criteria prioritization...")
            criteria = ["impact", "effort", "urgency", "risk", "resource_availability"]
            weights = [0.3, 0.2, 0.25, 0.15, 0.1]
            
            multi_prioritized = self.action_prioritizer.prioritize_actions(
                actions=actions,
                criteria=criteria,
                weights=weights
            )
            
            assert multi_prioritized is not None, "Should handle multi-criteria prioritization"
            assert all("priority_score" in action for action in multi_prioritized), "All actions should have priority scores"
            test_results["multi_criteria_prioritization"] = True
            
            # Test 3: Priority scoring
            logger.info("   Testing priority scoring...")
            detailed_actions = [
                {
                    "name": "Strategic Initiative A",
                    "impact": 9,
                    "effort": 7,
                    "urgency": 8,
                    "risk": 6,
                    "resource_availability": 8,
                    "dependencies": ["Market Research"],
                    "timeline": "3 months"
                }
            ]
            
            scored_actions = self.action_prioritizer.prioritize_actions(
                actions=detailed_actions,
                include_detailed_scoring=True
            )
            
            assert hasattr(scored_actions[0], 'detailed_scores'), "Should include detailed scoring"
            assert hasattr(scored_actions[0], 'rationale'), "Should include scoring rationale"
            test_results["priority_scoring"] = True
            
            # Test 4: Resource allocation
            logger.info("   Testing resource allocation...")
            resource_constraints = {
                "budget": 500000,
                "personnel": 10,
                "timeline": "6 months"
            }
            
            allocated_actions = self.action_prioritizer.allocate_resources(
                actions=actions,
                constraints=resource_constraints
            )
            
            assert allocated_actions is not None, "Should allocate resources"
            assert hasattr(allocated_actions, 'resource_allocation'), "Should have resource allocation"
            test_results["resource_allocation"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            many_actions = [
                {"name": f"Action {i}", "impact": i % 10, "effort": (i + 2) % 10, "urgency": (i + 1) % 10}
                for i in range(50)
            ]
            
            perf_start = time.time()
            self.action_prioritizer.prioritize_actions(many_actions)
            perf_time = time.time() - perf_start
            
            assert perf_time < 3.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Action Prioritizer Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Action Prioritizer Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_implementation_planner_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of implementation planner."""
        logger.info("🧪 Testing Implementation Planner (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_planning": False,
                "phase_planning": False,
                "resource_planning": False,
                "timeline_planning": False,
                "performance": False
            }
            
            # Test 1: Basic planning
            logger.info("   Testing basic planning...")
            action = {
                "name": "Market Expansion Initiative",
                "description": "Expand into new geographic markets",
                "priority_score": 8.5,
                "estimated_effort": 7,
                "required_resources": ["marketing_team", "sales_team", "budget"]
            }
            
            plan = self.implementation_planner.create_implementation_plan(action)
            
            assert plan is not None, "Should create implementation plan"
            assert hasattr(plan, 'phases'), "Should have implementation phases"
            assert hasattr(plan, 'timeline'), "Should have timeline"
            assert hasattr(plan, 'resource_requirements'), "Should have resource requirements"
            test_results["basic_planning"] = True
            
            # Test 2: Phase planning
            logger.info("   Testing phase planning...")
            detailed_action = {
                **action,
                "complexity": "high",
                "dependencies": ["Market Research", "Product Localization"],
                "success_criteria": ["15% market share", "positive ROI"]
            }
            
            phased_plan = self.implementation_planner.create_implementation_plan(
                action=detailed_action,
                include_phases=True
            )
            
            assert len(phased_plan.phases) > 1, "Should have multiple phases"
            for phase in phased_plan.phases:
                assert hasattr(phase, 'tasks'), "Each phase should have tasks"
                assert hasattr(phase, 'milestones'), "Each phase should have milestones"
            test_results["phase_planning"] = True
            
            # Test 3: Resource planning
            logger.info("   Testing resource planning...")
            resource_plan = self.implementation_planner.plan_resources(
                action=action,
                available_resources={
                    "budget": 1000000,
                    "personnel": 20,
                    "time": "12 months"
                }
            )
            
            assert resource_plan is not None, "Should plan resources"
            assert hasattr(resource_plan, 'resource_allocation'), "Should have resource allocation"
            assert hasattr(resource_plan, 'cost_breakdown'), "Should have cost breakdown"
            test_results["resource_planning"] = True
            
            # Test 4: Timeline planning
            logger.info("   Testing timeline planning...")
            timeline_plan = self.implementation_planner.create_timeline(
                action=action,
                start_date="2024-01-01",
                constraints={"deadline": "2024-12-31", "critical_path": True}
            )
            
            assert timeline_plan is not None, "Should create timeline"
            assert hasattr(timeline_plan, 'schedule'), "Should have schedule"
            assert hasattr(timeline_plan, 'critical_path'), "Should have critical path"
            test_results["timeline_planning"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            complex_actions = [
                {
                    "name": f"Complex Initiative {i}",
                    "description": f"Complex business initiative {i}",
                    "priority_score": 8.0 + (i * 0.1),
                    "estimated_effort": 6 + (i % 4),
                    "required_resources": ["team", "budget", "technology"],
                    "dependencies": [f"Dependency {j}" for j in range(3)]
                }
                for i in range(20)
            ]
            
            perf_start = time.time()
            for action in complex_actions:
                self.implementation_planner.create_implementation_plan(action)
            perf_time = time.time() - perf_start
            
            assert perf_time < 10.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Implementation Planner Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Implementation Planner Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_success_predictor_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of success predictor."""
        logger.info("🧪 Testing Success Predictor (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_prediction": False,
                "multi_factor_prediction": False,
                "confidence_analysis": False,
                "risk_assessment": False,
                "performance": False
            }
            
            # Test 1: Basic prediction
            logger.info("   Testing basic prediction...")
            action = {
                "name": "Product Launch",
                "type": "strategic",
                "complexity": "medium",
                "resource_availability": "high",
                "team_experience": "expert"
            }
            
            prediction = self.success_predictor.predict_success(action)
            
            assert prediction is not None, "Should predict success"
            assert hasattr(prediction, 'success_probability'), "Should have success probability"
            assert 0 <= prediction.success_probability <= 1, "Success probability should be between 0 and 1"
            test_results["basic_prediction"] = True
            
            # Test 2: Multi-factor prediction
            logger.info("   Testing multi-factor prediction...")
            detailed_action = {
                **action,
                "historical_success_rate": 0.75,
                "market_conditions": "favorable",
                "competition_level": "moderate",
                "organizational_support": "high",
                "timeline_realism": "realistic"
            }
            
            detailed_prediction = self.success_predictor.predict_success(
                action=detailed_action,
                include_factors=True
            )
            
            assert hasattr(detailed_prediction, 'contributing_factors'), "Should include contributing factors"
            assert hasattr(detailed_prediction, 'risk_factors'), "Should include risk factors"
            test_results["multi_factor_prediction"] = True
            
            # Test 3: Confidence analysis
            logger.info("   Testing confidence analysis...")
            confidence_analysis = self.success_predictor.analyze_confidence(
                action=action,
                include_uncertainty=True
            )
            
            assert hasattr(confidence_analysis, 'confidence_level'), "Should have confidence level"
            assert hasattr(confidence_analysis, 'uncertainty_factors'), "Should have uncertainty factors"
            test_results["confidence_analysis"] = True
            
            # Test 4: Risk assessment
            logger.info("   Testing risk assessment...")
            risk_assessment = self.success_predictor.assess_risks(action)
            
            assert risk_assessment is not None, "Should assess risks"
            assert hasattr(risk_assessment, 'risk_factors'), "Should have risk factors"
            assert hasattr(risk_assessment, 'mitigation_suggestions'), "Should have mitigation suggestions"
            test_results["risk_assessment"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            many_actions = [
                {
                    "name": f"Action {i}",
                    "type": "strategic" if i % 2 == 0 else "operational",
                    "complexity": "low" if i % 3 == 0 else "medium" if i % 3 == 1 else "high",
                    "resource_availability": "high" if i % 2 == 0 else "medium",
                    "team_experience": "expert" if i % 4 == 0 else "intermediate"
                }
                for i in range(30)
            ]
            
            perf_start = time.time()
            for action in many_actions:
                self.success_predictor.predict_success(action)
            perf_time = time.time() - perf_start
            
            assert perf_time < 5.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Success Predictor Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Success Predictor Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_decision_agent_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of decision support agent."""
        logger.info("🧪 Testing Decision Support Agent (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_analysis": False,
                "recommendation_generation": False,
                "action_planning": False,
                "success_prediction": False,
                "performance": False
            }
            
            # Test 1: Basic analysis
            logger.info("   Testing basic analysis...")
            request = AnalysisRequest(
                content="Help me make strategic decisions for business growth",
                data_type=DataType.TEXT,
                analysis_type="decision_support",
                language="en"
            )
            
            result = await self.decision_agent.analyze(request)
            
            assert isinstance(result, AnalysisResult), "Should return AnalysisResult"
            assert result.success, "Analysis should be successful"
            test_results["basic_analysis"] = True
            
            # Test 2: Recommendation generation
            logger.info("   Testing recommendation generation...")
            recommendation_request = AnalysisRequest(
                content="Generate recommendations for improving customer satisfaction",
                data_type=DataType.TEXT,
                analysis_type="decision_support",
                language="en"
            )
            
            result = await self.decision_agent.analyze(recommendation_request)
            assert result.success, "Recommendation generation should be successful"
            test_results["recommendation_generation"] = True
            
            # Test 3: Action planning
            logger.info("   Testing action planning...")
            planning_request = AnalysisRequest(
                content="Create implementation plan for market expansion strategy",
                data_type=DataType.TEXT,
                analysis_type="decision_support",
                language="en"
            )
            
            result = await self.decision_agent.analyze(planning_request)
            assert result.success, "Action planning should be successful"
            test_results["action_planning"] = True
            
            # Test 4: Success prediction
            logger.info("   Testing success prediction...")
            prediction_request = AnalysisRequest(
                content="Predict success probability for new product launch",
                data_type=DataType.TEXT,
                analysis_type="decision_support",
                language="en"
            )
            
            result = await self.decision_agent.analyze(prediction_request)
            assert result.success, "Success prediction should be successful"
            test_results["success_prediction"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            large_request = AnalysisRequest(
                content="Comprehensive decision support analysis with multiple business scenarios and strategic planning " * 50,
                data_type=DataType.TEXT,
                analysis_type="decision_support",
                language="en"
            )
            
            perf_start = time.time()
            result = await self.decision_agent.analyze(large_request)
            perf_time = time.time() - perf_start
            
            assert perf_time < 60.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Decision Support Agent Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Decision Support Agent Comprehensive",
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
                "data_consistency": False,
                "result_validation": False,
                "system_stability": False
            }
            
            # Test 1: Orchestrator integration
            logger.info("   Testing orchestrator integration...")
            request = AnalysisRequest(
                content="Comprehensive decision support for strategic planning",
                data_type=DataType.TEXT,
                analysis_type="decision_support",
                language="en"
            )
            
            result = await self.orchestrator.analyze(request)
            assert result.success, "Orchestrator should handle decision support"
            test_results["orchestrator_integration"] = True
            
            # Test 2: Workflow completion
            logger.info("   Testing workflow completion...")
            # Test complete workflow: Generate recommendations -> Prioritize actions -> Plan implementation -> Predict success
            business_context = self.generate_test_business_context()
            
            # Generate recommendations
            recommendations = self.recommendation_engine.generate_recommendations(
                business_context=business_context,
                analysis_type="strategic_planning"
            )
            
            # Prioritize actions
            actions = [{"name": rec.name, "impact": 8, "effort": 6, "urgency": 7} 
                      for rec in recommendations.recommendations[:3]]
            prioritized_actions = self.action_prioritizer.prioritize_actions(actions)
            
            # Plan implementation
            implementation_plans = []
            for action in prioritized_actions[:2]:
                plan = self.implementation_planner.create_implementation_plan(action)
                implementation_plans.append(plan)
            
            # Predict success
            success_predictions = []
            for action in prioritized_actions[:2]:
                prediction = self.success_predictor.predict_success(action)
                success_predictions.append(prediction)
            
            assert all([recommendations, prioritized_actions, implementation_plans, success_predictions]), "Complete workflow should work"
            test_results["workflow_completion"] = True
            
            # Test 3: Data consistency
            logger.info("   Testing data consistency...")
            # Verify that same action produces consistent results across components
            test_action = prioritized_actions[0]
            
            # Check consistency in prioritization
            priority1 = self.action_prioritizer.prioritize_actions([test_action])
            priority2 = self.action_prioritizer.prioritize_actions([test_action])
            assert priority1[0]["priority_score"] == priority2[0]["priority_score"], "Prioritization should be consistent"
            
            # Check consistency in success prediction
            prediction1 = self.success_predictor.predict_success(test_action)
            prediction2 = self.success_predictor.predict_success(test_action)
            assert abs(prediction1.success_probability - prediction2.success_probability) < 0.1, "Success prediction should be consistent"
            
            test_results["data_consistency"] = True
            
            # Test 4: Result validation
            logger.info("   Testing result validation...")
            # Validate that results make sense
            for action in prioritized_actions:
                assert 0 <= action["priority_score"] <= 10, "Priority score should be valid"
                
                prediction = self.success_predictor.predict_success(action)
                assert 0 <= prediction.success_probability <= 1, "Success probability should be valid"
                
                plan = self.implementation_planner.create_implementation_plan(action)
                assert plan is not None, "Implementation plan should be created"
            
            test_results["result_validation"] = True
            
            # Test 5: System stability
            logger.info("   Testing system stability...")
            # Run multiple operations to test stability
            operations = []
            for i in range(10):
                op_request = AnalysisRequest(
                    content=f"Stability test decision support {i}",
                    data_type=DataType.TEXT,
                    analysis_type="decision_support",
                    language="en"
                )
                operations.append(self.decision_agent.analyze(op_request))
            
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
        logger.info("🚀 Starting Comprehensive Decision Support Testing...")
        
        await self.setup()
        
        # Run all test categories
        test_categories = [
            ("Recommendation Engine", self.test_recommendation_engine_comprehensive),
            ("Action Prioritizer", self.test_action_prioritizer_comprehensive),
            ("Implementation Planner", self.test_implementation_planner_comprehensive),
            ("Success Predictor", self.test_success_predictor_comprehensive),
            ("Decision Support Agent", self.test_decision_agent_comprehensive),
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
        with open("Test/comprehensive_decision_support_results.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("COMPREHENSIVE DECISION SUPPORT TESTING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        logger.info(f"Total Duration: {overall_duration:.2f}s")
        logger.info(f"Results saved to: Test/comprehensive_decision_support_results.json")
        
        return report


async def main():
    """Main test execution function."""
    tester = ComprehensiveDecisionSupportTester()
    results = await tester.run_all_tests()
    return results


if __name__ == "__main__":
    # Run the comprehensive test suite
    results = asyncio.run(main())
    print(f"\nTest execution completed. Success rate: {results['test_summary']['success_rate']*100:.1f}%")
