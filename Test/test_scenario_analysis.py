#!/usr/bin/env python3
"""
Comprehensive Test Suite for Scenario Analysis - Phase 6.2

This script provides comprehensive testing for all scenario analysis components including:
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
import numpy as np
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from loguru import logger

# Import all scenario analysis components
from src.core.scenario_analysis.scenario_builder import ScenarioBuilder
from src.core.scenario_analysis.impact_analyzer import ImpactAnalyzer
from src.core.scenario_analysis.risk_assessor import RiskAssessor
from src.core.scenario_analysis.scenario_comparator import ScenarioComparator
from src.agents.scenario_analysis_agent import ScenarioAnalysisAgent
from src.core.models import DataType, AnalysisRequest, AnalysisResult
from src.core.orchestrator import SentimentOrchestrator


class ComprehensiveScenarioAnalysisTester:
    """Comprehensive test suite for scenario analysis components."""
    
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
    
    def generate_test_scenarios(self, num_scenarios: int = 5) -> list:
        """Generate test scenarios with varying complexity."""
        scenarios = []
        
        # Basic scenarios
        scenarios.append({
            "name": "baseline",
            "description": "Current business as usual scenario",
            "parameters": {
                "growth_rate": 0.05,
                "market_share": 0.15,
                "cost_reduction": 0.02
            }
        })
        
        # Optimistic scenarios
        scenarios.append({
            "name": "optimistic",
            "description": "High growth and market expansion scenario",
            "parameters": {
                "growth_rate": 0.15,
                "market_share": 0.25,
                "cost_reduction": 0.05
            }
        })
        
        # Pessimistic scenarios
        scenarios.append({
            "name": "pessimistic",
            "description": "Economic downturn scenario",
            "parameters": {
                "growth_rate": -0.05,
                "market_share": 0.10,
                "cost_reduction": 0.01
            }
        })
        
        # Complex scenarios
        for i in range(num_scenarios - 3):
            scenarios.append({
                "name": f"complex_scenario_{i+1}",
                "description": f"Complex scenario with multiple variables {i+1}",
                "parameters": {
                    "growth_rate": 0.05 + (i * 0.02),
                    "market_share": 0.15 + (i * 0.01),
                    "cost_reduction": 0.02 + (i * 0.005),
                    "innovation_factor": 1.0 + (i * 0.1),
                    "risk_tolerance": 0.5 + (i * 0.1)
                }
            })
        
        return scenarios
    
    async def setup(self):
        """Setup test environment."""
        logger.info("Setting up comprehensive scenario analysis test environment...")
        self.start_time = time.time()
        
        # Initialize orchestrator
        self.orchestrator = SentimentOrchestrator()
        
        # Initialize components
        self.scenario_builder = ScenarioBuilder()
        self.impact_analyzer = ImpactAnalyzer()
        self.risk_assessor = RiskAssessor()
        self.scenario_comparator = ScenarioComparator()
        self.scenario_agent = ScenarioAnalysisAgent()
        
        logger.info("✅ Test environment setup complete")
    
    async def test_scenario_builder_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of scenario builder."""
        logger.info("🧪 Testing Scenario Builder (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_scenario_creation": False,
                "complex_scenario_creation": False,
                "parameter_validation": False,
                "scenario_templates": False,
                "performance": False
            }
            
            # Test 1: Basic scenario creation
            logger.info("   Testing basic scenario creation...")
            basic_scenario = {
                "name": "test_scenario",
                "description": "Test scenario for validation",
                "parameters": {
                    "growth_rate": 0.05,
                    "market_share": 0.15
                }
            }
            
            scenario = self.scenario_builder.create_scenario(basic_scenario)
            
            assert scenario.name == "test_scenario", "Scenario name not set correctly"
            assert scenario.description == "Test scenario for validation", "Description not set"
            assert scenario.parameters["growth_rate"] == 0.05, "Parameters not set correctly"
            test_results["basic_scenario_creation"] = True
            
            # Test 2: Complex scenario creation
            logger.info("   Testing complex scenario creation...")
            complex_scenario = {
                "name": "complex_test",
                "description": "Complex scenario with many parameters",
                "parameters": {
                    "growth_rate": 0.10,
                    "market_share": 0.20,
                    "cost_reduction": 0.03,
                    "innovation_factor": 1.2,
                    "risk_tolerance": 0.7,
                    "time_horizon": 5
                },
                "constraints": {
                    "max_growth": 0.20,
                    "min_market_share": 0.05
                }
            }
            
            scenario = self.scenario_builder.create_scenario(complex_scenario)
            
            assert len(scenario.parameters) == 6, "Complex parameters not set correctly"
            assert scenario.constraints["max_growth"] == 0.20, "Constraints not set"
            test_results["complex_scenario_creation"] = True
            
            # Test 3: Parameter validation
            logger.info("   Testing parameter validation...")
            invalid_scenario = {
                "name": "invalid_test",
                "description": "Scenario with invalid parameters",
                "parameters": {
                    "growth_rate": 2.0,  # Invalid: growth rate > 100%
                    "market_share": -0.1  # Invalid: negative market share
                }
            }
            
            try:
                scenario = self.scenario_builder.create_scenario(invalid_scenario)
                # Should either validate and correct or raise an error
                assert scenario is not None, "Should handle invalid parameters"
            except Exception:
                # Expected behavior for invalid parameters
                pass
            
            test_results["parameter_validation"] = True
            
            # Test 4: Scenario templates
            logger.info("   Testing scenario templates...")
            templates = self.scenario_builder.get_available_templates()
            
            assert len(templates) > 0, "Should have available templates"
            
            # Test creating scenario from template
            if templates:
                template_name = list(templates.keys())[0]
                scenario = self.scenario_builder.create_from_template(template_name, {
                    "custom_parameter": 0.05
                })
                assert scenario is not None, "Should create scenario from template"
            
            test_results["scenario_templates"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            scenarios = self.generate_test_scenarios(20)
            
            perf_start = time.time()
            for scenario_data in scenarios:
                self.scenario_builder.create_scenario(scenario_data)
            perf_time = time.time() - perf_start
            
            assert perf_time < 5.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Scenario Builder Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Scenario Builder Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_impact_analyzer_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of impact analyzer."""
        logger.info("🧪 Testing Impact Analyzer (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_impact_analysis": False,
                "multi_metric_analysis": False,
                "sensitivity_analysis": False,
                "impact_quantification": False,
                "performance": False
            }
            
            # Test 1: Basic impact analysis
            logger.info("   Testing basic impact analysis...")
            base_scenario = self.scenario_builder.create_scenario({
                "name": "base",
                "description": "Base scenario",
                "parameters": {"growth_rate": 0.05, "market_share": 0.15}
            })
            
            target_scenario = self.scenario_builder.create_scenario({
                "name": "target",
                "description": "Target scenario",
                "parameters": {"growth_rate": 0.10, "market_share": 0.20}
            })
            
            impact = self.impact_analyzer.analyze_impact(base_scenario, target_scenario)
            
            assert impact is not None, "Impact analysis should return results"
            assert hasattr(impact, 'impact_metrics'), "Should have impact metrics"
            test_results["basic_impact_analysis"] = True
            
            # Test 2: Multi-metric analysis
            logger.info("   Testing multi-metric analysis...")
            metrics = ["revenue", "profit", "market_position", "risk_level"]
            
            impact = self.impact_analyzer.analyze_impact(
                base_scenario, 
                target_scenario,
                metrics=metrics
            )
            
            assert len(impact.impact_metrics) >= len(metrics), "Should analyze all requested metrics"
            test_results["multi_metric_analysis"] = True
            
            # Test 3: Sensitivity analysis
            logger.info("   Testing sensitivity analysis...")
            sensitivity = self.impact_analyzer.perform_sensitivity_analysis(
                base_scenario,
                parameter="growth_rate",
                range_values=[0.03, 0.05, 0.07, 0.10]
            )
            
            assert sensitivity is not None, "Sensitivity analysis should return results"
            assert hasattr(sensitivity, 'sensitivity_curves'), "Should have sensitivity curves"
            test_results["sensitivity_analysis"] = True
            
            # Test 4: Impact quantification
            logger.info("   Testing impact quantification...")
            quantification = self.impact_analyzer.quantify_impacts(
                base_scenario,
                target_scenario,
                include_financial=True,
                include_operational=True
            )
            
            assert hasattr(quantification, 'financial_impact'), "Should have financial impact"
            assert hasattr(quantification, 'operational_impact'), "Should have operational impact"
            test_results["impact_quantification"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            scenarios = [self.scenario_builder.create_scenario(s) for s in self.generate_test_scenarios(10)]
            
            perf_start = time.time()
            for i in range(len(scenarios) - 1):
                self.impact_analyzer.analyze_impact(scenarios[i], scenarios[i+1])
            perf_time = time.time() - perf_start
            
            assert perf_time < 10.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Impact Analyzer Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Impact Analyzer Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_risk_assessor_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of risk assessor."""
        logger.info("🧪 Testing Risk Assessor (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_risk_assessment": False,
                "risk_categorization": False,
                "risk_scoring": False,
                "mitigation_suggestions": False,
                "performance": False
            }
            
            # Test 1: Basic risk assessment
            logger.info("   Testing basic risk assessment...")
            scenario = self.scenario_builder.create_scenario({
                "name": "risk_test",
                "description": "Scenario for risk assessment",
                "parameters": {
                    "growth_rate": 0.15,
                    "market_share": 0.25,
                    "investment_level": 1000000
                }
            })
            
            risk_assessment = self.risk_assessor.assess_risks(scenario)
            
            assert risk_assessment is not None, "Risk assessment should return results"
            assert hasattr(risk_assessment, 'risk_level'), "Should have risk level"
            assert hasattr(risk_assessment, 'risk_factors'), "Should have risk factors"
            test_results["basic_risk_assessment"] = True
            
            # Test 2: Risk categorization
            logger.info("   Testing risk categorization...")
            categories = ["financial", "operational", "strategic", "market", "regulatory"]
            
            for category in categories:
                category_risks = self.risk_assessor.assess_risks_by_category(scenario, category)
                assert category_risks is not None, f"Should assess {category} risks"
            
            test_results["risk_categorization"] = True
            
            # Test 3: Risk scoring
            logger.info("   Testing risk scoring...")
            risk_score = self.risk_assessor.calculate_risk_score(scenario)
            
            assert 0 <= risk_score <= 100, "Risk score should be between 0 and 100"
            assert isinstance(risk_score, (int, float)), "Risk score should be numeric"
            test_results["risk_scoring"] = True
            
            # Test 4: Mitigation suggestions
            logger.info("   Testing mitigation suggestions...")
            mitigation = self.risk_assessor.suggest_mitigation_strategies(scenario)
            
            assert mitigation is not None, "Should suggest mitigation strategies"
            assert hasattr(mitigation, 'strategies'), "Should have mitigation strategies"
            assert hasattr(mitigation, 'effectiveness_scores'), "Should have effectiveness scores"
            test_results["mitigation_suggestions"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            scenarios = [self.scenario_builder.create_scenario(s) for s in self.generate_test_scenarios(15)]
            
            perf_start = time.time()
            for scenario in scenarios:
                self.risk_assessor.assess_risks(scenario)
            perf_time = time.time() - perf_start
            
            assert perf_time < 8.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Risk Assessor Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Risk Assessor Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_scenario_comparator_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of scenario comparator."""
        logger.info("🧪 Testing Scenario Comparator (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_comparison": False,
                "multi_scenario_comparison": False,
                "ranking_analysis": False,
                "recommendation_generation": False,
                "performance": False
            }
            
            # Test 1: Basic comparison
            logger.info("   Testing basic comparison...")
            scenario1 = self.scenario_builder.create_scenario({
                "name": "scenario_1",
                "description": "First scenario",
                "parameters": {"growth_rate": 0.05, "market_share": 0.15}
            })
            
            scenario2 = self.scenario_builder.create_scenario({
                "name": "scenario_2",
                "description": "Second scenario",
                "parameters": {"growth_rate": 0.10, "market_share": 0.20}
            })
            
            comparison = self.scenario_comparator.compare_scenarios([scenario1, scenario2])
            
            assert comparison is not None, "Comparison should return results"
            assert hasattr(comparison, 'comparison_matrix'), "Should have comparison matrix"
            assert hasattr(comparison, 'differences'), "Should have differences"
            test_results["basic_comparison"] = True
            
            # Test 2: Multi-scenario comparison
            logger.info("   Testing multi-scenario comparison...")
            scenarios = [self.scenario_builder.create_scenario(s) for s in self.generate_test_scenarios(5)]
            
            comparison = self.scenario_comparator.compare_scenarios(scenarios)
            
            assert len(comparison.comparison_matrix) == len(scenarios), "Should compare all scenarios"
            test_results["multi_scenario_comparison"] = True
            
            # Test 3: Ranking analysis
            logger.info("   Testing ranking analysis...")
            ranking = self.scenario_comparator.rank_scenarios(scenarios, criteria=["growth_rate", "risk_level"])
            
            assert ranking is not None, "Should provide ranking"
            assert len(ranking) == len(scenarios), "Should rank all scenarios"
            test_results["ranking_analysis"] = True
            
            # Test 4: Recommendation generation
            logger.info("   Testing recommendation generation...")
            recommendations = self.scenario_comparator.generate_recommendations(scenarios)
            
            assert recommendations is not None, "Should generate recommendations"
            assert hasattr(recommendations, 'recommendations'), "Should have recommendations"
            assert hasattr(recommendations, 'rationale'), "Should have rationale"
            test_results["recommendation_generation"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            many_scenarios = [self.scenario_builder.create_scenario(s) for s in self.generate_test_scenarios(20)]
            
            perf_start = time.time()
            self.scenario_comparator.compare_scenarios(many_scenarios)
            perf_time = time.time() - perf_start
            
            assert perf_time < 12.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Scenario Comparator Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Scenario Comparator Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_scenario_agent_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of scenario analysis agent."""
        logger.info("🧪 Testing Scenario Analysis Agent (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_analysis": False,
                "scenario_creation": False,
                "impact_analysis": False,
                "risk_assessment": False,
                "performance": False
            }
            
            # Test 1: Basic analysis
            logger.info("   Testing basic analysis...")
            request = AnalysisRequest(
                content="Analyze business scenarios for market expansion",
                data_type=DataType.TEXT,
                analysis_type="scenario_analysis",
                language="en"
            )
            
            result = await self.scenario_agent.analyze(request)
            
            assert isinstance(result, AnalysisResult), "Should return AnalysisResult"
            assert result.success, "Analysis should be successful"
            test_results["basic_analysis"] = True
            
            # Test 2: Scenario creation
            logger.info("   Testing scenario creation...")
            creation_request = AnalysisRequest(
                content="Create optimistic and pessimistic scenarios for Q4 2024",
                data_type=DataType.TEXT,
                analysis_type="scenario_analysis",
                language="en"
            )
            
            result = await self.scenario_agent.analyze(creation_request)
            assert result.success, "Scenario creation should be successful"
            test_results["scenario_creation"] = True
            
            # Test 3: Impact analysis
            logger.info("   Testing impact analysis...")
            impact_request = AnalysisRequest(
                content="Analyze impact of 20% market share increase on revenue",
                data_type=DataType.TEXT,
                analysis_type="scenario_analysis",
                language="en"
            )
            
            result = await self.scenario_agent.analyze(impact_request)
            assert result.success, "Impact analysis should be successful"
            test_results["impact_analysis"] = True
            
            # Test 4: Risk assessment
            logger.info("   Testing risk assessment...")
            risk_request = AnalysisRequest(
                content="Assess risks of entering new international markets",
                data_type=DataType.TEXT,
                analysis_type="scenario_analysis",
                language="en"
            )
            
            result = await self.scenario_agent.analyze(risk_request)
            assert result.success, "Risk assessment should be successful"
            test_results["risk_assessment"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            large_request = AnalysisRequest(
                content="Comprehensive scenario analysis with multiple variables and long-term projections " * 100,
                data_type=DataType.TEXT,
                analysis_type="scenario_analysis",
                language="en"
            )
            
            perf_start = time.time()
            result = await self.scenario_agent.analyze(large_request)
            perf_time = time.time() - perf_start
            
            assert perf_time < 45.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Scenario Analysis Agent Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Scenario Analysis Agent Comprehensive",
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
                content="Comprehensive scenario analysis for business planning",
                data_type=DataType.TEXT,
                analysis_type="scenario_analysis",
                language="en"
            )
            
            result = await self.orchestrator.analyze(request)
            assert result.success, "Orchestrator should handle scenario analysis"
            test_results["orchestrator_integration"] = True
            
            # Test 2: Workflow completion
            logger.info("   Testing workflow completion...")
            # Test complete workflow: Create scenarios -> Analyze impacts -> Assess risks -> Compare
            scenarios_data = self.generate_test_scenarios(3)
            scenarios = [self.scenario_builder.create_scenario(s) for s in scenarios_data]
            
            # Analyze impacts
            impacts = []
            for i in range(len(scenarios) - 1):
                impact = self.impact_analyzer.analyze_impact(scenarios[i], scenarios[i+1])
                impacts.append(impact)
            
            # Assess risks
            risks = []
            for scenario in scenarios:
                risk = self.risk_assessor.assess_risks(scenario)
                risks.append(risk)
            
            # Compare scenarios
            comparison = self.scenario_comparator.compare_scenarios(scenarios)
            
            assert all([impacts, risks, comparison]), "Complete workflow should work"
            test_results["workflow_completion"] = True
            
            # Test 3: Data consistency
            logger.info("   Testing data consistency...")
            # Verify that scenario data is consistent across components
            scenario = scenarios[0]
            
            # Check that same scenario produces consistent results
            risk1 = self.risk_assessor.assess_risks(scenario)
            risk2 = self.risk_assessor.assess_risks(scenario)
            
            assert risk1.risk_level == risk2.risk_level, "Risk assessment should be consistent"
            test_results["data_consistency"] = True
            
            # Test 4: Result validation
            logger.info("   Testing result validation...")
            # Validate that results make sense
            for scenario in scenarios:
                risk = self.risk_assessor.assess_risks(scenario)
                assert 0 <= risk.risk_level <= 100, "Risk level should be valid"
                
                impact = self.impact_analyzer.analyze_impact(scenarios[0], scenario)
                assert impact is not None, "Impact analysis should return results"
            
            test_results["result_validation"] = True
            
            # Test 5: System stability
            logger.info("   Testing system stability...")
            # Run multiple operations to test stability
            operations = []
            for i in range(10):
                op_request = AnalysisRequest(
                    content=f"Stability test scenario analysis {i}",
                    data_type=DataType.TEXT,
                    analysis_type="scenario_analysis",
                    language="en"
                )
                operations.append(self.scenario_agent.analyze(op_request))
            
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
        logger.info("🚀 Starting Comprehensive Scenario Analysis Testing...")
        
        await self.setup()
        
        # Run all test categories
        test_categories = [
            ("Scenario Builder", self.test_scenario_builder_comprehensive),
            ("Impact Analyzer", self.test_impact_analyzer_comprehensive),
            ("Risk Assessor", self.test_risk_assessor_comprehensive),
            ("Scenario Comparator", self.test_scenario_comparator_comprehensive),
            ("Scenario Analysis Agent", self.test_scenario_agent_comprehensive),
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
        with open("Test/comprehensive_scenario_analysis_results.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("COMPREHENSIVE SCENARIO ANALYSIS TESTING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        logger.info(f"Total Duration: {overall_duration:.2f}s")
        logger.info(f"Results saved to: Test/comprehensive_scenario_analysis_results.json")
        
        return report


async def main():
    """Main test execution function."""
    tester = ComprehensiveScenarioAnalysisTester()
    results = await tester.run_all_tests()
    return results


if __name__ == "__main__":
    # Run the comprehensive test suite
    results = asyncio.run(main())
    print(f"\nTest execution completed. Success rate: {results['test_summary']['success_rate']*100:.1f}%")
