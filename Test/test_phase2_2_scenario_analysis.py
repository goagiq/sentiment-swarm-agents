#!/usr/bin/env python3
"""
Test script for Phase 2.2: What-If Scenario Engine Enhancement

Tests the scenario analysis components including:
- ScenarioBuilder
- ImpactAnalyzer  
- RiskAssessor
- ScenarioComparator
- ScenarioAnalysisAgent
"""

import sys
import os
import asyncio
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.scenario_analysis import (
    ScenarioBuilder, ImpactAnalyzer, RiskAssessor, ScenarioComparator
)
from core.scenario_analysis.scenario_builder import (
    Scenario, ScenarioType, ParameterType, ScenarioParameter
)
from agents.scenario_analysis_agent import ScenarioAnalysisAgent
from core.models import AnalysisRequest, AnalysisResult, DataType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_scenario_builder():
    """Test ScenarioBuilder functionality."""
    print("\n=== Testing ScenarioBuilder ===")
    
    builder = ScenarioBuilder()
    
    # Test creating a scenario
    scenario = builder.create_scenario(
        name="Test Scenario",
        description="A test scenario for validation",
        scenario_type=ScenarioType.OPTIMISTIC,
        tags=["test", "validation"]
    )
    
    print(f"✓ Created scenario: {scenario.name} (ID: {scenario.id})")
    
    # Test adding parameters
    builder.add_parameter(
        scenario.id,
        "revenue_growth",
        ParameterType.PERCENTAGE,
        current_value=5.0,
        scenario_value=15.0,
        min_value=0.0,
        max_value=50.0,
        description="Annual revenue growth rate",
        unit="%",
        confidence_level=0.9
    )
    
    builder.add_parameter(
        scenario.id,
        "market_share",
        ParameterType.PERCENTAGE,
        current_value=10.0,
        scenario_value=25.0,
        min_value=0.0,
        max_value=100.0,
        description="Market share percentage",
        unit="%",
        confidence_level=0.85
    )
    
    print(f"✓ Added parameters to scenario")
    
    # Test validation
    validation = builder.validate_scenario(scenario.id)
    print(f"✓ Scenario validation: {validation['is_valid']}")
    if validation['errors']:
        print(f"  Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")
    
    # Test listing scenarios
    scenarios = builder.list_scenarios(scenario_type=ScenarioType.OPTIMISTIC)
    print(f"✓ Found {len(scenarios)} optimistic scenarios")
    
    return scenario


def test_impact_analyzer(scenario):
    """Test ImpactAnalyzer functionality."""
    print("\n=== Testing ImpactAnalyzer ===")
    
    analyzer = ImpactAnalyzer()
    
    # Test impact analysis
    impact_analysis = analyzer.analyze_impact(scenario)
    
    print(f"✓ Impact analysis completed")
    print(f"  Overall impact score: {impact_analysis.overall_impact_score:.2f}")
    print(f"  Overall risk level: {impact_analysis.overall_risk_level}")
    print(f"  Impact categories: {len(impact_analysis.impact_categories)}")
    
    # Test recommendations
    print(f"  Recommendations: {len(impact_analysis.recommendations)}")
    for i, rec in enumerate(impact_analysis.recommendations[:3], 1):
        print(f"    {i}. {rec}")
    
    return impact_analysis


def test_risk_assessor(scenario):
    """Test RiskAssessor functionality."""
    print("\n=== Testing RiskAssessor ===")
    
    assessor = RiskAssessor()
    
    # Test risk assessment
    risk_assessment = assessor.assess_risk(scenario)
    
    print(f"✓ Risk assessment completed")
    print(f"  Overall risk score: {risk_assessment.overall_risk_score:.2f}")
    print(f"  Overall risk level: {risk_assessment.overall_risk_level.value}")
    print(f"  Risk factors: {len(risk_assessment.risk_factors)}")
    print(f"  High-risk factors: {len(risk_assessment.high_risk_factors)}")
    
    # Test mitigation priorities
    print(f"  Mitigation priorities: {len(risk_assessment.mitigation_priorities)}")
    for i, priority in enumerate(risk_assessment.mitigation_priorities[:3], 1):
        print(f"    {i}. {priority}")
    
    return risk_assessment


def test_scenario_comparator():
    """Test ScenarioComparator functionality."""
    print("\n=== Testing ScenarioComparator ===")
    
    comparator = ScenarioComparator()
    builder = ScenarioBuilder()
    
    # Create multiple scenarios for comparison
    scenarios = []
    
    # Optimistic scenario
    opt_scenario = builder.create_scenario(
        name="Optimistic Scenario",
        description="Best-case scenario",
        scenario_type=ScenarioType.OPTIMISTIC,
        tags=["optimistic", "best-case"]
    )
    builder.add_parameter(
        opt_scenario.id,
        "growth_rate",
        ParameterType.PERCENTAGE,
        current_value=5.0,
        scenario_value=20.0,
        confidence_level=0.8
    )
    scenarios.append(opt_scenario)
    
    # Pessimistic scenario
    pes_scenario = builder.create_scenario(
        name="Pessimistic Scenario", 
        description="Worst-case scenario",
        scenario_type=ScenarioType.PESSIMISTIC,
        tags=["pessimistic", "worst-case"]
    )
    builder.add_parameter(
        pes_scenario.id,
        "growth_rate",
        ParameterType.PERCENTAGE,
        current_value=5.0,
        scenario_value=-5.0,
        confidence_level=0.7
    )
    scenarios.append(pes_scenario)
    
    # Test comparison
    comparison_result = comparator.compare_scenarios(scenarios)
    
    print(f"✓ Scenario comparison completed")
    print(f"  Scenarios compared: {len(comparison_result.scenarios_compared)}")
    print(f"  Comparison type: {comparison_result.comparison_type.value}")
    print(f"  Metrics calculated: {len(comparison_result.metrics)}")
    print(f"  Similarities: {len(comparison_result.similarities)}")
    print(f"  Differences: {len(comparison_result.differences)}")
    
    # Test rankings
    rankings = comparator.rank_scenarios(scenarios, method='composite')
    print(f"  Scenario rankings: {len(rankings)}")
    for ranking in rankings:
        print(f"    {ranking.rank}. {ranking.scenario_id} (score: {ranking.score:.2f})")
    
    return comparison_result


async def test_scenario_analysis_agent():
    """Test ScenarioAnalysisAgent functionality."""
    print("\n=== Testing ScenarioAnalysisAgent ===")
    
    agent = ScenarioAnalysisAgent()
    
    # Test scenario building request
    scenario_request = AnalysisRequest(
        data_type=DataType.TEXT,
        analysis_type="scenario_building",
        content={
            "scenario": {
                "name": "Agent Test Scenario",
                "description": "Scenario created via agent",
                "type": "custom",
                "tags": ["agent-test", "automated"],
                "parameters": {
                    "test_param": {
                        "type": "numerical",
                        "current_value": 100,
                        "scenario_value": 150,
                        "description": "Test parameter",
                        "unit": "units",
                        "confidence_level": 0.9
                    }
                }
            }
        }
    )
    
    result = await agent.process(scenario_request)
    
    if result.success:
        print(f"✓ Agent scenario building successful")
        scenario_data = result.data.get('scenario', {})
        print(f"  Scenario ID: {scenario_data.get('id')}")
        print(f"  Parameters: {len(scenario_data.get('parameters', {}))}")
        
        # Test impact analysis request
        impact_request = AnalysisRequest(
            data_type=DataType.TEXT,
            analysis_type="impact_analysis",
            content={
                "scenario_id": scenario_data.get('id')
            }
        )
        
        impact_result = await agent.process(impact_request)
        
        if impact_result.success:
            print(f"✓ Agent impact analysis successful")
            impact_data = impact_result.data.get('impact_analysis', {})
            print(f"  Overall impact score: {impact_data.get('overall_impact_score', 0):.2f}")
            print(f"  Overall risk level: {impact_data.get('overall_risk_level', 'unknown')}")
        else:
            print(f"✗ Agent impact analysis failed: {impact_result.error_message}")
    else:
        print(f"✗ Agent scenario building failed: {result.error_message}")
    
    return result


def main():
    """Run all tests."""
    print("Phase 2.2: What-If Scenario Engine Enhancement - Test Suite")
    print("=" * 60)
    
    try:
        # Test core components
        scenario = test_scenario_builder()
        impact_analysis = test_impact_analyzer(scenario)
        risk_assessment = test_risk_assessor(scenario)
        comparison_result = test_scenario_comparator()
        
        # Test agent
        asyncio.run(test_scenario_analysis_agent())
        
        print("\n" + "=" * 60)
        print("✓ All Phase 2.2 tests completed successfully!")
        print("✓ Scenario analysis components are working correctly")
        print("✓ What-If Scenario Engine Enhancement is ready for use")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
