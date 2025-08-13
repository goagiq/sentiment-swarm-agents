#!/usr/bin/env python3
"""
Simplified test script for Phase 2.2: What-If Scenario Engine Enhancement

Tests the core scenario analysis components without complex agent interactions.
"""

import sys
import os
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.scenario_analysis import (
    ScenarioBuilder, ImpactAnalyzer, RiskAssessor, ScenarioComparator
)
from core.scenario_analysis.scenario_builder import (
    ScenarioType, ParameterType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_core_components():
    """Test core scenario analysis components."""
    print("Phase 2.2: What-If Scenario Engine Enhancement - Core Test")
    print("=" * 60)
    
    try:
        # Test ScenarioBuilder
        print("\n=== Testing ScenarioBuilder ===")
        builder = ScenarioBuilder()
        
        # Create a simple scenario
        scenario = builder.create_scenario(
            name="Simple Test Scenario",
            description="A simple test scenario",
            scenario_type=ScenarioType.OPTIMISTIC,
            tags=["test", "simple"]
        )
        
        print(f"✓ Created scenario: {scenario.name}")
        
        # Add a parameter
        builder.add_parameter(
            scenario.id,
            "test_param",
            ParameterType.NUMERICAL,
            current_value=10.0,
            scenario_value=20.0,
            description="Test parameter"
        )
        
        print(f"✓ Added parameter to scenario")
        
        # Test validation
        validation = builder.validate_scenario(scenario.id)
        print(f"✓ Scenario validation: {validation['is_valid']}")
        
        # Test ImpactAnalyzer
        print("\n=== Testing ImpactAnalyzer ===")
        analyzer = ImpactAnalyzer()
        impact_analysis = analyzer.analyze_impact(scenario)
        
        print(f"✓ Impact analysis completed")
        print(f"  Overall impact score: {impact_analysis.overall_impact_score:.2f}")
        print(f"  Overall risk level: {impact_analysis.overall_risk_level}")
        
        # Test RiskAssessor
        print("\n=== Testing RiskAssessor ===")
        assessor = RiskAssessor()
        risk_assessment = assessor.assess_risk(scenario)
        
        print(f"✓ Risk assessment completed")
        print(f"  Overall risk score: {risk_assessment.overall_risk_score:.2f}")
        print(f"  Overall risk level: {risk_assessment.overall_risk_level.value}")
        
        # Test ScenarioComparator
        print("\n=== Testing ScenarioComparator ===")
        comparator = ScenarioComparator()
        
        # Create another scenario for comparison
        scenario2 = builder.create_scenario(
            name="Second Test Scenario",
            description="Another test scenario",
            scenario_type=ScenarioType.PESSIMISTIC,
            tags=["test", "second"]
        )
        
        builder.add_parameter(
            scenario2.id,
            "test_param",
            ParameterType.NUMERICAL,
            current_value=10.0,
            scenario_value=5.0,
            description="Test parameter"
        )
        
        # Compare scenarios
        comparison_result = comparator.compare_scenarios([scenario, scenario2])
        
        print(f"✓ Scenario comparison completed")
        print(f"  Scenarios compared: {len(comparison_result.scenarios_compared)}")
        print(f"  Comparison type: {comparison_result.comparison_type.value}")
        
        print("\n" + "=" * 60)
        print("✓ All Phase 2.2 core components working correctly!")
        print("✓ What-If Scenario Engine Enhancement is functional")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_core_components()
    sys.exit(0 if success else 1)
