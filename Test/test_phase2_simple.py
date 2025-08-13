"""
Simple Test for Phase 2.1: Predictive Analytics Components

This script tests the core predictive analytics components directly.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_forecasting_engine():
    """Test the forecasting engine directly"""
    print("🧪 Testing Forecasting Engine...")
    
    try:
        # Import directly
        from core.predictive_analytics.forecasting_engine import ForecastingEngine
        
        # Initialize engine
        engine = ForecastingEngine()
        
        # Generate test data
        data = np.random.normal(100, 10, 30)  # 30 data points
        
        # Test ensemble forecasting
        result = engine.forecast(
            data=data,
            model_type='ensemble',
            forecast_horizon=12
        )
        
        # Verify results
        assert len(result.predictions) == 12, f"Expected 12 predictions, got {len(result.predictions)}"
        assert result.model_type == 'ensemble', f"Expected ensemble model, got {result.model_type}"
        
        print("✅ Forecasting Engine: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Forecasting Engine: FAILED - {str(e)}")
        return False


def test_confidence_calculator():
    """Test the confidence calculator directly"""
    print("🧪 Testing Confidence Calculator...")
    
    try:
        # Import directly
        from core.predictive_analytics.confidence_calculator import ConfidenceCalculator
        
        # Initialize calculator
        calculator = ConfidenceCalculator()
        
        # Generate test data
        historical_data = np.random.normal(100, 10, 30)
        predictions = np.array([100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155])
        
        # Test parametric confidence intervals
        ci = calculator.calculate_confidence_intervals(
            predictions=predictions,
            historical_data=historical_data,
            confidence_level=0.95,
            method='parametric'
        )
        
        # Verify results
        assert len(ci.lower_bound) == len(predictions), "Lower bound length mismatch"
        assert len(ci.upper_bound) == len(predictions), "Upper bound length mismatch"
        assert ci.confidence_level == 0.95, "Confidence level mismatch"
        
        print("✅ Confidence Calculator: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Confidence Calculator: FAILED - {str(e)}")
        return False


def test_scenario_forecaster():
    """Test the scenario forecaster directly"""
    print("🧪 Testing Scenario Forecaster...")
    
    try:
        # Import directly
        from core.predictive_analytics.scenario_forecaster import ScenarioForecaster
        
        # Initialize forecaster
        forecaster = ScenarioForecaster()
        
        # Generate test data
        base_data = np.random.normal(100, 10, 30)
        
        # Test scenario generation
        scenarios = forecaster.generate_scenarios(
            base_data=base_data,
            forecast_horizon=12
        )
        
        # Verify results
        assert len(scenarios) > 0, "No scenarios generated"
        
        for scenario in scenarios:
            assert len(scenario.predictions) == 12, "Scenario predictions length mismatch"
            assert scenario.probability > 0, "Scenario probability must be positive"
        
        print("✅ Scenario Forecaster: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Scenario Forecaster: FAILED - {str(e)}")
        return False


def test_forecast_validator():
    """Test the forecast validator directly"""
    print("🧪 Testing Forecast Validator...")
    
    try:
        # Import directly
        from core.predictive_analytics.forecast_validator import ForecastValidator
        
        # Initialize validator
        validator = ForecastValidator()
        
        # Generate test data
        actual = np.random.normal(100, 10, 20)
        predicted = actual + np.random.normal(0, 2, 20)  # Add some noise
        
        # Test holdout validation
        validation_result = validator.validate_forecast(
            actual=actual,
            predicted=predicted,
            validation_type='holdout'
        )
        
        # Verify results
        assert len(validation_result.accuracy_metrics) > 0, "No accuracy metrics"
        assert validation_result.model_performance in ['excellent', 'good', 'fair', 'poor'], "Invalid performance level"
        
        print("✅ Forecast Validator: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Forecast Validator: FAILED - {str(e)}")
        return False


def main():
    """Run all Phase 2.1 tests"""
    print("🚀 Phase 2.1 Predictive Analytics Simple Test Suite")
    print("=" * 50)
    
    test_results = []
    
    # Test individual components
    test_results.append(test_forecasting_engine())
    test_results.append(test_confidence_calculator())
    test_results.append(test_scenario_forecaster())
    test_results.append(test_forecast_validator())
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"   Total Tests: {total}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {total - passed}")
    
    if passed == total:
        print("🎉 All Phase 2.1 core components PASSED!")
        print("✅ Phase 2.1 Predictive Analytics core components are working!")
    else:
        print("❌ Some Phase 2.1 tests FAILED!")
        print("⚠️  Please check the failed components above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
