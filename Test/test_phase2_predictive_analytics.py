"""
Test Phase 2.1: Predictive Analytics Implementation

This script tests the Phase 2.1 components including:
- Forecasting Engine
- Confidence Calculator
- Scenario Forecaster
- Forecast Validator
- Predictive Analytics Agent
"""

import sys
import os
import asyncio
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.predictive_analytics.forecasting_engine import ForecastingEngine
from core.predictive_analytics.confidence_calculator import ConfidenceCalculator
from core.predictive_analytics.scenario_forecaster import ScenarioForecaster
from core.predictive_analytics.forecast_validator import ForecastValidator
from agents.predictive_analytics_agent import PredictiveAnalyticsAgent
from core.models import DataType, AnalysisRequest


def generate_test_data(n_points: int = 50) -> np.ndarray:
    """Generate test time series data"""
    # Generate trend + noise
    trend = np.linspace(0, 10, n_points)
    noise = np.random.normal(0, 1, n_points)
    seasonal = 2 * np.sin(2 * np.pi * np.arange(n_points) / 12)
    
    return trend + noise + seasonal


def test_forecasting_engine():
    """Test the forecasting engine"""
    print("🧪 Testing Forecasting Engine...")
    
    try:
        # Initialize engine
        engine = ForecastingEngine()
        
        # Generate test data
        data = generate_test_data(30)
        
        # Test different model types
        models_to_test = [
            'ensemble',
            'simple_average',
            'exponential_smoothing',
            'arima'
        ]
        
        for model_type in models_to_test:
            print(f"   Testing {model_type} model...")
            
            # Generate forecast
            result = engine.forecast(
                data=data,
                model_type=model_type,
                forecast_horizon=12
            )
            
            # Verify results
            assert len(result.predictions) == 12, f"Expected 12 predictions, got {len(result.predictions)}"
            assert result.model_type == model_type, f"Expected model type {model_type}, got {result.model_type}"
            assert result.forecast_horizon == 12, f"Expected horizon 12, got {result.forecast_horizon}"
            
            print(f"   ✅ {model_type} model: completed")
        
        # Test available models
        available_models = engine.get_available_models()
        assert len(available_models) > 0, "No models available"
        
        # Test model info
        model_info = engine.get_model_info('ensemble')
        assert 'name' in model_info, "Model info missing name"
        
        print("✅ Forecasting Engine: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Forecasting Engine: FAILED - {str(e)}")
        return False


def test_confidence_calculator():
    """Test the confidence calculator"""
    print("🧪 Testing Confidence Calculator...")
    
    try:
        # Initialize calculator
        calculator = ConfidenceCalculator()
        
        # Generate test data
        historical_data = generate_test_data(30)
        predictions = np.array([100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155])
        
        # Test different confidence interval methods
        methods = ['parametric', 'bootstrap', 'empirical']
        
        for method in methods:
            print(f"   Testing {method} confidence intervals...")
            
            # Calculate confidence intervals
            ci = calculator.calculate_confidence_intervals(
                predictions=predictions,
                historical_data=historical_data,
                confidence_level=0.95,
                method=method
            )
            
            # Verify results
            assert len(ci.lower_bound) == len(predictions), "Lower bound length mismatch"
            assert len(ci.upper_bound) == len(predictions), "Upper bound length mismatch"
            assert ci.confidence_level == 0.95, "Confidence level mismatch"
            assert ci.method == method, "Method mismatch"
            
            print(f"   ✅ {method} method: completed")
        
        # Test uncertainty metrics
        uncertainty_metrics = calculator.calculate_uncertainty_metrics(
            predictions=predictions,
            historical_data=historical_data
        )
        
        assert len(uncertainty_metrics) > 0, "No uncertainty metrics calculated"
        
        # Test confidence level info
        info = calculator.get_confidence_level_info(0.95)
        assert info['confidence_level'] == 0.95, "Confidence level info mismatch"
        
        print("✅ Confidence Calculator: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Confidence Calculator: FAILED - {str(e)}")
        return False


def test_scenario_forecaster():
    """Test the scenario forecaster"""
    print("🧪 Testing Scenario Forecaster...")
    
    try:
        # Initialize forecaster
        forecaster = ScenarioForecaster()
        
        # Generate test data
        base_data = generate_test_data(30)
        
        # Test scenario generation
        print("   Testing scenario generation...")
        scenarios = forecaster.generate_scenarios(
            base_data=base_data,
            forecast_horizon=12
        )
        
        # Verify results
        assert len(scenarios) > 0, "No scenarios generated"
        
        for scenario in scenarios:
            assert len(scenario.predictions) == 12, "Scenario predictions length mismatch"
            assert scenario.probability > 0, "Scenario probability must be positive"
            assert scenario.name in ['baseline', 'optimistic', 'pessimistic'], f"Unexpected scenario: {scenario.name}"
        
        print("   ✅ Scenario generation: completed")
        
        # Test scenario comparison
        print("   Testing scenario comparison...")
        comparison = forecaster.compare_scenarios(scenarios)
        
        assert comparison.best_scenario is not None, "Best scenario not identified"
        assert comparison.worst_scenario is not None, "Worst scenario not identified"
        assert len(comparison.comparison_metrics) > 0, "No comparison metrics"
        
        print("   ✅ Scenario comparison: completed")
        
        # Test scenario templates
        templates = forecaster.get_scenario_templates()
        assert len(templates) > 0, "No scenario templates available"
        
        print("✅ Scenario Forecaster: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Scenario Forecaster: FAILED - {str(e)}")
        return False


def test_forecast_validator():
    """Test the forecast validator"""
    print("🧪 Testing Forecast Validator...")
    
    try:
        # Initialize validator
        validator = ForecastValidator()
        
        # Generate test data
        actual = generate_test_data(20)
        predicted = actual + np.random.normal(0, 2, 20)  # Add some noise
        
        # Test holdout validation
        print("   Testing holdout validation...")
        validation_result = validator.validate_forecast(
            actual=actual,
            predicted=predicted,
            validation_type='holdout'
        )
        
        # Verify results
        assert len(validation_result.accuracy_metrics) > 0, "No accuracy metrics"
        assert validation_result.model_performance in ['excellent', 'good', 'fair', 'poor'], "Invalid performance level"
        assert len(validation_result.recommendations) > 0, "No recommendations generated"
        
        print("   ✅ Holdout validation: completed")
        
        # Test performance summary
        summary = validator.get_performance_summary(validation_result)
        assert 'performance_level' in summary, "Performance summary missing level"
        assert 'key_metrics' in summary, "Performance summary missing metrics"
        
        print("✅ Forecast Validator: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Forecast Validator: FAILED - {str(e)}")
        return False


async def test_predictive_analytics_agent():
    """Test the predictive analytics agent"""
    print("🧪 Testing Predictive Analytics Agent...")
    
    try:
        # Initialize agent
        agent = PredictiveAnalyticsAgent()
        
        # Generate test data
        data = generate_test_data(30)
        
        # Test forecasting analysis
        print("   Testing forecasting analysis...")
        request = AnalysisRequest(
            data_type=DataType.TIME_SERIES,
            content=data.tolist(),
            parameters={
                'analysis_type': 'forecast',
                'model_type': 'ensemble',
                'forecast_horizon': 12,
                'confidence_level': 0.95
            }
        )
        
        result = await agent.process(request)
        
        assert result.success, f"Forecasting failed: {result.error_message}"
        assert 'forecast' in result.data, "No forecast data in result"
        assert 'confidence_intervals' in result.data, "No confidence intervals in result"
        
        print("   ✅ Forecasting analysis: completed")
        
        # Test scenario analysis
        print("   Testing scenario analysis...")
        request = AnalysisRequest(
            data_type=DataType.TIME_SERIES,
            content=data.tolist(),
            parameters={
                'analysis_type': 'scenario',
                'forecast_horizon': 12
            }
        )
        
        result = await agent.process(request)
        
        assert result.success, f"Scenario analysis failed: {result.error_message}"
        assert 'scenarios' in result.data, "No scenarios in result"
        assert 'comparison' in result.data, "No comparison in result"
        
        print("   ✅ Scenario analysis: completed")
        
        # Test validation analysis
        print("   Testing validation analysis...")
        request = AnalysisRequest(
            data_type=DataType.TIME_SERIES,
            content=data.tolist(),
            parameters={
                'analysis_type': 'validation',
                'validation_type': 'holdout'
            }
        )
        
        result = await agent.process(request)
        
        assert result.success, f"Validation failed: {result.error_message}"
        assert 'validation' in result.data, "No validation data in result"
        assert 'performance_summary' in result.data, "No performance summary in result"
        
        print("   ✅ Validation analysis: completed")
        
        # Test capabilities
        capabilities = agent.get_capabilities()
        assert 'analysis_types' in capabilities, "Capabilities missing analysis types"
        assert 'available_models' in capabilities, "Capabilities missing models"
        
        print("✅ Predictive Analytics Agent: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Predictive Analytics Agent: FAILED - {str(e)}")
        return False


async def main():
    """Run all Phase 2.1 tests"""
    print("🚀 Phase 2.1 Predictive Analytics Test Suite")
    print("=" * 50)
    
    test_results = []
    
    # Test individual components
    test_results.append(test_forecasting_engine())
    test_results.append(test_confidence_calculator())
    test_results.append(test_scenario_forecaster())
    test_results.append(test_forecast_validator())
    test_results.append(await test_predictive_analytics_agent())
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"   Total Tests: {total}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {total - passed}")
    
    if passed == total:
        print("🎉 All Phase 2.1 tests PASSED!")
        print("✅ Phase 2.1 Predictive Analytics is working correctly!")
    else:
        print("❌ Some Phase 2.1 tests FAILED!")
        print("⚠️  Please check the failed components above.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
