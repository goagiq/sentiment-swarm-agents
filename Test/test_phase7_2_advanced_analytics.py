"""
Test script for Phase 7.2 Advanced Analytics Components

This script tests the implementation of enhanced predictive analytics features
including multivariate forecasting, causal inference, scenario analysis, and
anomaly detection.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

# Import modules directly
try:
    from core.advanced_analytics.multivariate_forecasting import MultivariateForecastingEngine
    from core.advanced_analytics.causal_inference_engine import CausalInferenceEngine
    from core.advanced_analytics.scenario_analysis import ScenarioAnalysisEngine
    from core.advanced_analytics.confidence_intervals import ConfidenceIntervalCalculator
    from core.advanced_analytics.advanced_anomaly_detection import AdvancedAnomalyDetector
    from core.advanced_analytics.model_optimization import ModelOptimizer
    from core.advanced_analytics.feature_engineering import FeatureEngineer
    from core.advanced_analytics.performance_monitoring import AdvancedPerformanceMonitor
    from agents.advanced_forecasting_agent import AdvancedForecastingAgent
    from agents.causal_analysis_agent import CausalAnalysisAgent
    from agents.anomaly_detection_agent import AnomalyDetectionAgent
    from config.advanced_analytics_config import AdvancedAnalyticsConfig
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    IMPORTS_SUCCESSFUL = False

def test_advanced_analytics_imports():
    """Test that all advanced analytics components can be imported"""
    print("Testing advanced analytics imports...")
    
    if IMPORTS_SUCCESSFUL:
        print("✅ All advanced analytics components imported successfully")
        return True
    else:
        print("❌ Import failed")
        return False

def test_multivariate_forecasting():
    """Test multivariate forecasting functionality"""
    print("\nTesting multivariate forecasting...")
    
    if not IMPORTS_SUCCESSFUL:
        print("❌ Skipping test due to import failure")
        return False
    
    try:
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'sales': np.random.normal(1000, 200, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 100,
            'temperature': np.random.normal(20, 10, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 5,
            'advertising': np.random.normal(500, 100, len(dates))
        })
        data.set_index('timestamp', inplace=True)
        
        # Initialize engine
        engine = MultivariateForecastingEngine()
        
        # Test data preparation
        features, targets = engine.prepare_multivariate_data(
            data, 
            target_columns=['sales', 'temperature'],
            lag_features=True,
            max_lags=7
        )
        
        print(f"✅ Data preparation successful: {features.shape[0]} samples, {features.shape[1]} features")
        
        # Test causal relationship detection
        causal_relationships = engine.detect_causal_relationships(
            data, 
            target_columns=['sales', 'temperature']
        )
        
        print(f"✅ Causal relationship detection successful: {len(causal_relationships)} relationships found")
        
        return True
        
    except Exception as e:
        print(f"❌ Multivariate forecasting test failed: {e}")
        return False

def test_anomaly_detection():
    """Test anomaly detection functionality"""
    print("\nTesting anomaly detection...")
    
    if not IMPORTS_SUCCESSFUL:
        print("❌ Skipping test due to import failure")
        return False
    
    try:
        # Create sample data with anomalies
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 1000)
        anomaly_data = np.random.normal(5, 1, 50)  # Anomalies
        data = pd.DataFrame({
            'value': np.concatenate([normal_data, anomaly_data])
        })
        
        # Initialize detector
        detector = AdvancedAnomalyDetector()
        
        # Test statistical anomaly detection
        result = detector.detect_statistical_outliers(data, method='zscore', threshold=3.0)
        
        print(f"✅ Statistical anomaly detection successful: {len(result.anomalies)} anomalies found")
        
        # Test multivariate anomaly detection
        multivariate_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        })
        
        # Add some anomalies
        multivariate_data.iloc[95:100, 0] = 5  # Anomalies in feature1
        
        results = detector.detect_multivariate_anomalies(
            multivariate_data,
            methods=['isolation_forest', 'statistical'],
            ensemble=True
        )
        
        print(f"✅ Multivariate anomaly detection successful: {len(results)} methods tested")
        
        return True
        
    except Exception as e:
        print(f"❌ Anomaly detection test failed: {e}")
        return False

def test_causal_inference():
    """Test causal inference functionality"""
    print("\nTesting causal inference...")
    
    if not IMPORTS_SUCCESSFUL:
        print("❌ Skipping test due to import failure")
        return False
    
    try:
        # Create sample data with causal relationships
        np.random.seed(42)
        n_samples = 1000
        
        # Create causal relationship: X -> Y
        X = np.random.normal(0, 1, n_samples)
        Y = 0.7 * X + np.random.normal(0, 0.3, n_samples)
        Z = np.random.normal(0, 1, n_samples)  # Independent variable
        
        data = pd.DataFrame({
            'X': X,
            'Y': Y,
            'Z': Z
        })
        
        # Initialize engine
        engine = CausalInferenceEngine()
        
        # Test correlation-based causality
        relationships = engine.detect_correlation_causality(
            data,
            variables=['X', 'Y', 'Z'],
            method='pearson'
        )
        
        print(f"✅ Correlation causality detection successful: {len(relationships)} relationships found")
        
        # Test comprehensive causal analysis
        result = engine.perform_causal_analysis(
            data,
            methods=['correlation', 'conditional_independence'],
            variables=['X', 'Y', 'Z']
        )
        
        print(f"✅ Comprehensive causal analysis successful: {len(result.relationships)} relationships found")
        
        return True
        
    except Exception as e:
        print(f"❌ Causal inference test failed: {e}")
        return False

def test_scenario_analysis():
    """Test scenario analysis functionality"""
    print("\nTesting scenario analysis...")
    
    if not IMPORTS_SUCCESSFUL:
        print("❌ Skipping test due to import failure")
        return False
    
    try:
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        data = pd.DataFrame({
            'timestamp': dates,
            'revenue': np.random.normal(10000, 2000, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 2000,
            'costs': np.random.normal(6000, 1000, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 500
        })
        data.set_index('timestamp', inplace=True)
        
        # Initialize engine
        engine = ScenarioAnalysisEngine()
        
        # Create scenarios
        baseline_scenario = engine.create_scenario(
            name="baseline",
            description="Baseline scenario with current trends",
            parameters={'revenue_trend': 0.0, 'costs_trend': 0.0},
            probability=0.6
        )
        
        optimistic_scenario = engine.create_scenario(
            name="optimistic",
            description="Optimistic scenario with growth",
            parameters={'revenue_trend': 0.1, 'costs_trend': -0.05},
            probability=0.2
        )
        
        print(f"✅ Scenario creation successful: {len(engine.scenarios)} scenarios created")
        
        # Test default scenario creation
        default_scenarios = engine.create_default_scenarios(data)
        
        print(f"✅ Default scenario creation successful: {len(default_scenarios)} scenarios created")
        
        return True
        
    except Exception as e:
        print(f"❌ Scenario analysis test failed: {e}")
        return False

def test_confidence_intervals():
    """Test confidence interval calculation"""
    print("\nTesting confidence intervals...")
    
    if not IMPORTS_SUCCESSFUL:
        print("❌ Skipping test due to import failure")
        return False
    
    try:
        # Create sample data
        np.random.seed(42)
        data = pd.DataFrame({
            'values': np.random.normal(100, 15, 1000)
        })
        
        # Initialize calculator
        calculator = ConfidenceIntervalCalculator()
        
        # Test parametric confidence intervals
        interval = calculator.calculate_parametric_intervals(
            data['values'],
            confidence_level=0.95,
            distribution='normal'
        )
        
        print(f"✅ Parametric confidence interval calculation successful")
        print(f"   Interval: [{interval.lower_bound:.2f}, {interval.upper_bound:.2f}]")
        print(f"   Confidence level: {interval.confidence_level}")
        
        # Test bootstrap confidence intervals
        bootstrap_interval = calculator.calculate_bootstrap_intervals(
            data['values'],
            confidence_level=0.95,
            n_bootstrap=100
        )
        
        print(f"✅ Bootstrap confidence interval calculation successful")
        print(f"   Interval: [{bootstrap_interval.lower_bound:.2f}, {bootstrap_interval.upper_bound:.2f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Confidence interval test failed: {e}")
        return False

def test_agents():
    """Test advanced analytics agents"""
    print("\nTesting advanced analytics agents...")
    
    if not IMPORTS_SUCCESSFUL:
        print("❌ Skipping test due to import failure")
        return False
    
    try:
        # Test agent initialization
        forecasting_agent = AdvancedForecastingAgent()
        causal_agent = CausalAnalysisAgent()
        anomaly_agent = AnomalyDetectionAgent()
        
        print(f"✅ Advanced forecasting agent initialized: {forecasting_agent.agent_id}")
        print(f"✅ Causal analysis agent initialized: {causal_agent.agent_id}")
        print(f"✅ Anomaly detection agent initialized: {anomaly_agent.agent_id}")
        
        # Test agent capabilities
        print(f"Forecasting agent capabilities: {forecasting_agent.metadata.get('capabilities', [])}")
        print(f"Causal agent capabilities: {causal_agent.metadata.get('capabilities', [])}")
        print(f"Anomaly agent capabilities: {anomaly_agent.metadata.get('capabilities', [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

def main():
    """Run all Phase 7.2 tests"""
    print("🚀 Phase 7.2 Advanced Analytics Component Tests")
    print("=" * 60)
    
    tests = [
        ("Advanced Analytics Imports", test_advanced_analytics_imports),
        ("Multivariate Forecasting", test_multivariate_forecasting),
        ("Anomaly Detection", test_anomaly_detection),
        ("Causal Inference", test_causal_inference),
        ("Scenario Analysis", test_scenario_analysis),
        ("Confidence Intervals", test_confidence_intervals),
        ("Advanced Analytics Agents", test_agents)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 7.2 tests passed! Advanced analytics components are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
