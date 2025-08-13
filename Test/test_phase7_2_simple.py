"""
Simple test for Phase 7.2 Advanced Analytics Components

This script tests basic import and initialization of the advanced analytics components.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_basic_imports():
    """Test basic imports of advanced analytics components"""
    print("Testing basic imports...")
    
    try:
        # Test core components
        from core.advanced_analytics.multivariate_forecasting import MultivariateForecastingEngine
        from core.advanced_analytics.causal_inference_engine import CausalInferenceEngine
        from core.advanced_analytics.scenario_analysis import ScenarioAnalysisEngine
        from core.advanced_analytics.confidence_intervals import ConfidenceIntervalCalculator
        from core.advanced_analytics.advanced_anomaly_detection import AdvancedAnomalyDetector
        from core.advanced_analytics.model_optimization import ModelOptimizer
        from core.advanced_analytics.feature_engineering import FeatureEngineer
        from core.advanced_analytics.performance_monitoring import AdvancedPerformanceMonitor
        
        print("✅ All core advanced analytics components imported successfully")
        
        # Test initialization
        engine = MultivariateForecastingEngine()
        print("✅ MultivariateForecastingEngine initialized successfully")
        
        causal_engine = CausalInferenceEngine()
        print("✅ CausalInferenceEngine initialized successfully")
        
        scenario_engine = ScenarioAnalysisEngine()
        print("✅ ScenarioAnalysisEngine initialized successfully")
        
        confidence_calc = ConfidenceIntervalCalculator()
        print("✅ ConfidenceIntervalCalculator initialized successfully")
        
        anomaly_detector = AdvancedAnomalyDetector()
        print("✅ AdvancedAnomalyDetector initialized successfully")
        
        model_optimizer = ModelOptimizer()
        print("✅ ModelOptimizer initialized successfully")
        
        feature_engineer = FeatureEngineer()
        print("✅ FeatureEngineer initialized successfully")
        
        performance_monitor = AdvancedPerformanceMonitor()
        print("✅ AdvancedPerformanceMonitor initialized successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_configuration():
    """Test configuration import"""
    print("\nTesting configuration...")
    
    try:
        from config.advanced_analytics_config import AdvancedAnalyticsConfig
        
        config = AdvancedAnalyticsConfig()
        print("✅ AdvancedAnalyticsConfig initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run basic tests"""
    print("🚀 Phase 7.2 Basic Component Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Configuration", test_configuration)
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
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Basic Phase 7.2 tests passed! Components are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
