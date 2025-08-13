"""
Debug script to identify which import is causing the relative import issue
"""

import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..', 'src')
sys.path.insert(0, src_dir)

print("Testing imports one by one...")

# Test each import individually
try:
    print("Testing multivariate_forecasting...")
    from core.advanced_analytics.multivariate_forecasting import MultivariateForecastingEngine
    print("✅ multivariate_forecasting imported successfully")
except Exception as e:
    print(f"❌ multivariate_forecasting failed: {e}")

try:
    print("Testing causal_inference_engine...")
    from core.advanced_analytics.causal_inference_engine import CausalInferenceEngine
    print("✅ causal_inference_engine imported successfully")
except Exception as e:
    print(f"❌ causal_inference_engine failed: {e}")

try:
    print("Testing scenario_analysis...")
    from core.advanced_analytics.scenario_analysis import ScenarioAnalysisEngine
    print("✅ scenario_analysis imported successfully")
except Exception as e:
    print(f"❌ scenario_analysis failed: {e}")

try:
    print("Testing confidence_intervals...")
    from core.advanced_analytics.confidence_intervals import ConfidenceIntervalCalculator
    print("✅ confidence_intervals imported successfully")
except Exception as e:
    print(f"❌ confidence_intervals failed: {e}")

try:
    print("Testing advanced_anomaly_detection...")
    from core.advanced_analytics.advanced_anomaly_detection import AdvancedAnomalyDetector
    print("✅ advanced_anomaly_detection imported successfully")
except Exception as e:
    print(f"❌ advanced_anomaly_detection failed: {e}")

try:
    print("Testing model_optimization...")
    from core.advanced_analytics.model_optimization import ModelOptimizer
    print("✅ model_optimization imported successfully")
except Exception as e:
    print(f"❌ model_optimization failed: {e}")

try:
    print("Testing feature_engineering...")
    from core.advanced_analytics.feature_engineering import FeatureEngineer
    print("✅ feature_engineering imported successfully")
except Exception as e:
    print(f"❌ feature_engineering failed: {e}")

try:
    print("Testing performance_monitoring...")
    from core.advanced_analytics.performance_monitoring import AdvancedPerformanceMonitor
    print("✅ performance_monitoring imported successfully")
except Exception as e:
    print(f"❌ performance_monitoring failed: {e}")

try:
    print("Testing advanced_forecasting_agent...")
    from agents.advanced_forecasting_agent import AdvancedForecastingAgent
    print("✅ advanced_forecasting_agent imported successfully")
except Exception as e:
    print(f"❌ advanced_forecasting_agent failed: {e}")

try:
    print("Testing causal_analysis_agent...")
    from agents.causal_analysis_agent import CausalAnalysisAgent
    print("✅ causal_analysis_agent imported successfully")
except Exception as e:
    print(f"❌ causal_analysis_agent failed: {e}")

try:
    print("Testing anomaly_detection_agent...")
    from agents.anomaly_detection_agent import AnomalyDetectionAgent
    print("✅ anomaly_detection_agent imported successfully")
except Exception as e:
    print(f"❌ anomaly_detection_agent failed: {e}")

try:
    print("Testing advanced_analytics_config...")
    from config.advanced_analytics_config import AdvancedAnalyticsConfig
    print("✅ advanced_analytics_config imported successfully")
except Exception as e:
    print(f"❌ advanced_analytics_config failed: {e}")

print("\nDebug complete.")
