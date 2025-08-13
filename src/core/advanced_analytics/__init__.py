"""
Advanced Analytics Module

This module provides enhanced predictive analytics capabilities including:
- Multi-variate forecasting
- Causal inference engine
- Scenario analysis
- Advanced anomaly detection
- Model optimization
- Feature engineering
- Performance monitoring

All components follow the Design Framework standards and integrate with existing MCP tools.
"""

from .multivariate_forecasting import MultivariateForecastingEngine
from .causal_inference_engine import CausalInferenceEngine
from .scenario_analysis import ScenarioAnalysisEngine
from .confidence_intervals import ConfidenceIntervalCalculator
from .advanced_anomaly_detection import AdvancedAnomalyDetector
from .model_optimization import ModelOptimizer
from .feature_engineering import FeatureEngineer
from .performance_monitoring import AdvancedPerformanceMonitor

__all__ = [
    'MultivariateForecastingEngine',
    'CausalInferenceEngine', 
    'ScenarioAnalysisEngine',
    'ConfidenceIntervalCalculator',
    'AdvancedAnomalyDetector',
    'ModelOptimizer',
    'FeatureEngineer',
    'AdvancedPerformanceMonitor'
]

__version__ = "1.0.0"
__author__ = "Advanced Analytics Team"

