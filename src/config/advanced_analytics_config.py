"""
Advanced Analytics Configuration

Configuration settings for Phase 7.2 advanced analytics features including
multivariate forecasting, causal inference, scenario analysis, and performance monitoring.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class ForecastingMethod(Enum):
    """Supported forecasting methods"""
    ENSEMBLE = "ensemble"
    VAR = "var"
    LSTM = "lstm"
    PROPHET = "prophet"
    ARIMA = "arima"


class CausalMethod(Enum):
    """Supported causal inference methods"""
    GRANGER_CAUSALITY = "granger_causality"
    CORRELATION = "correlation"
    CONDITIONAL_INDEPENDENCE = "conditional_independence"


class AnomalyMethod(Enum):
    """Supported anomaly detection methods"""
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    STATISTICAL = "statistical"


class OptimizationMethod(Enum):
    """Supported optimization methods"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"


@dataclass
class MultivariateForecastingConfig:
    """Configuration for multivariate forecasting"""
    default_method: ForecastingMethod = ForecastingMethod.ENSEMBLE
    default_forecast_horizon: int = 12
    max_lags: int = 12
    confidence_level: float = 0.95
    validation_split: float = 0.2
    random_state: int = 42
    
    # Ensemble configuration
    ensemble_methods: List[str] = field(default_factory=lambda: [
        'random_forest', 'gradient_boosting', 'ridge', 'lasso'
    ])
    
    # VAR configuration
    var_max_lags: int = 12
    var_trend: str = 'c'
    
    # Feature engineering
    create_lag_features: bool = True
    create_rolling_features: bool = True
    create_seasonal_features: bool = True
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 30])


@dataclass
class CausalInferenceConfig:
    """Configuration for causal inference"""
    default_methods: List[CausalMethod] = field(default_factory=lambda: [
        CausalMethod.CORRELATION,
        CausalMethod.GRANGER_CAUSALITY,
        CausalMethod.CONDITIONAL_INDEPENDENCE
    ])
    significance_level: float = 0.05
    max_lag: int = 10
    min_relationship_strength: float = 0.1
    
    # Granger causality configuration
    granger_max_lag: int = 5
    granger_verbose: bool = False
    
    # Correlation configuration
    correlation_method: str = 'pearson'
    correlation_threshold: float = 0.3


@dataclass
class ScenarioAnalysisConfig:
    """Configuration for scenario analysis"""
    default_scenarios: List[str] = field(default_factory=lambda: [
        'baseline', 'optimistic', 'pessimistic'
    ])
    
    # Impact thresholds
    impact_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.05,
        'medium': 0.15,
        'high': 0.30
    })
    
    # Scenario parameters
    baseline_probability: float = 0.6
    optimistic_probability: float = 0.2
    pessimistic_probability: float = 0.2
    
    # Sensitivity analysis
    sensitivity_variations: List[float] = field(default_factory=lambda: [0.9, 1.1])
    sensitivity_threshold: float = 0.1


@dataclass
class ConfidenceIntervalConfig:
    """Configuration for confidence intervals"""
    default_confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    default_method: str = 'parametric'
    
    # Method-specific configurations
    parametric_distribution: str = 'normal'
    bootstrap_n_iter: int = 1000
    
    # Thresholds
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.80,
        'medium': 0.90,
        'high': 0.95,
        'very_high': 0.99
    })


@dataclass
class AnomalyDetectionConfig:
    """Configuration for anomaly detection"""
    default_methods: List[AnomalyMethod] = field(default_factory=lambda: [
        AnomalyMethod.ISOLATION_FOREST,
        AnomalyMethod.ONE_CLASS_SVM,
        AnomalyMethod.STATISTICAL
    ])
    contamination: float = 0.1
    threshold_method: str = 'percentile'
    
    # Severity thresholds
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.3,
        'medium': 0.6,
        'high': 0.8
    })
    
    # Method-specific configurations
    isolation_forest_n_estimators: int = 100
    one_class_svm_kernel: str = 'rbf'
    one_class_svm_nu: float = 0.1
    lof_n_neighbors: int = 20
    statistical_threshold: float = 3.0


@dataclass
class ModelOptimizationConfig:
    """Configuration for model optimization"""
    default_method: OptimizationMethod = OptimizationMethod.GRID_SEARCH
    cv_folds: int = 5
    scoring: str = 'neg_mean_squared_error'
    n_iter: int = 100
    
    # Cross-validation
    cv_strategy: str = 'kfold'
    cv_shuffle: bool = True
    
    # Feature selection
    max_features: int = 20
    feature_selection_method: str = 'recursive'
    
    # Ensemble configuration
    ensemble_method: str = 'voting'
    ensemble_cv: int = 5


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering"""
    max_features: int = 100
    correlation_threshold: float = 0.95
    feature_importance_threshold: float = 0.01
    
    # Time series features
    lag_periods: List[int] = field(default_factory=lambda: [1, 3, 7, 14, 30])
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 30])
    
    # Interaction features
    max_interactions: int = 10
    interaction_methods: List[str] = field(default_factory=lambda: [
        'multiplication', 'division', 'addition', 'subtraction'
    ])
    
    # Scaling methods
    scaling_methods: List[str] = field(default_factory=lambda: [
        'standard', 'minmax', 'robust'
    ])
    
    # Dimensionality reduction
    reduction_methods: List[str] = field(default_factory=lambda: [
        'pca', 'feature_selection'
    ])
    max_components: int = 10


@dataclass
class PerformanceMonitoringConfig:
    """Configuration for performance monitoring"""
    monitoring_interval: int = 60  # seconds
    metric_history_size: int = 1000
    
    # Alert thresholds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'response_time': 5.0,
        'error_rate': 0.05,
        'memory_usage': 0.8,
        'cpu_usage': 0.8
    })
    
    # Report generation
    report_interval: int = 3600  # seconds
    auto_generate_reports: bool = True
    
    # Metric types
    metric_types: List[str] = field(default_factory=lambda: [
        'system', 'model_performance', 'prediction_latency', 'data_quality'
    ])


@dataclass
class AdvancedAnalyticsConfig:
    """Main configuration for advanced analytics"""
    
    # Component configurations
    multivariate_forecasting: MultivariateForecastingConfig = field(
        default_factory=MultivariateForecastingConfig
    )
    causal_inference: CausalInferenceConfig = field(
        default_factory=CausalInferenceConfig
    )
    scenario_analysis: ScenarioAnalysisConfig = field(
        default_factory=ScenarioAnalysisConfig
    )
    confidence_intervals: ConfidenceIntervalConfig = field(
        default_factory=ConfidenceIntervalConfig
    )
    anomaly_detection: AnomalyDetectionConfig = field(
        default_factory=AnomalyDetectionConfig
    )
    model_optimization: ModelOptimizationConfig = field(
        default_factory=ModelOptimizationConfig
    )
    feature_engineering: FeatureEngineeringConfig = field(
        default_factory=FeatureEngineeringConfig
    )
    performance_monitoring: PerformanceMonitoringConfig = field(
        default_factory=PerformanceMonitoringConfig
    )
    
    # General settings
    enable_logging: bool = True
    log_level: str = 'INFO'
    cache_results: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Performance settings
    max_workers: int = 4
    chunk_size: int = 1000
    memory_limit: str = '2GB'
    
    # Integration settings
    mcp_integration: bool = True
    api_integration: bool = True
    dashboard_integration: bool = True
    
    # Export settings
    export_formats: List[str] = field(default_factory=lambda: ['json', 'csv', 'excel'])
    export_path: str = 'exports/advanced_analytics'
    
    # Validation settings
    validate_inputs: bool = True
    validate_outputs: bool = True
    strict_mode: bool = False


# Default configuration instance
DEFAULT_ADVANCED_ANALYTICS_CONFIG = AdvancedAnalyticsConfig()


def get_advanced_analytics_config(config_path: Optional[str] = None) -> AdvancedAnalyticsConfig:
    """
    Get advanced analytics configuration
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        AdvancedAnalyticsConfig instance
    """
    if config_path:
        # Load from file if provided
        try:
            import json
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Create config from data
            config = AdvancedAnalyticsConfig(**config_data)
            return config
            
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            print("Using default configuration")
    
    return DEFAULT_ADVANCED_ANALYTICS_CONFIG


def save_advanced_analytics_config(
    config: AdvancedAnalyticsConfig,
    config_path: str
) -> None:
    """
    Save advanced analytics configuration to file
    
    Args:
        config: Configuration to save
        config_path: Path to save configuration
    """
    try:
        import json
        from dataclasses import asdict
        
        config_data = asdict(config)
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
            
        print(f"Configuration saved to {config_path}")
        
    except Exception as e:
        print(f"Error saving configuration: {e}")


def validate_advanced_analytics_config(config: AdvancedAnalyticsConfig) -> List[str]:
    """
    Validate advanced analytics configuration
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Validate forecasting config
    if config.multivariate_forecasting.default_forecast_horizon <= 0:
        errors.append("Forecast horizon must be positive")
    
    if config.multivariate_forecasting.confidence_level <= 0 or config.multivariate_forecasting.confidence_level >= 1:
        errors.append("Confidence level must be between 0 and 1")
    
    # Validate causal inference config
    if config.causal_inference.significance_level <= 0 or config.causal_inference.significance_level >= 1:
        errors.append("Significance level must be between 0 and 1")
    
    # Validate anomaly detection config
    if config.anomaly_detection.contamination <= 0 or config.anomaly_detection.contamination >= 1:
        errors.append("Contamination must be between 0 and 1")
    
    # Validate performance monitoring config
    if config.performance_monitoring.monitoring_interval <= 0:
        errors.append("Monitoring interval must be positive")
    
    if config.performance_monitoring.metric_history_size <= 0:
        errors.append("Metric history size must be positive")
    
    return errors
