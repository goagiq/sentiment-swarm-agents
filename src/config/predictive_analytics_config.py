"""
Predictive Analytics Configuration

Configuration settings for predictive analytics components including
forecasting, confidence intervals, scenario analysis, and validation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class ForecastModelType(Enum):
    """Supported forecasting model types"""
    ARIMA = "arima"
    PROPHET = "prophet"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"
    SIMPLE_AVERAGE = "simple_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"


class ConfidenceMethod(Enum):
    """Confidence interval calculation methods"""
    PARAMETRIC = "parametric"
    BOOTSTRAP = "bootstrap"
    EMPIRICAL = "empirical"
    SIMPLE_PERCENTAGE = "simple_percentage"


class ValidationType(Enum):
    """Validation types"""
    HOLDOUT = "holdout"
    CROSS_VALIDATION = "cross_validation"


@dataclass
class ForecastingConfig:
    """Configuration for forecasting engine"""
    default_forecast_horizon: int = 12
    default_confidence_level: float = 0.95
    min_data_points: int = 10
    max_data_points: int = 10000
    ensemble_method: str = "weighted_average"
    validation_split: float = 0.2
    random_state: int = 42
    default_model: ForecastModelType = ForecastModelType.ENSEMBLE
    
    # Model-specific configurations
    arima_config: Dict[str, Any] = field(default_factory=lambda: {
        'order': (1, 1, 1),
        'seasonal_order': (1, 1, 1, 12),
        'trend': 'c'
    })
    
    prophet_config: Dict[str, Any] = field(default_factory=lambda: {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'holidays_prior_scale': 10.0
    })
    
    lstm_config: Dict[str, Any] = field(default_factory=lambda: {
        'units': 50,
        'layers': 2,
        'dropout': 0.2,
        'epochs': 100,
        'batch_size': 32
    })
    
    exponential_smoothing_config: Dict[str, Any] = field(default_factory=lambda: {
        'alpha': 0.3,
        'beta': 0.1,
        'gamma': 0.1
    })


@dataclass
class ConfidenceConfig:
    """Configuration for confidence interval calculation"""
    default_confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    prediction_interval_method: ConfidenceMethod = ConfidenceMethod.PARAMETRIC
    uncertainty_quantification: bool = True
    
    # Confidence level thresholds
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.80,
        'medium': 0.90,
        'high': 0.95,
        'very_high': 0.99
    })


@dataclass
class ScenarioConfig:
    """Configuration for scenario analysis"""
    default_scenarios: List[str] = field(default_factory=lambda: [
        'baseline', 'optimistic', 'pessimistic'
    ])
    
    scenario_probabilities: Dict[str, float] = field(default_factory=lambda: {
        'baseline': 0.6,
        'optimistic': 0.2,
        'pessimistic': 0.2
    })
    
    parameter_variation: float = 0.2  # 20% variation
    max_scenarios: int = 10
    confidence_level: float = 0.95
    
    # Scenario templates
    scenario_templates: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'growth_scenario': {
            'description': 'High growth scenario',
            'parameters': {
                'trend_multiplier': 0.15,
                'volatility_multiplier': 0.9,
                'level_shift': 0.1
            }
        },
        'recession_scenario': {
            'description': 'Economic recession scenario',
            'parameters': {
                'trend_multiplier': -0.2,
                'volatility_multiplier': 1.5,
                'level_shift': -0.15
            }
        },
        'stable_scenario': {
            'description': 'Stable growth scenario',
            'parameters': {
                'trend_multiplier': 0.05,
                'volatility_multiplier': 0.7,
                'level_shift': 0.02
            }
        },
        'volatile_scenario': {
            'description': 'High volatility scenario',
            'parameters': {
                'trend_multiplier': 0.0,
                'volatility_multiplier': 2.0,
                'level_shift': 0.0
            }
        }
    })


@dataclass
class ValidationConfig:
    """Configuration for forecast validation"""
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'excellent': 0.9,
        'good': 0.8,
        'fair': 0.7,
        'poor': 0.6
    })
    
    min_test_size: int = 5
    max_test_size: int = 100
    
    # Validation metrics weights
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        'mae': 0.3,
        'mape': 0.3,
        'rmse': 0.2,
        'r_squared': 0.2
    })


@dataclass
class PredictiveAnalyticsConfig:
    """Main configuration for predictive analytics system"""
    
    # Component configurations
    forecasting: ForecastingConfig = field(default_factory=ForecastingConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    # General settings
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # Performance settings
    max_concurrent_requests: int = 10
    request_timeout: int = 300  # seconds
    
    # Data processing settings
    chunk_size: int = 1000
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Model persistence
    save_models: bool = True
    model_storage_path: str = "models/predictive_analytics"
    
    # Export settings
    export_formats: List[str] = field(default_factory=lambda: [
        'json', 'csv', 'excel'
    ])
    
    # Monitoring settings
    enable_monitoring: bool = True
    metrics_collection_interval: int = 60  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'forecasting': {
                'default_forecast_horizon': self.forecasting.default_forecast_horizon,
                'default_confidence_level': self.forecasting.default_confidence_level,
                'min_data_points': self.forecasting.min_data_points,
                'max_data_points': self.forecasting.max_data_points,
                'ensemble_method': self.forecasting.ensemble_method,
                'validation_split': self.forecasting.validation_split,
                'random_state': self.forecasting.random_state,
                'default_model': self.forecasting.default_model.value,
                'arima_config': self.forecasting.arima_config,
                'prophet_config': self.forecasting.prophet_config,
                'lstm_config': self.forecasting.lstm_config,
                'exponential_smoothing_config': self.forecasting.exponential_smoothing_config
            },
            'confidence': {
                'default_confidence_level': self.confidence.default_confidence_level,
                'bootstrap_samples': self.confidence.bootstrap_samples,
                'prediction_interval_method': self.confidence.prediction_interval_method.value,
                'uncertainty_quantification': self.confidence.uncertainty_quantification,
                'confidence_thresholds': self.confidence.confidence_thresholds
            },
            'scenario': {
                'default_scenarios': self.scenario.default_scenarios,
                'scenario_probabilities': self.scenario.scenario_probabilities,
                'parameter_variation': self.scenario.parameter_variation,
                'max_scenarios': self.scenario.max_scenarios,
                'confidence_level': self.scenario.confidence_level,
                'scenario_templates': self.scenario.scenario_templates
            },
            'validation': {
                'validation_split': self.validation.validation_split,
                'cross_validation_folds': self.validation.cross_validation_folds,
                'performance_thresholds': self.validation.performance_thresholds,
                'min_test_size': self.validation.min_test_size,
                'max_test_size': self.validation.max_test_size,
                'metric_weights': self.validation.metric_weights
            },
            'general': {
                'enable_logging': self.enable_logging,
                'log_level': self.log_level,
                'max_concurrent_requests': self.max_concurrent_requests,
                'request_timeout': self.request_timeout,
                'chunk_size': self.chunk_size,
                'enable_caching': self.enable_caching,
                'cache_ttl': self.cache_ttl,
                'save_models': self.save_models,
                'model_storage_path': self.model_storage_path,
                'export_formats': self.export_formats,
                'enable_monitoring': self.enable_monitoring,
                'metrics_collection_interval': self.metrics_collection_interval
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PredictiveAnalyticsConfig':
        """Create configuration from dictionary"""
        forecasting_config = ForecastingConfig(**config_dict.get('forecasting', {}))
        confidence_config = ConfidenceConfig(**config_dict.get('confidence', {}))
        scenario_config = ScenarioConfig(**config_dict.get('scenario', {}))
        validation_config = ValidationConfig(**config_dict.get('validation', {}))
        
        general_config = config_dict.get('general', {})
        
        return cls(
            forecasting=forecasting_config,
            confidence=confidence_config,
            scenario=scenario_config,
            validation=validation_config,
            enable_logging=general_config.get('enable_logging', True),
            log_level=general_config.get('log_level', 'INFO'),
            max_concurrent_requests=general_config.get('max_concurrent_requests', 10),
            request_timeout=general_config.get('request_timeout', 300),
            chunk_size=general_config.get('chunk_size', 1000),
            enable_caching=general_config.get('enable_caching', True),
            cache_ttl=general_config.get('cache_ttl', 3600),
            save_models=general_config.get('save_models', True),
            model_storage_path=general_config.get('model_storage_path', 'models/predictive_analytics'),
            export_formats=general_config.get('export_formats', ['json', 'csv', 'excel']),
            enable_monitoring=general_config.get('enable_monitoring', True),
            metrics_collection_interval=general_config.get('metrics_collection_interval', 60)
        )


# Default configuration instance
DEFAULT_CONFIG = PredictiveAnalyticsConfig()
