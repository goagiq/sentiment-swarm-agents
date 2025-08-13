"""
Advanced Machine Learning Configuration

This module provides comprehensive configuration for advanced machine learning
features including deep learning, transfer learning, ensemble methods, and
automated ML pipelines.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass, field
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DeepLearningConfig:
    """Configuration for deep learning models and neural networks."""
    
    # Framework selection
    framework: str = "tensorflow"  # "tensorflow" or "pytorch"
    
    # Neural network architectures
    architectures: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "mlp": {
            "layers": [512, 256, 128, 64],
            "activation": "relu",
            "dropout": 0.3,
            "batch_normalization": True
        },
        "cnn": {
            "filters": [32, 64, 128, 256],
            "kernel_sizes": [3, 3, 3, 3],
            "pooling": "max",
            "dropout": 0.5
        },
        "lstm": {
            "units": [128, 64, 32],
            "dropout": 0.2,
            "recurrent_dropout": 0.2,
            "bidirectional": True
        },
        "transformer": {
            "num_layers": 6,
            "d_model": 512,
            "num_heads": 8,
            "dff": 2048,
            "dropout": 0.1
        }
    })
    
    # Training configuration
    training: Dict[str, Any] = field(default_factory=lambda: {
        "batch_size": 32,
        "epochs": 100,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss_function": "categorical_crossentropy",
        "early_stopping": True,
        "patience": 10,
        "validation_split": 0.2
    })
    
    # Model storage
    model_storage: Dict[str, Any] = field(default_factory=lambda: {
        "base_path": "src/models/neural_networks",
        "versioning": True,
        "auto_save": True,
        "save_format": "h5"  # "h5" or "pb"
    })

@dataclass
class TransferLearningConfig:
    """Configuration for transfer learning and pre-trained models."""
    
    # Pre-trained models
    pre_trained_models: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "bert": {
            "model_name": "bert-base-uncased",
            "max_length": 512,
            "fine_tune_layers": 2
        },
        "gpt2": {
            "model_name": "gpt2",
            "max_length": 1024,
            "fine_tune_layers": 4
        },
        "resnet": {
            "model_name": "resnet50",
            "include_top": False,
            "fine_tune_layers": 10
        },
        "vgg": {
            "model_name": "vgg16",
            "include_top": False,
            "fine_tune_layers": 5
        }
    })
    
    
    # Transfer learning settings
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "freeze_base": True,
        "gradual_unfreezing": True,
        "learning_rate_multiplier": 0.1,
        "fine_tune_epochs": 50
    })

@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods and model combination."""
    
    # Ensemble methods
    methods: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8
        },
        "stacking": {
            "base_models": ["random_forest", "gradient_boosting", "svm"],
            "meta_model": "logistic_regression",
            "cv_folds": 5
        },
        "voting": {
            "voting_type": "soft",  # "hard" or "soft"
            "weights": None  # None for equal weights
        }
    })
    
    # Ensemble optimization
    optimization: Dict[str, Any] = field(default_factory=lambda: {
        "auto_optimize": True,
        "optimization_metric": "accuracy",
        "cross_validation": True,
        "cv_folds": 5
    })

@dataclass
class TimeSeriesConfig:
    """Configuration for time series models and forecasting."""
    
    # Time series models
    models: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "lstm": {
            "units": [128, 64, 32],
            "dropout": 0.2,
            "recurrent_dropout": 0.2,
            "bidirectional": True
        },
        "gru": {
            "units": [128, 64, 32],
            "dropout": 0.2,
            "recurrent_dropout": 0.2
        },
        "transformer": {
            "num_layers": 4,
            "d_model": 256,
            "num_heads": 8,
            "dff": 1024,
            "dropout": 0.1
        },
        "prophet": {
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "holidays_prior_scale": 10.0
        }
    })
    
    # Forecasting settings
    forecasting: Dict[str, Any] = field(default_factory=lambda: {
        "forecast_horizon": 30,
        "confidence_intervals": True,
        "confidence_level": 0.95,
        "seasonality_detection": True
    })

@dataclass
class ClusteringConfig:
    """Configuration for clustering algorithms."""
    
    # Clustering algorithms
    algorithms: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "kmeans": {
            "n_clusters": 8,
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300
        },
        "dbscan": {
            "eps": 0.5,
            "min_samples": 5
        },
        "hierarchical": {
            "n_clusters": 8,
            "linkage": "ward",
            "distance_threshold": None
        },
        "spectral": {
            "n_clusters": 8,
            "affinity": "rbf",
            "gamma": 1.0
        }
    })
    
    # Clustering optimization
    optimization: Dict[str, Any] = field(default_factory=lambda: {
        "auto_optimize": True,
        "optimization_metric": "silhouette",
        "max_clusters": 20
    })

@dataclass
class DimensionalityReductionConfig:
    """Configuration for dimensionality reduction techniques."""
    
    # Dimensionality reduction methods
    methods: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "pca": {
            "n_components": 0.95,  # Explained variance ratio
            "whiten": False
        },
        "tsne": {
            "n_components": 2,
            "perplexity": 30.0,
            "learning_rate": 200.0
        },
        "umap": {
            "n_components": 2,
            "n_neighbors": 15,
            "min_dist": 0.1
        },
        "autoencoder": {
            "encoding_dim": 64,
            "layers": [256, 128, 64],
            "dropout": 0.2
        }
    })

@dataclass
class AutoMLConfig:
    """Configuration for automated machine learning pipeline."""
    
    # AutoML settings
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "time_limit": 3600,  # 1 hour
        "max_models": 50,
        "ensemble_size": 5,
        "cross_validation": True,
        "cv_folds": 5
    })
    
    # Model selection
    model_selection: Dict[str, Any] = field(default_factory=lambda: {
        "include_models": [
            "random_forest", "gradient_boosting", "xgboost", "lightgbm",
            "logistic_regression", "svm", "neural_network"
        ],
        "exclude_models": [],
        "prefer_fast": False
    })
    
    # Feature engineering
    feature_engineering: Dict[str, Any] = field(default_factory=lambda: {
        "auto_feature_engineering": True,
        "feature_selection": True,
        "polynomial_features": True,
        "interaction_features": True
    })

@dataclass
class ModelVersioningConfig:
    """Configuration for model versioning and lifecycle management."""
    
    # Versioning settings
    versioning: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "version_format": "semantic",  # "semantic" or "timestamp"
        "auto_version": True,
        "backup_versions": 10
    })
    
    # Model registry
    registry: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "registry_path": "src/models/registry",
        "metadata_storage": True,
        "performance_tracking": True
    })
    
    # Model deployment
    deployment: Dict[str, Any] = field(default_factory=lambda: {
        "auto_deploy": False,
        "deployment_threshold": 0.9,  # Minimum performance for auto-deployment
        "rollback_enabled": True,
        "a_b_testing": False
    })

class AdvancedMLConfig:
    """Main configuration class for advanced machine learning features."""
    
    def __init__(self):
        self.deep_learning = DeepLearningConfig()
        self.transfer_learning = TransferLearningConfig()
        self.ensemble = EnsembleConfig()
        self.time_series = TimeSeriesConfig()
        self.clustering = ClusteringConfig()
        self.dimensionality_reduction = DimensionalityReductionConfig()
        self.automl = AutoMLConfig()
        self.model_versioning = ModelVersioningConfig()
        
        # Global settings
        self.global_settings = {
            "enable_gpu": True,
            "gpu_memory_fraction": 0.8,
            "parallel_processing": True,
            "cache_models": True,
            "model_cache_size": 10,
            "logging_level": "INFO"
        }
        
        # Performance settings
        self.performance = {
            "batch_processing": True,
            "batch_size": 1000,
            "memory_optimization": True,
            "model_compression": False
        }
        
        # Integration settings
        self.integration = {
            "mcp_integration": True,
            "api_endpoints": True,
            "webhook_notifications": False,
            "real_time_monitoring": True
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
        return {
            "deep_learning": self.deep_learning.__dict__,
            "transfer_learning": self.transfer_learning.__dict__,
            "ensemble": self.ensemble.__dict__,
            "time_series": self.time_series.__dict__,
            "clustering": self.clustering.__dict__,
            "dimensionality_reduction": self.dimensionality_reduction.__dict__,
            "automl": self.automl.__dict__,
            "model_versioning": self.model_versioning.__dict__,
            "global_settings": self.global_settings,
            "performance": self.performance,
            "integration": self.integration
        }
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """Update configuration with new settings."""
        for section, settings in config_updates.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                if hasattr(section_obj, '__dict__'):
                    section_obj.__dict__.update(settings)
                else:
                    setattr(self, section, settings)
            else:
                setattr(self, section, settings)
        
        logger.info(f"Updated advanced ML configuration: {list(config_updates.keys())}")
    
    def validate_config(self) -> bool:
        """Validate configuration settings."""
        try:
            # Validate deep learning config
            if self.deep_learning.framework not in ["tensorflow", "pytorch"]:
                logger.error("Invalid deep learning framework")
                return False
            
            # Validate model storage paths
            model_paths = [
                self.deep_learning.model_storage["base_path"],
                self.model_versioning.registry["registry_path"]
            ]
            
            for path in model_paths:
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)
                    logger.info(f"Created model directory: {path}")
            
            # Validate performance settings
            if self.performance["batch_size"] <= 0:
                logger.error("Invalid batch size")
                return False
            
            logger.info("Advanced ML configuration validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False

# Global configuration instance
advanced_ml_config = AdvancedMLConfig()

def get_advanced_ml_config() -> AdvancedMLConfig:
    """Get the global advanced ML configuration instance."""
    return advanced_ml_config

def update_advanced_ml_config(config_updates: Dict[str, Any]) -> None:
    """Update the global advanced ML configuration."""
    advanced_ml_config.update_config(config_updates)

def validate_advanced_ml_config() -> bool:
    """Validate the global advanced ML configuration."""
    return advanced_ml_config.validate_config()
