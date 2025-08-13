"""
Advanced Machine Learning Module

This module provides advanced machine learning capabilities including:
- Deep learning models and neural networks
- Transfer learning and pre-trained models
- Ensemble methods and model combination
- Time series forecasting
- Clustering algorithms
- Dimensionality reduction
- Automated machine learning (AutoML)
- Model versioning and lifecycle management
"""

# Import components that don't require deep learning frameworks
from .model_versioning import ModelVersioning
from .ensemble_methods import EnsembleMethods
from .time_series_models import TimeSeriesModels
from .clustering_algorithms import ClusteringAlgorithms
from .dimensionality_reduction import DimensionalityReduction
from .automl_pipeline import AutoMLPipeline

# Import deep learning components conditionally
try:
    from .deep_learning_engine import DeepLearningEngine
    from .transfer_learning_service import TransferLearningService
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DeepLearningEngine = None
    TransferLearningService = None
    DEEP_LEARNING_AVAILABLE = False

__all__ = [
    'ModelVersioning',
    'AutoMLPipeline',
    'EnsembleMethods',
    'TimeSeriesModels',
    'ClusteringAlgorithms',
    'DimensionalityReduction',
    'DEEP_LEARNING_AVAILABLE'
]

# Add deep learning components if available
if DEEP_LEARNING_AVAILABLE:
    __all__.extend(['DeepLearningEngine', 'TransferLearningService'])
