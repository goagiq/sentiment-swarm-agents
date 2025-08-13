"""
Pattern Recognition Configuration

This module provides configuration settings for pattern recognition components including:
- Temporal analysis settings
- Seasonal detection parameters
- Trend analysis configuration
- Pattern storage settings
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class TemporalAnalysisConfig:
    """Configuration for temporal pattern analysis."""
    
    # Data requirements
    min_data_points: int = 10
    max_data_points: int = 10000
    
    # Trend analysis
    trend_threshold: float = 0.1
    confidence_threshold: float = 0.7
    
    # Seasonal periods to test (in days)
    seasonal_periods: List[int] = None
    
    # Analysis methods
    enable_autocorrelation: bool = True
    enable_fourier_analysis: bool = True
    enable_trend_analysis: bool = True
    
    def __post_init__(self):
        if self.seasonal_periods is None:
            self.seasonal_periods = [7, 14, 30, 90, 180, 365]


@dataclass
class SeasonalDetectionConfig:
    """Configuration for seasonal pattern detection."""
    
    # Detection parameters
    min_periods: int = 3
    max_periods: int = 365
    autocorr_threshold: float = 0.3
    seasonal_strength_threshold: float = 0.1
    fourier_threshold: float = 0.1
    
    # Analysis methods
    enable_autocorrelation: bool = True
    enable_fourier_analysis: bool = True
    enable_seasonal_decomposition: bool = True
    enable_periodicity_detection: bool = True
    
    # Seasonal strength calculation
    enable_seasonal_strength: bool = True
    strength_categories: Dict[str, float] = None
    
    def __post_init__(self):
        if self.strength_categories is None:
            self.strength_categories = {
                "weak": 0.3,
                "moderate": 0.6,
                "strong": 0.9
            }


@dataclass
class TrendAnalysisConfig:
    """Configuration for trend analysis."""
    
    # Data requirements
    min_data_points: int = 5
    max_data_points: int = 10000
    
    # Trend detection
    trend_threshold: float = 0.1
    confidence_threshold: float = 0.7
    
    # Forecasting
    forecast_periods: int = 10
    enable_forecasting: bool = True
    
    # Moving averages
    smoothing_window: int = 3
    enable_moving_averages: bool = True
    
    # Pattern detection
    enable_trend_reversals: bool = True
    enable_trend_acceleration: bool = True
    enable_trend_consolidation: bool = True
    
    # Analysis methods
    enable_linear_trend: bool = True
    enable_moving_average_trend: bool = True
    enable_trend_comparison: bool = True


@dataclass
class PatternStorageConfig:
    """Configuration for pattern storage."""
    
    # Storage settings
    storage_path: str = "data/pattern_storage"
    enable_persistence: bool = True
    auto_backup: bool = True
    
    # Data management
    max_patterns: int = 10000
    cleanup_deleted_after_days: int = 30
    enable_versioning: bool = True
    
    # Search settings
    default_search_limit: int = 100
    enable_metadata_search: bool = True
    enable_pattern_type_filtering: bool = True


@dataclass
class VectorPatternConfig:
    """Configuration for vector-based pattern discovery."""
    
    # Vector analysis
    enable_vector_clustering: bool = True
    enable_anomaly_detection: bool = True
    enable_pattern_classification: bool = True
    enable_cross_modal_matching: bool = True
    
    # Clustering parameters
    clustering_algorithm: str = "kmeans"
    min_cluster_size: int = 3
    max_clusters: int = 50
    
    # Anomaly detection
    anomaly_threshold: float = 0.95
    enable_statistical_outliers: bool = True
    enable_isolation_forest: bool = True
    
    # Pattern classification
    classification_confidence_threshold: float = 0.7
    enable_auto_classification: bool = True


@dataclass
class GeospatialPatternConfig:
    """Configuration for geospatial pattern analysis."""
    
    # Geospatial analysis
    enable_spatial_analysis: bool = True
    enable_temporal_spatial_correlation: bool = True
    enable_geographic_clustering: bool = True
    enable_map_visualization: bool = True
    
    # Spatial parameters
    spatial_threshold_km: float = 10.0
    enable_distance_calculation: bool = True
    enable_spatial_indexing: bool = True
    
    # Clustering parameters
    spatial_clustering_algorithm: str = "dbscan"
    min_spatial_cluster_size: int = 3
    spatial_epsilon_km: float = 5.0
    
    # Visualization
    map_provider: str = "openstreetmap"
    enable_interactive_maps: bool = True
    map_zoom_levels: List[int] = None
    
    def __post_init__(self):
        if self.map_zoom_levels is None:
            self.map_zoom_levels = [5, 10, 15, 20]


@dataclass
class PatternRecognitionConfig:
    """Main configuration for pattern recognition system."""
    
    # Component configurations
    temporal: TemporalAnalysisConfig = None
    seasonal: SeasonalDetectionConfig = None
    trend: TrendAnalysisConfig = None
    storage: PatternStorageConfig = None
    vector: VectorPatternConfig = None
    geospatial: GeospatialPatternConfig = None
    
    # System settings
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_caching: bool = True
    
    # Performance settings
    max_concurrent_analyses: int = 10
    analysis_timeout_seconds: int = 300
    enable_parallel_processing: bool = True
    
    # Integration settings
    enable_mcp_integration: bool = True
    enable_api_endpoints: bool = True
    enable_streamlit_integration: bool = True
    
    def __post_init__(self):
        if self.temporal is None:
            self.temporal = TemporalAnalysisConfig()
        if self.seasonal is None:
            self.seasonal = SeasonalDetectionConfig()
        if self.trend is None:
            self.trend = TrendAnalysisConfig()
        if self.storage is None:
            self.storage = PatternStorageConfig()
        if self.vector is None:
            self.vector = VectorPatternConfig()
        if self.geospatial is None:
            self.geospatial = GeospatialPatternConfig()


# Default configuration instance
pattern_recognition_config = PatternRecognitionConfig()


def get_pattern_recognition_config() -> PatternRecognitionConfig:
    """Get the pattern recognition configuration."""
    return pattern_recognition_config


def update_pattern_recognition_config(config_updates: Dict[str, Any]) -> PatternRecognitionConfig:
    """Update the pattern recognition configuration with new settings."""
    global pattern_recognition_config
    
    # Update configuration based on provided updates
    for key, value in config_updates.items():
        if hasattr(pattern_recognition_config, key):
            setattr(pattern_recognition_config, key, value)
    
    return pattern_recognition_config
