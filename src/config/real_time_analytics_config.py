"""
Real-Time Analytics Configuration

Configuration settings for real-time analytics dashboard features including:
- Live data streaming
- Interactive visualizations
- Analytics workflow management
- Alert management
"""

from typing import Dict, List, Any
from pydantic import BaseModel, Field


class StreamProcessingConfig(BaseModel):
    """Configuration for real-time data stream processing."""
    
    # Stream processing settings
    buffer_size: int = Field(default=1000, description="Maximum buffer size for data points")
    batch_size: int = Field(default=100, description="Batch size for processing")
    processing_interval: float = Field(default=0.1, description="Processing interval in seconds")
    max_latency: float = Field(default=1.0, description="Maximum processing latency in seconds")
    
    # Data quality settings
    enable_validation: bool = Field(default=True, description="Enable data validation")
    enable_filtering: bool = Field(default=True, description="Enable data filtering")
    enable_aggregation: bool = Field(default=True, description="Enable data aggregation")
    
    # Performance settings
    enable_batching: bool = Field(default=True, description="Enable batch processing")
    enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    max_workers: int = Field(default=4, description="Maximum number of worker threads")


class VisualizationConfig(BaseModel):
    """Configuration for interactive visualizations."""
    
    # Chart settings
    default_chart_type: str = Field(default="line", description="Default chart type")
    max_data_points: int = Field(default=1000, description="Maximum data points per chart")
    update_interval: float = Field(default=1.0, description="Chart update interval in seconds")
    
    # Interactive features
    enable_zoom: bool = Field(default=True, description="Enable chart zooming")
    enable_pan: bool = Field(default=True, description="Enable chart panning")
    enable_drill_down: bool = Field(default=True, description="Enable drill-down capabilities")
    enable_cross_filtering: bool = Field(default=True, description="Enable cross-chart filtering")
    
    # Theme settings
    theme: str = Field(default="plotly_white", description="Chart theme")
    colors: List[str] = Field(
        default=[
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ],
        description="Chart color palette"
    )


class DashboardConfig(BaseModel):
    """Configuration for analytics dashboard."""
    
    # Layout settings
    default_layout: str = Field(default="grid", description="Default dashboard layout")
    max_widgets: int = Field(default=12, description="Maximum widgets per dashboard")
    widget_sizes: Dict[str, Dict[str, int]] = Field(
        default={
            "small": {"width": 4, "height": 3},
            "medium": {"width": 6, "height": 4},
            "large": {"width": 12, "height": 6}
        },
        description="Widget size configurations"
    )
    
    # Dashboard features
    enable_customization: bool = Field(default=True, description="Enable dashboard customization")
    enable_sharing: bool = Field(default=True, description="Enable dashboard sharing")
    enable_export: bool = Field(default=True, description="Enable dashboard export")
    enable_scheduling: bool = Field(default=True, description="Enable scheduled reports")


class AlertConfig(BaseModel):
    """Configuration for alert management."""
    
    # Alert settings
    enable_alerts: bool = Field(default=True, description="Enable alert system")
    alert_check_interval: float = Field(default=5.0, description="Alert check interval in seconds")
    max_alerts: int = Field(default=100, description="Maximum number of active alerts")
    
    # Alert types
    alert_types: List[str] = Field(
        default=["threshold", "anomaly", "trend", "custom"],
        description="Supported alert types"
    )
    
    # Notification settings
    notification_channels: List[str] = Field(
        default=["dashboard", "email", "webhook"],
        description="Available notification channels"
    )
    
    # Alert severity levels
    severity_levels: Dict[str, Dict[str, Any]] = Field(
        default={
            "critical": {"color": "#d62728", "priority": 1, "timeout": 300},
            "high": {"color": "#ff7f0e", "priority": 2, "timeout": 600},
            "medium": {"color": "#ffdc00", "priority": 3, "timeout": 1800},
            "low": {"color": "#2ca02c", "priority": 4, "timeout": 3600}
        },
        description="Alert severity level configurations"
    )


class WorkflowConfig(BaseModel):
    """Configuration for analytics workflow management."""
    
    # Workflow settings
    enable_workflows: bool = Field(default=True, description="Enable workflow management")
    max_workflows: int = Field(default=50, description="Maximum number of workflows")
    workflow_timeout: int = Field(default=3600, description="Workflow timeout in seconds")
    
    # Workflow types
    workflow_types: List[str] = Field(
        default=["scheduled", "triggered", "manual", "continuous"],
        description="Supported workflow types"
    )
    
    # Collaboration settings
    enable_collaboration: bool = Field(default=True, description="Enable collaborative analytics")
    max_collaborators: int = Field(default=10, description="Maximum collaborators per workflow")
    enable_version_control: bool = Field(default=True, description="Enable workflow versioning")


class DataSourceConfig(BaseModel):
    """Configuration for data sources."""
    
    # Supported data sources
    supported_sources: List[str] = Field(
        default=[
            "api", "database", "file", "stream", "webhook", "message_queue"
        ],
        description="Supported data source types"
    )
    
    # Connection settings
    connection_timeout: int = Field(default=30, description="Connection timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")
    
    # Data format settings
    supported_formats: List[str] = Field(
        default=["json", "csv", "xml", "yaml", "avro", "parquet"],
        description="Supported data formats"
    )


class PerformanceConfig(BaseModel):
    """Configuration for performance optimization."""
    
    # Caching settings
    enable_caching: bool = Field(default=True, description="Enable data caching")
    cache_ttl: int = Field(default=300, description="Cache TTL in seconds")
    max_cache_size: int = Field(default=1000, description="Maximum cache size in MB")
    
    # Memory management
    max_memory_usage: float = Field(default=0.8, description="Maximum memory usage ratio")
    garbage_collection_interval: int = Field(default=300, description="GC interval in seconds")
    
    # Processing optimization
    enable_compression: bool = Field(default=True, description="Enable data compression")
    enable_deduplication: bool = Field(default=True, description="Enable data deduplication")


class RealTimeAnalyticsConfig(BaseModel):
    """Main configuration for real-time analytics dashboard."""
    
    # Component configurations
    stream_processing: StreamProcessingConfig = Field(
        default_factory=StreamProcessingConfig,
        description="Stream processing configuration"
    )
    
    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig,
        description="Visualization configuration"
    )
    
    dashboard: DashboardConfig = Field(
        default_factory=DashboardConfig,
        description="Dashboard configuration"
    )
    
    alerts: AlertConfig = Field(
        default_factory=AlertConfig,
        description="Alert configuration"
    )
    
    workflow: WorkflowConfig = Field(
        default_factory=WorkflowConfig,
        description="Workflow configuration"
    )
    
    data_sources: DataSourceConfig = Field(
        default_factory=DataSourceConfig,
        description="Data source configuration"
    )
    
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance configuration"
    )
    
    # System settings
    enable_real_time: bool = Field(default=True, description="Enable real-time features")
    enable_websockets: bool = Field(default=True, description="Enable WebSocket connections")
    websocket_port: int = Field(default=8004, description="WebSocket server port")
    
    # Integration settings
    api_endpoints: List[str] = Field(
        default=[
            "/analytics/stream",
            "/analytics/dashboard",
            "/analytics/alerts",
            "/analytics/workflows"
        ],
        description="Available API endpoints"
    )
    
    # Security settings
    enable_authentication: bool = Field(default=False, description="Enable authentication")
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    max_requests_per_minute: int = Field(default=1000, description="Maximum requests per minute")


# Default configuration instance
default_config = RealTimeAnalyticsConfig()


def get_real_time_analytics_config() -> RealTimeAnalyticsConfig:
    """Get the real-time analytics configuration."""
    return default_config


def update_real_time_analytics_config(config: RealTimeAnalyticsConfig) -> None:
    """Update the real-time analytics configuration."""
    global default_config
    default_config = config
