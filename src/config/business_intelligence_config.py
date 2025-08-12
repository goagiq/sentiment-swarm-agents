"""
Business Intelligence Configuration for the sentiment analysis system.
"""

from typing import List

from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class DashboardConfig(BaseSettings):
    """Configuration for business dashboards."""
    
    # Dashboard settings
    theme: str = Field(
        default="business",
        description="Dashboard theme: 'business', 'executive', 'technical'"
    )
    refresh_rate: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Dashboard refresh rate in seconds"
    )
    max_data_points: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Maximum data points for visualizations"
    )
    
    # Visualization settings
    default_chart_types: List[str] = Field(
        default=["trend", "distribution", "correlation"],
        description="Default chart types for visualizations"
    )
    interactive_mode: bool = Field(
        default=True,
        description="Enable interactive visualizations"
    )
    export_format: str = Field(
        default="html",
        description="Default export format for visualizations"
    )
    
    # Performance settings
    enable_caching: bool = Field(
        default=True,
        description="Enable dashboard caching"
    )
    cache_duration: int = Field(
        default=1800,
        ge=300,
        le=7200,
        description="Cache duration in seconds"
    )
    max_concurrent_dashboards: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum concurrent dashboard generations"
    )


class ReportingConfig(BaseSettings):
    """Configuration for business reporting."""
    
    # Report settings
    default_format: str = Field(
        default="pdf",
        description="Default report format: 'pdf', 'excel', 'html', 'json'"
    )
    include_visualizations: bool = Field(
        default=True,
        description="Include visualizations in reports"
    )
    executive_summary_length: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Executive summary length in characters"
    )
    
    # Report templates
    available_templates: List[str] = Field(
        default=["business", "technical", "stakeholder", "executive"],
        description="Available report templates"
    )
    default_template: str = Field(
        default="business",
        description="Default report template"
    )
    
    # Export settings
    export_formats: List[str] = Field(
        default=["pdf", "excel", "html", "json"],
        description="Available export formats"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include metadata in exports"
    )
    
    # Automation settings
    enable_automated_reports: bool = Field(
        default=True,
        description="Enable automated report generation"
    )
    default_schedule: str = Field(
        default="weekly",
        description="Default report schedule: 'daily', 'weekly', 'monthly'"
    )


class TrendAnalysisConfig(BaseSettings):
    """Configuration for trend analysis and forecasting."""
    
    # Trend analysis settings
    default_trend_period: str = Field(
        default="30d",
        description="Default trend analysis period"
    )
    analysis_types: List[str] = Field(
        default=["sentiment", "topics", "entities", "comprehensive"],
        description="Available analysis types"
    )
    include_forecasting: bool = Field(
        default=True,
        description="Include forecasting in trend analysis"
    )
    
    # Forecasting settings
    default_forecast_period: str = Field(
        default="90d",
        description="Default forecasting period"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.8,
        le=0.99,
        description="Forecasting confidence level"
    )
    include_scenarios: bool = Field(
        default=True,
        description="Include multiple scenarios in forecasts"
    )
    
    # Comparative analysis
    enable_comparative_analysis: bool = Field(
        default=True,
        description="Enable comparative analysis"
    )
    comparison_types: List[str] = Field(
        default=["performance", "sentiment", "trends"],
        description="Available comparison types"
    )
    include_benchmarks: bool = Field(
        default=True,
        description="Include benchmarks in comparative analysis"
    )


class BusinessIntelligenceConfig(BaseSettings):
    """Main business intelligence configuration."""
    
    model_config = ConfigDict(env_prefix="BI_")
    
    # Dashboard configuration
    dashboard: DashboardConfig = Field(
        default_factory=DashboardConfig,
        description="Dashboard configuration"
    )
    
    # Reporting configuration
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="Reporting configuration"
    )
    
    # Trend analysis configuration
    trend_analysis: TrendAnalysisConfig = Field(
        default_factory=TrendAnalysisConfig,
        description="Trend analysis configuration"
    )
    
    # Business intelligence settings
    enable_business_insights: bool = Field(
        default=True,
        description="Enable business intelligence features"
    )
    enable_executive_summaries: bool = Field(
        default=True,
        description="Enable executive summary generation"
    )
    enable_trend_analysis: bool = Field(
        default=True,
        description="Enable trend analysis and forecasting"
    )
    
    # Performance settings
    max_processing_time: int = Field(
        default=300,
        ge=60,
        le=1800,
        description="Maximum processing time for BI operations in seconds"
    )
    enable_parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing for BI operations"
    )
    max_parallel_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum parallel workers for BI operations"
    )
    
    # Storage settings
    bi_data_path: str = Field(
        default="./Results/business_intelligence",
        description="Path for storing business intelligence data"
    )
    enable_data_persistence: bool = Field(
        default=True,
        description="Enable persistence of BI data"
    )
    data_retention_days: int = Field(
        default=365,
        ge=30,
        le=1095,
        description="Data retention period in days"
    )


# Global configuration instance
bi_config = BusinessIntelligenceConfig()
