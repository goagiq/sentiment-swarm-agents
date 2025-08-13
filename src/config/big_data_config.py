"""
Big Data Configuration

Configuration for big data processing components:
- Apache Spark settings
- Data lake connections
- Batch processing parameters
- Data governance settings
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Configuration loading function
def get_config() -> Dict[str, Any]:
    """Get base configuration as dictionary."""
    return {}


@dataclass
class SparkConfig:
    """Apache Spark configuration."""
    app_name: str = "SentimentAnalytics"
    master: str = "local[*]"
    driver_memory: str = "2g"
    executor_memory: str = "2g"
    executor_cores: int = 2
    spark_sql_adaptive_enabled: bool = True
    spark_sql_adaptive_coalesce_partitions_enabled: bool = True
    spark_sql_adaptive_skew_join_enabled: bool = True


@dataclass
class DataLakeConfig:
    """Data lake configuration."""
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    azure_connection_string: str = ""
    azure_container: str = ""
    gcs_bucket: str = ""
    gcs_credentials_path: str = ""


@dataclass
class BatchProcessingConfig:
    """Batch processing configuration."""
    max_workers: int = 4
    chunk_size: int = 1000
    max_memory_mb: int = 1024
    batch_timeout: int = 3600
    enable_parallel_processing: bool = True
    enable_progress_tracking: bool = True


@dataclass
class DataGovernanceConfig:
    """Data governance configuration."""
    enable_lineage_tracking: bool = True
    enable_quality_validation: bool = True
    enable_compliance_monitoring: bool = True
    quality_threshold: float = 0.95
    retention_days: int = 365
    enable_metadata_management: bool = True


@dataclass
class BigDataConfig:
    """Complete big data configuration."""
    spark: SparkConfig = field(default_factory=SparkConfig)
    data_lake: DataLakeConfig = field(default_factory=DataLakeConfig)
    batch_processing: BatchProcessingConfig = field(default_factory=BatchProcessingConfig)
    data_governance: DataGovernanceConfig = field(default_factory=DataGovernanceConfig)
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600
    enable_compression: bool = True
    compression_format: str = "snappy"
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_interval: int = 60
    enable_alerting: bool = True
    
    # Storage settings
    temp_directory: str = "./temp"
    output_directory: str = "./output"
    enable_backup: bool = True
    backup_retention_days: int = 30


# Global configuration instance
_big_data_config: Optional[BigDataConfig] = None


def get_big_data_config() -> Dict[str, Any]:
    """
    Get big data configuration as dictionary.
    
    Returns:
        Dict containing big data configuration
    """
    global _big_data_config
    
    if _big_data_config is None:
        _big_data_config = _load_big_data_config()
    
    return _config_to_dict(_big_data_config)


def _load_big_data_config() -> BigDataConfig:
    """Load big data configuration from environment and config files."""
    config = get_config()
    
    # Create base configuration
    big_data_config = BigDataConfig()
    
    # Load Spark configuration
    big_data_config.spark.app_name = os.getenv(
        'SPARK_APP_NAME', 
        config.get('spark', {}).get('app_name', 'SentimentAnalytics')
    )
    big_data_config.spark.master = os.getenv(
        'SPARK_MASTER', 
        config.get('spark', {}).get('master', 'local[*]')
    )
    big_data_config.spark.driver_memory = os.getenv(
        'SPARK_DRIVER_MEMORY', 
        config.get('spark', {}).get('driver_memory', '2g')
    )
    big_data_config.spark.executor_memory = os.getenv(
        'SPARK_EXECUTOR_MEMORY', 
        config.get('spark', {}).get('executor_memory', '2g')
    )
    big_data_config.spark.executor_cores = int(os.getenv(
        'SPARK_EXECUTOR_CORES', 
        config.get('spark', {}).get('executor_cores', 2)
    ))
    
    # Load data lake configuration
    big_data_config.data_lake.s3_bucket = os.getenv(
        'S3_BUCKET', 
        config.get('data_lake', {}).get('s3_bucket', '')
    )
    big_data_config.data_lake.s3_region = os.getenv(
        'S3_REGION', 
        config.get('data_lake', {}).get('s3_region', 'us-east-1')
    )
    big_data_config.data_lake.s3_access_key = os.getenv(
        'S3_ACCESS_KEY', 
        config.get('data_lake', {}).get('s3_access_key', '')
    )
    big_data_config.data_lake.s3_secret_key = os.getenv(
        'S3_SECRET_KEY', 
        config.get('data_lake', {}).get('s3_secret_key', '')
    )
    big_data_config.data_lake.azure_connection_string = os.getenv(
        'AZURE_CONNECTION_STRING', 
        config.get('data_lake', {}).get('azure_connection_string', '')
    )
    big_data_config.data_lake.azure_container = os.getenv(
        'AZURE_CONTAINER', 
        config.get('data_lake', {}).get('azure_container', '')
    )
    big_data_config.data_lake.gcs_bucket = os.getenv(
        'GCS_BUCKET', 
        config.get('data_lake', {}).get('gcs_bucket', '')
    )
    big_data_config.data_lake.gcs_credentials_path = os.getenv(
        'GCS_CREDENTIALS_PATH', 
        config.get('data_lake', {}).get('gcs_credentials_path', '')
    )
    
    # Load batch processing configuration
    big_data_config.batch_processing.max_workers = int(os.getenv(
        'BATCH_MAX_WORKERS', 
        config.get('batch_processing', {}).get('max_workers', 4)
    ))
    big_data_config.batch_processing.chunk_size = int(os.getenv(
        'BATCH_CHUNK_SIZE', 
        config.get('batch_processing', {}).get('chunk_size', 1000)
    ))
    big_data_config.batch_processing.max_memory_mb = int(os.getenv(
        'BATCH_MAX_MEMORY_MB', 
        config.get('batch_processing', {}).get('max_memory_mb', 1024)
    ))
    big_data_config.batch_processing.batch_timeout = int(os.getenv(
        'BATCH_TIMEOUT', 
        config.get('batch_processing', {}).get('batch_timeout', 3600)
    ))
    
    # Load data governance configuration
    big_data_config.data_governance.enable_lineage_tracking = bool(os.getenv(
        'ENABLE_LINEAGE_TRACKING', 
        config.get('data_governance', {}).get('enable_lineage_tracking', True)
    ))
    big_data_config.data_governance.enable_quality_validation = bool(os.getenv(
        'ENABLE_QUALITY_VALIDATION', 
        config.get('data_governance', {}).get('enable_quality_validation', True)
    ))
    big_data_config.data_governance.enable_compliance_monitoring = bool(os.getenv(
        'ENABLE_COMPLIANCE_MONITORING', 
        config.get('data_governance', {}).get('enable_compliance_monitoring', True)
    ))
    big_data_config.data_governance.quality_threshold = float(os.getenv(
        'QUALITY_THRESHOLD', 
        config.get('data_governance', {}).get('quality_threshold', 0.95)
    ))
    big_data_config.data_governance.retention_days = int(os.getenv(
        'RETENTION_DAYS', 
        config.get('data_governance', {}).get('retention_days', 365)
    ))
    
    # Load performance settings
    big_data_config.enable_caching = bool(os.getenv(
        'ENABLE_CACHING', 
        config.get('big_data', {}).get('enable_caching', True)
    ))
    big_data_config.cache_ttl = int(os.getenv(
        'CACHE_TTL', 
        config.get('big_data', {}).get('cache_ttl', 3600)
    ))
    big_data_config.enable_compression = bool(os.getenv(
        'ENABLE_COMPRESSION', 
        config.get('big_data', {}).get('enable_compression', True)
    ))
    big_data_config.compression_format = os.getenv(
        'COMPRESSION_FORMAT', 
        config.get('big_data', {}).get('compression_format', 'snappy')
    )
    
    # Load monitoring settings
    big_data_config.enable_metrics = bool(os.getenv(
        'ENABLE_METRICS', 
        config.get('big_data', {}).get('enable_metrics', True)
    ))
    big_data_config.metrics_interval = int(os.getenv(
        'METRICS_INTERVAL', 
        config.get('big_data', {}).get('metrics_interval', 60)
    ))
    big_data_config.enable_alerting = bool(os.getenv(
        'ENABLE_ALERTING', 
        config.get('big_data', {}).get('enable_alerting', True)
    ))
    
    # Load storage settings
    big_data_config.temp_directory = os.getenv(
        'TEMP_DIRECTORY', 
        config.get('big_data', {}).get('temp_directory', './temp')
    )
    big_data_config.output_directory = os.getenv(
        'OUTPUT_DIRECTORY', 
        config.get('big_data', {}).get('output_directory', './output')
    )
    big_data_config.enable_backup = bool(os.getenv(
        'ENABLE_BACKUP', 
        config.get('big_data', {}).get('enable_backup', True)
    ))
    big_data_config.backup_retention_days = int(os.getenv(
        'BACKUP_RETENTION_DAYS', 
        config.get('big_data', {}).get('backup_retention_days', 30)
    ))
    
    return big_data_config


def _config_to_dict(config: BigDataConfig) -> Dict[str, Any]:
    """Convert BigDataConfig to dictionary."""
    return {
        'spark': {
            'app_name': config.spark.app_name,
            'master': config.spark.master,
            'driver_memory': config.spark.driver_memory,
            'executor_memory': config.spark.executor_memory,
            'executor_cores': config.spark.executor_cores,
            'spark_sql_adaptive_enabled': config.spark.spark_sql_adaptive_enabled,
            'spark_sql_adaptive_coalesce_partitions_enabled': 
                config.spark.spark_sql_adaptive_coalesce_partitions_enabled,
            'spark_sql_adaptive_skew_join_enabled': 
                config.spark.spark_sql_adaptive_skew_join_enabled
        },
        'data_lake': {
            's3_bucket': config.data_lake.s3_bucket,
            's3_region': config.data_lake.s3_region,
            's3_access_key': config.data_lake.s3_access_key,
            's3_secret_key': config.data_lake.s3_secret_key,
            'azure_connection_string': config.data_lake.azure_connection_string,
            'azure_container': config.data_lake.azure_container,
            'gcs_bucket': config.data_lake.gcs_bucket,
            'gcs_credentials_path': config.data_lake.gcs_credentials_path
        },
        'batch_processing': {
            'max_workers': config.batch_processing.max_workers,
            'chunk_size': config.batch_processing.chunk_size,
            'max_memory_mb': config.batch_processing.max_memory_mb,
            'batch_timeout': config.batch_processing.batch_timeout,
            'enable_parallel_processing': config.batch_processing.enable_parallel_processing,
            'enable_progress_tracking': config.batch_processing.enable_progress_tracking
        },
        'data_governance': {
            'enable_lineage_tracking': config.data_governance.enable_lineage_tracking,
            'enable_quality_validation': config.data_governance.enable_quality_validation,
            'enable_compliance_monitoring': config.data_governance.enable_compliance_monitoring,
            'quality_threshold': config.data_governance.quality_threshold,
            'retention_days': config.data_governance.retention_days,
            'enable_metadata_management': config.data_governance.enable_metadata_management
        },
        'enable_caching': config.enable_caching,
        'cache_ttl': config.cache_ttl,
        'enable_compression': config.enable_compression,
        'compression_format': config.compression_format,
        'enable_metrics': config.enable_metrics,
        'metrics_interval': config.metrics_interval,
        'enable_alerting': config.enable_alerting,
        'temp_directory': config.temp_directory,
        'output_directory': config.output_directory,
        'enable_backup': config.enable_backup,
        'backup_retention_days': config.backup_retention_days
    }


def get_spark_config() -> Dict[str, str]:
    """Get Spark configuration as dictionary."""
    config = get_big_data_config()
    return config.get('spark', {})


def get_data_lake_config() -> Dict[str, str]:
    """Get data lake configuration as dictionary."""
    config = get_big_data_config()
    return config.get('data_lake', {})


def get_batch_processing_config() -> Dict[str, Any]:
    """Get batch processing configuration as dictionary."""
    config = get_big_data_config()
    return config.get('batch_processing', {})


def get_data_governance_config() -> Dict[str, Any]:
    """Get data governance configuration as dictionary."""
    config = get_big_data_config()
    return config.get('data_governance', {})
