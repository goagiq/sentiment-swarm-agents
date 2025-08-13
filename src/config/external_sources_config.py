"""
External Sources Configuration

Configuration settings for external data integration including:
- API configurations
- Database configurations  
- Data synchronization settings
- Quality monitoring rules
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum

from src.core.external_integration.api_connector import AuthType, APIConfig
from src.core.external_integration.database_connector import DatabaseType, DatabaseConfig
from src.core.external_integration.data_synchronizer import SyncDirection, ConflictResolution, SyncConfig
from src.core.external_integration.quality_monitor import ValidationRule, QualityRule


@dataclass
class ExternalSourcesConfig:
    """Main configuration for external data sources"""
    
    # API Configurations
    api_configs: Dict[str, APIConfig] = field(default_factory=dict)
    
    # Database Configurations
    database_configs: Dict[str, DatabaseConfig] = field(default_factory=dict)
    
    # Synchronization Configurations
    sync_configs: Dict[str, SyncConfig] = field(default_factory=dict)
    
    # Quality Rules
    quality_rules: Dict[str, List[QualityRule]] = field(default_factory=dict)
    
    # General Settings
    enable_monitoring: bool = True
    enable_sync: bool = True
    enable_quality_checks: bool = True
    sync_interval: int = 300  # 5 minutes
    quality_check_interval: int = 3600  # 1 hour
    max_retry_attempts: int = 3
    timeout: int = 30


# Default API configurations
DEFAULT_API_CONFIGS = {
    "weather_api": APIConfig(
        name="weather_api",
        base_url="https://api.openweathermap.org/data/2.5",
        auth_type=AuthType.API_KEY,
        auth_credentials={"api_key": "your_api_key_here"},
        rate_limit=60,
        timeout=30,
        retry_attempts=3,
        headers={"Content-Type": "application/json"}
    ),
    
    "news_api": APIConfig(
        name="news_api", 
        base_url="https://newsapi.org/v2",
        auth_type=AuthType.API_KEY,
        auth_credentials={"api_key": "your_api_key_here"},
        rate_limit=100,
        timeout=30,
        retry_attempts=3,
        headers={"Content-Type": "application/json"}
    ),
    
    "financial_api": APIConfig(
        name="financial_api",
        base_url="https://api.example.com/financial",
        auth_type=AuthType.BEARER_TOKEN,
        auth_credentials={"token": "your_token_here"},
        rate_limit=1000,
        timeout=60,
        retry_attempts=5,
        headers={"Content-Type": "application/json"}
    )
}


# Default database configurations
DEFAULT_DATABASE_CONFIGS = {
    "main_database": DatabaseConfig(
        name="main_database",
        db_type=DatabaseType.POSTGRESQL,
        host="localhost",
        port=5432,
        database="sentiment_analysis",
        username="postgres",
        password="password",
        pool_size=20,
        timeout=30
    ),
    
    "cache_database": DatabaseConfig(
        name="cache_database",
        db_type=DatabaseType.REDIS,
        host="localhost", 
        port=6379,
        database="0",
        pool_size=10,
        timeout=10
    ),
    
    "analytics_database": DatabaseConfig(
        name="analytics_database",
        db_type=DatabaseType.MONGODB,
        host="localhost",
        port=27017,
        database="analytics",
        connection_string="mongodb://localhost:27017/",
        pool_size=15,
        timeout=30
    )
}


# Default synchronization configurations
DEFAULT_SYNC_CONFIGS = {
    "api_to_main": SyncConfig(
        name="api_to_main",
        source_name="weather_api",
        target_name="main_database",
        direction=SyncDirection.SOURCE_TO_TARGET,
        conflict_resolution=ConflictResolution.SOURCE_WINS,
        sync_interval=300,  # 5 minutes
        batch_size=100,
        enabled=True,
        transform_rules={
            "temperature": {"type": "format", "format": "celsius"},
            "humidity": {"type": "range_check", "min": 0, "max": 100}
        },
        filters={
            "country": "US",
            "temperature": {"min": -50, "max": 150}
        },
        mapping={
            "temp": "temperature",
            "hum": "humidity",
            "desc": "description"
        }
    ),
    
    "main_to_analytics": SyncConfig(
        name="main_to_analytics",
        source_name="main_database", 
        target_name="analytics_database",
        direction=SyncDirection.SOURCE_TO_TARGET,
        conflict_resolution=ConflictResolution.TIMESTAMP,
        sync_interval=600,  # 10 minutes
        batch_size=500,
        enabled=True,
        transform_rules={
            "timestamp": {"type": "format", "format": "iso"},
            "sentiment_score": {"type": "calculation", "formula": "score * 100"}
        },
        filters={},
        mapping={
            "id": "_id",
            "created_at": "timestamp",
            "sentiment": "sentiment_score"
        }
    )
}


# Default quality rules
DEFAULT_QUALITY_RULES = {
    "weather_data": [
        QualityRule(
            field="temperature",
            rule_type=ValidationRule.RANGE_CHECK,
            parameters={"min": -100, "max": 150},
            severity="error",
            description="Temperature must be between -100 and 150 degrees"
        ),
        QualityRule(
            field="humidity",
            rule_type=ValidationRule.RANGE_CHECK,
            parameters={"min": 0, "max": 100},
            severity="error", 
            description="Humidity must be between 0 and 100 percent"
        ),
        QualityRule(
            field="location",
            rule_type=ValidationRule.REQUIRED,
            severity="error",
            description="Location is required"
        ),
        QualityRule(
            field="timestamp",
            rule_type=ValidationRule.PATTERN_MATCH,
            parameters={"pattern": r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"},
            severity="warning",
            description="Timestamp should be in ISO format"
        )
    ],
    
    "sentiment_data": [
        QualityRule(
            field="text",
            rule_type=ValidationRule.REQUIRED,
            severity="error",
            description="Text content is required"
        ),
        QualityRule(
            field="sentiment_score",
            rule_type=ValidationRule.RANGE_CHECK,
            parameters={"min": -1, "max": 1},
            severity="error",
            description="Sentiment score must be between -1 and 1"
        ),
        QualityRule(
            field="confidence",
            rule_type=ValidationRule.RANGE_CHECK,
            parameters={"min": 0, "max": 1},
            severity="warning",
            description="Confidence should be between 0 and 1"
        ),
        QualityRule(
            field="language",
            rule_type=ValidationRule.PATTERN_MATCH,
            parameters={"pattern": r"^[a-z]{2}$"},
            severity="warning",
            description="Language should be a 2-letter code"
        )
    ],
    
    "user_data": [
        QualityRule(
            field="email",
            rule_type=ValidationRule.PATTERN_MATCH,
            parameters={"pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
            severity="error",
            description="Email must be in valid format"
        ),
        QualityRule(
            field="age",
            rule_type=ValidationRule.RANGE_CHECK,
            parameters={"min": 13, "max": 120},
            severity="error",
            description="Age must be between 13 and 120"
        ),
        QualityRule(
            field="username",
            rule_type=ValidationRule.UNIQUE_CHECK,
            severity="error",
            description="Username must be unique"
        )
    ]
}


def get_default_config() -> ExternalSourcesConfig:
    """
    Get default external sources configuration
    
    Returns:
        Default configuration object
    """
    return ExternalSourcesConfig(
        api_configs=DEFAULT_API_CONFIGS,
        database_configs=DEFAULT_DATABASE_CONFIGS,
        sync_configs=DEFAULT_SYNC_CONFIGS,
        quality_rules=DEFAULT_QUALITY_RULES
    )


def load_config_from_file(config_path: str) -> ExternalSourcesConfig:
    """
    Load configuration from file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    import json
    import os
    
    if not os.path.exists(config_path):
        return get_default_config()
        
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            
        # Convert JSON data to configuration objects
        # This is a simplified version - in practice you'd want more robust parsing
        
        api_configs = {}
        for name, data in config_data.get('api_configs', {}).items():
            api_configs[name] = APIConfig(**data)
            
        database_configs = {}
        for name, data in config_data.get('database_configs', {}).items():
            database_configs[name] = DatabaseConfig(**data)
            
        sync_configs = {}
        for name, data in config_data.get('sync_configs', {}).items():
            sync_configs[name] = SyncConfig(**data)
            
        quality_rules = {}
        for name, rules_data in config_data.get('quality_rules', {}).items():
            quality_rules[name] = [QualityRule(**rule_data) for rule_data in rules_data]
            
        return ExternalSourcesConfig(
            api_configs=api_configs,
            database_configs=database_configs,
            sync_configs=sync_configs,
            quality_rules=quality_rules,
            enable_monitoring=config_data.get('enable_monitoring', True),
            enable_sync=config_data.get('enable_sync', True),
            enable_quality_checks=config_data.get('enable_quality_checks', True),
            sync_interval=config_data.get('sync_interval', 300),
            quality_check_interval=config_data.get('quality_check_interval', 3600),
            max_retry_attempts=config_data.get('max_retry_attempts', 3),
            timeout=config_data.get('timeout', 30)
        )
        
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        return get_default_config()


def save_config_to_file(config: ExternalSourcesConfig, config_path: str):
    """
    Save configuration to file
    
    Args:
        config: Configuration object
        config_path: Path to save configuration file
    """
    import json
    
    config_data = {
        'api_configs': {
            name: {
                'name': api_config.name,
                'base_url': api_config.base_url,
                'auth_type': api_config.auth_type.value,
                'auth_credentials': api_config.auth_credentials,
                'rate_limit': api_config.rate_limit,
                'timeout': api_config.timeout,
                'retry_attempts': api_config.retry_attempts,
                'headers': api_config.headers
            }
            for name, api_config in config.api_configs.items()
        },
        'database_configs': {
            name: {
                'name': db_config.name,
                'db_type': db_config.db_type.value,
                'host': db_config.host,
                'port': db_config.port,
                'database': db_config.database,
                'username': db_config.username,
                'password': db_config.password,
                'pool_size': db_config.pool_size,
                'timeout': db_config.timeout
            }
            for name, db_config in config.database_configs.items()
        },
        'sync_configs': {
            name: {
                'name': sync_config.name,
                'source_name': sync_config.source_name,
                'target_name': sync_config.target_name,
                'direction': sync_config.direction.value,
                'conflict_resolution': sync_config.conflict_resolution.value,
                'sync_interval': sync_config.sync_interval,
                'batch_size': sync_config.batch_size,
                'enabled': sync_config.enabled,
                'transform_rules': sync_config.transform_rules,
                'filters': sync_config.filters,
                'mapping': sync_config.mapping
            }
            for name, sync_config in config.sync_configs.items()
        },
        'quality_rules': {
            name: [
                {
                    'field': rule.field,
                    'rule_type': rule.rule_type.value,
                    'parameters': rule.parameters,
                    'severity': rule.severity,
                    'description': rule.description
                }
                for rule in rules
            ]
            for name, rules in config.quality_rules.items()
        },
        'enable_monitoring': config.enable_monitoring,
        'enable_sync': config.enable_sync,
        'enable_quality_checks': config.enable_quality_checks,
        'sync_interval': config.sync_interval,
        'quality_check_interval': config.quality_check_interval,
        'max_retry_attempts': config.max_retry_attempts,
        'timeout': config.timeout
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
