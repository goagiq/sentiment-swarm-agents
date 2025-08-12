"""
Monitoring Configuration for Phase 2 Performance Optimization.
Language-specific monitoring settings and performance thresholds.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class MonitoringSettings:
    """Monitoring settings for a specific language."""
    max_processing_time: float
    max_memory_mb: int
    error_rate_threshold: float
    alert_level: str
    monitoring_frequency: int


class MonitoringConfig:
    """Language-specific monitoring configuration."""
    
    # Language-specific monitoring settings
    LANGUAGE_MONITORING_SETTINGS = {
        "zh": MonitoringSettings(
            max_processing_time=5.0,  # Chinese: longer processing expected
            max_memory_mb=1500,
            error_rate_threshold=0.05,
            alert_level="medium",
            monitoring_frequency=30
        ),
        "ja": MonitoringSettings(
            max_processing_time=4.0,  # Japanese: medium processing time
            max_memory_mb=1200,
            error_rate_threshold=0.04,
            alert_level="medium",
            monitoring_frequency=45
        ),
        "ko": MonitoringSettings(
            max_processing_time=4.0,  # Korean: similar to Japanese
            max_memory_mb=1200,
            error_rate_threshold=0.04,
            alert_level="medium",
            monitoring_frequency=45
        ),
        "ru": MonitoringSettings(
            max_processing_time=3.0,  # Russian: standard processing
            max_memory_mb=1000,
            error_rate_threshold=0.03,
            alert_level="low",
            monitoring_frequency=60
        ),
        "en": MonitoringSettings(
            max_processing_time=2.5,  # English: fastest processing
            max_memory_mb=800,
            error_rate_threshold=0.02,
            alert_level="low",
            monitoring_frequency=60
        )
    }
    
    # Global monitoring settings
    GLOBAL_MONITORING_SETTINGS = {
        "max_history_size": 1000,
        "default_monitoring_interval": 30,
        "enable_alerts": True,
        "enable_metrics_collection": True,
        "enable_performance_tracking": True,
        "alert_retention_days": 7
    }
    
    @classmethod
    def get_language_monitoring_settings(cls, language: str) -> MonitoringSettings:
        """Get monitoring settings for a specific language."""
        return cls.LANGUAGE_MONITORING_SETTINGS.get(
            language, cls.LANGUAGE_MONITORING_SETTINGS["en"]
        )
    
    @classmethod
    def get_global_monitoring_settings(cls) -> Dict[str, Any]:
        """Get global monitoring settings."""
        return cls.GLOBAL_MONITORING_SETTINGS.copy()
    
    @classmethod
    def get_all_language_settings(cls) -> Dict[str, MonitoringSettings]:
        """Get all language monitoring settings."""
        return cls.LANGUAGE_MONITORING_SETTINGS.copy()
    
    @classmethod
    def update_language_settings(cls, language: str, settings: Dict[str, Any]) -> None:
        """Update monitoring settings for a specific language."""
        if language in cls.LANGUAGE_MONITORING_SETTINGS:
            current_settings = cls.LANGUAGE_MONITORING_SETTINGS[language]
            for key, value in settings.items():
                if hasattr(current_settings, key):
                    setattr(current_settings, key, value)
    
    @classmethod
    def get_monitoring_optimization_recommendations(cls, language: str) -> Dict[str, Any]:
        """Get monitoring optimization recommendations for a language."""
        settings = cls.get_language_monitoring_settings(language)
        
        recommendations = {
            "language": language,
            "current_settings": {
                "max_processing_time": settings.max_processing_time,
                "max_memory_mb": settings.max_memory_mb,
                "error_rate_threshold": settings.error_rate_threshold,
                "alert_level": settings.alert_level,
                "monitoring_frequency": settings.monitoring_frequency
            },
            "recommendations": []
        }
        
        # Language-specific recommendations
        if language == "zh":
            recommendations["recommendations"].extend([
                "Use higher processing time thresholds for Chinese content",
                "Implement more frequent monitoring for Chinese processing",
                "Apply medium alert levels for Chinese content",
                "Consider character density in monitoring thresholds"
            ])
        elif language == "ja":
            recommendations["recommendations"].extend([
                "Use moderate processing time thresholds for Japanese",
                "Implement balanced monitoring frequency for Japanese",
                "Apply medium alert levels for Japanese content",
                "Consider honorific patterns in monitoring"
            ])
        elif language == "ko":
            recommendations["recommendations"].extend([
                "Apply similar optimizations as Japanese",
                "Use moderate processing time thresholds for Korean",
                "Implement balanced monitoring for Korean content",
                "Consider formal speech patterns in monitoring"
            ])
        elif language == "ru":
            recommendations["recommendations"].extend([
                "Use standard processing time thresholds for Russian",
                "Implement standard monitoring frequency for Russian",
                "Apply low alert levels for Russian content",
                "Consider case sensitivity in monitoring"
            ])
        
        return recommendations
    
    @classmethod
    def get_alert_strategy(cls, language: str) -> Dict[str, Any]:
        """Get alert strategy for a specific language."""
        settings = cls.get_language_monitoring_settings(language)
        
        if language == "zh":
            return {
                "strategy": "aggressive",
                "alert_threshold": 0.8,
                "escalation_time_minutes": 5,
                "notification_channels": ["log", "email", "dashboard"],
                "auto_recovery_enabled": True
            }
        elif language == "ja":
            return {
                "strategy": "balanced",
                "alert_threshold": 0.85,
                "escalation_time_minutes": 10,
                "notification_channels": ["log", "dashboard"],
                "auto_recovery_enabled": True
            }
        elif language == "ko":
            return {
                "strategy": "balanced",
                "alert_threshold": 0.85,
                "escalation_time_minutes": 10,
                "notification_channels": ["log", "dashboard"],
                "auto_recovery_enabled": True
            }
        else:  # Default for Russian, English, and others
            return {
                "strategy": "conservative",
                "alert_threshold": 0.9,
                "escalation_time_minutes": 15,
                "notification_channels": ["log"],
                "auto_recovery_enabled": False
            }
