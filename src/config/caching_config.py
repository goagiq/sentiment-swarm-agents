"""
Caching Configuration for Phase 2 Performance Optimization.
Language-specific caching settings and optimization parameters.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class CacheSettings:
    """Cache settings for a specific language."""
    memory_ttl_multiplier: float
    compression_level: int
    cache_priority: str
    max_cache_size: int
    cleanup_frequency: float


class CachingConfig:
    """Language-specific caching configuration."""
    
    # Language-specific cache settings
    LANGUAGE_CACHE_SETTINGS = {
        "zh": CacheSettings(
            memory_ttl_multiplier=2.0,  # Longer caching for Chinese
            compression_level=6,  # Higher compression for Chinese text
            cache_priority="high",
            max_cache_size=2000,
            cleanup_frequency=0.7
        ),
        "ja": CacheSettings(
            memory_ttl_multiplier=1.5,  # Medium caching for Japanese
            compression_level=5,
            cache_priority="medium",
            max_cache_size=1500,
            cleanup_frequency=0.75
        ),
        "ko": CacheSettings(
            memory_ttl_multiplier=1.5,  # Medium caching for Korean
            compression_level=5,
            cache_priority="medium",
            max_cache_size=1500,
            cleanup_frequency=0.75
        ),
        "ru": CacheSettings(
            memory_ttl_multiplier=1.2,  # Slightly longer for Russian
            compression_level=4,
            cache_priority="medium",
            max_cache_size=1200,
            cleanup_frequency=0.8
        ),
        "en": CacheSettings(
            memory_ttl_multiplier=1.0,  # Standard caching
            compression_level=3,
            cache_priority="normal",
            max_cache_size=1000,
            cleanup_frequency=0.8
        )
    }
    
    # Global cache settings
    GLOBAL_CACHE_SETTINGS = {
        "memory_cache_size": 1000,
        "disk_cache_size_mb": 1024,
        "default_ttl_seconds": 3600,
        "cleanup_interval_seconds": 300,
        "compression_enabled": True,
        "cache_persistence": True
    }
    
    @classmethod
    def get_language_cache_settings(cls, language: str) -> CacheSettings:
        """Get cache settings for a specific language."""
        return cls.LANGUAGE_CACHE_SETTINGS.get(language, cls.LANGUAGE_CACHE_SETTINGS["en"])
    
    @classmethod
    def get_global_cache_settings(cls) -> Dict[str, Any]:
        """Get global cache settings."""
        return cls.GLOBAL_CACHE_SETTINGS.copy()
    
    @classmethod
    def get_all_language_settings(cls) -> Dict[str, CacheSettings]:
        """Get all language cache settings."""
        return cls.LANGUAGE_CACHE_SETTINGS.copy()
    
    @classmethod
    def update_language_settings(cls, language: str, settings: Dict[str, Any]) -> None:
        """Update cache settings for a specific language."""
        if language in cls.LANGUAGE_CACHE_SETTINGS:
            current_settings = cls.LANGUAGE_CACHE_SETTINGS[language]
            for key, value in settings.items():
                if hasattr(current_settings, key):
                    setattr(current_settings, key, value)
    
    @classmethod
    def get_cache_optimization_recommendations(cls, language: str) -> Dict[str, Any]:
        """Get cache optimization recommendations for a language."""
        settings = cls.get_language_cache_settings(language)
        
        recommendations = {
            "language": language,
            "current_settings": {
                "memory_ttl_multiplier": settings.memory_ttl_multiplier,
                "compression_level": settings.compression_level,
                "cache_priority": settings.cache_priority,
                "max_cache_size": settings.max_cache_size
            },
            "recommendations": []
        }
        
        # Language-specific recommendations
        if language == "zh":
            recommendations["recommendations"].extend([
                "Use higher compression for Chinese text due to character density",
                "Implement longer TTL for frequently accessed Chinese content",
                "Consider aggressive caching strategy for Chinese documents"
            ])
        elif language == "ja":
            recommendations["recommendations"].extend([
                "Balance compression and performance for Japanese mixed text",
                "Use moderate caching for Japanese content",
                "Consider honorific-aware caching strategies"
            ])
        elif language == "ko":
            recommendations["recommendations"].extend([
                "Apply similar optimizations as Japanese",
                "Use moderate compression for Korean text",
                "Consider formal speech pattern caching"
            ])
        elif language == "ru":
            recommendations["recommendations"].extend([
                "Use standard compression for Russian text",
                "Implement moderate caching for Russian content",
                "Consider case-sensitive caching for Russian"
            ])
        
        return recommendations
