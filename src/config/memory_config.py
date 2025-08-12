"""
Memory Configuration for Phase 2 Performance Optimization.
Language-specific memory management settings and optimization parameters.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class MemorySettings:
    """Memory settings for a specific language."""
    chunk_size: int
    memory_multiplier: float
    cleanup_frequency: float
    max_memory_mb: int
    processing_mode: str


class MemoryConfig:
    """Language-specific memory configuration."""
    
    # Language-specific memory settings
    LANGUAGE_MEMORY_SETTINGS = {
        "zh": MemorySettings(
            chunk_size=5000,  # Chinese: larger chunks due to character density
            memory_multiplier=1.5,
            cleanup_frequency=0.7,
            max_memory_mb=1500,
            processing_mode="streaming"
        ),
        "ja": MemorySettings(
            chunk_size=4000,  # Japanese: medium chunks
            memory_multiplier=1.3,
            cleanup_frequency=0.75,
            max_memory_mb=1200,
            processing_mode="chunked"
        ),
        "ko": MemorySettings(
            chunk_size=4000,  # Korean: medium chunks
            memory_multiplier=1.3,
            cleanup_frequency=0.75,
            max_memory_mb=1200,
            processing_mode="chunked"
        ),
        "ru": MemorySettings(
            chunk_size=3000,  # Russian: smaller chunks
            memory_multiplier=1.1,
            cleanup_frequency=0.8,
            max_memory_mb=1000,
            processing_mode="standard"
        ),
        "en": MemorySettings(
            chunk_size=2500,  # English: standard chunks
            memory_multiplier=1.0,
            cleanup_frequency=0.8,
            max_memory_mb=800,
            processing_mode="standard"
        )
    }
    
    # Global memory settings
    GLOBAL_MEMORY_SETTINGS = {
        "default_max_memory_mb": 1024,
        "cleanup_threshold": 0.8,
        "monitoring_interval_seconds": 30,
        "enable_auto_cleanup": True,
        "enable_memory_tracking": True,
        "enable_weak_references": True
    }
    
    @classmethod
    def get_language_memory_settings(cls, language: str) -> MemorySettings:
        """Get memory settings for a specific language."""
        return cls.LANGUAGE_MEMORY_SETTINGS.get(
            language, cls.LANGUAGE_MEMORY_SETTINGS["en"]
        )
    
    @classmethod
    def get_global_memory_settings(cls) -> Dict[str, Any]:
        """Get global memory settings."""
        return cls.GLOBAL_MEMORY_SETTINGS.copy()
    
    @classmethod
    def get_all_language_settings(cls) -> Dict[str, MemorySettings]:
        """Get all language memory settings."""
        return cls.LANGUAGE_MEMORY_SETTINGS.copy()
    
    @classmethod
    def update_language_settings(cls, language: str, settings: Dict[str, Any]) -> None:
        """Update memory settings for a specific language."""
        if language in cls.LANGUAGE_MEMORY_SETTINGS:
            current_settings = cls.LANGUAGE_MEMORY_SETTINGS[language]
            for key, value in settings.items():
                if hasattr(current_settings, key):
                    setattr(current_settings, key, value)
    
    @classmethod
    def get_memory_optimization_recommendations(cls, language: str) -> Dict[str, Any]:
        """Get memory optimization recommendations for a language."""
        settings = cls.get_language_memory_settings(language)
        
        recommendations = {
            "language": language,
            "current_settings": {
                "chunk_size": settings.chunk_size,
                "memory_multiplier": settings.memory_multiplier,
                "cleanup_frequency": settings.cleanup_frequency,
                "max_memory_mb": settings.max_memory_mb,
                "processing_mode": settings.processing_mode
            },
            "recommendations": []
        }
        
        # Language-specific recommendations
        if language == "zh":
            recommendations["recommendations"].extend([
                "Use larger chunk sizes for Chinese due to character density",
                "Implement streaming processing for large Chinese documents",
                "Apply aggressive memory cleanup for Chinese content",
                "Consider higher memory limits for complex Chinese processing"
            ])
        elif language == "ja":
            recommendations["recommendations"].extend([
                "Use moderate chunk sizes for Japanese mixed text",
                "Implement chunked processing for Japanese content",
                "Apply balanced memory cleanup for Japanese",
                "Consider honorific-aware memory management"
            ])
        elif language == "ko":
            recommendations["recommendations"].extend([
                "Apply similar optimizations as Japanese",
                "Use moderate chunk sizes for Korean text",
                "Implement chunked processing for Korean content",
                "Consider formal speech pattern memory management"
            ])
        elif language == "ru":
            recommendations["recommendations"].extend([
                "Use standard chunk sizes for Russian text",
                "Implement standard processing mode for Russian",
                "Apply conservative memory cleanup for Russian content",
                "Consider case-sensitive memory management"
            ])
        
        return recommendations
    
    @classmethod
    def get_memory_cleanup_strategy(cls, language: str) -> Dict[str, Any]:
        """Get memory cleanup strategy for a specific language."""
        settings = cls.get_language_memory_settings(language)
        
        if language == "zh":
            return {
                "strategy": "aggressive",
                "cleanup_threshold": settings.cleanup_frequency,
                "cleanup_interval_seconds": 60,
                "memory_reduction_target": 0.3,
                "enable_compression": True
            }
        elif language == "ja":
            return {
                "strategy": "balanced",
                "cleanup_threshold": settings.cleanup_frequency,
                "cleanup_interval_seconds": 90,
                "memory_reduction_target": 0.25,
                "enable_compression": True
            }
        elif language == "ko":
            return {
                "strategy": "balanced",
                "cleanup_threshold": settings.cleanup_frequency,
                "cleanup_interval_seconds": 90,
                "memory_reduction_target": 0.25,
                "enable_compression": True
            }
        else:  # Default for Russian, English, and others
            return {
                "strategy": "conservative",
                "cleanup_threshold": settings.cleanup_frequency,
                "cleanup_interval_seconds": 120,
                "memory_reduction_target": 0.2,
                "enable_compression": False
            }
