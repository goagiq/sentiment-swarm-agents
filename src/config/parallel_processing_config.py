"""
Parallel Processing Configuration for Phase 2 Performance Optimization.
Language-specific parallel processing settings and optimization parameters.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ParallelSettings:
    """Parallel processing settings for a specific language."""
    chunk_size: int
    max_workers: int
    processing_mode: str
    load_balancing: str
    timeout_seconds: float


class ParallelProcessingConfig:
    """Language-specific parallel processing configuration."""
    
    # Language-specific parallel processing settings
    LANGUAGE_PARALLEL_SETTINGS = {
        "zh": ParallelSettings(
            chunk_size=1000,  # Chinese: larger chunks due to character density
            max_workers=6,
            processing_mode="streaming",
            load_balancing="adaptive",
            timeout_seconds=30.0
        ),
        "ja": ParallelSettings(
            chunk_size=800,  # Japanese: medium chunks
            max_workers=5,
            processing_mode="chunked",
            load_balancing="round_robin",
            timeout_seconds=25.0
        ),
        "ko": ParallelSettings(
            chunk_size=800,  # Korean: medium chunks
            max_workers=5,
            processing_mode="chunked",
            load_balancing="round_robin",
            timeout_seconds=25.0
        ),
        "ru": ParallelSettings(
            chunk_size=600,  # Russian: smaller chunks
            max_workers=4,
            processing_mode="standard",
            load_balancing="simple",
            timeout_seconds=20.0
        ),
        "en": ParallelSettings(
            chunk_size=500,  # English: standard chunks
            max_workers=4,
            processing_mode="standard",
            load_balancing="simple",
            timeout_seconds=15.0
        )
    }
    
    # Global parallel processing settings
    GLOBAL_PARALLEL_SETTINGS = {
        "max_total_workers": 32,
        "max_total_processes": 8,
        "default_timeout_seconds": 30,
        "retry_attempts": 3,
        "retry_delay_seconds": 1.0,
        "enable_load_balancing": True,
        "enable_monitoring": True
    }
    
    @classmethod
    def get_language_parallel_settings(cls, language: str) -> ParallelSettings:
        """Get parallel processing settings for a specific language."""
        return cls.LANGUAGE_PARALLEL_SETTINGS.get(
            language, cls.LANGUAGE_PARALLEL_SETTINGS["en"]
        )
    
    @classmethod
    def get_global_parallel_settings(cls) -> Dict[str, Any]:
        """Get global parallel processing settings."""
        return cls.GLOBAL_PARALLEL_SETTINGS.copy()
    
    @classmethod
    def get_all_language_settings(cls) -> Dict[str, ParallelSettings]:
        """Get all language parallel processing settings."""
        return cls.LANGUAGE_PARALLEL_SETTINGS.copy()
    
    @classmethod
    def update_language_settings(cls, language: str, settings: Dict[str, Any]) -> None:
        """Update parallel processing settings for a specific language."""
        if language in cls.LANGUAGE_PARALLEL_SETTINGS:
            current_settings = cls.LANGUAGE_PARALLEL_SETTINGS[language]
            for key, value in settings.items():
                if hasattr(current_settings, key):
                    setattr(current_settings, key, value)
    
    @classmethod
    def get_parallel_optimization_recommendations(cls, language: str) -> Dict[str, Any]:
        """Get parallel processing optimization recommendations for a language."""
        settings = cls.get_language_parallel_settings(language)
        
        recommendations = {
            "language": language,
            "current_settings": {
                "chunk_size": settings.chunk_size,
                "max_workers": settings.max_workers,
                "processing_mode": settings.processing_mode,
                "load_balancing": settings.load_balancing,
                "timeout_seconds": settings.timeout_seconds
            },
            "recommendations": []
        }
        
        # Language-specific recommendations
        if language == "zh":
            recommendations["recommendations"].extend([
                "Use larger chunk sizes for Chinese due to character density",
                "Implement streaming processing for large Chinese documents",
                "Apply adaptive load balancing for Chinese content",
                "Consider longer timeouts for complex Chinese processing"
            ])
        elif language == "ja":
            recommendations["recommendations"].extend([
                "Use moderate chunk sizes for Japanese mixed text",
                "Implement chunked processing for Japanese content",
                "Apply round-robin load balancing for Japanese",
                "Consider honorific-aware parallel processing"
            ])
        elif language == "ko":
            recommendations["recommendations"].extend([
                "Apply similar optimizations as Japanese",
                "Use moderate chunk sizes for Korean text",
                "Implement chunked processing for Korean content",
                "Consider formal speech pattern parallel processing"
            ])
        elif language == "ru":
            recommendations["recommendations"].extend([
                "Use standard chunk sizes for Russian text",
                "Implement standard processing mode for Russian",
                "Apply simple load balancing for Russian content",
                "Consider case-sensitive parallel processing"
            ])
        
        return recommendations
    
    @classmethod
    def get_worker_allocation_strategy(cls, language: str) -> Dict[str, Any]:
        """Get worker allocation strategy for a specific language."""
        settings = cls.get_language_parallel_settings(language)
        
        if language == "zh":
            return {
                "strategy": "adaptive",
                "initial_workers": settings.max_workers,
                "max_workers": settings.max_workers * 2,
                "scaling_factor": 1.5,
                "load_threshold": 0.7
            }
        elif language == "ja":
            return {
                "strategy": "balanced",
                "initial_workers": settings.max_workers,
                "max_workers": settings.max_workers * 1.5,
                "scaling_factor": 1.2,
                "load_threshold": 0.8
            }
        elif language == "ko":
            return {
                "strategy": "balanced",
                "initial_workers": settings.max_workers,
                "max_workers": settings.max_workers * 1.5,
                "scaling_factor": 1.2,
                "load_threshold": 0.8
            }
        else:  # Default for Russian, English, and others
            return {
                "strategy": "fixed",
                "initial_workers": settings.max_workers,
                "max_workers": settings.max_workers,
                "scaling_factor": 1.0,
                "load_threshold": 0.9
            }
