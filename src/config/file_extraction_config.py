"""
File Extraction Configuration with language-specific optimizations.
This file stores language-specific extraction settings, chunking strategies, and parallel processing parameters.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
import os


@dataclass
class LanguageExtractionConfig:
    """Language-specific extraction configuration."""
    # Parallel processing settings
    max_workers: int = 4
    chunk_size: int = 2  # Increased from 1 for better parallelization
    
    # OCR and text extraction settings
    ocr_confidence_threshold: float = 0.7
    min_text_length: int = 10
    max_text_length: int = 50000
    
    # Language-specific patterns for text validation
    text_validation_patterns: List[str] = None
    noise_filter_patterns: List[str] = None
    
    # Memory management
    memory_cleanup_threshold: int = 5
    max_memory_usage_mb: int = 512
    
    # Timeout settings
    timeout_per_page: int = 60
    timeout_per_chunk: int = 300
    
    def __post_init__(self):
        if self.text_validation_patterns is None:
            self.text_validation_patterns = []
        if self.noise_filter_patterns is None:
            self.noise_filter_patterns = []


# Language-specific extraction configurations
LANGUAGE_EXTRACTION_CONFIGS = {
    "en": LanguageExtractionConfig(
        max_workers=4,
        chunk_size=3,  # English text is typically shorter
        ocr_confidence_threshold=0.7,
        min_text_length=5,
        text_validation_patterns=[
            r'[a-zA-Z]',  # Must contain letters
            r'\b\w+\b',   # Must contain words
        ],
        noise_filter_patterns=[
            r'^\s*$',     # Empty lines
            r'^\d+$',     # Just numbers
        ]
    ),
    
    "zh": LanguageExtractionConfig(
        max_workers=6,  # Chinese text is denser, needs more workers
        chunk_size=1,   # Chinese characters are more complex, smaller chunks
        ocr_confidence_threshold=0.8,  # Higher confidence for Chinese
        min_text_length=3,  # Chinese characters are more meaningful
        text_validation_patterns=[
            r'[\u4e00-\u9fff]',  # Must contain Chinese characters
            r'[\u4e00-\u9fff]{2,}',  # Must contain multiple Chinese characters
        ],
        noise_filter_patterns=[
            r'^\s*$',     # Empty lines
            r'^[^\u4e00-\u9fff]*$',  # Lines without Chinese characters
        ]
    ),
    
    "ru": LanguageExtractionConfig(
        max_workers=5,  # Russian text is medium complexity
        chunk_size=2,   # Medium chunk size
        ocr_confidence_threshold=0.75,  # Medium confidence
        min_text_length=8,  # Russian words are longer
        text_validation_patterns=[
            r'[а-яёА-ЯЁ]',  # Must contain Cyrillic characters
            r'\b[а-яёА-ЯЁ]+\b',  # Must contain Cyrillic words
        ],
        noise_filter_patterns=[
            r'^\s*$',     # Empty lines
            r'^[^а-яёА-ЯЁ]*$',  # Lines without Cyrillic characters
        ]
    )
}

# Default configuration for unknown languages
DEFAULT_EXTRACTION_CONFIG = LanguageExtractionConfig(
    max_workers=4,
    chunk_size=2,
    ocr_confidence_threshold=0.7,
    min_text_length=10,
    text_validation_patterns=[
        r'[a-zA-Zа-яёА-ЯЁ\u4e00-\u9fff]',  # Any script
    ],
    noise_filter_patterns=[
        r'^\s*$',  # Empty lines
    ]
)


def get_extraction_config(language: str = "en") -> LanguageExtractionConfig:
    """Get language-specific extraction configuration."""
    return LANGUAGE_EXTRACTION_CONFIGS.get(language, DEFAULT_EXTRACTION_CONFIG)


def get_optimal_chunk_size(language: str, text_length: int) -> int:
    """Calculate optimal chunk size based on language and text length."""
    config = get_extraction_config(language)
    base_chunk_size = config.chunk_size
    
    # Adjust based on text length
    if text_length > 100000:  # Very long text
        return max(1, base_chunk_size // 2)
    elif text_length > 50000:  # Long text
        return base_chunk_size
    elif text_length > 10000:  # Medium text
        return min(5, base_chunk_size + 1)
    else:  # Short text
        return min(8, base_chunk_size + 2)


def get_optimal_workers(language: str, system_cores: int = None) -> int:
    """Calculate optimal number of workers based on language and system resources."""
    if system_cores is None:
        system_cores = os.cpu_count() or 4
    
    config = get_extraction_config(language)
    base_workers = config.max_workers
    
    # Adjust based on system resources
    optimal_workers = min(base_workers, system_cores - 1)  # Leave one core free
    return max(1, optimal_workers)  # At least 1 worker


# Performance optimization settings
PERFORMANCE_CONFIG = {
    "enable_dynamic_chunking": True,
    "enable_memory_monitoring": True,
    "enable_progress_tracking": True,
    "enable_adaptive_timeouts": True,
    "max_concurrent_pdfs": 2,  # Limit concurrent PDF processing
    "memory_threshold_mb": 1024,  # Memory threshold for cleanup
    "progress_update_interval": 0.5,  # Progress update frequency in seconds
}


def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration."""
    return PERFORMANCE_CONFIG.copy()
