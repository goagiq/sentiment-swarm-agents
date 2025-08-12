"""
Language-specific configuration package for enhanced multilingual processing.
This package provides comprehensive configuration files for each language with optimized regex patterns.
"""

from .base_config import BaseLanguageConfig, LanguageConfigFactory
from .chinese_config import ChineseConfig
from .russian_config import RussianConfig
from .english_config import EnglishConfig
from .japanese_config import JapaneseConfig
from .korean_config import KoreanConfig
from .arabic_config import ArabicConfig
from .hindi_config import HindiConfig

__all__ = [
    'BaseLanguageConfig',
    'LanguageConfigFactory', 
    'ChineseConfig',
    'RussianConfig',
    'EnglishConfig',
    'JapaneseConfig',
    'KoreanConfig',
    'ArabicConfig',
    'HindiConfig'
]
