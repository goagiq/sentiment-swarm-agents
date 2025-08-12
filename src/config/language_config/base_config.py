"""
Base configuration classes for language-specific processing.
Provides common functionality and abstract methods for language-specific implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class EntityPatterns:
    """Entity patterns for a specific language."""
    person: List[str]
    organization: List[str]
    location: List[str]
    concept: List[str]


@dataclass
class ProcessingSettings:
    """Processing settings for a specific language."""
    min_entity_length: int
    max_entity_length: int
    confidence_threshold: float
    use_enhanced_extraction: bool
    relationship_prompt_simplified: bool
    use_hierarchical_relationships: bool = False
    entity_clustering_enabled: bool = False
    fallback_strategies: List[str] = None


@dataclass
class OllamaModelConfig:
    """Ollama model configuration for a specific language."""
    model_id: str
    temperature: float
    max_tokens: int
    system_prompt: str
    keep_alive: str = "10m"


class BaseLanguageConfig(ABC):
    """Base class for language-specific configurations."""
    
    def __init__(self):
        self.language_code: str = ""
        self.language_name: str = ""
        self.entity_patterns: EntityPatterns = None
        self.processing_settings: ProcessingSettings = None
        self.relationship_templates: Dict[str, str] = {}
        self.detection_patterns: List[str] = []
        self.ollama_config: Dict[str, Dict[str, Any]] = {}
        
    @abstractmethod
    def get_entity_patterns(self) -> EntityPatterns:
        """Get entity patterns for this language."""
        pass
    
    @abstractmethod
    def get_processing_settings(self) -> ProcessingSettings:
        """Get processing settings for this language."""
        pass
    
    @abstractmethod
    def get_relationship_templates(self) -> Dict[str, str]:
        """Get relationship templates for this language."""
        pass
    
    @abstractmethod
    def get_detection_patterns(self) -> List[str]:
        """Get language detection patterns."""
        pass
    
    def get_ollama_config(self) -> Dict[str, Dict[str, Any]]:
        """Get Ollama model configuration for this language."""
        return {
            "text_model": {
                "model_id": "llama3.2:latest",
                "temperature": 0.7,
                "max_tokens": 1000,
                "system_prompt": "You are a professional text analysis assistant.",
                "keep_alive": "10m"
            },
            "vision_model": {
                "model_id": "llava:latest",
                "temperature": 0.7,
                "max_tokens": 1000,
                "system_prompt": "You are a professional image analysis assistant.",
                "keep_alive": "15m"
            },
            "audio_model": {
                "model_id": "llava:latest",
                "temperature": 0.7,
                "max_tokens": 1000,
                "system_prompt": "You are a professional audio analysis assistant.",
                "keep_alive": "15m"
            }
        }
    
    def get_ollama_model_config(self, model_type: str = "text") -> Optional[Dict[str, Any]]:
        """Get specific Ollama model configuration."""
        return self.ollama_config.get(model_type)
    
    def detect_language(self, text: str) -> bool:
        """Detect if text is in this language."""
        if not self.detection_patterns:
            return False
        
        text_lower = text.lower()
        for pattern in self.detection_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def get_relationship_prompt(self, entities: List[str], text: str) -> str:
        """Get language-specific relationship prompt."""
        if self.processing_settings.relationship_prompt_simplified:
            return self._get_simplified_prompt(entities, text)
        else:
            return self._get_standard_prompt(entities, text)
    
    def _get_simplified_prompt(self, entities: List[str], text: str) -> str:
        """Get simplified relationship prompt."""
        entities_str = ", ".join(entities[:10])  # Limit to first 10 entities
        return f"""
Analyze the following text and identify relationships between these entities: {entities_str}

Text: {text[:1000]}...

Please identify relationships in this format:
Entity1 | Relationship | Entity2

Focus on the most important relationships. Be concise and accurate.
"""
    
    def _get_standard_prompt(self, entities: List[str], text: str) -> str:
        """Get standard relationship prompt."""
        entities_str = ", ".join(entities[:10])
        return f"""
Please analyze the following text and identify relationships between the entities: {entities_str}

Text: {text[:1000]}...

Please provide relationships in the following format:
Entity1 | Relationship Type | Entity2

Relationship types should be specific and meaningful, such as:
- WORKS_FOR, LOCATED_IN, PART_OF, CREATED_BY, USES, IMPLEMENTS
- SIMILAR_TO, OPPOSES, SUPPORTS, LEADS_TO, DEPENDS_ON

Provide detailed and accurate relationships based on the text content.
"""


class LanguageConfigFactory:
    """Factory for creating language-specific configurations."""
    
    _configs = {}
    
    @classmethod
    def register_config(cls, language_code: str, config_class: type):
        """Register a language configuration."""
        cls._configs[language_code] = config_class
    
    @classmethod
    def get_config(cls, language_code: str) -> BaseLanguageConfig:
        """Get configuration for a specific language."""
        if language_code not in cls._configs:
            raise ValueError(f"No configuration found for language: {language_code}")
        
        config_class = cls._configs[language_code]
        return config_class()
    
    @classmethod
    def detect_language_from_text(cls, text: str) -> str:
        """Detect language from text using registered configurations."""
        best_match = "en"  # Default to English
        best_score = 0
        
        for language_code, config_class in cls._configs.items():
            config = config_class()
            if config.detect_language(text):
                # Count matching patterns for confidence
                score = sum(1 for pattern in config.detection_patterns 
                          if re.search(pattern, text.lower()))
                if score > best_score:
                    best_score = score
                    best_match = language_code
        
        return best_match
    
    @classmethod
    def get_available_languages(cls) -> List[str]:
        """Get list of available language codes."""
        return list(cls._configs.keys())


# Register all language configurations
def register_language_configs():
    """Register all available language configurations."""
    from .chinese_config import ChineseConfig
    from .russian_config import RussianConfig
    from .english_config import EnglishConfig
    from .japanese_config import JapaneseConfig
    from .korean_config import KoreanConfig
    from .arabic_config import ArabicConfig
    from .hindi_config import HindiConfig
    
    LanguageConfigFactory.register_config("zh", ChineseConfig)
    LanguageConfigFactory.register_config("ru", RussianConfig)
    LanguageConfigFactory.register_config("en", EnglishConfig)
    LanguageConfigFactory.register_config("ja", JapaneseConfig)
    LanguageConfigFactory.register_config("ko", KoreanConfig)
    LanguageConfigFactory.register_config("ar", ArabicConfig)
    LanguageConfigFactory.register_config("hi", HindiConfig)


# Auto-register configurations when module is imported
register_language_configs()
