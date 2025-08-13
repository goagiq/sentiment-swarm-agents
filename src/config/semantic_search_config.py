"""
Semantic Search Configuration

This module provides configuration settings for semantic search functionality,
including search parameters, performance settings, and language-specific options.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class SearchType(Enum):
    """Types of semantic search available."""
    SEMANTIC = "semantic"
    CONCEPTUAL = "conceptual"
    MULTILINGUAL = "multilingual"
    CROSS_CONTENT = "cross_content"
    KNOWLEDGE_GRAPH = "knowledge_graph"


class ContentType(Enum):
    """Supported content types for search."""
    TEXT = "text"
    PDF = "pdf"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    WEB = "web"
    DOCUMENT = "document"


@dataclass
class SearchParameters:
    """Default search parameters."""
    # General search settings
    default_n_results: int = 10
    max_n_results: int = 100
    default_similarity_threshold: float = 0.7
    min_similarity_threshold: float = 0.3
    max_similarity_threshold: float = 0.95
    
    # Performance settings
    search_timeout_seconds: int = 30
    batch_size: int = 50
    cache_results: bool = True
    cache_ttl_seconds: int = 3600
    
    # Language settings
    default_language: str = "en"
    supported_languages: List[str] = None
    
    # Content type settings
    default_content_types: List[str] = None
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = [
                "en", "zh", "ru", "ja", "ko", "ar", "hi", "es", "fr", "de"
            ]
        
        if self.default_content_types is None:
            self.default_content_types = [
                "text", "pdf", "document", "web"
            ]


@dataclass
class LanguageSpecificSettings:
    """Language-specific search settings."""
    language_code: str
    similarity_threshold: float
    content_types: List[str]
    search_strategies: List[str]
    
    # Language-specific search parameters
    use_stemming: bool = True
    use_synonyms: bool = True
    case_sensitive: bool = False
    
    # Cultural and linguistic considerations
    cultural_context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.cultural_context is None:
            self.cultural_context = {}


class SemanticSearchConfig:
    """Main configuration class for semantic search."""
    
    def __init__(self):
        self.search_params = SearchParameters()
        self.language_settings = self._initialize_language_settings()
        self.search_strategies = self._initialize_search_strategies()
    
    def _initialize_language_settings(self) -> Dict[str, LanguageSpecificSettings]:
        """Initialize language-specific settings."""
        settings = {}
        
        # English settings
        settings["en"] = LanguageSpecificSettings(
            language_code="en",
            similarity_threshold=0.7,
            content_types=["text", "pdf", "document", "web"],
            search_strategies=["semantic", "conceptual", "keyword"],
            use_stemming=True,
            use_synonyms=True,
            case_sensitive=False
        )
        
        # Chinese settings
        settings["zh"] = LanguageSpecificSettings(
            language_code="zh",
            similarity_threshold=0.65,
            content_types=["text", "pdf", "document"],
            search_strategies=["semantic", "conceptual", "character"],
            use_stemming=False,
            use_synonyms=True,
            case_sensitive=False,
            cultural_context={
                "use_character_level": True,
                "consider_tone": False
            }
        )
        
        # Russian settings
        settings["ru"] = LanguageSpecificSettings(
            language_code="ru",
            similarity_threshold=0.7,
            content_types=["text", "pdf", "document"],
            search_strategies=["semantic", "conceptual", "morphological"],
            use_stemming=True,
            use_synonyms=True,
            case_sensitive=False
        )
        
        # Japanese settings
        settings["ja"] = LanguageSpecificSettings(
            language_code="ja",
            similarity_threshold=0.65,
            content_types=["text", "pdf", "document"],
            search_strategies=["semantic", "conceptual", "kanji"],
            use_stemming=False,
            use_synonyms=True,
            case_sensitive=False,
            cultural_context={
                "use_kanji_level": True,
                "consider_particles": True
            }
        )
        
        # Korean settings
        settings["ko"] = LanguageSpecificSettings(
            language_code="ko",
            similarity_threshold=0.65,
            content_types=["text", "pdf", "document"],
            search_strategies=["semantic", "conceptual", "morpheme"],
            use_stemming=True,
            use_synonyms=True,
            case_sensitive=False
        )
        
        # Arabic settings
        settings["ar"] = LanguageSpecificSettings(
            language_code="ar",
            similarity_threshold=0.65,
            content_types=["text", "pdf", "document"],
            search_strategies=["semantic", "conceptual", "root"],
            use_stemming=True,
            use_synonyms=True,
            case_sensitive=False,
            cultural_context={
                "use_root_analysis": True,
                "consider_diacritics": False
            }
        )
        
        # Hindi settings
        settings["hi"] = LanguageSpecificSettings(
            language_code="hi",
            similarity_threshold=0.65,
            content_types=["text", "pdf", "document"],
            search_strategies=["semantic", "conceptual", "morphological"],
            use_stemming=True,
            use_synonyms=True,
            case_sensitive=False
        )
        
        return settings
    
    def _initialize_search_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize search strategy configurations."""
        return {
            "semantic": {
                "description": "Semantic similarity search using embeddings",
                "default_threshold": 0.7,
                "performance_optimized": True,
                "supports_multilingual": True
            },
            "conceptual": {
                "description": "Concept-based search for related ideas",
                "default_threshold": 0.6,
                "performance_optimized": False,
                "supports_multilingual": True
            },
            "multilingual": {
                "description": "Cross-language semantic search",
                "default_threshold": 0.65,
                "performance_optimized": True,
                "supports_multilingual": True
            },
            "cross_content": {
                "description": "Search across different content types",
                "default_threshold": 0.7,
                "performance_optimized": True,
                "supports_multilingual": True
            },
            "knowledge_graph": {
                "description": "Knowledge graph-based search",
                "default_threshold": 0.7,
                "performance_optimized": False,
                "supports_multilingual": True
            }
        }
    
    def get_language_settings(self, language_code: str) -> LanguageSpecificSettings:
        """Get settings for a specific language."""
        return self.language_settings.get(
            language_code, 
            self.language_settings["en"]
        )
    
    def get_search_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for a specific search strategy."""
        return self.search_strategies.get(strategy_name, {})
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(self.language_settings.keys())
    
    def get_supported_content_types(self) -> List[str]:
        """Get list of supported content types."""
        return [ct.value for ct in ContentType]
    
    def validate_search_parameters(
        self,
        n_results: int,
        similarity_threshold: float,
        language: str
    ) -> Dict[str, Any]:
        """Validate and adjust search parameters."""
        # Validate n_results
        if n_results > self.search_params.max_n_results:
            n_results = self.search_params.max_n_results
        elif n_results < 1:
            n_results = self.search_params.default_n_results
        
        # Validate similarity threshold
        if similarity_threshold > self.search_params.max_similarity_threshold:
            similarity_threshold = self.search_params.max_similarity_threshold
        elif similarity_threshold < self.search_params.min_similarity_threshold:
            similarity_threshold = self.search_params.min_similarity_threshold
        
        # Get language-specific threshold if available
        lang_settings = self.get_language_settings(language)
        if similarity_threshold < lang_settings.similarity_threshold:
            similarity_threshold = lang_settings.similarity_threshold
        
        return {
            "n_results": n_results,
            "similarity_threshold": similarity_threshold,
            "language": language,
            "content_types": lang_settings.content_types,
            "search_strategies": lang_settings.search_strategies
        }


# Global configuration instance
semantic_search_config = SemanticSearchConfig()
