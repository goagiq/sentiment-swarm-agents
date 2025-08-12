"""
Language Processing Service for isolated language-specific processing.
This service uses the new isolated language configurations to prevent conflicts.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import re

from src.config.language_config import LanguageConfigFactory, BaseLanguageConfig
from src.core.error_handler import with_error_handling


class LanguageProcessingService:
    """Service for language-specific processing using isolated configurations."""
    
    def __init__(self):
        self.config_factory = LanguageConfigFactory()
        self._cache = {}
    
    @with_error_handling("language_detection")
    def detect_language(self, text: str) -> str:
        """Detect language from text using isolated configurations."""
        if not text or len(text.strip()) < 10:
            return "en"  # Default to English for short text
        
        # Use the factory's language detection
        detected_language = self.config_factory.detect_language_from_text(text)
        
        # Cache the result for performance
        text_hash = hash(text[:100])  # Use first 100 chars for caching
        self._cache[text_hash] = detected_language
        
        return detected_language
    
    @with_error_handling("language_config_loading")
    def get_language_config(self, language_code: str) -> BaseLanguageConfig:
        """Get isolated configuration for a specific language."""
        try:
            return self.config_factory.get_config(language_code)
        except ValueError:
            # Fallback to English if language not found
            return self.config_factory.get_config("en")
    
    @with_error_handling("entity_extraction")
    def extract_entities_with_config(self, text: str, language_code: str = "auto") -> Dict[str, Any]:
        """Extract entities using language-specific configuration."""
        if language_code == "auto":
            language_code = self.detect_language(text)
        
        config = self.get_language_config(language_code)
        
        # Get language-specific patterns and settings
        patterns = config.get_entity_patterns()
        settings = config.get_processing_settings()
        
        entities = {
            "person": [],
            "organization": [],
            "location": [],
            "concept": []
        }
        
        # Extract entities using language-specific patterns
        for entity_type, pattern_list in [
            ("person", patterns.person),
            ("organization", patterns.organization),
            ("location", patterns.location),
            ("concept", patterns.concept)
        ]:
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = " ".join(match)
                    
                    # Apply language-specific filtering
                    if self._is_valid_entity(match, settings):
                        if match not in entities[entity_type]:
                            entities[entity_type].append(match)
        
        return {
            "language": language_code,
            "entities": entities,
            "settings": {
                "min_entity_length": settings.min_entity_length,
                "max_entity_length": settings.max_entity_length,
                "confidence_threshold": settings.confidence_threshold,
                "use_enhanced_extraction": settings.use_enhanced_extraction
            }
        }
    
    @with_error_handling("relationship_mapping")
    def map_relationships_with_config(self, text: str, entities: List[str], language_code: str = "auto") -> Dict[str, Any]:
        """Map relationships using language-specific configuration."""
        if language_code == "auto":
            language_code = self.detect_language(text)
        
        config = self.get_language_config(language_code)
        settings = config.get_processing_settings()
        
        # Get language-specific relationship prompt
        if hasattr(config, 'get_hierarchical_relationship_prompt') and settings.use_hierarchical_relationships:
            prompt = config.get_hierarchical_relationship_prompt(entities, text)
        else:
            prompt = config.get_relationship_prompt(entities, text)
        
        # Get language-specific relationship templates
        templates = config.get_relationship_templates()
        
        return {
            "language": language_code,
            "prompt": prompt,
            "templates": templates,
            "settings": {
                "relationship_prompt_simplified": settings.relationship_prompt_simplified,
                "use_hierarchical_relationships": settings.use_hierarchical_relationships,
                "entity_clustering_enabled": settings.entity_clustering_enabled,
                "fallback_strategies": settings.fallback_strategies
            }
        }
    
    def _is_valid_entity(self, entity: str, settings) -> bool:
        """Check if entity is valid according to language-specific settings."""
        if not entity or not entity.strip():
            return False
        
        entity = entity.strip()
        
        # Check length constraints
        if len(entity) < settings.min_entity_length:
            return False
        
        if len(entity) > settings.max_entity_length:
            return False
        
        # Check for common invalid patterns
        invalid_patterns = [
            r'^\d+$',  # Pure numbers
            r'^[^\w\s]+$',  # Pure punctuation
            r'^\s+$',  # Pure whitespace
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, entity):
                return False
        
        return True
    
    @with_error_handling("language_validation")
    def validate_language_processing(self, language_code: str) -> Dict[str, Any]:
        """Validate that a language configuration is properly set up."""
        try:
            config = self.get_language_config(language_code)
            
            return {
                "language_code": language_code,
                "language_name": config.language_name,
                "has_entity_patterns": config.entity_patterns is not None,
                "has_processing_settings": config.processing_settings is not None,
                "has_relationship_templates": bool(config.relationship_templates),
                "has_detection_patterns": bool(config.detection_patterns),
                "is_valid": True
            }
        except Exception as e:
            return {
                "language_code": language_code,
                "is_valid": False,
                "error": str(e)
            }
    
    def get_available_languages(self) -> List[str]:
        """Get list of available language codes."""
        return self.config_factory.get_available_languages()
    
    def get_language_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available languages."""
        languages = {}
        
        for language_code in self.get_available_languages():
            validation = self.validate_language_processing(language_code)
            languages[language_code] = validation
        
        return languages
