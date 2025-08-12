"""
Configuration Validator for Phase 3 optimization.
Provides comprehensive validation for language configurations, regex patterns, and processing settings.
"""

import re
from typing import Dict, List, Any, Optional
import logging


class ConfigValidator:
    """Configuration validator with comprehensive validation rules."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.required_fields = {
            "entity_patterns": ["person", "organization", "location", "concept"],
            "processing_settings": ["min_entity_length", "max_entity_length", 
                                  "confidence_threshold", "use_enhanced_extraction"],
            "detection_patterns": ["language_indicators", "script_patterns"]
        }
    
    def validate_language_config(self, config: dict) -> bool:
        """Validate language configuration structure."""
        try:
            # Check required top-level fields
            if not isinstance(config, dict):
                self.logger.error("Configuration must be a dictionary")
                return False
            
            # Validate entity patterns
            if "entity_patterns" in config:
                if not self._validate_entity_patterns(config["entity_patterns"]):
                    return False
            
            # Validate processing settings
            if "processing_settings" in config:
                if not self._validate_processing_settings(config["processing_settings"]):
                    return False
            
            # Validate detection patterns
            if "detection_patterns" in config:
                if not self._validate_detection_patterns(config["detection_patterns"]):
                    return False
            
            # Validate regex patterns if present
            if "regex_patterns" in config:
                if not self._validate_regex_patterns(config["regex_patterns"]):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating language config: {e}")
            return False
    
    def _validate_entity_patterns(self, patterns: dict) -> bool:
        """Validate entity patterns structure."""
        try:
            if not isinstance(patterns, dict):
                self.logger.error("Entity patterns must be a dictionary")
                return False
            
            # Check required entity types
            required_entities = self.required_fields["entity_patterns"]
            for entity_type in required_entities:
                if entity_type not in patterns:
                    self.logger.warning(f"Missing entity type: {entity_type}")
                    continue
                
                if not isinstance(patterns[entity_type], list):
                    self.logger.error(f"Entity patterns for {entity_type} must be a list")
                    return False
                
                # Validate each regex pattern
                for pattern in patterns[entity_type]:
                    if not self._validate_regex_pattern(pattern):
                        self.logger.error(f"Invalid regex pattern in {entity_type}: {pattern}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating entity patterns: {e}")
            return False
    
    def _validate_processing_settings(self, settings: dict) -> bool:
        """Validate processing settings."""
        try:
            if not isinstance(settings, dict):
                self.logger.error("Processing settings must be a dictionary")
                return False
            
            # Check required settings
            required_settings = self.required_fields["processing_settings"]
            for setting in required_settings:
                if setting not in settings:
                    self.logger.warning(f"Missing processing setting: {setting}")
                    continue
                
                # Validate setting types
                if setting in ["min_entity_length", "max_entity_length"]:
                    if not isinstance(settings[setting], int) or settings[setting] < 0:
                        self.logger.error(f"Invalid {setting}: must be positive integer")
                        return False
                
                elif setting == "confidence_threshold":
                    if not isinstance(settings[setting], (int, float)) or not 0 <= settings[setting] <= 1:
                        self.logger.error(f"Invalid {setting}: must be between 0 and 1")
                        return False
                
                elif setting == "use_enhanced_extraction":
                    if not isinstance(settings[setting], bool):
                        self.logger.error(f"Invalid {setting}: must be boolean")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating processing settings: {e}")
            return False
    
    def _validate_detection_patterns(self, patterns: dict) -> bool:
        """Validate detection patterns."""
        try:
            if not isinstance(patterns, dict):
                self.logger.error("Detection patterns must be a dictionary")
                return False
            
            # Check required detection types
            required_detections = self.required_fields["detection_patterns"]
            for detection_type in required_detections:
                if detection_type not in patterns:
                    self.logger.warning(f"Missing detection type: {detection_type}")
                    continue
                
                if not isinstance(patterns[detection_type], list):
                    self.logger.error(f"Detection patterns for {detection_type} must be a list")
                    return False
                
                # Validate each pattern
                for pattern in patterns[detection_type]:
                    if not self._validate_regex_pattern(pattern):
                        self.logger.error(f"Invalid detection pattern in {detection_type}: {pattern}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating detection patterns: {e}")
            return False
    
    def _validate_regex_patterns(self, patterns: dict) -> bool:
        """Validate regex patterns structure."""
        try:
            if not isinstance(patterns, dict):
                self.logger.error("Regex patterns must be a dictionary")
                return False
            
            for category, pattern_list in patterns.items():
                if not isinstance(pattern_list, list):
                    self.logger.error(f"Regex patterns for {category} must be a list")
                    return False
                
                for pattern in pattern_list:
                    if not self._validate_regex_pattern(pattern):
                        self.logger.error(f"Invalid regex pattern in {category}: {pattern}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating regex patterns: {e}")
            return False
    
    def _validate_regex_pattern(self, pattern: str) -> bool:
        """Validate individual regex pattern."""
        try:
            if not isinstance(pattern, str):
                self.logger.error("Regex pattern must be a string")
                return False
            
            # Test if pattern compiles
            re.compile(pattern)
            return True
            
        except re.error as e:
            self.logger.error(f"Invalid regex pattern '{pattern}': {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error validating regex pattern '{pattern}': {e}")
            return False
    
    def validate_language_specific_config(self, language_code: str, config: dict) -> bool:
        """Validate language-specific configuration."""
        try:
            # Basic validation
            if not self.validate_language_config(config):
                return False
            
            # Language-specific validation
            if language_code == "zh":
                return self._validate_chinese_config(config)
            elif language_code == "ru":
                return self._validate_russian_config(config)
            elif language_code == "ja":
                return self._validate_japanese_config(config)
            elif language_code == "ko":
                return self._validate_korean_config(config)
            elif language_code == "ar":
                return self._validate_arabic_config(config)
            elif language_code == "hi":
                return self._validate_hindi_config(config)
            else:
                # Default validation for other languages
                return True
                
        except Exception as e:
            self.logger.error(f"Error validating {language_code} config: {e}")
            return False
    
    def _validate_chinese_config(self, config: dict) -> bool:
        """Validate Chinese-specific configuration."""
        try:
            # Check for Classical Chinese patterns
            if "classical_patterns" in config:
                classical_patterns = config["classical_patterns"]
                required_classical = ["particles", "grammar_structures", "classical_entities"]
                
                for pattern_type in required_classical:
                    if pattern_type not in classical_patterns:
                        self.logger.warning(f"Missing Classical Chinese pattern type: {pattern_type}")
            
            # Check for Chinese-specific entity patterns
            if "entity_patterns" in config:
                entity_patterns = config["entity_patterns"]
                if "person" in entity_patterns:
                    # Check for Chinese character patterns
                    chinese_patterns = [p for p in entity_patterns["person"] if r'\u4e00-\u9fff' in p]
                    if not chinese_patterns:
                        self.logger.warning("No Chinese character patterns found in person entities")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating Chinese config: {e}")
            return False
    
    def _validate_russian_config(self, config: dict) -> bool:
        """Validate Russian-specific configuration."""
        try:
            # Check for Russian-specific patterns
            if "grammar_patterns" in config:
                grammar_patterns = config["grammar_patterns"]
                required_grammar = ["cases", "verb_forms", "prepositions"]
                
                for pattern_type in required_grammar:
                    if pattern_type not in grammar_patterns:
                        self.logger.warning(f"Missing Russian grammar pattern type: {pattern_type}")
            
            # Check for Cyrillic patterns
            if "entity_patterns" in config:
                entity_patterns = config["entity_patterns"]
                if "person" in entity_patterns:
                    # Check for Cyrillic patterns
                    cyrillic_patterns = [p for p in entity_patterns["person"] if r'[А-ЯЁа-яё]' in p]
                    if not cyrillic_patterns:
                        self.logger.warning("No Cyrillic patterns found in person entities")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating Russian config: {e}")
            return False
    
    def _validate_japanese_config(self, config: dict) -> bool:
        """Validate Japanese-specific configuration."""
        try:
            # Check for Japanese-specific patterns
            if "entity_patterns" in config:
                entity_patterns = config["entity_patterns"]
                if "person" in entity_patterns:
                    # Check for Japanese character patterns
                    japanese_patterns = [p for p in entity_patterns["person"] 
                                       if r'\u3040-\u309F' in p or r'\u30A0-\u30FF' in p or r'\u4E00-\u9FAF' in p]
                    if not japanese_patterns:
                        self.logger.warning("No Japanese character patterns found in person entities")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating Japanese config: {e}")
            return False
    
    def _validate_korean_config(self, config: dict) -> bool:
        """Validate Korean-specific configuration."""
        try:
            # Check for Korean-specific patterns
            if "entity_patterns" in config:
                entity_patterns = config["entity_patterns"]
                if "person" in entity_patterns:
                    # Check for Korean character patterns
                    korean_patterns = [p for p in entity_patterns["person"] if r'[가-힣]' in p]
                    if not korean_patterns:
                        self.logger.warning("No Korean character patterns found in person entities")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating Korean config: {e}")
            return False
    
    def _validate_arabic_config(self, config: dict) -> bool:
        """Validate Arabic-specific configuration."""
        try:
            # Check for Arabic-specific patterns
            if "entity_patterns" in config:
                entity_patterns = config["entity_patterns"]
                if "person" in entity_patterns:
                    # Check for Arabic character patterns
                    arabic_patterns = [p for p in entity_patterns["person"] if r'[\u0600-\u06FF]' in p]
                    if not arabic_patterns:
                        self.logger.warning("No Arabic character patterns found in person entities")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating Arabic config: {e}")
            return False
    
    def _validate_hindi_config(self, config: dict) -> bool:
        """Validate Hindi-specific configuration."""
        try:
            # Check for Hindi-specific patterns
            if "entity_patterns" in config:
                entity_patterns = config["entity_patterns"]
                if "person" in entity_patterns:
                    # Check for Devanagari character patterns
                    hindi_patterns = [p for p in entity_patterns["person"] if r'[\u0900-\u097F]' in p]
                    if not hindi_patterns:
                        self.logger.warning("No Hindi character patterns found in person entities")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating Hindi config: {e}")
            return False


# Global instance
config_validator = ConfigValidator()
