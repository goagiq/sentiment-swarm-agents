# Configuration Management Guide

## Overview

This guide provides comprehensive documentation for managing configurations in the multilingual sentiment analysis system. It covers language-specific configurations, dynamic updates, validation, and best practices for maintaining system configurations.

## Table of Contents

1. [Configuration Architecture](#configuration-architecture)
2. [Language-Specific Configurations](#language-specific-configurations)
3. [Dynamic Configuration Management](#dynamic-configuration-management)
4. [Configuration Validation](#configuration-validation)
5. [Environment-Based Configuration](#environment-based-configuration)
6. [Configuration Files Structure](#configuration-files-structure)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Configuration Architecture

### Core Components

The configuration system consists of several key components:

#### 1. Base Configuration (`src/config/language_config/base_config.py`)
- **BaseLanguageConfig**: Abstract base class for all language configurations
- **LanguageConfigFactory**: Factory pattern for creating language-specific configurations
- **Configuration Registry**: Central registry for all available language configurations

#### 2. Language-Specific Configurations
- **ChineseConfig**: Chinese language patterns and settings
- **RussianConfig**: Russian language patterns and settings
- **JapaneseConfig**: Japanese language patterns and settings
- **KoreanConfig**: Korean language patterns and settings
- **ArabicConfig**: Arabic language patterns and settings
- **HindiConfig**: Hindi language patterns and settings

#### 3. Dynamic Configuration Manager (`src/config/dynamic_config_manager.py`)
- **Runtime Updates**: Hot-reload capabilities
- **Configuration Validation**: Real-time validation
- **Environment Adaptation**: Environment-specific settings

#### 4. Configuration Validator (`src/config/config_validator.py`)
- **Pattern Validation**: Regex pattern validation
- **Structure Validation**: Configuration structure validation
- **Compatibility Checks**: Cross-configuration compatibility

### Configuration Hierarchy

```
BaseLanguageConfig (Abstract)
├── ChineseConfig
├── RussianConfig
├── JapaneseConfig
├── KoreanConfig
├── ArabicConfig
└── HindiConfig

LanguageConfigFactory
├── Configuration Registry
├── Dynamic Updates
└── Validation

DynamicConfigManager
├── Hot Reload
├── Environment Adaptation
└── Backup/Restore
```

## Language-Specific Configurations

### Configuration Structure

Each language configuration implements the following interface:

```python
class LanguageConfig(BaseLanguageConfig):
    def get_entity_patterns(self) -> EntityPatterns:
        """Return language-specific entity extraction patterns."""
        
    def get_grammar_patterns(self) -> Dict[str, str]:
        """Return grammar and syntax patterns."""
        
    def get_advanced_patterns(self) -> Dict[str, str]:
        """Return advanced cultural and technical patterns."""
        
    def get_processing_settings(self) -> ProcessingSettings:
        """Return language-specific processing parameters."""
        
    def get_relationship_templates(self) -> List[str]:
        """Return relationship extraction templates."""
        
    def get_detection_patterns(self) -> List[str]:
        """Return language detection patterns."""
```

### Entity Patterns

Entity patterns define how to extract different types of entities:

```python
@dataclass
class EntityPatterns:
    person: str
    organization: str
    location: str
    concept: str
```

#### Example: Chinese Entity Patterns
```python
def get_entity_patterns(self) -> EntityPatterns:
    return EntityPatterns(
        person=r'[\u4e00-\u9fff]{2,4}(?:先生|女士|老师|教授|博士|硕士|学士)',
        organization=r'[\u4e00-\u9fff]+(?:公司|集团|企业|学校|大学|学院|医院|政府|部门)',
        location=r'[\u4e00-\u9fff]+(?:省|市|县|区|街道|路|号|楼|室)',
        concept=r'[\u4e00-\u9fff]+(?:主义|思想|理论|方法|技术|艺术|文化|传统)'
    )
```

### Grammar Patterns

Grammar patterns capture language-specific syntax:

```python
def get_grammar_patterns(self) -> Dict[str, str]:
    return {
        "definite_article": r"the|a|an",
        "verb_forms": r"\b(?:is|are|was|were|be|been|being)\b",
        "prepositions": r"\b(?:in|on|at|by|for|with|without|through)\b",
        "conjunctions": r"\b(?:and|or|but|because|although|however)\b",
        "question_words": r"\b(?:what|when|where|who|why|how)\b",
        "numbers": r"\b\d+(?:\.\d+)?\b"
    }
```

### Processing Settings

Processing settings control how text is processed:

```python
@dataclass
class ProcessingSettings:
    max_entity_length: int
    confidence_threshold: float
    use_enhanced_extraction: bool
    enable_caching: bool
    cache_ttl: int
    parallel_processing: bool
    max_workers: int
```

#### Example: Arabic Processing Settings
```python
def get_processing_settings(self) -> ProcessingSettings:
    return ProcessingSettings(
        max_entity_length=50,
        confidence_threshold=0.7,
        use_enhanced_extraction=True,
        enable_caching=True,
        cache_ttl=3600,
        parallel_processing=True,
        max_workers=4
    )
```

## Dynamic Configuration Management

### Runtime Updates

The system supports dynamic configuration updates without restart:

```python
from src.config.dynamic_config_manager import dynamic_config_manager

# Update language configuration
new_config = {
    "entity_patterns": {
        "person": r"updated_person_pattern",
        "organization": r"updated_org_pattern"
    },
    "processing_settings": {
        "confidence_threshold": 0.8
    }
}

await dynamic_config_manager.update_language_config("zh", new_config)
```

### Hot Reload

Configuration hot-reload capabilities:

```python
# Reload all configurations
await dynamic_config_manager.reload_configurations()

# Reload specific language configuration
await dynamic_config_manager.reload_language_config("zh")

# Reload processing settings
await dynamic_config_manager.reload_processing_settings()
```

### Configuration Backup and Restore

```python
# Backup current configuration
backup = await dynamic_config_manager.backup_configurations()

# Restore from backup
await dynamic_config_manager.restore_configurations(backup)

# Export configuration to file
await dynamic_config_manager.export_config("config_backup.json")

# Import configuration from file
await dynamic_config_manager.import_config("config_backup.json")
```

## Configuration Validation

### Pattern Validation

Validate regex patterns for correctness:

```python
from src.config.config_validator import config_validator

# Validate entity patterns
errors = await config_validator.validate_entity_patterns(patterns)

# Validate grammar patterns
errors = await config_validator.validate_grammar_patterns(patterns)

# Validate advanced patterns
errors = await config_validator.validate_advanced_patterns(patterns)
```

### Structure Validation

Validate configuration structure:

```python
# Validate language configuration
is_valid = await config_validator.validate_language_config("zh", config)

# Validate processing settings
is_valid = await config_validator.validate_processing_settings(settings)

# Validate relationship templates
is_valid = await config_validator.validate_relationship_templates(templates)
```

### Compatibility Validation

Check configuration compatibility:

```python
# Check language configuration compatibility
compatibility = await config_validator.check_language_compatibility("zh", "en")

# Validate cross-language patterns
errors = await config_validator.validate_cross_language_patterns()
```

## Environment-Based Configuration

### Environment Variables

Configuration adapts to different environments:

```python
# Development environment
ENVIRONMENT = "development"
DEBUG = True
LOG_LEVEL = "DEBUG"
CACHE_TTL = 300
MAX_WORKERS = 2

# Production environment
ENVIRONMENT = "production"
DEBUG = False
LOG_LEVEL = "INFO"
CACHE_TTL = 3600
MAX_WORKERS = 8
```

### Configuration Files

Environment-specific configuration files:

```python
# config/development.py
DEVELOPMENT_CONFIG = {
    "cache_ttl": 300,
    "max_workers": 2,
    "debug": True,
    "log_level": "DEBUG"
}

# config/production.py
PRODUCTION_CONFIG = {
    "cache_ttl": 3600,
    "max_workers": 8,
    "debug": False,
    "log_level": "INFO"
}
```

### Dynamic Environment Adaptation

```python
# Get environment-specific configuration
config = dynamic_config_manager.get_environment_config()

# Update configuration based on environment
await dynamic_config_manager.adapt_to_environment("production")

# Validate environment configuration
is_valid = await config_validator.validate_environment_config()
```

## Configuration Files Structure

### Directory Structure

```
src/config/
├── __init__.py
├── language_config/
│   ├── __init__.py
│   ├── base_config.py
│   ├── chinese_config.py
│   ├── russian_config.py
│   ├── japanese_config.py
│   ├── korean_config.py
│   ├── arabic_config.py
│   └── hindi_config.py
├── dynamic_config_manager.py
├── config_validator.py
├── caching_config.py
├── parallel_processing_config.py
├── memory_config.py
├── monitoring_config.py
└── language_specific_regex_config.py
```

### Configuration File Templates

#### Language Configuration Template
```python
from typing import Dict, List
from dataclasses import dataclass
from .base_config import BaseLanguageConfig, EntityPatterns, ProcessingSettings

class LanguageConfig(BaseLanguageConfig):
    """Language-specific configuration."""
    
    def get_entity_patterns(self) -> EntityPatterns:
        """Return entity extraction patterns."""
        return EntityPatterns(
            person=r"person_pattern",
            organization=r"organization_pattern",
            location=r"location_pattern",
            concept=r"concept_pattern"
        )
    
    def get_grammar_patterns(self) -> Dict[str, str]:
        """Return grammar patterns."""
        return {
            "definite_article": r"article_pattern",
            "verb_forms": r"verb_pattern",
            "prepositions": r"preposition_pattern",
            "conjunctions": r"conjunction_pattern",
            "question_words": r"question_pattern",
            "numbers": r"number_pattern"
        }
    
    def get_advanced_patterns(self) -> Dict[str, str]:
        """Return advanced patterns."""
        return {
            "cultural_terms": r"cultural_pattern",
            "technical_terms": r"technical_pattern",
            "literary_forms": r"literary_pattern"
        }
    
    def get_processing_settings(self) -> ProcessingSettings:
        """Return processing settings."""
        return ProcessingSettings(
            max_entity_length=50,
            confidence_threshold=0.7,
            use_enhanced_extraction=True,
            enable_caching=True,
            cache_ttl=3600,
            parallel_processing=True,
            max_workers=4
        )
    
    def get_relationship_templates(self) -> List[str]:
        """Return relationship templates."""
        return [
            "{entity1} is related to {entity2}",
            "{entity1} works for {entity2}",
            "{entity1} is located in {entity2}"
        ]
    
    def get_detection_patterns(self) -> List[str]:
        """Return language detection patterns."""
        return [
            r"language_specific_pattern1",
            r"language_specific_pattern2"
        ]
```

## Best Practices

### Configuration Design

1. **Modular Design**: Keep configurations modular and focused
2. **Default Values**: Provide sensible default values
3. **Validation**: Always validate configurations before use
4. **Documentation**: Document all configuration options

### Language-Specific Configurations

1. **Pattern Testing**: Test regex patterns with real data
2. **Performance Optimization**: Optimize patterns for performance
3. **Maintainability**: Keep patterns readable and maintainable
4. **Extensibility**: Design for easy extension and modification

### Dynamic Configuration

1. **Atomic Updates**: Make configuration updates atomic
2. **Rollback Capability**: Always maintain rollback capability
3. **Validation**: Validate configurations before applying
4. **Monitoring**: Monitor configuration changes and their impact

### Environment Management

1. **Environment Isolation**: Keep environments isolated
2. **Configuration Versioning**: Version control configurations
3. **Deployment Automation**: Automate configuration deployment
4. **Security**: Secure sensitive configuration data

## Troubleshooting

### Common Configuration Issues

#### Invalid Regex Patterns
```python
# Check pattern validity
try:
    re.compile(pattern)
except re.error as e:
    print(f"Invalid regex pattern: {e}")
```

#### Missing Configuration Files
```python
# Check file existence
import os
if not os.path.exists(config_path):
    print(f"Configuration file not found: {config_path}")
```

#### Configuration Validation Errors
```python
# Validate configuration
errors = await config_validator.validate_language_config("zh", config)
if errors:
    print(f"Configuration validation errors: {errors}")
```

### Debug Tools

#### Configuration Inspector
```python
from src.config.dynamic_config_manager import dynamic_config_manager

# Inspect current configuration
config = dynamic_config_manager.get_current_config()
print(config)

# Inspect language configuration
lang_config = dynamic_config_manager.get_language_config("zh")
print(lang_config)
```

#### Configuration Validator
```python
from src.config.config_validator import config_validator

# Validate all configurations
errors = await config_validator.validate_all_configurations()
print(f"Validation errors: {errors}")

# Test specific patterns
test_text = "Sample text for testing"
matches = config_validator.test_patterns(test_text, patterns)
print(f"Pattern matches: {matches}")
```

### Performance Optimization

#### Configuration Caching
```python
# Enable configuration caching
await dynamic_config_manager.enable_caching()

# Cache language configurations
await dynamic_config_manager.cache_language_configs()

# Monitor cache performance
cache_stats = dynamic_config_manager.get_cache_stats()
print(f"Cache hit rate: {cache_stats['hit_rate']}")
```

#### Pattern Optimization
```python
# Optimize regex patterns
optimized_patterns = config_validator.optimize_patterns(patterns)

# Benchmark pattern performance
performance = config_validator.benchmark_patterns(patterns, test_data)
print(f"Pattern performance: {performance}")
```

## Conclusion

This configuration management guide provides comprehensive documentation for managing configurations in the multilingual sentiment analysis system. By following these guidelines and best practices, you can effectively manage, validate, and optimize system configurations for maximum performance and reliability.

For additional support or questions, refer to the optimization guide and performance monitoring guide for more detailed information.
