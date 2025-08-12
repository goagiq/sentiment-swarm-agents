#!/usr/bin/env python3
"""
Simple script to check language configuration status.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def check_language_configs():
    """Check all language configurations for completeness."""
    print("🔍 Checking Language Configurations...")
    print("=" * 50)
    
    try:
        from src.config.language_config.base_config import LanguageConfigFactory
        
        available_languages = LanguageConfigFactory.get_available_languages()
        print(f"Available languages: {available_languages}")
        
        required_entity_types = ['person', 'organization', 'location', 'concept']
        required_processing_settings = ['min_entity_length', 'max_entity_length', 'confidence_threshold', 'use_enhanced_extraction']
        
        for lang_code in available_languages:
            print(f"\n📋 Checking {lang_code.upper()} configuration:")
            
            try:
                config = LanguageConfigFactory.get_config(lang_code)
                
                # Check entity patterns
                print(f"  Entity patterns:")
                for entity_type in required_entity_types:
                    if hasattr(config.entity_patterns, entity_type):
                        patterns = getattr(config.entity_patterns, entity_type)
                        if patterns and len(patterns) > 0:
                            print(f"    ✅ {entity_type}: {len(patterns)} patterns")
                        else:
                            print(f"    ❌ {entity_type}: No patterns found")
                    else:
                        print(f"    ❌ {entity_type}: Missing attribute")
                
                # Check processing settings
                print(f"  Processing settings:")
                for setting in required_processing_settings:
                    if hasattr(config.processing_settings, setting):
                        value = getattr(config.processing_settings, setting)
                        print(f"    ✅ {setting}: {value}")
                    else:
                        print(f"    ❌ {setting}: Missing")
                
                # Check additional features
                additional_features = ['classical_patterns', 'grammar_patterns', 'honorific_patterns', 'advanced_patterns']
                print(f"  Additional features:")
                for feature in additional_features:
                    if hasattr(config, feature):
                        feature_data = getattr(config, feature)
                        if feature_data:
                            print(f"    ✅ {feature}: Available")
                        else:
                            print(f"    ⚠️ {feature}: Empty")
                    else:
                        print(f"    ❌ {feature}: Missing")
                
            except Exception as e:
                print(f"  ❌ Error checking {lang_code}: {e}")
        
    except Exception as e:
        print(f"❌ Error accessing language configurations: {e}")

if __name__ == "__main__":
    check_language_configs()
