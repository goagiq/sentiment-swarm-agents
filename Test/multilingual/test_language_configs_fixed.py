#!/usr/bin/env python3
"""
Test script to verify language configurations are working properly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_language_configs():
    """Test all language configurations."""
    print("🔍 Testing Language Configurations...")
    print("=" * 50)
    
    try:
        from src.config.language_config.base_config import LanguageConfigFactory
        
        available_languages = LanguageConfigFactory.get_available_languages()
        print(f"Available languages: {available_languages}")
        
        required_entity_types = ['person', 'organization', 'location', 'concept']
        required_processing_settings = ['min_entity_length', 'max_entity_length', 'confidence_threshold', 'use_enhanced_extraction']
        
        successful_configs = 0
        total_configs = len(available_languages)
        
        for lang_code in available_languages:
            print(f"\n📋 Testing {lang_code.upper()} configuration:")
            
            try:
                config = LanguageConfigFactory.get_config(lang_code)
                
                # Check entity patterns
                entity_patterns_ok = True
                for entity_type in required_entity_types:
                    if hasattr(config.entity_patterns, entity_type):
                        patterns = getattr(config.entity_patterns, entity_type)
                        if patterns and len(patterns) > 0:
                            print(f"    ✅ {entity_type}: {len(patterns)} patterns")
                        else:
                            print(f"    ❌ {entity_type}: No patterns found")
                            entity_patterns_ok = False
                    else:
                        print(f"    ❌ {entity_type}: Missing attribute")
                        entity_patterns_ok = False
                
                # Check processing settings
                processing_settings_ok = True
                for setting in required_processing_settings:
                    if hasattr(config.processing_settings, setting):
                        value = getattr(config.processing_settings, setting)
                        print(f"    ✅ {setting}: {value}")
                    else:
                        print(f"    ❌ {setting}: Missing")
                        processing_settings_ok = False
                
                # Check detection patterns
                if hasattr(config, 'detection_patterns') and config.detection_patterns:
                    print(f"    ✅ detection_patterns: {len(config.detection_patterns)} patterns")
                else:
                    print(f"    ❌ detection_patterns: Missing or empty")
                    entity_patterns_ok = False
                
                if entity_patterns_ok and processing_settings_ok:
                    successful_configs += 1
                    print(f"    ✅ {lang_code.upper()}: Configuration OK")
                else:
                    print(f"    ❌ {lang_code.upper()}: Configuration has issues")
                
            except Exception as e:
                print(f"    ❌ Error testing {lang_code}: {e}")
        
        success_rate = successful_configs / total_configs
        print(f"\n📊 Results: {successful_configs}/{total_configs} configurations successful ({success_rate:.1%})")
        
        if success_rate >= 0.8:
            print("✅ Language configurations are working properly!")
            return True
        else:
            print("❌ Some language configurations have issues.")
            return False
        
    except Exception as e:
        print(f"❌ Error testing language configurations: {e}")
        return False

if __name__ == "__main__":
    success = test_language_configs()
    sys.exit(0 if success else 1)
