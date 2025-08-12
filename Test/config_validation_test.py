#!/usr/bin/env python3
"""
Configuration validation test script for Classical Chinese processing.
Tests Chinese language configuration loading, validates Classical Chinese patterns,
and tests Ollama model configuration.
"""

import sys
import os
import asyncio
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.language_config.chinese_config import ChineseConfig
from config.language_config.base_config import BaseLanguageConfig
from config.ollama_config import OptimizedOllamaConfig
from config.config import config


class ConfigurationValidator:
    """Validates configuration for Classical Chinese processing."""
    
    def __init__(self):
        self.chinese_config = ChineseConfig()
        self.ollama_config = OptimizedOllamaConfig()
        self.test_results = {
            "passed": [],
            "failed": [],
            "warnings": []
        }
    
    def test_chinese_config_loading(self):
        """Test Chinese configuration loading."""
        try:
            # Test basic configuration loading
            assert self.chinese_config.language_code == "zh"
            assert self.chinese_config.language_name == "Chinese"
            
            # Test entity patterns
            assert hasattr(self.chinese_config, 'entity_patterns')
            assert hasattr(self.chinese_config.entity_patterns, 'person')
            assert hasattr(self.chinese_config.entity_patterns, 'organization')
            assert hasattr(self.chinese_config.entity_patterns, 'location')
            assert hasattr(self.chinese_config.entity_patterns, 'concept')
            
            self.test_results["passed"].append("Chinese configuration loading")
            print("✅ Chinese configuration loaded successfully")
            
        except Exception as e:
            self.test_results["failed"].append(f"Chinese configuration loading: {e}")
            print(f"❌ Chinese configuration loading failed: {e}")
    
    def test_classical_chinese_patterns(self):
        """Test Classical Chinese patterns."""
        try:
            # Test Classical Chinese patterns
            classical_patterns = self.chinese_config.get_classical_chinese_patterns()
            assert len(classical_patterns) > 0
            
            # Test specific Classical Chinese patterns
            required_keys = ['particles', 'grammar_structures', 'classical_entities']
            for key in required_keys:
                assert key in classical_patterns, f"Missing key: {key}"
                assert len(classical_patterns[key]) > 0, f"Empty patterns for: {key}"
            
            # Check for Classical Chinese entity patterns
            classical_entities = classical_patterns['classical_entities']
            assert any('子' in pattern for pattern in classical_entities), "Missing classical titles"
            assert any('国' in pattern for pattern in classical_entities), "Missing classical locations"
            assert any('仁' in pattern for pattern in classical_entities), "Missing classical virtues"
            
            self.test_results["passed"].append("Classical Chinese patterns")
            print("✅ Classical Chinese patterns validated")
            
        except Exception as e:
            self.test_results["failed"].append(f"Classical Chinese patterns: {e}")
            print(f"❌ Classical Chinese patterns validation failed: {e}")
    
    def test_ollama_model_configuration(self):
        """Test Ollama model configuration."""
        try:
            # Test Ollama configuration
            ollama_config = self.chinese_config.get_ollama_config()
            
            # Check required model types
            required_models = ['text_model', 'vision_model', 'audio_model']
            for model_type in required_models:
                assert model_type in ollama_config
                assert 'model_id' in ollama_config[model_type]
                assert 'temperature' in ollama_config[model_type]
                assert 'max_tokens' in ollama_config[model_type]
                assert 'system_prompt' in ollama_config[model_type]
            
            # Test Chinese-specific model configuration
            text_model = ollama_config['text_model']
            assert 'qwen2.5:7b' in text_model['model_id']  # Chinese model
            assert text_model['temperature'] <= 0.3  # Should be low for accuracy
            
            self.test_results["passed"].append("Ollama model configuration")
            print("✅ Ollama model configuration validated")
            
        except Exception as e:
            self.test_results["failed"].append(f"Ollama model configuration: {e}")
            print(f"❌ Ollama model configuration validation failed: {e}")
    
    def test_entity_extraction_patterns(self):
        """Test entity extraction patterns."""
        try:
            patterns = self.chinese_config.entity_patterns
            
            # Test person patterns
            assert len(patterns.person) > 0
            # Check for Classical Chinese person patterns
            classical_person_patterns = [p for p in patterns.person if any(char in p for char in ['子', '君', '公', '卿', '氏'])]
            assert len(classical_person_patterns) > 0
            
            # Test organization patterns
            assert len(patterns.organization) > 0
            # Check for Classical Chinese organization patterns
            classical_org_patterns = [p for p in patterns.organization if any(char in p for char in ['国', '朝', '府', '衙', '寺'])]
            assert len(classical_org_patterns) > 0
            
            # Test location patterns
            assert len(patterns.location) > 0
            # Check for Classical Chinese location patterns
            classical_loc_patterns = [p for p in patterns.location if any(char in p for char in ['国', '州', '郡', '县', '邑'])]
            assert len(classical_loc_patterns) > 0
            
            # Test concept patterns
            assert len(patterns.concept) > 0
            # Check for Classical Chinese concept patterns
            classical_concept_patterns = [p for p in patterns.concept if any(char in p for char in ['仁', '义', '礼', '智', '信', '道', '德'])]
            assert len(classical_concept_patterns) > 0
            
            self.test_results["passed"].append("Entity extraction patterns")
            print("✅ Entity extraction patterns validated")
            
        except Exception as e:
            self.test_results["failed"].append(f"Entity extraction patterns: {e}")
            print(f"❌ Entity extraction patterns validation failed: {e}")
    
    def test_processing_settings(self):
        """Test processing settings."""
        try:
            settings = self.chinese_config.get_processing_settings()
            
            # Test required settings
            assert hasattr(settings, 'min_entity_length')
            assert hasattr(settings, 'max_entity_length')
            assert hasattr(settings, 'confidence_threshold')
            assert hasattr(settings, 'use_enhanced_extraction')
            
            # Test Classical Chinese specific settings
            assert hasattr(settings, 'fallback_strategies')
            assert 'classical_patterns' in settings.fallback_strategies
            
            self.test_results["passed"].append("Processing settings")
            print("✅ Processing settings validated")
            
        except Exception as e:
            self.test_results["failed"].append(f"Processing settings: {e}")
            print(f"❌ Processing settings validation failed: {e}")
    
    def run_all_tests(self):
        """Run all configuration validation tests."""
        print("🔍 Starting Configuration Validation Tests...")
        print("=" * 60)
        
        self.test_chinese_config_loading()
        self.test_classical_chinese_patterns()
        self.test_ollama_model_configuration()
        self.test_entity_extraction_patterns()
        self.test_processing_settings()
        
        print("=" * 60)
        print("📊 Test Results Summary:")
        print(f"✅ Passed: {len(self.test_results['passed'])}")
        print(f"❌ Failed: {len(self.test_results['failed'])}")
        print(f"⚠️  Warnings: {len(self.test_results['warnings'])}")
        
        if self.test_results['failed']:
            print("\n❌ Failed Tests:")
            for failure in self.test_results['failed']:
                print(f"  - {failure}")
        
        if self.test_results['warnings']:
            print("\n⚠️  Warnings:")
            for warning in self.test_results['warnings']:
                print(f"  - {warning}")
        
        # Save test results
        results_dir = Path("../Results")
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / "config_validation_results.json", "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Test results saved to: {results_dir / 'config_validation_results.json'}")
        
        return len(self.test_results['failed']) == 0


async def main():
    """Main test function."""
    validator = ConfigurationValidator()
    success = validator.run_all_tests()
    
    if success:
        print("\n🎉 All configuration validation tests passed!")
        return 0
    else:
        print("\n💥 Some configuration validation tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
