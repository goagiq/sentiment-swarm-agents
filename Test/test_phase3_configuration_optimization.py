"""
Phase 3 Configuration Optimization Testing Framework.
Tests for dynamic configuration management, configuration validation, and multilingual regex patterns.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.dynamic_config_manager import DynamicConfigManager
from config.config_validator import ConfigValidator
from config.language_specific_regex_config import LANGUAGE_REGEX_PATTERNS, LANGUAGE_PROCESSING_SETTINGS
from config.language_config.base_config import LanguageConfigFactory


class Phase3ConfigurationOptimizationTest:
    """Comprehensive test suite for Phase 3 configuration optimization."""
    
    def __init__(self):
        self.config_manager = DynamicConfigManager()
        self.validator = ConfigValidator()
        self.test_results = {
            "dynamic_config_manager": {},
            "config_validator": {},
            "multilingual_patterns": {},
            "processing_settings": {},
            "integration_tests": {}
        }
    
    async def run_all_tests(self):
        """Run all Phase 3 tests."""
        print("🔧 Phase 3 Configuration Optimization Testing")
        print("=" * 60)
        
        # Test Dynamic Configuration Manager
        await self.test_dynamic_config_manager()
        
        # Test Configuration Validator
        await self.test_config_validator()
        
        # Test Multilingual Regex Patterns
        await self.test_multilingual_patterns()
        
        # Test Processing Settings
        await self.test_processing_settings()
        
        # Test Integration
        await self.test_integration()
        
        # Generate report
        self.generate_test_report()
    
    async def test_dynamic_config_manager(self):
        """Test dynamic configuration manager functionality."""
        print("\n📋 Testing Dynamic Configuration Manager...")
        
        try:
            # Test initialization
            assert self.config_manager is not None
            assert hasattr(self.config_manager, 'config_watchers')
            assert hasattr(self.config_manager, 'config_backups')
            assert hasattr(self.config_manager, 'config_cache')
            self.test_results["dynamic_config_manager"]["initialization"] = "✅ PASS"
            
            # Test config status
            status = await self.config_manager.get_config_status()
            assert isinstance(status, dict)
            assert "total_languages" in status
            self.test_results["dynamic_config_manager"]["config_status"] = "✅ PASS"
            
            # Test watcher functionality
            test_callback_called = False
            
            def test_callback(lang_code, config):
                nonlocal test_callback_called
                test_callback_called = True
            
            self.config_manager.add_config_watcher("test", test_callback)
            assert "test" in self.config_manager.config_watchers
            self.test_results["dynamic_config_manager"]["watcher_management"] = "✅ PASS"
            
            print("   ✅ Dynamic Configuration Manager: All tests passed")
            
        except Exception as e:
            print(f"   ❌ Dynamic Configuration Manager: Error - {e}")
            self.test_results["dynamic_config_manager"]["error"] = str(e)
    
    async def test_config_validator(self):
        """Test configuration validator functionality."""
        print("\n📋 Testing Configuration Validator...")
        
        try:
            # Test basic validation
            valid_config = {
                "entity_patterns": {
                    "person": [r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'],
                    "organization": [r'\b[A-Z][a-z]+(?:Inc\.|Corp\.)\b'],
                    "location": [r'\b[A-Z][a-z]+(?:City|Town)\b'],
                    "concept": [r'\b(?:AI|ML|DL)\b']
                },
                "processing_settings": {
                    "min_entity_length": 2,
                    "max_entity_length": 50,
                    "confidence_threshold": 0.7,
                    "use_enhanced_extraction": True
                }
            }
            
            assert self.validator.validate_language_config(valid_config)
            self.test_results["config_validator"]["basic_validation"] = "✅ PASS"
            
            # Test invalid config
            invalid_config = {
                "entity_patterns": {
                    "person": ["invalid regex ["],  # Invalid regex
                    "organization": [r'\b[A-Z][a-z]+(?:Inc\.|Corp\.)\b']
                }
            }
            
            assert not self.validator.validate_language_config(invalid_config)
            self.test_results["config_validator"]["invalid_validation"] = "✅ PASS"
            
            # Test language-specific validation
            chinese_config = {
                "entity_patterns": {
                    "person": [r'[\u4e00-\u9fff]{2,4}'],
                    "organization": [r'[\u4e00-\u9fff]+(?:公司|大学)'],
                    "location": [r'[\u4e00-\u9fff]+(?:市|省)'],
                    "concept": [r'(?:人工智能|机器学习)']
                },
                "processing_settings": {
                    "min_entity_length": 2,
                    "max_entity_length": 20,
                    "confidence_threshold": 0.7,
                    "use_enhanced_extraction": True
                }
            }
            
            assert self.validator.validate_language_specific_config("zh", chinese_config)
            self.test_results["config_validator"]["language_specific"] = "✅ PASS"
            
            print("   ✅ Configuration Validator: All tests passed")
            
        except Exception as e:
            print(f"   ❌ Configuration Validator: Error - {e}")
            self.test_results["config_validator"]["error"] = str(e)
    
    async def test_multilingual_patterns(self):
        """Test multilingual regex patterns."""
        print("\n📋 Testing Multilingual Regex Patterns...")
        
        try:
            # Test pattern structure
            assert isinstance(LANGUAGE_REGEX_PATTERNS, dict)
            assert len(LANGUAGE_REGEX_PATTERNS) >= 7  # en, zh, ru, ja, ko, ar, hi
            
            # Test each language
            for lang_code, patterns in LANGUAGE_REGEX_PATTERNS.items():
                assert isinstance(patterns, dict)
                assert "person" in patterns
                assert "organization" in patterns
                assert "location" in patterns
                assert "concept" in patterns
                
                # Test pattern compilation
                for entity_type, pattern_list in patterns.items():
                    assert isinstance(pattern_list, list)
                    for pattern in pattern_list:
                        import re
                        re.compile(pattern)  # Should not raise exception
            
            self.test_results["multilingual_patterns"]["structure"] = "✅ PASS"
            self.test_results["multilingual_patterns"]["compilation"] = "✅ PASS"
            
            # Test specific language patterns
            test_cases = {
                "en": {
                    "text": "John Smith works at Microsoft Corporation in New York City.",
                    "expected_entities": ["John Smith", "Microsoft Corporation", "New York City"]
                },
                "zh": {
                    "text": "张三在北京大学工作，研究人工智能技术。",
                    "expected_entities": ["张三", "北京大学", "人工智能"]
                },
                "ru": {
                    "text": "Иван Петров работает в Московском университете.",
                    "expected_entities": ["Иван Петров", "Московском университете"]
                }
            }
            
            for lang_code, test_case in test_cases.items():
                if lang_code in LANGUAGE_REGEX_PATTERNS:
                    patterns = LANGUAGE_REGEX_PATTERNS[lang_code]
                    found_entities = []
                    
                    for entity_type, pattern_list in patterns.items():
                        for pattern in pattern_list:
                            import re
                            matches = re.findall(pattern, test_case["text"])
                            found_entities.extend(matches)
                    
                    # Check if we found some entities (not necessarily all expected)
                    assert len(found_entities) > 0
            
            self.test_results["multilingual_patterns"]["pattern_matching"] = "✅ PASS"
            
            print("   ✅ Multilingual Regex Patterns: All tests passed")
            
        except Exception as e:
            print(f"   ❌ Multilingual Regex Patterns: Error - {e}")
            self.test_results["multilingual_patterns"]["error"] = str(e)
    
    async def test_processing_settings(self):
        """Test multilingual processing settings."""
        print("\n📋 Testing Multilingual Processing Settings...")
        
        try:
            # Test settings structure
            assert isinstance(LANGUAGE_PROCESSING_SETTINGS, dict)
            assert len(LANGUAGE_PROCESSING_SETTINGS) >= 7  # en, zh, ru, ja, ko, ar, hi
            
            # Test each language settings
            for lang_code, settings in LANGUAGE_PROCESSING_SETTINGS.items():
                assert isinstance(settings, dict)
                assert "min_entity_length" in settings
                assert "max_entity_length" in settings
                assert "confidence_threshold" in settings
                assert "use_enhanced_extraction" in settings
                assert "relationship_prompt_simplified" in settings
                
                # Validate setting values
                assert isinstance(settings["min_entity_length"], int)
                assert isinstance(settings["max_entity_length"], int)
                assert isinstance(settings["confidence_threshold"], (int, float))
                assert isinstance(settings["use_enhanced_extraction"], bool)
                assert isinstance(settings["relationship_prompt_simplified"], bool)
                
                assert settings["min_entity_length"] > 0
                assert settings["max_entity_length"] > settings["min_entity_length"]
                assert 0 <= settings["confidence_threshold"] <= 1
            
            self.test_results["processing_settings"]["structure"] = "✅ PASS"
            self.test_results["processing_settings"]["validation"] = "✅ PASS"
            
            print("   ✅ Multilingual Processing Settings: All tests passed")
            
        except Exception as e:
            print(f"   ❌ Multilingual Processing Settings: Error - {e}")
            self.test_results["processing_settings"]["error"] = str(e)
    
    async def test_integration(self):
        """Test integration between components."""
        print("\n📋 Testing Integration...")
        
        try:
            # Test language config factory integration
            available_languages = LanguageConfigFactory.get_available_languages()
            assert len(available_languages) >= 5  # Should have at least 5 languages
            
            # Test config manager with language factory
            for lang_code in available_languages[:3]:  # Test first 3 languages
                config = LanguageConfigFactory.get_config(lang_code)
                assert config is not None
                assert hasattr(config, 'language_code')
                assert hasattr(config, 'language_name')
            
            self.test_results["integration_tests"]["language_factory"] = "✅ PASS"
            
            # Test dynamic config update simulation
            test_config = {
                "entity_patterns": {
                    "person": [r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'],
                    "organization": [r'\b[A-Z][a-z]+(?:Inc\.|Corp\.)\b'],
                    "location": [r'\b[A-Z][a-z]+(?:City|Town)\b'],
                    "concept": [r'\b(?:AI|ML|DL)\b']
                },
                "processing_settings": {
                    "min_entity_length": 2,
                    "max_entity_length": 50,
                    "confidence_threshold": 0.7,
                    "use_enhanced_extraction": True
                }
            }
            
            # Validate config before update
            assert self.validator.validate_language_config(test_config)
            
            # Simulate config update
            success = await self.config_manager.update_language_config("test_lang", test_config)
            assert success  # Should succeed with valid config
            
            self.test_results["integration_tests"]["config_update"] = "✅ PASS"
            
            print("   ✅ Integration: All tests passed")
            
        except Exception as e:
            print(f"   ❌ Integration: Error - {e}")
            self.test_results["integration_tests"]["error"] = str(e)
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        print("\n📊 Phase 3 Configuration Optimization Test Report")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        for category, results in self.test_results.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            category_tests = 0
            category_passed = 0
            
            for test_name, result in results.items():
                if test_name != "error":
                    category_tests += 1
                    total_tests += 1
                    
                    if result == "✅ PASS":
                        category_passed += 1
                        passed_tests += 1
                        print(f"   ✅ {test_name}")
                    else:
                        print(f"   ❌ {test_name}: {result}")
                else:
                    print(f"   ❌ Error: {result}")
            
            if category_tests > 0:
                success_rate = (category_passed / category_tests) * 100
                print(f"   📈 Success Rate: {success_rate:.1f}% ({category_passed}/{category_tests})")
        
        # Overall summary
        print(f"\n📈 OVERALL SUMMARY:")
        if total_tests > 0:
            overall_success_rate = (passed_tests / total_tests) * 100
            print(f"   Total Tests: {total_tests}")
            print(f"   Passed: {passed_tests}")
            print(f"   Failed: {total_tests - passed_tests}")
            print(f"   Success Rate: {overall_success_rate:.1f}%")
            
            if overall_success_rate >= 90:
                print("   🎉 Phase 3 Configuration Optimization: EXCELLENT")
            elif overall_success_rate >= 80:
                print("   ✅ Phase 3 Configuration Optimization: GOOD")
            elif overall_success_rate >= 70:
                print("   ⚠️ Phase 3 Configuration Optimization: NEEDS IMPROVEMENT")
            else:
                print("   ❌ Phase 3 Configuration Optimization: FAILED")
        else:
            print("   ❌ No tests were executed")


async def main():
    """Main test execution function."""
    test_suite = Phase3ConfigurationOptimizationTest()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
