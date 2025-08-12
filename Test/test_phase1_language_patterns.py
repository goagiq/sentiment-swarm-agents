#!/usr/bin/env python3
"""
Test script for Phase 1: Enhanced Language-Specific Regex Patterns
Validates the comprehensive regex patterns for Chinese, Russian, Japanese, and Korean.
"""

import sys
import os
import asyncio
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.language_config.base_config import LanguageConfigFactory


class LanguagePatternTester:
    """Test class for validating enhanced language-specific regex patterns."""
    
    def __init__(self):
        self.test_results = {}
        self.languages = ['zh', 'ru', 'ja', 'ko', 'en']
    
    def test_chinese_patterns(self):
        """Test enhanced Chinese patterns including Classical Chinese."""
        print("🔍 Testing Chinese Language Patterns...")
        
        config = LanguageConfigFactory.get_config('zh')
        
        # Test modern Chinese patterns
        modern_texts = [
            "人工智能技术发展迅速，机器学习在各个领域都有应用。",
            "北京大学的张教授在研究深度学习算法。",
            "华为公司发布了新的5G技术。",
            "上海市政府的数字化转型项目取得了成功。"
        ]
        
        # Test Classical Chinese patterns
        classical_texts = [
            "孔子曰：学而时习之，不亦说乎？",
            "仁者爱人，义者正己。",
            "道可道，非常道。名可名，非常名。",
            "君子之道，造次必于是，颠沛必于是。"
        ]
        
        results = {
            'modern_entity_extraction': [],
            'classical_entity_extraction': [],
            'classical_detection': [],
            'grammar_patterns': []
        }
        
        # Test modern entity extraction
        for text in modern_texts:
            entities = self._extract_entities_with_patterns(config, text)
            results['modern_entity_extraction'].append({
                'text': text,
                'entities': entities
            })
        
        # Test Classical Chinese detection
        for text in classical_texts:
            is_classical = config.is_classical_chinese(text)
            results['classical_detection'].append({
                'text': text,
                'is_classical': is_classical
            })
            
            entities = self._extract_entities_with_patterns(config, text)
            results['classical_entity_extraction'].append({
                'text': text,
                'entities': entities
            })
        
        # Test grammar patterns
        grammar_text = "人工智能之发展，乃科技进步之体现。"
        grammar_matches = self._test_grammar_patterns(config, grammar_text)
        results['grammar_patterns'] = grammar_matches
        
        self.test_results['chinese'] = results
        print(f"✅ Chinese patterns tested: {len(results['modern_entity_extraction'])} modern, {len(results['classical_entity_extraction'])} classical")
    
    def test_russian_patterns(self):
        """Test enhanced Russian patterns."""
        print("🔍 Testing Russian Language Patterns...")
        
        config = LanguageConfigFactory.get_config('ru')
        
        test_texts = [
            "Искусственный интеллект развивается быстро в России.",
            "Профессор Иванов работает в Московском университете.",
            "Компания ООО 'Технологии будущего' разрабатывает новые алгоритмы.",
            "В Санкт-Петербурге открылся новый исследовательский центр."
        ]
        
        results = {
            'entity_extraction': [],
            'grammar_patterns': [],
            'formal_detection': []
        }
        
        # Test entity extraction
        for text in test_texts:
            entities = self._extract_entities_with_patterns(config, text)
            results['entity_extraction'].append({
                'text': text,
                'entities': entities
            })
        
        # Test formal Russian detection
        formal_text = "Уважаемый господин, прошу рассмотреть наше предложение."
        is_formal = config.is_formal_russian(formal_text)
        results['formal_detection'] = {
            'text': formal_text,
            'is_formal': is_formal
        }
        
        # Test grammar patterns
        grammar_text = "Новый алгоритм работает эффективно и быстро."
        grammar_matches = self._test_grammar_patterns(config, grammar_text)
        results['grammar_patterns'] = grammar_matches
        
        self.test_results['russian'] = results
        print(f"✅ Russian patterns tested: {len(results['entity_extraction'])} texts")
    
    def test_japanese_patterns(self):
        """Test enhanced Japanese patterns."""
        print("🔍 Testing Japanese Language Patterns...")
        
        config = LanguageConfigFactory.get_config('ja')
        
        test_texts = [
            "人工知能技術が急速に発展している。",
            "東京大学の田中教授が新しい研究を発表した。",
            "株式会社ソニーが新しい製品を開発した。",
            "京都の古い寺院で伝統文化を体験できる。"
        ]
        
        results = {
            'entity_extraction': [],
            'honorific_patterns': [],
            'formal_detection': []
        }
        
        # Test entity extraction
        for text in test_texts:
            entities = self._extract_entities_with_patterns(config, text)
            results['entity_extraction'].append({
                'text': text,
                'entities': entities
            })
        
        # Test formal Japanese detection
        formal_text = "お世話になっております。田中でございます。"
        is_formal = config.is_formal_japanese(formal_text)
        results['formal_detection'] = {
            'text': formal_text,
            'is_formal': is_formal
        }
        
        # Test honorific patterns
        honorific_text = "田中様、お疲れ様です。"
        honorific_matches = self._test_honorific_patterns(config, honorific_text)
        results['honorific_patterns'] = honorific_matches
        
        self.test_results['japanese'] = results
        print(f"✅ Japanese patterns tested: {len(results['entity_extraction'])} texts")
    
    def test_korean_patterns(self):
        """Test enhanced Korean patterns."""
        print("🔍 Testing Korean Language Patterns...")
        
        config = LanguageConfigFactory.get_config('ko')
        
        test_texts = [
            "인공지능 기술이 빠르게 발전하고 있다.",
            "서울대학교 김교수가 새로운 연구를 발표했다.",
            "삼성전자 주식회사가 새로운 제품을 개발했다.",
            "부산의 해운대에서 아름다운 경치를 볼 수 있다."
        ]
        
        results = {
            'entity_extraction': [],
            'formal_detection': []
        }
        
        # Test entity extraction
        for text in test_texts:
            entities = self._extract_entities_with_patterns(config, text)
            results['entity_extraction'].append({
                'text': text,
                'entities': entities
            })
        
        # Test formal Korean detection
        formal_text = "안녕하세요. 김철수입니다. 감사합니다."
        is_formal = config.is_formal_korean(formal_text)
        results['formal_detection'] = {
            'text': formal_text,
            'is_formal': is_formal
        }
        
        self.test_results['korean'] = results
        print(f"✅ Korean patterns tested: {len(results['entity_extraction'])} texts")
    
    def test_language_detection(self):
        """Test language detection with enhanced patterns."""
        print("🔍 Testing Language Detection...")
        
        test_cases = [
            ("人工智能技术发展迅速", "zh"),
            ("Искусственный интеллект развивается быстро", "ru"),
            ("人工知能技術が急速に発展している", "ja"),
            ("인공지능 기술이 빠르게 발전하고 있다", "ko"),
            ("Artificial intelligence is developing rapidly", "en")
        ]
        
        results = []
        
        for text, expected_lang in test_cases:
            detected_lang = LanguageConfigFactory.detect_language_from_text(text)
            correct = detected_lang == expected_lang
            results.append({
                'text': text,
                'expected': expected_lang,
                'detected': detected_lang,
                'correct': correct
            })
        
        self.test_results['language_detection'] = results
        print(f"✅ Language detection tested: {len(results)} cases")
    
    def _extract_entities_with_patterns(self, config, text):
        """Extract entities using language-specific patterns."""
        entities = []
        
        # Test person patterns
        for pattern in config.entity_patterns.person:
            matches = re.findall(pattern, text)
            entities.extend([{'type': 'PERSON', 'text': match} for match in matches])
        
        # Test organization patterns
        for pattern in config.entity_patterns.organization:
            matches = re.findall(pattern, text)
            entities.extend([{'type': 'ORGANIZATION', 'text': match} for match in matches])
        
        # Test location patterns
        for pattern in config.entity_patterns.location:
            matches = re.findall(pattern, text)
            entities.extend([{'type': 'LOCATION', 'text': match} for match in matches])
        
        # Test concept patterns
        for pattern in config.entity_patterns.concept:
            matches = re.findall(pattern, text)
            entities.extend([{'type': 'CONCEPT', 'text': match} for match in matches])
        
        return entities
    
    def _test_grammar_patterns(self, config, text):
        """Test grammar patterns for a language."""
        matches = {}
        
        if hasattr(config, 'grammar_patterns'):
            for category, patterns in config.grammar_patterns.items():
                category_matches = []
                for pattern in patterns:
                    found = re.findall(pattern, text)
                    if found:
                        category_matches.extend(found)
                if category_matches:
                    matches[category] = category_matches
        
        return matches
    
    def _test_honorific_patterns(self, config, text):
        """Test honorific patterns for a language."""
        matches = {}
        
        if hasattr(config, 'honorific_patterns'):
            for category, patterns in config.honorific_patterns.items():
                category_matches = []
                for pattern in patterns:
                    found = re.findall(pattern, text)
                    if found:
                        category_matches.extend(found)
                if category_matches:
                    matches[category] = category_matches
        
        return matches
    
    def generate_report(self):
        """Generate a comprehensive test report."""
        print("\n" + "="*60)
        print("📊 PHASE 1 LANGUAGE PATTERN TEST REPORT")
        print("="*60)
        
        total_tests = 0
        passed_tests = 0
        
        # Language detection results
        if 'language_detection' in self.test_results:
            detection_results = self.test_results['language_detection']
            correct_detections = sum(1 for r in detection_results if r['correct'])
            print(f"\n🌍 Language Detection: {correct_detections}/{len(detection_results)} correct")
            total_tests += len(detection_results)
            passed_tests += correct_detections
            
            for result in detection_results:
                status = "✅" if result['correct'] else "❌"
                print(f"  {status} {result['expected']} -> {result['detected']}: {result['text'][:30]}...")
        
        # Individual language results
        for lang, results in self.test_results.items():
            if lang == 'language_detection':
                continue
                
            print(f"\n📝 {lang.upper()} Language Results:")
            
            # Entity extraction results
            if 'entity_extraction' in results:
                total_entities = sum(len(r['entities']) for r in results['entity_extraction'])
                print(f"  📍 Entity Extraction: {total_entities} entities found")
                total_tests += 1
                if total_entities > 0:
                    passed_tests += 1
            
            # Classical/Formal detection results
            if 'classical_detection' in results:
                classical_detected = sum(1 for r in results['classical_detection'] if r['is_classical'])
                print(f"  🏛️  Classical Detection: {classical_detected}/{len(results['classical_detection'])} detected")
                total_tests += 1
                if classical_detected > 0:
                    passed_tests += 1
            
            if 'formal_detection' in results:
                formal_detected = results['formal_detection']['is_formal']
                print(f"  📋 Formal Detection: {'✅' if formal_detected else '❌'}")
                total_tests += 1
                if formal_detected:
                    passed_tests += 1
        
        # Summary
        print(f"\n" + "="*60)
        print(f"📈 SUMMARY: {passed_tests}/{total_tests} tests passed")
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"🎯 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 Phase 1 Language Pattern Enhancement: SUCCESS!")
        else:
            print("⚠️  Phase 1 Language Pattern Enhancement: NEEDS IMPROVEMENT")
        
        print("="*60)
        
        return success_rate >= 80


async def main():
    """Main test function."""
    print("🚀 Starting Phase 1: Enhanced Language-Specific Regex Pattern Testing")
    print("="*70)
    
    tester = LanguagePatternTester()
    
    # Run all tests
    tester.test_chinese_patterns()
    tester.test_russian_patterns()
    tester.test_japanese_patterns()
    tester.test_korean_patterns()
    tester.test_language_detection()
    
    # Generate report
    success = tester.generate_report()
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
