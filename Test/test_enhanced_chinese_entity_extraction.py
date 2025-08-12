"""
Test Enhanced Chinese Entity Extraction
Validates the improved entity extraction methods for Chinese content.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.enhanced_chinese_entity_extraction import (
    EnhancedChineseEntityExtractor, 
    ChineseEntityValidator
)


class EnhancedChineseEntityExtractionTester:
    """Test enhanced Chinese entity extraction methods."""
    
    def __init__(self):
        self.extractor = EnhancedChineseEntityExtractor()
        self.validator = ChineseEntityValidator()
        self.results = {}
    
    async def test_pattern_based_extraction(self):
        """Test pattern-based entity extraction."""
        print("🧪 Testing Pattern-Based Entity Extraction")
        print("=" * 50)
        
        test_cases = [
            {
                "text": "习近平主席和李克强总理出席了会议。",
                "expected_persons": ["习近平", "李克强"],
                "expected_organizations": [],
                "expected_locations": [],
                "expected_concepts": []
            },
            {
                "text": "马云创立了阿里巴巴集团，马化腾创立了腾讯公司。",
                "expected_persons": ["马云", "马化腾"],
                "expected_organizations": ["阿里巴巴集团", "腾讯公司"],
                "expected_locations": [],
                "expected_concepts": []
            },
            {
                "text": "北京、上海、深圳是中国的主要城市。",
                "expected_persons": [],
                "expected_organizations": [],
                "expected_locations": ["北京", "上海", "深圳"],
                "expected_concepts": []
            },
            {
                "text": "人工智能和机器学习技术发展迅速。",
                "expected_persons": [],
                "expected_organizations": [],
                "expected_locations": [],
                "expected_concepts": ["人工智能", "机器学习"]
            }
        ]
        
        total_accuracy = 0
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {test_case['text']}")
            
            # Extract entities using patterns
            entities = self.extractor._extract_with_patterns(test_case['text'])
            
            # Group by type
            extracted = {
                'PERSON': [e.text for e in entities if e.entity_type == 'PERSON'],
                'ORGANIZATION': [e.text for e in entities if e.entity_type == 'ORGANIZATION'],
                'LOCATION': [e.text for e in entities if e.entity_type == 'LOCATION'],
                'CONCEPT': [e.text for e in entities if e.entity_type == 'CONCEPT']
            }
            
            # Calculate accuracy for each type
            person_accuracy = self._calculate_accuracy(
                extracted['PERSON'], test_case['expected_persons']
            )
            org_accuracy = self._calculate_accuracy(
                extracted['ORGANIZATION'], test_case['expected_organizations']
            )
            loc_accuracy = self._calculate_accuracy(
                extracted['LOCATION'], test_case['expected_locations']
            )
            concept_accuracy = self._calculate_accuracy(
                extracted['CONCEPT'], test_case['expected_concepts']
            )
            
            overall_accuracy = (person_accuracy + org_accuracy + 
                              loc_accuracy + concept_accuracy) / 4
            total_accuracy += overall_accuracy
            
            print(f"✅ Person Accuracy: {person_accuracy:.1%}")
            print(f"✅ Organization Accuracy: {org_accuracy:.1%}")
            print(f"✅ Location Accuracy: {loc_accuracy:.1%}")
            print(f"✅ Concept Accuracy: {concept_accuracy:.1%}")
            print(f"✅ Overall Accuracy: {overall_accuracy:.1%}")
        
        avg_accuracy = total_accuracy / len(test_cases)
        print(f"\n📊 Pattern-Based Extraction Average Accuracy: {avg_accuracy:.1%}")
        
        self.results['pattern_based'] = {
            'accuracy': avg_accuracy,
            'test_cases': len(test_cases)
        }
    
    async def test_dictionary_based_extraction(self):
        """Test dictionary-based entity extraction."""
        print("\n🧪 Testing Dictionary-Based Entity Extraction")
        print("=" * 50)
        
        test_cases = [
            {
                "text": "习近平主席访问了华为公司。",
                "expected_persons": ["习近平"],
                "expected_organizations": ["华为"],
                "expected_locations": [],
                "expected_concepts": []
            },
            {
                "text": "马云在杭州创立了阿里巴巴。",
                "expected_persons": ["马云"],
                "expected_organizations": ["阿里巴巴"],
                "expected_locations": ["杭州"],
                "expected_concepts": []
            },
            {
                "text": "清华大学研究人工智能技术。",
                "expected_persons": [],
                "expected_organizations": ["清华大学"],
                "expected_locations": [],
                "expected_concepts": ["人工智能"]
            }
        ]
        
        total_accuracy = 0
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {test_case['text']}")
            
            # Extract entities using dictionary
            entities = self.extractor._extract_with_dictionary(test_case['text'])
            
            # Group by type
            extracted = {
                'PERSON': [e.text for e in entities if e.entity_type == 'PERSON'],
                'ORGANIZATION': [e.text for e in entities if e.entity_type == 'ORGANIZATION'],
                'LOCATION': [e.text for e in entities if e.entity_type == 'LOCATION'],
                'CONCEPT': [e.text for e in entities if e.entity_type == 'CONCEPT']
            }
            
            # Calculate accuracy
            person_accuracy = self._calculate_accuracy(
                extracted['PERSON'], test_case['expected_persons']
            )
            org_accuracy = self._calculate_accuracy(
                extracted['ORGANIZATION'], test_case['expected_organizations']
            )
            loc_accuracy = self._calculate_accuracy(
                extracted['LOCATION'], test_case['expected_locations']
            )
            concept_accuracy = self._calculate_accuracy(
                extracted['CONCEPT'], test_case['expected_concepts']
            )
            
            overall_accuracy = (person_accuracy + org_accuracy + 
                              loc_accuracy + concept_accuracy) / 4
            total_accuracy += overall_accuracy
            
            print(f"✅ Person Accuracy: {person_accuracy:.1%}")
            print(f"✅ Organization Accuracy: {org_accuracy:.1%}")
            print(f"✅ Location Accuracy: {loc_accuracy:.1%}")
            print(f"✅ Concept Accuracy: {concept_accuracy:.1%}")
            print(f"✅ Overall Accuracy: {overall_accuracy:.1%}")
        
        avg_accuracy = total_accuracy / len(test_cases)
        print(f"\n📊 Dictionary-Based Extraction Average Accuracy: {avg_accuracy:.1%}")
        
        self.results['dictionary_based'] = {
            'accuracy': avg_accuracy,
            'test_cases': len(test_cases)
        }
    
    async def test_entity_validation(self):
        """Test entity validation methods."""
        print("\n🧪 Testing Entity Validation")
        print("=" * 50)
        
        # Test person name validation
        person_tests = [
            ("习近平", True),
            ("马云", True),
            ("李", False),  # Too short
            ("张三李四", True),
            ("John", False),  # Not Chinese
            ("习", False),  # Too short
        ]
        
        person_correct = 0
        for name, expected in person_tests:
            result = self.validator.validate_person_name(name)
            if result == expected:
                person_correct += 1
            print(f"✅ {name}: {result} (expected: {expected})")
        
        person_accuracy = person_correct / len(person_tests)
        print(f"📊 Person Validation Accuracy: {person_accuracy:.1%}")
        
        # Test organization validation
        org_tests = [
            ("华为公司", True),
            ("清华大学", True),
            ("中科院", True),
            ("公司", False),  # Too generic
            ("华为", True),  # In dictionary
            ("ABC", False),  # No Chinese characters
        ]
        
        org_correct = 0
        for org, expected in org_tests:
            result = self.validator.validate_organization_name(org)
            if result == expected:
                org_correct += 1
            print(f"✅ {org}: {result} (expected: {expected})")
        
        org_accuracy = org_correct / len(org_tests)
        print(f"📊 Organization Validation Accuracy: {org_accuracy:.1%}")
        
        # Test location validation
        loc_tests = [
            ("北京", True),
            ("上海市", True),
            ("中国", True),
            ("城市", False),  # Too generic
            ("ABC", False),  # No Chinese characters
        ]
        
        loc_correct = 0
        for loc, expected in loc_tests:
            result = self.validator.validate_location_name(loc)
            if result == expected:
                loc_correct += 1
            print(f"✅ {loc}: {result} (expected: {expected})")
        
        loc_accuracy = loc_correct / len(loc_tests)
        print(f"📊 Location Validation Accuracy: {loc_accuracy:.1%}")
        
        # Test concept validation
        concept_tests = [
            ("人工智能", True),
            ("机器学习", True),
            ("技术", False),  # Too generic
            ("ABC", False),  # No Chinese characters
        ]
        
        concept_correct = 0
        for concept, expected in concept_tests:
            result = self.validator.validate_technical_term(concept)
            if result == expected:
                concept_correct += 1
            print(f"✅ {concept}: {result} (expected: {expected})")
        
        concept_accuracy = concept_correct / len(concept_tests)
        print(f"📊 Concept Validation Accuracy: {concept_accuracy:.1%}")
        
        self.results['validation'] = {
            'person_accuracy': person_accuracy,
            'organization_accuracy': org_accuracy,
            'location_accuracy': loc_accuracy,
            'concept_accuracy': concept_accuracy,
            'overall_accuracy': (person_accuracy + org_accuracy + 
                               loc_accuracy + concept_accuracy) / 4
        }
    
    async def test_enhanced_extraction_pipeline(self):
        """Test the complete enhanced extraction pipeline."""
        print("\n🧪 Testing Enhanced Extraction Pipeline")
        print("=" * 50)
        
        test_cases = [
            {
                "text": "习近平主席在华为公司发表关于人工智能技术的讲话。",
                "expected_persons": ["习近平"],
                "expected_organizations": ["华为公司"],
                "expected_locations": [],
                "expected_concepts": ["人工智能"]
            },
            {
                "text": "马云在杭州创立了阿里巴巴集团，该公司专注于电子商务。",
                "expected_persons": ["马云"],
                "expected_organizations": ["阿里巴巴集团"],
                "expected_locations": ["杭州"],
                "expected_concepts": []
            }
        ]
        
        total_accuracy = 0
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {test_case['text']}")
            
            # Use enhanced extraction pipeline
            entities = await self.extractor.extract_entities_enhanced(
                test_case['text'], "all"
            )
            
            # Group by type
            extracted = {
                'PERSON': [e.text for e in entities if e.entity_type == 'PERSON'],
                'ORGANIZATION': [e.text for e in entities if e.entity_type == 'ORGANIZATION'],
                'LOCATION': [e.text for e in entities if e.entity_type == 'LOCATION'],
                'CONCEPT': [e.text for e in entities if e.entity_type == 'CONCEPT']
            }
            
            # Calculate accuracy
            person_accuracy = self._calculate_accuracy(
                extracted['PERSON'], test_case['expected_persons']
            )
            org_accuracy = self._calculate_accuracy(
                extracted['ORGANIZATION'], test_case['expected_organizations']
            )
            loc_accuracy = self._calculate_accuracy(
                extracted['LOCATION'], test_case['expected_locations']
            )
            concept_accuracy = self._calculate_accuracy(
                extracted['CONCEPT'], test_case['expected_concepts']
            )
            
            overall_accuracy = (person_accuracy + org_accuracy + 
                              loc_accuracy + concept_accuracy) / 4
            total_accuracy += overall_accuracy
            
            print(f"✅ Person Accuracy: {person_accuracy:.1%}")
            print(f"✅ Organization Accuracy: {org_accuracy:.1%}")
            print(f"✅ Location Accuracy: {loc_accuracy:.1%}")
            print(f"✅ Concept Accuracy: {concept_accuracy:.1%}")
            print(f"✅ Overall Accuracy: {overall_accuracy:.1%}")
        
        avg_accuracy = total_accuracy / len(test_cases)
        print(f"\n📊 Enhanced Pipeline Average Accuracy: {avg_accuracy:.1%}")
        
        self.results['enhanced_pipeline'] = {
            'accuracy': avg_accuracy,
            'test_cases': len(test_cases)
        }
    
    def _calculate_accuracy(self, extracted: list, expected: list) -> float:
        """Calculate accuracy of entity extraction."""
        if not expected:
            return 1.0 if not extracted else 0.0
        
        if not extracted:
            return 0.0
        
        # Calculate precision and recall
        correct = len(set(extracted) & set(expected))
        precision = correct / len(extracted) if extracted else 0.0
        recall = correct / len(expected) if expected else 0.0
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    async def run_all_tests(self):
        """Run all enhanced entity extraction tests."""
        print("🚀 Starting Enhanced Chinese Entity Extraction Tests")
        print("=" * 60)
        
        await self.test_pattern_based_extraction()
        await self.test_dictionary_based_extraction()
        await self.test_entity_validation()
        await self.test_enhanced_extraction_pipeline()
        
        self._generate_summary()
        
        return self.results
    
    def _generate_summary(self):
        """Generate test summary."""
        print("\n" + "=" * 60)
        print("📊 ENHANCED CHINESE ENTITY EXTRACTION TEST SUMMARY")
        print("=" * 60)
        
        if 'pattern_based' in self.results:
            print(f"📋 Pattern-Based Extraction: {self.results['pattern_based']['accuracy']:.1%}")
        
        if 'dictionary_based' in self.results:
            print(f"📋 Dictionary-Based Extraction: {self.results['dictionary_based']['accuracy']:.1%}")
        
        if 'validation' in self.results:
            print(f"📋 Entity Validation: {self.results['validation']['overall_accuracy']:.1%}")
        
        if 'enhanced_pipeline' in self.results:
            print(f"📋 Enhanced Pipeline: {self.results['enhanced_pipeline']['accuracy']:.1%}")
        
        # Calculate overall improvement
        original_accuracy = 0.50  # From Phase 6 results
        if 'enhanced_pipeline' in self.results:
            new_accuracy = self.results['enhanced_pipeline']['accuracy']
            improvement = ((new_accuracy - original_accuracy) / original_accuracy) * 100
            print(f"\n🎯 Improvement over Original: {improvement:+.1f}%")
            print(f"📈 Original Accuracy: {original_accuracy:.1%}")
            print(f"📈 Enhanced Accuracy: {new_accuracy:.1%}")


async def main():
    """Run the enhanced Chinese entity extraction tests."""
    tester = EnhancedChineseEntityExtractionTester()
    results = await tester.run_all_tests()
    return results


if __name__ == "__main__":
    asyncio.run(main())
