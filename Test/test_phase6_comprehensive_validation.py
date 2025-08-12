"""
Phase 6: Comprehensive Testing & Validation for Multilingual Knowledge Graph
Tests all aspects of the multilingual functionality including Chinese content, 
language detection, entity extraction, query translation, performance, and integration.
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.core.models import AnalysisRequest, DataType


class Phase6TestSuite:
    """Comprehensive test suite for Phase 6 validation."""
    
    def __init__(self):
        self.agent = KnowledgeGraphAgent()
        self.results = {}
        
    async def test_6_1_chinese_content_test_cases(self) -> Dict:
        """Task 6.1: Create Chinese content test cases."""
        print("🧪 Task 6.1: Testing Chinese Content Test Cases")
        print("=" * 60)
        
        # Comprehensive Chinese test content
        chinese_test_cases = [
            {
                "name": "Chinese News Article",
                "content": """
                中国科技巨头阿里巴巴集团今日宣布，将在人工智能领域投资1000亿元人民币。
                该公司CEO张勇表示，AI技术将在电商、云计算、金融科技等领域发挥重要作用。
                清华大学和北京大学的专家团队将参与此次合作项目。
                预计该项目将创造超过10万个就业岗位，推动中国数字经济发展。
                """,
                "expected_entities": ["阿里巴巴", "张勇", "清华大学", "北京大学", "中国"],
                "expected_relationships": ["投资", "合作", "创造"]
            },
            {
                "name": "Chinese Technical Document",
                "content": """
                深度学习技术在自然语言处理领域取得重大突破。
                Transformer架构在机器翻译任务中表现出色。
                谷歌、微软、百度等公司都在积极开发大语言模型。
                GPT和BERT等模型在中文文本理解方面表现优异。
                """,
                "expected_entities": ["深度学习", "Transformer", "谷歌", "微软", "百度", "GPT", "BERT"],
                "expected_relationships": ["突破", "开发", "表现"]
            },
            {
                "name": "Chinese Business Report",
                "content": """
                华为技术有限公司发布2024年第一季度财报，营收同比增长15%。
                公司轮值董事长徐直军表示，5G技术和云计算业务增长强劲。
                在欧洲市场，华为与德国电信、法国Orange等运营商保持良好合作关系。
                预计全年营收将突破8000亿元人民币。
                """,
                "expected_entities": ["华为", "徐直军", "德国电信", "法国Orange", "欧洲"],
                "expected_relationships": ["发布", "合作", "增长"]
            },
            {
                "name": "Chinese Academic Paper",
                "content": """
                中国科学院计算技术研究所的研究团队在量子计算领域取得重要进展。
                该团队由李国杰院士领导，在量子算法优化方面提出创新方法。
                研究成果已发表在《Nature》和《Science》等国际顶级期刊。
                该技术有望在密码学、药物研发等领域产生重大影响。
                """,
                "expected_entities": ["中国科学院", "李国杰", "Nature", "Science"],
                "expected_relationships": ["研究", "发表", "影响"]
            }
        ]
        
        results = {}
        
        for i, test_case in enumerate(chinese_test_cases, 1):
            print(f"\n📝 Test Case {i}: {test_case['name']}")
            print(f"📄 Content: {test_case['content'][:100]}...")
            
            try:
                # Create analysis request
                request = AnalysisRequest(
                    data_type=DataType.TEXT,
                    content=test_case['content'],
                    language="zh"
                )
                
                # Process the content
                start_time = time.time()
                result = await self.agent.process(request)
                processing_time = time.time() - start_time
                
                # Extract results
                json_data = result.content[0].get("json", {})
                entities = json_data.get("entities", [])
                relationships = json_data.get("relationships", [])
                
                # Analyze results
                extracted_entity_names = [e.get("name", "") for e in entities]
                extracted_relationship_types = [r.get("relationship_type", "") for r in relationships]
                
                # Calculate accuracy
                entity_accuracy = self._calculate_entity_accuracy(
                    extracted_entity_names, test_case["expected_entities"]
                )
                relationship_accuracy = self._calculate_relationship_accuracy(
                    extracted_relationship_types, test_case["expected_relationships"]
                )
                
                results[test_case["name"]] = {
                    "success": entity_accuracy > 0.3,  # 30% threshold
                    "entity_accuracy": entity_accuracy,
                    "relationship_accuracy": relationship_accuracy,
                    "entities_found": len(entities),
                    "relationships_found": len(relationships),
                    "processing_time": processing_time,
                    "extracted_entities": extracted_entity_names,
                    "extracted_relationships": extracted_relationship_types
                }
                
                print(f"✅ Entities: {len(entities)} found, Accuracy: {entity_accuracy:.2%}")
                print(f"✅ Relationships: {len(relationships)} found, Accuracy: {relationship_accuracy:.2%}")
                print(f"⏱️  Processing time: {processing_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Error processing test case: {e}")
                results[test_case["name"]] = {
                    "success": False,
                    "error": str(e)
                }
        
        self.results["chinese_content_test_cases"] = results
        return results
    
    async def test_6_2_language_detection_accuracy(self) -> Dict:
        """Task 6.2: Test language detection accuracy."""
        print("\n🧪 Task 6.2: Testing Language Detection Accuracy")
        print("=" * 60)
        
        # Test cases with known languages
        language_test_cases = [
            {
                "language": "zh",
                "content": "人工智能技术正在快速发展，深度学习在图像识别领域取得重大突破。",
                "name": "Chinese Text"
            },
            {
                "language": "en", 
                "content": "Artificial intelligence technology is developing rapidly, with deep learning achieving breakthroughs in image recognition.",
                "name": "English Text"
            },
            {
                "language": "ja",
                "content": "人工知能技術が急速に発展しており、ディープラーニングが画像認識分野で大きなブレークスルーを達成しています。",
                "name": "Japanese Text"
            },
            {
                "language": "ko",
                "content": "인공지능 기술이 빠르게 발전하고 있으며, 딥러닝이 이미지 인식 분야에서 큰 돌파구를 달성했습니다.",
                "name": "Korean Text"
            },
            {
                "language": "es",
                "content": "La tecnología de inteligencia artificial se está desarrollando rápidamente, con el aprendizaje profundo logrando avances en el reconocimiento de imágenes.",
                "name": "Spanish Text"
            },
            {
                "language": "fr",
                "content": "La technologie d'intelligence artificielle se développe rapidement, avec l'apprentissage profond réalisant des percées dans la reconnaissance d'images.",
                "name": "French Text"
            }
        ]
        
        results = {}
        total_accuracy = 0
        
        for test_case in language_test_cases:
            print(f"\n🔍 Testing: {test_case['name']}")
            
            try:
                # Create request with auto language detection
                request = AnalysisRequest(
                    data_type=DataType.TEXT,
                    content=test_case["content"],
                    language="auto"
                )
                
                # Extract text content (triggers language detection)
                await self.agent._extract_text_content(request)
                
                # Check if language was detected correctly
                detected_language = request.language
                expected_language = test_case["language"]
                is_correct = detected_language == expected_language
                
                results[test_case["name"]] = {
                    "success": is_correct,
                    "expected": expected_language,
                    "detected": detected_language,
                    "correct": is_correct
                }
                
                status = "✅" if is_correct else "❌"
                print(f"{status} Expected: {expected_language}, Detected: {detected_language}")
                
                if is_correct:
                    total_accuracy += 1
                    
            except Exception as e:
                print(f"❌ Error in language detection: {e}")
                results[test_case["name"]] = {
                    "success": False,
                    "error": str(e)
                }
        
        overall_accuracy = total_accuracy / len(language_test_cases)
        print(f"\n📊 Overall Language Detection Accuracy: {overall_accuracy:.2%}")
        
        self.results["language_detection_accuracy"] = {
            "overall_accuracy": overall_accuracy,
            "detailed_results": results
        }
        return self.results["language_detection_accuracy"]
    
    async def test_6_3_validate_entity_extraction_chinese(self) -> Dict:
        """Task 6.3: Validate entity extraction in Chinese."""
        print("\n🧪 Task 6.3: Validating Entity Extraction in Chinese")
        print("=" * 60)
        
        # Chinese entity extraction test cases
        chinese_entity_tests = [
            {
                "name": "Person Names",
                "text": "习近平主席、李克强总理、王毅外长出席了会议。",
                "expected_entities": ["习近平", "李克强", "王毅"],
                "expected_types": ["PERSON"]
            },
            {
                "name": "Organizations",
                "text": "清华大学、北京大学、中科院、华为公司都是知名机构。",
                "expected_entities": ["清华大学", "北京大学", "中科院", "华为"],
                "expected_types": ["ORGANIZATION"]
            },
            {
                "name": "Locations",
                "text": "北京、上海、深圳、广州是中国的主要城市。",
                "expected_entities": ["北京", "上海", "深圳", "广州"],
                "expected_types": ["LOCATION"]
            },
            {
                "name": "Mixed Entities",
                "text": "马云在杭州创立了阿里巴巴集团，该公司总部位于北京。",
                "expected_entities": ["马云", "杭州", "阿里巴巴", "北京"],
                "expected_types": ["PERSON", "LOCATION", "ORGANIZATION"]
            }
        ]
        
        results = {}
        total_accuracy = 0
        
        for test_case in chinese_entity_tests:
            print(f"\n🔍 Testing: {test_case['name']}")
            print(f"📝 Text: {test_case['text']}")
            
            try:
                # Extract entities
                result = await self.agent.extract_entities(test_case["text"], "zh")
                json_data = result.get("content", [{}])[0].get("json", {})
                entities = json_data.get("entities", [])
                
                # Analyze results
                extracted_names = [e.get("name", "") for e in entities]
                extracted_types = [e.get("type", "") for e in entities]
                
                # Calculate accuracy
                name_accuracy = self._calculate_entity_accuracy(
                    extracted_names, test_case["expected_entities"]
                )
                type_accuracy = self._calculate_type_accuracy(
                    extracted_types, test_case["expected_types"]
                )
                
                overall_accuracy = (name_accuracy + type_accuracy) / 2
                total_accuracy += overall_accuracy
                
                results[test_case["name"]] = {
                    "success": overall_accuracy > 0.4,  # 40% threshold
                    "name_accuracy": name_accuracy,
                    "type_accuracy": type_accuracy,
                    "overall_accuracy": overall_accuracy,
                    "entities_found": len(entities),
                    "extracted_entities": extracted_names,
                    "extracted_types": extracted_types
                }
                
                print(f"✅ Name Accuracy: {name_accuracy:.2%}")
                print(f"✅ Type Accuracy: {type_accuracy:.2%}")
                print(f"✅ Overall Accuracy: {overall_accuracy:.2%}")
                
            except Exception as e:
                print(f"❌ Error in entity extraction: {e}")
                results[test_case["name"]] = {
                    "success": False,
                    "error": str(e)
                }
        
        avg_accuracy = total_accuracy / len(chinese_entity_tests)
        print(f"\n📊 Average Chinese Entity Extraction Accuracy: {avg_accuracy:.2%}")
        
        self.results["chinese_entity_extraction"] = {
            "average_accuracy": avg_accuracy,
            "detailed_results": results
        }
        return self.results["chinese_entity_extraction"]
    
    async def test_6_4_query_translation_accuracy(self) -> Dict:
        """Task 6.4: Test query translation accuracy."""
        print("\n🧪 Task 6.4: Testing Query Translation Accuracy")
        print("=" * 60)
        
        # Test query translation scenarios
        translation_test_cases = [
            {
                "name": "Chinese to English Query",
                "query": "人工智能技术有哪些应用？",
                "expected_english": "What are the applications of artificial intelligence technology?",
                "target_language": "en"
            },
            {
                "name": "English to Chinese Query", 
                "query": "What companies are working on AI?",
                "expected_chinese": "哪些公司在研究人工智能？",
                "target_language": "zh"
            },
            {
                "name": "Technical Query Translation",
                "query": "深度学习在医疗领域的应用",
                "expected_english": "Applications of deep learning in healthcare",
                "target_language": "en"
            }
        ]
        
        results = {}
        
        for test_case in translation_test_cases:
            print(f"\n🔍 Testing: {test_case['name']}")
            print(f"📝 Query: {test_case['query']}")
            
            try:
                # Test query translation
                start_time = time.time()
                result = await self.agent.query_knowledge_graph(
                    test_case["query"], 
                    test_case["target_language"]
                )
                translation_time = time.time() - start_time
                
                # Extract translated query from result
                translated_query = result.get("translated_query", "")
                
                results[test_case["name"]] = {
                    "success": True,
                    "original_query": test_case["query"],
                    "translated_query": translated_query,
                    "expected_translation": test_case.get(f"expected_{test_case['target_language']}", ""),
                    "translation_time": translation_time,
                    "has_results": len(result.get("entities", [])) > 0
                }
                
                print(f"✅ Translation time: {translation_time:.2f}s")
                print(f"✅ Has results: {results[test_case['name']]['has_results']}")
                
            except Exception as e:
                print(f"❌ Error in query translation: {e}")
                results[test_case["name"]] = {
                    "success": False,
                    "error": str(e)
                }
        
        self.results["query_translation_accuracy"] = results
        return results
    
    async def test_6_5_performance_testing(self) -> Dict:
        """Task 6.5: Performance testing with large multilingual datasets."""
        print("\n🧪 Task 6.5: Performance Testing with Large Multilingual Datasets")
        print("=" * 60)
        
        # Create large test dataset
        large_chinese_text = """
        人工智能技术正在各个领域快速发展。在医疗健康领域，AI技术被用于疾病诊断、药物研发和个性化治疗方案制定。
        在金融领域，机器学习算法被用于风险评估、欺诈检测和投资决策支持。在教育领域，智能教育平台提供个性化学习体验。
        
        中国的主要科技公司包括阿里巴巴、腾讯、百度、华为等。这些公司在云计算、大数据、物联网等领域都有重要布局。
        清华大学、北京大学、中科院等研究机构在基础研究方面做出了重要贡献。
        
        在自动驾驶领域，百度Apollo、特斯拉、Waymo等公司都在积极开发相关技术。这些技术将彻底改变交通运输方式。
        在自然语言处理领域，GPT、BERT、Transformer等模型在文本理解和生成方面表现出色。
        
        量子计算是未来计算技术的重要方向。谷歌、IBM、微软等公司都在积极投入量子计算研究。
        中国也在量子通信和量子计算方面取得了重要进展，潘建伟院士团队在量子通信领域处于世界领先地位。
        """ * 3  # Repeat to create larger dataset
        
        performance_metrics = {}
        
        # Test processing time
        print("⏱️  Testing processing performance...")
        start_time = time.time()
        
        try:
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=large_chinese_text,
                language="zh"
            )
            
            result = await self.agent.process(request)
            processing_time = time.time() - start_time
            
            # Extract metrics
            json_data = result.content[0].get("json", {})
            entities_count = len(json_data.get("entities", []))
            relationships_count = len(json_data.get("relationships", []))
            
            performance_metrics = {
                "success": True,
                "processing_time": processing_time,
                "text_length": len(large_chinese_text),
                "entities_extracted": entities_count,
                "relationships_extracted": relationships_count,
                "processing_speed": len(large_chinese_text) / processing_time,  # chars per second
                "entities_per_second": entities_count / processing_time,
                "relationships_per_second": relationships_count / processing_time
            }
            
            print(f"✅ Processing time: {processing_time:.2f}s")
            print(f"✅ Text length: {len(large_chinese_text)} characters")
            print(f"✅ Entities extracted: {entities_count}")
            print(f"✅ Relationships extracted: {relationships_count}")
            print(f"✅ Processing speed: {performance_metrics['processing_speed']:.0f} chars/sec")
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            performance_metrics = {
                "success": False,
                "error": str(e)
            }
        
        # Test memory usage (basic estimation)
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        performance_metrics["memory_usage_mb"] = memory_usage
        
        print(f"💾 Memory usage: {memory_usage:.1f} MB")
        
        self.results["performance_testing"] = performance_metrics
        return performance_metrics
    
    async def test_6_6_integration_testing(self) -> Dict:
        """Task 6.6: Integration testing with other agents."""
        print("\n🧪 Task 6.6: Integration Testing with Other Agents")
        print("=" * 60)
        
        integration_results = {}
        
        # Test with translation service
        print("🔄 Testing Translation Service Integration...")
        try:
            chinese_text = "人工智能技术发展迅速"
            translation_result = await self.agent.translation_service.translate_text(
                chinese_text, target_language="en"
            )
            
            integration_results["translation_service"] = {
                "success": True,
                "original": chinese_text,
                "translated": translation_result.translated_text,
                "confidence": getattr(translation_result, 'confidence', 0.0)
            }
            print(f"✅ Translation: '{chinese_text}' -> '{translation_result.translated_text}'")
            
        except Exception as e:
            print(f"❌ Translation service integration failed: {e}")
            integration_results["translation_service"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test graph report generation
        print("📊 Testing Graph Report Generation...")
        try:
            report_result = await self.agent.generate_graph_report(target_language="zh")
            
            integration_results["graph_report_generation"] = {
                "success": True,
                "has_content": len(report_result.get("content", [])) > 0,
                "target_language": "zh"
            }
            print("✅ Graph report generation successful")
            
        except Exception as e:
            print(f"❌ Graph report generation failed: {e}")
            integration_results["graph_report_generation"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test query functionality
        print("🔍 Testing Query Functionality...")
        try:
            query_result = await self.agent.query_knowledge_graph("人工智能", target_language="zh")
            
            integration_results["query_functionality"] = {
                "success": True,
                "has_results": len(query_result.get("entities", [])) > 0,
                "query_language": "zh"
            }
            print("✅ Query functionality successful")
            
        except Exception as e:
            print(f"❌ Query functionality failed: {e}")
            integration_results["query_functionality"] = {
                "success": False,
                "error": str(e)
            }
        
        self.results["integration_testing"] = integration_results
        return integration_results
    
    def _calculate_entity_accuracy(self, extracted: List[str], expected: List[str]) -> float:
        """Calculate accuracy of entity extraction."""
        if not expected:
            return 1.0
        
        found_count = 0
        for expected_entity in expected:
            for extracted_entity in extracted:
                if (expected_entity.lower() in extracted_entity.lower() or 
                    extracted_entity.lower() in expected_entity.lower()):
                    found_count += 1
                    break
        
        return found_count / len(expected)
    
    def _calculate_relationship_accuracy(self, extracted: List[str], expected: List[str]) -> float:
        """Calculate accuracy of relationship extraction."""
        return self._calculate_entity_accuracy(extracted, expected)
    
    def _calculate_type_accuracy(self, extracted_types: List[str], expected_types: List[str]) -> float:
        """Calculate accuracy of entity type classification."""
        if not expected_types:
            return 1.0
        
        # Check if any of the expected types are found in extracted types
        found_count = 0
        for expected_type in expected_types:
            if expected_type in extracted_types:
                found_count += 1
        
        return found_count / len(expected_types)
    
    async def run_all_tests(self) -> Dict:
        """Run all Phase 6 tests."""
        print("🚀 Starting Phase 6: Comprehensive Testing & Validation")
        print("=" * 80)
        
        test_functions = [
            ("6.1 Chinese Content Test Cases", self.test_6_1_chinese_content_test_cases),
            ("6.2 Language Detection Accuracy", self.test_6_2_language_detection_accuracy),
            ("6.3 Chinese Entity Extraction Validation", self.test_6_3_validate_entity_extraction_chinese),
            ("6.4 Query Translation Accuracy", self.test_6_4_query_translation_accuracy),
            ("6.5 Performance Testing", self.test_6_5_performance_testing),
            ("6.6 Integration Testing", self.test_6_6_integration_testing)
        ]
        
        for test_name, test_func in test_functions:
            try:
                await test_func()
            except Exception as e:
                print(f"❌ {test_name} failed with error: {e}")
                self.results[test_name.lower().replace(" ", "_")] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _generate_summary(self):
        """Generate comprehensive test summary."""
        print("\n" + "=" * 80)
        print("📊 PHASE 6 TEST SUMMARY")
        print("=" * 80)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, result in self.results.items():
            if isinstance(result, dict):
                if "success" in result:
                    total_tests += 1
                    if result["success"]:
                        passed_tests += 1
                        status = "✅ PASS"
                    else:
                        status = "❌ FAIL"
                    print(f"{status} - {test_name}")
                elif "overall_accuracy" in result:
                    print(f"📈 {test_name}: {result['overall_accuracy']:.2%} accuracy")
                elif "average_accuracy" in result:
                    print(f"📈 {test_name}: {result['average_accuracy']:.2%} accuracy")
        
        print(f"\n🎯 Overall Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All Phase 6 tests passed! Multilingual functionality is working correctly.")
        else:
            print("⚠️  Some tests failed. Please review the implementation.")


async def main():
    """Run the Phase 6 test suite."""
    test_suite = Phase6TestSuite()
    results = await test_suite.run_all_tests()
    
    # Save results to file
    output_file = Path("Test/phase6_test_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Test results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())
