"""
Test Query Translation Accuracy for Knowledge Graph
Tests the accuracy and performance of query translation functionality.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.knowledge_graph_agent import KnowledgeGraphAgent


class QueryTranslationTester:
    """Tester for query translation functionality."""
    
    def __init__(self):
        self.agent = KnowledgeGraphAgent()
        self.results = {}
    
    async def test_basic_translation(self):
        """Test basic query translation scenarios."""
        print("🧪 Testing Basic Query Translation")
        print("=" * 50)
        
        test_cases = [
            {
                "query": "人工智能技术",
                "target_language": "en",
                "expected_keywords": ["artificial", "intelligence", "technology"],
                "description": "Chinese to English - AI technology"
            },
            {
                "query": "What is artificial intelligence?",
                "target_language": "zh",
                "expected_keywords": ["人工智能", "什么"],
                "description": "English to Chinese - AI question"
            },
            {
                "query": "深度学习应用",
                "target_language": "en",
                "expected_keywords": ["deep", "learning", "application"],
                "description": "Chinese to English - Deep learning"
            },
            {
                "query": "Machine learning companies",
                "target_language": "zh",
                "expected_keywords": ["机器学习", "公司"],
                "description": "English to Chinese - ML companies"
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {case['description']}")
            print(f"Query: {case['query']}")
            print(f"Target: {case['target_language']}")
            
            try:
                start_time = time.time()
                result = await self.agent.query_knowledge_graph(
                    case["query"], 
                    case["target_language"]
                )
                translation_time = time.time() - start_time
                
                # Extract translation info
                translated_query = result.get("translated_query", "")
                original_query = result.get("original_query", case["query"])
                
                # Check if expected keywords are present
                accuracy = self._check_keyword_accuracy(
                    translated_query, case["expected_keywords"]
                )
                
                result_data = {
                    "success": accuracy > 0.3,  # 30% threshold
                    "accuracy": accuracy,
                    "original_query": original_query,
                    "translated_query": translated_query,
                    "translation_time": translation_time,
                    "has_results": len(result.get("entities", [])) > 0
                }
                
                results.append(result_data)
                
                print(f"✅ Translation: '{translated_query}'")
                print(f"✅ Accuracy: {accuracy:.2%}")
                print(f"✅ Time: {translation_time:.2f}s")
                print(f"✅ Has results: {result_data['has_results']}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({"success": False, "error": str(e)})
        
        self.results["basic_translation"] = results
        return results
    
    async def test_technical_queries(self):
        """Test technical query translation."""
        print("\n🧪 Testing Technical Query Translation")
        print("=" * 50)
        
        test_cases = [
            {
                "query": "量子计算在密码学中的应用",
                "target_language": "en",
                "expected_keywords": ["quantum", "computing", "cryptography"],
                "description": "Quantum computing in cryptography"
            },
            {
                "query": "Transformer architecture in NLP",
                "target_language": "zh",
                "expected_keywords": ["Transformer", "架构", "自然语言"],
                "description": "Transformer architecture"
            },
            {
                "query": "5G网络技术发展",
                "target_language": "en",
                "expected_keywords": ["5G", "network", "technology"],
                "description": "5G network technology"
            },
            {
                "query": "Blockchain applications in finance",
                "target_language": "zh",
                "expected_keywords": ["区块链", "应用", "金融"],
                "description": "Blockchain in finance"
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {case['description']}")
            print(f"Query: {case['query']}")
            
            try:
                start_time = time.time()
                result = await self.agent.query_knowledge_graph(
                    case["query"], 
                    case["target_language"]
                )
                translation_time = time.time() - start_time
                
                translated_query = result.get("translated_query", "")
                accuracy = self._check_keyword_accuracy(
                    translated_query, case["expected_keywords"]
                )
                
                result_data = {
                    "success": accuracy > 0.3,
                    "accuracy": accuracy,
                    "original_query": case["query"],
                    "translated_query": translated_query,
                    "translation_time": translation_time,
                    "has_results": len(result.get("entities", [])) > 0
                }
                
                results.append(result_data)
                
                print(f"✅ Translation: '{translated_query}'")
                print(f"✅ Accuracy: {accuracy:.2%}")
                print(f"✅ Time: {translation_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({"success": False, "error": str(e)})
        
        self.results["technical_queries"] = results
        return results
    
    async def test_company_queries(self):
        """Test company and organization queries."""
        print("\n🧪 Testing Company Query Translation")
        print("=" * 50)
        
        test_cases = [
            {
                "query": "华为公司技术发展",
                "target_language": "en",
                "expected_keywords": ["Huawei", "technology", "development"],
                "description": "Huawei technology development"
            },
            {
                "query": "Alibaba cloud services",
                "target_language": "zh",
                "expected_keywords": ["阿里巴巴", "云", "服务"],
                "description": "Alibaba cloud services"
            },
            {
                "query": "腾讯游戏业务",
                "target_language": "en",
                "expected_keywords": ["Tencent", "game", "business"],
                "description": "Tencent gaming business"
            },
            {
                "query": "Google AI research",
                "target_language": "zh",
                "expected_keywords": ["谷歌", "人工智能", "研究"],
                "description": "Google AI research"
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {case['description']}")
            print(f"Query: {case['query']}")
            
            try:
                start_time = time.time()
                result = await self.agent.query_knowledge_graph(
                    case["query"], 
                    case["target_language"]
                )
                translation_time = time.time() - start_time
                
                translated_query = result.get("translated_query", "")
                accuracy = self._check_keyword_accuracy(
                    translated_query, case["expected_keywords"]
                )
                
                result_data = {
                    "success": accuracy > 0.3,
                    "accuracy": accuracy,
                    "original_query": case["query"],
                    "translated_query": translated_query,
                    "translation_time": translation_time,
                    "has_results": len(result.get("entities", [])) > 0
                }
                
                results.append(result_data)
                
                print(f"✅ Translation: '{translated_query}'")
                print(f"✅ Accuracy: {accuracy:.2%}")
                print(f"✅ Time: {translation_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({"success": False, "error": str(e)})
        
        self.results["company_queries"] = results
        return results
    
    async def test_complex_queries(self):
        """Test complex multi-part queries."""
        print("\n🧪 Testing Complex Query Translation")
        print("=" * 50)
        
        test_cases = [
            {
                "query": "人工智能在医疗健康领域的应用和发展趋势",
                "target_language": "en",
                "expected_keywords": ["artificial", "intelligence", "healthcare"],
                "description": "AI in healthcare applications and trends"
            },
            {
                "query": "How does machine learning improve cybersecurity?",
                "target_language": "zh",
                "expected_keywords": ["机器学习", "网络安全", "如何"],
                "description": "ML improving cybersecurity"
            },
            {
                "query": "自动驾驶技术在中国的商业化进程",
                "target_language": "en",
                "expected_keywords": ["autonomous", "driving", "China", "commercialization"],
                "description": "Autonomous driving commercialization in China"
            },
            {
                "query": "Future of quantum computing in drug discovery",
                "target_language": "zh",
                "expected_keywords": ["量子计算", "药物", "发现", "未来"],
                "description": "Quantum computing in drug discovery"
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {case['description']}")
            print(f"Query: {case['query']}")
            
            try:
                start_time = time.time()
                result = await self.agent.query_knowledge_graph(
                    case["query"], 
                    case["target_language"]
                )
                translation_time = time.time() - start_time
                
                translated_query = result.get("translated_query", "")
                accuracy = self._check_keyword_accuracy(
                    translated_query, case["expected_keywords"]
                )
                
                result_data = {
                    "success": accuracy > 0.3,
                    "accuracy": accuracy,
                    "original_query": case["query"],
                    "translated_query": translated_query,
                    "translation_time": translation_time,
                    "has_results": len(result.get("entities", [])) > 0
                }
                
                results.append(result_data)
                
                print(f"✅ Translation: '{translated_query}'")
                print(f"✅ Accuracy: {accuracy:.2%}")
                print(f"✅ Time: {translation_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({"success": False, "error": str(e)})
        
        self.results["complex_queries"] = results
        return results
    
    async def test_performance_benchmarks(self):
        """Test translation performance with various query lengths."""
        print("\n🧪 Testing Translation Performance")
        print("=" * 50)
        
        performance_tests = [
            {
                "query": "AI",
                "target_language": "zh",
                "description": "Short query"
            },
            {
                "query": "人工智能技术发展",
                "target_language": "en",
                "description": "Medium query"
            },
            {
                "query": "人工智能技术在医疗健康领域的应用和发展趋势分析",
                "target_language": "en",
                "description": "Long query"
            }
        ]
        
        results = []
        for i, test in enumerate(performance_tests, 1):
            print(f"\n📝 Test {i}: {test['description']}")
            print(f"Query: {test['query']}")
            
            try:
                # Run multiple times for average
                times = []
                for _ in range(3):
                    start_time = time.time()
                    result = await self.agent.query_knowledge_graph(
                        test["query"], 
                        test["target_language"]
                    )
                    translation_time = time.time() - start_time
                    times.append(translation_time)
                
                avg_time = sum(times) / len(times)
                translated_query = result.get("translated_query", "")
                
                result_data = {
                    "success": True,
                    "original_query": test["query"],
                    "translated_query": translated_query,
                    "average_time": avg_time,
                    "min_time": min(times),
                    "max_time": max(times),
                    "query_length": len(test["query"])
                }
                
                results.append(result_data)
                
                print(f"✅ Translation: '{translated_query}'")
                print(f"✅ Avg time: {avg_time:.3f}s")
                print(f"✅ Time range: {min(times):.3f}s - {max(times):.3f}s")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({"success": False, "error": str(e)})
        
        self.results["performance_benchmarks"] = results
        return results
    
    def _check_keyword_accuracy(self, translated_text, expected_keywords):
        """Check if expected keywords are present in translated text."""
        if not expected_keywords:
            return 1.0
        
        translated_lower = translated_text.lower()
        found_count = 0
        
        for keyword in expected_keywords:
            if keyword.lower() in translated_lower:
                found_count += 1
        
        return found_count / len(expected_keywords)
    
    async def run_all_tests(self):
        """Run all query translation tests."""
        print("🚀 Starting Query Translation Tests")
        print("=" * 60)
        
        await self.test_basic_translation()
        await self.test_technical_queries()
        await self.test_company_queries()
        await self.test_complex_queries()
        await self.test_performance_benchmarks()
        
        self._generate_summary()
        return self.results
    
    def _generate_summary(self):
        """Generate test summary."""
        print("\n" + "=" * 60)
        print("📊 QUERY TRANSLATION TEST SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        total_accuracy = 0
        total_time = 0
        
        for category, results in self.results.items():
            print(f"\n📋 {category.upper()}:")
            category_passed = 0
            category_total = len(results)
            category_accuracy = 0
            
            for i, result in enumerate(results, 1):
                if "success" in result:
                    total_tests += 1
                    if result["success"]:
                        passed_tests += 1
                        category_passed += 1
                        status = "✅"
                    else:
                        status = "❌"
                    
                    accuracy = result.get("accuracy", 0)
                    category_accuracy += accuracy
                    total_accuracy += accuracy
                    
                    translation_time = result.get("translation_time", 0)
                    total_time += translation_time
                    
                    print(f"  {status} Test {i}: {accuracy:.2%} accuracy, {translation_time:.2f}s")
            
            avg_category_accuracy = category_accuracy / category_total if category_total > 0 else 0
            print(f"  📈 Category Avg Accuracy: {avg_category_accuracy:.2%}")
        
        overall_accuracy = total_accuracy / total_tests if total_tests > 0 else 0
        avg_time = total_time / total_tests if total_tests > 0 else 0
        
        print(f"\n🎯 Overall Results:")
        print(f"  📊 Accuracy: {overall_accuracy:.2%}")
        print(f"  ⏱️  Avg Time: {avg_time:.3f}s")
        print(f"  ✅ Passed: {passed_tests}/{total_tests}")
        
        if overall_accuracy > 0.7:
            print("🎉 Query translation is working excellently!")
        elif overall_accuracy > 0.5:
            print("⚠️  Query translation needs improvement.")
        else:
            print("❌ Query translation needs significant improvement.")


async def main():
    """Run the query translation tests."""
    tester = QueryTranslationTester()
    results = await tester.run_all_tests()
    return results


if __name__ == "__main__":
    asyncio.run(main())
