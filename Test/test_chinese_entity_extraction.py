"""
Test Chinese Entity Extraction for Knowledge Graph
Specialized tests for Chinese language entity extraction accuracy and edge cases.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.knowledge_graph_agent import KnowledgeGraphAgent


class ChineseEntityExtractionTester:
    """Specialized tester for Chinese entity extraction."""
    
    def __init__(self):
        self.agent = KnowledgeGraphAgent()
        self.results = {}
    
    async def test_person_names(self):
        """Test Chinese person name extraction."""
        print("🧪 Testing Chinese Person Name Extraction")
        print("=" * 50)
        
        test_cases = [
            {
                "text": "习近平主席和李克强总理出席了会议。",
                "expected": ["习近平", "李克强"],
                "description": "Political leaders"
            },
            {
                "text": "马云创立了阿里巴巴，马化腾创立了腾讯。",
                "expected": ["马云", "马化腾"],
                "description": "Business leaders"
            },
            {
                "text": "李国杰院士和潘建伟院士在量子计算领域取得突破。",
                "expected": ["李国杰", "潘建伟"],
                "description": "Academic leaders"
            },
            {
                "text": "张勇接替马云成为阿里巴巴CEO。",
                "expected": ["张勇", "马云"],
                "description": "CEO succession"
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {case['description']}")
            print(f"Text: {case['text']}")
            
            try:
                result = await self.agent.extract_entities(case["text"], "zh")
                json_data = result.get("content", [{}])[0].get("json", {})
                entities = json_data.get("entities", [])
                
                extracted_names = [e.get("name", "") for e in entities]
                person_entities = [e for e in entities if e.get("type") == "PERSON"]
                
                accuracy = self._calculate_accuracy(extracted_names, case["expected"])
                
                result_data = {
                    "success": accuracy > 0.5,
                    "accuracy": accuracy,
                    "expected": case["expected"],
                    "extracted": extracted_names,
                    "person_entities": len(person_entities),
                    "total_entities": len(entities)
                }
                
                results.append(result_data)
                
                print(f"✅ Accuracy: {accuracy:.2%}")
                print(f"✅ Extracted: {extracted_names}")
                print(f"✅ Person entities: {len(person_entities)}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({"success": False, "error": str(e)})
        
        self.results["person_names"] = results
        return results
    
    async def test_organization_names(self):
        """Test Chinese organization name extraction."""
        print("\n🧪 Testing Chinese Organization Name Extraction")
        print("=" * 50)
        
        test_cases = [
            {
                "text": "清华大学和北京大学都是中国顶尖大学。",
                "expected": ["清华大学", "北京大学"],
                "description": "Universities"
            },
            {
                "text": "华为技术有限公司和阿里巴巴集团都是科技巨头。",
                "expected": ["华为", "阿里巴巴"],
                "description": "Tech companies"
            },
            {
                "text": "中国科学院计算技术研究所发布了新报告。",
                "expected": ["中国科学院", "计算技术研究所"],
                "description": "Research institutes"
            },
            {
                "text": "中国移动、中国联通、中国电信是三大运营商。",
                "expected": ["中国移动", "中国联通", "中国电信"],
                "description": "Telecom companies"
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {case['description']}")
            print(f"Text: {case['text']}")
            
            try:
                result = await self.agent.extract_entities(case["text"], "zh")
                json_data = result.get("content", [{}])[0].get("json", {})
                entities = json_data.get("entities", [])
                
                extracted_names = [e.get("name", "") for e in entities]
                org_entities = [e for e in entities if e.get("type") == "ORGANIZATION"]
                
                accuracy = self._calculate_accuracy(extracted_names, case["expected"])
                
                result_data = {
                    "success": accuracy > 0.5,
                    "accuracy": accuracy,
                    "expected": case["expected"],
                    "extracted": extracted_names,
                    "org_entities": len(org_entities),
                    "total_entities": len(entities)
                }
                
                results.append(result_data)
                
                print(f"✅ Accuracy: {accuracy:.2%}")
                print(f"✅ Extracted: {extracted_names}")
                print(f"✅ Organization entities: {len(org_entities)}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({"success": False, "error": str(e)})
        
        self.results["organizations"] = results
        return results
    
    async def test_location_names(self):
        """Test Chinese location name extraction."""
        print("\n🧪 Testing Chinese Location Name Extraction")
        print("=" * 50)
        
        test_cases = [
            {
                "text": "北京、上海、深圳、广州是中国的主要城市。",
                "expected": ["北京", "上海", "深圳", "广州"],
                "description": "Major cities"
            },
            {
                "text": "杭州是阿里巴巴的总部所在地。",
                "expected": ["杭州"],
                "description": "Company headquarters"
            },
            {
                "text": "中国、美国、日本、韩国都是重要经济体。",
                "expected": ["中国", "美国", "日本", "韩国"],
                "description": "Countries"
            },
            {
                "text": "长江、黄河是中国的母亲河。",
                "expected": ["长江", "黄河"],
                "description": "Rivers"
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {case['description']}")
            print(f"Text: {case['text']}")
            
            try:
                result = await self.agent.extract_entities(case["text"], "zh")
                json_data = result.get("content", [{}])[0].get("json", {})
                entities = json_data.get("entities", [])
                
                extracted_names = [e.get("name", "") for e in entities]
                location_entities = [e for e in entities if e.get("type") == "LOCATION"]
                
                accuracy = self._calculate_accuracy(extracted_names, case["expected"])
                
                result_data = {
                    "success": accuracy > 0.5,
                    "accuracy": accuracy,
                    "expected": case["expected"],
                    "extracted": extracted_names,
                    "location_entities": len(location_entities),
                    "total_entities": len(entities)
                }
                
                results.append(result_data)
                
                print(f"✅ Accuracy: {accuracy:.2%}")
                print(f"✅ Extracted: {extracted_names}")
                print(f"✅ Location entities: {len(location_entities)}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({"success": False, "error": str(e)})
        
        self.results["locations"] = results
        return results
    
    async def test_technical_terms(self):
        """Test Chinese technical term extraction."""
        print("\n🧪 Testing Chinese Technical Term Extraction")
        print("=" * 50)
        
        test_cases = [
            {
                "text": "人工智能、机器学习、深度学习是热门技术。",
                "expected": ["人工智能", "机器学习", "深度学习"],
                "description": "AI technologies"
            },
            {
                "text": "5G、云计算、大数据、物联网是新兴技术。",
                "expected": ["5G", "云计算", "大数据", "物联网"],
                "description": "Emerging technologies"
            },
            {
                "text": "Transformer、BERT、GPT都是重要的模型。",
                "expected": ["Transformer", "BERT", "GPT"],
                "description": "AI models"
            },
            {
                "text": "量子计算、区块链、虚拟现实是前沿技术。",
                "expected": ["量子计算", "区块链", "虚拟现实"],
                "description": "Frontier technologies"
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {case['description']}")
            print(f"Text: {case['text']}")
            
            try:
                result = await self.agent.extract_entities(case["text"], "zh")
                json_data = result.get("content", [{}])[0].get("json", {})
                entities = json_data.get("entities", [])
                
                extracted_names = [e.get("name", "") for e in entities]
                concept_entities = [e for e in entities if e.get("type") == "CONCEPT"]
                
                accuracy = self._calculate_accuracy(extracted_names, case["expected"])
                
                result_data = {
                    "success": accuracy > 0.3,  # Lower threshold for technical terms
                    "accuracy": accuracy,
                    "expected": case["expected"],
                    "extracted": extracted_names,
                    "concept_entities": len(concept_entities),
                    "total_entities": len(entities)
                }
                
                results.append(result_data)
                
                print(f"✅ Accuracy: {accuracy:.2%}")
                print(f"✅ Extracted: {extracted_names}")
                print(f"✅ Concept entities: {len(concept_entities)}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({"success": False, "error": str(e)})
        
        self.results["technical_terms"] = results
        return results
    
    async def test_edge_cases(self):
        """Test edge cases in Chinese entity extraction."""
        print("\n🧪 Testing Chinese Entity Extraction Edge Cases")
        print("=" * 50)
        
        test_cases = [
            {
                "text": "张三和李四都是王五的朋友。",
                "expected": ["张三", "李四", "王五"],
                "description": "Common names"
            },
            {
                "text": "2024年北京奥运会将在北京举行。",
                "expected": ["北京"],
                "description": "Repetitive entities"
            },
            {
                "text": "中国共产党的总书记习近平访问了美国。",
                "expected": ["中国共产党", "习近平", "美国"],
                "description": "Political entities"
            },
            {
                "text": "华为Mate60和iPhone15都是智能手机。",
                "expected": ["华为", "Mate60", "iPhone15"],
                "description": "Product names"
            }
        ]
        
        results = []
        for i, case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {case['description']}")
            print(f"Text: {case['text']}")
            
            try:
                result = await self.agent.extract_entities(case["text"], "zh")
                json_data = result.get("content", [{}])[0].get("json", {})
                entities = json_data.get("entities", [])
                
                extracted_names = [e.get("name", "") for e in entities]
                accuracy = self._calculate_accuracy(extracted_names, case["expected"])
                
                result_data = {
                    "success": accuracy > 0.3,
                    "accuracy": accuracy,
                    "expected": case["expected"],
                    "extracted": extracted_names,
                    "total_entities": len(entities)
                }
                
                results.append(result_data)
                
                print(f"✅ Accuracy: {accuracy:.2%}")
                print(f"✅ Extracted: {extracted_names}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({"success": False, "error": str(e)})
        
        self.results["edge_cases"] = results
        return results
    
    def _calculate_accuracy(self, extracted, expected):
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
    
    async def run_all_tests(self):
        """Run all Chinese entity extraction tests."""
        print("🚀 Starting Chinese Entity Extraction Tests")
        print("=" * 60)
        
        await self.test_person_names()
        await self.test_organization_names()
        await self.test_location_names()
        await self.test_technical_terms()
        await self.test_edge_cases()
        
        self._generate_summary()
        return self.results
    
    def _generate_summary(self):
        """Generate test summary."""
        print("\n" + "=" * 60)
        print("📊 CHINESE ENTITY EXTRACTION TEST SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        for category, results in self.results.items():
            print(f"\n📋 {category.upper()}:")
            category_passed = 0
            category_total = len(results)
            
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
                    print(f"  {status} Test {i}: {accuracy:.2%} accuracy")
            
            category_accuracy = category_passed / category_total if category_total > 0 else 0
            print(f"  📈 Category Accuracy: {category_accuracy:.2%}")
        
        overall_accuracy = passed_tests / total_tests if total_tests > 0 else 0
        print(f"\n🎯 Overall Accuracy: {overall_accuracy:.2%} ({passed_tests}/{total_tests})")
        
        if overall_accuracy > 0.7:
            print("🎉 Chinese entity extraction is working well!")
        elif overall_accuracy > 0.5:
            print("⚠️  Chinese entity extraction needs improvement.")
        else:
            print("❌ Chinese entity extraction needs significant improvement.")


async def main():
    """Run the Chinese entity extraction tests."""
    tester = ChineseEntityExtractionTester()
    results = await tester.run_all_tests()
    return results


if __name__ == "__main__":
    asyncio.run(main())
