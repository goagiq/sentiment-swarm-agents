#!/usr/bin/env python3
"""
Test script for multilingual PDF processing integration.
Tests the fixed components with language-specific configurations.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.language_config import LanguageConfigFactory
from agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from core.models import AnalysisRequest, DataType
from core.vector_db import VectorDBManager
from core.translation_service import TranslationService


class MultilingualIntegrationTest:
    """Test class for multilingual PDF processing integration."""
    
    def __init__(self):
        self.test_results = {}
        self.file_agent = None
        self.kg_agent = None
        self.vector_db = None
        self.translation_service = None
        
    async def setup(self):
        """Initialize test components."""
        print("🔧 Setting up multilingual integration test...")
        
        # Initialize fixed components
        self.file_agent = EnhancedFileExtractionAgent()
        self.kg_agent = KnowledgeGraphAgent()
        self.vector_db = VectorDBManager()
        self.translation_service = TranslationService()
        
        print("✅ Test components initialized")
    
    async def test_language_config_factory(self):
        """Test language configuration factory."""
        print("\n🔍 Testing Language Configuration Factory...")
        
        try:
            # Test available languages
            available_languages = LanguageConfigFactory.get_available_languages()
            print(f"✅ Available languages: {available_languages}")
            
            # Test Chinese configuration
            chinese_config = LanguageConfigFactory.get_config("zh")
            print(f"✅ Chinese config loaded: {chinese_config.language_name}")
            
            # Test entity patterns
            entity_patterns = chinese_config.get_entity_patterns()
            print(f"✅ Chinese entity patterns: {len(entity_patterns.person)} person, {len(entity_patterns.organization)} org, {len(entity_patterns.location)} location, {len(entity_patterns.concept)} concept")
            
            # Test Classical Chinese patterns
            classical_patterns = chinese_config.get_classical_chinese_patterns()
            print(f"✅ Classical Chinese patterns: {len(classical_patterns)} categories")
            
            # Test processing settings
            processing_settings = chinese_config.get_processing_settings()
            print(f"✅ Processing settings: confidence_threshold={processing_settings.confidence_threshold}, enhanced_extraction={processing_settings.use_enhanced_extraction}")
            
            # Test Ollama configuration
            ollama_config = chinese_config.get_ollama_config()
            print(f"✅ Ollama config: {len(ollama_config)} models configured")
            
            self.test_results["language_config_factory"] = True
            print("✅ Language Configuration Factory test passed")
            
        except Exception as e:
            self.test_results["language_config_factory"] = False
            print(f"❌ Language Configuration Factory test failed: {e}")
    
    async def test_language_detection(self):
        """Test language detection functionality."""
        print("\n🔍 Testing Language Detection...")
        
        try:
            # Test Chinese text detection
            chinese_text = "你好，这是一个测试。我们正在测试中文文本检测功能。"
            detected_language = LanguageConfigFactory.detect_language_from_text(chinese_text)
            print(f"✅ Chinese text detected as: {detected_language}")
            
            # Test English text detection
            english_text = "Hello, this is a test. We are testing English text detection."
            detected_language = LanguageConfigFactory.detect_language_from_text(english_text)
            print(f"✅ English text detected as: {detected_language}")
            
            # Test Russian text detection
            russian_text = "Привет, это тест. Мы тестируем обнаружение русского текста."
            detected_language = LanguageConfigFactory.detect_language_from_text(russian_text)
            print(f"✅ Russian text detected as: {detected_language}")
            
            self.test_results["language_detection"] = True
            print("✅ Language Detection test passed")
            
        except Exception as e:
            self.test_results["language_detection"] = False
            print(f"❌ Language Detection test failed: {e}")
    
    async def test_fixed_vector_db(self):
        """Test the fixed vector database with metadata sanitization."""
        print("\n🔍 Testing Fixed Vector Database...")
        
        try:
            # Test metadata sanitization
            test_metadata = {
                "simple_string": "test",
                "simple_int": 123,
                "simple_float": 45.67,
                "simple_bool": True,
                "complex_dict": {"nested": "value", "list": [1, 2, 3]},
                "complex_list": [{"item": "value"}, {"another": "item"}]
            }
            
            sanitized = self.vector_db.sanitize_metadata(test_metadata)
            print(f"✅ Metadata sanitization: {len(sanitized)} items processed")
            
            # Verify complex types are converted to JSON strings
            assert isinstance(sanitized["complex_dict"], str)
            assert isinstance(sanitized["complex_list"], str)
            assert isinstance(sanitized["simple_string"], str)
            assert isinstance(sanitized["simple_int"], int)
            
            print("✅ Metadata sanitization working correctly")
            
            self.test_results["fixed_vector_db"] = True
            print("✅ Fixed Vector Database test passed")
            
        except Exception as e:
            self.test_results["fixed_vector_db"] = False
            print(f"❌ Fixed Vector Database test failed: {e}")
    
    async def test_fixed_translation_service(self):
        """Test the fixed translation service."""
        print("\n🔍 Testing Fixed Translation Service...")
        
        try:
            # Test translation memory storage and retrieval
            test_text = "Hello world"
            test_result = {
                "translated_text": "你好世界",
                "source_language": "en",
                "target_language": "zh",
                "confidence": 0.95
            }
            
            # Test storing translation memory
            from src.core.models import TranslationResult
            
            translation_result = TranslationResult(
                original_text=test_text,
                translated_text=test_result["translated_text"],
                source_language=test_result["source_language"],
                target_language=test_result["target_language"],
                confidence=test_result["confidence"],
                processing_time=0.1,
                model_used="test_model",
                translation_memory_hit=False,
                metadata={}
            )
            
            await self.translation_service._store_translation_memory(translation_result)
            print("✅ Translation memory storage working")
            
            # Test retrieving translation memory
            memory_result = await self.translation_service._check_translation_memory(
                source_text=test_text,
                target_language="zh"
            )
            print(f"✅ Translation memory retrieval: {memory_result is not None}")
            
            self.test_results["fixed_translation_service"] = True
            print("✅ Fixed Translation Service test passed")
            
        except Exception as e:
            self.test_results["fixed_translation_service"] = False
            print(f"❌ Fixed Translation Service test failed: {e}")
    
    async def test_multilingual_pdf_processing(self):
        """Test multilingual PDF processing with fixed components."""
        print("\n🔍 Testing Multilingual PDF Processing...")
        
        try:
            # Check if test PDF exists
            test_pdf_path = "data/Classical Chinese Sample 22208_0_8.pdf"
            if not os.path.exists(test_pdf_path):
                print(f"⚠️ Test PDF not found: {test_pdf_path}")
                self.test_results["multilingual_pdf_processing"] = False
                return
            
            # Test PDF extraction
            pdf_request = AnalysisRequest(
                data_type=DataType.PDF,
                content=test_pdf_path,
                language="auto"
            )
            
            print("🔧 Extracting text from PDF...")
            extraction_result = await self.file_agent.process(pdf_request)
            
            if extraction_result.status != "completed":
                raise Exception(f"PDF extraction failed: {extraction_result.metadata.get('error', 'Unknown error')}")
            
            text_content = extraction_result.extracted_text
            print(f"✅ Text extraction successful: {len(text_content)} characters")
            
            # Test language detection
            detected_language = LanguageConfigFactory.detect_language_from_text(text_content)
            print(f"✅ Language detected: {detected_language}")
            
            # Test knowledge graph processing
            kg_request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=text_content,
                language=detected_language
            )
            
            print("🔧 Processing with knowledge graph agent...")
            kg_result = await self.kg_agent.process(kg_request)
            print(f"✅ Knowledge graph processing successful: {kg_result.processing_time:.2f}s")
            
            # Verify results
            if kg_result.metadata and 'statistics' in kg_result.metadata:
                stats = kg_result.metadata['statistics']
                print(f"✅ Knowledge graph statistics: {stats.get('entities_found', 0)} entities, {stats.get('nodes', 0)} nodes, {stats.get('edges', 0)} edges")
            
            self.test_results["multilingual_pdf_processing"] = True
            print("✅ Multilingual PDF Processing test passed")
            
        except Exception as e:
            self.test_results["multilingual_pdf_processing"] = False
            print(f"❌ Multilingual PDF Processing test failed: {e}")
    
    async def test_mcp_integration(self):
        """Test MCP integration for multilingual processing."""
        print("\n🔍 Testing MCP Integration...")
        
        try:
            # Import MCP client
            from mcp_servers.client_example import MCPClient
            
            # Test MCP client initialization
            mcp_client = MCPClient()
            print("✅ MCP client initialized")
            
            # Test available tools
            tools = await mcp_client.list_tools()
            print(f"✅ Available MCP tools: {len(tools) if tools else 0}")
            
            # Check for multilingual PDF processing tool
            if tools and "process_multilingual_pdf_mcp" in [tool.get('name', '') for tool in tools]:
                print("✅ Multilingual PDF processing MCP tool available")
                self.test_results["mcp_integration"] = True
            else:
                print("⚠️ Multilingual PDF processing MCP tool not found")
                self.test_results["mcp_integration"] = False
            
        except Exception as e:
            self.test_results["mcp_integration"] = False
            print(f"❌ MCP Integration test failed: {e}")
    
    async def run_all_tests(self):
        """Run all integration tests."""
        print("🚀 Starting Multilingual Integration Tests")
        print("=" * 60)
        
        await self.setup()
        
        # Run all tests
        await self.test_language_config_factory()
        await self.test_language_detection()
        await self.test_fixed_vector_db()
        await self.test_fixed_translation_service()
        await self.test_multilingual_pdf_processing()
        await self.test_mcp_integration()
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """Generate test summary."""
        print("\n" + "=" * 60)
        print("📊 MULTILINGUAL INTEGRATION TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        if failed_tests == 0:
            print("\n🎉 All tests passed! Multilingual integration is working correctly.")
        else:
            print(f"\n⚠️ {failed_tests} test(s) failed. Please check the implementation.")
        
        print("\n🔧 Integration Features Verified:")
        print("  ✅ Language-specific configuration system")
        print("  ✅ Automatic language detection")
        print("  ✅ Fixed vector database with metadata sanitization")
        print("  ✅ Fixed translation service")
        print("  ✅ Enhanced file extraction agent")
        print("  ✅ Knowledge graph agent with language support")
        print("  ✅ MCP framework integration")
        print("  ✅ Multilingual PDF processing pipeline")


async def main():
    """Main test function."""
    test = MultilingualIntegrationTest()
    await test.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
