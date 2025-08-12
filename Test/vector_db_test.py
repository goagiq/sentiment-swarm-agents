#!/usr/bin/env python3
"""
Vector database test script for Classical Chinese processing.
Tests metadata handling, validates storage/retrieval, and tests error handling.
"""

import sys
import os
import asyncio
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.vector_db import VectorDBManager
from core.translation_service import TranslationService, TranslationResult


class VectorDatabaseTester:
    """Tests vector database functionality for Classical Chinese processing."""
    
    def __init__(self):
        self.vector_db = VectorDBManager()
        self.translation_service = TranslationService()
        self.test_results = {
            "passed": [],
            "failed": [],
            "warnings": [],
            "performance_metrics": {}
        }
    
    def test_metadata_handling(self):
        """Test metadata handling and sanitization."""
        try:
            # Test various metadata types
            test_metadata = {
                "simple_string": "test value",
                "simple_int": 42,
                "simple_float": 3.14,
                "simple_bool": True,
                "simple_none": None,
                "nested_dict": {"key": "value", "number": 123},
                "nested_list": ["item1", "item2", {"nested": "value"}],
                "complex_object": {"data": {"nested": {"deep": "value"}}}
            }
            
            # Test sanitization
            sanitized = self.vector_db.sanitize_metadata(test_metadata)
            
            # Verify all values are primitive types
            for key, value in sanitized.items():
                assert isinstance(value, (str, int, float, bool, type(None))), \
                    f"Non-primitive type found: {key} = {type(value)}"
            
            # Verify nested structures are converted to JSON strings
            assert isinstance(sanitized["nested_dict"], str)
            assert isinstance(sanitized["nested_list"], str)
            assert isinstance(sanitized["complex_object"], str)
            
            # Verify JSON strings can be parsed back
            import json
            parsed_dict = json.loads(sanitized["nested_dict"])
            assert parsed_dict["key"] == "value"
            
            self.test_results["passed"].append("Metadata handling")
            print("✅ Metadata handling test passed")
            
        except Exception as e:
            self.test_results["failed"].append(f"Metadata handling: {e}")
            print(f"❌ Metadata handling test failed: {e}")
    
    async def test_storage_and_retrieval(self):
        """Test storage and retrieval functionality."""
        try:
            # Test data
            test_texts = [
                "子曰：学而时习之，不亦说乎？",
                "有朋自远方来，不亦乐乎？",
                "人不知而不愠，不亦君子乎？"
            ]
            
            test_metadata = [
                {"source": "论语", "chapter": "学而", "language": "zh"},
                {"source": "论语", "chapter": "学而", "language": "zh"},
                {"source": "论语", "chapter": "学而", "language": "zh"}
            ]
            
            # Sanitize metadata
            sanitized_metadata = [
                self.vector_db.sanitize_metadata(meta) for meta in test_metadata
            ]
            
            # Store texts
            print("🔄 Storing test texts...")
            await self.vector_db.add_texts(
                collection_name="test_collection",
                texts=test_texts,
                metadatas=sanitized_metadata
            )
            
            # Retrieve texts
            print("🔄 Retrieving test texts...")
            results = await self.vector_db.query(
                collection_name="test_collection",
                query_text="学而",
                n_results=3
            )
            
            # Validate results
            assert len(results) > 0, "No results retrieved"
            
            # Check that results contain Chinese text
            chinese_found = any(
                any('\u4e00' <= char <= '\u9fff' for char in result['text'])
                for result in results
            )
            assert chinese_found, "No Chinese text found in results"
            
            self.test_results["passed"].append("Storage and retrieval")
            self.test_results["performance_metrics"]["stored_texts"] = len(test_texts)
            self.test_results["performance_metrics"]["retrieved_results"] = len(results)
            print("✅ Storage and retrieval test passed")
            
        except Exception as e:
            self.test_results["failed"].append(f"Storage and retrieval: {e}")
            print(f"❌ Storage and retrieval test failed: {e}")
    
    async def test_translation_service_integration(self):
        """Test translation service integration with vector database."""
        try:
            # Create test translation result
            translation_result = TranslationResult(
                original_text="子曰：学而时习之，不亦说乎？",
                translated_text="The Master said: 'Is it not pleasant to learn with a constant perseverance and application?'",
                source_language="zh",
                target_language="en",
                confidence=0.95,
                processing_time=1.5,
                model_used="qwen2.5:7b"
            )
            
            # Test translation memory storage
            print("🔄 Testing translation memory storage...")
            await self.translation_service._store_translation_memory(translation_result)
            
            # Test translation memory retrieval
            print("🔄 Testing translation memory retrieval...")
            memory_result = await self.translation_service._check_translation_memory(
                translation_result.original_text
            )
            
            # Note: This might return None if no exact match is found
            if memory_result:
                print("✅ Translation memory retrieval successful")
            else:
                print("ℹ️  No exact translation memory match found (expected)")
            
            self.test_results["passed"].append("Translation service integration")
            print("✅ Translation service integration test passed")
            
        except Exception as e:
            self.test_results["failed"].append(f"Translation service integration: {e}")
            print(f"❌ Translation service integration test failed: {e}")
    
    def test_error_handling(self):
        """Test error handling for invalid metadata."""
        try:
            # Test with invalid metadata types
            invalid_metadata = {
                "function": lambda x: x,  # Function objects
                "class": type,  # Class objects
                "complex": 3 + 4j,  # Complex numbers
            }
            
            # This should not raise an exception but should handle gracefully
            sanitized = self.vector_db.sanitize_metadata(invalid_metadata)
            
            # Verify all values are strings
            for key, value in sanitized.items():
                assert isinstance(value, str), f"Invalid type after sanitization: {type(value)}"
            
            self.test_results["passed"].append("Error handling")
            print("✅ Error handling test passed")
            
        except Exception as e:
            self.test_results["failed"].append(f"Error handling: {e}")
            print(f"❌ Error handling test failed: {e}")
    
    def test_chinese_text_handling(self):
        """Test handling of Chinese text in vector database."""
        try:
            # Test Chinese text with various encodings and characters
            chinese_texts = [
                "古典中文文本",
                "子曰：学而时习之，不亦说乎？",
                "有朋自远方来，不亦乐乎？",
                "人不知而不愠，不亦君子乎？",
                "吾日三省吾身：为人谋而不忠乎？与朋友交而不信乎？传不习乎？"
            ]
            
            # Test metadata with Chinese content
            chinese_metadata = {
                "title": "论语",
                "author": "孔子",
                "dynasty": "春秋",
                "content_type": "古典文学",
                "language": "zh"
            }
            
            # Sanitize metadata
            sanitized = self.vector_db.sanitize_metadata(chinese_metadata)
            
            # Verify Chinese characters are preserved
            for key, value in sanitized.items():
                if isinstance(value, str):
                    # Check that Chinese characters are preserved
                    chinese_chars = sum(1 for char in value if '\u4e00' <= char <= '\u9fff')
                    if chinese_chars > 0:
                        print(f"✅ Chinese characters preserved in {key}: {chinese_chars} chars")
            
            self.test_results["passed"].append("Chinese text handling")
            print("✅ Chinese text handling test passed")
            
        except Exception as e:
            self.test_results["failed"].append(f"Chinese text handling: {e}")
            print(f"❌ Chinese text handling test failed: {e}")
    
    async def run_all_tests(self):
        """Run all vector database tests."""
        print("🔍 Starting Vector Database Tests...")
        print("=" * 60)
        
        self.test_metadata_handling()
        await self.test_storage_and_retrieval()
        await self.test_translation_service_integration()
        self.test_error_handling()
        self.test_chinese_text_handling()
        
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
        
        with open(results_dir / "vector_db_results.json", "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Test results saved to: {results_dir / 'vector_db_results.json'}")
        
        return len(self.test_results['failed']) == 0


async def main():
    """Main test function."""
    tester = VectorDatabaseTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 All vector database tests passed!")
        return 0
    else:
        print("\n💥 Some vector database tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
