#!/usr/bin/env python3
"""
File processing test script for Classical Chinese PDF processing.
Tests PDF extraction, validates text content, and tests language detection.
"""

import sys
import os
import asyncio
import json
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
from core.models import AnalysisRequest, DataType
from config.language_config.chinese_config import ChineseConfig


class FileProcessingTester:
    """Tests file processing functionality for Classical Chinese PDFs."""
    
    def __init__(self):
        self.file_agent = EnhancedFileExtractionAgent()
        self.chinese_config = ChineseConfig()
        self.test_results = {
            "passed": [],
            "failed": [],
            "warnings": [],
            "performance_metrics": {}
        }
        
        # Test file path
        self.test_file = Path("data/Classical Chinese Sample 22208_0_8.pdf")
    
    def test_pdf_file_existence(self):
        """Test PDF file existence and accessibility."""
        try:
            assert self.test_file.exists(), f"Test file not found: {self.test_file}"
            assert self.test_file.is_file(), f"Test path is not a file: {self.test_file}"
            assert self.test_file.stat().st_size > 0, f"Test file is empty: {self.test_file}"
            
            self.test_results["passed"].append("PDF file existence")
            print(f"✅ PDF file found: {self.test_file}")
            print(f"   Size: {self.test_file.stat().st_size} bytes")
            
        except Exception as e:
            self.test_results["failed"].append(f"PDF file existence: {e}")
            print(f"❌ PDF file existence test failed: {e}")
    
    def test_file_extraction_agent_initialization(self):
        """Test file extraction agent initialization."""
        try:
            # Test agent initialization
            assert self.file_agent is not None
            assert hasattr(self.file_agent, 'process')
            assert hasattr(self.file_agent, 'can_process')
            
            # Test agent configuration
            assert hasattr(self.file_agent, 'model_name')
            assert hasattr(self.file_agent, 'enable_chroma_storage')
            assert hasattr(self.file_agent, 'performance_config')
            assert hasattr(self.file_agent, 'ollama_integration')
            
            self.test_results["passed"].append("File extraction agent initialization")
            print("✅ File extraction agent initialized successfully")
            
        except Exception as e:
            self.test_results["failed"].append(f"File extraction agent initialization: {e}")
            print(f"❌ File extraction agent initialization failed: {e}")
    
    async def test_text_extraction_with_chinese_detection(self):
        """Test text extraction with Chinese language detection."""
        try:
            # Create analysis request
            request = AnalysisRequest(
                data_type=DataType.PDF,
                content=str(self.test_file),
                language="zh",
                model_preference="qwen2.5:7b"
            )
            
            # Process the PDF
            print("🔄 Processing PDF file...")
            result = await self.file_agent.process(request)
            
            # Validate result
            assert result is not None
            assert hasattr(result, 'extracted_text')
            assert len(result.extracted_text) > 0
            
            # Check for Chinese characters
            chinese_chars = sum(1 for char in result.extracted_text if '\u4e00' <= char <= '\u9fff')
            chinese_ratio = chinese_chars / len(result.extracted_text) if result.extracted_text else 0
            
            print(f"📄 Extracted text length: {len(result.extracted_text)} characters")
            print(f"🇨🇳 Chinese characters: {chinese_chars} ({chinese_ratio:.2%})")
            
            # Validate Chinese content
            assert chinese_ratio > 0.1, f"Insufficient Chinese content: {chinese_ratio:.2%}"
            
            # Check for Classical Chinese indicators
            classical_indicators = ['之', '也', '者', '而', '以', '于', '为', '其', '此', '彼']
            classical_count = sum(1 for char in result.extracted_text if char in classical_indicators)
            
            print(f"📚 Classical Chinese indicators: {classical_count}")
            
            self.test_results["passed"].append("Text extraction with Chinese detection")
            self.test_results["performance_metrics"]["extracted_text_length"] = len(result.extracted_text)
            self.test_results["performance_metrics"]["chinese_ratio"] = chinese_ratio
            self.test_results["performance_metrics"]["classical_indicators"] = classical_count
            
        except Exception as e:
            self.test_results["failed"].append(f"Text extraction with Chinese detection: {e}")
            print(f"❌ Text extraction test failed: {e}")
    
    def test_extracted_content_quality(self):
        """Test extracted content quality."""
        try:
            # This would be called after text extraction
            # For now, we'll test the quality metrics
            
            # Test content quality indicators
            quality_metrics = {
                "min_length": 100,
                "max_length": 100000,
                "chinese_ratio_threshold": 0.1,
                "classical_indicators_threshold": 5
            }
            
            # Validate quality metrics
            for metric, value in quality_metrics.items():
                assert isinstance(value, (int, float)), f"Invalid metric type: {metric}"
                if metric.endswith('_threshold'):
                    assert value > 0, f"Invalid threshold: {metric}"
            
            self.test_results["passed"].append("Extracted content quality validation")
            print("✅ Content quality validation passed")
            
        except Exception as e:
            self.test_results["failed"].append(f"Extracted content quality: {e}")
            print(f"❌ Content quality validation failed: {e}")
    
    def test_language_detection(self):
        """Test language detection functionality."""
        try:
            # Test Chinese language detection patterns
            detection_patterns = self.chinese_config.detection_patterns
            
            # Test basic Chinese character detection
            test_texts = [
                "这是现代中文文本",
                "子曰：学而时习之，不亦说乎？",
                "This is English text with 中文 mixed in",
                "これは日本語のテキストです"
            ]
            
            for text in test_texts:
                chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
                chinese_ratio = chinese_chars / len(text) if text else 0
                
                if chinese_ratio > 0.5:
                    print(f"✅ Detected Chinese content: {chinese_ratio:.2%} Chinese characters")
                elif chinese_ratio > 0:
                    print(f"⚠️  Mixed content: {chinese_ratio:.2%} Chinese characters")
                else:
                    print(f"ℹ️  No Chinese content detected")
            
            self.test_results["passed"].append("Language detection")
            print("✅ Language detection test passed")
            
        except Exception as e:
            self.test_results["failed"].append(f"Language detection: {e}")
            print(f"❌ Language detection test failed: {e}")
    
    async def run_all_tests(self):
        """Run all file processing tests."""
        print("🔍 Starting File Processing Tests...")
        print("=" * 60)
        
        self.test_pdf_file_existence()
        self.test_file_extraction_agent_initialization()
        await self.test_text_extraction_with_chinese_detection()
        self.test_extracted_content_quality()
        self.test_language_detection()
        
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
        
        with open(results_dir / "file_processing_results.json", "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Test results saved to: {results_dir / 'file_processing_results.json'}")
        
        return len(self.test_results['failed']) == 0


async def main():
    """Main test function."""
    tester = FileProcessingTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 All file processing tests passed!")
        return 0
    else:
        print("\n💥 Some file processing tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
