#!/usr/bin/env python3
"""
Simple test for generic Chinese PDF processing.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
from config.language_config.chinese_config import ChineseConfig


async def test_generic_chinese_pdf():
    """Test generic Chinese PDF processing."""
    
    print("🧪 Testing Generic Chinese PDF Processing")
    print("=" * 50)
    
    # Initialize components
    file_agent = EnhancedFileExtractionAgent()
    chinese_config = ChineseConfig()
    
    # Test PDF file
    pdf_file = "data/Classical Chinese Sample 22208_0_8.pdf"
    
    print(f"📄 Testing with: {pdf_file}")
    
    # Test 1: Sample extraction
    print("\n🔍 Test 1: Sample extraction")
    try:
        sample_result = await file_agent.extract_text_from_pdf(
            pdf_file, 
            {"sample_only": True, "language": "zh"}
        )
        
        if sample_result["status"] == "success":
            sample_text = sample_result["extracted_text"]
            print(f"✅ Sample extraction successful")
            print(f"📝 Sample length: {len(sample_text)} characters")
            
            # Test language detection
            pdf_type = chinese_config.detect_chinese_pdf_type(sample_text)
            print(f"🔤 Detected type: {pdf_type}")
            
        else:
            print(f"❌ Sample extraction failed: {sample_result.get('error')}")
            
    except Exception as e:
        print(f"❌ Sample extraction error: {e}")
    
    # Test 2: Full extraction
    print("\n📖 Test 2: Full extraction")
    try:
        full_result = await file_agent.extract_text_from_pdf(
            pdf_file, 
            {"language": "zh", "enhanced_processing": True}
        )
        
        if full_result["status"] == "success":
            full_text = full_result["extracted_text"]
            print(f"✅ Full extraction successful")
            print(f"📝 Full text length: {len(full_text)} characters")
            print(f"⏱️ Processing time: {full_result.get('processing_time', 0):.2f}s")
            print(f"🔤 PDF type: {full_result.get('pdf_type', 'unknown')}")
            print(f"🌐 Language: {full_result.get('language', 'unknown')}")
            
            # Count Chinese characters
            chinese_chars = sum(1 for char in full_text if '\u4e00' <= char <= '\u9fff')
            print(f"📊 Chinese characters: {chinese_chars}")
            
        else:
            print(f"❌ Full extraction failed: {full_result.get('error')}")
            
    except Exception as e:
        print(f"❌ Full extraction error: {e}")
    
    print("\n✅ Generic Chinese PDF processing test completed!")


if __name__ == "__main__":
    asyncio.run(test_generic_chinese_pdf())
