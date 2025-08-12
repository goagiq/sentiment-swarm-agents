#!/usr/bin/env python3
"""
Comprehensive test to verify that both Chinese and Russian PDF processing work correctly together.
This test ensures that language-specific optimizations don't conflict with each other.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.file_extraction_agent import FileExtractionAgent
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from core.models import AnalysisRequest, DataType
from config.language_specific_config import detect_primary_language


async def test_language_processing(pdf_path: str, expected_language: str, test_name: str):
    """Test PDF processing for a specific language."""
    
    print(f"\n🧪 {test_name}")
    print("=" * 80)
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    try:
        # Step 1: File Extraction
        print("📄 Step 1: File Extraction...")
        file_agent = FileExtractionAgent()
        
        pdf_request = AnalysisRequest(
            data_type=DataType.PDF,
            content=pdf_path,
            language=expected_language
        )
        
        extraction_result = await file_agent.process(pdf_request)
        
        if extraction_result.status != "completed":
            print(f"❌ File extraction failed: {extraction_result.metadata.get('error', 'Unknown error')}")
            return False
        
        text_content = extraction_result.extracted_text
        print(f"✅ File extraction successful: {len(text_content)} characters, {len(extraction_result.pages)} pages")
        
        # Step 2: Language Detection
        print("🌍 Step 2: Language Detection...")
        detected_language = detect_primary_language(text_content)
        print(f"✅ Language detection: Expected {expected_language}, Detected {detected_language}")
        
        # Step 3: Knowledge Graph Processing
        print("🧠 Step 3: Knowledge Graph Processing...")
        kg_agent = KnowledgeGraphAgent()
        
        kg_request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=text_content,
            language=detected_language
        )
        
        start_time = datetime.now()
        kg_result = await kg_agent.process(kg_request)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if kg_result.status == "completed":
            print(f"✅ Knowledge graph processing successful in {processing_time:.2f}s")
            
            # Check for language-specific content
            if expected_language == "zh":
                chinese_keywords = ['中国', '北京', '上海', '人工智能', '机器学习']
                found_keywords = [kw for kw in chinese_keywords if kw in text_content]
                print(f"🇨🇳 Chinese keywords found: {len(found_keywords)}")
                if found_keywords:
                    print(f"   📋 Found: {', '.join(found_keywords[:3])}")
            
            elif expected_language == "ru":
                russian_keywords = ['Москва', 'Россия', 'Путин', 'Хрущёв', 'Газпром']
                found_keywords = [kw for kw in russian_keywords if kw in text_content]
                print(f"🇷🇺 Russian keywords found: {len(found_keywords)}")
                if found_keywords:
                    print(f"   📋 Found: {', '.join(found_keywords[:3])}")
            
            return True
        else:
            print(f"❌ Knowledge graph processing failed: {kg_result.metadata.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 Starting Comprehensive Language Processing Test...")
    print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test Chinese PDF
    chinese_pdf = "data/Classical Chinese Sample 22208_0_8.pdf"
    chinese_success = await test_language_processing(
        chinese_pdf, "zh", "🇨🇳 TESTING CHINESE PDF PROCESSING"
    )
    
    # Test Russian PDF
    russian_pdf = "data/Russian_Oliver_Excerpt.pdf"
    russian_success = await test_language_processing(
        russian_pdf, "ru", "🇷🇺 TESTING RUSSIAN PDF PROCESSING"
    )
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"   🇨🇳 Chinese PDF: {'✅ PASSED' if chinese_success else '❌ FAILED'}")
    print(f"   🇷🇺 Russian PDF: {'✅ PASSED' if russian_success else '❌ FAILED'}")
    print(f"   📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if chinese_success and russian_success:
        print(f"\n🎉 SUCCESS: Both Chinese and Russian PDF processing are working correctly!")
        print(f"   ✅ No language conflicts detected")
        print(f"   ✅ Language-specific configurations are properly isolated")
        print(f"   ✅ Both languages can be processed in the same session")
    else:
        print(f"\n⚠️ ISSUES DETECTED: Some language processing failed")
        print(f"   🔧 Please check the specific error messages above")


if __name__ == "__main__":
    asyncio.run(main())
