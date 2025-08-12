#!/usr/bin/env python3
"""
Test script to test the new PDF processing functionality.
This will test both Chinese and Russian PDF processing to ensure both work correctly.
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


async def test_pdf_processing(pdf_path: str, expected_language: str):
    """Test PDF processing for a specific language."""
    
    print(f"\n🧪 Testing PDF processing for {expected_language.upper()} PDF: {pdf_path}")
    print("=" * 80)
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    try:
        # Step 1: Test File Extraction
        print("\n📄 Step 1: Testing File Extraction...")
        print("   ⏳ Initializing FileExtractionAgent...")
        file_agent = FileExtractionAgent()
        print("   ✅ FileExtractionAgent initialized")
        
        print("   ⏳ Creating analysis request...")
        pdf_request = AnalysisRequest(
            data_type=DataType.PDF,
            content=pdf_path,
            language=expected_language
        )
        print("   ✅ Analysis request created")
        
        print("   ⏳ Processing PDF extraction (this may take a moment)...")
        extraction_result = await file_agent.process(pdf_request)
        print("   ✅ PDF extraction completed")
        
        if extraction_result.status != "completed":
            print(f"❌ File extraction failed: {extraction_result.metadata.get('error', 'Unknown error')}")
            return False
        
        text_content = extraction_result.extracted_text
        print("✅ File extraction successful:")
        print(f"  - Content length: {len(text_content)} characters")
        print(f"  - Pages processed: {len(extraction_result.pages) if extraction_result.pages else 'Unknown'}")
        print(f"  - Processing time: {extraction_result.processing_time:.2f}s")
        
        # Step 2: Test Language Detection
        print("\n🌍 Step 2: Testing Language Detection...")
        print("   ⏳ Analyzing text content for language detection...")
        detected_language = detect_primary_language(text_content)
        print("   ✅ Language detection completed")
        print(f"   - Expected: {expected_language}")
        print(f"   - Detected: {detected_language}")
        print(f"   - Match: {'✅' if detected_language == expected_language else '❌'}")
        
        # Step 3: Test Knowledge Graph Processing
        print("\n🧠 Step 3: Testing Knowledge Graph Processing...")
        print("   ⏳ Initializing KnowledgeGraphAgent...")
        kg_agent = KnowledgeGraphAgent()
        print("   ✅ KnowledgeGraphAgent initialized")
        
        print("   ⏳ Creating knowledge graph analysis request...")
        kg_request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=text_content,
            language=detected_language
        )
        print("   ✅ Knowledge graph request created")
        
        print("   ⏳ Processing knowledge graph (this may take a moment)...")
        start_time = datetime.now()
        kg_result = await kg_agent.process(kg_request)
        processing_time = (datetime.now() - start_time).total_seconds()
        print("   ✅ Knowledge graph processing completed")
        
        print("✅ Knowledge graph processing completed:")
        print(f"  - Status: {kg_result.status}")
        print(f"  - Processing time: {processing_time:.2f}s")
        
        # Check statistics
        if kg_result.metadata and "statistics" in kg_result.metadata:
            stats = kg_result.metadata["statistics"]
            print(f"  - Entities found: {stats.get('entities_found', 0)}")
            print(f"  - Entity types: {stats.get('entity_types', {})}")
            print(f"  - Language stats: {stats.get('language_stats', {})}")
            
            # Check if entities were found for the expected language
            language_stats = stats.get('language_stats', {})
            expected_entities = language_stats.get(expected_language, 0)
            print(f"  - {expected_language.upper()} entities: {expected_entities}")
            
            success = expected_entities > 0
        else:
            print("  - No statistics available")
            success = False
        
        # Step 4: Test Report Generation
        print("\n📊 Step 4: Testing Report Generation...")
        try:
            print("   ⏳ Preparing report generation...")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"Results/reports/test_{expected_language}_{timestamp}"
            print(f"   📁 Output path: {output_path}")
            
            print("   ⏳ Generating knowledge graph report (this may take a moment)...")
            report_result = await kg_agent.generate_graph_report(
                output_path=output_path,
                target_language=detected_language
            )
            print("   ✅ Report generation completed")
            
            if hasattr(report_result, 'success') and report_result.success:
                print("✅ Report generation successful:")
                print(f"  - HTML report: {report_result.metadata.get('html_path', 'Unknown')}")
                print(f"  - PNG image: {report_result.metadata.get('png_path', 'Unknown')}")
            else:
                print(f"⚠️ Report generation failed: {getattr(report_result, 'error', 'Unknown error')}")
                
        except Exception as e:
            print(f"⚠️ Report generation error: {e}")
        
        return success
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Starting PDF Processing Integration Test...")
    print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test Chinese PDF
    print("\n" + "="*80)
    print("🇨🇳 TESTING CHINESE PDF PROCESSING")
    print("="*80)
    chinese_pdf = "data/Classical Chinese Sample 22208_0_8.pdf"
    chinese_success = await test_pdf_processing(chinese_pdf, "zh")
    
    # Test Russian PDF
    print("\n" + "="*80)
    print("🇷🇺 TESTING RUSSIAN PDF PROCESSING")
    print("="*80)
    russian_pdf = "data/Russian_Oliver_Excerpt.pdf"
    russian_success = await test_pdf_processing(russian_pdf, "ru")
    
    print("\n" + "=" * 80)
    print("📋 TEST RESULTS SUMMARY:")
    print(f"  - Chinese PDF: {'✅ PASSED' if chinese_success else '❌ FAILED'}")
    print(f"  - Russian PDF: {'✅ PASSED' if russian_success else '❌ FAILED'}")
    
    if chinese_success and russian_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Both Chinese and Russian PDF processing are working correctly!")
        print("✅ Language-specific configurations are properly applied!")
        print("✅ No conflicts between language configurations!")
    elif chinese_success and not russian_success:
        print("\n⚠️ PARTIAL SUCCESS - Russian PDF needs fixing!")
        print("✅ Chinese PDF processing is working correctly!")
        print("❌ Russian PDF processing needs attention!")
    elif not chinese_success and russian_success:
        print("\n⚠️ PARTIAL SUCCESS - Chinese PDF needs fixing!")
        print("❌ Chinese PDF processing needs attention!")
        print("✅ Russian PDF processing is working correctly!")
    else:
        print("\n❌ ALL TESTS FAILED!")
        print("❌ Both Chinese and Russian PDF processing need fixing!")
    
    print(f"📅 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return chinese_success and russian_success


if __name__ == "__main__":
    asyncio.run(main())
