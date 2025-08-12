#!/usr/bin/env python3
"""
Test script to establish baseline for Classical Chinese PDF processing.
This will help us understand the current working state before making changes.
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


async def test_classical_chinese_baseline():
    """Test Classical Chinese PDF processing to establish baseline."""
    
    print("🧪 Testing Classical Chinese PDF Baseline...")
    print("=" * 60)
    
    pdf_path = "data/Classical Chinese Sample 22208_0_8.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    try:
        # Step 1: Test File Extraction
        print("\n📄 Step 1: Testing File Extraction...")
        file_agent = FileExtractionAgent()
        
        # Create analysis request for PDF processing
        pdf_request = AnalysisRequest(
            data_type=DataType.PDF,
            content=pdf_path,
            language="zh"
        )
        
        extraction_result = await file_agent.process(pdf_request)
        
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
        detected_language = detect_primary_language(text_content)
        print(f"✅ Language detection result: {detected_language}")
        
        # Step 3: Test Knowledge Graph Processing
        print("\n🧠 Step 3: Testing Knowledge Graph Processing...")
        kg_agent = KnowledgeGraphAgent()
        
        # Create analysis request
        request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=text_content,
            language=detected_language
        )
        
        # Process with knowledge graph agent
        start_time = datetime.now()
        kg_result = await kg_agent.process(request)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print("✅ Knowledge graph processing completed:")
        print(f"  - Status: {kg_result.status}")
        print(f"  - Processing time: {processing_time:.2f}s")
        
        # Check statistics
        if kg_result.metadata and "statistics" in kg_result.metadata:
            stats = kg_result.metadata["statistics"]
            print(f"  - Entities found: {stats.get('entities_found', 0)}")
            print(f"  - Entity types: {stats.get('entity_types', {})}")
            print(f"  - Language stats: {stats.get('language_stats', {})}")
            
            # Check if Chinese entities were found
            language_stats = stats.get('language_stats', {})
            chinese_entities = language_stats.get('zh', 0)
            print(f"  - Chinese entities: {chinese_entities}")
            
            success = chinese_entities > 0
        else:
            print("  - No statistics available")
            success = False
        
        # Step 4: Test Report Generation
        print("\n📊 Step 4: Testing Report Generation...")
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"Results/reports/classical_chinese_baseline_{timestamp}"
            
            report_result = await kg_agent.generate_graph_report(
                output_path=output_path,
                target_language=detected_language
            )
            
            if report_result.success:
                print("✅ Report generation successful:")
                print(f"  - HTML report: {report_result.metadata.get('html_path', 'Unknown')}")
                print(f"  - PNG image: {report_result.metadata.get('png_path', 'Unknown')}")
            else:
                print(f"⚠️ Report generation failed: {report_result.error}")
                
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
    print("🚀 Starting Classical Chinese PDF Baseline Test...")
    print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    success = await test_classical_chinese_baseline()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Classical Chinese PDF Baseline Test PASSED!")
        print("🎉 Classical Chinese processing is working correctly!")
        print("📋 Baseline established - ready for enhancement work")
    else:
        print("❌ Classical Chinese PDF Baseline Test FAILED!")
        print("🔧 Classical Chinese processing needs fixing!")
        print("⚠️ Need to fix Classical Chinese processing before proceeding")
    
    print(f"📅 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())
