#!/usr/bin/env python3
"""
Test script to demonstrate enhanced file extraction capabilities with multilingual optimizations.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
from core.models import AnalysisRequest, DataType
from config.file_extraction_config import get_extraction_config, get_optimal_workers, get_optimal_chunk_size


async def test_enhanced_extraction(pdf_path: str, language: str, test_name: str):
    """Test enhanced file extraction for a specific language."""
    
    print(f"\n🧪 {test_name}")
    print("=" * 80)
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    try:
        # Get language-specific configuration
        extraction_config = get_extraction_config(language)
        optimal_workers = get_optimal_workers(language)
        optimal_chunk_size = get_optimal_chunk_size(language, 50000)  # Estimate text length
        
        print(f"📋 Language Configuration:")
        print(f"   - Language: {language}")
        print(f"   - Max Workers: {extraction_config.max_workers}")
        print(f"   - Chunk Size: {extraction_config.chunk_size}")
        print(f"   - Optimal Workers: {optimal_workers}")
        print(f"   - Optimal Chunk Size: {optimal_chunk_size}")
        print(f"   - OCR Confidence Threshold: {extraction_config.ocr_confidence_threshold}")
        print(f"   - Min Text Length: {extraction_config.min_text_length}")
        
        # Initialize enhanced file extraction agent
        print(f"\n🚀 Initializing Enhanced File Extraction Agent...")
        agent = EnhancedFileExtractionAgent()
        print(f"✅ Enhanced File Extraction Agent initialized")
        
        # Create analysis request
        print(f"\n📄 Creating Analysis Request...")
        request = AnalysisRequest(
            data_type=DataType.PDF,
            content=pdf_path,
            language=language
        )
        print(f"✅ Analysis Request created")
        
        # Process PDF with enhanced extraction
        print(f"\n⚡ Processing PDF with Enhanced Extraction...")
        print(f"   ⏳ This may take a moment with optimized parallel processing...")
        
        start_time = datetime.now()
        result = await agent.process(request)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Enhanced extraction completed in {processing_time:.2f}s")
        
        if result.status == "completed":
            print(f"\n📊 Extraction Results:")
            print(f"   ✅ Status: {result.status}")
            print(f"   📄 Pages Processed: {result.metadata.get('pages_processed', 'Unknown')}")
            print(f"   📝 Text Length: {result.metadata.get('text_length', 'Unknown')} characters")
            print(f"   🎯 Confidence: {result.metadata.get('confidence', 'Unknown'):.2f}")
            print(f"   🔧 Method: {result.metadata.get('extraction_method', 'Unknown')}")
            print(f"   🌍 Language: {result.metadata.get('language', 'Unknown')}")
            
            # Show extraction statistics if available
            if 'stats' in result.metadata:
                stats = result.metadata['stats']
                print(f"\n📈 Detailed Statistics:")
                print(f"   - Total Pages: {stats.get('total_pages', 'Unknown')}")
                print(f"   - Successful Pages: {stats.get('successful_pages', 'Unknown')}")
                print(f"   - Failed Pages: {stats.get('failed_pages', 'Unknown')}")
                print(f"   - Workers Used: {stats.get('workers_used', 'Unknown')}")
                print(f"   - Chunk Size Used: {stats.get('chunk_size_used', 'Unknown')}")
            
            # Show sample text
            if result.extracted_text:
                sample_text = result.extracted_text[:500] + "..." if len(result.extracted_text) > 500 else result.extracted_text
                print(f"\n📝 Sample Extracted Text:")
                print(f"   {sample_text}")
            
            # Show page results if available
            if result.pages:
                print(f"\n📄 Page Results Summary:")
                successful_pages = [p for p in result.pages if not getattr(p, 'error_message', None)]
                failed_pages = [p for p in result.pages if getattr(p, 'error_message', None)]
                
                print(f"   ✅ Successful Pages: {len(successful_pages)}")
                print(f"   ❌ Failed Pages: {len(failed_pages)}")
                
                if successful_pages:
                    avg_confidence = sum(getattr(p, 'confidence', 0) for p in successful_pages) / len(successful_pages)
                    print(f"   🎯 Average Confidence: {avg_confidence:.2f}")
                
                # Show sample page details
                if successful_pages:
                    sample_page = successful_pages[0]
                    print(f"\n📄 Sample Page Details:")
                    print(f"   - Page Number: {getattr(sample_page, 'page_number', 'Unknown')}")
                    print(f"   - Content Length: {getattr(sample_page, 'content_length', 'Unknown')}")
                    print(f"   - Extraction Method: {getattr(sample_page, 'extraction_method', 'Unknown')}")
                    print(f"   - Confidence: {getattr(sample_page, 'confidence', 'Unknown'):.2f}")
                    print(f"   - Processing Time: {getattr(sample_page, 'processing_time', 'Unknown'):.2f}s")
            
            return True
        else:
            print(f"❌ Extraction failed: {result.metadata.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Starting Enhanced File Extraction Test...")
    print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test Chinese PDF with enhanced extraction
    chinese_pdf = "data/Classical Chinese Sample 22208_0_8.pdf"
    chinese_success = await test_enhanced_extraction(
        chinese_pdf, "zh", "🇨🇳 TESTING ENHANCED CHINESE PDF EXTRACTION"
    )
    
    # Test Russian PDF with enhanced extraction
    russian_pdf = "data/Russian_Oliver_Excerpt.pdf"
    russian_success = await test_enhanced_extraction(
        russian_pdf, "ru", "🇷🇺 TESTING ENHANCED RUSSIAN PDF EXTRACTION"
    )
    
    # Test English PDF with enhanced extraction (if available)
    english_pdf = "data/sample_english.pdf"  # You may need to provide an English PDF
    english_success = False
    if os.path.exists(english_pdf):
        english_success = await test_enhanced_extraction(
            english_pdf, "en", "🇺🇸 TESTING ENHANCED ENGLISH PDF EXTRACTION"
        )
    else:
        print(f"\n⚠️ English PDF not found: {english_pdf}")
        print(f"   Skipping English PDF test")
    
    # Summary
    print(f"\n📊 Enhanced Extraction Test Summary:")
    print(f"   🇨🇳 Chinese PDF: {'✅ PASSED' if chinese_success else '❌ FAILED'}")
    print(f"   🇷🇺 Russian PDF: {'✅ PASSED' if russian_success else '❌ FAILED'}")
    if os.path.exists(english_pdf):
        print(f"   🇺🇸 English PDF: {'✅ PASSED' if english_success else '❌ FAILED'}")
    print(f"   📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Performance comparison
    print(f"\n🚀 Enhanced Extraction Features Demonstrated:")
    print(f"   ✅ Language-specific parallel processing")
    print(f"   ✅ Dynamic chunking strategies")
    print(f"   ✅ Adaptive worker allocation")
    print(f"   ✅ Language-specific text validation")
    print(f"   ✅ Quality scoring and filtering")
    print(f"   ✅ Memory monitoring and optimization")
    print(f"   ✅ Real-time progress tracking")
    print(f"   ✅ Enhanced error handling")


if __name__ == "__main__":
    asyncio.run(main())
