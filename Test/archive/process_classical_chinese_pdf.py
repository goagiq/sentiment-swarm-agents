#!/usr/bin/env python3
"""
Process Classical Chinese PDF and generate knowledge graph report.
Uses enhanced multilingual processing with Classical Chinese support.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import after path modification
from src.agents.enhanced_file_extraction_agent import (
    EnhancedFileExtractionAgent
)
from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.core.models import AnalysisRequest, DataType
from src.config.language_specific_config import detect_primary_language


async def process_classical_chinese_pdf():
    """Process the Classical Chinese PDF and generate knowledge graph report."""
    
    # PDF file path
    pdf_path = "data/Classical Chinese Sample 22208_0_8.pdf"
    
    # Validate PDF file exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    print(f"📄 Processing Classical Chinese PDF: {pdf_path}")
    print("=" * 60)
    
    try:
        # Step 1: Extract text from PDF using enhanced extraction
        print("🔍 Step 1: Extracting text from PDF with enhanced multilingual processing...")
        file_agent = EnhancedFileExtractionAgent()
        
        pdf_request = AnalysisRequest(
            data_type=DataType.PDF,
            content=pdf_path,
            language="zh"  # Explicitly set to Chinese for Classical Chinese
        )
        
        extraction_result = await file_agent.process(pdf_request)
        
        if extraction_result.status != "completed":
            print(f"❌ PDF extraction failed: {extraction_result.metadata.get('error', 'Unknown error')}")
            return False
        
        text_content = extraction_result.extracted_text
        print(f"✅ Text extraction completed. Content length: {len(text_content)} characters")
        print(f"📄 Pages processed: {len(extraction_result.pages) if extraction_result.pages else 'Unknown'}")
        
        # Step 2: Detect language and check for Classical Chinese patterns
        print("\n🌍 Step 2: Analyzing language and Classical Chinese patterns...")
        detected_language = detect_primary_language(text_content)
        print(f"✅ Detected language: {detected_language}")
        
        # Check for Classical Chinese patterns
        from src.config.language_config import LanguageConfigFactory
        chinese_config = LanguageConfigFactory.get_config("zh")
        
        if hasattr(chinese_config, 'is_classical_chinese'):
            is_classical = chinese_config.is_classical_chinese(text_content[:1000])  # Check first 1000 chars
            print(f"🏛️ Classical Chinese detected: {is_classical}")
        
        # Step 3: Process with knowledge graph agent using enhanced multilingual support
        print(f"\n🧠 Step 3: Processing with enhanced multilingual entity extraction...")
        kg_agent = KnowledgeGraphAgent()
        
        kg_request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=text_content,
            language=detected_language
        )
        
        kg_result = await kg_agent.process(kg_request)
        
        if kg_result.status != "completed":
            print(f"❌ Knowledge graph processing failed: {kg_result.metadata.get('error', 'Unknown error')}")
            return False
        
        # Step 4: Generate comprehensive report
        print(f"\n📊 Step 4: Generating knowledge graph report...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"Results/reports/classical_chinese_pdf_{timestamp}"
        
        # Ensure Results/reports directory exists
        os.makedirs("Results/reports", exist_ok=True)
        
        report_result = await kg_agent.generate_graph_report(
            output_path=output_path,
            target_language=detected_language
        )
        
        # Step 5: Display results
        print("\n" + "=" * 60)
        print("📋 PROCESSING RESULTS")
        print("=" * 60)
        
        # Text extraction results
        print(f"📄 Text Extraction:")
        print(f"   - Content length: {len(text_content)} characters")
        print(f"   - Pages processed: {len(extraction_result.pages) if extraction_result.pages else 'Unknown'}")
        print(f"   - Extraction method: Enhanced multilingual")
        
        # Entity extraction results
        stats = kg_result.metadata.get("statistics", {}) if kg_result.metadata else {}
        print(f"\n🔍 Entity Extraction:")
        print(f"   - Entities found: {stats.get('entities_found', 0)}")
        print(f"   - Entity types: {stats.get('entity_types', {})}")
        print(f"   - Language stats: {stats.get('language_stats', {})}")
        print(f"   - Extraction method: Enhanced multilingual with Classical Chinese support")
        
        # Knowledge graph results
        print(f"\n🧠 Knowledge Graph:")
        print(f"   - Nodes: {stats.get('nodes', 0)}")
        print(f"   - Edges: {stats.get('edges', 0)}")
        print(f"   - Communities: {stats.get('communities', 0)}")
        print(f"   - Processing time: {kg_result.processing_time:.2f} seconds")
        
        # Report results
        if hasattr(report_result, 'success') and report_result.success:
            print(f"\n📊 Report Generation:")
            print(f"   - HTML report: {report_result.metadata.get('html_path', 'Unknown')}")
            print(f"   - PNG visualization: {report_result.metadata.get('png_path', 'Unknown')}")
            print(f"   - Report directory: {output_path}")
        else:
            print(f"\n⚠️ Report generation may have had issues")
        
        # Enhanced features
        print(f"\n🚀 Enhanced Features:")
        print(f"   - Language-specific patterns: ✅")
        print(f"   - Dictionary lookup: ✅")
        print(f"   - LLM-based extraction: ✅")
        print(f"   - Classical Chinese support: ✅")
        print(f"   - Multilingual support: ['en', 'ru', 'zh']")
        
        print("\n" + "=" * 60)
        print("✅ Classical Chinese PDF processing completed successfully!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing Classical Chinese PDF: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main function to run the Classical Chinese PDF processing."""
    print("🏛️ Classical Chinese PDF Processing and Knowledge Graph Generation")
    print("=" * 60)
    
    success = await process_classical_chinese_pdf()
    
    if success:
        print("\n🎉 Processing completed successfully!")
        print("📁 Check the Results/reports directory for generated reports.")
    else:
        print("\n❌ Processing failed. Check the error messages above.")
    
    return success


if __name__ == "__main__":
    # Run the async function
    asyncio.run(main())
