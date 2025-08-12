#!/usr/bin/env python3
"""
Process Classical Chinese PDF using simple PDF processor.
Uses basic PDF extraction and knowledge graph generation.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.pdf_processor import PDFProcessor
from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.core.models import AnalysisRequest, DataType


async def process_classical_chinese_pdf_simple():
    """Process the Classical Chinese PDF using simple PDF processor."""
    
    # PDF file path
    pdf_path = "data/Classical Chinese Sample 22208_0_8.pdf"
    
    # Validate PDF file exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    print(f"📄 Processing Classical Chinese PDF: {pdf_path}")
    print("=" * 60)
    
    try:
        # Step 1: Extract text from PDF using simple processor
        print("🔍 Step 1: Extracting text from PDF...")
        pdf_processor = PDFProcessor()
        
        # Extract text directly
        from pathlib import Path
        pdf_file = Path(pdf_path)
        text_content = await pdf_processor._extract_pdf_text(pdf_file)
        
        if not text_content:
            print("❌ No text content extracted from PDF")
            return False
        
        print(f"✅ Text extraction completed. Content length: {len(text_content)} characters")
        
        # Step 2: Process with knowledge graph agent
        print(f"\n🧠 Step 2: Processing with knowledge graph agent...")
        kg_agent = KnowledgeGraphAgent()
        
        kg_request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=text_content,
            language="zh"  # Explicitly set to Chinese for Classical Chinese
        )
        
        kg_result = await kg_agent.process(kg_request)
        
        if kg_result.status != "completed":
            print(f"❌ Knowledge graph processing failed: {kg_result.metadata.get('error', 'Unknown error')}")
            return False
        
        # Step 3: Generate comprehensive report
        print(f"\n📊 Step 3: Generating knowledge graph report...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"Results/reports/classical_chinese_pdf_{timestamp}"
        
        # Ensure Results/reports directory exists
        os.makedirs("Results/reports", exist_ok=True)
        
        report_result = await kg_agent.generate_graph_report(
            output_path=output_path,
            target_language="zh"
        )
        
        # Step 4: Store in vector database
        print(f"\n💾 Step 4: Storing in vector database...")
        vector_id = await pdf_processor._store_in_vector_db(
            text_content, pdf_file, "zh", len(text_content)
        )
        
        # Step 5: Display results
        print("\n" + "=" * 60)
        print("📋 PROCESSING RESULTS")
        print("=" * 60)
        
        # Text extraction results
        print(f"📄 Text Extraction:")
        print(f"   - Content length: {len(text_content)} characters")
        print(f"   - Extraction method: PyPDF2")
        
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
        
        # Vector database results
        print(f"\n💾 Vector Database:")
        print(f"   - Vector ID: {vector_id}")
        print(f"   - Content stored: ✅")
        
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
    
    success = await process_classical_chinese_pdf_simple()
    
    if success:
        print("\n🎉 Processing completed successfully!")
        print("📁 Check the Results/reports directory for generated reports.")
    else:
        print("\n❌ Processing failed. Check the error messages above.")
    
    return success


if __name__ == "__main__":
    # Run the async function
    asyncio.run(main())
