#!/usr/bin/env python3
"""
Script to regenerate Russian PDF report with fixed entity extraction.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.file_extraction_agent import FileExtractionAgent
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from core.models import AnalysisRequest, DataType


async def regenerate_russian_report():
    """Regenerate the Russian PDF report with fixed entity extraction."""
    
    pdf_path = "data/Russian_Oliver_Excerpt.pdf"
    output_path = "Results/reports/russian_oliver_knowledge_graph_report_fixed_v2"
    
    print(f"🔄 Regenerating Russian PDF report...")
    print(f"📄 PDF: {pdf_path}")
    print(f"📊 Output: {output_path}")
    
    try:
        # Initialize agents
        file_agent = FileExtractionAgent()
        kg_agent = KnowledgeGraphAgent()
        
        # Extract text from PDF
        print("\n📄 Extracting text from PDF...")
        pdf_request = AnalysisRequest(
            data_type=DataType.PDF,
            content=pdf_path,
            language="ru"
        )
        
        extraction_result = await file_agent.process(pdf_request)
        
        # Get the extracted text
        text_content = extraction_result.extracted_text or extraction_result.raw_content
        
        if not text_content:
            print("❌ Failed to extract text from PDF")
            return False
        
        print(f"✅ Text extraction successful, content length: {len(text_content)}")
        
        # Process with knowledge graph agent
        print("\n🧠 Processing with knowledge graph agent...")
        kg_request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=text_content,
            language="ru"
        )
        
        kg_result = await kg_agent.process(kg_request)
        
        print(f"\n✅ Knowledge graph processing result:")
        print(f"  - Status: {kg_result.status}")
        print(f"  - Processing time: {kg_result.processing_time:.2f}s")
        
        # Check statistics
        if kg_result.metadata and "statistics" in kg_result.metadata:
            stats = kg_result.metadata["statistics"]
            print(f"  - Entities found: {stats.get('entities_found', 0)}")
            print(f"  - Entity types: {stats.get('entity_types', {})}")
            print(f"  - Language stats: {stats.get('language_stats', {})}")
            
            # Check if Russian entities were found
            language_stats = stats.get('language_stats', {})
            russian_entities = language_stats.get('ru', 0)
            print(f"  - Russian entities: {russian_entities}")
        
        # Generate report
        print(f"\n📊 Generating knowledge graph report...")
        report_result = await kg_agent.generate_graph_report(
            output_path=output_path,
            target_language="ru"
        )
        
        print(f"\n✅ Report generation result:")
        print(f"  - Files: {report_result.get('files', {})}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main function."""
    print("🚀 Starting Russian PDF report regeneration...")
    
    success = await regenerate_russian_report()
    
    if success:
        print("\n✅ Russian PDF report regeneration completed!")
        print("🎉 The report should now show Russian entities!")
    else:
        print("\n❌ Russian PDF report regeneration failed!")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())
