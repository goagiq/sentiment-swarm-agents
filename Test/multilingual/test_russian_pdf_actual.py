#!/usr/bin/env python3
"""
Test script to process the actual Russian PDF with fixed Russian entity extraction.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.file_extraction_agent import FileExtractionAgent
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from core.models import AnalysisRequest, DataType


async def main():
    """Main function to process Russian PDF with fixed Russian language support."""
    print("🚀 Starting Russian PDF Processing with Fixed Entity Extraction")
    print("=" * 60)
    
    # Initialize agents
    file_agent = FileExtractionAgent()
    kg_agent = KnowledgeGraphAgent()
    
    pdf_path = "data/Russian_Oliver_Excerpt.pdf"
    
    # Step 1: Extract text from PDF
    print(f"📄 Step 1: Extracting text from {pdf_path}")
    
    request = AnalysisRequest(
        data_type=DataType.PDF,
        content=pdf_path,
        request_id="russian_pdf_test_fixed"
    )
    
    try:
        extraction_result = await file_agent.process(request)
        
        if extraction_result.status == "completed":
            extracted_text = extraction_result.extracted_text
            print("✅ Text extraction successful!")
            print(f"   📊 Extracted {len(extracted_text)} characters")
            print(f"   📝 First 200 chars: {extracted_text[:200]}...")
            
            # Step 2: Process through knowledge graph with Russian language
            print("\n🧠 Step 2: Processing through knowledge graph with Russian language...")
            
            # Create request with Russian language explicitly set
            kg_request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=extracted_text,
                language="ru",  # Explicitly set Russian language
                request_id="russian_pdf_kg_fixed"
            )
            
            kg_result = await kg_agent.process(kg_request)
            
            if kg_result.status == "completed":
                print("✅ Knowledge graph processing successful!")
                
                # Generate report with Russian language
                print("\n📊 Step 3: Generating knowledge graph report with Russian...")
                report_path = "Results/russian_oliver_knowledge_graph_report_fixed.html"
                
                # Use the agent's report generation method with Russian language
                report_result = await kg_agent.generate_graph_report(
                    output_path=report_path,
                    target_language="ru"
                )
                
                if report_result.get("success", False):
                    print(f"✅ Knowledge graph report generated: {report_path}")
                    entities = report_result.get('entities_extracted', 0)
                    relationships = report_result.get('relationships_mapped', 0)
                    nodes = report_result.get('graph_nodes', 0)
                    edges = report_result.get('graph_edges', 0)
                    print(f"   📋 Entities extracted: {entities}")
                    print(f"   🔗 Relationships mapped: {relationships}")
                    print(f"   🎯 Graph nodes: {nodes}")
                    print(f"   🔗 Graph edges: {edges}")
                    
                    print("\n🎉 Russian PDF processing completed successfully!")
                    print("✅ Russian language support is working correctly")
                else:
                    print("❌ Failed to generate knowledge graph report")
                    error_msg = report_result.get('error', 'Unknown error')
                    print(f"   Error: {error_msg}")
            else:
                print("❌ Knowledge graph processing failed")
                error_msg = kg_result.metadata.get('error', 'Unknown error')
                print(f"   Error: {error_msg}")
        else:
            print("❌ Text extraction failed")
            error_msg = extraction_result.metadata.get('error', 'Unknown error')
            print(f"   Error: {error_msg}")
            
    except Exception as e:
        print(f"❌ Processing failed with exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
