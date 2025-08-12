#!/usr/bin/env python3
"""
Simple PDF processing test using optimized agents.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_simple_pdf_processing():
    """Test simple PDF processing using optimized agents."""
    print("🧪 Simple PDF Processing Test")
    print("=" * 40)
    
    try:
        print("📡 Importing agents...")
        
        from src.agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
        from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
        from src.core.models import AnalysisRequest, DataType
        
        print("✅ Agents imported successfully")
        
        print("\n📡 Creating agents...")
        
        # Create agents
        file_agent = EnhancedFileExtractionAgent()
        kg_agent = KnowledgeGraphAgent()
        
        print("✅ Agents created successfully")
        
        print("\n📡 Processing PDF...")
        
        # Step 1: Extract text from PDF
        pdf_request = AnalysisRequest(
            data_type=DataType.PDF,
            content="data/Classical Chinese Sample 22208_0_8.pdf",
            language="zh"
        )
        
        print("🔧 Extracting text from PDF...")
        extraction_result = await file_agent.process(pdf_request)
        
        if extraction_result.status != "completed":
            print(f"❌ PDF extraction failed: {extraction_result.metadata.get('error', 'Unknown error')}")
            return {"success": False, "error": "PDF extraction failed"}
        
        text_content = extraction_result.extracted_text
        print(f"✅ Text extraction successful! Length: {len(text_content)} characters")
        print(f"✅ Pages processed: {len(extraction_result.pages) if extraction_result.pages else 0}")
        
        # Step 2: Process with knowledge graph
        print("🔧 Processing with knowledge graph...")
        
        kg_request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=text_content,
            language="zh"
        )
        
        kg_result = await kg_agent.process(kg_request)
        
        print("✅ Knowledge graph processing successful!")
        print(f"✅ Processing time: {kg_result.processing_time:.2f}s")
        
        # Step 3: Generate report
        print("🔧 Generating knowledge graph report...")
        
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"Results/reports/enhanced_multilingual_pdf_zh_{timestamp}"
        
        report_result = await kg_agent.generate_graph_report(
            output_path=output_path,
            target_language="zh"
        )
        
        report_files = {}
        if hasattr(report_result, 'success') and report_result.success:
            report_files = {
                "html": report_result.metadata.get('html_path', 'Unknown'),
                "png": report_result.metadata.get('png_path', 'Unknown')
            }
            print("✅ Report generated successfully!")
            print(f"✅ HTML: {report_files.get('html', 'Unknown')}")
            print(f"✅ PNG: {report_files.get('png', 'Unknown')}")
        
        # Compile final results
        result = {
            "success": True,
            "pdf_path": "data/Classical Chinese Sample 22208_0_8.pdf",
            "language": "zh",
            "text_extraction": {
                "success": True,
                "content_length": len(text_content),
                "pages_processed": len(extraction_result.pages) if extraction_result.pages else 0,
                "extraction_method": "Enhanced File Extraction Agent"
            },
            "entity_extraction": {
                "entities_found": kg_result.metadata.get('statistics', {}).get('entities_found', 0) if kg_result.metadata else 0,
                "entity_types": kg_result.metadata.get('statistics', {}).get('entity_types', {}) if kg_result.metadata else {},
                "extraction_method": "Enhanced Knowledge Graph Agent"
            },
            "knowledge_graph": {
                "nodes": kg_result.metadata.get('statistics', {}).get('nodes', 0) if kg_result.metadata else 0,
                "edges": kg_result.metadata.get('statistics', {}).get('edges', 0) if kg_result.metadata else 0,
                "communities": kg_result.metadata.get('statistics', {}).get('communities', 0) if kg_result.metadata else 0,
                "processing_time": kg_result.processing_time
            },
            "report_files": report_files
        }
        
        print("\n🎉 PDF processing completed successfully!")
        print(f"📊 Entities Found: {result['entity_extraction']['entities_found']}")
        print(f"📊 Knowledge Graph Nodes: {result['knowledge_graph']['nodes']}")
        print(f"📊 Knowledge Graph Edges: {result['knowledge_graph']['edges']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = asyncio.run(test_simple_pdf_processing())
    print(f"\n🎯 Final Result: {'SUCCESS' if result.get('success', False) else 'FAILED'}")
