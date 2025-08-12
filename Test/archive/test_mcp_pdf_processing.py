#!/usr/bin/env python3
"""
Test script to process Classical Chinese PDF using MCP server.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_mcp_pdf_processing():
    """Test PDF processing using MCP server."""
    print("🔧 Testing MCP PDF Processing")
    print("=" * 40)
    
    try:
        # Import the function from main.py
        from main import process_classical_chinese_pdf_simple
        
        # Test PDF path
        pdf_path = "data/Classical Chinese Sample 22208_0_8.pdf"
        
        print(f"📄 Processing PDF: {pdf_path}")
        print("🔧 Using optimized agents directly...")
        
        # Process the PDF
        result = await process_classical_chinese_pdf_simple(
            pdf_path=pdf_path,
            language="zh",
            generate_report=True
        )
        
        if result.get("success", False):
            print("\n🎉 PDF processing completed successfully!")
            print(f"📊 Text extraction: {result['text_extraction']['content_length']} characters")
            print(f"📊 Pages processed: {result['text_extraction']['pages_processed']}")
            print(f"📊 Entities found: {result['entity_extraction']['entities_found']}")
            print(f"📊 Knowledge graph nodes: {result['knowledge_graph']['nodes']}")
            print(f"📊 Knowledge graph edges: {result['knowledge_graph']['edges']}")
            print(f"📊 Processing time: {result['knowledge_graph']['processing_time']:.2f}s")
            
            if result.get('report_files'):
                print("\n📋 Generated reports:")
                for report_type, path in result['report_files'].items():
                    print(f"   - {report_type.upper()}: {path}")
            
            return True
        else:
            print(f"\n❌ PDF processing failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during PDF processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting MCP PDF Processing Test")
    print("=" * 50)
    
    # Run the test
    success = asyncio.run(test_mcp_pdf_processing())
    
    if success:
        print("\n✅ PDF has been successfully processed and added to the knowledge graph!")
        print("✅ Report has been generated!")
    else:
        print("\n❌ PDF processing failed.")
        sys.exit(1)

