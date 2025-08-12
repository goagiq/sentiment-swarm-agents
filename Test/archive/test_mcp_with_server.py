#!/usr/bin/env python3
"""
Test MCP with server startup.
"""

import asyncio
import sys
import os
import time

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_mcp_with_server():
    """Test MCP with server startup."""
    print("🧪 Testing MCP with Server Startup")
    print("=" * 50)
    
    try:
        print("📡 Step 1: Starting MCP server...")
        
        from main import start_mcp_server
        
        # Start the MCP server
        mcp_server = start_mcp_server()
        
        if mcp_server is None:
            print("❌ Failed to start MCP server")
            return {"success": False, "error": "MCP server not available"}
        
        print("✅ MCP server started successfully")
        
        # Wait a moment for the server to fully start
        print("⏱️ Waiting for server to fully start...")
        await asyncio.sleep(2)
        
        print("\n📡 Step 2: Testing PDF processing...")
        
        # Test the PDF processing function
        from main import process_classical_chinese_pdf_via_mcp
        
        result = await process_classical_chinese_pdf_via_mcp(
            pdf_path="data/Classical Chinese Sample 22208_0_8.pdf",
            language="zh",
            generate_report=True
        )
        
        print("✅ PDF processing completed!")
        print(f"Success: {result.get('success', False)}")
        
        if result.get('success', False):
            print("🎉 PDF processing successful!")
            print(f"Language: {result.get('language', 'Unknown')}")
            
            # Print extraction info
            text_extraction = result.get('text_extraction', {})
            if text_extraction.get('success'):
                print(f"Text Length: {text_extraction.get('content_length', 'Unknown')}")
                print(f"Pages Processed: {text_extraction.get('pages_processed', 'Unknown')}")
            
            # Print entity info
            entity_extraction = result.get('entity_extraction', {})
            print(f"Entities Found: {entity_extraction.get('entities_found', 0)}")
            
            # Print knowledge graph info
            kg_info = result.get('knowledge_graph', {})
            print(f"Knowledge Graph Nodes: {kg_info.get('nodes', 0)}")
            print(f"Knowledge Graph Edges: {kg_info.get('edges', 0)}")
            
            # Print report info
            report_files = result.get('report_files', {})
            if report_files:
                print(f"Report HTML: {report_files.get('html', 'Unknown')}")
                print(f"Report PNG: {report_files.get('png', 'Unknown')}")
        else:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = asyncio.run(test_mcp_with_server())
    print(f"\n🎯 Final Result: {'SUCCESS' if result.get('success', False) else 'FAILED'}")
