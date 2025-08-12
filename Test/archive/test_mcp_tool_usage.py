#!/usr/bin/env python3
"""
Test script to verify MCP tool usage with proper MCP client.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import process_classical_chinese_pdf_via_mcp

async def test_mcp_tool():
    """Test the MCP tool usage with proper MCP client."""
    print("🧪 Testing MCP Tool Usage with Proper MCP Client")
    print("=" * 60)
    
    try:
        result = await process_classical_chinese_pdf_via_mcp(
            'data/Classical Chinese Sample 22208_0_8.pdf', 
            'zh', 
            True
        )
        
        print(f"\n📊 Test Result:")
        print(f"Success: {result.get('success', False)}")
        
        if not result.get('success', False):
            print(f"Error: {result.get('error', 'Unknown error')}")
        else:
            print("✅ MCP tool processing successful!")
            print(f"PDF Path: {result.get('pdf_path', 'Unknown')}")
            print(f"Detected Language: {result.get('detected_language', 'Unknown')}")
            
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
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = asyncio.run(test_mcp_tool())
    print(f"\n🎯 Final Result: {'SUCCESS' if result.get('success', False) else 'FAILED'}")
