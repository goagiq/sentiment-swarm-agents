#!/usr/bin/env python3
"""
Direct test of MCP client approach.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_mcp_client_direct():
    """Test MCP client directly."""
    print("🧪 Testing MCP Client Directly")
    print("=" * 50)
    
    try:
        from mcp.client.streamable_http import streamablehttp_client
        from strands.tools.mcp.mcp_client import MCPClient
        
        def create_streamable_http_transport():
            return streamablehttp_client("http://localhost:8000/mcp/")
        
        streamable_http_mcp_client = MCPClient(create_streamable_http_transport)
        
        print("📡 Initializing MCP client...")
        
        # Use the MCP server in a context manager
        with streamable_http_mcp_client:
            # Get the tools from the MCP server
            tools = streamable_http_mcp_client.list_tools_sync()
            print(f"✅ Available MCP tools: {len(tools)} tools")
            
            # List available tools
            for tool in tools:
                print(f"  - {tool.name}")
            
            # Call the specific tool
            print("🔧 Calling process_pdf_enhanced_multilingual tool...")
            result = streamable_http_mcp_client.call_tool_sync(
                "process_pdf_enhanced_multilingual",
                {
                    "pdf_path": "data/Classical Chinese Sample 22208_0_8.pdf",
                    "language": "zh",
                    "generate_report": True
                }
            )
            
            print("✅ MCP tool processing completed successfully")
            print(f"Result: {result}")
            
            return {"success": True, "result": result}
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = asyncio.run(test_mcp_client_direct())
    print(f"\n🎯 Final Result: {'SUCCESS' if result.get('success', False) else 'FAILED'}")
