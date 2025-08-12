#!/usr/bin/env python3
"""
Simple test to check MCP client functionality.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_simple_mcp():
    """Simple test of MCP client."""
    print("🧪 Simple MCP Client Test")
    print("=" * 40)
    
    try:
        print("📡 Testing MCP client import...")
        from mcp.client.streamable_http import streamablehttp_client
        from strands.tools.mcp.mcp_client import MCPClient
        
        print("✅ MCP client imports successful")
        
        def create_streamable_http_transport():
            return streamablehttp_client("http://localhost:8000/mcp/")
        
        streamable_http_mcp_client = MCPClient(create_streamable_http_transport)
        
        print("📡 Created MCP client")
        
        # Use the MCP server in a context manager
        with streamable_http_mcp_client:
            print("✅ Entered MCP client context")
            
            # Get the tools from the MCP server
            tools = streamable_http_mcp_client.list_tools_sync()
            print(f"✅ Available MCP tools: {len(tools)} tools")
            
            # List available tools
            for tool in tools:
                print(f"  - {tool.name}")
            
            print("✅ MCP client test completed successfully")
            return {"success": True, "tools_count": len(tools)}
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return {"success": False, "error": f"Import error: {e}"}
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = asyncio.run(test_simple_mcp())
    print(f"\n🎯 Final Result: {'SUCCESS' if result.get('success', False) else 'FAILED'}")
