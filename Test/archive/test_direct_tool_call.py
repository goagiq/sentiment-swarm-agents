#!/usr/bin/env python3
"""
Test direct tool call through MCP server.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_direct_tool_call():
    """Test calling the MCP tool directly through the server."""
    print("🧪 Testing Direct Tool Call Through MCP Server")
    print("=" * 60)
    
    try:
        # Import the MCP server and call the tool directly
        from src.core.mcp_server import OptimizedMCPServer
        
        # Create MCP server instance
        mcp_server = OptimizedMCPServer()
        
        print("📡 Created MCP server instance")
        
        # Check if the tool is available
        if hasattr(mcp_server, 'mcp') and mcp_server.mcp:
            print("✅ MCP server has mcp attribute")
            
            # Try to call the tool directly
            print("🔧 Calling process_pdf_enhanced_multilingual directly...")
            
            # Call the tool function directly
            result = await mcp_server.mcp.process_pdf_enhanced_multilingual(
                pdf_path="data/Classical Chinese Sample 22208_0_8.pdf",
                language="zh",
                generate_report=True
            )
            
            print("✅ Direct tool call successful!")
            print(f"Result: {result}")
            
            return {"success": True, "result": result}
        else:
            print("❌ MCP server does not have mcp attribute")
            return {"success": False, "error": "MCP server not properly initialized"}
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = asyncio.run(test_direct_tool_call())
    print(f"\n🎯 Final Result: {'SUCCESS' if result.get('success', False) else 'FAILED'}")
