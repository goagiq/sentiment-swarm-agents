#!/usr/bin/env python3
"""
Simple test for Strands MCP client connection.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all required imports work."""
    print("🔧 Testing imports...")
    
    try:
        from mcp.client.streamable_http import streamablehttp_client
        print("✅ MCP streamablehttp_client import successful")
    except Exception as e:
        print(f"❌ MCP import failed: {e}")
        return False
    
    try:
        from strands import Agent
        print("✅ Strands Agent import successful")
    except Exception as e:
        print(f"❌ Strands import failed: {e}")
        return False
    
    try:
        from strands.tools.mcp.mcp_client import MCPClient
        print("✅ MCPClient import successful")
    except Exception as e:
        print(f"❌ MCPClient import failed: {e}")
        return False
    
    return True

def test_mcp_connection():
    """Test MCP server connection."""
    print("\n🔧 Testing MCP server connection...")
    
    try:
        from mcp.client.streamable_http import streamablehttp_client
        from strands.tools.mcp.mcp_client import MCPClient
        
        def create_streamable_http_transport():
            return streamablehttp_client("http://localhost:8000/mcp/")
        
        streamable_http_mcp_client = MCPClient(create_streamable_http_transport)
        print("✅ MCP client created successfully")
        
        # Use the MCP server in a context manager
        with streamable_http_mcp_client:
            print("✅ Connected to MCP server")
            
            # Get the tools from the MCP server
            tools = streamable_http_mcp_client.list_tools_sync()
            print(f"✅ Found {len(tools)} tools from MCP server")
            
            # List available tools
            print("Available tools:")
            for tool in tools:
                print(f"   - {tool.name}")
            
            return True
            
    except Exception as e:
        print(f"❌ MCP connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("🧪 Simple Strands MCP Client Test")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed")
        return False
    
    # Test connection
    if not test_mcp_connection():
        print("\n❌ Connection test failed")
        return False
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    main()
