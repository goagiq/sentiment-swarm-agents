#!/usr/bin/env python3
"""
Simple test script to check MCP server initialization.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_mcp_initialization():
    """Test MCP server initialization."""
    try:
        from mcp.server import FastMCP
        print("✅ FastMCP import successful")
        
        mcp = FastMCP("Test Server")
        print("✅ FastMCP server created successfully")
        
        # Test tool registration
        @mcp.tool(description="Test tool")
        def test_tool():
            return "Test successful"
        
        print("✅ Tool registered successfully")
        
        # Test running the server
        print("🚀 Testing server run...")
        # Don't actually run it, just test the setup
        print("✅ MCP server setup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing MCP server initialization...")
    success = test_mcp_initialization()
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Tests failed!")

