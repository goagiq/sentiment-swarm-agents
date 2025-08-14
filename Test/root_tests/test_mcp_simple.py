#!/usr/bin/env python3
"""
Simple test to check MCP server initialization.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.mcp_servers.unified_mcp_server import create_unified_mcp_server
    print("✅ Successfully imported create_unified_mcp_server")
    
    # Create MCP server
    mcp_server = create_unified_mcp_server()
    print(f"✅ MCP server created: {mcp_server}")
    print(f"   MCP object: {mcp_server.mcp}")
    
    if mcp_server.mcp:
        print("✅ MCP server is properly initialized")
        
        # Try to get HTTP app
        try:
            http_app = mcp_server.get_http_app(path="/mcp")
            print(f"✅ HTTP app created: {http_app}")
            
            if http_app:
                print("✅ HTTP app is available")
            else:
                print("❌ HTTP app is None")
                
        except Exception as e:
            print(f"❌ Error creating HTTP app: {e}")
    else:
        print("❌ MCP server is None")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

