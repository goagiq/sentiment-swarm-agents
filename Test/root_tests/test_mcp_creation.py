#!/usr/bin/env python3
"""
Test to verify MCP server creation in main server.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    # Import the main server components
    from src.mcp_servers.unified_mcp_server import create_unified_mcp_server
    from src.api.main import app
    
    print("✅ Successfully imported components")
    
    # Test MCP server creation
    print("🔧 Testing MCP server creation...")
    mcp_server = create_unified_mcp_server()
    print(f"✅ MCP server created: {mcp_server}")
    
    if mcp_server and mcp_server.mcp:
        print("✅ MCP server is properly initialized")
        
        # Test HTTP app creation
        print("🔧 Testing HTTP app creation...")
        http_app = mcp_server.get_http_app(path="")
        print(f"✅ HTTP app created: {http_app}")
        
        if http_app:
            print("✅ HTTP app is available")
            
            # Check if the app has routes
            if hasattr(http_app, 'routes'):
                print(f"✅ HTTP app has {len(http_app.routes)} routes")
            else:
                print("⚠️ HTTP app has no routes attribute")
                
        else:
            print("❌ HTTP app is None")
    else:
        print("❌ MCP server is None or not properly initialized")
    
    # Test FastAPI app
    print("\n🔧 Testing FastAPI app...")
    print(f"✅ FastAPI app: {app}")
    
    # Check if MCP routes are mounted
    print("🔧 Checking mounted routes...")
    for route in app.routes:
        if hasattr(route, 'path') and 'mcp' in str(route.path):
            print(f"✅ Found MCP route: {route.path}")
        elif hasattr(route, 'app') and hasattr(route.app, 'routes'):
            print(f"✅ Found mounted app: {route.app}")
            
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

