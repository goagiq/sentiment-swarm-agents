#!/usr/bin/env python3
"""
Detailed test to check MCP server mounting and HTTP app.
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
        
        # Try to get HTTP app with different paths
        for path in ["", "/mcp", "/test"]:
            try:
                print(f"\n🔧 Testing path: '{path}'")
                http_app = mcp_server.get_http_app(path=path)
                print(f"   HTTP app created: {http_app}")
                
                if http_app:
                    print("   ✅ HTTP app is available")
                    # Try to get the app type
                    print(f"   App type: {type(http_app)}")
                    
                    # Try to check if it has routes
                    if hasattr(http_app, 'routes'):
                        print(f"   Routes: {len(http_app.routes)} routes")
                    elif hasattr(http_app, 'app') and hasattr(http_app.app, 'routes'):
                        print(f"   Routes: {len(http_app.app.routes)} routes")
                    else:
                        print("   No routes attribute found")
                        
                else:
                    print("   ❌ HTTP app is None")
                    
            except Exception as e:
                print(f"   ❌ Error creating HTTP app with path '{path}': {e}")
                import traceback
                traceback.print_exc()
    else:
        print("❌ MCP server is None")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

