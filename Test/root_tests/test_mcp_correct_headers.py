#!/usr/bin/env python3
"""
Test MCP with correct headers that FastMCP expects.
"""

import sys
from pathlib import Path
import uvicorn
import threading
import time
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.mcp_servers.unified_mcp_server import create_unified_mcp_server
    print("✅ Successfully imported create_unified_mcp_server")
    
    # Create MCP server
    mcp_server = create_unified_mcp_server()
    print(f"✅ MCP server created: {mcp_server}")
    
    if mcp_server.mcp:
        print("✅ MCP server is properly initialized")
        
        # Get HTTP app
        http_app = mcp_server.get_http_app(path="")
        print(f"✅ HTTP app created: {http_app}")
        
        if http_app:
            print("✅ HTTP app is available")
            
            # Start the HTTP app in a separate thread
            def run_server():
                uvicorn.run(http_app, host="localhost", port=8004, log_level="info")
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            # Wait for server to start
            print("⏳ Waiting for server to start...")
            time.sleep(5)
            
            # Test with correct headers that FastMCP expects
            base_url = "http://localhost:8004"
            
            # Correct headers based on the error message
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
            
            # MCP protocol format
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            }
            
            try:
                print("🔧 Testing with correct headers...")
                response = requests.post(
                    f"{base_url}/mcp/",
                    json=mcp_request,
                    headers=headers,
                    timeout=5
                )
                print(f"✅ Correct headers test: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
                if response.status_code == 200:
                    print("🎉 SUCCESS! MCP server is working!")
                else:
                    print(f"❌ Still getting error: {response.status_code}")
                
            except Exception as e:
                print(f"❌ Error testing with correct headers: {e}")
                
        else:
            print("❌ HTTP app is None")
    else:
        print("❌ MCP server is None")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

