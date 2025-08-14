#!/usr/bin/env python3
"""
Test MCP session initialization and different methods.
"""

import sys
from pathlib import Path
import uvicorn
import threading
import time
import requests
import uuid

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
            
            # Test with session initialization
            base_url = "http://localhost:8004"
            
            # Correct headers
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
            
            # Test different MCP methods
            methods_to_test = [
                "initialize",
                "tools/list",
                "tools/call",
                "sessions/list",
                "session/initialize"
            ]
            
            for method in methods_to_test:
                try:
                    print(f"\n🔧 Testing method: {method}")
                    
                    # Create MCP request
                    mcp_request = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": method,
                        "params": {}
                    }
                    
                    # Add session ID to params for initialize
                    if method == "initialize":
                        mcp_request["params"] = {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {
                                "tools": {}
                            },
                            "clientInfo": {
                                "name": "test-client",
                                "version": "1.0.0"
                            }
                        }
                    
                    response = requests.post(
                        f"{base_url}/mcp/",
                        json=mcp_request,
                        headers=headers,
                        timeout=5
                    )
                    print(f"✅ Method {method}: {response.status_code}")
                    print(f"   Response: {response.text[:200]}...")
                    
                    if response.status_code == 200:
                        print("🎉 SUCCESS! Found working method!")
                        break
                        
                except Exception as e:
                    print(f"❌ Error with method {method}: {e}")
                
        else:
            print("❌ HTTP app is None")
    else:
        print("❌ MCP server is None")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

