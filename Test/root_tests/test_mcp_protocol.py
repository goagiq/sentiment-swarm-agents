#!/usr/bin/env python3
"""
Test MCP protocol with proper format.
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
            
            # Test with proper MCP protocol format
            base_url = "http://localhost:8004"
            
            # Test 1: Try with proper MCP headers
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # Test 2: Try with MCP protocol format
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {}
            }
            
            try:
                print("🔧 Testing MCP protocol format...")
                response = requests.post(
                    f"{base_url}/mcp/",
                    json=mcp_request,
                    headers=headers,
                    timeout=5
                )
                print(f"✅ MCP protocol test: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
            except Exception as e:
                print(f"❌ Error testing MCP protocol: {e}")
            
            # Test 3: Try with different content types
            content_types = [
                "application/json",
                "application/json-rpc",
                "application/jsonrpc",
                "text/plain"
            ]
            
            for content_type in content_types:
                try:
                    headers = {"Content-Type": content_type}
                    response = requests.post(
                        f"{base_url}/mcp/",
                        json=mcp_request,
                        headers=headers,
                        timeout=5
                    )
                    print(f"✅ Content-Type {content_type}: {response.status_code}")
                    if response.status_code == 200:
                        print(f"   Response: {response.text[:100]}...")
                        
                except Exception as e:
                    print(f"❌ Error with {content_type}: {e}")
            
            # Test 4: Try GET request to see what the server expects
            try:
                response = requests.get(f"{base_url}/mcp/", headers=headers, timeout=5)
                print(f"✅ GET /mcp/: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
            except Exception as e:
                print(f"❌ Error with GET: {e}")
                
        else:
            print("❌ HTTP app is None")
    else:
        print("❌ MCP server is None")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

