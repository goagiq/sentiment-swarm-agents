#!/usr/bin/env python3
"""
Direct test of MCP server HTTP app.
"""

import sys
from pathlib import Path
import asyncio
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
            
            # Test the server
            try:
                # Test root endpoint
                response = requests.get("http://localhost:8004/", timeout=5)
                print(f"✅ Root endpoint: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
                # Test tools/list endpoint
                response = requests.post(
                    "http://localhost:8004/",
                    json={"method": "tools/list", "params": {}},
                    timeout=5
                )
                print(f"✅ Tools list endpoint: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
            except Exception as e:
                print(f"❌ Error testing server: {e}")
                
        else:
            print("❌ HTTP app is None")
    else:
        print("❌ MCP server is None")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

