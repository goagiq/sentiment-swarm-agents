#!/usr/bin/env python3
"""
Test MCP with session ID that FastMCP expects.
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
            
            # Test with session ID
            base_url = "http://localhost:8004"
            
            # Correct headers
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
            
            # Test different session ID formats
            session_formats = [
                # Test 1: Session ID in headers
                {"X-Session-ID": str(uuid.uuid4())},
                {"Session-ID": str(uuid.uuid4())},
                {"session-id": str(uuid.uuid4())},
                
                # Test 2: Session ID in params
                {"params": {"session_id": str(uuid.uuid4())}},
                {"params": {"sessionId": str(uuid.uuid4())}},
                
                # Test 3: Session ID in request body
                {"session_id": str(uuid.uuid4())},
                {"sessionId": str(uuid.uuid4())},
            ]
            
            for i, session_format in enumerate(session_formats):
                try:
                    print(f"\n🔧 Testing session format {i+1}: {session_format}")
                    
                    # Create MCP request with session
                    mcp_request = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/list",
                        "params": {}
                    }
                    
                    # Add session format to request
                    if isinstance(session_format, dict) and "params" in session_format:
                        # Session in params
                        mcp_request["params"].update(session_format["params"])
                    elif isinstance(session_format, dict):
                        # Session in request body
                        mcp_request.update(session_format)
                    
                    # Add session to headers if it's a header format
                    test_headers = headers.copy()
                    if "X-Session-ID" in session_format:
                        test_headers["X-Session-ID"] = session_format["X-Session-ID"]
                    elif "Session-ID" in session_format:
                        test_headers["Session-ID"] = session_format["Session-ID"]
                    elif "session-id" in session_format:
                        test_headers["session-id"] = session_format["session-id"]
                    
                    response = requests.post(
                        f"{base_url}/mcp/",
                        json=mcp_request,
                        headers=test_headers,
                        timeout=5
                    )
                    print(f"✅ Session format {i+1}: {response.status_code}")
                    print(f"   Response: {response.text[:200]}...")
                    
                    if response.status_code == 200:
                        print("🎉 SUCCESS! Found working session format!")
                        break
                        
                except Exception as e:
                    print(f"❌ Error with session format {i+1}: {e}")
                
        else:
            print("❌ HTTP app is None")
    else:
        print("❌ MCP server is None")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

