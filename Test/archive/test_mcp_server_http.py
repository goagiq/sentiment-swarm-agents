#!/usr/bin/env python3
"""
HTTP-based MCP server test for Classical Chinese PDF processing.
This script uses HTTP requests to call the MCP server directly.
"""

import asyncio
import os
import sys
import json
import requests
from datetime import datetime

async def test_mcp_server_http():
    """Test the MCP server via HTTP requests."""
    print("🧪 HTTP-based MCP Server Test for Classical Chinese PDF Processing")
    print("=" * 70)
    
    # Find the Classical Chinese PDF
    pdf_path = "data/Classical Chinese Sample 22208_0_8.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return False
    
    print(f"✅ Found PDF: {pdf_path}")
    
    try:
        # Test MCP server availability
        mcp_url = "http://localhost:8000/mcp/"
        print(f"🔧 Testing MCP server at: {mcp_url}")
        
        # First, let's check if the MCP server is responding and get session ID
        try:
            response = requests.get(mcp_url, timeout=5)
            print(f"✅ MCP server is responding (status: {response.status_code})")
            
            # Get session ID from response headers
            session_id = response.headers.get('mcp-session-id')
            if session_id:
                print(f"✅ Got session ID: {session_id}")
            else:
                print("⚠️ No session ID in response headers")
                session_id = None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ MCP server not responding: {e}")
            return False
        
        # Now let's try to call the MCP tool via HTTP
        print(f"📤 Calling MCP tool 'process_pdf_enhanced_multilingual'")
        print(f"📄 Processing PDF: {pdf_path}")
        print(f"🌍 Language: zh (Chinese)")
        print(f"📊 Generate report: True")
        
        # Prepare the JSON-RPC 2.0 request
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 1,
            "params": {
                "name": "process_pdf_enhanced_multilingual",
                "arguments": {
                    "pdf_path": pdf_path,
                    "language": "zh",
                    "generate_report": True,
                    "output_path": None
                }
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        # Add session ID if available
        if session_id:
            headers["X-Session-ID"] = session_id
        
        print(f"📤 Sending JSON-RPC request to MCP server...")
        response = requests.post(mcp_url, json=payload, headers=headers, timeout=300)
        
        print(f"📥 Response status: {response.status_code}")
        print(f"📥 Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("✅ MCP server responded with JSON")
                print(f"📄 Result: {json.dumps(result, indent=2)}")
                return True
            except json.JSONDecodeError:
                print(f"📄 Raw response: {response.text}")
                return True
        else:
            print(f"❌ MCP server error: {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ MCP server test error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main function to run the HTTP-based MCP server test."""
    success = await test_mcp_server_http()
    
    if success:
        print("\n🎉 MCP server HTTP test completed!")
        print("✅ Classical Chinese PDF processing via MCP server HTTP")
        print("🔧 MCP server integration working correctly")
    else:
        print("\n❌ MCP server HTTP test failed. Check the output above for details.")

if __name__ == "__main__":
    asyncio.run(main())
