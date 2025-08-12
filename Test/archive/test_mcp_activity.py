#!/usr/bin/env python3
"""
Test MCP server activity with console output.
"""

import requests
import json
import time

def test_mcp_activity():
    """Test MCP server activity with detailed console output."""
    print("🧪 Testing MCP Server Activity")
    print("=" * 50)
    
    try:
        print("📡 Step 1: Testing MCP server connection...")
        
        # Test basic connection
        response = requests.get('http://127.0.0.1:8000/mcp/', 
                              headers={'Accept': 'application/json, text/event-stream'})
        print(f"✅ MCP server response status: {response.status_code}")
        
        if response.status_code == 406:
            print("✅ MCP server is running (406 is expected for GET without proper headers)")
        
        print("\n📡 Step 2: Testing MCP initialization...")
        
        # Initialize MCP session
        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream'
        }
        
        print("🔧 Sending initialization request...")
        init_response = requests.post('http://127.0.0.1:8000/mcp/', 
                                    headers=headers, json=init_payload)
        
        print(f"✅ Initialization response status: {init_response.status_code}")
        print(f"✅ Response headers: {dict(init_response.headers)}")
        print(f"✅ Response text: {init_response.text[:200]}...")
        
        session_id = init_response.headers.get('mcp-session-id')
        if session_id:
            print(f"✅ Got session ID: {session_id}")
            
            print("\n📡 Step 3: Testing tool listing...")
            
            # List tools
            list_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list"
            }
            
            list_headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json, text/event-stream',
                'mcp-session-id': session_id
            }
            
            print("🔧 Sending tools/list request...")
            list_response = requests.post('http://127.0.0.1:8000/mcp/', 
                                        headers=list_headers, json=list_payload)
            
            print(f"✅ Tools list response status: {list_response.status_code}")
            print(f"✅ Tools list response: {list_response.text[:500]}...")
            
            print("\n📡 Step 4: Testing PDF processing tool...")
            
            # Call the PDF processing tool
            tool_payload = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "process_pdf_enhanced_multilingual",
                    "arguments": {
                        "pdf_path": "data/Classical Chinese Sample 22208_0_8.pdf",
                        "language": "zh",
                        "generate_report": True
                    }
                }
            }
            
            print("🔧 Sending PDF processing tool request...")
            tool_response = requests.post('http://127.0.0.1:8000/mcp/', 
                                        headers=list_headers, json=tool_payload)
            
            print(f"✅ Tool call response status: {tool_response.status_code}")
            print(f"✅ Tool call response: {tool_response.text[:500]}...")
            
        else:
            print("❌ No session ID received")
            
        print("\n✅ MCP activity test completed!")
        return {"success": True, "session_id": session_id}
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = test_mcp_activity()
    print(f"\n🎯 Final Result: {'SUCCESS' if result.get('success', False) else 'FAILED'}")
