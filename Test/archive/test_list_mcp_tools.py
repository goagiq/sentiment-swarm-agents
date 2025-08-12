#!/usr/bin/env python3
"""
Test to list available MCP tools.
"""

import requests
import json

def test_list_mcp_tools():
    """Test listing MCP tools."""
    print("🧪 Testing MCP Tools Listing")
    print("=" * 50)
    
    # Step 1: Initialize MCP session with proper parameters
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/event-stream'
    }
    
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
                "name": "sentiment-analyzer",
                "version": "1.0.0"
            }
        }
    }
    
    print("📡 Initializing MCP session...")
    response = requests.post('http://127.0.0.1:8000/mcp/', headers=headers, json=init_payload)
    
    if response.status_code != 200:
        print(f"❌ Failed to initialize: {response.text}")
        return False
    
    session_id = response.headers.get('mcp-session-id')
    if not session_id:
        print("❌ No session ID received")
        return False
    
    print(f"✅ Session ID: {session_id}")
    
    # Step 2: List tools
    tool_headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/event-stream',
        'mcp-session-id': session_id
    }
    
    list_payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }
    
    print("🔧 Listing MCP tools...")
    list_response = requests.post('http://127.0.0.1:8000/mcp/', headers=tool_headers, json=list_payload)
    print(f"Status: {list_response.status_code}")
    print(f"Response: {list_response.text[:1000]}")
    
    if list_response.status_code == 200:
        print("✅ Tools listing successful!")
        return True
    else:
        print("❌ Tools listing failed")
        return False

if __name__ == "__main__":
    success = test_list_mcp_tools()
    print(f"\n🎯 Result: {'SUCCESS' if success else 'FAILED'}")
