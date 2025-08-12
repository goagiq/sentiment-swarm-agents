#!/usr/bin/env python3
"""
Test direct MCP tool call.
"""

import requests
import json

def test_mcp_tool():
    """Test calling the MCP tool directly."""
    print("🧪 Testing Direct MCP Tool Call")
    print("=" * 50)
    
    # Step 1: Initialize session
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/event-stream'
    }
    
    init_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize"
    }
    
    print("📡 Initializing MCP session...")
    response = requests.post('http://127.0.0.1:8000/mcp/', headers=headers, json=init_payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code != 200:
        print(f"❌ Failed to initialize: {response.text}")
        return False
    
    session_id = response.headers.get('mcp-session-id')
    if not session_id:
        print("❌ No session ID received")
        return False
    
    print(f"✅ Session ID: {session_id}")
    
    # Step 2: Call the MCP tool
    tool_headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/event-stream',
        'mcp-session-id': session_id
    }
    
    tool_payload = {
        "jsonrpc": "2.0",
        "id": 2,
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
    
    print("🔧 Calling MCP tool: process_pdf_enhanced_multilingual")
    tool_response = requests.post('http://127.0.0.1:8000/mcp/', headers=tool_headers, json=tool_payload)
    print(f"Status: {tool_response.status_code}")
    print(f"Response: {tool_response.text[:500]}")
    
    if tool_response.status_code == 200:
        print("✅ MCP tool call successful!")
        return True
    else:
        print("❌ MCP tool call failed")
        return False

if __name__ == "__main__":
    success = test_mcp_tool()
    print(f"\n🎯 Result: {'SUCCESS' if success else 'FAILED'}")
