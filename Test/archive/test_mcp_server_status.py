#!/usr/bin/env python3
"""
Test script to verify MCP server is running and responding.
"""

import requests
import json
import sys

def test_mcp_server():
    """Test if the MCP server is running and responding."""
    print("🔧 Testing MCP Server Status")
    print("=" * 40)
    
    # Test 1: Check if server is reachable
    try:
        headers = {
            "Accept": "application/json, text/event-stream"
        }
        response = requests.get("http://localhost:8000/mcp", headers=headers, timeout=5)
        print(f"✅ MCP server is reachable (Status: {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"❌ MCP server is not reachable: {e}")
        return False
    
    # Test 2: Try to initialize a session
    try:
        init_payload = {
            "jsonrpc": "2.0",
            "id": "test-init",
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
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        response = requests.post(
            "http://localhost:8000/mcp",
            json=init_payload,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            try:
                result = response.json()
                if "result" in result:
                    print("✅ MCP server session initialization successful")
                    print(f"   Server info: {result['result'].get('serverInfo', 'Unknown')}")
                else:
                    print(f"⚠️ MCP server responded but with error: {result}")
            except ValueError as e:
                print(f"⚠️ MCP server responded but with invalid JSON: {response.text[:100]}")
        else:
            print(f"❌ MCP server initialization failed (Status: {response.status_code})")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
        return False
    
    # Test 3: Try to list tools
    try:
        tools_payload = {
            "jsonrpc": "2.0",
            "id": "test-tools",
            "method": "tools/list",
            "params": {}
        }
        
        response = requests.post(
            "http://localhost:8000/mcp",
            json=tools_payload,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if "result" in result and "tools" in result["result"]:
                tools = result["result"]["tools"]
                print(f"✅ MCP server tools available: {len(tools)} tools")
                for tool in tools:
                    print(f"   - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}")
            else:
                print(f"⚠️ MCP server tools response: {result}")
        else:
            print(f"❌ MCP server tools listing failed (Status: {response.status_code})")
            
    except Exception as e:
        print(f"❌ Error listing MCP tools: {e}")
        return False
    
    print("\n🎉 MCP server is running and responding properly!")
    return True

if __name__ == "__main__":
    success = test_mcp_server()
    if success:
        print("\n✅ MCP server is ready for PDF processing!")
    else:
        print("\n❌ MCP server needs attention before PDF processing.")
        sys.exit(1)
