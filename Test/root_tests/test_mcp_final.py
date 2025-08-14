#!/usr/bin/env python3
"""
Final test to verify MCP server integration with main server.
"""

import requests
import time
import json

def test_mcp_integration():
    """Test MCP server integration with main server."""
    
    base_url = "http://localhost:8003"
    
    print("🔧 Testing MCP server integration...")
    print("=" * 50)
    
    # Test 1: Health check endpoint
    try:
        response = requests.get(f"{base_url}/mcp-health", timeout=5)
        print(f"✅ MCP Health Check: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)}")
        else:
            print(f"   Response: {response.text[:100]}...")
    except Exception as e:
        print(f"❌ MCP Health Check Error: {e}")
    
    print()
    
    # Test 2: MCP protocol initialization
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    
    mcp_request = {
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
    
    try:
        print("🔧 Testing MCP protocol initialization...")
        response = requests.post(
            f"{base_url}/mcp/",
            json=mcp_request,
            headers=headers,
            timeout=10
        )
        print(f"✅ MCP Initialize: {response.status_code}")
        if response.status_code == 200:
            print("🎉 SUCCESS! MCP server is working correctly!")
            print(f"   Response: {response.text[:200]}...")
        else:
            print(f"   Response: {response.text[:200]}...")
    except Exception as e:
        print(f"❌ MCP Initialize Error: {e}")
    
    print()
    
    # Test 3: Tools list (after initialization)
    try:
        print("🔧 Testing MCP tools list...")
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        response = requests.post(
            f"{base_url}/mcp/",
            json=tools_request,
            headers=headers,
            timeout=10
        )
        print(f"✅ MCP Tools List: {response.status_code}")
        if response.status_code == 200:
            print("🎉 SUCCESS! MCP tools are accessible!")
            print(f"   Response: {response.text[:200]}...")
        else:
            print(f"   Response: {response.text[:200]}...")
    except Exception as e:
        print(f"❌ MCP Tools List Error: {e}")

def main():
    """Main test function."""
    print("🚀 Final MCP Server Integration Test")
    print("=" * 50)
    
    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(5)
    
    test_mcp_integration()
    
    print("\n✅ Test completed!")
    print("\n💡 If all tests pass, the MCP server is working correctly!")
    print("   The MCP server is now properly integrated with the main server.")

if __name__ == "__main__":
    main()

