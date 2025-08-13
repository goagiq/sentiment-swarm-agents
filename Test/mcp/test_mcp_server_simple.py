"""
Simple test for MCP server with proper headers.
"""

import requests
import json

def test_mcp_server():
    """Test MCP server with proper headers."""
    
    print("🧪 Testing MCP Server with Proper Headers")
    print("=" * 50)
    
    # Test 1: Check server status with proper headers
    print("\n1. Testing server status...")
    try:
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        response = requests.get("http://localhost:8000/mcp/", headers=headers, timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Content: {response.text[:200]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Test tools listing
    print("\n2. Testing tools listing...")
    try:
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        response = requests.get("http://localhost:8000/mcp/tools", headers=headers, timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            tools = response.json()
            print(f"✅ Found {len(tools.get('tools', []))} tools")
        else:
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Test JSON-RPC call
    print("\n3. Testing JSON-RPC call...")
    try:
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        }
        
        response = requests.post("http://localhost:8000/mcp/", 
                               json=payload, 
                               headers=headers, 
                               timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ JSON-RPC call successful")
            print(f"Result: {result}")
        else:
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 MCP Server Test Complete!")

if __name__ == "__main__":
    test_mcp_server()
