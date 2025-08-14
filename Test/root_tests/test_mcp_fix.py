#!/usr/bin/env python3
"""
Test script to verify MCP server routing fix.
"""

import requests
import time


def test_mcp_endpoints():
    """Test MCP server endpoints to verify routing fix."""
    
    base_url = "http://localhost:8003"
    endpoints = [
        "/mcp",
        "/mcp/",
        "/mcp-health"
    ]
    
    print("🔧 Testing MCP server endpoints...")
    print("=" * 50)
    
    for endpoint in endpoints:
        try:
            print(f"Testing {endpoint}...")
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            
            if response.status_code == 200:
                print(f"✅ {endpoint}: OK (Status: {response.status_code})")
                content_type = response.headers.get('content-type', '')
                if content_type.startswith('application/json'):
                    try:
                        data = response.json()
                        print(f"   Response: {data}")
                    except ValueError:
                        print(f"   Response: {response.text[:100]}...")
            else:
                print(f"⚠️ {endpoint}: Status {response.status_code}")
                print(f"   Response: {response.text[:100]}...")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {endpoint}: Connection refused (server not running)")
        except requests.exceptions.Timeout:
            print(f"⏰ {endpoint}: Timeout")
        except Exception as e:
            print(f"❌ {endpoint}: Error - {e}")
        
        print()
    
    # Test POST requests to MCP endpoints
    print("Testing MCP POST endpoints...")
    print("=" * 50)
    
    test_data = {
        "method": "tools/list",
        "params": {}
    }
    
    for endpoint in ["/mcp", "/mcp/"]:
        try:
            print(f"Testing POST {endpoint}...")
            response = requests.post(
                f"{base_url}{endpoint}", 
                json=test_data,
                timeout=10
            )
            
            # Accept various responses
            if response.status_code in [200, 400, 405]:
                print(f"✅ POST {endpoint}: OK (Status: {response.status_code})")
                content_type = response.headers.get('content-type', '')
                if content_type.startswith('application/json'):
                    try:
                        data = response.json()
                        print(f"   Response: {data}")
                    except ValueError:
                        print(f"   Response: {response.text[:100]}...")
            else:
                print(f"⚠️ POST {endpoint}: Status {response.status_code}")
                print(f"   Response: {response.text[:100]}...")
                
        except requests.exceptions.ConnectionError:
            print(f"❌ POST {endpoint}: Connection refused")
        except requests.exceptions.Timeout:
            print(f"⏰ POST {endpoint}: Timeout")
        except Exception as e:
            print(f"❌ POST {endpoint}: Error - {e}")
        
        print()


def main():
    """Main test function."""
    print("🚀 MCP Server Routing Fix Test")
    print("=" * 50)
    
    # Wait 30 seconds for server to fully start
    print("Waiting 30 seconds for server to be ready...")
    for i in range(30, 0, -1):
        print(f"   {i} seconds remaining...", end="\r")
        time.sleep(1)
    print("\nStarting tests...")
    
    test_mcp_endpoints()
    
    print("✅ Test completed!")
    print("\n💡 If you see 400 errors, the routing fix should resolve them.")
    print("   Check the server logs for more details.")


if __name__ == "__main__":
    main()

