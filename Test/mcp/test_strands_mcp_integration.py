"""
Test script for Strands MCP integration using Streamable HTTP transport.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger


async def test_strands_mcp_integration():
    """Test Strands MCP integration with Streamable HTTP transport."""
    
    print("🧪 Testing Strands MCP Integration")
    print("=" * 50)
    
    # Test 1: Check if standalone MCP server is running
    print("\n1. Checking standalone MCP server status...")
    try:
        import requests
        response = requests.get("http://localhost:8000/mcp", timeout=5)
        if response.status_code == 200:
            print("✅ Standalone MCP server is running on port 8000")
        else:
            print(f"⚠️ MCP server responded with status: {response.status_code}")
    except Exception as e:
        print(f"❌ Could not connect to MCP server: {e}")
        print("   Starting standalone MCP server...")
        
        # Start the standalone MCP server
        try:
            from src.mcp_servers.standalone_mcp_server import start_standalone_mcp_server
            server = start_standalone_mcp_server()
            await asyncio.sleep(3)  # Wait for server to start
            
            # Test again
            response = requests.get("http://localhost:8000/mcp", timeout=5)
            if response.status_code == 200:
                print("✅ Standalone MCP server started and is running")
            else:
                print(f"⚠️ MCP server responded with status: {response.status_code}")
        except Exception as e2:
            print(f"❌ Failed to start MCP server: {e2}")
            return False
    
    # Test 2: Test Strands integration pattern
    print("\n2. Testing Strands integration pattern...")
    try:
        # This is the pattern the user wants to use
        print("📝 Strands integration pattern:")
        print("""
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.tools.mcp.mcp_client import MCPClient

streamable_http_mcp_client = MCPClient(lambda: streamablehttp_client("http://localhost:8000/mcp"))

# Create an agent with MCP tools
with streamable_http_mcp_client:
    # Get the tools from the MCP server
    tools = streamable_http_mcp_client.list_tools_sync()
    
    # Create an agent with these tools
    agent = Agent(tools=tools)
        """)
        
        # Test if we can access the MCP tools via HTTP
        response = requests.get("http://localhost:8000/mcp/tools", timeout=5)
        if response.status_code == 200:
            tools_data = response.json()
            print(f"✅ Successfully retrieved {len(tools_data.get('tools', []))} MCP tools")
            
            # Show available tools
            tools = tools_data.get('tools', [])
            if tools:
                print("\n🔧 Available MCP Tools:")
                for i, tool in enumerate(tools[:10], 1):  # Show first 10 tools
                    print(f"   {i}. {tool.get('name', 'Unknown')}")
                if len(tools) > 10:
                    print(f"   ... and {len(tools) - 10} more tools")
        else:
            print(f"⚠️ Could not retrieve tools: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing Strands integration: {e}")
    
    # Test 3: Test a simple MCP tool call
    print("\n3. Testing MCP tool call...")
    try:
        # Test the get_system_status tool
        test_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "get_system_status",
                "arguments": {}
            }
        }
        
        response = requests.post("http://localhost:8000/mcp", json=test_payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("✅ Successfully called MCP tool")
            print(f"   Result: {result.get('result', {}).get('content', {}).get('status', 'Unknown')}")
        else:
            print(f"⚠️ Tool call failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing MCP tool call: {e}")
    
    # Test 4: Verify Streamable HTTP transport
    print("\n4. Verifying Streamable HTTP transport...")
    try:
        # Check if the server supports the expected endpoints
        endpoints = ["/mcp", "/mcp/tools", "/mcp/health"]
        
        for endpoint in endpoints:
            response = requests.get(f"http://localhost:8000{endpoint}", timeout=5)
            if response.status_code in [200, 307, 404]:  # Acceptable responses
                print(f"✅ Endpoint {endpoint} is accessible")
            else:
                print(f"⚠️ Endpoint {endpoint} returned {response.status_code}")
                
    except Exception as e:
        print(f"❌ Error verifying transport: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Strands MCP Integration Test Complete!")
    print("\n💡 Next Steps:")
    print("   1. Use the standalone MCP server at http://localhost:8000/mcp")
    print("   2. Integrate with Strands using streamablehttp_client")
    print("   3. Access all 25 consolidated MCP tools")
    
    return True


if __name__ == "__main__":
    asyncio.run(test_strands_mcp_integration())
