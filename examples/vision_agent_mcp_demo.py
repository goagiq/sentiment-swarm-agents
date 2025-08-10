"""
Demo script for VisionAgent MCP Server.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp.vision_agent_server import create_vision_agent_mcp_server


async def demo_vision_agent_mcp_server():
    """Demonstrate the VisionAgent MCP server capabilities."""
    
    print("VisionAgent MCP Server Demo")
    print("=" * 40)
    
    try:
        # Create server
        print("1. Creating VisionAgent MCP server...")
        server = create_vision_agent_mcp_server()
        print(f"   ✅ Server created: {type(server).__name__}")
        
        # Show agent info
        print("\n2. VisionAgent Information:")
        agent = server.vision_agent
        print(f"   Agent ID: {agent.agent_id}")
        print(f"   Model: {agent.metadata.get('model', 'default')}")
        print(f"   Supported formats: {agent.metadata.get('supported_formats', [])}")
        print(f"   Capabilities: {agent.metadata.get('capabilities', [])}")
        
        # Show MCP server info
        print("\n3. MCP Server Information:")
        print(f"   MCP Type: {type(server.mcp).__name__}")
        print(f"   Port: 8003")
        
        # Show available tools
        print("\n4. Available MCP Tools:")
        if hasattr(server.mcp, 'tools'):
            for tool_name, tool_info in server.mcp.tools.items():
                print(f"   • {tool_name}: {tool_info.get('description', 'No description')}")
        else:
            print("   ⚠️  Mock MCP server - tools not accessible")
        
        # Test capabilities tool
        print("\n5. Testing capabilities tool...")
        try:
            # Get capabilities
            capabilities = server.mcp.tools.get("get_vision_agent_capabilities", {})
            if capabilities:
                print("   ✅ Capabilities tool available")
            else:
                print("   ⚠️  Capabilities tool not accessible in mock mode")
        except Exception as e:
            print(f"   ⚠️  Capabilities test: {e}")
        
        print("\n🎉 VisionAgent MCP Server demo completed!")
        print("\nTo use this server:")
        print("1. Install FastMCP: pip install fastmcp")
        print("2. Run: python src/mcp/vision_agent_server.py")
        print("3. Server will start on port 8003")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(demo_vision_agent_mcp_server())
    
    if success:
        print("\n✅ Demo completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Demo failed!")
        sys.exit(1)
