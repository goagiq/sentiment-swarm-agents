"""
Demo script for AudioAgent MCP Server.
"""

import asyncio
import sys
import os

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from loguru import logger
from mcp.audio_agent_server import create_audio_agent_mcp_server


async def demo_audio_agent_mcp():
    """Demonstrate AudioAgent MCP server capabilities."""
    logger.info("🚀 AudioAgent MCP Server Demo")
    
    try:
        # Create the MCP server
        logger.info("1. Creating AudioAgent MCP server...")
        server = create_audio_agent_mcp_server()
        logger.info("✅ MCP Server created")
        
        # Show server info
        logger.info("2. Server Information:")
        logger.info(f"   Server Type: {type(server).__name__}")
        logger.info(f"   AudioAgent ID: {server.audio_agent.agent_id}")
        logger.info(f"   Model: {server.audio_agent.metadata.get('model', 'default')}")
        
        # Show MCP server info
        logger.info("3. MCP Server Information:")
        logger.info(f"   MCP Type: {type(server.mcp).__name__}")
        logger.info(f"   Port: 8007")
        
        # Show available MCP tools
        logger.info("4. Available MCP Tools:")
        if hasattr(server.mcp, 'tools'):
            for tool_name, tool_info in server.mcp.tools.items():
                logger.info(f"   ✅ {tool_name}: {tool_info.get('description', 'No description')}")
        else:
            logger.info("   ⚠️  Mock MCP server - tools not accessible")
        
        # Test capabilities
        logger.info("5. Testing Capabilities:")
        if hasattr(server.mcp, 'tools'):
            capabilities = server.mcp.tools.get("get_audio_agent_capabilities", {})
            if capabilities:
                cap_func = capabilities.get("function")
                if cap_func:
                    try:
                        cap_result = cap_func()
                        logger.info(f"   ✅ Agent ID: {cap_result.get('agent_id', 'N/A')}")
                        logger.info(f"   ✅ Model: {cap_result.get('model', 'N/A')}")
                        logger.info(f"   ✅ Supported Formats: {cap_result.get('supported_formats', [])}")
                        logger.info(f"   ✅ Capabilities: {cap_result.get('capabilities', [])}")
                        logger.info(f"   ✅ Available Tools: {cap_result.get('available_tools', [])}")
                    except Exception as e:
                        logger.warning(f"   ⚠️  Capabilities test failed: {e}")
            else:
                logger.info("   ⚠️  Capabilities tool not found")
        else:
            logger.info("   ⚠️  Mock server - capabilities not accessible")
        
        # Show supported audio formats
        logger.info("6. Audio Support:")
        supported_formats = server.audio_agent.metadata.get("supported_formats", [])
        max_duration = server.audio_agent.metadata.get("max_audio_duration", 300)
        logger.info(f"   ✅ Supported Formats: {', '.join(supported_formats)}")
        logger.info(f"   ✅ Max Duration: {max_duration} seconds ({max_duration/60:.1f} minutes)")
        
        # Show integration examples
        logger.info("7. Integration Examples:")
        logger.info("   📝 With Strands Agents:")
        logger.info("      from mcp.audio_agent_server import create_audio_agent_mcp_server")
        logger.info("      server = create_audio_agent_mcp_server()")
        logger.info("      # Use server.run() to start the MCP server")
        
        logger.info("   🌐 With FastMCP Clients:")
        logger.info("      from fastmcp import FastMCPClient")
        logger.info("      client = FastMCPClient('http://localhost:8007')")
        logger.info("      # Use client to call MCP tools")
        
        logger.info("🎉 AudioAgent MCP Server demo completed!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Install FastMCP: pip install fastmcp")
        logger.info("2. Run the server: python src/mcp/audio_agent_server.py")
        logger.info("3. Test with MCP clients")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return False


if __name__ == "__main__":
    # Run the demo
    success = asyncio.run(demo_audio_agent_mcp())
    
    if success:
        logger.info("✅ Demo completed successfully!")
    else:
        logger.error("❌ Demo failed!")
        sys.exit(1)
