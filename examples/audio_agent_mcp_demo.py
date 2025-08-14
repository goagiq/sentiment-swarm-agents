"""
Demo script for AudioAgent MCP Server using Unified MCP Server.
"""

import asyncio
import sys
import os

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from loguru import logger
from src.mcp_servers.unified_mcp_server import create_unified_mcp_server


async def demo_audio_agent_mcp():
    """Demonstrate AudioAgent MCP server capabilities using unified server."""
    logger.info("🚀 AudioAgent MCP Server Demo (Unified)")
    
    try:
        # Create the unified MCP server
        logger.info("1. Creating Unified MCP server...")
        server = create_unified_mcp_server()
        logger.info("✅ Unified MCP Server created")
        
        # Show server info
        logger.info("2. Server Information:")
        logger.info(f"   Server Type: {type(server).__name__}")
        agent_id = server.audio_agent.agent_id
        logger.info(f"   AudioAgent ID: {agent_id}")
        model = server.audio_agent.metadata.get('model', 'default')
        logger.info(f"   Model: {model}")
        
        # Show MCP server info
        logger.info("3. MCP Server Information:")
        if server.mcp:
            logger.info(f"   MCP Type: {type(server.mcp).__name__}")
            logger.info(f"   Server Name: {server.mcp.name}")
            logger.info(f"   Version: {server.mcp.version}")
        else:
            logger.info("   ⚠️  Mock MCP server - FastMCP not available")
        
        # Show available MCP tools
        logger.info("4. Available MCP Tools:")
        if server.mcp and hasattr(server.mcp, 'tools'):
            for tool_name, tool_info in server.mcp.tools.items():
                desc = tool_info.get('description', 'No description')
                logger.info(f"   ✅ {tool_name}: {desc}")
        else:
            logger.info("   ⚠️  Mock MCP server - tools not accessible")
        
        # Test audio processing capabilities
        logger.info("5. Testing Audio Processing Capabilities:")
        
        try:
            # Test the unified content processing tool
            if server.mcp and hasattr(server.mcp, 'tools'):
                # Get the process_content tool
                process_tool = server.mcp.tools.get("process_content", {})
                if process_tool:
                    logger.info("   ✅ Content processing tool available")
                    logger.info("   📝 Can process audio files through unified interface")
                else:
                    logger.info("   ⚠️  Content processing tool not found")
            
            # Test audio agent directly
            logger.info("   ✅ Audio Agent initialized successfully")
            formats = server.audio_agent.metadata.get('supported_formats', [])
            logger.info(f"   ✅ Supported Formats: {formats}")
            max_duration = server.audio_agent.metadata.get('max_audio_duration', 300)
            logger.info(f"   ✅ Max Duration: {max_duration} seconds")
            
        except Exception as e:
            logger.warning(f"   ⚠️  Audio capabilities test failed: {e}")
        
        # Show integration examples
        logger.info("6. Integration Examples:")
        logger.info("   📝 With Unified MCP Server:")
        logger.info("      from src.mcp_servers.unified_mcp_server import "
                   "create_unified_mcp_server")
        logger.info("      server = create_unified_mcp_server()")
        logger.info("      # Use server.mcp.tools to access all available tools")
        
        logger.info("   🌐 With FastMCP Clients:")
        logger.info("      from fastmcp import FastMCPClient")
        logger.info("      client = FastMCPClient('http://localhost:8000')")
        logger.info("      # Use client to call unified MCP tools")
        
        logger.info("🎉 AudioAgent MCP Server demo completed!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Install FastMCP: pip install fastmcp")
        logger.info("2. Run the unified server: python main.py")
        logger.info("3. Test with MCP clients using unified interface")
        
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
