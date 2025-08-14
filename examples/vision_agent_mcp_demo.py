"""
Demo script for VisionAgent MCP Server using Unified MCP Server.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from src.mcp_servers.unified_mcp_server import create_unified_mcp_server


async def demo_vision_agent_mcp_server():
    """Demonstrate the VisionAgent MCP server capabilities using unified server."""
    
    logger.info("VisionAgent MCP Server Demo (Unified)")
    logger.info("=" * 40)
    
    try:
        # Create unified server
        logger.info("1. Creating Unified MCP server...")
        server = create_unified_mcp_server()
        logger.info(f"   ✅ Server created: {type(server).__name__}")
        
        # Show agent info
        logger.info("\n2. VisionAgent Information:")
        agent = server.vision_agent
        logger.info(f"   Agent ID: {agent.agent_id}")
        logger.info(f"   Model: {agent.metadata.get('model', 'default')}")
        logger.info(f"   Supported formats: {agent.metadata.get('supported_formats', [])}")
        logger.info(f"   Capabilities: {agent.metadata.get('capabilities', [])}")
        
        # Show MCP server info
        logger.info("\n3. MCP Server Information:")
        if server.mcp:
            logger.info(f"   MCP Type: {type(server.mcp).__name__}")
            logger.info(f"   Server Name: {server.mcp.name}")
            logger.info(f"   Version: {server.mcp.version}")
        else:
            logger.info("   ⚠️  Mock MCP server - FastMCP not available")
        
        # Show available tools
        logger.info("\n4. Available MCP Tools:")
        if server.mcp and hasattr(server.mcp, 'tools'):
            for tool_name, tool_info in server.mcp.tools.items():
                desc = tool_info.get('description', 'No description')
                logger.info(f"   • {tool_name}: {desc}")
        else:
            logger.info("   ⚠️  Mock MCP server - tools not accessible")
        
        # Test capabilities tool
        logger.info("\n5. Testing capabilities tool...")
        try:
            # Test the unified content processing tool
            if server.mcp and hasattr(server.mcp, 'tools'):
                process_tool = server.mcp.tools.get("process_content", {})
                if process_tool:
                    logger.info("   ✅ Content processing tool available")
                    logger.info("   📝 Can process images through unified interface")
                else:
                    logger.info("   ⚠️  Content processing tool not found")
            
            # Test vision agent directly
            logger.info("   ✅ Vision Agent initialized successfully")
            logger.info(f"   ✅ Agent ID: {agent.agent_id}")
            logger.info(f"   ✅ Model: {agent.metadata.get('model', 'default')}")
            
        except Exception as e:
            logger.warning(f"   ⚠️  Capabilities test: {e}")
        
        logger.info("\n🎉 VisionAgent MCP Server demo completed!")
        logger.info("\nTo use this server:")
        logger.info("1. Install FastMCP: pip install fastmcp")
        logger.info("2. Run: python main.py")
        logger.info("3. Server will start on port 8000")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level="INFO"
    )
    
    success = asyncio.run(demo_vision_agent_mcp_server())
    
    if success:
        logger.info("\n✅ Demo completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n❌ Demo failed!")
        sys.exit(1)
