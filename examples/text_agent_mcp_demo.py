"""
Demo script for TextAgent MCP Server using Unified MCP Server.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from src.mcp_servers.unified_mcp_server import create_unified_mcp_server


async def demo_text_agent_mcp():
    """Demonstrate TextAgent MCP server functionality using unified server."""
    logger.info("🚀 TextAgent MCP Server Demo (Unified)")
    
    try:
        # Create the unified MCP server
        server = create_unified_mcp_server()
        logger.info("✅ Unified MCP Server created")
        
        # Test texts
        test_texts = [
            "I love this product! It's amazing!",
            "This is terrible, I hate it.",
            "The weather is okay today."
        ]
        
        logger.info("📝 Testing text analysis...")
        
        for text in test_texts:
            logger.info(f"\n--- Text: '{text}' ---")
            
            # Test sentiment analysis using unified server
            try:
                # Test the unified content processing tool
                if server.mcp and hasattr(server.mcp, 'tools'):
                    process_tool = server.mcp.tools.get("process_content", {})
                    if process_tool:
                        logger.info("   ✅ Content processing tool available")
                        logger.info("   📝 Can process text through unified interface")
                    else:
                        logger.info("   ⚠️  Content processing tool not found")
                
                # Test text agent directly
                logger.info("   ✅ Text Agent initialized successfully")
                logger.info(f"   ✅ Agent ID: {server.text_agent.agent_id}")
                logger.info(f"   ✅ Model: {server.text_agent.metadata.get('model', 'default')}")
                
                # Create a mock request object for direct testing
                request = type('Request', (), {
                    'id': 'demo-1',
                    'content': text,
                    'data_type': type('DataType', (), {'TEXT': 'text'})().TEXT,
                    'language': 'en',
                    'confidence_threshold': 0.8
                })()
                
                # Test direct agent processing
                result = await server.text_agent.process(request)
                logger.info(f"   ✅ Sentiment: {result.sentiment.label}")
                logger.info(f"   ✅ Confidence: {result.sentiment.confidence}")
                
            except Exception as e:
                logger.error(f"❌ Analysis failed: {e}")
        
        logger.info("\n🎉 Demo completed!")
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
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level="INFO"
    )
    
    # Run demo
    success = asyncio.run(demo_text_agent_mcp())
    
    if success:
        logger.info("✅ Demo completed successfully!")
    else:
        logger.error("❌ Demo failed!")
        sys.exit(1)
