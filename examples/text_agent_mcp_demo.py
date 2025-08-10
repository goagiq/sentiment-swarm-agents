"""
Demo script for TextAgent MCP Server.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from mcp.text_agent_server import create_text_agent_mcp_server


async def demo_text_agent_mcp():
    """Demonstrate TextAgent MCP server functionality."""
    logger.info("🚀 TextAgent MCP Server Demo")
    
    # Create the MCP server
    server = create_text_agent_mcp_server()
    logger.info("✅ MCP Server created")
    
    # Test texts
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible, I hate it.",
        "The weather is okay today."
    ]
    
    logger.info("📝 Testing text analysis...")
    
    for text in test_texts:
        logger.info(f"\n--- Text: '{text}' ---")
        
        # Test sentiment analysis
        try:
            # Create a mock request object
            request = type('Request', (), {
                'id': 'demo-1',
                'content': text,
                'data_type': type('DataType', (), {'TEXT': 'text'})().TEXT,
                'language': 'en',
                'confidence_threshold': 0.8
            })()
            
            result = await server.text_agent.process(request)
            logger.info(f"✅ Sentiment: {result.sentiment.label}")
            logger.info(f"✅ Confidence: {result.sentiment.confidence}")
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
    
    logger.info("\n🎉 Demo completed!")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Run demo
    asyncio.run(demo_text_agent_mcp())
