"""
Example client for using the Sentiment Analysis MCP server with Strands agents.
"""

import asyncio
from typing import Dict, Any, List
from loguru import logger

# Try to import MCP client components
try:
    from mcp.client.streamable_http import streamablehttp_client
    from src.core.strands_mock import Agent

    # Mock MCPClient for testing
    class MCPClient:
        """Mock MCP client for testing."""
        
        def __init__(self, create_transport):
            self.create_transport = create_transport
            self.tools = []
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        
        async def list_tools_async(self):
            return self.tools

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP client components not available. Install required packages.")


class SentimentMCPClient:
    """Client for interacting with the Sentiment Analysis MCP server."""
    
    def __init__(self, server_url: str = "http://localhost:8001/mcp/"):
        self.server_url = server_url
        self.mcp_client = None
        self.agent = None
        
    async def connect(self):
        """Connect to the MCP server."""
        if not MCP_AVAILABLE:
            logger.error("MCP client not available")
            return False
            
        try:
            # Create MCP client
            def create_transport():
                return streamablehttp_client(self.server_url)
            
            self.mcp_client = MCPClient(create_transport)
            
            # Connect and get tools
            async with self.mcp_client:
                tools = await self.mcp_client.list_tools_async()
                logger.info(f"Connected to MCP server. Available tools: {len(tools)}")
                
                # Create Strands agent with MCP tools
                self.agent = Agent(tools=tools)
                logger.info("Strands agent created with MCP tools")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return False
    
    async def analyze_text_sentiment(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Analyze text sentiment using the MCP server."""
        if not self.agent:
            logger.error("Agent not initialized. Call connect() first.")
            return {"error": "Agent not initialized"}
        
        try:
            prompt = f"Analyze the sentiment of this text: '{text}'"
            response = await self.agent.invoke_async(prompt)
            
            # Extract sentiment information from response
            # This would depend on how the agent formats its response
            return {
                "text": text,
                "language": language,
                "response": str(response),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {e}")
            return {"error": str(e), "text": text}
    
    async def analyze_image_sentiment(self, image_url: str, language: str = "en") -> Dict[str, Any]:
        """Analyze image sentiment using the MCP server."""
        if not self.agent:
            logger.error("Agent not initialized. Call connect() first.")
            return {"error": "Agent not initialized"}
        
        try:
            prompt = f"Analyze the sentiment of this image: {image_url}"
            response = await self.agent.invoke_async(prompt)
            
            return {
                "image_url": image_url,
                "language": language,
                "response": str(response),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image sentiment: {e}")
            return {"error": str(e), "image_url": image_url}
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get available capabilities from the MCP server."""
        if not self.agent:
            logger.error("Agent not initialized. Call connect() first.")
            return {"error": "Agent not initialized"}
        
        try:
            prompt = "What capabilities do you have for sentiment analysis?"
            response = await self.agent.invoke_async(prompt)
            
            return {
                "capabilities": str(response),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error getting capabilities: {e}")
            return {"error": str(e)}
    
    async def batch_analyze_texts(self, texts: List[str], language: str = "en") -> List[Dict[str, Any]]:
        """Analyze multiple texts in batch."""
        results = []
        for text in texts:
            result = await self.analyze_text_sentiment(text, language)
            results.append(result)
        return results


async def demo_mcp_integration():
    """Demonstrate MCP integration with Strands agents."""
    logger.info("🚀 Starting MCP Integration Demo")
    
    # Create client
    client = SentimentMCPClient()
    
    # Connect to server
    if not await client.connect():
        logger.error("Failed to connect to MCP server")
        return
    
    # Test capabilities
    logger.info("📊 Testing capabilities...")
    capabilities = await client.get_capabilities()
    logger.info(f"Capabilities: {capabilities}")
    
    # Test text sentiment analysis
    logger.info("📝 Testing text sentiment analysis...")
    text_samples = [
        "I love this product! It's amazing!",
        "This is terrible, I hate it.",
        "The weather is okay today."
    ]
    
    for text in text_samples:
        result = await client.analyze_text_sentiment(text)
        logger.info(f"Text: '{text}' -> Result: {result}")
    
    # Test batch analysis
    logger.info("🔄 Testing batch analysis...")
    batch_results = await client.batch_analyze_texts(text_samples)
    logger.info(f"Batch results: {batch_results}")
    
    logger.info("✅ MCP Integration Demo completed!")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_mcp_integration())
