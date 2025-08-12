#!/usr/bin/env python3
"""
Example script demonstrating proper MCP tools integration with Strands.
This follows the official Strands documentation for MCP tool setup.
"""

import asyncio
from typing import List
from loguru import logger

# Import the proper MCP client
from src.core.strands_mcp_client import (
    create_mcp_agent, 
    create_mcp_swarm, 
    run_mcp_agent, 
    run_mcp_swarm,
    strands_mcp_client
)
from strands.types.content import ContentBlock


async def example_mcp_agent_usage():
    """Example of using MCP tools with a single agent."""
    logger.info("=== MCP Agent Example ===")
    
    try:
        # Create an agent with MCP tools
        agent = create_mcp_agent(
            name="sentiment_analyzer",
            system_prompt="You are a sentiment analysis expert with access to MCP tools."
        )
        
        # Check available tools
        tools = strands_mcp_client.get_available_tools()
        logger.info(f"Available MCP tools: {len(tools)}")
        for tool in tools:
            logger.info(f"  - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}")
        
        # Run the agent
        prompt = "Analyze the sentiment of this text: 'I love this product, it's amazing!'"
        result = await run_mcp_agent("sentiment_analyzer", prompt)
        
        logger.info(f"Agent result: {result}")
        
    except Exception as e:
        logger.error(f"Error in MCP agent example: {e}")


async def example_mcp_swarm_usage():
    """Example of using MCP tools with a swarm of agents."""
    logger.info("=== MCP Swarm Example ===")
    
    try:
        # Create specialized agents for different tasks
        image_analyzer = create_mcp_agent(
            name="image_analyzer", 
            system_prompt="You are an image analysis expert with access to MCP tools."
        )
        
        report_writer = create_mcp_agent(
            name="report_writer", 
            system_prompt="You are a report writing expert with access to MCP tools."
        )
        
        # Create the swarm
        swarm = create_mcp_swarm([image_analyzer, report_writer], name="analysis_swarm")
        
        # Create content blocks with text and image
        content_blocks = [
            ContentBlock(text="Analyze this image and create a report about what you see:"),
            # Note: In a real scenario, you would add image content here
            # ContentBlock(image={"format": "png", "source": {"bytes": image_bytes}}),
        ]
        
        # Execute the swarm with multi-modal input
        result = await run_mcp_swarm("analysis_swarm", content_blocks)
        
        logger.info(f"Swarm result: {result}")
        
    except Exception as e:
        logger.error(f"Error in MCP swarm example: {e}")


async def example_direct_mcp_client_usage():
    """Example of using the MCP client directly."""
    logger.info("=== Direct MCP Client Example ===")
    
    try:
        # Get tools directly from MCP server
        tools = strands_mcp_client.get_tools_sync()
        logger.info(f"Retrieved {len(tools)} tools from MCP server")
        
        # Display tool information
        for i, tool in enumerate(tools, 1):
            logger.info(f"Tool {i}:")
            logger.info(f"  Name: {tool.get('name', 'Unknown')}")
            logger.info(f"  Description: {tool.get('description', 'No description')}")
            logger.info(f"  Input Schema: {tool.get('inputSchema', 'No schema')}")
            logger.info("")
        
        # Get agent status
        status = strands_mcp_client.get_agent_status()
        logger.info(f"Agent status: {status}")
        
    except Exception as e:
        logger.error(f"Error in direct MCP client example: {e}")


async def main():
    """Main function to run all examples."""
    logger.info("Starting MCP Tools Examples")
    
    try:
        # Test direct MCP client usage first
        await example_direct_mcp_client_usage()
        
        # Test agent usage
        await example_mcp_agent_usage()
        
        # Test swarm usage
        await example_mcp_swarm_usage()
        
        logger.success("All MCP tools examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())
