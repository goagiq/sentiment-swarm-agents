#!/usr/bin/env python3
"""
Simple test script to fix the process_content tool options parameter issue.
This uses the MCP tool directly without complex fallbacks.
"""

import asyncio
import json


async def simple_process_content_test():
    """Simple test of the process_content tool."""
    
    # Your query about language and cultural context
    query = ("How do language and cultural context affect strategic "
             "communication and negotiation?")
    
    print(f"🔍 Testing process_content with query: {query}")
    
    try:
        # Use the MCP tool directly
        from mcp_Sentiment import process_content
        
        print("🔄 Calling MCP process_content tool...")
        
        # Call with correct parameter types - this is the key fix
        result = await process_content(
            content=query,
            content_type="text", 
            language="en",
            options=None  # Use None instead of empty dict or other types
        )
        
        print("✅ Success! Result:")
        print(json.dumps(result, indent=2, default=str))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 The issue is likely with the 'options' parameter type.")
        print("Make sure to pass 'options=None' instead of an empty dict or other types.")


if __name__ == "__main__":
    asyncio.run(simple_process_content_test())
