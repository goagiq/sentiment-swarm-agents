#!/usr/bin/env python3
"""
Corrected test script to fix the process_content tool options parameter issue.
This uses the proper MCP client to call the process_content tool.
"""

import asyncio
import json


async def corrected_process_content_test():
    """Test the process_content tool using the proper MCP client."""
    
    # Your query about language and cultural context
    query = ("How do language and cultural context affect strategic "
             "communication and negotiation?")
    
    print(f"🔍 Testing process_content with query: {query}")
    
    try:
        # Use the proper MCP client
        from src.core.unified_mcp_client import call_unified_mcp_tool
        
        print("🔄 Calling process_content tool via MCP client...")
        
        # Call the process_content tool with correct parameter types
        result = await call_unified_mcp_tool(
            tool_name="process_content",
            parameters={
                "content": query,
                "content_type": "text",
                "language": "en",
                "options": None  # Use None instead of empty dict or other types
            }
        )
        
        print("✅ Success! Result:")
        print(json.dumps(result, indent=2, default=str))
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 The issue is likely with the 'options' parameter type.")
        print("Make sure to pass 'options=None' instead of an empty dict or other types.")


if __name__ == "__main__":
    asyncio.run(corrected_process_content_test())
