#!/usr/bin/env python3
"""
Simple test script to verify the process_content tool fix.
Tests the specific query that was causing the error.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_process_content_fix():
    """Test the process_content tool with the specific query that was failing."""
    
    # The query that was causing the error
    query = "How do language and cultural context affect strategic communication and negotiation?"
    
    print(f"🔍 Testing process_content with query: {query}")
    print("=" * 80)
    
    try:
        # Try to import the MCP functions
        try:
            from mcp_Sentiment import process_content as mcp_Sentiment_process_content
            print("✅ Successfully imported mcp_Sentiment.process_content")
        except ImportError:
            print("⚠️ mcp_Sentiment import failed, using MCP client fallback")
            from src.core.unified_mcp_client import UnifiedMCPClient
            
            async def mcp_Sentiment_process_content(**kwargs):
                client = UnifiedMCPClient()
                return await client.call_tool("process_content", kwargs)
        
        print("🔄 Calling process_content tool with correct parameters...")
        
        # Call the process_content tool with the correct parameters
        result = await mcp_Sentiment_process_content(
            content=query,
            content_type="text",
            language="en",
            options=None  # ✅ This is the fix - explicitly set to None
        )
        
        print("✅ process_content tool call successful!")
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing process_content: {e}")
        print(f"Error type: {type(e)}")
        return False

async def main():
    """Main test function."""
    print("🚀 Starting process_content tool fix test")
    print("=" * 80)
    
    success = await test_process_content_fix()
    
    if success:
        print("\n✅ Test completed successfully!")
        print("The process_content tool is now working correctly.")
    else:
        print("\n❌ Test failed!")
        print("The process_content tool still has issues.")
    
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
