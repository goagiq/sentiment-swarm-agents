#!/usr/bin/env python3
"""
Test script to fix the process_content tool options parameter issue.
This demonstrates the correct way to call the process_content tool.
"""

import asyncio
import json


async def test_process_content_fix():
    """Test the process_content tool with correct parameter types."""
    
    # Your query about language and cultural context
    query = ("How do language and cultural context affect strategic "
             "communication and negotiation?")
    
    print(f"🔍 Testing process_content with query: {query}")
    
    try:
        # Import the MCP tool function
        from mcp_Sentiment import process_content
        
        # Call the process_content tool with correct parameter types
        result = await process_content(
            content=query,
            content_type="text",
            language="en",
            options=None  # Use None instead of empty dict or other types
        )
        
        print("✅ Success! Result:")
        print(json.dumps(result, indent=2, default=str))
        
    except ImportError:
        print("❌ MCP tool not available. Let's try alternative approach...")
        
        # Alternative: Use the orchestrator directly
        try:
            from src.core.orchestrator import SentimentOrchestrator
            from src.core.models import AnalysisRequest, DataType
            
            print("🔄 Using orchestrator directly...")
            
            orchestrator = SentimentOrchestrator()
            
            # Create analysis request
            request = AnalysisRequest(
                data_type=DataType.TEXT,
                content=query,
                language="en",
                metadata={"source": "test_script"}
            )
            
            # Process the request
            result = await orchestrator.process(request)
            
            print("✅ Success! Result:")
            print(json.dumps(result.metadata, indent=2, default=str))
            
        except Exception as e:
            print(f"❌ Error with orchestrator: {e}")
            
            # Final fallback: Use the enhanced process content agent
            try:
                from src.agents.enhanced_process_content_agent import (
                    EnhancedProcessContentAgent
                )
                
                print("🔄 Using enhanced process content agent...")
                
                agent = EnhancedProcessContentAgent()
                
                # Call the method on the agent instance
                result = await agent.process_content(
                    content=query,
                    content_type="text",
                    language="en",
                    options=None
                )
                
                print("✅ Success! Result:")
                print(json.dumps(result, indent=2, default=str))
                
            except Exception as e2:
                print(f"❌ Error with enhanced agent: {e2}")
                
                # Try calling it as a bound method
                try:
                    print("🔄 Trying bound method call...")
                    
                    # Get the bound method
                    bound_method = agent.process_content.__get__(agent, type(agent))
                    result = await bound_method(
                        content=query,
                        content_type="text",
                        language="en",
                        options=None
                    )
                    
                    print("✅ Success with bound method! Result:")
                    print(json.dumps(result, indent=2, default=str))
                    
                except Exception as e3:
                    print(f"❌ Error with bound method: {e3}")
                    print("\n🔧 Troubleshooting steps:")
                    print("1. Check if the MCP server is running on port 8003")
                    print("2. Verify the process_content tool is properly registered")
                    print("3. Ensure the options parameter is passed as None or valid Dict")
                    print("4. Check the server logs for more detailed error info")


if __name__ == "__main__":
    asyncio.run(test_process_content_fix())
