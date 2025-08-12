#!/usr/bin/env python3
"""
Test script for MCP Business Intelligence tools.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import OptimizedMCPServer


async def test_mcp_business_intelligence():
    """Test MCP business intelligence tools."""
    print("🚀 Testing MCP Business Intelligence Tools")
    print("=" * 50)
    
    try:
        # Create MCP server
        mcp_server = OptimizedMCPServer()
        
        if mcp_server.mcp is None:
            print("❌ MCP server not available")
            return False
        
        print("✅ MCP server initialized successfully")
        
        # Test business dashboard generation
        print("\n🔧 Testing Business Dashboard Generation...")
        try:
            # This would normally be called through MCP, but we'll test the agent directly
            bi_agent = mcp_server.agents["business_intelligence"]
            result = await bi_agent.generate_business_dashboard("test data", "executive")
            
            if result and not result.get("error"):
                print("    ✅ Business dashboard generation successful")
            else:
                print("    ❌ Business dashboard generation failed")
                return False
        except Exception as e:
            print(f"    ❌ Business dashboard test failed: {e}")
            return False
        
        # Test data visualization
        print("\n🔧 Testing Data Visualization...")
        try:
            viz_agent = mcp_server.agents["data_visualization"]
            result = await viz_agent.generate_visualizations("test data", ["trend", "pie"])
            
            if result and not result.get("error"):
                print("    ✅ Data visualization successful")
            else:
                print("    ❌ Data visualization failed")
                return False
        except Exception as e:
            print(f"    ❌ Data visualization test failed: {e}")
            return False
        
        # Test executive reporting
        print("\n🔧 Testing Executive Reporting...")
        try:
            result = await bi_agent.generate_executive_report("test business data", "comprehensive")
            
            if result and not result.get("error"):
                print("    ✅ Executive reporting successful")
            else:
                print("    ❌ Executive reporting failed")
                return False
        except Exception as e:
            print(f"    ❌ Executive reporting test failed: {e}")
            return False
        
        # Test trend analysis
        print("\n🔧 Testing Trend Analysis...")
        try:
            result = await bi_agent.analyze_business_trends("test trend data", "30d")
            
            if result and not result.get("error"):
                print("    ✅ Trend analysis successful")
            else:
                print("    ❌ Trend analysis failed")
                return False
        except Exception as e:
            print(f"    ❌ Trend analysis test failed: {e}")
            return False
        
        print("\n" + "=" * 50)
        print("✅ All MCP Business Intelligence tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ MCP Business Intelligence test failed: {e}")
        return False


async def main():
    """Main test function."""
    success = await test_mcp_business_intelligence()
    
    if success:
        print("🎉 MCP Business Intelligence integration successful!")
        return 0
    else:
        print("⚠️ MCP Business Intelligence integration failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
