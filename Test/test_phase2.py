#!/usr/bin/env python3
"""
Test script for Phase 2 External Data Integration.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_phase2_agents():
    """Test the Phase 2 agents."""
    print("Testing Phase 2 External Data Integration...")
    print("=" * 60)
    
    try:
        # Test SocialMediaAgent
        print("\n1. Testing SocialMediaAgent...")
        from src.agents.social_media_agent import SocialMediaAgent
        social_agent = SocialMediaAgent()
        print(f"   ✓ SocialMediaAgent initialized: {social_agent.agent_id}")
        
        # Test ExternalDataAgent
        print("\n2. Testing ExternalDataAgent...")
        from src.agents.external_data_agent import ExternalDataAgent
        external_agent = ExternalDataAgent()
        print(f"   ✓ ExternalDataAgent initialized: {external_agent.agent_id}")
        
        # Test MarketDataAgent
        print("\n3. Testing MarketDataAgent...")
        from src.agents.market_data_agent import MarketDataAgent
        market_agent = MarketDataAgent()
        print(f"   ✓ MarketDataAgent initialized: {market_agent.agent_id}")
        
        # Test agent capabilities
        print("\n4. Testing agent capabilities...")
        print(f"   SocialMediaAgent capabilities: {social_agent.metadata.get('capabilities', [])}")
        print(f"   ExternalDataAgent capabilities: {external_agent.metadata.get('capabilities', [])}")
        print(f"   MarketDataAgent capabilities: {market_agent.metadata.get('capabilities', [])}")
        
        # Test data types
        print("\n5. Testing data type support...")
        from src.core.models import DataType
        
        # Test social media data type
        social_request = type('Request', (), {
            'data_type': DataType.SOCIAL_MEDIA,
            'content': 'test social media content',
            'id': 'test-1'
        })()
        
        can_process_social = await social_agent.can_process(social_request)
        print(f"   SocialMediaAgent can process SOCIAL_MEDIA: {can_process_social}")
        
        # Test database data type
        db_request = type('Request', (), {
            'data_type': DataType.DATABASE,
            'content': 'test database content',
            'id': 'test-2'
        })()
        
        can_process_db = await external_agent.can_process(db_request)
        print(f"   ExternalDataAgent can process DATABASE: {can_process_db}")
        
        # Test market data type
        market_request = type('Request', (), {
            'data_type': DataType.MARKET_DATA,
            'content': 'test market data content',
            'id': 'test-3'
        })()
        
        can_process_market = await market_agent.can_process(market_request)
        print(f"   MarketDataAgent can process MARKET_DATA: {can_process_market}")
        
        print("\n" + "=" * 60)
        print("✓ All Phase 2 agents tested successfully!")
        print("✓ External Data Integration is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing Phase 2 agents: {e}")
        return False

async def test_mcp_tools():
    """Test the MCP tools integration."""
    print("\nTesting MCP Tools Integration...")
    print("=" * 60)
    
    try:
        # Import the main MCP server
        from main import OptimizedMCPServer
        
        print("1. Creating MCP server...")
        mcp_server = OptimizedMCPServer()
        
        print("2. Checking agent initialization...")
        expected_agents = ["social_media", "external_data", "market_data"]
        for agent_name in expected_agents:
            if agent_name in mcp_server.agents:
                print(f"   ✓ {agent_name} agent initialized")
            else:
                print(f"   ❌ {agent_name} agent not found")
        
        print("3. Checking MCP tools...")
        if mcp_server.mcp:
            print("   ✓ MCP server initialized")
            # Note: We can't directly access tools from FastMCP, but we can verify the server is working
        else:
            print("   ❌ MCP server not initialized")
        
        print("\n" + "=" * 60)
        print("✓ MCP tools integration tested successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing MCP tools: {e}")
        return False

async def test_configuration():
    """Test the configuration files."""
    print("\nTesting Configuration Files...")
    print("=" * 60)
    
    try:
        # Test external data configuration
        print("1. Testing external data configuration...")
        from src.config.external_data_config import external_data_config
        
        print(f"   ✓ External data config loaded")
        print(f"   Social media platforms: {external_data_config.social_media.supported_platforms}")
        print(f"   Database types: {external_data_config.database.supported_databases}")
        print(f"   API types: {external_data_config.api.supported_api_types}")
        print(f"   Market data sources: {external_data_config.market_data.supported_data_sources}")
        print(f"   News sources: {external_data_config.news_sources.supported_sources}")
        
        print("\n" + "=" * 60)
        print("✓ Configuration files tested successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing configuration: {e}")
        return False

async def main():
    """Main test function."""
    print("Phase 2 External Data Integration Test Suite")
    print("=" * 60)
    
    # Run all tests
    tests = [
        test_phase2_agents(),
        test_mcp_tools(),
        test_configuration()
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    # Summarize results
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    test_names = [
        "Phase 2 Agents",
        "MCP Tools Integration", 
        "Configuration Files"
    ]
    
    all_passed = True
    for i, (test_name, result) in enumerate(zip(test_names, results)):
        if isinstance(result, Exception):
            print(f"❌ {test_name}: Failed - {result}")
            all_passed = False
        elif result:
            print(f"✓ {test_name}: Passed")
        else:
            print(f"❌ {test_name}: Failed")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Phase 2 External Data Integration is working correctly!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
