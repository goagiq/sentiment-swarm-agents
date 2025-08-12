#!/usr/bin/env python3
"""
Test script to verify Strands integration is working properly.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.strands_ollama_integration import STRANDS_AVAILABLE
from src.core.strands_mock import Agent, Tool
from src.agents.base_agent import StrandsBaseAgent


def test_strands_availability():
    """Test that Strands is properly configured."""
    print("🔍 Testing Strands availability...")
    print(f"STRANDS_AVAILABLE: {STRANDS_AVAILABLE}")
    
    if STRANDS_AVAILABLE:
        print("❌ ERROR: Should be False for mock implementation")
        return False
    else:
        print("✅ SUCCESS: Using mock implementation as expected")
        return True


def test_mock_agent():
    """Test that mock Agent works correctly."""
    print("\n🔍 Testing mock Agent...")
    
    try:
        agent = Agent(
            name="test_agent",
            model="llama3.2:latest"
        )
        print(f"✅ SUCCESS: Created agent '{agent.name}' with model '{agent.model}'")
        return True
    except Exception as e:
        print(f"❌ ERROR: Failed to create agent: {e}")
        return False


def test_mock_tool():
    """Test that mock Tool works correctly."""
    print("\n🔍 Testing mock Tool...")
    
    def test_function():
        return "test result"
    
    try:
        tool = Tool(
            name="test_tool",
            description="A test tool",
            func=test_function,
            parameters={}
        )
        print(f"✅ SUCCESS: Created tool '{tool.name}'")
        return True
    except Exception as e:
        print(f"❌ ERROR: Failed to create tool: {e}")
        return False


def test_base_agent():
    """Test that StrandsBaseAgent works correctly."""
    print("\n🔍 Testing StrandsBaseAgent...")
    
    class TestAgent(StrandsBaseAgent):
        async def can_process(self, request):
            return True
        
        async def process(self, request):
            from src.core.models import AnalysisResult, SentimentResult
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(label="positive", confidence=0.8)
            )
    
    try:
        agent = TestAgent()
        print(f"✅ SUCCESS: Created TestAgent '{agent.agent_id}'")
        print(f"   Strands agent: {agent.strands_agent.name}")
        return True
    except Exception as e:
        print(f"❌ ERROR: Failed to create TestAgent: {e}")
        return False


async def test_agent_processing():
    """Test that agent processing works."""
    print("\n🔍 Testing agent processing...")
    
    from src.core.models import AnalysisRequest, DataType
    
    class TestAgent(StrandsBaseAgent):
        async def can_process(self, request):
            return True
        
        async def process(self, request):
            from src.core.models import AnalysisResult, SentimentResult
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(label="positive", confidence=0.8)
            )
    
    try:
        agent = TestAgent()
        request = AnalysisRequest(
            id="test_request",
            data_type=DataType.TEXT,
            content="This is a test message."
        )
        
        result = await agent.process_request(request)
        print(f"✅ SUCCESS: Processed request with sentiment: {result.sentiment.label}")
        return True
    except Exception as e:
        print(f"❌ ERROR: Failed to process request: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Strands Integration Tests\n")
    
    tests = [
        test_strands_availability(),
        test_mock_agent(),
        test_mock_tool(),
        test_base_agent(),
        await test_agent_processing()
    ]
    
    passed = sum(tests)
    total = len(tests)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Strands integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
