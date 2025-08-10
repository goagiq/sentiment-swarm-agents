#!/usr/bin/env python3
"""
Test script for enhanced audio agent integration with orchestrator agent.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from agents.orchestrator_agent import OrchestratorAgent
from agents.audio_agent_enhanced import EnhancedAudioAgent
from core.models import AnalysisRequest, DataType


async def test_enhanced_audio_agent():
    """Test the enhanced audio agent directly."""
    logger.info("Testing Enhanced Audio Agent...")
    
    try:
        # Create enhanced audio agent
        audio_agent = EnhancedAudioAgent()
        
        # Test audio analysis with a sample audio path
        test_audio_path = "test_audio.mp3"  # This would be a real audio file in practice
        
        # Create analysis request
        request = AnalysisRequest(
            content=test_audio_path,
            data_type=DataType.AUDIO,
            language="en"
        )
        
        # Process the request
        result = await audio_agent.process(request)
        
        logger.info(f"Enhanced Audio Agent Test Results:")
        logger.info(f"  Agent ID: {result.agent_id}")
        logger.info(f"  Sentiment: {result.sentiment.label}")
        logger.info(f"  Confidence: {result.sentiment.confidence}")
        logger.info(f"  Processing Time: {result.processing_time}")
        logger.info(f"  Status: {result.status}")
        logger.info(f"  Method: {result.metadata.get('method', 'unknown')}")
        logger.info(f"  Enhanced Features: {result.metadata.get('enhanced_features', False)}")
        logger.info(f"  Tools Used: {result.metadata.get('tools_used', [])}")
        
        return True
        
    except Exception as e:
        logger.error(f"Enhanced Audio Agent test failed: {e}")
        return False


async def test_orchestrator_with_enhanced_audio():
    """Test the orchestrator agent with enhanced audio capabilities."""
    logger.info("Testing Orchestrator Agent with Enhanced Audio...")
    
    try:
        # Create orchestrator agent
        orchestrator = OrchestratorAgent()
        
        # Test audio analysis through orchestrator
        test_audio_path = "test_audio.mp3"  # This would be a real audio file in practice
        
        # Create analysis request
        request = AnalysisRequest(
            content=test_audio_path,
            data_type=DataType.AUDIO,
            language="en"
        )
        
        # Process the request through orchestrator
        result = await orchestrator.process(request)
        
        logger.info(f"Orchestrator Enhanced Audio Test Results:")
        logger.info(f"  Agent ID: {result.agent_id}")
        logger.info(f"  Sentiment: {result.sentiment.label}")
        logger.info(f"  Confidence: {result.sentiment.confidence}")
        logger.info(f"  Processing Time: {result.processing_time}")
        logger.info(f"  Status: {result.status}")
        logger.info(f"  Method: {result.metadata.get('method', 'unknown')}")
        logger.info(f"  Enhanced Features: {result.metadata.get('enhanced_features', False)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Orchestrator Enhanced Audio test failed: {e}")
        return False


async def test_enhanced_audio_tools():
    """Test individual enhanced audio tools."""
    logger.info("Testing Enhanced Audio Tools...")
    
    try:
        # Create enhanced audio agent
        audio_agent = EnhancedAudioAgent()
        
        # Test audio path
        test_audio_path = "test_audio.mp3"
        
        # Test individual tools
        tools_to_test = [
            ("transcribe_audio_enhanced", audio_agent.transcribe_audio_enhanced),
            ("analyze_audio_sentiment_enhanced", audio_agent.analyze_audio_sentiment_enhanced),
            ("extract_audio_features_enhanced", audio_agent.extract_audio_features_enhanced),
            ("analyze_audio_quality", audio_agent.analyze_audio_quality),
            ("validate_audio_format", audio_agent.validate_audio_format),
            ("get_audio_metadata", audio_agent.get_audio_metadata),
            ("analyze_audio_emotion", audio_agent.analyze_audio_emotion)
        ]
        
        for tool_name, tool_func in tools_to_test:
            try:
                logger.info(f"Testing {tool_name}...")
                
                if tool_name == "transcribe_audio_enhanced":
                    result = await tool_func(test_audio_path)
                elif tool_name == "analyze_audio_sentiment_enhanced":
                    result = await tool_func(test_audio_path)
                elif tool_name == "extract_audio_features_enhanced":
                    result = await tool_func(test_audio_path)
                elif tool_name == "analyze_audio_quality":
                    result = await tool_func(test_audio_path)
                elif tool_name == "validate_audio_format":
                    result = await tool_func(test_audio_path)
                elif tool_name == "get_audio_metadata":
                    result = await tool_func(test_audio_path)
                elif tool_name == "analyze_audio_emotion":
                    result = await tool_func(test_audio_path)
                
                if result.get("status") == "success":
                    logger.info(f"  ✓ {tool_name} - SUCCESS")
                else:
                    logger.warning(f"  ⚠ {tool_name} - FAILED: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"  ✗ {tool_name} - ERROR: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Enhanced Audio Tools test failed: {e}")
        return False


async def test_orchestrator_tools():
    """Test orchestrator tools including enhanced audio."""
    logger.info("Testing Orchestrator Tools...")
    
    try:
        # Create orchestrator agent
        orchestrator = OrchestratorAgent()
        
        # Get available tools
        tools = await orchestrator.get_available_tools()
        
        logger.info(f"Available Orchestrator Tools:")
        for tool in tools:
            logger.info(f"  - {tool['name']}: {tool['description']}")
        
        # Test enhanced audio tool specifically
        test_audio_path = "test_audio.mp3"
        
        # Import the enhanced audio tool function
        from agents.orchestrator_agent import enhanced_audio_sentiment_analysis
        
        logger.info("Testing Enhanced Audio Sentiment Analysis Tool...")
        result = await enhanced_audio_sentiment_analysis(test_audio_path)
        
        if result.get("status") == "success":
            logger.info("  ✓ Enhanced Audio Sentiment Analysis - SUCCESS")
            content = result.get("content", [{}])[0].get("json", {})
            logger.info(f"    Sentiment: {content.get('sentiment', 'unknown')}")
            logger.info(f"    Confidence: {content.get('confidence', 0.0)}")
            logger.info(f"    Method: {content.get('method', 'unknown')}")
            logger.info(f"    Enhanced Features: {content.get('enhanced_features', False)}")
        else:
            logger.warning(f"  ⚠ Enhanced Audio Sentiment Analysis - FAILED: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Orchestrator Tools test failed: {e}")
        return False


async def main():
    """Run all enhanced audio agent integration tests."""
    logger.info("Starting Enhanced Audio Agent Integration Tests")
    logger.info("=" * 50)
    
    test_results = []
    
    # Test 1: Enhanced Audio Agent
    logger.info("\n1. Testing Enhanced Audio Agent")
    result1 = await test_enhanced_audio_agent()
    test_results.append(("Enhanced Audio Agent", result1))
    
    # Test 2: Orchestrator with Enhanced Audio
    logger.info("\n2. Testing Orchestrator with Enhanced Audio")
    result2 = await test_orchestrator_with_enhanced_audio()
    test_results.append(("Orchestrator with Enhanced Audio", result2))
    
    # Test 3: Enhanced Audio Tools
    logger.info("\n3. Testing Enhanced Audio Tools")
    result3 = await test_enhanced_audio_tools()
    test_results.append(("Enhanced Audio Tools", result3))
    
    # Test 4: Orchestrator Tools
    logger.info("\n4. Testing Orchestrator Tools")
    result4 = await test_orchestrator_tools()
    test_results.append(("Orchestrator Tools", result4))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("ENHANCED AUDIO AGENT INTEGRATION TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASSED" if result else "FAILED"
        if result:
            passed += 1
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Enhanced Audio Agent Integration Tests PASSED!")
        return True
    else:
        logger.error(f"❌ {total - passed} Enhanced Audio Agent Integration Tests FAILED!")
        return False


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    
    if success:
        print("\n✅ Enhanced Audio Agent Integration Tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Enhanced Audio Agent Integration Tests failed!")
        sys.exit(1)
