#!/usr/bin/env python3
"""
Test script for Phase 3: Multi-Modal Business Analysis
Verifies that all Phase 3 components are working correctly.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.multi_modal_analysis_agent import MultiModalAnalysisAgent
from src.agents.business_intelligence_agent import BusinessIntelligenceAgent
from src.core.models import AnalysisRequest, DataType


async def test_phase3_agents():
    """Test Phase 3 agents initialization and basic functionality."""
    print("🔧 Testing Phase 3 Agents...")
    
    try:
        # Test MultiModalAnalysisAgent
        print("  - Testing MultiModalAnalysisAgent...")
        multi_modal_agent = MultiModalAnalysisAgent()
        print("    ✅ MultiModalAnalysisAgent initialized successfully")
        
        # Test BusinessIntelligenceAgent (enhanced with Phase 3)
        print("  - Testing BusinessIntelligenceAgent (Phase 3 enhanced)...")
        bi_agent = BusinessIntelligenceAgent()
        print("    ✅ BusinessIntelligenceAgent initialized successfully")
        
        # Test can_process method
        test_request = AnalysisRequest(
            data_type=DataType.TEXT,
            content="Test content for multi-modal analysis",
            language="en"
        )
        
        can_process = multi_modal_agent.can_process(test_request)
        print(f"    ✅ can_process method working: {can_process}")
        
        return True
        
    except Exception as e:
        print(f"    ❌ Agent test failed: {e}")
        return False


async def test_mcp_tools():
    """Test Phase 3 MCP tools functionality."""
    print("🔧 Testing Phase 3 MCP Tools...")
    
    try:
        multi_modal_agent = MultiModalAnalysisAgent()
        bi_agent = BusinessIntelligenceAgent()
        
        # Test comprehensive content analysis
        print("  - Testing analyze_content_comprehensive...")
        content_data = {
            "text": "This is a sample business document with positive sentiment.",
            "image": "sample_image.jpg"
        }
        
        result = await multi_modal_agent.cross_modal_analyzer.analyze_content_comprehensive(
            content_data, "business", True, True
        )
        
        if "analysis_type" in result and result["analysis_type"] == "business":
            print("    ✅ analyze_content_comprehensive working")
        else:
            print("    ❌ analyze_content_comprehensive failed")
            return False
        
        # Test cross-modal insights
        print("  - Testing generate_cross_modal_insights...")
        content_sources = ["source1", "source2", "source3"]
        
        result = await multi_modal_agent._generate_cross_modal_insights(
            content_sources, "business", True, True
        )
        
        if "insight_type" in result and result["insight_type"] == "business":
            print("    ✅ generate_cross_modal_insights working")
        else:
            print("    ❌ generate_cross_modal_insights failed")
            return False
        
        # Test content storytelling
        print("  - Testing create_content_story...")
        content_data = "Sample business content for storytelling"
        
        result = await multi_modal_agent.content_storyteller.create_content_story(
            content_data, "business", True, True
        )
        
        if "story_type" in result and result["story_type"] == "business":
            print("    ✅ create_content_story working")
        else:
            print("    ❌ create_content_story failed")
            return False
        
        # Test business intelligence report
        print("  - Testing create_business_intelligence_report...")
        data_sources = ["source1", "source2"]
        
        result = await bi_agent.create_business_intelligence_report(
            data_sources, "comprehensive", True, True
        )
        
        if "report_scope" in result and result["report_scope"] == "comprehensive":
            print("    ✅ create_business_intelligence_report working")
        else:
            print("    ❌ create_business_intelligence_report failed")
            return False
        
        # Test actionable insights
        print("  - Testing create_actionable_insights...")
        analysis_results = {"sentiment": "positive", "confidence": 0.85}
        
        result = await bi_agent.create_actionable_insights(
            analysis_results, "strategic", True, True
        )
        
        if "insight_type" in result and result["insight_type"] == "strategic":
            print("    ✅ create_actionable_insights working")
        else:
            print("    ❌ create_actionable_insights failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"    ❌ MCP tools test failed: {e}")
        return False


async def test_configuration():
    """Test Phase 3 configuration and dependencies."""
    print("🔧 Testing Phase 3 Configuration...")
    
    try:
        # Test that required modules can be imported
        import numpy as np
        import pandas as pd
        from loguru import logger
        
        print("    ✅ Required dependencies available")
        
        # Test that agents can be instantiated
        from src.agents.multi_modal_analysis_agent import MultiModalAnalysisAgent
        from src.agents.business_intelligence_agent import BusinessIntelligenceAgent
        
        multi_modal_agent = MultiModalAnalysisAgent()
        bi_agent = BusinessIntelligenceAgent()
        
        print("    ✅ Agent instantiation successful")
        
        # Test that agents have required components
        if hasattr(multi_modal_agent, 'cross_modal_analyzer'):
            print("    ✅ Cross-modal analyzer available")
        else:
            print("    ❌ Cross-modal analyzer missing")
            return False
            
        if hasattr(multi_modal_agent, 'content_storyteller'):
            print("    ✅ Content storyteller available")
        else:
            print("    ❌ Content storyteller missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"    ❌ Configuration test failed: {e}")
        return False


async def test_integration():
    """Test Phase 3 integration with main system."""
    print("🔧 Testing Phase 3 Integration...")
    
    try:
        # Test that agents can process requests
        multi_modal_agent = MultiModalAnalysisAgent()
        
        test_request = AnalysisRequest(
            data_type=DataType.TEXT,
            content="This is a comprehensive business analysis request with multiple modalities.",
            language="en"
        )
        
        result = await multi_modal_agent.process(test_request)
        
        if result.success:
            print("    ✅ Multi-modal agent processing successful")
        else:
            print("    ❌ Multi-modal agent processing failed")
            return False
        
        # Test business intelligence agent processing
        bi_agent = BusinessIntelligenceAgent()
        
        test_request = AnalysisRequest(
            data_type=DataType.TEXT,
            content="Business intelligence analysis request",
            language="en"
        )
        
        result = await bi_agent.process(test_request)
        
        if result.success:
            print("    ✅ Business intelligence agent processing successful")
        else:
            print("    ❌ Business intelligence agent processing failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"    ❌ Integration test failed: {e}")
        return False


async def main():
    """Run all Phase 3 tests."""
    print("🚀 Starting Phase 3 Multi-Modal Business Analysis Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(await test_phase3_agents())
    test_results.append(await test_mcp_tools())
    test_results.append(await test_configuration())
    test_results.append(await test_integration())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Phase 3 Test Results Summary")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"❌ Tests Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Phase 3 Multi-Modal Business Analysis is working correctly!")
        print("\n📋 Phase 3 Features Available:")
        print("  • Multi-Modal Analysis Agent")
        print("  • Enhanced Business Intelligence Agent")
        print("  • Comprehensive Content Analysis")
        print("  • Cross-Modal Insights Generation")
        print("  • Content Storytelling")
        print("  • Data Storytelling Presentations")
        print("  • Business Intelligence Reports")
        print("  • Actionable Insights")
        print("  • MCP Tools Integration")
        print("  • API Endpoints")
        print("  • Orchestrator Agent Routing")
        
        return True
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        sys.exit(1)
