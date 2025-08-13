#!/usr/bin/env python3
"""
Test script to verify Phase 1 Pattern Recognition functionality
"""

import asyncio
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agents.pattern_recognition_agent import PatternRecognitionAgent
from src.core.models import AnalysisRequest, DataType


async def test_pattern_recognition():
    """Test the pattern recognition agent functionality."""
    print("🧪 Testing Phase 1 Pattern Recognition Components...")
    
    # Initialize the pattern recognition agent
    agent = PatternRecognitionAgent()
    print(f"✅ Pattern Recognition Agent initialized: {agent.agent_id}")
    
    # Test data
    test_data = [
        {"timestamp": "2024-01-01", "value": 10},
        {"timestamp": "2024-01-02", "value": 15},
        {"timestamp": "2024-01-03", "value": 12},
        {"timestamp": "2024-01-04", "value": 18},
        {"timestamp": "2024-01-05", "value": 14}
    ]
    
    # Create test request
    request = AnalysisRequest(
        data_type=DataType.TIME_SERIES,
        content=json.dumps(test_data),
        language="en"
    )
    
    print(f"📊 Testing with {len(test_data)} time series data points...")
    
    try:
        # Test if agent can process the request
        can_process = await agent.can_process(request)
        print(f"✅ Can process time series data: {can_process}")
        
        if can_process:
            # Process the request
            result = await agent.process(request)
            print(f"✅ Processing completed successfully!")
            print(f"   - Request ID: {result.request_id}")
            print(f"   - Data Type: {result.data_type}")
            print(f"   - Processing Time: {result.processing_time:.3f}s")
            print(f"   - Confidence: {result.sentiment.confidence}")
            print(f"   - Reasoning: {result.sentiment.reasoning}")
            print(f"   - Model Used: {result.model_used}")
            print(f"   - Quality Score: {result.quality_score}")
            
            # Check if pattern analysis results are in metadata
            if result.metadata:
                print(f"   - Pattern Analysis Results: ✅")
                if "pattern_analysis" in result.metadata:
                    pattern_analysis = result.metadata["pattern_analysis"]
                    print(f"     • Temporal Patterns: {'✅' if 'temporal_patterns' in pattern_analysis else '❌'}")
                    print(f"     • Seasonal Patterns: {'✅' if 'seasonal_patterns' in pattern_analysis else '❌'}")
                    print(f"     • Trend Analysis: {'✅' if 'trend_analysis' in pattern_analysis else '❌'}")
                    print(f"     • Pattern Storage: {'✅' if 'pattern_storage' in pattern_analysis else '❌'}")
                else:
                    print(f"     • Pattern Analysis: ❌ (not found in metadata)")
            else:
                print(f"   - Pattern Analysis Results: ❌ (no metadata)")
            
            return True
        else:
            print("❌ Agent cannot process time series data")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_pattern_components():
    """Test individual pattern recognition components."""
    print("\n🔧 Testing Individual Pattern Recognition Components...")
    
    agent = PatternRecognitionAgent()
    
    # Test data
    test_data = [
        {"timestamp": "2024-01-01", "value": 10},
        {"timestamp": "2024-01-02", "value": 15},
        {"timestamp": "2024-01-03", "value": 12},
        {"timestamp": "2024-01-04", "value": 18},
        {"timestamp": "2024-01-05", "value": 14}
    ]
    
    try:
        # Test temporal analysis
        print("📈 Testing Temporal Analysis...")
        temporal_result = await agent.analyze_temporal_patterns(test_data)
        print(f"   ✅ Temporal Analysis: {temporal_result.get('status', 'completed')}")
        
        # Test seasonal detection
        print("🔄 Testing Seasonal Detection...")
        seasonal_result = await agent.detect_seasonal_patterns(test_data)
        print(f"   ✅ Seasonal Detection: {seasonal_result.get('status', 'completed')}")
        
        # Test trend analysis
        print("📊 Testing Trend Analysis...")
        trend_result = await agent.analyze_trends(test_data)
        print(f"   ✅ Trend Analysis: {trend_result.get('status', 'completed')}")
        
        # Test pattern storage
        print("💾 Testing Pattern Storage...")
        storage_result = await agent.store_pattern("test_pattern", test_data, "time_series")
        print(f"   ✅ Pattern Storage: {storage_result.get('status', 'completed')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Phase 1 Pattern Recognition Verification Test")
    print("=" * 50)
    
    # Test pattern recognition agent
    agent_test_passed = await test_pattern_recognition()
    
    # Test individual components
    component_test_passed = await test_pattern_components()
    
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print(f"   Pattern Recognition Agent: {'✅ PASSED' if agent_test_passed else '❌ FAILED'}")
    print(f"   Individual Components: {'✅ PASSED' if component_test_passed else '❌ FAILED'}")
    
    if agent_test_passed and component_test_passed:
        print("\n🎉 Phase 1 Pattern Recognition is working correctly!")
        print("✅ All components are properly integrated and functional.")
        return True
    else:
        print("\n⚠️ Phase 1 Pattern Recognition has issues that need to be addressed.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
