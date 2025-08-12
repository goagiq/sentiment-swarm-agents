#!/usr/bin/env python3
"""
Simple test to verify main.py integration is working correctly.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.models import AnalysisRequest, DataType
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from agents.file_extraction_agent import FileExtractionAgent

async def test_main_integration():
    """Test that the main integration is working correctly."""
    print("🚀 Testing Main Integration...")
    
    # Initialize agents
    kg_agent = KnowledgeGraphAgent()
    file_agent = FileExtractionAgent()
    
    # Test with English text (simulating PDF content)
    english_text = """
    John Smith works at Microsoft Corporation in Seattle. 
    He is a software engineer who develops AI applications.
    Microsoft is headquartered in Redmond, Washington.
    """
    
    # Create analysis request
    request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=english_text,
        language="en"
    )
    
    try:
        # Process with knowledge graph agent
        result = await kg_agent.process(request)
        
        # Check results
        if result.metadata:
            stats = result.metadata.get("statistics", {})
            graph_data = stats.get("graph_data", {})
            
            print("✅ Main Integration Results:")
            print(f"   - Entities found: {stats.get('entities_found', 0)}")
            print(f"   - Relationships mapped: {stats.get('relationships_found', 0)}")
            print(f"   - Graph nodes: {graph_data.get('nodes', 0)}")
            print(f"   - Graph edges: {graph_data.get('edges', 0)}")
            
            # Verify processing worked
            if stats.get('entities_found', 0) > 0 and graph_data.get('edges', 0) > 0:
                print("✅ Main integration is working correctly!")
                return True
            else:
                print("❌ Main integration failed!")
                return False
        else:
            print("❌ No metadata returned from processing!")
            return False
            
    except Exception as e:
        print(f"❌ Main integration failed with error: {e}")
        return False

async def test_russian_integration():
    """Test Russian processing integration."""
    print("\n🇷🇺 Testing Russian Integration...")
    
    # Initialize knowledge graph agent
    kg_agent = KnowledgeGraphAgent()
    
    # Test with Russian text (simulating PDF content)
    russian_text = """
    Владимир Путин является президентом России.
    Москва является столицей Российской Федерации.
    Газпром является крупнейшей энергетической компанией России.
    """
    
    # Create analysis request
    request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=russian_text,
        language="ru"
    )
    
    try:
        # Process with knowledge graph agent
        result = await kg_agent.process(request)
        
        # Check results
        if result.metadata:
            stats = result.metadata.get("statistics", {})
            graph_data = stats.get("graph_data", {})
            
            print("✅ Russian Integration Results:")
            print(f"   - Entities found: {stats.get('entities_found', 0)}")
            print(f"   - Relationships mapped: {stats.get('relationships_found', 0)}")
            print(f"   - Graph nodes: {graph_data.get('nodes', 0)}")
            print(f"   - Graph edges: {graph_data.get('edges', 0)}")
            
            # Check language-specific processing
            language_stats = stats.get("language_stats", {})
            if "ru" in language_stats:
                print(f"   - Russian entities: {language_stats['ru']}")
            
            # Verify processing worked
            if stats.get('entities_found', 0) > 0:
                print("✅ Russian integration is working correctly!")
                return True
            else:
                print("❌ Russian integration failed!")
                return False
        else:
            print("❌ No metadata returned from Russian processing!")
            return False
            
    except Exception as e:
        print(f"❌ Russian integration failed with error: {e}")
        return False

async def main():
    """Run the main integration tests."""
    print("🚀 Starting Main Integration Tests")
    print("=" * 50)
    
    # Run tests
    test_results = []
    
    tests = [
        ("Main Integration", test_main_integration),
        ("Russian Integration", test_russian_integration)
    ]
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Main Integration Test Results:")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All main integration tests passed! The system is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    
    if success:
        print("\n✅ Main integration test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Main integration test failed!")
        sys.exit(1)
