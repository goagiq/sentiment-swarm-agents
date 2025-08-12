#!/usr/bin/env python3
"""
Final Integration Test Script
Tests the complete integration of edge creation and Russian PDF processing fixes.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.models import AnalysisRequest, DataType
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from agents.entity_extraction_agent import EntityExtractionAgent
from agents.file_extraction_agent import FileExtractionAgent
from config.language_specific_regex_config import (
    get_entity_patterns_for_language, get_relationship_patterns_for_language
)


async def test_edge_creation_fix():
    """Test that edge creation is working correctly."""
    print("\n🔗 Testing Edge Creation Fix...")
    
    # Initialize knowledge graph agent
    kg_agent = KnowledgeGraphAgent()
    
    # Test English text
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
    
    # Process the request
    result = await kg_agent.process(request)
    
    # Check results
    if result.metadata:
        stats = result.metadata.get("statistics", {})
        graph_data = stats.get("graph_data", {})
        
        print("✅ English Processing Results:")
        print(f"   - Entities found: {stats.get('entities_found', 0)}")
        print(f"   - Relationships mapped: {stats.get('relationships_found', 0)}")
        print(f"   - Graph nodes: {graph_data.get('nodes', 0)}")
        print(f"   - Graph edges: {graph_data.get('edges', 0)}")
        
        # Verify edges were created
        if graph_data.get('edges', 0) > 0:
            print("✅ Edge creation is working correctly!")
            return True
        else:
            print("❌ No edges were created!")
            return False
    else:
        print("❌ No metadata returned from processing!")
        return False

async def test_russian_pdf_processing():
    """Test Russian PDF processing with the fixes."""
    print("\n🇷🇺 Testing Russian PDF Processing Fix...")
    
    # Initialize agents
    file_agent = FileExtractionAgent()
    kg_agent = KnowledgeGraphAgent()
    
    # Test Russian text (simulating extracted PDF content)
    russian_text = """
    Владимир Путин является президентом России.
    Москва является столицей Российской Федерации.
    Газпром является крупнейшей энергетической компанией России.
    """
    
    # Create analysis request for Russian text
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
            
            print("✅ Russian Processing Results:")
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
                print("✅ Russian PDF processing is working correctly!")
                return True
            else:
                print("❌ No Russian entities were extracted!")
                return False
        else:
            print("❌ No metadata returned from Russian processing!")
            return False
            
    except Exception as e:
        print(f"❌ Russian processing failed with error: {e}")
        return False

async def test_language_specific_config():
    """Test that language-specific configuration is working."""
    print("\n⚙️ Testing Language-Specific Configuration...")
    
    # Test English patterns
    en_patterns = get_entity_patterns_for_language("en")
    en_relationships = get_relationship_patterns_for_language("en")
    
    print("✅ English Configuration:")
    print(f"   - Entity patterns: {len(en_patterns)} types")
    print(f"   - Relationship patterns: {len(en_relationships)} types")
    
    # Test Russian patterns
    ru_patterns = get_entity_patterns_for_language("ru")
    ru_relationships = get_relationship_patterns_for_language("ru")
    
    print("✅ Russian Configuration:")
    print(f"   - Entity patterns: {len(ru_patterns)} types")
    print(f"   - Relationship patterns: {len(ru_relationships)} types")
    
    # Test Chinese patterns
    zh_patterns = get_entity_patterns_for_language("zh")
    zh_relationships = get_relationship_patterns_for_language("zh")
    
    print("✅ Chinese Configuration:")
    print(f"   - Entity patterns: {len(zh_patterns)} types")
    print(f"   - Relationship patterns: {len(zh_relationships)} types")
    
    # Verify all languages have configurations
    if en_patterns and ru_patterns and zh_patterns:
        print("✅ Language-specific configuration is working correctly!")
        return True
    else:
        print("❌ Missing language-specific configurations!")
        return False

async def test_entity_extraction_agent_fix():
    """Test that the entity extraction agent fix is working."""
    print("\n🔍 Testing Entity Extraction Agent Fix...")
    
    # Initialize entity extraction agent
    entity_agent = EntityExtractionAgent()
    
    # Test with English text
    english_text = "Apple Inc. is headquartered in Cupertino, California. Tim Cook is the CEO."
    
    # Create analysis request
    request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=english_text,
        language="en"
    )
    
    try:
        # Process the request
        result = await entity_agent.process(request)
        
        print(f"✅ Entity Extraction Results:")
        print(f"   - Processing time: {result.processing_time:.2f}s")
        print(f"   - Status: {result.status}")
        
        if result.status == "completed":
            print("✅ Entity extraction agent fix is working correctly!")
            return True
        else:
            print(f"❌ Entity extraction failed with status: {result.status}")
            return False
            
    except Exception as e:
        print(f"❌ Entity extraction failed with error: {e}")
        return False

async def test_comprehensive_integration():
    """Test comprehensive integration of all fixes."""
    print("\n🧪 Testing Comprehensive Integration...")
    
    # Initialize knowledge graph agent
    kg_agent = KnowledgeGraphAgent()
    
    # Test with mixed language content
    mixed_text = """
    English: John Smith works at Microsoft in Seattle.
    Russian: Владимир Путин является президентом России.
    Chinese: 习近平是中国国家主席。
    """
    
    # Create analysis request
    request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=mixed_text,
        language="auto"
    )
    
    try:
        # Process the request
        result = await kg_agent.process(request)
        
        # Check results
        if result.metadata:
            stats = result.metadata.get("statistics", {})
            graph_data = stats.get("graph_data", {})
            
            print("✅ Comprehensive Integration Results:")
            print(f"   - Total entities found: {stats.get('entities_found', 0)}")
            print(f"   - Total relationships mapped: {stats.get('relationships_found', 0)}")
            print(f"   - Graph nodes: {graph_data.get('nodes', 0)}")
            print(f"   - Graph edges: {graph_data.get('edges', 0)}")
            
            # Check language-specific stats
            language_stats = stats.get("language_stats", {})
            for lang, lang_data in language_stats.items():
                print(f"   - {lang.upper()}: {lang_data} entities")
            
            # Verify comprehensive processing worked
            if stats.get('entities_found', 0) > 0 and graph_data.get('edges', 0) > 0:
                print("✅ Comprehensive integration is working correctly!")
                return True
            else:
                print("❌ Comprehensive integration failed!")
                return False
        else:
            print("❌ No metadata returned from comprehensive processing!")
            return False
            
    except Exception as e:
        print(f"❌ Comprehensive integration failed with error: {e}")
        return False

async def main():
    """Run all integration tests."""
    print("🚀 Starting Final Integration Tests")
    print("=" * 50)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Edge Creation Fix", test_edge_creation_fix),
        ("Russian PDF Processing", test_russian_pdf_processing),
        ("Language-Specific Configuration", test_language_specific_config),
        ("Entity Extraction Agent Fix", test_entity_extraction_agent_fix),
        ("Comprehensive Integration", test_comprehensive_integration)
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
    print("📊 Final Integration Test Results:")
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
        print("🎉 All integration tests passed! The fixes are working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    
    if success:
        print("\n✅ Final integration test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Final integration test failed!")
        sys.exit(1)
