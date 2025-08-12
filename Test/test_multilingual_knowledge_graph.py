"""
Test file for multilingual knowledge graph functionality.
Tests language detection, entity extraction, and basic multilingual support.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.core.models import AnalysisRequest, DataType


async def test_language_detection():
    """Test language detection with Chinese content."""
    print("🧪 Testing Language Detection...")
    
    # Initialize the knowledge graph agent
    agent = KnowledgeGraphAgent()
    
    # Test Chinese content
    chinese_text = """
    人工智能技术正在快速发展。北京是中国的首都，有很多著名的大学和研究机构。
    清华大学和北京大学都是世界知名的高等学府。人工智能在医疗、教育、交通等领域都有广泛应用。
    """
    
    # Create analysis request with auto language detection
    request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=chinese_text,
        language="auto"  # Let the system detect the language
    )
    
    print(f"📝 Input text: {chinese_text[:100]}...")
    print(f"🌐 Requested language: {request.language}")
    
    # Extract text content (this should trigger language detection)
    text_content = await agent._extract_text_content(request)
    
    print(f"🔍 Detected language: {request.language}")
    print(f"📄 Extracted content length: {len(text_content)}")
    
    # Verify language detection worked
    if request.language == "zh":
        print("✅ Language detection successful - Chinese detected!")
    else:
        print(f"⚠️  Language detection may have failed - detected: {request.language}")
    
    return request.language == "zh"


async def test_english_content():
    """Test with English content to ensure it still works."""
    print("\n🧪 Testing English Content...")
    
    agent = KnowledgeGraphAgent()
    
    english_text = """
    Artificial intelligence technology is developing rapidly. Washington D.C. is the capital of the United States.
    Harvard University and MIT are world-renowned institutions of higher learning. 
    AI has widespread applications in healthcare, education, transportation, and other fields.
    """
    
    request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=english_text,
        language="auto"
    )
    
    print(f"📝 Input text: {english_text[:100]}...")
    print(f"🌐 Requested language: {request.language}")
    
    text_content = await agent._extract_text_content(request)
    
    print(f"🔍 Detected language: {request.language}")
    print(f"📄 Extracted content length: {len(text_content)}")
    
    if request.language == "en":
        print("✅ Language detection successful - English detected!")
    else:
        print(f"⚠️  Language detection may have failed - detected: {request.language}")
    
    return request.language == "en"


async def test_mixed_content():
    """Test with mixed language content."""
    print("\n🧪 Testing Mixed Language Content...")
    
    agent = KnowledgeGraphAgent()
    
    mixed_text = """
    This is a mixed language text. 这是混合语言文本。
    AI technology is advancing rapidly. 人工智能技术正在快速发展。
    We can process both English and Chinese content. 我们可以处理英文和中文内容。
    """
    
    request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=mixed_text,
        language="auto"
    )
    
    print(f"📝 Input text: {mixed_text[:100]}...")
    print(f"🌐 Requested language: {request.language}")
    
    text_content = await agent._extract_text_content(request)
    
    print(f"🔍 Detected language: {request.language}")
    print(f"📄 Extracted content length: {len(text_content)}")
    
    print("✅ Mixed content processing completed!")
    return True


async def test_translation_service_integration():
    """Test that the translation service is properly integrated."""
    print("\n🧪 Testing Translation Service Integration...")
    
    agent = KnowledgeGraphAgent()
    
    # Check if translation service is available
    if hasattr(agent, 'translation_service'):
        print("✅ TranslationService is properly integrated!")
        
        # Test basic translation
        try:
            chinese_text = "你好，世界"
            result = await agent.translation_service.translate_text(chinese_text, target_language="en")
            print(f"🔄 Translation test: '{chinese_text}' -> '{result.translated_text}'")
            print("✅ Translation service is working!")
            return True
        except Exception as e:
            print(f"❌ Translation service test failed: {e}")
            return False
    else:
        print("❌ TranslationService not found in agent!")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Multilingual Knowledge Graph Tests...\n")
    
    tests = [
        ("Language Detection (Chinese)", test_language_detection),
        ("Language Detection (English)", test_english_content),
        ("Mixed Language Content", test_mixed_content),
        ("Translation Service Integration", test_translation_service_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with error: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Phase 1 implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(main())
