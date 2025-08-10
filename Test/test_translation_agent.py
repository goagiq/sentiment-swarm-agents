#!/usr/bin/env python3
"""
Test script for the Translation Agent functionality.
Tests text, audio, video, and batch translation capabilities.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.translation_agent import TranslationAgent
from core.models import AnalysisRequest, DataType


async def test_text_translation():
    """Test basic text translation functionality."""
    print("🧪 Testing text translation...")
    
    agent = TranslationAgent()
    
    # Test Spanish text
    spanish_text = "Hola, ¿cómo estás? Me llamo Juan y vivo en Madrid."
    request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=spanish_text,
        language="auto"
    )
    
    result = await agent.process(request)
    
    print(f"✅ Spanish translation test:")
    print(f"   Original: {spanish_text}")
    print(f"   Translated: {result.extracted_text}")
    print(f"   Source Language: {result.metadata.get('original_language')}")
    print(f"   Model Used: {result.metadata.get('model_used')}")
    print(f"   Processing Time: {result.processing_time:.2f}s")
    print()
    
    # Test French text
    french_text = "Bonjour le monde! Comment allez-vous aujourd'hui?"
    request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=french_text,
        language="auto"
    )
    
    result = await agent.process(request)
    
    print(f"✅ French translation test:")
    print(f"   Original: {french_text}")
    print(f"   Translated: {result.extracted_text}")
    print(f"   Source Language: {result.metadata.get('original_language')}")
    print(f"   Model Used: {result.metadata.get('model_used')}")
    print(f"   Processing Time: {result.processing_time:.2f}s")
    print()
    
    # Test German text
    german_text = "Guten Tag! Wie geht es Ihnen? Ich freue mich, Sie kennenzulernen."
    request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=german_text,
        language="auto"
    )
    
    result = await agent.process(request)
    
    print(f"✅ German translation test:")
    print(f"   Original: {german_text}")
    print(f"   Translated: {result.extracted_text}")
    print(f"   Source Language: {result.metadata.get('original_language')}")
    print(f"   Model Used: {result.metadata.get('model_used')}")
    print(f"   Processing Time: {result.processing_time:.2f}s")
    print()
    
    # Test English text (should not translate)
    english_text = "Hello world! This is a test of the translation system."
    request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=english_text,
        language="auto"
    )
    
    result = await agent.process(request)
    
    print(f"✅ English text test (should not translate):")
    print(f"   Original: {english_text}")
    print(f"   Result: {result.extracted_text}")
    print(f"   Source Language: {result.metadata.get('original_language')}")
    print(f"   Model Used: {result.metadata.get('model_used')}")
    print(f"   Processing Time: {result.processing_time:.2f}s")
    print()


async def test_webpage_translation():
    """Test webpage translation functionality."""
    print("🌐 Testing webpage translation...")
    
    agent = TranslationAgent()
    
    # Test with a simple webpage (you can replace with actual URLs)
    test_url = "https://httpbin.org/html"
    
    request = AnalysisRequest(
        data_type=DataType.WEBPAGE,
        content=test_url,
        language="auto"
    )
    
    try:
        result = await agent.process(request)
        
        print(f"✅ Webpage translation test:")
        print(f"   URL: {test_url}")
        print(f"   Translated Content: {result.extracted_text[:200]}...")
        print(f"   Source Language: {result.metadata.get('original_language')}")
        print(f"   Model Used: {result.metadata.get('model_used')}")
        print(f"   Processing Time: {result.processing_time:.2f}s")
        print()
        
    except Exception as e:
        print(f"⚠️  Webpage translation test failed: {e}")
        print()


async def test_translation_memory():
    """Test translation memory functionality."""
    print("🧠 Testing translation memory...")
    
    agent = TranslationAgent()
    
    # First translation
    text1 = "Hola, ¿cómo estás?"
    request1 = AnalysisRequest(
        data_type=DataType.TEXT,
        content=text1,
        language="auto"
    )
    
    result1 = await agent.process(request1)
    print(f"✅ First translation:")
    print(f"   Original: {text1}")
    print(f"   Translated: {result1.extracted_text}")
    print(f"   Memory Hit: {result1.metadata.get('translation_memory_hit')}")
    print()
    
    # Similar text (should hit memory)
    text2 = "Hola, ¿cómo estás? ¿Todo bien?"
    request2 = AnalysisRequest(
        data_type=DataType.TEXT,
        content=text2,
        language="auto"
    )
    
    result2 = await agent.process(request2)
    print(f"✅ Second translation (similar text):")
    print(f"   Original: {text2}")
    print(f"   Translated: {result2.extracted_text}")
    print(f"   Memory Hit: {result2.metadata.get('translation_memory_hit')}")
    print()
    
    # Different text (should not hit memory)
    text3 = "Buenos días, señor. ¿Dónde está la biblioteca?"
    request3 = AnalysisRequest(
        data_type=DataType.TEXT,
        content=text3,
        language="auto"
    )
    
    result3 = await agent.process(request3)
    print(f"✅ Third translation (different text):")
    print(f"   Original: {text3}")
    print(f"   Translated: {result3.extracted_text}")
    print(f"   Memory Hit: {result3.metadata.get('translation_memory_hit')}")
    print()


async def test_batch_translation():
    """Test batch translation functionality."""
    print("📦 Testing batch translation...")
    
    agent = TranslationAgent()
    
    # Create multiple requests
    requests = [
        AnalysisRequest(
            data_type=DataType.TEXT,
            content="Bonjour le monde",
            language="auto"
        ),
        AnalysisRequest(
            data_type=DataType.TEXT,
            content="Hola mundo",
            language="auto"
        ),
        AnalysisRequest(
            data_type=DataType.TEXT,
            content="Guten Tag Welt",
            language="auto"
        ),
        AnalysisRequest(
            data_type=DataType.TEXT,
            content="Ciao mondo",
            language="auto"
        )
    ]
    
    results = await agent.batch_translate(requests)
    
    print(f"✅ Batch translation test:")
    print(f"   Total requests: {len(requests)}")
    print(f"   Completed: {len([r for r in results if r.status == 'completed'])}")
    print(f"   Failed: {len([r for r in results if r.status == 'failed'])}")
    print()
    
    for i, result in enumerate(results):
        print(f"   Request {i+1}:")
        print(f"     Original: {requests[i].content}")
        print(f"     Translated: {result.extracted_text}")
        print(f"     Status: {result.status}")
        print(f"     Language: {result.metadata.get('original_language')}")
        print()


async def test_language_detection():
    """Test language detection functionality."""
    print("🔍 Testing language detection...")
    
    agent = TranslationAgent()
    
    test_cases = [
        ("Hola, ¿cómo estás?", "Spanish"),
        ("Bonjour le monde!", "French"),
        ("Guten Tag!", "German"),
        ("Ciao bella!", "Italian"),
        ("Olá mundo!", "Portuguese"),
        ("Привет мир!", "Russian"),
        ("你好世界!", "Chinese"),
        ("こんにちは世界!", "Japanese"),
        ("안녕하세요 세계!", "Korean"),
        ("مرحبا بالعالم!", "Arabic"),
        ("नमस्ते दुनिया!", "Hindi"),
        ("สวัสดีโลก!", "Thai"),
        ("Hello world!", "English")
    ]
    
    for text, expected_lang in test_cases:
        detected_lang = await agent._detect_language(text)
        print(f"✅ {expected_lang}: '{text}' -> {detected_lang}")
    
    print()


async def test_agent_status():
    """Test agent status and capabilities."""
    print("📊 Testing agent status...")
    
    agent = TranslationAgent()
    status = agent.get_status()
    
    print(f"✅ Agent Status:")
    print(f"   Agent ID: {status['agent_id']}")
    print(f"   Status: {status['status']}")
    print(f"   Translation Models: {status['translation_models']}")
    print(f"   Supported Languages: {status['supported_languages']}")
    print(f"   Translation Memory: {status['translation_memory_enabled']}")
    print(f"   Batch Processing: {status['batch_processing_enabled']}")
    print()


async def main():
    """Run all translation tests."""
    print("🚀 Starting Translation Agent Tests")
    print("=" * 50)
    
    try:
        # Test basic functionality
        await test_text_translation()
        await test_translation_memory()
        await test_language_detection()
        await test_batch_translation()
        await test_agent_status()
        
        # Test webpage translation (may fail if no internet)
        await test_webpage_translation()
        
        print("✅ All translation tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
