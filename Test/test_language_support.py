#!/usr/bin/env python3
"""
Test script to verify Russian and Chinese language support.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.config import config
from core.orchestrator import SentimentOrchestrator


async def test_language_support():
    """Test Russian and Chinese language support."""
    print("🧪 Testing Language Support")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = SentimentOrchestrator()
    
    # Test texts in different languages
    test_cases = [
        {
            "language": "ru",
            "text": "Привет, как дела? Это тестовый текст на русском языке.",
            "description": "Russian text"
        },
        {
            "language": "zh", 
            "text": "你好，你好吗？这是中文测试文本。",
            "description": "Chinese text"
        },
        {
            "language": "en",
            "text": "Hello, how are you? This is a test text in English.",
            "description": "English text (control)"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📝 Testing {test_case['description']}")
        print(f"Language: {test_case['language']}")
        print(f"Text: {test_case['text']}")
        
        try:
            # Analyze the text
            result = await orchestrator.analyze_text(
                content=test_case['text'],
                language=test_case['language']
            )
            
            print("✅ Analysis successful!")
            print(f"   Sentiment: {result.sentiment.label}")
            print(f"   Confidence: {result.sentiment.confidence:.2%}")
            print(f"   Processing time: {result.processing_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Language support test completed!")


def test_config_languages():
    """Test that the configuration includes Russian and Chinese."""
    print("\n🔧 Testing Language Configuration")
    print("=" * 50)
    
    # Check if language config is available
    if hasattr(config, 'language'):
        print("✅ Language configuration found")
        
        # Check supported languages
        supported_languages = config.language.get_language_list()
        print(f"📋 Supported languages: {supported_languages}")
        
        # Check specific languages
        required_languages = ["ru", "zh"]
        for lang in required_languages:
            if lang in supported_languages:
                lang_name = config.language.get_language_name(lang)
                print(f"✅ {lang} ({lang_name}) is supported")
            else:
                print(f"❌ {lang} is NOT supported")
    else:
        print("❌ Language configuration not found")


if __name__ == "__main__":
    print("🚀 Starting Language Support Tests")
    
    # Test configuration first
    test_config_languages()
    
    # Test actual analysis
    asyncio.run(test_language_support())
