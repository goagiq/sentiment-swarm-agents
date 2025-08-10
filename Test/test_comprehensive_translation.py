#!/usr/bin/env python3
"""
Test script for comprehensive translation functionality.
Demonstrates automatic translation with analysis, summary, and sentiment analysis.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.translation_agent import TranslationAgent


async def test_comprehensive_translation():
    """Test the comprehensive translation functionality."""
    
    # Sample Chinese text about Taiwan (from the BBC article)
    chinese_text = """
    「八炯」納粹言行爭議:台灣政治圈為何頻觸國際紅線?
    
    近月台灣政壇多次出現納粹相關爭議,分析指這反映出台灣歷史教育的結構性問題,大眾對世界史認識淺薄,與國際社會存在明顯落差。
    
    這場由公民團體發起、執政民進黨支持的罷免行動,試圖翻轉2024年選舉後「朝小野大」的立法院格局,卻以全面潰敗收場。
    
    特朗普宣布對台徵收20%關稅 台灣業者為何表達憂慮
    
    特朗普對台20%關稅的宣布,既是台美經貿談判的階段性結果,也揭開了台灣經濟與區域政治的新挑戰。傳統產業向BBC中文親訴對於關稅的憂慮,未來關稅是否下調或不變,影響台灣經濟也考驗賴清德政府的施政能力。
    """
    
    print("=== Testing Comprehensive Translation ===")
    print(f"Original Chinese text length: {len(chinese_text)} characters")
    print()
    
    try:
        # Initialize translation agent
        agent = TranslationAgent()
        
        print("🔄 Performing comprehensive translation and analysis...")
        print()
        
        # Perform comprehensive translation
        result = await agent.comprehensive_translate_and_analyze(chinese_text, include_analysis=True)
        
        # Display results
        print("✅ Translation Results:")
        print("=" * 50)
        print(f"Source Language: {result['translation']['source_language']}")
        print(f"Model Used: {result['translation']['model_used']}")
        print(f"Confidence: {result['translation']['confidence']:.2f}")
        print(f"Processing Time: {result['translation']['processing_time']:.2f}s")
        print(f"Memory Hit: {result['translation']['translation_memory_hit']}")
        print()
        
        print("📝 Complete Translation:")
        print("=" * 50)
        print(result['translation']['translated_text'])
        print()
        
        # Display sentiment analysis
        if 'sentiment_analysis' in result and 'error' not in result['sentiment_analysis']:
            print("😊 Sentiment Analysis:")
            print("=" * 50)
            print(f"Sentiment: {result['sentiment_analysis']['sentiment']}")
            print(f"Confidence: {result['sentiment_analysis']['confidence']:.2f}")
            print(f"Reasoning: {result['sentiment_analysis']['reasoning']}")
            print()
        else:
            print("⚠️  Sentiment Analysis: Unavailable")
            print()
        
        # Display summary analysis
        if 'summary_analysis' in result and 'error' not in result['summary_analysis']:
            print("📋 Summary Analysis:")
            print("=" * 50)
            print(f"Word Count: {result['summary_analysis']['word_count']}")
            print(f"Key Themes: {', '.join(result['summary_analysis']['key_themes'])}")
            print()
            print("Detailed Summary:")
            print(result['summary_analysis']['summary'])
            print()
        else:
            print("⚠️  Summary Analysis: Unavailable")
            print()
        
        print("✅ Comprehensive translation test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during comprehensive translation: {e}")
        import traceback
        traceback.print_exc()


async def test_simple_translation():
    """Test simple translation without analysis."""
    
    chinese_text = "台灣是一個美麗的島嶼，擁有豐富的文化和歷史。"
    
    print("=== Testing Simple Translation ===")
    print(f"Original text: {chinese_text}")
    print()
    
    try:
        agent = TranslationAgent()
        
        print("🔄 Performing simple translation...")
        
        # Perform simple translation
        result = await agent.comprehensive_translate_and_analyze(chinese_text, include_analysis=False)
        
        print("✅ Simple Translation Result:")
        print("=" * 50)
        print(f"Original: {result['translation']['original_text']}")
        print(f"Translated: {result['translation']['translated_text']}")
        print(f"Source Language: {result['translation']['source_language']}")
        print(f"Processing Time: {result['translation']['processing_time']:.2f}s")
        print()
        
        print("✅ Simple translation test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during simple translation: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function."""
    print("🚀 Starting Comprehensive Translation Tests")
    print("=" * 60)
    print()
    
    # Test simple translation first
    await test_simple_translation()
    print()
    
    # Test comprehensive translation
    await test_comprehensive_translation()
    print()
    
    print("🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
