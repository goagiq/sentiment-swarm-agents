#!/usr/bin/env python3
"""
Demonstration script for TextAgent sentiment analysis.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agents.text_agent import TextAgent
from src.core.models import AnalysisRequest, DataType


async def demo_text_agent():
    """Demonstrate TextAgent functionality."""
    print("🚀 Starting TextAgent Demo")
    print("=" * 50)
    
    # Initialize the TextAgent
    print("📝 Initializing TextAgent...")
    agent = TextAgent()
    print(f"✅ Agent initialized: {agent.agent_id}")
    tools = [tool.__name__ for tool in agent._get_tools()]
    print(f"🔧 Available tools: {tools}")
    print()
    
    # Test texts with different sentiments
    test_texts = [
        "I absolutely love this product! It's amazing and wonderful!",
        "This is terrible. I hate it so much. Worst experience ever.",
        "The product is okay. It works as expected.",
        "I'm feeling great today! Everything is going perfectly!",
        "I'm so disappointed with the service. It's awful."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"📊 Test {i}: Analyzing sentiment")
        print(f"   Text: {text}")
        
        # Create analysis request
        request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=text,
            language="en"
        )
        
        # Check if agent can process
        can_process = await agent.can_process(request)
        print(f"   Can process: {can_process}")
        
        if can_process:
            try:
                # Process the request
                print("   🔄 Processing...")
                result = await agent.process(request)
                
                # Display results
                print(f"   ✅ Sentiment: {result.sentiment.label}")
                print(f"   📈 Confidence: {result.sentiment.confidence:.2f}")
                print(f"   ⏱️  Processing time: {result.processing_time:.2f}s")
                print(f"   🏷️  Status: {result.status}")
                
                # Show scores if available
                if result.sentiment.scores:
                    print("   📊 Scores:")
                    for sentiment, score in result.sentiment.scores.items():
                        print(f"      {sentiment}: {score:.2f}")
                
                # Show metadata
                if result.metadata:
                    method = result.metadata.get('method', 'unknown')
                    print(f"   🔍 Method: {method}")
                    tools_used = result.metadata.get('tools_used', [])
                    print(f"   🛠️  Tools used: {tools_used}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("-" * 50)
    
    print("🎉 Demo completed!")
    print(f"📊 Agent status: {agent.get_status()}")


async def demo_social_media():
    """Demonstrate TextAgent with social media content."""
    print("\n📱 Social Media Demo")
    print("=" * 50)
    
    agent = TextAgent()
    
    # Social media post
    social_post = {
        "text": "Just had the best coffee ever! ☕️ This place is fantastic!",
        "platform": "twitter",
        "user": "coffee_lover",
        "likes": 42
    }
    
    request = AnalysisRequest(
        data_type=DataType.SOCIAL_MEDIA,
        content=social_post,
        language="en"
    )
    
    print(f"📝 Social media post: {social_post['text']}")
    print(f"📱 Platform: {social_post['platform']}")
    
    can_process = await agent.can_process(request)
    print(f"✅ Can process social media: {can_process}")
    
    if can_process:
        try:
            result = await agent.process(request)
            print(f"🎯 Sentiment: {result.sentiment.label}")
            print(f"📈 Confidence: {result.sentiment.confidence:.2f}")
            print(f"🔍 Extracted text: {result.extracted_text}")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("TextAgent Sentiment Analysis Demo")
    print("This script demonstrates the TextAgent's capabilities")
    print()
    
    try:
        # Run the main demo
        asyncio.run(demo_text_agent())
        
        # Run social media demo
        asyncio.run(demo_social_media())
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)
