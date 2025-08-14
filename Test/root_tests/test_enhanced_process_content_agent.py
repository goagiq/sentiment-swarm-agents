#!/usr/bin/env python3
"""
Test script for the Enhanced Process Content Agent with Open Library integration.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
from src.agents.enhanced_process_content_agent import EnhancedProcessContentAgent


async def test_enhanced_process_content_agent():
    """Test the enhanced process content agent with various content types."""
    
    logger.info("🚀 Testing Enhanced Process Content Agent...")
    
    # Initialize the agent
    agent = EnhancedProcessContentAgent()
    
    # Test 1: Open Library URL
    logger.info("\n📚 Test 1: Processing Open Library URL")
    war_and_peace_url = "https://openlibrary.org/books/OL14047767M/Voina_i_mir_%D0%92%D0%9E%D0%99%D0%9D%D0%90_%D0%B8_%D0%9C%D0%98%D0%A0%D0%AA"
    
    try:
        result = await agent.process_content(
            content=war_and_peace_url,
            content_type="auto",
            language="en"
        )
        
        if result["success"]:
            logger.info("✅ Open Library processing successful!")
            logger.info(f"Content type: {result['content_type']}")
            logger.info(f"Sentiment: {result['result']['sentiment']}")
            logger.info(f"Confidence: {result['result']['confidence']}")
            logger.info(f"Processing time: {result['result']['processing_time']:.2f}s")
            
            # Show metadata
            metadata = result['result']['metadata']
            logger.info(f"Title: {metadata.get('title', 'Unknown')}")
            logger.info(f"Vector ID: {metadata.get('vector_id', 'N/A')}")
            logger.info(f"Entities extracted: {metadata.get('entities_count', 0)}")
            logger.info(f"Relationships created: {metadata.get('relationships_count', 0)}")
            logger.info(f"Knowledge graph nodes: {metadata.get('knowledge_graph_nodes', 0)}")
            logger.info(f"Knowledge graph edges: {metadata.get('knowledge_graph_edges', 0)}")
        else:
            logger.error(f"❌ Open Library processing failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"❌ Error testing Open Library processing: {e}")
    
    # Test 2: Direct text content
    logger.info("\n📝 Test 2: Processing direct text content")
    sample_text = """
    This is a sample text for testing the enhanced process content agent.
    It contains positive words like great, excellent, and wonderful.
    The text is about artificial intelligence and machine learning.
    """
    
    try:
        result = await agent.process_content(
            content=sample_text,
            content_type="text",
            language="en"
        )
        
        if result["success"]:
            logger.info("✅ Text processing successful!")
            logger.info(f"Content type: {result['content_type']}")
            logger.info(f"Sentiment: {result['result']['sentiment']}")
            logger.info(f"Confidence: {result['result']['confidence']}")
            logger.info(f"Processing time: {result['result']['processing_time']:.2f}s")
        else:
            logger.error(f"❌ Text processing failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"❌ Error testing text processing: {e}")
    
    # Test 3: Content type detection
    logger.info("\n🔍 Test 3: Content type detection")
    
    test_contents = [
        "https://openlibrary.org/books/OL123456M/Test_Book",
        "https://example.com/document.pdf",
        "https://example.com/audio.mp3",
        "https://example.com/video.mp4",
        "https://example.com/image.jpg",
        "This is plain text content",
        "/path/to/local/file.txt"
    ]
    
    for content in test_contents:
        try:
            result = await agent.detect_content_type(content)
            
            if result["success"]:
                logger.info(f"Content: {content[:50]}...")
                logger.info(f"  Type: {result['content_type']}")
                logger.info(f"  Is Open Library: {result['is_openlibrary']}")
            else:
                logger.error(f"❌ Content type detection failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"❌ Error detecting content type: {e}")
    
    # Test 4: Download Open Library content directly
    logger.info("\n📥 Test 4: Direct Open Library download")
    
    try:
        result = await agent.download_openlibrary_content(war_and_peace_url)
        
        if result["success"]:
            logger.info("✅ Open Library download successful!")
            logger.info(f"Title: {result['title']}")
            logger.info(f"Content length: {result['content_length']} characters")
            logger.info(f"Vector ID: {result['vector_id']}")
            
            # Show extracted metadata
            metadata = result['metadata']
            logger.info(f"Author: {metadata.get('author', 'Unknown')}")
            logger.info(f"Publication year: {metadata.get('publication_year', 'Unknown')}")
            logger.info(f"Genre: {metadata.get('genre', 'Unknown')}")
            logger.info(f"Subjects: {', '.join(metadata.get('subjects', []))}")
        else:
            logger.error(f"❌ Open Library download failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"❌ Error testing Open Library download: {e}")
    
    # Test 5: Text extraction
    logger.info("\n📄 Test 5: Text extraction")
    
    try:
        result = await agent.extract_text_from_content(war_and_peace_url)
        
        if result["success"]:
            logger.info("✅ Text extraction successful!")
            logger.info(f"Title: {result['title']}")
            logger.info(f"Text length: {len(result['text'])} characters")
            logger.info(f"Language: {result['language']}")
            logger.info(f"Text preview: {result['text'][:200]}...")
        else:
            logger.error(f"❌ Text extraction failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"❌ Error testing text extraction: {e}")
    
    # Test 6: Content summarization
    logger.info("\n📋 Test 6: Content summarization")
    
    try:
        result = await agent.summarize_content(war_and_peace_url)
        
        if result["success"]:
            logger.info("✅ Content summarization successful!")
            logger.info(f"Title: {result['title']}")
            logger.info(f"Summary: {result['summary']}")
        else:
            logger.error(f"❌ Content summarization failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"❌ Error testing content summarization: {e}")
    
    # Test 7: Knowledge graph creation
    logger.info("\n🕸️ Test 7: Knowledge graph creation")
    
    sample_content = """
    War and Peace by Leo Tolstoy is a monumental Russian novel published between 1864-1869.
    It tells the story of five aristocratic families during the Napoleonic Wars (1805-1813).
    Key characters include Pierre Bezukhov, Prince Andrei Bolkonsky, and Natasha Rostova.
    The novel explores themes of war and peace, love and relationships, and social class.
    """
    
    try:
        result = await agent.create_knowledge_graph(sample_content, "War and Peace")
        
        if result["success"]:
            logger.info("✅ Knowledge graph creation successful!")
            logger.info(f"Nodes: {result['nodes']}")
            logger.info(f"Edges: {result['edges']}")
            logger.info(f"Entities extracted: {result['entities_count']}")
            logger.info(f"Relationships created: {result['relationships_count']}")
        else:
            logger.error(f"❌ Knowledge graph creation failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"❌ Error testing knowledge graph creation: {e}")
    
    # Test 8: Sentiment analysis
    logger.info("\n😊 Test 8: Sentiment analysis")
    
    positive_text = "This is a wonderful and excellent book that I love reading!"
    negative_text = "This is a terrible and awful book that I hate reading!"
    neutral_text = "This is a book about history and philosophy."
    
    test_texts = [
        ("Positive", positive_text),
        ("Negative", negative_text),
        ("Neutral", neutral_text)
    ]
    
    for sentiment_type, text in test_texts:
        try:
            result = await agent.analyze_sentiment(text)
            
            if result["success"]:
                logger.info(f"✅ {sentiment_type} sentiment analysis successful!")
                logger.info(f"  Detected sentiment: {result['sentiment']}")
                logger.info(f"  Confidence: {result['confidence']}")
                logger.info(f"  Metadata: {result['metadata']}")
            else:
                logger.error(f"❌ {sentiment_type} sentiment analysis failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"❌ Error testing {sentiment_type} sentiment analysis: {e}")
    
    logger.info("\n🎉 Enhanced Process Content Agent testing completed!")


async def main():
    """Main function to run the test."""
    logger.info("🚀 Starting Enhanced Process Content Agent Test Suite...")
    
    await test_enhanced_process_content_agent()
    
    logger.info("✅ Test suite completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
