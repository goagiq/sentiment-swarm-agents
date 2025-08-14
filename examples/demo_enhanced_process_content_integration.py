#!/usr/bin/env python3
"""
Demo script for Enhanced Process Content Agent with Open Library integration.
Demonstrates the integration following the Design Framework pattern.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
from src.mcp_servers.enhanced_unified_mcp_server import EnhancedUnifiedMCPServer


async def demo_enhanced_process_content_integration():
    """Demonstrate the enhanced process_content agent with Open Library integration."""
    
    logger.info("🚀 Enhanced Process Content Agent Integration Demo")
    logger.info("=" * 60)
    
    # Initialize the enhanced MCP server
    server = EnhancedUnifiedMCPServer()
    
    # Demo 1: Process Open Library URL directly
    logger.info("\n📚 Demo 1: Processing Open Library URL")
    logger.info("-" * 40)
    
    war_and_peace_url = "https://openlibrary.org/books/OL14047767M/Voina_i_mir_%D0%92%D0%9E%D0%99%D0%9D%D0%90_%D0%B8_%D0%9C%D0%98%D0%A0%D0%AA"
    
    try:
        # Use the enhanced process_content tool
        result = await server.mcp.tools["process_content"](
            content=war_and_peace_url,
            content_type="auto",
            language="en"
        )
        
        if result["success"]:
            logger.info("✅ Open Library processing successful!")
            logger.info(f"Content type: {result['content_type']}")
            
            # Show detailed results
            if result['content_type'] == 'open_library':
                logger.info(f"Title: {result['result']['title']}")
                logger.info(f"Vector ID: {result['result']['vector_id']}")
                logger.info(f"Summary ID: {result['result']['summary_id']}")
                logger.info(f"Entities extracted: {result['result']['entities_count']}")
                logger.info(f"Relationships created: {result['result']['relationships_count']}")
                logger.info(f"Knowledge graph nodes: {result['result']['knowledge_graph_nodes']}")
                logger.info(f"Knowledge graph edges: {result['result']['knowledge_graph_edges']}")
                logger.info(f"Content length: {result['result']['content_length']} characters")
                logger.info(f"Summary: {result['result']['summary'][:200]}...")
        else:
            logger.error(f"❌ Open Library processing failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"❌ Error in Demo 1: {e}")
    
    # Demo 2: Extract text from Open Library URL
    logger.info("\n📄 Demo 2: Extract text from Open Library URL")
    logger.info("-" * 40)
    
    try:
        result = await server.mcp.tools["extract_text_from_content"](
            content=war_and_peace_url,
            content_type="auto",
            language="en"
        )
        
        if result["success"]:
            logger.info("✅ Text extraction successful!")
            logger.info(f"Title: {result['title']}")
            logger.info(f"Text length: {len(result['text'])} characters")
            logger.info(f"Language: {result['language']}")
            logger.info(f"Content type: {result['content_type']}")
            logger.info(f"Text preview: {result['text'][:300]}...")
        else:
            logger.error(f"❌ Text extraction failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"❌ Error in Demo 2: {e}")
    
    # Demo 3: Summarize Open Library content
    logger.info("\n📋 Demo 3: Summarize Open Library content")
    logger.info("-" * 40)
    
    try:
        result = await server.mcp.tools["summarize_content"](
            content=war_and_peace_url,
            summary_length="medium"
        )
        
        if result["success"]:
            logger.info("✅ Content summarization successful!")
            logger.info(f"Title: {result['title']}")
            logger.info(f"Content type: {result['content_type']}")
            logger.info(f"Summary: {result['summary']}")
        else:
            logger.error(f"❌ Content summarization failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"❌ Error in Demo 3: {e}")
    
    # Demo 4: Download Open Library content directly
    logger.info("\n📥 Demo 4: Direct Open Library download")
    logger.info("-" * 40)
    
    try:
        result = await server.mcp.tools["download_openlibrary_content"](
            url=war_and_peace_url
        )
        
        if result["success"]:
            logger.info("✅ Open Library download successful!")
            logger.info(f"Title: {result['title']}")
            logger.info(f"Content length: {result['content_length']} characters")
            logger.info(f"Vector ID: {result['vector_id']}")
            logger.info(f"Summary ID: {result['summary_id']}")
            logger.info(f"Entities extracted: {result['entities_count']}")
            logger.info(f"Relationships created: {result['relationships_count']}")
            logger.info(f"Knowledge graph nodes: {result['knowledge_graph_nodes']}")
            logger.info(f"Knowledge graph edges: {result['knowledge_graph_edges']}")
            logger.info(f"Summary: {result['summary'][:200]}...")
            
            # Show extracted metadata
            metadata = result['metadata']
            logger.info(f"Author: {metadata.get('author', 'Unknown')}")
            logger.info(f"Publication year: {metadata.get('publication_year', 'Unknown')}")
            logger.info(f"Genre: {metadata.get('genre', 'Unknown')}")
            logger.info(f"Subjects: {', '.join(metadata.get('subjects', []))}")
        else:
            logger.error(f"❌ Open Library download failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"❌ Error in Demo 4: {e}")
    
    # Demo 5: Content type detection
    logger.info("\n🔍 Demo 5: Content type detection")
    logger.info("-" * 40)
    
    test_contents = [
        war_and_peace_url,
        "https://example.com/document.pdf",
        "https://example.com/audio.mp3",
        "This is plain text content"
    ]
    
    for content in test_contents:
        try:
            result = await server.mcp.tools["detect_content_type"](
                content=content
            )
            
            if result["success"]:
                logger.info(f"Content: {content[:50]}...")
                logger.info(f"  Type: {result['content_type']}")
                logger.info(f"  Is Open Library: {result['is_openlibrary']}")
            else:
                logger.error(f"❌ Content type detection failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            logger.error(f"❌ Error detecting content type: {e}")
    
    # Demo 6: Translate Open Library content
    logger.info("\n🌐 Demo 6: Translate Open Library content")
    logger.info("-" * 40)
    
    try:
        result = await server.mcp.tools["translate_content"](
            content=war_and_peace_url,
            target_language="es",
            source_language="auto"
        )
        
        if result["success"]:
            logger.info("✅ Content translation successful!")
            logger.info(f"Target language: {result['target_language']}")
            logger.info(f"Source language: {result['source_language']}")
            logger.info(f"Content type: {result['content_type']}")
            logger.info(f"Translated text preview: {result['translated_text'][:300]}...")
        else:
            logger.error(f"❌ Content translation failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"❌ Error in Demo 6: {e}")
    
    # Demo 7: Process regular text content
    logger.info("\n📝 Demo 7: Process regular text content")
    logger.info("-" * 40)
    
    sample_text = """
    This is a sample text for testing the enhanced process content agent.
    It contains positive words like great, excellent, and wonderful.
    The text is about artificial intelligence and machine learning.
    """
    
    try:
        result = await server.mcp.tools["process_content"](
            content=sample_text,
            content_type="text",
            language="en"
        )
        
        if result["success"]:
            logger.info("✅ Text processing successful!")
            logger.info(f"Content type: {result['content_type']}")
            logger.info(f"Result: {result['result']}")
        else:
            logger.error(f"❌ Text processing failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"❌ Error in Demo 7: {e}")
    
    logger.info("\n🎉 Enhanced Process Content Agent Integration Demo Completed!")
    logger.info("=" * 60)


async def main():
    """Main function to run the demo."""
    logger.info("🚀 Starting Enhanced Process Content Agent Integration Demo...")
    
    await demo_enhanced_process_content_integration()
    
    logger.info("✅ Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
