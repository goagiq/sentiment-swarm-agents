#!/usr/bin/env python3
"""
Test script for the Enhanced Unified MCP Server with Open Library integration.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
from src.mcp_servers.enhanced_unified_mcp_server import EnhancedUnifiedMCPServer


async def test_enhanced_mcp_server():
    """Test the enhanced MCP server with Open Library functionality."""
    
    logger.info("🚀 Testing Enhanced Unified MCP Server...")
    
    # Initialize the enhanced MCP server
    server = EnhancedUnifiedMCPServer()
    
    # Test 1: Content type detection
    logger.info("\n🔍 Test 1: Content type detection")
    
    test_contents = [
        "https://openlibrary.org/books/OL14047767M/Voina_i_mir_%D0%92%D0%9E%D0%99%D0%9D%D0%90_%D0%B8_%D0%9C%D0%98%D0%A0%D0%AA",
        "https://example.com/document.pdf",
        "https://example.com/audio.mp3",
        "This is plain text content"
    ]
    
    for content in test_contents:
        try:
            content_type = server._detect_content_type(content)
            is_openlibrary = 'openlibrary.org' in content.lower()
            
            logger.info(f"Content: {content[:50]}...")
            logger.info(f"  Detected type: {content_type}")
            logger.info(f"  Is Open Library: {is_openlibrary}")
        
        except Exception as e:
            logger.error(f"❌ Error detecting content type: {e}")
    
    # Test 2: Open Library download
    logger.info("\n📥 Test 2: Open Library download")
    
    war_and_peace_url = "https://openlibrary.org/books/OL14047767M/Voina_i_mir_%D0%92%D0%9E%D0%99%D0%9D%D0%90_%D0%B8_%D0%9C%D0%98%D0%A0%D0%AA"
    
    try:
        webpage_content = await server._download_openlibrary_content(war_and_peace_url)
        
        if webpage_content:
            logger.info("✅ Open Library download successful!")
            logger.info(f"Title: {webpage_content.get('title', 'Unknown')}")
            logger.info(f"Content length: {len(webpage_content.get('text', ''))} characters")
            logger.info(f"Status code: {webpage_content.get('status_code', 'Unknown')}")
        else:
            logger.error("❌ Open Library download failed")
    
    except Exception as e:
        logger.error(f"❌ Error testing Open Library download: {e}")
    
    # Test 3: Metadata extraction
    logger.info("\n📋 Test 3: Metadata extraction")
    
    sample_content = """
    War and Peace by Leo Tolstoy is a monumental Russian novel published between 1864-1869.
    It tells the story of five aristocratic families during the Napoleonic Wars (1805-1813).
    The novel explores themes of war and peace, love and relationships, and social class.
    """
    
    try:
        metadata = server._extract_metadata_from_content(sample_content, "War and Peace", war_and_peace_url)
        
        logger.info("✅ Metadata extraction successful!")
        logger.info(f"Author: {metadata.get('author', 'Unknown')}")
        logger.info(f"Publication year: {metadata.get('publication_year', 'Unknown')}")
        logger.info(f"Genre: {metadata.get('genre', 'Unknown')}")
        logger.info(f"Subjects: {', '.join(metadata.get('subjects', []))}")
        logger.info(f"Source: {metadata.get('source', 'Unknown')}")
    
    except Exception as e:
        logger.error(f"❌ Error testing metadata extraction: {e}")
    
    # Test 4: Summary generation
    logger.info("\n📝 Test 4: Summary generation")
    
    try:
        summary = server._generate_summary(sample_content, "War and Peace", "Leo Tolstoy")
        
        logger.info("✅ Summary generation successful!")
        logger.info(f"Summary: {summary}")
    
    except Exception as e:
        logger.error(f"❌ Error testing summary generation: {e}")
    
    # Test 5: Open Library content processing
    logger.info("\n📚 Test 5: Open Library content processing")
    
    try:
        result = await server._process_openlibrary_content(war_and_peace_url, "en", {})
        
        if result["success"]:
            logger.info("✅ Open Library content processing successful!")
            logger.info(f"Title: {result.get('title', 'Unknown')}")
            logger.info(f"Vector ID: {result.get('vector_id', 'N/A')}")
            logger.info(f"Summary ID: {result.get('summary_id', 'N/A')}")
            logger.info(f"Entities extracted: {result.get('entities_count', 0)}")
            logger.info(f"Relationships created: {result.get('relationships_count', 0)}")
            logger.info(f"Knowledge graph nodes: {result.get('knowledge_graph_nodes', 0)}")
            logger.info(f"Knowledge graph edges: {result.get('knowledge_graph_edges', 0)}")
            logger.info(f"Content length: {result.get('content_length', 0)} characters")
            logger.info(f"Summary: {result.get('summary', 'N/A')}")
        else:
            logger.error(f"❌ Open Library content processing failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"❌ Error testing Open Library content processing: {e}")
    
    logger.info("\n🎉 Enhanced MCP Server testing completed!")


async def main():
    """Main function to run the test."""
    logger.info("🚀 Starting Enhanced MCP Server Test Suite...")
    
    await test_enhanced_mcp_server()
    
    logger.info("✅ Test suite completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
