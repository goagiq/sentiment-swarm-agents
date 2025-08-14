#!/usr/bin/env python3
"""
Demonstration script for the integrated Data Ingestion Service.
Shows how to use the multilingual content processing pipeline.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
from src.core.data_ingestion_service import data_ingestion_service


async def demo_war_and_peace_ingestion():
    """Demonstrate War and Peace content ingestion with Russian language support."""
    
    logger.info("🎭 Demo: War and Peace Content Ingestion")
    logger.info("=" * 50)
    
    # War and Peace content (mixed Russian and English)
    war_and_peace_content = """
    Voina i mir = ВОЙНА и МИРЪ by Лев Толстой (Leo Tolstoy)
    
    War and Peace delineates in graphic detail events surrounding the French invasion of Russia, and the impact of the Napoleonic era on Tsarist society, as seen through the eyes of five Russian aristocratic families. The novel begins in the year 1805 during the reign of Tsar Alexander I and leads up to the 1812 French invasion of Russia by Napoleon.
    
    Key Characters:
    - Pierre Bezukhov: The illegitimate son of a wealthy count, who inherits his father's fortune
    - Prince Andrei Bolkonsky: A thoughtful and philosophical officer in the Russian army
    - Natasha Rostova: A young, vivacious girl from the Rostov family
    - Nikolai Rostov: Natasha's brother, a young officer
    - Princess Marya Bolkonskaya: Prince Andrei's sister, a religious and kind woman
    
    Historical Context:
    - Set during the Napoleonic Wars (1805-1813)
    - Focuses on Napoleon's invasion of Russia in 1812
    - Explores the impact of war on Russian society
    
    Themes:
    - War and peace as contrasting forces in human life
    - The role of individuals in historical events
    - Love, marriage, and family relationships
    - Social class and aristocracy in 19th century Russia
    """
    
    # Metadata for the content
    metadata = {
        "title": "War and Peace (Voina i mir = ВОЙНА и МИРЪ)",
        "author": "Лев Толстой (Leo Tolstoy)",
        "publication_year": "1864-1869",
        "genre": "Historical Fiction",
        "category": "Classic Literature",
        "source": "Open Library",
        "source_url": "https://openlibrary.org/books/OL14047767M/Voina_i_mir_%D0%92%D0%9E%D0%99%D0%9D%D0%90_%D0%B8_%D0%9C%D0%98%D0%A0%D0%AA",
        "content_type": "book_description",
        "subjects": [
            "Classic Literature", "Historical Fiction", "Russian Literature",
            "Napoleonic Wars", "Russian Empire", "Aristocracy"
        ],
        "key_characters": [
            "Pierre Bezukhov", "Prince Andrei Bolkonsky", "Natasha Rostova",
            "Nikolai Rostov", "Princess Marya Bolkonskaya"
        ],
        "historical_period": "1805-1813",
        "setting": "Russia during Napoleonic Wars"
    }
    
    try:
        # Ingest content with auto language detection
        logger.info("🔄 Starting War and Peace content ingestion...")
        result = await data_ingestion_service.ingest_content(
            content=war_and_peace_content,
            metadata=metadata,
            language_code=None,  # Auto-detect
            auto_detect_language=True,
            generate_summary=True,
            extract_entities=True,
            create_knowledge_graph=True,
            store_in_vector_db=True
        )
        
        if result["success"]:
            logger.info("✅ War and Peace ingestion completed successfully!")
            logger.info(f"   Language detected: {result['language_code']}")
            logger.info(f"   Language config used: {result['language_config_used']}")
            logger.info(f"   Vector IDs: {result['vector_ids']}")
            logger.info(f"   Entities extracted: {len(result['entities'])}")
            logger.info(f"   Relationships extracted: {len(result['relationships'])}")
            logger.info(f"   Knowledge graph: {result['knowledge_graph']['nodes']} nodes, {result['knowledge_graph']['edges']} edges")
            
            # Show extracted entities
            if result['entities']:
                logger.info("   Extracted entities:")
                for entity in result['entities']:
                    logger.info(f"     - {entity.get('text', '')} ({entity.get('type', '')})")
        else:
            logger.error(f"❌ War and Peace ingestion failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")


async def demo_english_content_ingestion():
    """Demonstrate English content ingestion."""
    
    logger.info("\n📚 Demo: English Content Ingestion")
    logger.info("=" * 50)
    
    english_content = """
    The Great Gatsby by F. Scott Fitzgerald
    
    The Great Gatsby is a 1925 novel by American writer F. Scott Fitzgerald. Set in the Jazz Age on Long Island, near New York City, the novel depicts first-person narrator Nick Carraway's interactions with mysterious millionaire Jay Gatsby and Gatsby's obsession to reunite with his former lover, Daisy Buchanan.
    
    Key Characters:
    - Jay Gatsby: A mysterious millionaire who throws lavish parties
    - Nick Carraway: The narrator, a Yale graduate and World War I veteran
    - Daisy Buchanan: Gatsby's former lover, now married to Tom
    - Tom Buchanan: Daisy's husband, a wealthy and arrogant man
    - Jordan Baker: A professional golfer and friend of Daisy
    
    Themes:
    - The American Dream and its corruption
    - Wealth and social class in 1920s America
    - Love and relationships
    - The Jazz Age and its excesses
    """
    
    metadata = {
        "title": "The Great Gatsby",
        "author": "F. Scott Fitzgerald",
        "publication_year": "1925",
        "genre": "Literary Fiction",
        "category": "American Literature",
        "content_type": "book_description",
        "setting": "Long Island, New York, 1920s",
        "themes": ["American Dream", "Wealth", "Love", "Jazz Age"]
    }
    
    try:
        logger.info("🔄 Starting English content ingestion...")
        result = await data_ingestion_service.ingest_content(
            content=english_content,
            metadata=metadata,
            language_code="en",
            auto_detect_language=False,
            generate_summary=True,
            extract_entities=True,
            create_knowledge_graph=True,
            store_in_vector_db=True
        )
        
        if result["success"]:
            logger.info("✅ English content ingestion completed successfully!")
            logger.info(f"   Language: {result['language_code']}")
            logger.info(f"   Vector IDs: {result['vector_ids']}")
            logger.info(f"   Entities extracted: {len(result['entities'])}")
            logger.info(f"   Knowledge graph: {result['knowledge_graph']['nodes']} nodes")
        else:
            logger.error(f"❌ English content ingestion failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")


async def demo_language_configurations():
    """Demonstrate language configuration capabilities."""
    
    logger.info("\n🌍 Demo: Language Configuration Information")
    logger.info("=" * 50)
    
    # Get supported languages
    languages = data_ingestion_service.get_supported_languages()
    logger.info(f"✅ Supported languages ({len(languages)}):")
    for code, name in languages.items():
        logger.info(f"   - {code}: {name}")
    
    # Get detailed configuration for Russian
    logger.info("\n🔧 Russian language configuration:")
    try:
        russian_config = data_ingestion_service.get_language_config_info("ru")
        if "error" not in russian_config:
            logger.info(f"   Language name: {russian_config['language_name']}")
            logger.info(f"   Entity patterns: {len(russian_config['entity_patterns'])} types")
            logger.info(f"   Processing settings: {russian_config['processing_settings']}")
        else:
            logger.error(f"   {russian_config['error']}")
    except Exception as e:
        logger.error(f"   Failed to get Russian config: {e}")


async def demo_url_ingestion():
    """Demonstrate URL-based content ingestion."""
    
    logger.info("\n🌐 Demo: URL-based Content Ingestion")
    logger.info("=" * 50)
    
    # Note: This is a demonstration URL - in practice, you'd use real URLs
    demo_url = "https://example.com/sample-content"
    
    logger.info(f"🔄 Attempting to ingest content from: {demo_url}")
    logger.info("   (This is a demo URL - in practice, use real URLs)")
    
    try:
        result = await data_ingestion_service.ingest_from_url(
            url=demo_url,
            metadata={"demo": True, "source": "demo_url"},
            language_code=None,
            auto_detect_language=True
        )
        
        if result["success"]:
            logger.info("✅ URL ingestion completed successfully!")
            logger.info(f"   Language detected: {result['language_code']}")
            logger.info(f"   Vector IDs: {result['vector_ids']}")
        else:
            logger.info(f"⚠️ URL ingestion result: {result.get('error', 'Expected for demo URL')}")
            
    except Exception as e:
        logger.info(f"⚠️ Expected error for demo URL: {e}")


async def main():
    """Main demonstration function."""
    logger.info("🚀 Starting Data Ingestion Service Demonstration")
    logger.info("=" * 60)
    
    try:
        # Demo 1: War and Peace with Russian language support
        await demo_war_and_peace_ingestion()
        
        # Demo 2: English content
        await demo_english_content_ingestion()
        
        # Demo 3: Language configurations
        await demo_language_configurations()
        
        # Demo 4: URL ingestion (demo)
        await demo_url_ingestion()
        
        logger.info("\n🎉 All demonstrations completed!")
        logger.info("=" * 60)
        logger.info("💡 The Data Ingestion Service is now integrated into the main system.")
        logger.info("🌐 You can also use the API endpoints:")
        logger.info("   - POST /ingest/content - Ingest content directly")
        logger.info("   - POST /ingest/url - Ingest from URL")
        logger.info("   - POST /ingest/file - Ingest from file")
        logger.info("   - GET /ingest/languages - Get supported languages")
        logger.info("   - POST /ingest/language-config - Get language config")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
