#!/usr/bin/env python3
"""
Generic script to download any Open Library book content and add to both 
vector database and knowledge graph.
Uses the existing EnhancedWebAgent for downloading and processing.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
from src.core.vector_db import VectorDBManager
from src.core.improved_knowledge_graph_utility import ImprovedKnowledgeGraphUtility
from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.agents.web_agent_enhanced import EnhancedWebAgent


async def download_openlibrary_content(url: str) -> Optional[Dict[str, Any]]:
    """Download content from Open Library URL using EnhancedWebAgent."""
    
    # Initialize the enhanced web agent
    web_agent = EnhancedWebAgent()
    
    logger.info(f"🔍 Downloading content from: {url}")
    
    try:
        # Use the web agent's _fetch_webpage method directly
        webpage_data = await web_agent._fetch_webpage(url)
        
        # Process the webpage data
        cleaned_text = web_agent._clean_webpage_text(webpage_data["html"])
        
        webpage_content = {
            "url": url,
            "title": webpage_data["title"],
            "text": cleaned_text,
            "html": webpage_data["html"],
            "status_code": webpage_data["status_code"]
        }
        
        if "error" in webpage_content:
            logger.error(f"❌ Failed to download content: {webpage_content['error']}")
            return None
        
        logger.info("✅ Successfully downloaded webpage content")
        logger.info(f"📄 Content length: {len(webpage_content.get('text', ''))} characters")
        logger.info(f"📋 Title: {webpage_content.get('title', 'Unknown')}")
        
        return webpage_content
        
    except Exception as e:
        logger.error(f"❌ Error downloading content: {e}")
        return None


async def add_book_to_database(
    url: str, 
    language: str = "en",
    book_title: Optional[str] = None,
    author: Optional[str] = None
) -> Dict[str, Any]:
    """Add book content from Open Library URL to vector database and knowledge graph."""
    
    # Initialize services
    vector_db = VectorDBManager()
    kg_utility = ImprovedKnowledgeGraphUtility()
    kg_agent = KnowledgeGraphAgent()
    
    try:
        logger.info(f"🚀 Starting book database addition for: {url}")
        
        # 1. Download content from Open Library
        webpage_content = await download_openlibrary_content(url)
        
        if not webpage_content:
            logger.error("❌ Failed to download content")
            return {
                "success": False,
                "error": "Failed to download content from URL"
            }
        
        # Extract text content
        content_text = webpage_content.get("text", "")
        title = book_title or webpage_content.get("title", "Unknown Book")
        source_url = webpage_content.get("url", url)
        
        # Basic metadata extraction from content
        extracted_metadata = extract_metadata_from_content(content_text, title, author)
        
        # Metadata for the content
        metadata = {
            "title": title,
            "author": author or extracted_metadata.get("author", "Unknown"),
            "publication_year": extracted_metadata.get("publication_year", "Unknown"),
            "language": language,
            "genre": extracted_metadata.get("genre", "Unknown"),
            "category": extracted_metadata.get("category", "Literature"),
            "source": "Open Library",
            "source_url": source_url,
            "content_type": "book_description",
            "subjects": extracted_metadata.get("subjects", []),
            "download_method": "EnhancedWebAgent",
            "content_length": len(content_text),
            "extracted_metadata": extracted_metadata
        }
        
        # 2. Store in vector database
        logger.info("Storing content in vector database...")
        vector_id = await vector_db.store_content(content_text, metadata)
        logger.info(f"✅ Content stored in vector database with ID: {vector_id}")
        
        # 3. Extract entities using appropriate language configuration
        logger.info(f"Extracting entities with {language} language support...")
        entities_result = await kg_agent.extract_entities(content_text, language)
        entities = entities_result.get("content", [{}])[0].get("json", {}).get("entities", [])
        logger.info(f"✅ Extracted {len(entities)} entities")
        
        # 4. Extract relationships
        logger.info("Extracting relationships...")
        relationships_result = await kg_agent.map_relationships(content_text, entities)
        relationships = relationships_result.get("content", [{}])[0].get("json", {}).get("relationships", [])
        logger.info(f"✅ Extracted {len(relationships)} relationships")
        
        # 5. Create knowledge graph with extracted entities and relationships
        logger.info("Creating knowledge graph...")
        
        # Transform entities to expected format
        transformed_entities = []
        for entity in entities:
            transformed_entities.append({
                "name": entity.get("text", ""),
                "type": entity.get("type", "CONCEPT"),
                "confidence": entity.get("confidence", 0.0),
                "source": title
            })
        
        # Transform relationships to expected format
        transformed_relationships = []
        for rel in relationships:
            transformed_relationships.append({
                "source": rel.get("source", ""),
                "target": rel.get("target", ""),
                "relationship_type": rel.get("type", "RELATED_TO"),
                "confidence": rel.get("confidence", 0.0),
                "source_type": title
            })
        
        kg_result = await kg_utility.create_knowledge_graph(transformed_entities, transformed_relationships)
        logger.info(f"✅ Knowledge graph created with {kg_result.number_of_nodes()} nodes and {kg_result.number_of_edges()} edges")
        
        # 6. Display extracted entities
        if entities:
            logger.info("Entities extracted successfully:")
            for entity in entities[:10]:  # Show first 10
                logger.info(f"  - {entity.get('text', '')} ({entity.get('type', '')})")
        
        # 7. Display extracted relationships
        if relationships:
            logger.info("Relationships extracted successfully:")
            for rel in relationships[:10]:  # Show first 10
                logger.info(f"  - {rel.get('source', '')} -> {rel.get('target', '')} ({rel.get('type', '')})")
        
        # 8. Generate summary
        logger.info("Generating content summary...")
        try:
            # Create a simple summary based on content
            summary = generate_summary(content_text, title, author)
            logger.info("✅ Summary generated successfully")
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            summary = "Summary generation failed"
        
        # 9. Store summary in vector database
        logger.info("Storing summary in vector database...")
        summary_metadata = metadata.copy()
        summary_metadata.update({
            "content_type": "summary",
            "parent_id": vector_id,
            "summary_type": "book_summary"
        })
        summary_id = await vector_db.store_content(summary, summary_metadata)
        logger.info(f"✅ Summary stored in vector database with ID: {summary_id}")
        
        # 10. Generate knowledge graph visualization
        logger.info("Generating knowledge graph visualization...")
        try:
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            viz_result = await kg_utility.generate_graph_visualization(f"{safe_title}_knowledge_graph")
            logger.info(f"✅ Knowledge graph visualization generated: {viz_result}")
        except Exception as e:
            logger.warning(f"Failed to generate visualization: {e}")
            viz_result = {"error": str(e)}
        
        logger.info(f"🎉 Successfully added {title} to both vector database and knowledge graph!")
        
        # Return comprehensive results
        return {
            "success": True,
            "book_title": title,
            "author": author or extracted_metadata.get("author", "Unknown"),
            "vector_id": vector_id,
            "summary_id": summary_id,
            "knowledge_graph": {
                "nodes": kg_result.number_of_nodes(),
                "edges": kg_result.number_of_edges()
            },
            "visualization": viz_result,
            "entities_count": len(entities) if entities else 0,
            "relationships_count": len(relationships) if relationships else 0,
            "language": language,
            "summary": summary[:200] + "..." if len(summary) > 200 else summary,
            "metadata": metadata,
            "download_success": True,
            "content_length": len(content_text)
        }
        
    except Exception as e:
        logger.error(f"❌ Error adding book to database: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def extract_metadata_from_content(content: str, title: str, author: Optional[str] = None) -> Dict[str, Any]:
    """Extract metadata from content text."""
    content_lower = content.lower()
    
    # Try to extract author if not provided
    if not author:
        # Look for common author patterns
        author_patterns = [
            "by ", "author:", "written by", "author is"
        ]
        for pattern in author_patterns:
            if pattern in content_lower:
                start_idx = content_lower.find(pattern) + len(pattern)
                end_idx = content.find("\n", start_idx)
                if end_idx == -1:
                    end_idx = start_idx + 100
                author = content[start_idx:end_idx].strip()
                break
    
    # Try to extract publication year
    import re
    year_pattern = r'\b(19|20)\d{2}\b'
    years = re.findall(year_pattern, content)
    publication_year = years[0] if years else "Unknown"
    
    # Try to determine genre/category
    genre_keywords = {
        "fiction": ["novel", "story", "tale", "fiction"],
        "non-fiction": ["history", "biography", "memoir", "essay"],
        "poetry": ["poem", "poetry", "verse"],
        "drama": ["play", "drama", "theater", "theatre"],
        "science": ["science", "physics", "chemistry", "biology"],
        "philosophy": ["philosophy", "philosophical", "ethics"],
        "religion": ["religion", "religious", "spiritual", "theology"]
    }
    
    detected_genre = "Literature"
    for genre, keywords in genre_keywords.items():
        if any(keyword in content_lower for keyword in keywords):
            detected_genre = genre.title()
            break
    
    # Extract subjects/topics
    subjects = []
    subject_keywords = [
        "history", "war", "peace", "love", "family", "politics", 
        "society", "culture", "art", "music", "science", "philosophy",
        "religion", "nature", "travel", "adventure", "mystery"
    ]
    
    for subject in subject_keywords:
        if subject in content_lower:
            subjects.append(subject.title())
    
    return {
        "author": author or "Unknown",
        "publication_year": publication_year,
        "genre": detected_genre,
        "category": "Classic Literature" if "classic" in content_lower else detected_genre,
        "subjects": subjects[:10]  # Limit to 10 subjects
    }


def generate_summary(content: str, title: str, author: Optional[str] = None) -> str:
    """Generate a summary of the book content."""
    # Simple summary generation based on content analysis
    content_lower = content.lower()
    
    # Extract key themes
    themes = []
    theme_keywords = {
        "war": ["war", "battle", "conflict", "military"],
        "love": ["love", "romance", "relationship", "marriage"],
        "family": ["family", "parent", "child", "sibling"],
        "society": ["society", "social", "class", "aristocracy"],
        "philosophy": ["philosophy", "meaning", "purpose", "existence"],
        "history": ["history", "historical", "past", "era"]
    }
    
    for theme, keywords in theme_keywords.items():
        if any(keyword in content_lower for keyword in keywords):
            themes.append(theme)
    
    # Create summary
    author_text = f" by {author}" if author else ""
    themes_text = ", ".join(themes[:3]) if themes else "various themes"
    
    summary = f"{title}{author_text} is a literary work that explores {themes_text}. "
    summary += f"The book contains {len(content.split())} words and covers topics related to {', '.join(themes[:5]) if themes else 'literature and human experience'}."
    
    return summary


async def main():
    """Main function to run the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and process Open Library books")
    parser.add_argument("url", help="Open Library URL to download")
    parser.add_argument("--language", "-l", default="en", help="Language code (default: en)")
    parser.add_argument("--title", "-t", help="Book title (optional)")
    parser.add_argument("--author", "-a", help="Book author (optional)")
    
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting Open Library book download and processing...")
    logger.info(f"URL: {args.url}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Title: {args.title or 'Auto-detected'}")
    logger.info(f"Author: {args.author or 'Auto-detected'}")
    
    result = await add_book_to_database(
        url=args.url,
        language=args.language,
        book_title=args.title,
        author=args.author
    )
    
    if result["success"]:
        logger.info("✅ Script completed successfully!")
        logger.info(f"Book: {result['book_title']}")
        logger.info(f"Author: {result['author']}")
        logger.info(f"Content length: {result.get('content_length', 0)} characters")
        logger.info(f"Vector ID: {result['vector_id']}")
        logger.info(f"Summary ID: {result['summary_id']}")
        logger.info(f"Entities extracted: {result['entities_count']}")
        logger.info(f"Relationships created: {result['relationships_count']}")
        logger.info(f"Knowledge graph nodes: {result['knowledge_graph']['nodes']}")
        logger.info(f"Knowledge graph edges: {result['knowledge_graph']['edges']}")
        logger.info(f"Language: {result['language']}")
        logger.info(f"Summary: {result['summary']}")
    else:
        logger.error(f"❌ Script failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
