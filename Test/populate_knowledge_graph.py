#!/usr/bin/env python3
"""
Script to populate the knowledge graph with sample data for testing.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.vector_db import VectorDBManager
from loguru import logger


async def populate_knowledge_graph():
    """Populate the knowledge graph with sample data about The Art of War."""
    
    logger.info("🚀 Starting knowledge graph population...")
    
    # Initialize vector database
    vector_db = VectorDBManager()
    
    # Sample data about The Art of War and resource planning
    sample_data = [
        {
            "text": "Sun Tzu's Art of War emphasizes that resource planning is fundamental to military success. The text states that victory requires careful assessment of available resources before engaging in conflict.",
            "metadata": {
                "source": "The Art of War",
                "chapter": "Chapter 1: Laying Plans",
                "topic": "resource_planning",
                "author": "Sun Tzu",
                "type": "principle"
            }
        },
        {
            "text": "Resource planning in warfare involves understanding the five fundamental factors: The Way, Heaven, Earth, Leadership, and Method. These factors determine the outcome of military campaigns.",
            "metadata": {
                "source": "The Art of War",
                "chapter": "Chapter 1: Laying Plans",
                "topic": "resource_planning",
                "author": "Sun Tzu",
                "type": "principle"
            }
        },
        {
            "text": "The relationship between resource planning and war success is direct: 'Victorious warriors win first and then go to war, while defeated warriors go to war first and then seek to win.'",
            "metadata": {
                "source": "The Art of War",
                "chapter": "Chapter 4: Tactical Dispositions",
                "topic": "resource_planning",
                "author": "Sun Tzu",
                "type": "principle"
            }
        },
        {
            "text": "Resource efficiency is crucial in warfare. Sun Tzu warns that prolonged warfare drains resources: 'When you engage in actual fighting, if victory is long in coming, then men's weapons will grow dull and their ardor will be damped.'",
            "metadata": {
                "source": "The Art of War",
                "chapter": "Chapter 2: Waging War",
                "topic": "resource_efficiency",
                "author": "Sun Tzu",
                "type": "principle"
            }
        },
        {
            "text": "The relationship between logistics and victory is clear: 'The line between disorder and order lies in logistics.' Proper resource planning ensures military discipline and effectiveness.",
            "metadata": {
                "source": "The Art of War",
                "chapter": "Chapter 2: Waging War",
                "topic": "logistics",
                "author": "Sun Tzu",
                "type": "principle"
            }
        },
        {
            "text": "Strategic resource allocation requires understanding terrain and conditions. 'Water shapes its course according to the nature of the ground over which it flows.' Resources must be adapted to circumstances.",
            "metadata": {
                "source": "The Art of War",
                "chapter": "Chapter 6: Weak Points and Strong",
                "topic": "strategic_allocation",
                "author": "Sun Tzu",
                "type": "principle"
            }
        },
        {
            "text": "The relationship between preparation and resource planning is fundamental. Sun Tzu states that successful commanders prepare thoroughly before committing resources to battle.",
            "metadata": {
                "source": "The Art of War",
                "chapter": "Chapter 1: Laying Plans",
                "topic": "preparation",
                "author": "Sun Tzu",
                "type": "principle"
            }
        },
        {
            "text": "Resource planning must consider both material and human factors. The text emphasizes that leadership quality and troop morale are as important as physical resources.",
            "metadata": {
                "source": "The Art of War",
                "chapter": "Chapter 1: Laying Plans",
                "topic": "human_resources",
                "author": "Sun Tzu",
                "type": "principle"
            }
        }
    ]
    
    try:
        # Add data to the knowledge graph collection
        for i, data in enumerate(sample_data):
            await vector_db.add_texts(
                collection_name="knowledge_graph",
                texts=[data["text"]],
                metadatas=[data["metadata"]],
                ids=[f"art_of_war_{i+1}"]
            )
            logger.info(f"✅ Added knowledge graph entry {i+1}: {data['metadata']['topic']}")
        
        # Also add to semantic search collection
        for i, data in enumerate(sample_data):
            await vector_db.add_texts(
                collection_name="semantic_search",
                texts=[data["text"]],
                metadatas=[data["metadata"]],
                ids=[f"semantic_{i+1}"]
            )
            logger.info(f"✅ Added semantic search entry {i+1}: {data['metadata']['topic']}")
        
        logger.info("🎉 Knowledge graph population completed successfully!")
        
        # Get statistics
        stats = await vector_db.get_search_statistics()
        logger.info(f"📊 Current database statistics: {stats}")
        
    except Exception as e:
        logger.error(f"❌ Error populating knowledge graph: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(populate_knowledge_graph())
