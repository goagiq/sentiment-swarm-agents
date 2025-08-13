#!/usr/bin/env python3
"""
Debug script to see what entities are extracted from Russian text.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.knowledge_graph_agent import KnowledgeGraphAgent

async def main():
    """Debug Russian entity extraction."""
    print("Debugging Russian entity extraction...")
    
    # Initialize agent
    kg_agent = KnowledgeGraphAgent()
    
    # Sample Russian text from the PDF
    russian_text = """
    Первый круг
    Russian Full Circle
    A First-Year Russian Textbook
    Donna Oliver
    Beloit College
    with Edie Furniss
    The Pennsylvania State University
    New Haven and London
    From 'Russian Full Circle'
    """
    
    print(f"Testing with Russian text: {russian_text[:200]}...")
    
    # Extract entities
    extraction_result = await kg_agent.extract_entities(russian_text, "ru")
    json_data = extraction_result.get("content", [{}])[0].get("json", {})
    entities = json_data.get("entities", [])
    
    print(f"\nExtracted {len(entities)} entities:")
    for i, entity in enumerate(entities):
        print(f"  {i+1}. {entity.get('text', 'N/A')} ({entity.get('type', 'N/A')}) - confidence: {entity.get('confidence', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(main())
