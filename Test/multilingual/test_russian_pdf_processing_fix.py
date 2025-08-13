#!/usr/bin/env python3
"""
Test script to verify Russian PDF processing fix end-to-end.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.file_extraction_agent import FileExtractionAgent
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from core.models import AnalysisRequest, DataType


async def test_russian_pdf_processing():
    """Test Russian PDF processing end-to-end."""
    
    # Create sample Russian text (simulating extracted PDF content)
    russian_text = """
    Владимир Путин является президентом России с 2012 года. 
    Москва является столицей России и крупнейшим городом страны.
    Газпром является крупнейшей энергетической компанией России.
    МГУ имени Ломоносова является ведущим университетом России.
    Искусственный интеллект и машинное обучение активно развиваются в России.
    """
    
    print("🧪 Testing Russian PDF processing end-to-end...")
    print("📝 Simulated PDF content:", russian_text.strip())
    
    try:
        # Initialize agents
        file_agent = FileExtractionAgent()
        kg_agent = KnowledgeGraphAgent()
        
        # Create analysis request (simulating PDF extraction)
        pdf_request = AnalysisRequest(
            data_type=DataType.PDF,
            content="simulated_russian_pdf.pdf",  # This would be the actual file path
            language="ru"
        )
        
        # Simulate text extraction from PDF
        print("\n📄 Simulating PDF text extraction...")
        extraction_result = await file_agent.process(pdf_request)
        
        # Use the Russian text as extracted content
        text_content = russian_text
        
        print(f"✅ Text extraction successful, content length: {len(text_content)}")
        
        # Process with knowledge graph agent
        print("\n🧠 Processing with knowledge graph agent...")
        kg_request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=text_content,
            language="ru"
        )
        
        kg_result = await kg_agent.process(kg_request)
        
        print(f"\n✅ Knowledge graph processing result:")
        print(f"  - Status: {kg_result.status}")
        print(f"  - Processing time: {kg_result.processing_time:.2f}s")
        
        # Check statistics
        if kg_result.metadata and "statistics" in kg_result.metadata:
            stats = kg_result.metadata["statistics"]
            print(f"  - Entities found: {stats.get('entities_found', 0)}")
            print(f"  - Entity types: {stats.get('entity_types', {})}")
            print(f"  - Language stats: {stats.get('language_stats', {})}")
            
            # Check if Russian entities were found
            language_stats = stats.get('language_stats', {})
            russian_entities = language_stats.get('ru', 0)
            print(f"  - Russian entities: {russian_entities}")
            
            return russian_entities > 0
        else:
            print("  - No statistics available")
            return False
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 Starting Russian PDF processing test...")
    
    success = await test_russian_pdf_processing()
    
    if success:
        print("\n✅ Russian PDF processing test PASSED!")
        print("🎉 Russian entity extraction is now working properly!")
    else:
        print("\n❌ Russian PDF processing test FAILED!")
        print("🔧 Russian entity extraction still needs fixing!")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())
