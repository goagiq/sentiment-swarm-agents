#!/usr/bin/env python3
"""
Demonstration script for bulk import functionality using the enhanced process_content agent.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
from src.agents.enhanced_process_content_agent import EnhancedProcessContentAgent
from src.core.models import AnalysisRequest, DataType


async def demo_bulk_import():
    """Demonstrate bulk import functionality with the specific URLs."""
    
    # Initialize the enhanced process content agent
    agent = EnhancedProcessContentAgent()
    
    # The bulk import request with the specific URLs
    bulk_request = """add @https://ctext.org/art-of-war and @https://openlibrary.org/books/OL14047767M/Voina_i_mir_%D0%92%D0%9E%D0%99%D0%9D%D0%90_%D0%B8_%D0%9C%D0%98%D0%A0%D0%AA to vector and knowledge graph db"""
    
    logger.info("🚀 Starting bulk import demonstration...")
    logger.info(f"📝 Request: {bulk_request}")
    
    try:
        # Test content type detection
        content_type = agent._detect_content_type(bulk_request)
        logger.info(f"✅ Detected content type: {content_type}")
        
        # Test URL extraction
        urls = agent._extract_urls_from_request(bulk_request)
        logger.info(f"✅ Extracted URLs: {urls}")
        
        # Process the bulk import request
        logger.info("🔄 Processing bulk import request...")
        
        # Create analysis request
        request = AnalysisRequest(
            id="bulk_import_demo",
            content=bulk_request,
            data_type=DataType.TEXT,
            language="en"
        )
        
        result = await agent.process(request)
        
        if result.status == "completed":
            logger.info("✅ Bulk import completed successfully!")
            logger.info(f"📊 URLs processed: {result.metadata.get('urls_processed', 0)}")
            logger.info(f"✅ Successful imports: {result.metadata.get('successful_imports', 0)}")
            logger.info(f"❌ Failed imports: {result.metadata.get('failed_imports', 0)}")
            logger.info(f"🏷️ Total entities: {result.metadata.get('total_entities', 0)}")
            logger.info(f"🔗 Total relationships: {result.metadata.get('total_relationships', 0)}")
            logger.info(f"📈 Knowledge graph nodes: {result.metadata.get('total_knowledge_graph_nodes', 0)}")
            logger.info(f"📉 Knowledge graph edges: {result.metadata.get('total_knowledge_graph_edges', 0)}")
            
            # Show detailed results
            if "results" in result.metadata:
                logger.info("📋 Detailed results:")
                for i, url_result in enumerate(result.metadata["results"], 1):
                    if url_result["success"]:
                        logger.info(f"  {i}. ✅ {url_result['url']}")
                        logger.info(f"     Title: {url_result.get('title', 'Unknown')}")
                        logger.info(f"     Entities: {url_result.get('entities_count', 0)}")
                        logger.info(f"     Relationships: {url_result.get('relationships_count', 0)}")
                        logger.info(f"     Vector ID: {url_result.get('vector_id', 'N/A')}")
                    else:
                        logger.info(f"  {i}. ❌ {url_result['url']}")
                        logger.info(f"     Error: {url_result.get('error', 'Unknown error')}")
            
            logger.info("\n🎉 Bulk import demonstration completed successfully!")
            return result
            
        else:
            logger.error(f"❌ Bulk import failed: {result.metadata.get('error', 'Unknown error')}")
            return result
            
    except Exception as e:
        logger.error(f"❌ Demonstration failed with error: {e}")
        return {"success": False, "error": str(e)}


async def main():
    """Main demonstration function."""
    logger.info("🎯 Enhanced Process Content Agent - Bulk Import Demonstration")
    logger.info("=" * 70)
    
    # Run the bulk import demonstration
    await demo_bulk_import()
    
    logger.info("\n✅ Demonstration completed!")


if __name__ == "__main__":
    asyncio.run(main())
