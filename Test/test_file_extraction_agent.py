#!/usr/bin/env python3
"""
Test script for File Extraction Agent
"""

import asyncio
import logging
from pathlib import Path
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agents.file_extraction_agent import FileExtractionAgent
from src.core.models import AnalysisRequest, DataType
from src.config.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_file_extraction_agent():
    """Test the file extraction agent with sample PDF files."""
    
    # Initialize the agent
    agent = FileExtractionAgent(
        agent_id="test_file_extraction_agent",
        max_capacity=3,
        model_name="llava:latest",
        max_workers=2,
        chunk_size=1,
        retry_attempts=1
    )
    
    logger.info("Starting File Extraction Agent test")
    logger.info(f"Agent ID: {agent.agent_id}")
    logger.info(f"Model: {agent.model_name}")
    logger.info(f"Max workers: {agent.max_workers}")
    
    # Test directory setup
    test_dir = Path(config.test_dir)
    results_dir = Path(config.results_dir)
    
    # Ensure directories exist
    test_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    # Look for PDF files in the test directory
    pdf_files = list(test_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found in test directory")
        logger.info(f"Please place PDF files in: {test_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {pdf_file.name}")
        logger.info(f"File size: {pdf_file.stat().st_size / 1024:.1f} KB")
        
        try:
            # Create analysis request
            request = AnalysisRequest(
                data_type=DataType.PDF,
                content=str(pdf_file),
                language="en"
            )
            
            # Check if agent can process this request
            can_process = await agent.can_process(request)
            if not can_process:
                logger.error(f"Agent cannot process {pdf_file.name}")
                continue
            
            # Process the request
            logger.info("Starting PDF extraction...")
            result = await agent.process(request)
            
            # Log results
            logger.info(f"Processing completed in {result.processing_time:.2f}s")
            logger.info(f"Status: {result.status}")
            logger.info(f"Pages processed: {result.metadata.get('pages_processed', 0)}")
            logger.info(f"Method used: {result.metadata.get('method', 'unknown')}")
            logger.info(f"Confidence: {result.quality_score:.2f}")
            
            if result.extracted_text:
                text_length = len(result.extracted_text)
                logger.info(f"Extracted text length: {text_length} characters")
                
                # Save extracted text to results directory
                output_file = results_dir / f"{pdf_file.stem}_extracted.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result.extracted_text)
                logger.info(f"Extracted text saved to: {output_file}")
                
                # Show first 500 characters as preview
                preview = result.extracted_text[:500]
                if len(result.extracted_text) > 500:
                    preview += "..."
                logger.info(f"Text preview:\n{preview}")
            else:
                logger.warning("No text was extracted")
            
            # Log any errors
            if result.status == "failed":
                error_msg = result.metadata.get('error', 'Unknown error')
                logger.error(f"Extraction failed: {error_msg}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {e}")
    
    # Show final statistics
    logger.info(f"\n{'='*60}")
    logger.info("FINAL STATISTICS")
    logger.info(f"{'='*60}")
    
    stats = agent.get_stats()
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    
    logger.info("\nTest completed!")


async def test_chroma_integration():
    """Test ChromaDB integration for storing extracted text."""
    
    from src.core.vector_db import vector_db
    
    logger.info("\nTesting ChromaDB integration...")
    
    try:
        # Get database stats
        db_stats = vector_db.get_database_stats()
        logger.info(f"Database stats: {db_stats}")
        
        # Search for recent PDF extractions
        search_results = await vector_db.search_similar_results(
            "PDF extraction", n_results=5
        )
        
        logger.info(f"Found {len(search_results)} recent PDF extractions")
        
        for result in search_results:
            logger.info(f"ID: {result['id']}")
            logger.info(f"Method: {result['metadata'].get('method', 'unknown')}")
            logger.info(f"Pages: {result['metadata'].get('pages_processed', 0)}")
            logger.info(f"Text length: {len(result['text'])}")
            logger.info("---")
        
    except Exception as e:
        logger.error(f"ChromaDB test failed: {e}")


async def main():
    """Main test function."""
    logger.info("File Extraction Agent Test Suite")
    logger.info("=" * 50)
    
    # Test file extraction
    await test_file_extraction_agent()
    
    # Test ChromaDB integration
    await test_chroma_integration()


if __name__ == "__main__":
    asyncio.run(main())
