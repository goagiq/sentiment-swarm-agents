#!/usr/bin/env python3
"""
File Extraction Agent Demo

This demo shows how to use the File Extraction Agent to process PDF files
with parallel processing, real-time progress tracking, and ChromaDB integration.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agents.file_extraction_agent import FileExtractionAgent
from src.core.models import AnalysisRequest, DataType
from src.config.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_basic_extraction():
    """Demonstrate basic PDF extraction."""
    logger.info("=== Basic PDF Extraction Demo ===")
    
    # Initialize agent
    agent = FileExtractionAgent(
        agent_id="demo_agent",
        max_capacity=3,
        model_name="llava:latest",
        max_workers=2,
        retry_attempts=1
    )
    
    # Create a sample request (you would replace this with actual PDF path)
    sample_pdf = Path(config.test_dir) / "sample_document.pdf"
    
    if not sample_pdf.exists():
        logger.warning(f"Sample PDF not found: {sample_pdf}")
        logger.info("Please place a PDF file in the Test directory to run this demo")
        return
    
    # Create request
    request = AnalysisRequest(
        data_type=DataType.PDF,
        content=str(sample_pdf),
        language="en"
    )
    
    # Process the PDF
    logger.info(f"Processing: {sample_pdf.name}")
    result = await agent.process(request)
    
    # Display results
    logger.info(f"Processing completed in {result.processing_time:.2f}s")
    logger.info(f"Status: {result.status}")
    logger.info(f"Method used: {result.metadata.get('method', 'unknown')}")
    logger.info(f"Pages processed: {result.metadata.get('pages_processed', 0)}")
    logger.info(f"Confidence: {result.quality_score:.2f}")
    
    if result.extracted_text:
        logger.info(f"Extracted {len(result.extracted_text)} characters")
        # Show first 300 characters
        preview = result.extracted_text[:300]
        if len(result.extracted_text) > 300:
            preview += "..."
        logger.info(f"Text preview:\n{preview}")
    
    return result


async def demo_parallel_processing():
    """Demonstrate parallel processing of multiple PDFs."""
    logger.info("\n=== Parallel Processing Demo ===")
    
    # Initialize high-performance agent
    agent = FileExtractionAgent(
        agent_id="parallel_demo_agent",
        max_capacity=5,
        model_name="llava:latest",
        max_workers=4,  # More workers for parallel processing
        retry_attempts=1
    )
    
    # Find all PDF files in test directory
    test_dir = Path(config.test_dir)
    pdf_files = list(test_dir.glob("*.pdf"))
    
    if len(pdf_files) < 2:
        logger.warning("Need at least 2 PDF files for parallel processing demo")
        logger.info(f"Found {len(pdf_files)} PDF files in {test_dir}")
        return
    
    logger.info(f"Processing {len(pdf_files)} PDF files in parallel")
    
    # Create requests for all PDFs
    requests = []
    for pdf_file in pdf_files:
        request = AnalysisRequest(
            data_type=DataType.PDF,
            content=str(pdf_file),
            language="en"
        )
        requests.append((pdf_file, request))
    
    # Process all requests concurrently
    results = []
    for pdf_file, request in requests:
        logger.info(f"Starting processing: {pdf_file.name}")
        result = await agent.process(request)
        results.append((pdf_file, result))
        
        # Log individual results
        logger.info(f"✓ {pdf_file.name}: {result.status} "
                   f"({result.processing_time:.2f}s)")
    
    # Summary
    successful = sum(1 for _, r in results if r.status == "completed")
    total_time = sum(r.processing_time for _, r in results)
    
    logger.info(f"\nParallel processing summary:")
    logger.info(f"Successfully processed: {successful}/{len(results)} files")
    logger.info(f"Total processing time: {total_time:.2f}s")
    logger.info(f"Average time per file: {total_time/len(results):.2f}s")
    
    return results


async def demo_chroma_integration():
    """Demonstrate ChromaDB integration."""
    logger.info("\n=== ChromaDB Integration Demo ===")
    
    from src.core.vector_db import vector_db
    
    try:
        # Get database statistics
        db_stats = vector_db.get_database_stats()
        logger.info(f"ChromaDB Statistics:")
        logger.info(f"  Total documents: {db_stats.get('total_documents', 0)}")
        logger.info(f"  Collections: {list(db_stats.get('collections', {}).keys())}")
        
        # Search for recent PDF extractions
        search_results = await vector_db.search_similar_results(
            "PDF document", n_results=5
        )
        
        logger.info(f"\nRecent PDF extractions in ChromaDB:")
        for i, result in enumerate(search_results, 1):
            metadata = result['metadata']
            logger.info(f"  {i}. {metadata.get('file_path', 'Unknown')}")
            logger.info(f"     Method: {metadata.get('method', 'unknown')}")
            logger.info(f"     Pages: {metadata.get('pages_processed', 0)}")
            logger.info(f"     Confidence: {metadata.get('confidence', 0):.2f}")
            logger.info(f"     Text length: {len(result['text'])} chars")
        
        # Filter by extraction method
        vision_results = await vector_db.get_results_by_filter({
            "method": "vision_ocr"
        }, n_results=3)
        
        if vision_results:
            logger.info(f"\nVision OCR extractions: {len(vision_results)}")
            for result in vision_results:
                logger.info(f"  - {result['metadata'].get('file_path', 'Unknown')}")
        
    except Exception as e:
        logger.error(f"ChromaDB demo failed: {e}")


async def demo_performance_monitoring():
    """Demonstrate performance monitoring and statistics."""
    logger.info("\n=== Performance Monitoring Demo ===")
    
    # Initialize agent with monitoring
    agent = FileExtractionAgent(
        agent_id="monitoring_demo_agent",
        max_capacity=3,
        model_name="llava:latest",
        max_workers=2,
        retry_attempts=1
    )
    
    # Process a few files to generate statistics
    test_dir = Path(config.test_dir)
    pdf_files = list(test_dir.glob("*.pdf"))[:3]  # Process up to 3 files
    
    if not pdf_files:
        logger.warning("No PDF files found for performance monitoring demo")
        return
    
    logger.info(f"Processing {len(pdf_files)} files for statistics...")
    
    for pdf_file in pdf_files:
        request = AnalysisRequest(
            data_type=DataType.PDF,
            content=str(pdf_file),
            language="en"
        )
        await agent.process(request)
    
    # Display comprehensive statistics
    stats = agent.get_stats()
    logger.info(f"\nAgent Performance Statistics:")
    logger.info(f"  Total files processed: {stats['total_files']}")
    logger.info(f"  Successful extractions: {stats['successful_extractions']}")
    logger.info(f"  Failed extractions: {stats['failed_extractions']}")
    logger.info(f"  Total pages processed: {stats['pages_processed']}")
    logger.info(f"  PyPDF2 successes: {stats['pypdf2_success']}")
    logger.info(f"  Vision OCR successes: {stats['vision_ocr_success']}")
    logger.info(f"  Success rate: {stats['successful_extractions']/stats['total_files']*100:.1f}%")
    
    # Agent configuration
    logger.info(f"\nAgent Configuration:")
    logger.info(f"  Agent ID: {stats['agent_id']}")
    logger.info(f"  Model: {stats['model_name']}")
    logger.info(f"  Max workers: {stats['max_workers']}")
    logger.info(f"  Chunk size: {stats['chunk_size']}")
    logger.info(f"  Retry attempts: {stats['retry_attempts']}")


async def main():
    """Main demo function."""
    logger.info("File Extraction Agent Demo")
    logger.info("=" * 50)
    
    # Check if test directory exists and has PDF files
    test_dir = Path(config.test_dir)
    test_dir.mkdir(exist_ok=True)
    
    pdf_files = list(test_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in test directory")
        logger.info(f"Please place some PDF files in: {test_dir}")
        logger.info("You can download sample PDFs or create test documents")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files for testing")
    
    # Run demos
    try:
        # Basic extraction demo
        await demo_basic_extraction()
        
        # Parallel processing demo
        await demo_parallel_processing()
        
        # ChromaDB integration demo
        await demo_chroma_integration()
        
        # Performance monitoring demo
        await demo_performance_monitoring()
        
        logger.info("\n" + "=" * 50)
        logger.info("All demos completed successfully!")
        logger.info("Check the Results directory for extracted text files")
        logger.info("Check ChromaDB for stored vector embeddings")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        logger.info("Make sure Ollama is running and llava:latest model is available")


if __name__ == "__main__":
    asyncio.run(main())
