#!/usr/bin/env python3
"""
Simple test script for optimized file extraction agent.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add the parent directory to the path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.file_extraction_agent import FileExtractionAgent
from src.core.models import AnalysisRequest, DataType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_optimized_file_extraction():
    """Test the optimized file extraction agent."""
    
    # Initialize the optimized file extraction agent
    agent = FileExtractionAgent(
        agent_id="optimized_test_agent",
        model_name="llava:latest",
        max_workers=2,
        enable_chroma_storage=False  # Disable for faster testing
    )
    
    # Test with the Classical Chinese PDF
    pdf_path = "data/Classical Chinese Sample 22208_0_8.pdf"
    
    if not Path(pdf_path).exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return
    
    logger.info(f"Testing optimized file extraction on: {pdf_path}")
    
    # Create analysis request
    request = AnalysisRequest(
        id="test_optimized_extraction",
        data_type=DataType.PDF,
        content=pdf_path,
        language="zh"
    )
    
    # Process the request
    logger.info("Starting optimized PDF extraction...")
    result = await agent.process(request)
    
    # Display results
    logger.info(f"Extraction completed with status: {result.status}")
    logger.info(f"Processing time: {result.processing_time:.2f} seconds")
    logger.info(f"Model used: {result.model_used}")
    logger.info(f"Quality score: {result.quality_score}")
    
    # Display structured page information
    if result.pages:
        logger.info(f"\nStructured page data ({len(result.pages)} pages):")
        logger.info("=" * 60)
        
        for page in result.pages:
            status = "✓" if not page.error_message else "✗"
            logger.info(f"Page {page.page_number}: {status}")
            logger.info(f"  Content length: {page.content_length} characters")
            logger.info(f"  Extraction method: {page.extraction_method}")
            logger.info(f"  Confidence: {page.confidence:.2f}")
            if page.processing_time:
                logger.info(f"  Processing time: {page.processing_time:.2f}s")
            if page.error_message:
                logger.info(f"  Error: {page.error_message}")
            logger.info(f"  Content preview: {page.content[:100]}...")
            logger.info("-" * 40)
    
    # Display enhanced metadata
    if result.metadata:
        logger.info(f"\nEnhanced metadata:")
        logger.info("=" * 60)
        
        # Page extraction details
        page_details = result.metadata.get("page_extraction_details", {})
        if page_details:
            logger.info(f"Successful pages: {page_details.get('successful_pages', 0)}")
            logger.info(f"Failed pages: {page_details.get('failed_pages', 0)}")
            logger.info(f"Average confidence: {page_details.get('average_confidence', 0.0):.2f}")
        
        # Performance metrics
        perf_metrics = result.metadata.get("performance_metrics", {})
        if perf_metrics:
            logger.info(f"Memory cleanups: {perf_metrics.get('memory_cleanups', 0)}")
            logger.info(f"Average page time: {perf_metrics.get('average_page_time', 0.0):.2f}s")
    
    # Save detailed results to file
    output_file = "Results/optimized_file_extraction_test.json"
    Path("Results").mkdir(exist_ok=True)
    
    # Convert result to dict for JSON serialization
    result_dict = {
        "request_id": result.request_id,
        "status": result.status,
        "processing_time": result.processing_time,
        "model_used": result.model_used,
        "quality_score": result.quality_score,
        "total_pages": len(result.pages) if result.pages else 0,
        "pages": [
            {
                "page_number": page.page_number,
                "content_length": page.content_length,
                "extraction_method": page.extraction_method,
                "confidence": page.confidence,
                "processing_time": page.processing_time,
                "error_message": page.error_message,
                "content_preview": page.content[:200] + "..." if len(page.content) > 200 else page.content,
                "metadata": page.metadata
            }
            for page in (result.pages or [])
        ],
        "metadata": result.metadata,
        "extracted_text_preview": result.extracted_text[:500] + "..." if result.extracted_text and len(result.extracted_text) > 500 else result.extracted_text
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nDetailed results saved to: {output_file}")
    
    # Display agent statistics
    stats = agent.get_stats()
    logger.info(f"\nAgent statistics:")
    logger.info("=" * 60)
    logger.info(f"Total files processed: {stats['total_files']}")
    logger.info(f"Successful extractions: {stats['successful_extractions']}")
    logger.info(f"Failed extractions: {stats['failed_extractions']}")
    logger.info(f"Total pages processed: {stats['pages_processed']}")
    logger.info(f"PyPDF2 successes: {stats['pypdf2_success']}")
    logger.info(f"Vision OCR successes: {stats['vision_ocr_success']}")
    logger.info(f"Memory cleanups: {stats['memory_cleanups']}")


async def main():
    """Main function."""
    try:
        await test_optimized_file_extraction()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
