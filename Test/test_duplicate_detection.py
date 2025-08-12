"""
Test script for duplicate detection functionality.
Demonstrates how the system handles duplicate file processing.
"""

import asyncio
import os
from pathlib import Path

from loguru import logger

from src.core.orchestrator import SentimentOrchestrator
from src.core.duplicate_detection_service import DuplicateDetectionService
from src.core.models import AnalysisRequest, DataType, SentimentLabel


async def test_duplicate_detection():
    """Test duplicate detection with the Classical Chinese PDF file."""
    
    # Initialize orchestrator
    orchestrator = SentimentOrchestrator()
    
    # Test file path
    test_file = "data/Classical Chinese Sample 22208_0_8.pdf"
    
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return
    
    logger.info("=== Duplicate Detection Test ===")
    logger.info(f"Testing with file: {test_file}")
    
    # First processing
    logger.info("\n1. First processing of the file...")
    request1 = AnalysisRequest(
        data_type=DataType.PDF,
        content=test_file,
        language="zh",
        reflection_enabled=False
    )
    
    result1 = await orchestrator.analyze(request1)
    logger.info(f"First processing result: {result1.sentiment.label} (confidence: {result1.sentiment.confidence:.3f})")
    
    # Second processing (should be detected as duplicate)
    logger.info("\n2. Second processing of the same file...")
    request2 = AnalysisRequest(
        data_type=DataType.PDF,
        content=test_file,
        language="zh",
        reflection_enabled=False
    )
    
    result2 = await orchestrator.analyze(request2)
    logger.info(f"Second processing result: {result2.sentiment.label} (confidence: {result2.sentiment.confidence:.3f})")
    
    if result2.metadata.get("duplicate_detected"):
        logger.info("✅ Duplicate detected successfully!")
        logger.info(f"Duplicate type: {result2.metadata.get('duplicate_type')}")
        logger.info(f"Recommendation: {result2.metadata.get('recommendation')}")
    else:
        logger.warning("❌ Duplicate not detected")
    
    # Third processing with force reprocess
    logger.info("\n3. Third processing with force reprocess...")
    request3 = AnalysisRequest(
        data_type=DataType.PDF,
        content=test_file,
        language="zh",
        reflection_enabled=False
    )
    request3.force_reprocess = True  # Force reprocessing
    
    result3 = await orchestrator.analyze(request3)
    logger.info(f"Force reprocess result: {result3.sentiment.label} (confidence: {result3.sentiment.confidence:.3f})")
    
    if not result3.metadata.get("duplicate_detected"):
        logger.info("✅ Force reprocess worked - duplicate detection bypassed")
    else:
        logger.warning("❌ Force reprocess didn't work as expected")
    
    # Get duplicate detection statistics
    logger.info("\n4. Duplicate detection statistics...")
    stats = await orchestrator.get_duplicate_stats()
    
    logger.info(f"Total processed files: {stats.get('total_files', 0)}")
    logger.info(f"Files by type: {stats.get('files_by_type', {})}")
    logger.info(f"Recent activity (24h): {stats.get('recent_activity_24h', 0)}")
    
    most_processed = stats.get('most_processed', [])
    if most_processed:
        logger.info("Most processed files:")
        for file_info in most_processed[:3]:
            logger.info(f"  - {file_info['file_path']}: {file_info['processing_count']} times")
    
    # Test with different content (should not be detected as duplicate)
    logger.info("\n5. Testing with different content...")
    different_content = "This is completely different content that should not be detected as a duplicate."
    
    request4 = AnalysisRequest(
        data_type=DataType.TEXT,
        content=different_content,
        language="en",
        reflection_enabled=False
    )
    
    result4 = await orchestrator.analyze(request4)
    logger.info(f"Different content result: {result4.sentiment.label} (confidence: {result4.sentiment.confidence:.3f})")
    
    if not result4.metadata.get("duplicate_detected"):
        logger.info("✅ Different content correctly not detected as duplicate")
    else:
        logger.warning("❌ Different content incorrectly detected as duplicate")
    
    # Test file path duplicate detection
    logger.info("\n6. Testing file path duplicate detection...")
    request5 = AnalysisRequest(
        data_type=DataType.PDF,
        content=test_file,
        language="zh",
        reflection_enabled=False
    )
    
    result5 = await orchestrator.analyze(request5)
    logger.info(f"File path test result: {result5.sentiment.label} (confidence: {result5.sentiment.confidence:.3f})")
    
    # Check if it's a duplicate result or if duplicate was detected
    if (result5.sentiment.label == SentimentLabel.DUPLICATE or 
        result5.metadata.get("duplicate_detected") or
        "duplicate" in str(result5.sentiment.label).lower()):
        logger.info("✅ File path duplicate detection working")
    else:
        logger.warning("❌ File path duplicate detection not working")
        logger.info(f"Result metadata: {result5.metadata}")
        logger.info(f"Sentiment label: {result5.sentiment.label}")
    
    logger.info("\n=== Test Complete ===")


async def test_duplicate_detection_service_directly():
    """Test the duplicate detection service directly."""
    
    logger.info("\n=== Direct Duplicate Detection Service Test ===")
    
    # Initialize service
    service = DuplicateDetectionService()
    
    # Test file
    test_file = "data/Classical Chinese Sample 22208_0_8.pdf"
    
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        return
    
    # Test duplicate detection
    logger.info("Testing duplicate detection...")
    duplicate_result = await service.detect_duplicates(
        file_path=test_file,
        data_type="pdf",
        agent_id="test_agent"
    )
    
    logger.info(f"Duplicate detected: {duplicate_result.is_duplicate}")
    logger.info(f"Duplicate type: {duplicate_result.duplicate_type}")
    logger.info(f"Confidence: {duplicate_result.confidence}")
    logger.info(f"Recommendation: {duplicate_result.recommendation}")
    
    if duplicate_result.existing_metadata:
        logger.info(f"Existing metadata: {duplicate_result.existing_metadata}")
    
    # Test with force reprocess
    logger.info("\nTesting with force reprocess...")
    force_result = await service.detect_duplicates(
        file_path=test_file,
        data_type="pdf",
        agent_id="test_agent",
        force_reprocess=True
    )
    
    logger.info(f"Force reprocess result: {force_result.is_duplicate}")
    logger.info(f"Recommendation: {force_result.recommendation}")
    
    # Get statistics
    logger.info("\nGetting statistics...")
    stats = await service.get_processing_stats()
    logger.info(f"Statistics: {stats}")


async def main():
    """Main test function."""
    try:
        # Test orchestrator integration
        await test_duplicate_detection()
        
        # Test service directly
        await test_duplicate_detection_service_directly()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
