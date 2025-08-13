#!/usr/bin/env python3
"""
Direct Classical Chinese PDF MCP Test
Directly tests the MCP tools for processing Classical Chinese PDFs.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_classical_chinese_pdf_processing():
    """Test Classical Chinese PDF processing using MCP tools."""
    logger.info("🚀 Starting Classical Chinese PDF MCP Processing Test")
    
    # Check if PDF file exists
    pdf_path = Path("data/Classical Chinese Sample 22208_0_8.pdf")
    if not pdf_path.exists():
        logger.error(f"❌ PDF file not found: {pdf_path}")
        return
    
    logger.info(f"✅ Found PDF file: {pdf_path}")
    
    # Test results storage
    test_results = {
        "test_name": "Classical Chinese PDF MCP Direct Test",
        "timestamp": datetime.now().isoformat(),
        "pdf_path": str(pdf_path),
        "language": "zh",  # Chinese
        "results": {}
    }
    
    try:
        # Test 1: Check agent status
        logger.info("📋 Test 1: Checking MCP agent status")
        agent_status = await mcp_Sentiment_get_all_agents_status("test")
        test_results["results"]["agent_status"] = {
            "success": True,
            "result": agent_status
        }
        logger.info("✅ Agent status check completed")
        
        # Test 2: Process PDF with enhanced multilingual support
        logger.info("📋 Test 2: Processing Classical Chinese PDF")
        output_path = f"Results/classical_chinese_mcp_direct_{int(time.time())}"
        
        pdf_result = await mcp_Sentiment_process_pdf_enhanced_multilingual(
            pdf_path=str(pdf_path),
            language="zh",
            generate_report=True,
            output_path=output_path
        )
        
        test_results["results"]["pdf_processing"] = {
            "success": True,
            "result": pdf_result,
            "output_path": output_path
        }
        logger.info("✅ PDF processing completed")
        
        # Test 3: Extract entities from sample text
        logger.info("📋 Test 3: Testing entity extraction")
        sample_text = "孔子曰：学而时习之，不亦说乎。有朋自远方来，不亦乐乎。"
        
        entity_result = await mcp_Sentiment_extract_entities(
            text=sample_text,
            language="zh"
        )
        
        test_results["results"]["entity_extraction"] = {
            "success": True,
            "result": entity_result,
            "sample_text": sample_text
        }
        logger.info("✅ Entity extraction completed")
        
        # Test 4: Analyze text sentiment
        logger.info("📋 Test 4: Testing text analysis")
        text_result = await mcp_Sentiment_analyze_text_sentiment(
            text=sample_text,
            agent_type="standard",
            language="zh"
        )
        
        test_results["results"]["text_analysis"] = {
            "success": True,
            "result": text_result,
            "sample_text": sample_text
        }
        logger.info("✅ Text analysis completed")
        
        # Test 5: Generate knowledge graph report
        logger.info("📋 Test 5: Generating knowledge graph report")
        report_result = await mcp_Sentiment_generate_graph_report(
            output_path=f"Results/classical_chinese_graph_report_{int(time.time())}",
            target_language="zh"
        )
        
        test_results["results"]["report_generation"] = {
            "success": True,
            "result": report_result
        }
        logger.info("✅ Report generation completed")
        
        # Save test results
        results_dir = Path("Results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f"classical_chinese_mcp_direct_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Test results saved to: {results_file}")
        
        # Print summary
        logger.info("📊 Test Summary:")
        logger.info("=" * 50)
        for test_name, result in test_results["results"].items():
            if result.get("success"):
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.info(f"❌ {test_name}: FAILED")
        logger.info("=" * 50)
        
        logger.info("🎉 Classical Chinese PDF MCP Direct Test completed!")
        
        return test_results
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        test_results["results"]["error"] = {
            "success": False,
            "error": str(e)
        }
        return test_results


async def main():
    """Main test function."""
    results = await test_classical_chinese_pdf_processing()
    return results


if __name__ == "__main__":
    asyncio.run(main())
