#!/usr/bin/env python3
"""
Simple Classical Chinese PDF Processing Test
Directly tests the agents for processing Classical Chinese PDFs.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
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
    """Test Classical Chinese PDF processing using direct agent calls."""
    logger.info("🚀 Starting Classical Chinese PDF Processing Test")
    
    # Check if PDF file exists
    pdf_path = Path("data/Classical Chinese Sample 22208_0_8.pdf")
    if not pdf_path.exists():
        logger.error(f"❌ PDF file not found: {pdf_path}")
        return
    
    logger.info(f"✅ Found PDF file: {pdf_path}")
    
    # Test results storage
    test_results = {
        "test_name": "Classical Chinese PDF Direct Processing Test",
        "timestamp": datetime.now().isoformat(),
        "pdf_path": str(pdf_path),
        "language": "zh",  # Chinese
        "results": {}
    }
    
    try:
        # Import required modules
        from src.agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
        from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
        from src.core.models import AnalysisRequest, DataType
        from src.config.language_config.chinese_config import ChineseConfig
        
        # Test 1: Configuration test
        logger.info("📋 Test 1: Testing Chinese configuration")
        chinese_config = ChineseConfig()
        config_tests = {
            "language_code": chinese_config.language_code,
            "language_name": chinese_config.language_name,
            "entity_patterns": len(chinese_config.entity_patterns.person),
            "classical_patterns": len(chinese_config.classical_patterns.get("particles", [])),
            "ollama_config": bool(chinese_config.ollama_config.get("classical_chinese_model")),
            "processing_settings": chinese_config.processing_settings.use_enhanced_extraction
        }
        
        test_results["results"]["configuration"] = {
            "success": True,
            "config_tests": config_tests
        }
        logger.info("✅ Configuration test completed")
        
        # Test 2: File extraction test
        logger.info("📋 Test 2: Testing file extraction")
        file_agent = EnhancedFileExtractionAgent()
        
        extraction_request = AnalysisRequest(
            request_id=f"classical_chinese_test_{int(time.time())}",
            content=str(pdf_path),
            data_type=DataType.PDF,
            language="zh",
            analysis_type="extraction"
        )
        
        extraction_result = await file_agent.process(extraction_request)
        
        test_results["results"]["file_extraction"] = {
            "success": extraction_result.status == "completed",
            "result": {
                "status": extraction_result.status,
                "pages_processed": len(extraction_result.pages) if extraction_result.pages else 0,
                "content_length": len(extraction_result.extracted_text) if extraction_result.extracted_text else 0,
                "processing_time": extraction_result.processing_time
            }
        }
        logger.info("✅ File extraction test completed")
        
        # Test 3: Knowledge graph processing
        logger.info("📋 Test 3: Testing knowledge graph processing")
        kg_agent = KnowledgeGraphAgent()
        
        if extraction_result.extracted_text:
            kg_request = AnalysisRequest(
                request_id=f"classical_chinese_kg_{int(time.time())}",
                content=extraction_result.extracted_text,
                data_type=DataType.TEXT,
                language="zh",
                analysis_type="knowledge_graph"
            )
            
            kg_result = await kg_agent.process(kg_request)
            
            test_results["results"]["knowledge_graph"] = {
                "success": kg_result.status == "completed",
                "result": {
                    "status": kg_result.status,
                    "entities_found": kg_result.metadata.get('statistics', {}).get('entities_found', 0) if kg_result.metadata else 0,
                    "nodes": kg_result.metadata.get('statistics', {}).get('nodes', 0) if kg_result.metadata else 0,
                    "edges": kg_result.metadata.get('statistics', {}).get('edges', 0) if kg_result.metadata else 0,
                    "processing_time": kg_result.processing_time
                }
            }
            logger.info("✅ Knowledge graph test completed")
        else:
            test_results["results"]["knowledge_graph"] = {
                "success": False,
                "error": "No extracted text available"
            }
            logger.warning("⚠️ Knowledge graph test skipped - no extracted text")
        
        # Test 4: Report generation
        logger.info("📋 Test 4: Testing report generation")
        try:
            output_path = f"Results/classical_chinese_report_{int(time.time())}"
            report_result = await kg_agent.generate_graph_report(
                output_path=output_path,
                target_language="zh"
            )
            
            test_results["results"]["report_generation"] = {
                "success": report_result.success if hasattr(report_result, 'success') else True,
                "output_path": output_path,
                "processing_time": report_result.processing_time if hasattr(report_result, 'processing_time') else 0.0
            }
            logger.info("✅ Report generation test completed")
        except Exception as e:
            test_results["results"]["report_generation"] = {
                "success": False,
                "error": str(e)
            }
            logger.error(f"❌ Report generation failed: {e}")
        
        # Save test results
        results_dir = Path("Results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f"classical_chinese_simple_test_results_{timestamp}.json"
        
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
                error_msg = result.get("error", "Unknown error")
                logger.info(f"❌ {test_name}: FAILED - {error_msg}")
        logger.info("=" * 50)
        
        logger.info("🎉 Classical Chinese PDF Processing Test completed!")
        
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
