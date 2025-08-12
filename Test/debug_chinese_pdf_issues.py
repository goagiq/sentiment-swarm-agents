#!/usr/bin/env python3
"""
Debug script to identify issues with Chinese PDF processing and graph visualization.
"""

import asyncio
from pathlib import Path
from loguru import logger

from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.core.models import AnalysisRequest, DataType


async def test_chinese_pdf_processing():
    """Test Chinese PDF processing to identify issues."""
    
    # Initialize the knowledge graph agent
    agent = KnowledgeGraphAgent()
    
    # Test with a simple Chinese text first
    test_text = """
    中国科技巨头阿里巴巴集团今日宣布，将在人工智能领域投入1000亿元人民币。
    预计该项目将创造超过10万个就业岗位，推动中国数字经济发展。
    在欧洲市场，华为与德国电信、法国Orange等运营商保持良好的合作关系。
    中国科学院的量子计算研究团队在量子计算领域取得重要进展。
    该团队由李国杰院士领导，在量子算法优化方面提出创新方法。
    研究成果已发表在《Nature》和《Science》等国际顶级期刊。
    中国的主要科技公司包括阿里巴巴、腾讯、百度、华为等。这些公司在云计算、大数据、物联网等领域都有重要布局。
    中国也在量子通信和量子计算方面取得了重要进展，潘建伟院士团队在量子通信领域处于世界领先地位。
    """
    
    logger.info("Testing Chinese text processing...")
    
    # Create analysis request
    request = AnalysisRequest(
        content=test_text,
        data_type=DataType.TEXT,
        language="zh",
        metadata={
            "source": "test_chinese_text",
            "domain": "technology"
        }
    )
    
    try:
        # Process the request
        result = await agent.process(request)
        
        logger.info(f"Processing completed. Status: {result.status}")
        
        # Get entities and relationships from metadata
        entities_count = result.metadata.get("entities_extracted", 0)
        relationships_count = result.metadata.get("relationships_extracted", 0)
        
        logger.info(f"Entities extracted: {entities_count}")
        logger.info(f"Relationships mapped: {relationships_count}")
        
        # Generate graph report
        logger.info("Generating graph report...")
        report_result = await agent.generate_graph_report(
            output_path="Test/debug_chinese_report.html",
            target_language="en"
        )
        
        # Extract file paths from the result
        if report_result and "content" in report_result:
            content = report_result["content"][0].get("json", {})
            html_file = content.get("html_file")
            png_file = content.get("png_file")
            
            logger.info(f"Report result: {content.get('message', 'N/A')}")
            logger.info(f"HTML file: {html_file}")
            logger.info(f"PNG file: {png_file}")
            
            # Check the generated HTML file
            if html_file:
                html_path = Path(html_file)
                if html_path.exists():
                    logger.info(f"HTML file created successfully: {html_path}")
                    
                    # Read and check the HTML content
                    with open(html_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    # Check for tooltip functionality
                    if 'tooltip' in html_content and 'mouseover' in html_content:
                        logger.info("✓ Tooltip functionality found in HTML")
                    else:
                        logger.warning("✗ Tooltip functionality missing from HTML")
                    
                    # Check for node data
                    if 'const nodes =' in html_content:
                        logger.info("✓ Node data found in HTML")
                        
                        # Check if nodes have proper data structure
                        if '"id":' in html_content and '"type":' in html_content:
                            logger.info("✓ Node data structure looks correct")
                        else:
                            logger.warning("✗ Node data structure may be incomplete")
                    else:
                        logger.warning("✗ Node data missing from HTML")
                        
                else:
                    logger.error(f"✗ HTML file was not created at {html_path}")
            else:
                logger.error("✗ No HTML file path returned in result")
                
        else:
            logger.error("✗ Invalid report result format")
            
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


async def test_entity_extraction_methods():
    """Test different entity extraction methods."""
    
    agent = KnowledgeGraphAgent()
    
    test_text = "中国科技巨头阿里巴巴集团今日宣布，将在人工智能领域投入1000亿元人民币。"
    
    logger.info("Testing entity extraction methods...")
    
    # Test the main extract_entities method
    try:
        entities_result = await agent.extract_entities(test_text, language="zh")
        logger.info(f"Main extraction result: {len(entities_result.get('entities', []))} entities")
        
        # Test enhanced Chinese extraction
        if hasattr(agent, 'enhanced_chinese_extractor'):
            enhanced_result = await agent.enhanced_chinese_extractor.extract_entities_enhanced(test_text)
            logger.info(f"Enhanced extraction result: {len(enhanced_result)} entities")
        
    except Exception as e:
        logger.error(f"Error in entity extraction: {e}")


async def main():
    """Main test function."""
    logger.info("Starting Chinese PDF processing debug...")
    
    await test_entity_extraction_methods()
    await test_chinese_pdf_processing()
    
    logger.info("Debug testing completed.")


if __name__ == "__main__":
    asyncio.run(main())
