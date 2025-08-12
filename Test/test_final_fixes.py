#!/usr/bin/env python3
"""
Final comprehensive test to verify all fixes for Chinese PDF processing and graph visualization.
"""

import asyncio
from pathlib import Path
from loguru import logger

from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.core.models import AnalysisRequest, DataType
from src.config.language_specific_config import should_use_enhanced_extraction
from src.config.font_config import get_font_family, configure_font_for_language, test_font_rendering


async def test_chinese_processing_complete():
    """Test complete Chinese processing pipeline."""
    logger.info("=== Testing Complete Chinese Processing Pipeline ===")
    
    # Initialize agent
    agent = KnowledgeGraphAgent()
    
    # Test text with Chinese content
    test_text = """
    中国科技巨头阿里巴巴集团今日宣布，将在人工智能领域投入1000亿元人民币。
    预计该项目将创造超过10万个就业岗位，推动中国数字经济发展。
    在欧洲市场，华为与德国电信、法国Orange等运营商保持良好的合作关系。
    中国科学院的量子计算研究团队在量子计算领域取得重要进展。
    该团队由李国杰院士领导，在量子算法优化方面提出创新方法。
    研究成果已发表在《Nature》和《Science》等国际顶级期刊。
    中国的量子计算研究已经走在世界前列，潘建伟院士的团队在量子通信方面取得重大突破。
    """
    
    # Create analysis request
    request = AnalysisRequest(
        content=test_text,
        data_type=DataType.TEXT,
        language="zh",
        analysis_type="knowledge_graph"
    )
    
    # Process the request
    logger.info("Processing Chinese text...")
    result = await agent.process_request(request)
    
    # Check results
    logger.info(f"Processing completed successfully")
    logger.info(f"Entities extracted: {len(result.metadata.get('entities', []))}")
    logger.info(f"Relationships extracted: {len(result.metadata.get('relationships', []))}")
    
    # Generate graph report
    logger.info("Generating graph report...")
    report_result = await agent.generate_graph_report(
        output_path="final_chinese_test_report",
        target_language="zh"
    )
    
    if report_result:
        logger.info("✅ Graph report generated successfully")
        
        # Check if files exist
        html_file = Path("Results/reports/final_chinese_test_report.html")
        png_file = Path("Results/reports/final_chinese_test_report.png")
        
        logger.info(f"HTML file exists: {html_file.exists()}")
        logger.info(f"PNG file exists: {png_file.exists()}")
        
        # Check HTML content
        if html_file.exists():
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
                has_tooltip = 'tooltip' in html_content.lower()
                has_chinese = any('\u4e00' <= char <= '\u9fff' for char in html_content)
                logger.info(f"HTML has tooltip: {has_tooltip}")
                logger.info(f"HTML has Chinese content: {has_chinese}")
    else:
        logger.error("❌ Failed to generate graph report")


async def test_font_configuration():
    """Test font configuration for different languages."""
    logger.info("=== Testing Font Configuration ===")
    
    languages = ["zh", "ja", "ko", "ru", "en"]
    
    for lang in languages:
        logger.info(f"Testing font for language: {lang}")
        
        # Get font family
        font_family = get_font_family(lang)
        logger.info(f"  Font family: {font_family}")
        
        # Configure font
        configured = configure_font_for_language(lang)
        logger.info(f"  Configured: {configured}")
        
        # Test rendering
        can_render = test_font_rendering(lang)
        logger.info(f"  Can render: {can_render}")
        
        if lang == "zh":
            if can_render:
                logger.info("✅ Chinese font rendering works")
            else:
                logger.warning("⚠️ Chinese font rendering may have issues")


async def test_language_specific_config():
    """Test language-specific configuration."""
    logger.info("=== Testing Language-Specific Configuration ===")
    
    languages = ["zh", "zh-cn", "en", "ru", "ja"]
    
    for lang in languages:
        uses_enhanced = should_use_enhanced_extraction(lang)
        logger.info(f"Language {lang}: enhanced extraction = {uses_enhanced}")


async def main():
    """Run all tests."""
    logger.info("Starting final comprehensive test...")
    
    # Test 1: Language-specific configuration
    await test_language_specific_config()
    
    # Test 2: Font configuration
    await test_font_configuration()
    
    # Test 3: Complete Chinese processing
    await test_chinese_processing_complete()
    
    logger.info("=== Test Summary ===")
    logger.info("All tests completed. Check the generated files:")
    logger.info("- Results/reports/final_chinese_test_report.html")
    logger.info("- Results/reports/final_chinese_test_report.png")


if __name__ == "__main__":
    asyncio.run(main())
