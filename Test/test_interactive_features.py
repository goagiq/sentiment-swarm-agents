#!/usr/bin/env python3
"""
Test script to verify Chinese PDF processing and interactive visualization features.
"""

import asyncio
from pathlib import Path
from loguru import logger

from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
from src.core.models import AnalysisRequest, DataType
from src.config.language_specific_config import should_use_enhanced_extraction


async def test_chinese_pdf_processing():
    """Test Chinese PDF processing to ensure it still works."""
    logger.info("=== Testing Chinese PDF Processing ===")
    
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
    entities = result.metadata.get('entities', [])
    relationships = result.metadata.get('relationships', [])
    
    logger.info(f"Entities extracted: {len(entities)}")
    logger.info(f"Relationships extracted: {len(relationships)}")
    
    # Check if Chinese entities were extracted
    chinese_entities = [e for e in entities if any('\u4e00' <= char <= '\u9fff' for char in str(e.get('text', '')))]
    logger.info(f"Chinese entities found: {len(chinese_entities)}")
    
    if chinese_entities:
        logger.info("✅ Chinese entity extraction working")
        for i, entity in enumerate(chinese_entities[:5]):  # Show first 5
            logger.info(f"  {i+1}. {entity.get('text', 'N/A')} ({entity.get('type', 'N/A')})")
    else:
        logger.warning("⚠️ No Chinese entities found")
    
    return result


async def test_interactive_visualization():
    """Test interactive visualization features."""
    logger.info("=== Testing Interactive Visualization ===")
    
    # Initialize agent
    agent = KnowledgeGraphAgent()
    
    # Generate graph report
    logger.info("Generating interactive graph report...")
    report_result = await agent.generate_graph_report(
        output_path="interactive_test_report",
        target_language="zh"
    )
    
    if report_result:
        logger.info("✅ Graph report generated successfully")
        
        # Check if files exist
        html_file = Path("Results/reports/interactive_test_report.html")
        png_file = Path("Results/reports/interactive_test_report.png")
        
        logger.info(f"HTML file exists: {html_file.exists()}")
        logger.info(f"PNG file exists: {png_file.exists()}")
        
        # Check HTML content for interactive features
        if html_file.exists():
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
                
            # Check for zoom controls
            has_zoom_controls = 'zoom-controls' in html_content
            has_zoom_in = 'zoomIn' in html_content
            has_zoom_out = 'zoomOut' in html_content
            has_reset_zoom = 'resetZoom' in html_content
            
            logger.info(f"Zoom controls: {has_zoom_controls}")
            logger.info(f"Zoom in button: {has_zoom_in}")
            logger.info(f"Zoom out button: {has_zoom_out}")
            logger.info(f"Reset zoom button: {has_reset_zoom}")
            
            # Check for tooltip functionality
            has_tooltip = 'tooltip' in html_content.lower()
            has_mouseover = 'mouseover' in html_content
            has_mouseout = 'mouseout' in html_content
            
            logger.info(f"Tooltip functionality: {has_tooltip}")
            logger.info(f"Mouseover events: {has_mouseover}")
            logger.info(f"Mouseout events: {has_mouseout}")
            
            # Check for D3.js zoom behavior
            has_d3_zoom = 'd3.zoom()' in html_content
            has_zoom_behavior = 'zoom.scaleBy' in html_content
            
            logger.info(f"D3.js zoom: {has_d3_zoom}")
            logger.info(f"Zoom behavior: {has_zoom_behavior}")
            
            # Check for Chinese content
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in html_content)
            logger.info(f"Chinese content in HTML: {has_chinese}")
            
            # Overall assessment
            if has_zoom_controls and has_tooltip and has_d3_zoom:
                logger.info("✅ All interactive features present")
            else:
                logger.warning("⚠️ Some interactive features missing")
                
        else:
            logger.error("❌ HTML file not created")
    else:
        logger.error("❌ Failed to generate graph report")


async def test_language_configuration():
    """Test language-specific configuration."""
    logger.info("=== Testing Language Configuration ===")
    
    languages = ["zh", "zh-cn", "en", "ru", "ja"]
    
    for lang in languages:
        uses_enhanced = should_use_enhanced_extraction(lang)
        logger.info(f"Language {lang}: enhanced extraction = {uses_enhanced}")
        
        if lang == "zh" and not uses_enhanced:
            logger.warning(f"⚠️ Chinese ({lang}) should use enhanced extraction")


async def main():
    """Run all tests."""
    logger.info("Starting comprehensive interactive features test...")
    
    # Test 1: Language configuration
    await test_language_configuration()
    
    # Test 2: Chinese PDF processing
    await test_chinese_pdf_processing()
    
    # Test 3: Interactive visualization
    await test_interactive_visualization()
    
    logger.info("=== Test Summary ===")
    logger.info("All tests completed. Check the generated files:")
    logger.info("- Results/reports/interactive_test_report.html")
    logger.info("- Results/reports/interactive_test_report.png")


if __name__ == "__main__":
    asyncio.run(main())
