#!/usr/bin/env python3
"""
Script to process Classical Chinese Sample PDF and add to vector/knowledge graph 
database. Then generate a comprehensive graph report.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the parent directory to the path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.file_extraction_agent import FileExtractionAgent
from src.core.models import AnalysisRequest, DataType
from src.agents.knowledge_graph_agent import KnowledgeGraphAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def process_chinese_pdf_to_database(pdf_path: str) -> bool:
    """Process the Classical Chinese Sample PDF and add to vector/knowledge graph 
    database."""
    
    logger.info("Processing Classical Chinese PDF: {}".format(pdf_path))
    
    # Initialize optimized file extraction agent for Chinese processing
    agent = FileExtractionAgent(
        agent_id="classical_chinese_agent",
        model_name="llava:latest",
        max_workers=4,
        chunk_size=2,
        enable_chroma_storage=True
    )
    
    # Create analysis request for Chinese PDF
    request = AnalysisRequest(
        id="classical_chinese_sample",
        data_type=DataType.PDF,
        content=pdf_path,
        language="zh"  # Explicitly set Chinese language
    )
    
    try:
        # Process the PDF
        logger.info("Starting Classical Chinese PDF extraction for: {}".format(pdf_path))
        result = await agent.process(request)
        
        if result.status == "completed":
            logger.info("✅ Successfully processed Classical Chinese PDF: {}".format(pdf_path))
            logger.info("   Pages extracted: {}".format(len(result.pages)))
            logger.info("   Processing time: {:.2f}s".format(result.processing_time))
            logger.info("   Quality score: {:.2f}".format(result.quality_score))
            
            # Add to knowledge graph
            await add_chinese_content_to_knowledge_graph(result, pdf_path)
            
            return True
        else:
            logger.error("❌ Failed to process Classical Chinese PDF {}: {}".format(
                pdf_path, result.status))
            return False
            
    except Exception as e:
        logger.error("❌ Error processing Classical Chinese PDF {}: {}".format(pdf_path, e))
        return False


async def add_chinese_content_to_knowledge_graph(result, pdf_path: str):
    """Add extracted Classical Chinese content to knowledge graph."""
    
    logger.info("Adding Classical Chinese content from {} to knowledge graph...".format(
        pdf_path))
    
    try:
        # Initialize knowledge graph agent
        kg_agent = KnowledgeGraphAgent()
        
        # Process each page with the knowledge graph agent
        for page in result.pages:
            if page.content and not page.error_message:
                # Create analysis request for the page with Chinese language
                page_request = AnalysisRequest(
                    id="chinese_page_{}".format(page.page_number),
                    data_type=DataType.TEXT,
                    content=page.content,
                    language="zh"  # Explicitly set Chinese language
                )
                
                # Process with knowledge graph agent
                kg_result = await kg_agent.process(page_request)
                
                if kg_result.status == "completed":
                    logger.info("✅ Processed Classical Chinese page {}".format(
                        page.page_number))
                else:
                    logger.warning("⚠️ Failed to process Classical Chinese page {}: {}".format(
                        page.page_number, kg_result.status))
        
        logger.info("✅ Added Classical Chinese content from {} to knowledge graph".format(
            pdf_path))
        
    except Exception as e:
        logger.error("❌ Error adding Classical Chinese content to knowledge graph: {}".format(e))


async def generate_comprehensive_graph_report():
    """Generate a comprehensive knowledge graph report including Classical Chinese content."""
    
    logger.info("Generating comprehensive knowledge graph report...")
    
    try:
        # Initialize knowledge graph agent
        kg_agent = KnowledgeGraphAgent()
        
        # Generate graph report using the correct method
        report_result = await kg_agent.generate_graph_report(
            output_path="classical_chinese_knowledge_graph_report",
            target_language="en"
        )
        
        # Extract the report content
        if report_result and 'content' in report_result:
            report_content = report_result['content'][0].get('json', {})
            
            # Save report to Results directory
            report_path = Path("Results/classical_chinese_knowledge_graph_report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(report_content, f, indent=2, ensure_ascii=False)
            
            logger.info("✅ Knowledge graph report saved to: {}".format(report_path))
            
            # Display comprehensive summary
            logger.info("\n" + "="*70)
            logger.info("CLASSICAL CHINESE SAMPLE KNOWLEDGE GRAPH REPORT")
            logger.info("="*70)
            
            # Display graph statistics
            if 'graph_stats' in report_content:
                stats = report_content['graph_stats']
                logger.info("📊 GRAPH STATISTICS:")
                logger.info("   Total nodes: {}".format(stats.get('total_nodes', 0)))
                logger.info("   Total edges: {}".format(stats.get('total_edges', 0)))
                logger.info("   Graph density: {:.4f}".format(stats.get('density', 0)))
                logger.info("   Average degree: {:.2f}".format(stats.get('avg_degree', 0)))
                
                if 'language_breakdown' in stats:
                    lang_breakdown = stats['language_breakdown']
                    logger.info("   Language breakdown: {}".format(lang_breakdown))
                
                if 'entity_types' in stats:
                    entity_types = stats['entity_types']
                    logger.info("   Entity types: {}".format(entity_types))
            
            # Display report files generated
            logger.info("\n📁 REPORT FILES GENERATED:")
            if report_content.get('png_file'):
                logger.info("   PNG visualization: {}".format(report_content['png_file']))
            if report_content.get('html_file'):
                logger.info("   HTML report: {}".format(report_content['html_file']))
            if report_content.get('md_file'):
                logger.info("   Markdown report: {}".format(report_content['md_file']))
            
            logger.info("\n✅ Report generation completed successfully!")
            logger.info("📁 Report saved to: {}".format(report_path))
            
            return report_content
        else:
            logger.error("❌ Failed to generate report - no content returned")
            return None
        
    except Exception as e:
        logger.error("❌ Error generating report: {}".format(e))
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Main function to process Classical Chinese PDF and generate report."""
    
    logger.info("🚀 Starting Classical Chinese Sample PDF Processing and Knowledge Graph Generation")
    logger.info("="*80)
    
    # Define the Chinese PDF to process
    pdf_path = "data/Classical Chinese Sample 22208_0_8.pdf"
    
    # Check if file exists
    if not Path(pdf_path).exists():
        logger.error("❌ Classical Chinese PDF file not found: {}".format(pdf_path))
        logger.error("Please ensure the file exists in the data directory")
        return
    
    logger.info("📄 Processing Classical Chinese PDF: {}".format(pdf_path))
    logger.info("-" * 60)
    
    # Process the Chinese PDF
    success = await process_chinese_pdf_to_database(pdf_path)
    
    if success:
        logger.info("\n✅ Classical Chinese PDF processing completed successfully!")
        
        # Generate comprehensive knowledge graph report
        logger.info("\n📈 Generating comprehensive knowledge graph report...")
        report = await generate_comprehensive_graph_report()
        
        if report:
            logger.info("\n🎉 SUCCESS: Classical Chinese Sample PDF processed and report generated!")
            logger.info("📊 The knowledge graph now includes Classical Chinese content analysis")
            logger.info("🔍 You can explore the relationships and entities in the generated report")
        else:
            logger.error("❌ Failed to generate knowledge graph report")
    else:
        logger.error("❌ Classical Chinese PDF processing failed")


if __name__ == "__main__":
    asyncio.run(main())
