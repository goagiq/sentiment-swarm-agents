#!/usr/bin/env python3
"""
Script to process Russian Oliver Excerpt PDF and add to vector/knowledge graph 
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


async def process_russian_pdf_to_database(pdf_path: str) -> bool:
    """Process the Russian Oliver Excerpt PDF and add to vector/knowledge graph 
    database."""
    
    logger.info("Processing Russian PDF: {}".format(pdf_path))
    
    # Initialize optimized file extraction agent for Russian processing
    agent = FileExtractionAgent(
        agent_id="russian_oliver_agent",
        model_name="llava:latest",
        max_workers=4,
        chunk_size=2,
        enable_chroma_storage=True
    )
    
    # Create analysis request for Russian PDF
    request = AnalysisRequest(
        id="russian_oliver_excerpt",
        data_type=DataType.PDF,
        content=pdf_path,
        language="ru"  # Explicitly set Russian language
    )
    
    try:
        # Process the PDF
        logger.info("Starting Russian PDF extraction for: {}".format(pdf_path))
        result = await agent.process(request)
        
        if result.status == "completed":
            logger.info("✅ Successfully processed Russian PDF: {}".format(pdf_path))
            logger.info("   Pages extracted: {}".format(len(result.pages)))
            logger.info("   Processing time: {:.2f}s".format(result.processing_time))
            logger.info("   Quality score: {:.2f}".format(result.quality_score))
            
            # Add to knowledge graph
            await add_russian_content_to_knowledge_graph(result, pdf_path)
            
            return True
        else:
            logger.error("❌ Failed to process Russian PDF {}: {}".format(
                pdf_path, result.status))
            return False
            
    except Exception as e:
        logger.error("❌ Error processing Russian PDF {}: {}".format(pdf_path, e))
        return False


async def add_russian_content_to_knowledge_graph(result, pdf_path: str):
    """Add extracted Russian content to knowledge graph."""
    
    logger.info("Adding Russian content from {} to knowledge graph...".format(
        pdf_path))
    
    try:
        # Initialize knowledge graph agent
        kg_agent = KnowledgeGraphAgent()
        
        # Process each page with the knowledge graph agent
        for page in result.pages:
            if page.content and not page.error_message:
                # Create analysis request for the page with Russian language
                page_request = AnalysisRequest(
                    id="russian_page_{}".format(page.page_number),
                    data_type=DataType.TEXT,
                    content=page.content,
                    language="ru"  # Explicitly set Russian language
                )
                
                # Process with knowledge graph agent
                kg_result = await kg_agent.process(page_request)
                
                if kg_result.status == "completed":
                    logger.info("✅ Processed Russian page {}".format(
                        page.page_number))
                else:
                    logger.warning("⚠️ Failed to process Russian page {}: {}".format(
                        page.page_number, kg_result.status))
        
        logger.info("✅ Added Russian content from {} to knowledge graph".format(
            pdf_path))
        
    except Exception as e:
        logger.error("❌ Error adding Russian content to knowledge graph: {}".format(e))


async def generate_comprehensive_graph_report():
    """Generate a comprehensive knowledge graph report including Russian content."""
    
    logger.info("Generating comprehensive knowledge graph report...")
    
    try:
        # Initialize knowledge graph agent
        kg_agent = KnowledgeGraphAgent()
        
        # Generate graph report using the correct method
        report_result = await kg_agent.generate_graph_report(
            output_path="russian_oliver_knowledge_graph_report",
            target_language="en"
        )
        
        # Extract the report content
        if report_result and 'content' in report_result:
            report_content = report_result['content'][0].get('json', {})
            
            # Save report to Results directory
            report_path = Path("Results/russian_oliver_knowledge_graph_report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(report_content, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Knowledge graph report saved to: {report_path}")
            
            # Display comprehensive summary
            logger.info("\n" + "="*70)
            logger.info("RUSSIAN OLIVER EXCERPT KNOWLEDGE GRAPH REPORT")
            logger.info("="*70)
            
            # Display graph statistics
            if 'graph_stats' in report_content:
                stats = report_content['graph_stats']
                logger.info(f"📊 GRAPH STATISTICS:")
                logger.info(f"   Total nodes: {stats.get('total_nodes', 0)}")
                logger.info(f"   Total edges: {stats.get('total_edges', 0)}")
                logger.info(f"   Graph density: {stats.get('density', 0):.4f}")
                logger.info(f"   Average degree: {stats.get('avg_degree', 0):.2f}")
                
                if 'language_breakdown' in stats:
                    lang_breakdown = stats['language_breakdown']
                    logger.info(f"   Language breakdown: {lang_breakdown}")
                
                if 'entity_types' in stats:
                    entity_types = stats['entity_types']
                    logger.info(f"   Entity types: {entity_types}")
            
            # Display report files generated
            logger.info(f"\n📁 REPORT FILES GENERATED:")
            if report_content.get('png_file'):
                logger.info(f"   PNG visualization: {report_content['png_file']}")
            if report_content.get('html_file'):
                logger.info(f"   HTML report: {report_content['html_file']}")
            if report_content.get('md_file'):
                logger.info(f"   Markdown report: {report_content['md_file']}")
            
            logger.info(f"\n✅ Report generation completed successfully!")
            logger.info(f"📁 Report saved to: {report_path}")
            
            return report_content
        else:
            logger.error("❌ Failed to generate report - no content returned")
            return None
        
    except Exception as e:
        logger.error(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Main function to process Russian Oliver PDF and generate report."""
    
    logger.info("🚀 Starting Russian Oliver Excerpt PDF Processing and Knowledge Graph Generation")
    logger.info("="*80)
    
    # Define the Russian PDF to process
    pdf_path = "data/Russian_Oliver_Excerpt.pdf"
    
    # Check if file exists
    if not Path(pdf_path).exists():
        logger.error(f"❌ Russian PDF file not found: {pdf_path}")
        logger.error("Please ensure the file exists in the data directory")
        return
    
    logger.info(f"📄 Processing Russian PDF: {pdf_path}")
    logger.info("-" * 60)
    
    # Process the Russian PDF
    success = await process_russian_pdf_to_database(pdf_path)
    
    if success:
        logger.info(f"\n✅ Russian PDF processing completed successfully!")
        
        # Generate comprehensive knowledge graph report
        logger.info(f"\n📈 Generating comprehensive knowledge graph report...")
        report = await generate_comprehensive_graph_report()
        
        if report:
            logger.info("\n🎉 SUCCESS: Russian Oliver Excerpt PDF processed and report generated!")
            logger.info("📊 The knowledge graph now includes Russian content analysis")
            logger.info("🔍 You can explore the relationships and entities in the generated report")
        else:
            logger.error("❌ Failed to generate knowledge graph report")
    else:
        logger.error("❌ Russian PDF processing failed")


if __name__ == "__main__":
    asyncio.run(main())
