#!/usr/bin/env python3
"""
Fast script to process specific PDFs and add to vector/knowledge graph database.
Optimized for speed using the enhanced FileExtractionAgent.
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


async def process_pdf_to_database(pdf_path: str, language: str = "en") -> bool:
    """Process a single PDF and add to vector/knowledge graph database."""
    
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Initialize optimized file extraction agent
    agent = FileExtractionAgent(
        agent_id=f"fast_agent_{Path(pdf_path).stem}",
        model_name="llava:latest",
        max_workers=4,  # Optimized for speed
        chunk_size=2,   # Process 2 pages at a time
        enable_chroma_storage=True  # Enable storage
    )
    
    # Create analysis request
    request = AnalysisRequest(
        id=f"process_{Path(pdf_path).stem}",
        data_type=DataType.PDF,
        content=pdf_path,
        language=language
    )
    
    try:
        # Process the PDF
        logger.info(f"Starting extraction for: {pdf_path}")
        result = await agent.process(request)
        
        if result.status == "completed":
            logger.info(f"✅ Successfully processed {pdf_path}")
            logger.info(f"   Pages extracted: {len(result.pages)}")
            logger.info(f"   Processing time: {result.processing_time:.2f}s")
            logger.info(f"   Quality score: {result.quality_score:.2f}")
            
            # Add to knowledge graph
            await add_to_knowledge_graph(result, pdf_path)
            
            return True
        else:
            logger.error(f"❌ Failed to process {pdf_path}: {result.status}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error processing {pdf_path}: {e}")
        return False


async def add_to_knowledge_graph(result, pdf_path: str):
    """Add extracted content to knowledge graph."""
    
    logger.info(f"Adding {pdf_path} to knowledge graph...")
    
    try:
        # Initialize knowledge graph agent
        kg_agent = KnowledgeGraphAgent()
        
        # Process each page with the knowledge graph agent
        for page in result.pages:
            if page.content and not page.error_message:
                # Create analysis request for the page
                page_request = AnalysisRequest(
                    id=f"page_{page.page_number}_{Path(pdf_path).stem}",
                    data_type=DataType.TEXT,
                    content=page.content,
                    language="zh" if "chinese" in pdf_path.lower() else "en"
                )
                
                # Process with knowledge graph agent
                kg_result = await kg_agent.process(page_request)
                
                if kg_result.status == "completed":
                    logger.info(f"✅ Processed page {page.page_number}")
        
        logger.info(f"✅ Added {pdf_path} to knowledge graph")
        
    except Exception as e:
        logger.error(f"❌ Error adding to knowledge graph: {e}")


async def generate_graph_report():
    """Generate a comprehensive knowledge graph report."""
    
    logger.info("Generating knowledge graph report...")
    
    try:
        # Initialize knowledge graph agent
        kg_agent = KnowledgeGraphAgent()
        
        # Generate comprehensive report using the agent
        report = await kg_agent.generate_comprehensive_report()
        
        # Save report to Results directory
        report_path = Path("Results/reports/knowledge_graph_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Knowledge graph report saved to: {report_path}")
        
        # Display summary
        logger.info("\n" + "="*60)
        logger.info("KNOWLEDGE GRAPH REPORT SUMMARY")
        logger.info("="*60)
        
        if 'summary' in report:
            summary = report['summary']
            logger.info(f"Total entities: {summary.get('total_entities', 0)}")
            logger.info(f"Total relationships: {summary.get('total_relationships', 0)}")
            logger.info(f"Documents processed: {summary.get('documents_processed', 0)}")
            logger.info(f"Entity types: {summary.get('entity_types', [])}")
        
        if 'top_entities' in report:
            logger.info(f"\nTop entities:")
            for entity in report['top_entities'][:10]:
                logger.info(f"  - {entity.get('name', '')} ({entity.get('type', '')})")
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Error generating report: {e}")
        return None


async def main():
    """Main function to process PDFs and generate report."""
    
    logger.info("🚀 Starting optimized PDF processing and knowledge graph generation")
    logger.info("="*70)
    
    # Define PDFs to process
    pdfs_to_process = [
        ("data/chinese-all.pdf", "zh"),  # Chinese PDF
        ("data/mackenzie_king_pdf.pdf", "en")  # English PDF
    ]
    
    # Check if files exist
    for pdf_path, language in pdfs_to_process:
        if not Path(pdf_path).exists():
            logger.error(f"❌ PDF file not found: {pdf_path}")
            return
    
    # Process each PDF
    successful_processing = 0
    
    for pdf_path, language in pdfs_to_process:
        logger.info(f"\n📄 Processing: {pdf_path}")
        logger.info("-" * 50)
        
        success = await process_pdf_to_database(pdf_path, language)
        if success:
            successful_processing += 1
    
    logger.info(f"\n📊 Processing Summary:")
    logger.info(f"   Total PDFs: {len(pdfs_to_process)}")
    logger.info(f"   Successfully processed: {successful_processing}")
    
    if successful_processing > 0:
        # Generate knowledge graph report
        logger.info(f"\n📈 Generating knowledge graph report...")
        report = await generate_graph_report()
        
        if report:
            logger.info("✅ Knowledge graph report generated successfully!")
        else:
            logger.error("❌ Failed to generate knowledge graph report")
    else:
        logger.error("❌ No PDFs were successfully processed")


if __name__ == "__main__":
    asyncio.run(main())
