#!/usr/bin/env python3
"""
Script to process all PDFs in the data directory and add them to vector database 
and knowledge graph. Then generates a comprehensive graph report.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import after path modification
from agents.unified_file_extraction_agent import UnifiedFileExtractionAgent
from agents.enhanced_knowledge_graph_agent import EnhancedKnowledgeGraphAgent
from agents.knowledge_graph_coordinator import KnowledgeGraphCoordinator
from core.vector_db import vector_db
from core.improved_knowledge_graph_utility import ImprovedKnowledgeGraphUtility
from config.config import config
from core.models import AnalysisRequest, DataType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Process all PDFs in the data directory and build knowledge graph."""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.results_dir = Path("Results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize agents
        self.file_extraction_agent = UnifiedFileExtractionAgent()
        self.knowledge_graph_agent = EnhancedKnowledgeGraphAgent()
        self.knowledge_graph_coordinator = KnowledgeGraphCoordinator()
        self.knowledge_graph_utility = ImprovedKnowledgeGraphUtility()
        
        # Track processing results
        self.processed_files = []
        self.failed_files = []
        self.extracted_texts = []
        
    async def find_pdf_files(self) -> List[Path]:
        """Find all PDF files in the data directory."""
        pdf_files = []
        
        # Look for PDF files in the data directory
        for pdf_file in self.data_dir.glob("*.pdf"):
            if pdf_file.is_file():
                pdf_files.append(pdf_file)
                logger.info(f"Found PDF: {pdf_file}")
        
        # Also check for PDF files in subdirectories
        for pdf_file in self.data_dir.rglob("*.pdf"):
            if pdf_file.is_file() and pdf_file not in pdf_files:
                pdf_files.append(pdf_file)
                logger.info(f"Found PDF in subdirectory: {pdf_file}")
        
        logger.info(f"Total PDF files found: {len(pdf_files)}")
        return pdf_files
    
    async def process_single_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Process a single PDF file."""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Create analysis request
            request = AnalysisRequest(
                content=str(pdf_path),
                data_type=DataType.PDF,
                language="auto",
                request_id=f"pdf_processing_{pdf_path.stem}"
            )
            
            # Extract text from PDF
            extraction_result = await self.file_extraction_agent.process(request)
            
            if extraction_result.status == "completed":
                logger.info(f"Successfully extracted text from {pdf_path}")
                
                # Store in vector database
                await self._store_in_vector_db(extraction_result, pdf_path)
                
                # Add to knowledge graph
                await self._add_to_knowledge_graph(extraction_result, pdf_path)
                
                return {
                    "file": str(pdf_path),
                    "status": "success",
                    "extracted_text": extraction_result.extracted_text,
                    "metadata": extraction_result.metadata
                }
            else:
                logger.error(f"Failed to extract text from {pdf_path}: {extraction_result.error_message}")
                return {
                    "file": str(pdf_path),
                    "status": "failed",
                    "error": extraction_result.error_message
                }
                
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {
                "file": str(pdf_path),
                "status": "error",
                "error": str(e)
            }
    
    async def _store_in_vector_db(self, extraction_result, pdf_path: Path):
        """Store extracted text in vector database."""
        try:
            # Add to vector database
            await vector_db.add_texts(
                texts=[extraction_result.extracted_text],
                metadatas=[{
                    "source": str(pdf_path),
                    "type": "pdf",
                    "extraction_method": extraction_result.metadata.get("extraction_method", "unknown"),
                    "language": extraction_result.metadata.get("language", "unknown"),
                    "page_count": extraction_result.metadata.get("page_count", 0),
                    "processing_timestamp": extraction_result.metadata.get("processing_timestamp", "")
                }]
            )
            logger.info(f"Stored {pdf_path} in vector database")
        except Exception as e:
            logger.error(f"Error storing {pdf_path} in vector database: {str(e)}")
    
    async def _add_to_knowledge_graph(self, extraction_result, pdf_path: Path):
        """Add extracted content to knowledge graph."""
        try:
            # Process with knowledge graph agent
            kg_result = await self.knowledge_graph_agent.process_content(
                content=extraction_result.extracted_text,
                source=str(pdf_path),
                language=extraction_result.metadata.get("language", "en")
            )
            
            if kg_result:
                logger.info(f"Added {pdf_path} to knowledge graph")
                self.extracted_texts.append(extraction_result.extracted_text)
            else:
                logger.warning(f"Failed to add {pdf_path} to knowledge graph")
                
        except Exception as e:
            logger.error(f"Error adding {pdf_path} to knowledge graph: {str(e)}")
    
    async def process_all_pdfs(self):
        """Process all PDF files in the data directory."""
        logger.info("Starting PDF processing...")
        
        # Find all PDF files
        pdf_files = await self.find_pdf_files()
        
        if not pdf_files:
            logger.warning("No PDF files found in data directory")
            return
        
        # Process each PDF file
        for pdf_file in pdf_files:
            result = await self.process_single_pdf(pdf_file)
            
            if result["status"] == "success":
                self.processed_files.append(result)
            else:
                self.failed_files.append(result)
        
        logger.info(f"Processing complete. Success: {len(self.processed_files)}, Failed: {len(self.failed_files)}")
    
    async def generate_graph_report(self):
        """Generate a comprehensive graph report."""
        try:
            logger.info("Generating comprehensive graph report...")
            
            # Generate report using knowledge graph coordinator
            report_path = self.results_dir / f"comprehensive_pdf_graph_report_{config.get_timestamp()}.html"
            
            report_result = await self.knowledge_graph_coordinator.generate_comprehensive_report(
                output_path=str(report_path),
                include_statistics=True,
                include_visualizations=True,
                include_entity_analysis=True,
                include_relationship_analysis=True,
                include_community_detection=True
            )
            
            if report_result:
                logger.info(f"Graph report generated successfully: {report_path}")
                return str(report_path)
            else:
                logger.error("Failed to generate graph report")
                return None
                
        except Exception as e:
            logger.error(f"Error generating graph report: {str(e)}")
            return None
    
    async def generate_processing_summary(self):
        """Generate a summary of the processing results."""
        summary = {
            "total_files": len(self.processed_files) + len(self.failed_files),
            "successful_files": len(self.processed_files),
            "failed_files": len(self.failed_files),
            "success_rate": len(self.processed_files) / (len(self.processed_files) + len(self.failed_files)) * 100 if (len(self.processed_files) + len(self.failed_files)) > 0 else 0,
            "processed_files": [f["file"] for f in self.processed_files],
            "failed_files": [f["file"] for f in self.failed_files],
            "total_extracted_texts": len(self.extracted_texts)
        }
        
        # Save summary to file
        summary_path = self.results_dir / f"pdf_processing_summary_{config.get_timestamp()}.json"
        import json
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processing summary saved to: {summary_path}")
        return summary


async def main():
    """Main function to process all PDFs and generate report."""
    processor = PDFProcessor()
    
    try:
        # Process all PDFs
        await processor.process_all_pdfs()
        
        # Generate graph report
        report_path = await processor.generate_graph_report()
        
        # Generate processing summary
        summary = await processor.generate_processing_summary()
        
        # Print results
        print("\n" + "="*60)
        print("PDF PROCESSING COMPLETE")
        print("="*60)
        print(f"Total files processed: {summary['total_files']}")
        print(f"Successful: {summary['successful_files']}")
        print(f"Failed: {summary['failed_files']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"Total texts extracted: {summary['total_extracted_texts']}")
        
        if report_path:
            print(f"\nGraph report generated: {report_path}")
        
        print(f"\nProcessing summary saved: {processor.results_dir}/pdf_processing_summary_*.json")
        
        if summary['failed_files']:
            print("\nFailed files:")
            for failed_file in summary['failed_files']:
                print(f"  - {failed_file}")
        
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
