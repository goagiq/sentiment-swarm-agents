"""
PDF processing utility for adding content to vector and knowledge graph databases.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

import PyPDF2
from loguru import logger

from .vector_db import VectorDBManager
from .models import AnalysisResult, SentimentResult, ProcessingStatus, SentimentLabel


class PDFProcessor:
    """Utility class for processing PDFs and adding to databases."""
    
    def __init__(self):
        self.vector_db = VectorDBManager()
    
    async def process_pdf_to_databases(
        self,
        pdf_path: str,
        language: str = "en",
        generate_report: bool = True,
        output_path: Optional[str] = None,
        knowledge_graph_agent = None
    ) -> Dict[str, Any]:
        """
        Process PDF file and add content to both vector and knowledge graph databases.
        
        Args:
            pdf_path: Path to the PDF file
            language: Language of the PDF content
            generate_report: Whether to generate a knowledge graph report
            output_path: Custom output path for the report
            knowledge_graph_agent: Knowledge graph agent instance
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Extract text from PDF
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                return {
                    "success": False,
                    "error": f"PDF file not found: {pdf_path}"
                }
            
            # Read PDF content
            text_content = await self._extract_pdf_text(pdf_file)
            if not text_content:
                return {
                    "success": False,
                    "error": "No text content extracted from PDF"
                }
            
            # Store in vector database
            vector_id = await self._store_in_vector_db(
                text_content, pdf_file, language, len(text_content)
            )
            
            # Process with knowledge graph
            kg_result = {}
            if knowledge_graph_agent:
                kg_result = await knowledge_graph_agent.process_content(
                    content=text_content,
                    data_type="text",
                    language=language
                )
            
            # Generate graph report if requested
            report_files = {}
            if generate_report and knowledge_graph_agent:
                report_files = await self._generate_report(
                    knowledge_graph_agent, output_path
                )
            
            return {
                "success": True,
                "vector_database_id": vector_id,
                "knowledge_graph_result": kg_result,
                "content_length": len(text_content),
                "pages": len(text_content.split('\n')),  # Rough estimate
                "report_files": report_files
            }
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _extract_pdf_text(self, pdf_file: Path) -> str:
        """Extract text content from PDF file."""
        try:
            with open(pdf_file, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text_content = ""
                for page in reader.pages:
                    text_content += page.extract_text()
                return text_content.strip()
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {pdf_file}: {e}")
            return ""
    
    async def _store_in_vector_db(
        self, 
        text_content: str, 
        pdf_file: Path, 
        language: str,
        content_length: int
    ) -> str:
        """Store PDF content in vector database."""
        try:
            # Create AnalysisResult for vector database
            result = AnalysisResult(
                id=str(uuid.uuid4()),
                request_id=f"pdf_upload_{datetime.now().isoformat()}",
                data_type="pdf",
                content=text_content,
                language=language,
                processing_status=ProcessingStatus.COMPLETED,
                timestamp=datetime.now().isoformat(),
                processing_time=0.0,
                model_used="pdf_extraction",
                agent_id="pdf_processor",
                sentiment=SentimentResult(
                    label=SentimentLabel.NEUTRAL,
                    confidence=0.8,
                    scores={"positive": 0.3, "negative": 0.2, "neutral": 0.5},
                    reasoning="PDF content - educational/documentary material"
                ),
                metadata={
                    "source_file": pdf_file.name,
                    "content_type": "pdf",
                    "extraction_method": "PyPDF2",
                    "content_length": content_length
                }
            )
            
            # Store in vector database
            vector_id = await self.vector_db.store_result(result)
            logger.info(f"Stored PDF content in vector database with ID: {vector_id}")
            return vector_id
            
        except Exception as e:
            logger.error(f"Failed to store in vector database: {e}")
            raise
    
    async def _generate_report(
        self, 
        knowledge_graph_agent, 
        output_path: Optional[str] = None
    ) -> Dict[str, str]:
        """Generate knowledge graph report."""
        try:
            if not output_path:
                output_path = f"Results/reports/pdf_knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            report_result = await knowledge_graph_agent.generate_graph_report(
                output_path=output_path
            )
            return report_result.get("files", {})
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {}


# Convenience function for easy usage
async def process_pdf_to_databases(
    pdf_path: str,
    language: str = "en",
    generate_report: bool = True,
    output_path: Optional[str] = None,
    knowledge_graph_agent = None
) -> Dict[str, Any]:
    """
    Convenience function to process PDF and add to databases.
    
    Args:
        pdf_path: Path to the PDF file
        language: Language of the PDF content
        generate_report: Whether to generate a knowledge graph report
        output_path: Custom output path for the report
        knowledge_graph_agent: Knowledge graph agent instance
        
    Returns:
        Dictionary with processing results
    """
    processor = PDFProcessor()
    return await processor.process_pdf_to_databases(
        pdf_path=pdf_path,
        language=language,
        generate_report=generate_report,
        output_path=output_path,
        knowledge_graph_agent=knowledge_graph_agent
    )
