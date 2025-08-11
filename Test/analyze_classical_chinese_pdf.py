#!/usr/bin/env python3
"""
Script to analyze Classical Chinese PDF file page by page.
Extracts text from each page and performs sentiment analysis.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.file_extraction_agent import FileExtractionAgent
from src.agents.text_agent import TextAgent
from src.core.models import AnalysisRequest, DataType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClassicalChineseAnalyzer:
    """Analyzer for Classical Chinese PDF documents."""
    
    def __init__(self):
        self.file_agent = FileExtractionAgent()
        self.text_agent = TextAgent()
        
    async def analyze_pdf_pages(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze each page of the PDF file."""
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        logger.info(f"Starting analysis of PDF: {pdf_file.name}")
        
        # Create analysis request for PDF extraction
        extraction_request = AnalysisRequest(
            request_id=f"pdf_analysis_{pdf_file.stem}",
            content=str(pdf_file),
            data_type=DataType.PDF,
            language="zh",  # Chinese
            analysis_type="extraction",
            metadata={
                "source": "classical_chinese_pdf",
                "filename": pdf_file.name
            }
        )
        
        # Extract text from PDF
        logger.info("Extracting text from PDF...")
        extraction_result = await self.file_agent.process(extraction_request)
        
        if extraction_result.status != "completed":
            error_msg = extraction_result.metadata.get("error", "Unknown error")
            logger.error(f"PDF extraction failed: {error_msg}")
            return {
                "error": "PDF extraction failed", 
                "details": error_msg
            }
        
        # Get the extracted content
        extracted_text = extraction_result.extracted_text or ""
        
        # Split the extracted text into pages based on page markers
        if extracted_text:
            # Split by page markers
            page_splits = extracted_text.split("--- Page")
            pages = []
            for i, split in enumerate(page_splits):
                if split.strip():
                    # Remove the page number and clean up
                    if "---" in split:
                        content = split.split("---", 1)[1].strip()
                    else:
                        content = split.strip()
                    if content:
                        pages.append(content)
            
            logger.info(f"Successfully extracted {len(pages)} pages from text")
        else:
            pages = []
            logger.info("No text content extracted")
        
        # If no pages were extracted, try to get the raw content
        if not pages and extraction_result.raw_content:
            pages = [extraction_result.raw_content]
            logger.info("Using raw content as single page")
        
        # Analyze each page
        page_analyses = []
        for page_num, page_content in enumerate(pages, 1):
            logger.info(f"Analyzing page {page_num}...")
            
            page_analysis = await self._analyze_single_page(
                page_num, page_content, {"pages": pages}
            )
            page_analyses.append(page_analysis)
        
        # Create comprehensive summary
        comprehensive_analysis = await self._create_comprehensive_summary(
            pdf_file.name, page_analyses, {"pages": pages}
        )
        
        return {
            "pdf_filename": pdf_file.name,
            "total_pages": len(pages),
            "page_analyses": page_analyses,
            "comprehensive_analysis": comprehensive_analysis,
            "extraction_metadata": extraction_result.metadata
        }
    
    async def _analyze_single_page(self, page_num: int, page_content: str, 
                                 extracted_content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single page of the PDF."""
        
        # Create text analysis request for this page
        text_request = AnalysisRequest(
            request_id=f"page_analysis_{page_num}",
            content=page_content,
            data_type=DataType.TEXT,
            language="zh",  # Chinese
            analysis_type="sentiment_summary",
            metadata={
                "page_number": page_num,
                "source": "classical_chinese_pdf"
            }
        )
        
        # Perform text analysis
        text_result = await self.text_agent.process(text_request)
        
        if text_result.status != "completed":
            error_msg = text_result.metadata.get("error", "Unknown error")
            logger.warning(f"Page {page_num} analysis failed: {error_msg}")
            return {
                "page_number": page_num,
                "status": "failed",
                "error": error_msg,
                "raw_content": page_content[:500] + "..." if len(page_content) > 500 else page_content
            }
        
        # Extract analysis results
        analysis_data = text_result.metadata
        
        return {
            "page_number": page_num,
            "status": "completed",
            "content_length": len(page_content),
            "raw_content": page_content[:500] + "..." if len(page_content) > 500 else page_content,
            "summary": analysis_data.get("summary", ""),
            "sentiment": analysis_data.get("sentiment", {}),
            "key_themes": analysis_data.get("key_themes", []),
            "language_detected": analysis_data.get("language_detected", "zh"),
            "analysis_metadata": analysis_data.get("metadata", {})
        }
    
    async def _create_comprehensive_summary(self, filename: str, 
                                          page_analyses: List[Dict[str, Any]],
                                          extracted_content: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive summary of all pages."""
        
        # Combine all page content for overall analysis
        all_content = "\n\n".join([
            analysis.get("raw_content", "") 
            for analysis in page_analyses 
            if analysis.get("status") == "completed"
        ])
        
        if not all_content.strip():
            return {
                "overall_summary": "No content could be extracted from the PDF",
                "overall_sentiment": {"sentiment": "neutral", "confidence": 0.0},
                "document_themes": [],
                "analysis_status": "failed"
            }
        
        # Create comprehensive analysis request
        comprehensive_request = AnalysisRequest(
            request_id=f"comprehensive_analysis_{filename}",
            content=all_content,
            data_type=DataType.TEXT,
            language="zh",
            analysis_type="comprehensive_analysis",
            metadata={
                "source": "classical_chinese_pdf",
                "filename": filename,
                "total_pages": len(page_analyses)
            }
        )
        
        # Perform comprehensive analysis
        comprehensive_result = await self.text_agent.process(comprehensive_request)
        
        if comprehensive_result.status != "completed":
            error_msg = comprehensive_result.metadata.get("error", "Unknown error")
            logger.warning(f"Comprehensive analysis failed: {error_msg}")
            return {
                "overall_summary": "Comprehensive analysis failed",
                "overall_sentiment": {"sentiment": "neutral", "confidence": 0.0},
                "document_themes": [],
                "analysis_status": "failed",
                "error": error_msg
            }
        
        analysis_data = comprehensive_result.metadata
        
        return {
            "overall_summary": analysis_data.get("summary", ""),
            "overall_sentiment": analysis_data.get("sentiment", {}),
            "document_themes": analysis_data.get("key_themes", []),
            "analysis_status": "completed",
            "total_pages_analyzed": len([a for a in page_analyses if a.get("status") == "completed"]),
            "analysis_metadata": analysis_data.get("metadata", {})
        }


async def main():
    """Main function to run the analysis."""
    pdf_path = "data/Classical Chinese Sample 22208_0_8.pdf"
    
    analyzer = ClassicalChineseAnalyzer()
    
    try:
        logger.info("Starting Classical Chinese PDF analysis...")
        
        # Perform the analysis
        results = await analyzer.analyze_pdf_pages(pdf_path)
        
        # Save results to file
        output_file = Path("Results/classical_chinese_analysis.json")
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Analysis completed. Results saved to: {output_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("CLASSICAL CHINESE PDF ANALYSIS SUMMARY")
        print("="*80)
        
        if "error" in results:
            print(f"Analysis failed: {results['error']}")
            print(f"Details: {results.get('details', 'No details available')}")
            return 1
            
        print(f"PDF File: {results['pdf_filename']}")
        print(f"Total Pages: {results['total_pages']}")
        print(f"Pages Successfully Analyzed: {results['comprehensive_analysis'].get('total_pages_analyzed', 0)}")
        
        if results['comprehensive_analysis'].get('analysis_status') == 'completed':
            print(f"\nOverall Sentiment: {results['comprehensive_analysis'].get('overall_sentiment', {}).get('sentiment', 'unknown')}")
            print(f"Document Themes: {', '.join(results['comprehensive_analysis'].get('document_themes', []))}")
            print(f"\nOverall Summary:")
            print(results['comprehensive_analysis'].get('overall_summary', 'No summary available'))
        
        print("\n" + "="*80)
        print("DETAILED PAGE-BY-PAGE ANALYSIS")
        print("="*80)
        
        for page_analysis in results['page_analyses']:
            print(f"\nPage {page_analysis['page_number']}:")
            print(f"  Status: {page_analysis['status']}")
            if page_analysis['status'] == 'completed':
                print(f"  Sentiment: {page_analysis['sentiment'].get('sentiment', 'unknown')}")
                print(f"  Key Themes: {', '.join(page_analysis['key_themes'])}")
                print(f"  Summary: {page_analysis['summary'][:200]}...")
            else:
                print(f"  Error: {page_analysis.get('error', 'Unknown error')}")
        
        print(f"\nDetailed results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
