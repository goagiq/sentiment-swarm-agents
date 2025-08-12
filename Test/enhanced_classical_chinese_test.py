#!/usr/bin/env python3
"""
Enhanced Classical Chinese PDF Analysis Test
Tests the consolidated MCP server architecture with Classical Chinese processing.
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
from src.agents.unified_text_agent import UnifiedTextAgent
from src.core.models import AnalysisRequest, DataType
from src.config.language_config.chinese_config import ChineseConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedClassicalChineseAnalyzer:
    """Enhanced analyzer for Classical Chinese PDF documents with better error handling."""
    
    def __init__(self):
        self.file_agent = FileExtractionAgent()
        self.text_agent = UnifiedTextAgent(use_strands=True, use_swarm=False)
        self.chinese_config = ChineseConfig()
        
    async def analyze_pdf_pages(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze each page of the PDF file with enhanced error handling."""
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        logger.info(f"Starting enhanced analysis of PDF: {pdf_file.name}")
        
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
        
        # Analyze each page with enhanced error handling
        page_analyses = []
        for page_num, page_content in enumerate(pages, 1):
            logger.info(f"Analyzing page {page_num}...")
            
            page_analysis = await self._analyze_single_page_enhanced(
                page_num, page_content, {"pages": pages}
            )
            page_analyses.append(page_analysis)
        
        # Create comprehensive summary
        comprehensive_analysis = await self._create_comprehensive_summary_enhanced(
            pdf_file.name, page_analyses, {"pages": pages}
        )
        
        return {
            "pdf_filename": pdf_file.name,
            "total_pages": len(pages),
            "page_analyses": page_analyses,
            "comprehensive_analysis": comprehensive_analysis,
            "extraction_metadata": extraction_result.metadata
        }
    
    async def _analyze_single_page_enhanced(self, page_num: int, page_content: str, 
                                          extracted_content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single page with enhanced error handling and Classical Chinese support."""
        
        # Create text analysis request for this page
        text_request = AnalysisRequest(
            request_id=f"page_analysis_{page_num}",
            content=page_content,
            data_type=DataType.TEXT,
            language="zh",  # Chinese
            analysis_type="sentiment_summary",
            metadata={
                "page_number": page_num,
                "source": "classical_chinese_pdf",
                "processing_mode": "enhanced"
            }
        )
        
        try:
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
            
            # Extract analysis results with enhanced parsing
            analysis_data = self._parse_analysis_result(text_result, page_content)
            
            return {
                "page_number": page_num,
                "status": "completed",
                "content_length": len(page_content),
                "raw_content": page_content[:500] + "..." if len(page_content) > 500 else page_content,
                "summary": analysis_data.get("summary", ""),
                "sentiment": analysis_data.get("sentiment", {}),
                "key_themes": analysis_data.get("key_themes", []),
                "language_detected": analysis_data.get("language_detected", "zh"),
                "analysis_metadata": analysis_data.get("metadata", {}),
                "processing_time": text_result.processing_time
            }
            
        except Exception as e:
            logger.error(f"Error analyzing page {page_num}: {e}")
            return {
                "page_number": page_num,
                "status": "failed",
                "error": str(e),
                "raw_content": page_content[:500] + "..." if len(page_content) > 500 else page_content
            }
    
    def _parse_analysis_result(self, result, page_content: str) -> Dict[str, Any]:
        """Parse the AnalysisResult into the expected format."""
        try:
            # Extract sentiment information
            sentiment_data = {
                "sentiment": result.sentiment.label.value if result.sentiment else "neutral",
                "confidence": result.sentiment.confidence if result.sentiment else 0.0,
                "reasoning": result.sentiment.reasoning if result.sentiment else "",
                "scores": result.sentiment.scores if result.sentiment else {}
            }
            
            # Generate summary from sentiment reasoning or content
            summary = ""
            if result.sentiment and result.sentiment.reasoning:
                summary = result.sentiment.reasoning
            else:
                # Create a basic summary from the content
                summary = self._generate_basic_summary(page_content)
            
            # Extract key themes from metadata or generate them
            key_themes = []
            if result.metadata and "key_themes" in result.metadata:
                key_themes = result.metadata["key_themes"]
            else:
                key_themes = self._extract_key_themes(page_content)
            
            # Detect language
            language_detected = "zh"  # Default for Classical Chinese
            if result.metadata and "language_detected" in result.metadata:
                language_detected = result.metadata["language_detected"]
            
            return {
                "summary": summary,
                "sentiment": sentiment_data,
                "key_themes": key_themes,
                "language_detected": language_detected,
                "metadata": result.metadata or {}
            }
            
        except Exception as e:
            logger.error(f"Error parsing analysis result: {e}")
            return {
                "summary": "Analysis parsing failed",
                "sentiment": {"sentiment": "neutral", "confidence": 0.0},
                "key_themes": [],
                "language_detected": "zh",
                "metadata": {"error": str(e)}
            }
    
    def _generate_basic_summary(self, content: str) -> str:
        """Generate a basic summary for Classical Chinese content."""
        # For Classical Chinese, create a basic summary based on content length and type
        if len(content) < 100:
            return "Short Classical Chinese text"
        elif "文言" in content or "古典" in content:
            return "Classical Chinese text with traditional grammar patterns"
        elif "之" in content or "其" in content or "者" in content:
            return "Classical Chinese text with common classical particles"
        else:
            return "Classical Chinese text requiring specialized analysis"
    
    def _extract_key_themes(self, content: str) -> List[str]:
        """Extract key themes from Classical Chinese content."""
        themes = []
        
        # Look for Classical Chinese patterns
        if "文言" in content:
            themes.append("Classical Chinese")
        if "之" in content:
            themes.append("Classical particles")
        if "其" in content:
            themes.append("Possessive pronouns")
        if "者" in content:
            themes.append("Nominalization")
        if "也" in content or "乃" in content:
            themes.append("Copula particles")
        if "以" in content:
            themes.append("Prepositional usage")
        if "所" in content:
            themes.append("Nominalization patterns")
        if "为" in content:
            themes.append("Passive voice")
        
        # Add general themes if no specific patterns found
        if not themes:
            themes = ["Classical Chinese", "Traditional grammar"]
        
        return themes
    
    async def _create_comprehensive_summary_enhanced(self, filename: str, 
                                                   page_analyses: List[Dict[str, Any]],
                                                   extracted_content: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive summary with enhanced Classical Chinese support."""
        
        # Count successful analyses
        successful_analyses = [a for a in page_analyses if a.get("status") == "completed"]
        
        if not successful_analyses:
            return {
                "overall_summary": "No content could be successfully analyzed from the Classical Chinese PDF",
                "overall_sentiment": {"sentiment": "neutral", "confidence": 0.0},
                "document_themes": ["Classical Chinese", "Analysis failed"],
                "analysis_status": "failed",
                "error": "All page analyses failed"
            }
        
        # Combine all successful page content
        all_content = "\n\n".join([
            analysis.get("raw_content", "") 
            for analysis in successful_analyses
        ])
        
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
                "total_pages": len(page_analyses),
                "successful_pages": len(successful_analyses)
            }
        )
        
        try:
            # Perform comprehensive analysis
            comprehensive_result = await self.text_agent.process(comprehensive_request)
            
            if comprehensive_result.status != "completed":
                error_msg = comprehensive_result.metadata.get("error", "Unknown error")
                logger.warning(f"Comprehensive analysis failed: {error_msg}")
                return {
                    "overall_summary": "Comprehensive analysis failed",
                    "overall_sentiment": {"sentiment": "neutral", "confidence": 0.0},
                    "document_themes": ["Classical Chinese", "Analysis failed"],
                    "analysis_status": "failed",
                    "error": error_msg
                }
            
            # Parse comprehensive result
            analysis_data = self._parse_analysis_result(comprehensive_result, all_content)
            
            return {
                "overall_summary": analysis_data.get("summary", ""),
                "overall_sentiment": analysis_data.get("sentiment", {}),
                "document_themes": analysis_data.get("key_themes", []),
                "analysis_status": "completed",
                "total_pages_analyzed": len(successful_analyses),
                "analysis_metadata": analysis_data.get("metadata", {})
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {
                "overall_summary": "Comprehensive analysis failed due to error",
                "overall_sentiment": {"sentiment": "neutral", "confidence": 0.0},
                "document_themes": ["Classical Chinese", "Analysis error"],
                "analysis_status": "failed",
                "error": str(e)
            }


async def main():
    """Main function to run the enhanced analysis."""
    pdf_path = "data/Classical Chinese Sample 22208_0_8.pdf"
    
    analyzer = EnhancedClassicalChineseAnalyzer()
    
    try:
        logger.info("Starting Enhanced Classical Chinese PDF analysis...")
        
        # Perform the analysis
        results = await analyzer.analyze_pdf_pages(pdf_path)
        
        # Save results to file
        output_file = Path("Results/enhanced_classical_chinese_analysis.json")
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Enhanced analysis completed. Results saved to: {output_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("ENHANCED CLASSICAL CHINESE PDF ANALYSIS SUMMARY")
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
                sentiment = page_analysis.get('sentiment', {})
                print(f"  Sentiment: {sentiment.get('sentiment', 'unknown')} (confidence: {sentiment.get('confidence', 0.0):.2f})")
                print(f"  Key Themes: {', '.join(page_analysis.get('key_themes', []))}")
                print(f"  Summary: {page_analysis.get('summary', '')[:200]}...")
                print(f"  Processing Time: {page_analysis.get('processing_time', 0.0):.2f}s")
            else:
                print(f"  Error: {page_analysis.get('error', 'Unknown error')}")
        
        print(f"\nDetailed results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
