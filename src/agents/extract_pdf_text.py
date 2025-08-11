#!/usr/bin/env python3
"""
Simple PDF text extraction utility using PyPDF2.
This is a standalone utility for basic PDF text extraction.
For advanced PDF processing with OCR and parallel processing,
use the FileExtractionAgent instead.
"""

import logging
from pathlib import Path
from typing import Optional

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logging.warning("PyPDF2 not available. Install with: pip install PyPDF2")

logger = logging.getLogger(__name__)


def extract_pdf_text(pdf_path: str) -> Optional[str]:
    """
    Extract text from PDF file using PyPDF2.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as string, or None if extraction failed
    """
    if not PYPDF2_AVAILABLE:
        logger.error("PyPDF2 is not available")
        return None
    
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return None
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            logger.info(f"PDF has {len(pdf_reader.pages)} pages")
            
            if len(pdf_reader.pages) == 0:
                logger.warning("PDF appears to be empty or corrupted")
                return None
            
            text = ""
            for i, page in enumerate(pdf_reader.pages):
                logger.info(f"Processing page {i+1}...")
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {i+1} ---\n"
                    text += page_text
                    text += "\n"
                else:
                    logger.warning(f"No text found on page {i+1}")
            
            return text if text.strip() else None
            
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return None


def extract_pdf_text_to_file(pdf_path: str, output_path: str) -> bool:
    """
    Extract text from PDF and save to file.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path to save the extracted text
        
    Returns:
        True if successful, False otherwise
    """
    text = extract_pdf_text(pdf_path)
    
    if text is None:
        return False
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Text saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving text to file: {e}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    pdf_path = "data/Classical Chinese Sample 22208_0_8.pdf"
    text = extract_pdf_text(pdf_path)
    
    if text:
        print("\nExtracted text:")
        print("=" * 50)
        print(text)
        print("=" * 50)
        
        # Save to file
        output_path = "extracted_text.txt"
        if extract_pdf_text_to_file(pdf_path, output_path):
            print(f"\nText saved to {output_path}")
        else:
            print("\nFailed to save text to file")
    else:
        print("No text could be extracted from the PDF")
