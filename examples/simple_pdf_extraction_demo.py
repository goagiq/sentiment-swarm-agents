#!/usr/bin/env python3
"""
Simple PDF Extraction Demo

This script demonstrates how to use the standalone PDF text extraction utility
that has been moved to src/agents/extract_pdf_text.py.

For advanced PDF processing with OCR and parallel processing,
use the FileExtractionAgent instead.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.extract_pdf_text import extract_pdf_text, extract_pdf_text_to_file


def demo_basic_extraction():
    """Demonstrate basic PDF text extraction."""
    print("🔍 Basic PDF Text Extraction Demo")
    print("=" * 40)
    
    # Example PDF path (update this to point to your PDF file)
    pdf_path = "data/sample.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        print("Please update the pdf_path variable to point to a valid PDF file.")
        return
    
    print(f"📄 Processing PDF: {pdf_path}")
    
    # Extract text
    text = extract_pdf_text(pdf_path)
    
    if text:
        print("✅ Text extraction successful!")
        print(f"📊 Extracted {len(text)} characters")
        print("\n📝 First 500 characters:")
        print("-" * 40)
        print(text[:500] + "..." if len(text) > 500 else text)
        print("-" * 40)
    else:
        print("❌ Text extraction failed")


def demo_save_to_file():
    """Demonstrate saving extracted text to file."""
    print("\n💾 Save to File Demo")
    print("=" * 40)
    
    # Example PDF path
    pdf_path = "data/sample.pdf"
    output_path = "Results/extracted_text_demo.txt"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    # Ensure output directory exists
    os.makedirs("Results", exist_ok=True)
    
    print(f"📄 Processing PDF: {pdf_path}")
    print(f"💾 Saving to: {output_path}")
    
    # Extract and save
    success = extract_pdf_text_to_file(pdf_path, output_path)
    
    if success:
        print("✅ Text saved successfully!")
        
        # Show file info
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"📊 File size: {file_size} bytes")
    else:
        print("❌ Failed to save text to file")


def demo_error_handling():
    """Demonstrate error handling."""
    print("\n⚠️ Error Handling Demo")
    print("=" * 40)
    
    # Test with non-existent file
    non_existent_pdf = "data/non_existent.pdf"
    print(f"📄 Testing with non-existent file: {non_existent_pdf}")
    
    text = extract_pdf_text(non_existent_pdf)
    if text is None:
        print("✅ Correctly handled non-existent file")
    else:
        print("❌ Unexpected result for non-existent file")
    
    # Test with invalid file
    invalid_file = "data/invalid.txt"
    print(f"📄 Testing with invalid file: {invalid_file}")
    
    # Create an invalid file
    os.makedirs("data", exist_ok=True)
    with open(invalid_file, "w") as f:
        f.write("This is not a PDF file")
    
    text = extract_pdf_text(invalid_file)
    if text is None:
        print("✅ Correctly handled invalid file")
    else:
        print("❌ Unexpected result for invalid file")
    
    # Clean up
    if os.path.exists(invalid_file):
        os.remove(invalid_file)


def main():
    """Main demo function."""
    print("🚀 Simple PDF Extraction Utility Demo")
    print("=" * 50)
    print("This demo shows how to use the standalone PDF text extraction utility.")
    print("For advanced features, use the FileExtractionAgent instead.\n")
    
    # Run demos
    demo_basic_extraction()
    demo_save_to_file()
    demo_error_handling()
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed!")
    print("\n📋 Next Steps:")
    print("1. Update the pdf_path variable to point to your PDF files")
    print("2. Run the script to test with your own PDFs")
    print("3. For advanced features, use the FileExtractionAgent")
    print("4. Check the Results/ directory for extracted text files")


if __name__ == "__main__":
    main()
