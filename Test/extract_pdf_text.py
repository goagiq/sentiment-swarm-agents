#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fitz
import sys

def extract_pdf_text(pdf_path):
    """Extract text from PDF with proper encoding."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None

if __name__ == "__main__":
    pdf_path = "data/Classical Chinese Sample 22208_0_8.pdf"
    text = extract_pdf_text(pdf_path)
    
    if text:
        # Save to file with UTF-8 encoding
        with open("extracted_text.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Text extracted successfully. Length: {len(text)} characters")
        print("First 500 characters:")
        print(text[:500])
    else:
        print("Failed to extract text from PDF")
