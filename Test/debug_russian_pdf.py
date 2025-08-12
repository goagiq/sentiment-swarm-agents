#!/usr/bin/env python3
"""
Debug script to understand PDF extraction issues.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.file_extraction_agent import FileExtractionAgent
from core.models import AnalysisRequest, DataType


async def main():
    """Debug PDF extraction."""
    print("Debugging Russian PDF extraction...")
    
    # Initialize agent
    file_agent = FileExtractionAgent()
    
    pdf_path = "data/Russian_Oliver_Excerpt.pdf"
    
    # Check if file exists
    if not Path(pdf_path).exists():
        print(f"✗ File not found: {pdf_path}")
        return
    
    print(f"✓ File exists: {pdf_path}")
    print(f"  - Size: {Path(pdf_path).stat().st_size} bytes")
    
    # Create request
    request = AnalysisRequest(
        id="debug_russian_pdf",
        content=pdf_path,
        data_type=DataType.PDF
    )
    
    try:
        print("Processing PDF...")
        extraction_result = await file_agent.process(request)
        
        print(f"Status: {extraction_result.status}")
        print(f"Processing time: {extraction_result.processing_time}")
        print(f"Model used: {extraction_result.model_used}")
        print(f"Quality score: {extraction_result.quality_score}")
        
        if extraction_result.metadata:
            print("Metadata:")
            for key, value in extraction_result.metadata.items():
                print(f"  {key}: {value}")
        
        if extraction_result.status == "COMPLETED":
            print(f"✓ Success! Extracted {len(extraction_result.extracted_text)} characters")
            print(f"First 500 chars: {extraction_result.extracted_text[:500]}")
        else:
            print("✗ Failed")
            
    except Exception as e:
        print(f"✗ Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
