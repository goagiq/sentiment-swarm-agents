#!/usr/bin/env python3
"""
Simple test to debug PDF extraction.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.file_extraction_agent import FileExtractionAgent
from src.core.models import AnalysisRequest, DataType

async def test_pdf_extraction():
    """Test PDF extraction directly."""
    pdf_path = "data/Classical Chinese Sample 22208_0_8.pdf"
    
    # Create file extraction agent
    agent = FileExtractionAgent()
    
    # Create analysis request
    request = AnalysisRequest(
        request_id="test_pdf_extraction",
        content=pdf_path,
        data_type=DataType.PDF,
        language="zh",
        analysis_type="extraction"
    )
    
    # Process the request
    result = await agent.process(request)
    
    print(f"Status: {result.status}")
    print(f"Raw content length: {len(result.raw_content) if result.raw_content else 0}")
    print(f"Metadata keys: {list(result.metadata.keys())}")
    
    if result.metadata.get("extracted_content"):
        extracted = result.metadata["extracted_content"]
        print(f"Extracted content keys: {list(extracted.keys())}")
        print(f"Pages: {len(extracted.get('pages', []))}")
        
        if extracted.get('pages'):
            print(f"First page preview: {extracted['pages'][0][:200]}...")
    
    return result

if __name__ == "__main__":
    import asyncio
    result = asyncio.run(test_pdf_extraction())
