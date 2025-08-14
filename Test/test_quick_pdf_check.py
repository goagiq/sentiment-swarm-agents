#!/usr/bin/env python3
"""
Quick test to check PDF processing without MCP server.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent


async def test_quick_pdf_check():
    """Quick test for PDF processing."""
    
    print("🔍 Quick PDF Processing Check")
    print("=" * 40)
    
    # Initialize file agent directly
    file_agent = EnhancedFileExtractionAgent()
    
    # Test files
    test_files = [
        "data/Classical Chinese Sample 22208_0_8.pdf",
        "data/Russian_Oliver_Excerpt.pdf", 
        "data/paulbouvetpdf.pdf"
    ]
    
    for pdf_file in test_files:
        print(f"\n📄 Testing: {pdf_file}")
        
        # Check if file exists
        if not Path(pdf_file).exists():
            print(f"❌ File not found: {pdf_file}")
            continue
            
        try:
            # Quick sample extraction
            print("🔍 Extracting sample...")
            result = await file_agent.extract_text_from_pdf(
                pdf_file, 
                {"sample_only": True}
            )
            
            if result["status"] == "success":
                sample_text = result["extracted_text"]
                print(f"✅ Success! Sample length: {len(sample_text)} chars")
                print(f"📝 Preview: {sample_text[:100]}...")
            else:
                print(f"❌ Failed: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ Quick test completed!")


if __name__ == "__main__":
    asyncio.run(test_quick_pdf_check())


