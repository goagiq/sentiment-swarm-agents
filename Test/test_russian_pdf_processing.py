#!/usr/bin/env python3
"""
Quick test for Russian PDF processing.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent


async def test_russian_pdf():
    """Test Russian PDF processing."""
    
    print("🧪 Testing Russian PDF Processing")
    print("=" * 40)
    
    # Initialize file agent
    file_agent = EnhancedFileExtractionAgent()
    
    # Test Russian PDF
    russian_pdf = "data/Russian_Oliver_Excerpt.pdf"
    
    print(f"📄 Testing with: {russian_pdf}")
    
    # Test 1: Sample extraction
    print("\n🔍 Test 1: Sample extraction")
    try:
        sample_result = await file_agent.extract_text_from_pdf(
            russian_pdf, 
            {"sample_only": True, "language": "auto"}
        )
        
        if sample_result["status"] == "success":
            sample_text = sample_result["extracted_text"]
            print(f"✅ Sample extraction successful")
            print(f"📝 Sample length: {len(sample_text)} characters")
            print(f"📝 Sample preview: {sample_text[:200]}...")
            
            # Check for Russian characters
            russian_chars = sum(1 for char in sample_text if '\u0400' <= char <= '\u04FF')
            print(f"🔤 Russian characters found: {russian_chars}")
            
        else:
            print(f"❌ Sample extraction failed: {sample_result.get('error')}")
            
    except Exception as e:
        print(f"❌ Sample extraction error: {e}")
    
    print("\n✅ Russian PDF processing test completed!")


if __name__ == "__main__":
    asyncio.run(test_russian_pdf())
