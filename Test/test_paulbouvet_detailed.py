#!/usr/bin/env python3
"""
Detailed test for paulbouvet PDF content.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent


async def test_paulbouvet_detailed():
    """Test paulbouvet PDF with detailed content analysis."""
    
    print("🧪 Testing Paulbouvet PDF - Detailed Analysis")
    print("=" * 50)
    
    # Initialize file agent
    file_agent = EnhancedFileExtractionAgent()
    
    # Test paulbouvet PDF
    pdf_file = "data/paulbouvetpdf.pdf"
    
    print(f"📄 Testing with: {pdf_file}")
    
    # Test sample extraction
    try:
        sample_result = await file_agent.extract_text_from_pdf(
            pdf_file, 
            {"sample_only": True, "language": "auto"}
        )
        
        if sample_result["status"] == "success":
            sample_text = sample_result["extracted_text"]
            print(f"✅ Sample extraction successful")
            print(f"📝 Sample length: {len(sample_text)} characters")
            print(f"\n📄 FULL SAMPLE TEXT:")
            print("=" * 50)
            print(sample_text)
            print("=" * 50)
            
            # Character analysis
            russian_chars = sum(1 for char in sample_text if '\u0400' <= char <= '\u04FF')
            chinese_chars = sum(1 for char in sample_text if '\u4e00' <= char <= '\u9fff')
            latin_chars = sum(1 for char in sample_text if '\u0020' <= char <= '\u007F')
            
            print(f"\n🔤 Character Analysis:")
            print(f"   Russian characters: {russian_chars}")
            print(f"   Chinese characters: {chinese_chars}")
            print(f"   Latin characters: {latin_chars}")
            
            # Language detection
            if russian_chars > 10:
                print("🌐 Detected language: Russian")
            elif chinese_chars > 10:
                print("🌐 Detected language: Chinese")
            else:
                print("🌐 Detected language: English/Latin")
            
        else:
            print(f"❌ Sample extraction failed: {sample_result.get('error')}")
            
    except Exception as e:
        print(f"❌ Sample extraction error: {e}")
    
    print("\n✅ Detailed analysis completed!")


if __name__ == "__main__":
    asyncio.run(test_paulbouvet_detailed())
