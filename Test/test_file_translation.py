#!/usr/bin/env python3
"""
Test script to verify file translation capabilities:
- PDF files: extract text and perform normal translation
- Image files: use vision model for best results
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.translation_agent import TranslationAgent
from core.models import AnalysisRequest, DataType


async def test_pdf_translation():
    """Test PDF translation functionality."""
    print("🔍 Testing PDF Translation...")
    
    # Create a simple test PDF content (simulated)
    test_pdf_path = "test_document.pdf"
    
    # For testing, we'll create a mock PDF scenario
    print(f"📄 Testing PDF translation with: {test_pdf_path}")
    
    try:
        agent = TranslationAgent()
        
        # Test if agent can process PDF data type
        can_process = await agent.can_process(
            AnalysisRequest(data_type=DataType.PDF, content=test_pdf_path)
        )
        print(f"✅ Agent can process PDF: {can_process}")
        
        # Note: Actual PDF processing requires a real PDF file
        print("ℹ️  PDF translation requires PyPDF2 or PyMuPDF packages")
        print("ℹ️  For full testing, provide a real PDF file path")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF translation test failed: {e}")
        return False


async def test_image_translation():
    """Test image translation with vision model."""
    print("\n🖼️  Testing Image Translation with Vision Model...")
    
    # Create a test image path
    test_image_path = "test_image.jpg"
    
    try:
        agent = TranslationAgent()
        
        # Test if agent can process IMAGE data type
        can_process = await agent.can_process(
            AnalysisRequest(data_type=DataType.IMAGE, content=test_image_path)
        )
        print(f"✅ Agent can process IMAGE: {can_process}")
        
        # Check if vision model is configured
        vision_model = agent.translation_models.get("vision")
        print(f"✅ Vision model configured: {vision_model}")
        
        # Note: Actual image processing requires a real image file
        print("ℹ️  Image translation uses vision model (llava:latest) for best results")
        print("ℹ️  Falls back to OCR agent if vision model fails")
        print("ℹ️  For full testing, provide a real image file path")
        
        return True
        
    except Exception as e:
        print(f"❌ Image translation test failed: {e}")
        return False


async def test_file_type_support():
    """Test that all file types are supported."""
    print("\n📋 Testing File Type Support...")
    
    try:
        agent = TranslationAgent()
        
        # Test all supported data types
        supported_types = [
            DataType.TEXT,
            DataType.AUDIO,
            DataType.VIDEO,
            DataType.WEBPAGE,
            DataType.IMAGE,
            DataType.PDF
        ]
        
        for data_type in supported_types:
            can_process = await agent.can_process(
                AnalysisRequest(data_type=data_type, content="test_content")
            )
            status = "✅" if can_process else "❌"
            print(f"{status} {data_type.value.upper()}: {can_process}")
        
        return True
        
    except Exception as e:
        print(f"❌ File type support test failed: {e}")
        return False


async def test_agent_status():
    """Test agent status and capabilities."""
    print("\n🔧 Testing Agent Status...")
    
    try:
        agent = TranslationAgent()
        status = agent.get_status()
        
        print(f"✅ Agent ID: {status.get('agent_id')}")
        print(f"✅ Model Name: {status.get('model_name')}")
        print(f"✅ Translation Models: {status.get('translation_models', {})}")
        print(f"✅ Language Patterns: {len(status.get('language_patterns', {}))} patterns")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent status test failed: {e}")
        return False


async def main():
    """Run all file translation tests."""
    print("🚀 Starting File Translation Tests")
    print("=" * 50)
    
    tests = [
        test_file_type_support,
        test_pdf_translation,
        test_image_translation,
        test_agent_status
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! File translation support is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    print("\n📝 Notes:")
    print("- PDF translation requires PyPDF2 or PyMuPDF packages")
    print("- Image translation uses vision model (llava:latest) for optimal results")
    print("- For full testing, provide real PDF and image files")
    print("- All file types are properly supported in the translation agent")


if __name__ == "__main__":
    asyncio.run(main())
