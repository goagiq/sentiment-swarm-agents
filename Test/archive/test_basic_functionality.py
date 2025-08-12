#!/usr/bin/env python3
"""
Basic functionality test without file processing.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_basic_functionality():
    """Test basic functionality without file processing."""
    print("🧪 Basic Functionality Test")
    print("=" * 40)
    
    try:
        print("📡 Testing imports...")
        
        # Test basic imports
        from src.agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
        from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
        from src.core.models import AnalysisRequest, DataType
        
        print("✅ All imports successful")
        
        # Test agent creation
        print("📡 Testing agent creation...")
        file_agent = EnhancedFileExtractionAgent()
        kg_agent = KnowledgeGraphAgent()
        
        print("✅ Agents created successfully")
        
        # Test basic request creation
        print("📡 Testing request creation...")
        test_request = AnalysisRequest(
            data_type=DataType.TEXT,
            content="Test content",
            language="en"
        )
        
        print("✅ Request creation successful")
        print(f"Request data type: {test_request.data_type}")
        print(f"Request language: {test_request.language}")
        
        print("✅ Basic functionality test completed successfully")
        return {"success": True, "message": "All basic functionality working"}
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = asyncio.run(test_basic_functionality())
    print(f"\n🎯 Final Result: {'SUCCESS' if result.get('success', False) else 'FAILED'}")
