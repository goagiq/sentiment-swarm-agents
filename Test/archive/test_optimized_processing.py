#!/usr/bin/env python3
"""
Test optimized PDF processing with shorter timeout.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_optimized_processing():
    """Test optimized PDF processing."""
    print("🧪 Optimized PDF Processing Test")
    print("=" * 45)
    
    try:
        print("📡 Testing optimized processing...")
        
        # Import the optimized agents directly
        from src.agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
        from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
        from src.core.models import AnalysisRequest, DataType
        
        print("✅ Optimized agents imported successfully")
        
        # Create optimized agents
        file_agent = EnhancedFileExtractionAgent()
        kg_agent = KnowledgeGraphAgent()
        
        print("✅ Optimized agents created successfully")
        
        # Test with a shorter timeout
        print("⏱️ Setting 30-second timeout for testing...")
        
        try:
            # Test PDF extraction only (without full processing)
            pdf_request = AnalysisRequest(
                data_type=DataType.PDF,
                content="data/Classical Chinese Sample 22208_0_8.pdf",
                language="zh"
            )
            
            print("🔧 Testing PDF extraction with optimized agent...")
            extraction_result = await asyncio.wait_for(
                file_agent.process(pdf_request),
                timeout=30.0  # 30 second timeout
            )
            
            print("✅ PDF extraction completed within timeout")
            print(f"Status: {extraction_result.status}")
            print(f"Pages Processed: {len(extraction_result.pages) if extraction_result.pages else 0}")
            
            if extraction_result.status == "completed":
                print("🎉 Optimized PDF extraction successful!")
                print(f"Text Length: {len(extraction_result.extracted_text) if extraction_result.extracted_text else 0}")
                print(f"Processing Time: {extraction_result.processing_time:.2f}s")
                
                return {"success": True, "extraction": "successful", "processing_time": extraction_result.processing_time}
            else:
                print(f"❌ Extraction failed: {extraction_result.metadata.get('error', 'Unknown error')}")
                return {"success": False, "error": extraction_result.metadata.get('error', 'Unknown error')}
                
        except asyncio.TimeoutError:
            print("⏰ Processing timed out after 30 seconds")
            return {"success": False, "error": "Processing timed out"}
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = asyncio.run(test_optimized_processing())
    print(f"\n🎯 Final Result: {'SUCCESS' if result.get('success', False) else 'FAILED'}")
