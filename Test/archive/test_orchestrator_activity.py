#!/usr/bin/env python3
"""
Test orchestrator activity with detailed console output.
"""

import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_orchestrator_activity():
    """Test orchestrator with detailed console output."""
    print("🧪 Testing Orchestrator Activity")
    print("=" * 50)
    
    try:
        print("📡 Step 1: Importing orchestrator...")
        
        from src.core.orchestrator import SentimentOrchestrator
        from src.core.models import AnalysisRequest, DataType
        
        print("✅ Orchestrator imported successfully")
        
        print("\n📡 Step 2: Creating orchestrator instance...")
        orchestrator = SentimentOrchestrator()
        print("✅ Orchestrator created successfully")
        
        print("\n📡 Step 3: Testing PDF processing...")
        
        # Test PDF processing with timeout
        try:
            print("🔧 Processing PDF with orchestrator...")
            result = await asyncio.wait_for(
                orchestrator.analyze_pdf("data/Classical Chinese Sample 22208_0_8.pdf", language="zh"),
                timeout=60.0  # 60 second timeout
            )
            
            print("✅ PDF processing completed!")
            print(f"Status: {result.status}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            
            if result.status == "completed":
                print("🎉 PDF processing successful!")
                print(f"Text Length: {len(result.extracted_text) if result.extracted_text else 0}")
                print(f"Pages: {len(result.pages) if result.pages else 0}")
                
                return {"success": True, "result": "PDF processing successful"}
            else:
                print(f"❌ Processing failed: {result.metadata.get('error', 'Unknown error')}")
                return {"success": False, "error": result.metadata.get('error', 'Unknown error')}
                
        except asyncio.TimeoutError:
            print("⏰ Processing timed out after 60 seconds")
            return {"success": False, "error": "Processing timed out"}
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = asyncio.run(test_orchestrator_activity())
    print(f"\n🎯 Final Result: {'SUCCESS' if result.get('success', False) else 'FAILED'}")
