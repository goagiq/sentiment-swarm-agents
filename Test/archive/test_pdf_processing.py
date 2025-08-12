#!/usr/bin/env python3
"""
Test PDF processing with timeout.
"""

import asyncio
import sys
import os
import signal

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_pdf_processing():
    """Test PDF processing with timeout."""
    print("🧪 PDF Processing Test")
    print("=" * 40)
    
    try:
        print("📡 Starting PDF processing...")
        
        # Import the main function
        from main import process_classical_chinese_pdf_via_mcp
        
        # Set a timeout for the processing
        print("⏱️ Setting 60-second timeout...")
        
        # Run the processing with timeout
        try:
            result = await asyncio.wait_for(
                process_classical_chinese_pdf_via_mcp(
                    'data/Classical Chinese Sample 22208_0_8.pdf', 
                    'zh', 
                    True
                ),
                timeout=60.0  # 60 second timeout
            )
            
            print("✅ PDF processing completed within timeout")
            print(f"Success: {result.get('success', False)}")
            
            if result.get('success', False):
                print("🎉 PDF processing successful!")
                print(f"Detected Language: {result.get('detected_language', 'Unknown')}")
                
                # Print extraction info
                text_extraction = result.get('text_extraction', {})
                if text_extraction.get('success'):
                    print(f"Text Length: {text_extraction.get('content_length', 'Unknown')}")
                    print(f"Pages Processed: {text_extraction.get('pages_processed', 'Unknown')}")
                
                # Print entity info
                entity_extraction = result.get('entity_extraction', {})
                print(f"Entities Found: {entity_extraction.get('entities_found', 0)}")
                
                # Print knowledge graph info
                kg_info = result.get('knowledge_graph', {})
                print(f"Knowledge Graph Nodes: {kg_info.get('nodes', 0)}")
                print(f"Knowledge Graph Edges: {kg_info.get('edges', 0)}")
            else:
                print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except asyncio.TimeoutError:
            print("⏰ Processing timed out after 60 seconds")
            return {"success": False, "error": "Processing timed out"}
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    result = asyncio.run(test_pdf_processing())
    print(f"\n🎯 Final Result: {'SUCCESS' if result.get('success', False) else 'FAILED'}")
