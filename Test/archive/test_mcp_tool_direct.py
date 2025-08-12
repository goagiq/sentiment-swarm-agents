#!/usr/bin/env python3
"""
Direct MCP tool test for Classical Chinese PDF processing.
This script directly uses the MCP tool to process the Classical Chinese PDF.
"""

import asyncio
import os
import sys
import json
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_mcp_tool_direct():
    """Test the MCP tool directly for Classical Chinese PDF processing."""
    print("🧪 Direct MCP Tool Test for Classical Chinese PDF Processing")
    print("=" * 70)
    
    # Find the Classical Chinese PDF
    pdf_path = "data/Classical Chinese Sample 22208_0_8.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        return False
    
    print(f"✅ Found PDF: {pdf_path}")
    
    try:
        # Import the MCP tool directly from the server
        from src.core.mcp_server import OptimizedMCPServer
        
        print("🔧 Initializing MCP server...")
        mcp_server = OptimizedMCPServer()
        
        # Check if the tool is available
        if hasattr(mcp_server, 'mcp') and hasattr(mcp_server.mcp, 'process_pdf_enhanced_multilingual'):
            print("✅ MCP tool 'process_pdf_enhanced_multilingual' is available")
            
            print(f"📤 Calling MCP tool with PDF: {pdf_path}")
            print(f"🌍 Language: zh (Chinese)")
            print(f"📊 Generate report: True")
            
            # Call the MCP tool directly
            result = await mcp_server.mcp.process_pdf_enhanced_multilingual(
                pdf_path=pdf_path,
                language="zh",
                generate_report=True,
                output_path=None
            )
            
            print("✅ MCP tool processing completed successfully!")
            display_mcp_results(result)
            return True
            
        else:
            print("❌ MCP tool 'process_pdf_enhanced_multilingual' not found")
            if hasattr(mcp_server, 'mcp'):
                print("Available MCP methods:")
                for method in dir(mcp_server.mcp):
                    if not method.startswith('_'):
                        print(f"   - {method}")
            return False
            
    except Exception as e:
        print(f"❌ MCP tool test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def display_mcp_results(result):
    """Display MCP tool results."""
    print("\n" + "=" * 60)
    print("📋 MCP TOOL RESULTS")
    print("=" * 60)
    
    if isinstance(result, dict):
        # Check for success status
        success = result.get('success', False)
        print(f"✅ Success: {success}")
        
        # Display key information
        if 'pdf_path' in result:
            print(f"📄 PDF Path: {result['pdf_path']}")
        
        if 'detected_language' in result:
            print(f"🌍 Detected Language: {result['detected_language']}")
        
        # Text extraction results
        if 'text_extraction' in result:
            text_extraction = result['text_extraction']
            print(f"\n📄 Text Extraction:")
            print(f"   - Success: {text_extraction.get('success', False)}")
            print(f"   - Content length: {text_extraction.get('content_length', 0)} characters")
            print(f"   - Pages processed: {text_extraction.get('pages_processed', 'Unknown')}")
            print(f"   - Extraction method: {text_extraction.get('extraction_method', 'Unknown')}")
        
        # Entity extraction results
        if 'entity_extraction' in result:
            entity_extraction = result['entity_extraction']
            print(f"\n🔍 Entity Extraction:")
            print(f"   - Entities found: {entity_extraction.get('entities_found', 0)}")
            print(f"   - Entity types: {entity_extraction.get('entity_types', {})}")
            print(f"   - Language stats: {entity_extraction.get('language_stats', {})}")
            print(f"   - Extraction method: {entity_extraction.get('extraction_method', 'Unknown')}")
        
        # Knowledge graph results
        if 'knowledge_graph' in result:
            knowledge_graph = result['knowledge_graph']
            print(f"\n🧠 Knowledge Graph:")
            print(f"   - Nodes: {knowledge_graph.get('nodes', 0)}")
            print(f"   - Edges: {knowledge_graph.get('edges', 0)}")
            print(f"   - Communities: {knowledge_graph.get('communities', 0)}")
            print(f"   - Processing time: {knowledge_graph.get('processing_time', 0):.2f} seconds")
        
        # Vector database results
        if 'vector_database' in result:
            vector_db = result['vector_database']
            print(f"\n💾 Vector Database:")
            print(f"   - Vector ID: {vector_db.get('vector_id', 'Unknown')}")
            print(f"   - Content stored: {vector_db.get('content_stored', False)}")
        
        # Report results
        if 'report_files' in result:
            report_files = result['report_files']
            print(f"\n📊 Report Generation:")
            if isinstance(report_files, dict):
                for key, value in report_files.items():
                    print(f"   - {key}: {value}")
            else:
                print(f"   - Report files: {report_files}")
        
        # Enhanced features
        if 'enhanced_features' in result:
            enhanced_features = result['enhanced_features']
            print(f"\n🚀 Enhanced Features:")
            for feature, enabled in enhanced_features.items():
                status = "✅" if enabled else "❌"
                print(f"   - {feature}: {status}")
    else:
        print(f"📄 Result: {result}")
    
    print("\n" + "=" * 60)

async def main():
    """Main function to run the direct MCP tool test."""
    success = await test_mcp_tool_direct()
    
    if success:
        print("\n🎉 MCP tool test completed successfully!")
        print("✅ Classical Chinese PDF processed via MCP tool")
        print("🔧 MCP tool integration working correctly")
    else:
        print("\n❌ MCP tool test failed. Check the output above for details.")

if __name__ == "__main__":
    asyncio.run(main())
