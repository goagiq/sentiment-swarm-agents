#!/usr/bin/env python3
"""
Test MCP direct client integration for Chinese PDF processing.
Uses the MCP client directly instead of HTTP requests.
"""

import os
import sys
import asyncio
import json
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.language_config import LanguageConfigFactory


class MCPDirectClientTester:
    """MCP direct client tester for Chinese PDF processing."""
    
    def __init__(self):
        self.chinese_config = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize language configurations."""
        try:
            # Initialize Chinese language configuration
            self.chinese_config = LanguageConfigFactory.get_config("zh")
            print("✅ Chinese language configuration loaded")
            
            # Check for Classical Chinese patterns
            if hasattr(self.chinese_config, 'classical_patterns'):
                print(f"✅ Classical Chinese patterns available: {len(self.chinese_config.classical_patterns)} categories")
            
            if hasattr(self.chinese_config, 'is_classical_chinese'):
                print("✅ Classical Chinese detection method available")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize language components: {e}")
    
    def find_chinese_pdfs(self, data_dir="data"):
        """Find all Chinese PDFs in the data directory."""
        print(f"\n📁 Searching for Chinese PDFs in {data_dir}...")
        
        chinese_pdfs = []
        
        if not os.path.exists(data_dir):
            print(f"❌ Data directory not found: {data_dir}")
            return chinese_pdfs
        
        for file in os.listdir(data_dir):
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(data_dir, file)
                
                # Check if filename contains Chinese characters or Chinese-related keywords
                chinese_keywords = ['chinese', 'china', 'mandarin', 'cantonese', 'classical', 'zh']
                if any(keyword in file.lower() for keyword in chinese_keywords):
                    chinese_pdfs.append(file_path)
                    print(f"✅ Found Chinese PDF: {file}")
        
        if not chinese_pdfs:
            print("⚠️ No Chinese PDFs found in data directory")
        
        return chinese_pdfs
    
    async def test_pdf_processing_direct(self, pdf_path):
        """Test PDF processing using direct MCP client."""
        print(f"\n🔬 Testing PDF processing via direct MCP client: {pdf_path}")
        
        try:
            # Import the MCP server directly
            from src.core.mcp_server import OptimizedMCPServer
            
            # Create MCP server instance
            mcp_server = OptimizedMCPServer()
            
            if not mcp_server.mcp:
                print("❌ MCP server not available")
                return None
            
            print("✅ MCP server initialized successfully")
            
            # Call the process_pdf_enhanced_multilingual function directly
            print(f"📤 Calling process_pdf_enhanced_multilingual directly")
            print(f"📄 Processing PDF: {pdf_path}")
            print(f"🌍 Language: zh (Chinese)")
            print(f"📊 Generate report: True")
            
            # Get the function from the MCP server
            process_function = None
            for tool_name, tool_func in mcp_server.mcp.tools.items():
                if tool_name == "process_pdf_enhanced_multilingual":
                    process_function = tool_func
                    break
            
            if not process_function:
                print("❌ process_pdf_enhanced_multilingual function not found")
                return None
            
            # Call the function directly
            result = await process_function(
                pdf_path=pdf_path,
                language="zh",
                generate_report=True,
                output_path=None
            )
            
            print("✅ PDF processing successful via direct MCP client")
            self._display_direct_results(result)
            return result
            
        except Exception as e:
            print(f"❌ MCP direct processing error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _display_direct_results(self, result):
        """Display direct MCP processing results."""
        print("\n" + "=" * 60)
        print("📋 DIRECT MCP PROCESSING RESULTS")
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
            
            # Display any error information
            if 'error' in result:
                print(f"\n❌ Error: {result['error']}")
        else:
            print(f"   - Result: {result}")
        
        print("\n" + "=" * 60)
    
    async def run_comprehensive_direct_test(self):
        """Run comprehensive test of direct MCP client for Chinese PDF processing."""
        print("🧪 Direct MCP Client Test Suite for Chinese PDF Processing")
        print("=" * 70)
        
        # Test 1: Find Chinese PDFs
        chinese_pdfs = self.find_chinese_pdfs()
        
        if not chinese_pdfs:
            print("\n❌ No Chinese PDFs found for testing")
            return False
        
        # Test 2: Process each Chinese PDF via direct MCP client
        results = []
        
        for pdf_path in chinese_pdfs:
            print(f"\n{'='*70}")
            print(f"🔬 Testing PDF via Direct MCP Client: {os.path.basename(pdf_path)}")
            print(f"{'='*70}")
            
            pdf_result = {
                "pdf_path": pdf_path,
                "direct_mcp_result": None
            }
            
            # Test direct MCP processing
            pdf_result["direct_mcp_result"] = await self.test_pdf_processing_direct(pdf_path)
            
            results.append(pdf_result)
        
        # Save results
        self._save_test_results(results)
        
        # Display summary
        self._display_test_summary(results)
        
        return True
    
    def _save_test_results(self, results):
        """Save test results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"Results/direct_mcp_client_test_results_{timestamp}.json"
        
        os.makedirs("Results", exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Test results saved to: {results_file}")
    
    def _display_test_summary(self, results):
        """Display test summary."""
        print("\n" + "=" * 70)
        print("📊 DIRECT MCP CLIENT TEST SUMMARY")
        print("=" * 70)
        
        total_pdfs = len(results)
        successful_mcp = sum(1 for r in results if r.get("direct_mcp_result") and r["direct_mcp_result"].get("success"))
        
        print(f"📄 Total Chinese PDFs tested: {total_pdfs}")
        print(f"✅ Successful Direct MCP processing: {successful_mcp}/{total_pdfs}")
        print(f"🔧 Method: Direct MCP Client (no HTTP)")
        print(f"🛠️ Tool: process_pdf_enhanced_multilingual")
        
        for result in results:
            pdf_name = os.path.basename(result["pdf_path"])
            mcp_status = "✅" if (result.get("direct_mcp_result") and 
                                result["direct_mcp_result"].get("success")) else "❌"
            
            print(f"   - {pdf_name}: Direct MCP {mcp_status}")
            
            # Show additional info for successful results
            if result.get("direct_mcp_result") and result["direct_mcp_result"].get("success"):
                mcp_result = result["direct_mcp_result"]
                if "text_extraction" in mcp_result:
                    content_length = mcp_result["text_extraction"].get("content_length", 0)
                    print(f"     └─ Content length: {content_length} characters")
                if "entity_extraction" in mcp_result:
                    entities_found = mcp_result["entity_extraction"].get("entities_found", 0)
                    print(f"     └─ Entities found: {entities_found}")
                if "knowledge_graph" in mcp_result:
                    nodes = mcp_result["knowledge_graph"].get("nodes", 0)
                    edges = mcp_result["knowledge_graph"].get("edges", 0)
                    print(f"     └─ Knowledge graph: {nodes} nodes, {edges} edges")
        
        print("\n" + "=" * 70)
        print("🎉 Direct MCP Client test completed!")
        print("=" * 70)


async def main():
    """Main function to run the direct MCP client test."""
    tester = MCPDirectClientTester()
    success = await tester.run_comprehensive_direct_test()
    
    if success:
        print("\n🎉 All direct MCP client tests completed successfully!")
        print("✅ All tests used the MCP tool directly (no HTTP session issues)")
    else:
        print("\n❌ Some direct MCP client tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())
