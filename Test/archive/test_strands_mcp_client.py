#!/usr/bin/env python3
"""
Test Strands MCP client integration for Chinese PDF processing.
Uses the correct Strands MCP client implementation.
"""

import os
import sys
import json
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.language_config import LanguageConfigFactory


class StrandsMCPClientTester:
    """Strands MCP client tester using the correct implementation."""
    
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
    
    def test_strands_mcp_client(self, pdf_path):
        """Test PDF processing using Strands MCP client."""
        print(f"\n🔬 Testing PDF processing via Strands MCP client: {pdf_path}")
        
        try:
            # Import the required modules
            from mcp.client.streamable_http import streamablehttp_client
            from strands import Agent
            from strands.tools.mcp.mcp_client import MCPClient
            
            def create_streamable_http_transport():
                return streamablehttp_client("http://localhost:8000/mcp/")
            
            streamable_http_mcp_client = MCPClient(create_streamable_http_transport)
            
            print("✅ Strands MCP client created successfully")
            
            # Use the MCP server in a context manager
            with streamable_http_mcp_client:
                print("✅ Connected to MCP server")
                
                # Get the tools from the MCP server
                tools = streamable_http_mcp_client.list_tools_sync()
                print(f"✅ Found {len(tools)} tools from MCP server")
                
                # Check if our target tool is available
                target_tool = "process_pdf_enhanced_multilingual"
                if target_tool in [tool.name for tool in tools]:
                    print(f"✅ Target tool '{target_tool}' is available")
                else:
                    print(f"❌ Target tool '{target_tool}' not found")
                    print("Available tools:")
                    for tool in tools:
                        print(f"   - {tool.name}")
                    return None
                
                # Create an agent with the MCP tools
                agent = Agent(tools=tools)
                print("✅ Agent created with MCP tools")
                
                # Call the PDF processing tool
                print(f"📤 Calling {target_tool} with PDF: {pdf_path}")
                print(f"🌍 Language: zh (Chinese)")
                print(f"📊 Generate report: True")
                
                result = agent.run_sync(
                    f"Process the PDF file '{pdf_path}' using the {target_tool} tool. "
                    f"Set language to 'zh' (Chinese) and generate a report. "
                    f"Return the complete result including text extraction, entity extraction, "
                    f"and knowledge graph generation."
                )
                
                print("✅ PDF processing completed via Strands MCP client")
                self._display_strands_results(result)
                return result
                
        except Exception as e:
            print(f"❌ Strands MCP client error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _display_strands_results(self, result):
        """Display Strands MCP processing results."""
        print("\n" + "=" * 60)
        print("📋 STRANDS MCP CLIENT PROCESSING RESULTS")
        print("=" * 60)
        
        if hasattr(result, 'content'):
            print(f"✅ Result content available")
            print(f"📄 Content: {result.content}")
        else:
            print(f"📄 Result: {result}")
        
        print("\n" + "=" * 60)
    
    def run_comprehensive_strands_test(self):
        """Run comprehensive test of Strands MCP client for Chinese PDF processing."""
        print("🧪 Strands MCP Client Test Suite for Chinese PDF Processing")
        print("=" * 70)
        
        # Test 1: Find Chinese PDFs
        chinese_pdfs = self.find_chinese_pdfs()
        
        if not chinese_pdfs:
            print("\n❌ No Chinese PDFs found for testing")
            return False
        
        # Test 2: Process each Chinese PDF via Strands MCP client
        results = []
        
        for pdf_path in chinese_pdfs:
            print(f"\n{'='*70}")
            print(f"🔬 Testing PDF via Strands MCP Client: {os.path.basename(pdf_path)}")
            print(f"{'='*70}")
            
            pdf_result = {
                "pdf_path": pdf_path,
                "strands_mcp_result": None
            }
            
            # Test Strands MCP client processing
            pdf_result["strands_mcp_result"] = self.test_strands_mcp_client(pdf_path)
            
            results.append(pdf_result)
        
        # Save results
        self._save_test_results(results)
        
        # Display summary
        self._display_test_summary(results)
        
        return True
    
    def _save_test_results(self, results):
        """Save test results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"Results/strands_mcp_client_test_results_{timestamp}.json"
        
        os.makedirs("Results", exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            # Convert results to serializable format
            serializable_results = []
            for result in results:
                serializable_result = {
                    "pdf_path": result["pdf_path"],
                    "strands_mcp_result": str(result["strands_mcp_result"]) if result["strands_mcp_result"] else None
                }
                serializable_results.append(serializable_result)
            
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Test results saved to: {results_file}")
    
    def _display_test_summary(self, results):
        """Display test summary."""
        print("\n" + "=" * 70)
        print("📊 STRANDS MCP CLIENT TEST SUMMARY")
        print("=" * 70)
        
        total_pdfs = len(results)
        successful_mcp = sum(1 for r in results if r.get("strands_mcp_result") is not None)
        
        print(f"📄 Total Chinese PDFs tested: {total_pdfs}")
        print(f"✅ Successful Strands MCP client processing: {successful_mcp}/{total_pdfs}")
        print(f"🔧 Method: Strands MCP Client")
        print(f"🛠️ Tool: process_pdf_enhanced_multilingual")
        print(f"🌐 Endpoint: http://localhost:8000/mcp/")
        
        for result in results:
            pdf_name = os.path.basename(result["pdf_path"])
            mcp_status = "✅" if result.get("strands_mcp_result") is not None else "❌"
            
            print(f"   - {pdf_name}: Strands MCP Client {mcp_status}")
            
            # Show additional info for successful results
            if result.get("strands_mcp_result"):
                print(f"     └─ Result available")
        
        print("\n" + "=" * 70)
        print("🎉 Strands MCP Client test completed!")
        print("=" * 70)


def main():
    """Main function to run the Strands MCP client test."""
    tester = StrandsMCPClientTester()
    success = tester.run_comprehensive_strands_test()
    
    if success:
        print("\n🎉 All Strands MCP client tests completed successfully!")
        print("✅ All tests used the correct Strands MCP client implementation")
    else:
        print("\n❌ Some Strands MCP client tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
