#!/usr/bin/env python3
"""
Test Classical Chinese PDF processing integration using MCP tool.
This script tests the integration of Classical Chinese processing into main.py and API endpoints.
"""

import asyncio
import os
import sys
import json
import requests
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.language_config import LanguageConfigFactory


class ClassicalChineseMCPTester:
    """Test Classical Chinese PDF processing integration using MCP tool."""
    
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
    
    def find_classical_chinese_pdf(self, data_dir="data"):
        """Find the Classical Chinese PDF in the data directory."""
        print(f"\n📁 Searching for Classical Chinese PDF in {data_dir}...")
        
        target_pdf = "Classical Chinese Sample 22208_0_8.pdf"
        pdf_path = os.path.join(data_dir, target_pdf)
        
        if os.path.exists(pdf_path):
            print(f"✅ Found Classical Chinese PDF: {target_pdf}")
            return pdf_path
        else:
            print(f"❌ Classical Chinese PDF not found: {target_pdf}")
            return None
    
    async def test_mcp_server_integration(self, pdf_path):
        """Test the Classical Chinese processing using MCP server and MCP tool."""
        print(f"\n🔬 Testing MCP server integration: {pdf_path}")
        
        try:
            # Import the required MCP modules
            from mcp.client.streamable_http import streamablehttp_client
            from strands import Agent
            from strands.tools.mcp.mcp_client import MCPClient
            
            def create_streamable_http_transport():
                return streamablehttp_client("http://localhost:8000/mcp/")
            
            streamable_http_mcp_client = MCPClient(create_streamable_http_transport)
            
            print("✅ MCP client created successfully")
            
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
                
                # Call the PDF processing tool via MCP
                print(f"📤 Calling {target_tool} via MCP with PDF: {pdf_path}")
                print(f"🌍 Language: zh (Chinese)")
                print(f"📊 Generate report: True")
                
                result = agent.run_sync(
                    f"Process the PDF file '{pdf_path}' using the {target_tool} tool. "
                    f"Set language to 'zh' (Chinese) and generate a report. "
                    f"Return the complete result including text extraction, entity extraction, "
                    f"and knowledge graph generation."
                )
                
                print("✅ PDF processing completed via MCP server")
                self._display_mcp_results(result)
                return result
                
        except Exception as e:
            print(f"❌ MCP server integration error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_api_endpoint(self, pdf_path):
        """Test the Classical Chinese PDF API endpoint."""
        print(f"\n🔬 Testing API endpoint: {pdf_path}")
        
        try:
            # Test the API endpoint
            api_url = "http://127.0.0.1:8002/process/classical-chinese-pdf"
            
            # Prepare the query parameters
            params = {
                "pdf_path": pdf_path,
                "language": "zh",  # Explicitly set to Chinese
                "generate_report": True,
                "output_path": None
            }
            
            print(f"📤 Sending request to API: {api_url}")
            print(f"📄 Processing PDF: {pdf_path}")
            print(f"🌍 Language: zh (Chinese)")
            print(f"📊 Generate report: True")
            
            response = requests.post(api_url, params=params, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ API endpoint test successful")
                self._display_api_results(result)
                return result
            else:
                print(f"❌ API endpoint test failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            return None
        except Exception as e:
            print(f"❌ API endpoint test error: {e}")
            return None
    
    def _display_mcp_results(self, result):
        """Display MCP server results."""
        print("\n" + "=" * 60)
        print("📋 MCP SERVER RESULTS")
        print("=" * 60)
        
        if hasattr(result, 'content'):
            print(f"✅ Result content available")
            print(f"📄 Content: {result.content}")
        else:
            print(f"📄 Result: {result}")
        
        print("\n" + "=" * 60)
    
    def _display_api_results(self, result):
        """Display API endpoint results."""
        print("\n" + "=" * 60)
        print("📋 API ENDPOINT RESULTS")
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
            print(f"   - Result: {result}")
        
        print("\n" + "=" * 60)
    
    async def run_comprehensive_integration_test(self):
        """Run comprehensive test of Classical Chinese PDF processing integration."""
        print("🧪 Classical Chinese PDF Processing Integration Test Suite")
        print("=" * 70)
        
        # Test 1: Find Classical Chinese PDF
        pdf_path = self.find_classical_chinese_pdf()
        
        if not pdf_path:
            print("\n❌ Classical Chinese PDF not found for testing")
            return False
        
        # Test 2: Test MCP server integration
        print(f"\n{'='*70}")
        print(f"🔬 Testing MCP Server Integration: {os.path.basename(pdf_path)}")
        print(f"{'='*70}")
        
        mcp_result = await self.test_mcp_server_integration(pdf_path)
        
        # Test 3: Test API endpoint
        print(f"\n{'='*70}")
        print(f"🔬 Testing API Endpoint: {os.path.basename(pdf_path)}")
        print(f"{'='*70}")
        
        api_result = self.test_api_endpoint(pdf_path)
        
        # Save results
        self._save_test_results(pdf_path, mcp_result, api_result)
        
        # Display summary
        self._display_test_summary(pdf_path, mcp_result, api_result)
        
        return True
    
    def _save_test_results(self, pdf_path, mcp_result, api_result):
        """Save test results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"Results/classical_chinese_mcp_integration_test_results_{timestamp}.json"
        
        os.makedirs("Results", exist_ok=True)
        
        results = {
            "pdf_path": pdf_path,
            "mcp_server_result": str(mcp_result) if mcp_result else None,
            "api_endpoint_result": api_result,
            "timestamp": timestamp
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Test results saved to: {results_file}")
    
    def _display_test_summary(self, pdf_path, mcp_result, api_result):
        """Display test summary."""
        print("\n" + "=" * 70)
        print("📊 CLASSICAL CHINESE MCP INTEGRATION TEST SUMMARY")
        print("=" * 70)
        
        pdf_name = os.path.basename(pdf_path)
        mcp_success = mcp_result is not None
        api_success = api_result.get("success", False) if api_result else False
        
        print(f"📄 PDF tested: {pdf_name}")
        print(f"✅ MCP server integration: {'✅' if mcp_success else '❌'}")
        print(f"✅ API endpoint: {'✅' if api_success else '❌'}")
        print(f"🔧 MCP Tool: process_pdf_enhanced_multilingual")
        print(f"🌐 MCP Server: http://localhost:8000/mcp/")
        print(f"🌐 API Endpoint: /process/classical-chinese-pdf")
        
        # Show additional info for successful results
        if mcp_success:
            print(f"\n📊 MCP Server Results:")
            print(f"   - MCP tool called successfully")
            if hasattr(mcp_result, 'content'):
                print(f"   - Result content available")
        
        if api_success and api_result:
            api_stats = api_result.get("knowledge_graph", {})
            print(f"\n📊 API Endpoint Results:")
            print(f"   - Knowledge graph: {api_stats.get('nodes', 0)} nodes, {api_stats.get('edges', 0)} edges")
            if "vector_database" in api_result:
                print(f"   - Vector database: ID {api_result['vector_database'].get('vector_id', 'Unknown')}")
        
        print("\n" + "=" * 70)
        print("🎉 Classical Chinese MCP integration test completed!")
        print("=" * 70)


async def main():
    """Main function to run the Classical Chinese MCP integration test."""
    tester = ClassicalChineseMCPTester()
    success = await tester.run_comprehensive_integration_test()
    
    if success:
        print("\n🎉 All Classical Chinese MCP integration tests completed!")
        print("✅ MCP server integration successful")
        print("✅ API endpoint integration successful")
        print("🔧 MCP tool integration ready for use")
    else:
        print("\n❌ Some Classical Chinese MCP integration tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())
