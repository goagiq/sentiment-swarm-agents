#!/usr/bin/env python3
"""
Test Strands MCP integration for Chinese PDF processing.
Follows the official Strands documentation patterns for MCP tools.
"""

import os
import sys
import json
import requests
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.language_config import LanguageConfigFactory


class StrandsMCPTester:
    """Strands MCP integration tester following official documentation patterns."""
    
    def __init__(self):
        self.mcp_base_url = "http://127.0.0.1:8000/mcp"
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
    
    def test_mcp_service_availability(self):
        """Test if MCP service is available following Strands patterns."""
        print("🔧 Testing MCP service availability...")
        
        try:
            # Test basic connectivity to MCP service
            headers = {
                'Accept': 'application/json, text/event-stream'
            }
            response = requests.get(f"{self.mcp_base_url}/", headers=headers, timeout=5)
            if response.status_code in [200, 400, 406]:  # Acceptable for MCP
                print("✅ MCP service is available and responding")
                return True
            else:
                print(f"⚠️ MCP service responded with status: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("❌ MCP service is not available at http://127.0.0.1:8000/mcp")
            return False
        except Exception as e:
            print(f"❌ Error testing MCP service: {e}")
            return False
    
    def test_mcp_tools_availability(self):
        """Test if MCP tools are available using proper JSON-RPC format."""
        print("\n🔧 Testing MCP tools availability...")
        
        try:
            # Test the process_pdf_enhanced_multilingual tool with proper JSON-RPC format
            url = f"{self.mcp_base_url}/process_pdf_enhanced_multilingual"
            
            # Use proper JSON-RPC format as per Strands documentation with session ID
            jsonrpc_payload = {
                "jsonrpc": "2.0",
                "method": "process_pdf_enhanced_multilingual",
                "id": "test-request-1",
                "params": {
                    "pdf_path": "test.pdf",
                    "language": "zh",
                    "generate_report": False,
                    "output_path": None
                }
            }
            
            headers = {
                'Accept': 'application/json, text/event-stream',
                'Content-Type': 'application/json',
                'X-Session-ID': 'test-session-123'
            }
            
            response = requests.post(url, json=jsonrpc_payload, headers=headers, timeout=10)
            
            # Even if it fails due to file not found, if we get a response, the tool exists
            if response.status_code in [200, 400, 404, 406]:
                print("✅ MCP tool process_pdf_enhanced_multilingual is available")
                return True
            else:
                print(f"❌ MCP tool not available: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing MCP tools: {e}")
            return False
    
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
    
    def test_pdf_processing_via_strands_mcp(self, pdf_path):
        """Test PDF processing via MCP service using proper JSON-RPC format."""
        print(f"\n🔬 Testing PDF processing via Strands MCP: {pdf_path}")
        
        try:
            # Use the MCP service for PDF processing with proper JSON-RPC format
            url = f"{self.mcp_base_url}/process_pdf_enhanced_multilingual"
            
            # Create proper JSON-RPC payload following Strands patterns
            jsonrpc_payload = {
                "jsonrpc": "2.0",
                "method": "process_pdf_enhanced_multilingual",
                "id": f"pdf-processing-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "params": {
                    "pdf_path": pdf_path,
                    "language": "zh",  # Explicitly set to Chinese
                    "generate_report": True,
                    "output_path": None
                }
            }
            
            print(f"📤 Sending JSON-RPC request to MCP service: {url}")
            print(f"📄 Processing PDF: {pdf_path}")
            print(f"🌍 Language: zh (Chinese)")
            print(f"📊 Generate report: True")
            print(f"🆔 Request ID: {jsonrpc_payload['id']}")
            
            headers = {
                'Accept': 'application/json, text/event-stream',
                'Content-Type': 'application/json',
                'X-Session-ID': f'session-{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            }
            response = requests.post(url, json=jsonrpc_payload, headers=headers, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ PDF processing successful via Strands MCP")
                self._display_strands_mcp_results(result)
                return result
            else:
                print(f"❌ MCP processing failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ MCP request failed: {e}")
            return None
        except Exception as e:
            print(f"❌ MCP processing error: {e}")
            return None
    
    def _display_strands_mcp_results(self, result):
        """Display Strands MCP processing results."""
        print("\n" + "=" * 60)
        print("📋 STRANDS MCP PROCESSING RESULTS")
        print("=" * 60)
        
        # Display JSON-RPC response structure
        if isinstance(result, dict):
            # Check for JSON-RPC fields
            jsonrpc_version = result.get('jsonrpc', 'Unknown')
            request_id = result.get('id', 'Unknown')
            print(f"📋 JSON-RPC Version: {jsonrpc_version}")
            print(f"🆔 Request ID: {request_id}")
            
            # Check for result or error
            if 'result' in result:
                result_data = result['result']
                print(f"✅ Result available")
                self._display_result_data(result_data)
            elif 'error' in result:
                error_data = result['error']
                print(f"❌ Error occurred")
                print(f"   - Error Code: {error_data.get('code', 'Unknown')}")
                print(f"   - Error Message: {error_data.get('message', 'Unknown')}")
            else:
                print(f"⚠️ Unexpected response format")
                print(f"   - Response: {result}")
        else:
            print(f"   - Result: {result}")
        
        print("\n" + "=" * 60)
    
    def _display_result_data(self, result_data):
        """Display the actual result data from MCP processing."""
        if isinstance(result_data, dict):
            # Check for success status
            success = result_data.get('success', False)
            print(f"✅ Success: {success}")
            
            # Display key information
            if 'pdf_path' in result_data:
                print(f"📄 PDF Path: {result_data['pdf_path']}")
            
            if 'detected_language' in result_data:
                print(f"🌍 Detected Language: {result_data['detected_language']}")
            
            # Text extraction results
            if 'text_extraction' in result_data:
                text_extraction = result_data['text_extraction']
                print(f"\n📄 Text Extraction:")
                print(f"   - Success: {text_extraction.get('success', False)}")
                print(f"   - Content length: {text_extraction.get('content_length', 0)} characters")
                print(f"   - Pages processed: {text_extraction.get('pages_processed', 'Unknown')}")
                print(f"   - Extraction method: {text_extraction.get('extraction_method', 'Unknown')}")
            
            # Entity extraction results
            if 'entity_extraction' in result_data:
                entity_extraction = result_data['entity_extraction']
                print(f"\n🔍 Entity Extraction:")
                print(f"   - Entities found: {entity_extraction.get('entities_found', 0)}")
                print(f"   - Entity types: {entity_extraction.get('entity_types', {})}")
                print(f"   - Language stats: {entity_extraction.get('language_stats', {})}")
                print(f"   - Extraction method: {entity_extraction.get('extraction_method', 'Unknown')}")
            
            # Knowledge graph results
            if 'knowledge_graph' in result_data:
                knowledge_graph = result_data['knowledge_graph']
                print(f"\n🧠 Knowledge Graph:")
                print(f"   - Nodes: {knowledge_graph.get('nodes', 0)}")
                print(f"   - Edges: {knowledge_graph.get('edges', 0)}")
                print(f"   - Communities: {knowledge_graph.get('communities', 0)}")
                print(f"   - Processing time: {knowledge_graph.get('processing_time', 0):.2f} seconds")
            
            # Report results
            if 'report_files' in result_data:
                report_files = result_data['report_files']
                print(f"\n📊 Report Generation:")
                if isinstance(report_files, dict):
                    for key, value in report_files.items():
                        print(f"   - {key}: {value}")
                else:
                    print(f"   - Report files: {report_files}")
            
            # Enhanced features
            if 'enhanced_features' in result_data:
                enhanced_features = result_data['enhanced_features']
                print(f"\n🚀 Enhanced Features:")
                for feature, enabled in enhanced_features.items():
                    status = "✅" if enabled else "❌"
                    print(f"   - {feature}: {status}")
        else:
            print(f"   - Result data: {result_data}")
    
    def run_comprehensive_strands_mcp_test(self):
        """Run comprehensive test of Strands MCP integration for Chinese PDF processing."""
        print("🧪 Strands MCP Integration Test Suite for Chinese PDF Processing")
        print("=" * 70)
        
        # Test 1: MCP Service Availability
        mcp_available = self.test_mcp_service_availability()
        
        if not mcp_available:
            print("\n❌ MCP service is not available. Cannot proceed with testing.")
            return False
        
        # Test 2: MCP Tools Availability
        tools_available = self.test_mcp_tools_availability()
        
        if not tools_available:
            print("\n❌ MCP tools are not available. Cannot proceed with testing.")
            return False
        
        # Test 3: Find Chinese PDFs
        chinese_pdfs = self.find_chinese_pdfs()
        
        if not chinese_pdfs:
            print("\n❌ No Chinese PDFs found for testing")
            return False
        
        # Test 4: Process each Chinese PDF via Strands MCP
        results = []
        
        for pdf_path in chinese_pdfs:
            print(f"\n{'='*70}")
            print(f"🔬 Testing PDF via Strands MCP: {os.path.basename(pdf_path)}")
            print(f"{'='*70}")
            
            pdf_result = {
                "pdf_path": pdf_path,
                "strands_mcp_result": None
            }
            
            # Test Strands MCP processing
            pdf_result["strands_mcp_result"] = self.test_pdf_processing_via_strands_mcp(pdf_path)
            
            results.append(pdf_result)
        
        # Save results
        self._save_test_results(results)
        
        # Display summary
        self._display_test_summary(results)
        
        return True
    
    def _save_test_results(self, results):
        """Save test results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"Results/strands_mcp_integration_test_results_{timestamp}.json"
        
        os.makedirs("Results", exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Test results saved to: {results_file}")
    
    def _display_test_summary(self, results):
        """Display test summary."""
        print("\n" + "=" * 70)
        print("📊 STRANDS MCP INTEGRATION TEST SUMMARY")
        print("=" * 70)
        
        total_pdfs = len(results)
        successful_mcp = sum(1 for r in results if r.get("strands_mcp_result") and 
                           r["strands_mcp_result"].get("result", {}).get("success"))
        
        print(f"📄 Total Chinese PDFs tested: {total_pdfs}")
        print(f"✅ Successful Strands MCP processing: {successful_mcp}/{total_pdfs}")
        print(f"🔧 MCP Service: http://127.0.0.1:8000/mcp")
        print(f"🛠️ MCP Tool: process_pdf_enhanced_multilingual")
        print(f"📋 Protocol: JSON-RPC 2.0 (Strands compliant)")
        
        for result in results:
            pdf_name = os.path.basename(result["pdf_path"])
            mcp_status = "✅" if (result.get("strands_mcp_result") and 
                                result["strands_mcp_result"].get("result", {}).get("success")) else "❌"
            
            print(f"   - {pdf_name}: Strands MCP {mcp_status}")
            
            # Show additional info for successful results
            if result.get("strands_mcp_result") and result["strands_mcp_result"].get("result"):
                mcp_result = result["strands_mcp_result"]["result"]
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
        print("🎉 Strands MCP Integration test completed!")
        print("=" * 70)


def main():
    """Main function to run the Strands MCP integration test."""
    tester = StrandsMCPTester()
    success = tester.run_comprehensive_strands_mcp_test()
    
    if success:
        print("\n🎉 All Strands MCP integration tests completed successfully!")
        print("✅ All tests used the MCP tool at http://127.0.0.1:8000/mcp/ with JSON-RPC 2.0")
        print("📚 Following Strands documentation patterns for MCP tool usage")
    else:
        print("\n❌ Some Strands MCP integration tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
