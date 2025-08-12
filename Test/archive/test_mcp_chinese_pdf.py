#!/usr/bin/env python3
"""
Test Chinese PDF processing using MCP service at http://127.0.0.1:8000/mcp/
Uses the process_pdf_enhanced_multilingual tool for multilingual PDF processing.
"""

import os
import sys
import requests
import json
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.language_config import LanguageConfigFactory


class MCPChinesePDFTester:
    """Tester for Chinese PDF processing using MCP service."""
    
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
        """Test if MCP service is available."""
        print("🔧 Testing MCP service availability...")
        
        try:
            # Test basic connectivity to MCP service
            headers = {
                'Accept': 'application/json, text/event-stream'
            }
            response = requests.get(f"{self.mcp_base_url}/", headers=headers, timeout=5)
            if response.status_code == 200:
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
    
    def test_pdf_processing_via_mcp(self, pdf_path):
        """Test PDF processing via MCP service using process_pdf_enhanced_multilingual."""
        print(f"\n🔬 Testing PDF processing via MCP: {pdf_path}")
        
        try:
            # Use the MCP service for PDF processing
            url = f"{self.mcp_base_url}/process_pdf_enhanced_multilingual"
            
            payload = {
                "pdf_path": pdf_path,
                "language": "zh",  # Explicitly set to Chinese
                "generate_report": True,
                "output_path": None
            }
            
            print(f"📤 Sending request to MCP service: {url}")
            print(f"📄 Processing PDF: {pdf_path}")
            print(f"🌍 Language: zh (Chinese)")
            print(f"📊 Generate report: True")
            
            headers = {
                'Accept': 'application/json, text/event-stream',
                'Content-Type': 'application/json'
            }
            response = requests.post(url, json=payload, headers=headers, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ PDF processing successful via MCP")
                self._display_mcp_results(result)
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
    
    def _display_mcp_results(self, result):
        """Display MCP processing results."""
        print("\n" + "=" * 60)
        print("📋 MCP PROCESSING RESULTS")
        print("=" * 60)
        
        # Display MCP-specific results
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
            
            # Display any other fields
            for key, value in result.items():
                if key not in ['success', 'pdf_path', 'detected_language', 'text_extraction', 
                              'entity_extraction', 'knowledge_graph', 'report_files', 'enhanced_features']:
                    print(f"   - {key}: {value}")
        else:
            print(f"   - Result: {result}")
        
        print("\n" + "=" * 60)
    
    def run_comprehensive_test(self):
        """Run comprehensive test of Chinese PDF processing via MCP."""
        print("🧪 MCP Chinese PDF Processing Test Suite")
        print("=" * 60)
        
        # Test 1: MCP Service Availability
        mcp_available = self.test_mcp_service_availability()
        
        if not mcp_available:
            print("\n❌ MCP service is not available. Cannot proceed with testing.")
            return False
        
        # Test 2: Find Chinese PDFs
        chinese_pdfs = self.find_chinese_pdfs()
        
        if not chinese_pdfs:
            print("\n❌ No Chinese PDFs found for testing")
            return False
        
        # Test 3: Process each Chinese PDF
        results = []
        
        for pdf_path in chinese_pdfs:
            print(f"\n{'='*60}")
            print(f"🔬 Testing PDF: {os.path.basename(pdf_path)}")
            print(f"{'='*60}")
            
            pdf_result = {
                "pdf_path": pdf_path,
                "mcp_result": None
            }
            
            # Test MCP processing
            pdf_result["mcp_result"] = self.test_pdf_processing_via_mcp(pdf_path)
            
            results.append(pdf_result)
        
        # Save results
        self._save_test_results(results)
        
        # Display summary
        self._display_test_summary(results)
        
        return True
    
    def _save_test_results(self, results):
        """Save test results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"Results/mcp_chinese_pdf_test_results_{timestamp}.json"
        
        os.makedirs("Results", exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Test results saved to: {results_file}")
    
    def _display_test_summary(self, results):
        """Display test summary."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_pdfs = len(results)
        successful_mcp = sum(1 for r in results if r.get("mcp_result") and r["mcp_result"].get("success"))
        
        print(f"📄 Total Chinese PDFs tested: {total_pdfs}")
        print(f"✅ Successful MCP processing: {successful_mcp}/{total_pdfs}")
        
        for result in results:
            pdf_name = os.path.basename(result["pdf_path"])
            mcp_status = "✅" if result.get("mcp_result") and result["mcp_result"].get("success") else "❌"
            
            print(f"   - {pdf_name}: MCP {mcp_status}")
            
            # Show additional info for successful results
            if result.get("mcp_result") and result["mcp_result"].get("success"):
                mcp_result = result["mcp_result"]
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
        
        print("\n" + "=" * 60)
        print("🎉 MCP Chinese PDF processing test completed!")
        print("=" * 60)


def main():
    """Main function to run the MCP Chinese PDF test."""
    tester = MCPChinesePDFTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
