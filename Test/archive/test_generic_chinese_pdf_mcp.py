#!/usr/bin/env python3
"""
Generic Chinese PDF processing test using existing MCP service.
Tests any Chinese PDF using the MCP service at http://127.0.0.1:8000/mcp
"""

import asyncio
import os
import sys
import requests
import json
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.language_config import LanguageConfigFactory


class GenericChinesePDFTester:
    """Generic tester for Chinese PDF processing using MCP service."""
    
    def __init__(self):
        self.mcp_base_url = "http://127.0.0.1:8000/mcp"
        self.api_base_url = "http://127.0.0.1:8000"
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
            response = requests.get(f"{self.mcp_base_url}/health", timeout=5)
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
    
    def test_api_endpoints(self):
        """Test available API endpoints."""
        print("\n🌐 Testing API endpoints...")
        
        endpoints_to_test = [
            "/process/pdf-enhanced-multilingual",
            "/analyze/pdf",
            "/process/pdf-to-databases"
        ]
        
        available_endpoints = []
        
        for endpoint in endpoints_to_test:
            try:
                response = requests.get(f"{self.api_base_url}{endpoint}", timeout=5)
                if response.status_code in [200, 405]:  # 405 means method not allowed (endpoint exists)
                    print(f"✅ Endpoint available: {endpoint}")
                    available_endpoints.append(endpoint)
                else:
                    print(f"❌ Endpoint not available: {endpoint} (status: {response.status_code})")
            except Exception as e:
                print(f"❌ Error testing endpoint {endpoint}: {e}")
        
        return available_endpoints
    
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
    
    def test_pdf_processing_via_api(self, pdf_path):
        """Test PDF processing via API endpoint."""
        print(f"\n🔬 Testing PDF processing via API: {pdf_path}")
        
        try:
            # Use the enhanced multilingual PDF processing endpoint
            url = f"{self.api_base_url}/process/pdf-enhanced-multilingual"
            
            payload = {
                "pdf_path": pdf_path,
                "language": "zh",  # Explicitly set to Chinese
                "generate_report": True,
                "output_path": None
            }
            
            print(f"📤 Sending request to: {url}")
            print(f"📄 Processing PDF: {pdf_path}")
            
            response = requests.post(url, json=payload, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ PDF processing successful via API")
                self._display_api_results(result)
                return result
            else:
                print(f"❌ API processing failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            return None
        except Exception as e:
            print(f"❌ API processing error: {e}")
            return None
    
    def test_pdf_processing_via_mcp(self, pdf_path):
        """Test PDF processing via MCP service."""
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
            
            response = requests.post(url, json=payload, timeout=300)
            
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
    
    def _display_api_results(self, result):
        """Display API processing results."""
        print("\n" + "=" * 60)
        print("📋 API PROCESSING RESULTS")
        print("=" * 60)
        
        print(f"📄 PDF Path: {result.get('pdf_path', 'Unknown')}")
        print(f"🌍 Detected Language: {result.get('detected_language', 'Unknown')}")
        
        # Text extraction results
        text_extraction = result.get('text_extraction', {})
        print(f"\n📄 Text Extraction:")
        print(f"   - Success: {text_extraction.get('success', False)}")
        print(f"   - Content length: {text_extraction.get('content_length', 0)} characters")
        print(f"   - Pages processed: {text_extraction.get('pages_processed', 'Unknown')}")
        print(f"   - Extraction method: {text_extraction.get('extraction_method', 'Unknown')}")
        
        # Entity extraction results
        entity_extraction = result.get('entity_extraction', {})
        print(f"\n🔍 Entity Extraction:")
        print(f"   - Entities found: {entity_extraction.get('entities_found', 0)}")
        print(f"   - Entity types: {entity_extraction.get('entity_types', {})}")
        print(f"   - Language stats: {entity_extraction.get('language_stats', {})}")
        print(f"   - Extraction method: {entity_extraction.get('extraction_method', 'Unknown')}")
        
        # Knowledge graph results
        knowledge_graph = result.get('knowledge_graph', {})
        print(f"\n🧠 Knowledge Graph:")
        print(f"   - Nodes: {knowledge_graph.get('nodes', 0)}")
        print(f"   - Edges: {knowledge_graph.get('edges', 0)}")
        print(f"   - Communities: {knowledge_graph.get('communities', 0)}")
        print(f"   - Processing time: {knowledge_graph.get('processing_time', 0):.2f} seconds")
        
        # Report results
        report_files = result.get('report_files', {})
        if report_files:
            print(f"\n📊 Report Generation:")
            for key, value in report_files.items():
                print(f"   - {key}: {value}")
        
        # Enhanced features
        enhanced_features = result.get('enhanced_features', {})
        if enhanced_features:
            print(f"\n🚀 Enhanced Features:")
            for feature, enabled in enhanced_features.items():
                status = "✅" if enabled else "❌"
                print(f"   - {feature}: {status}")
        
        print("\n" + "=" * 60)
    
    def _display_mcp_results(self, result):
        """Display MCP processing results."""
        print("\n" + "=" * 60)
        print("📋 MCP PROCESSING RESULTS")
        print("=" * 60)
        
        # Display MCP-specific results
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"   - {key}: {value}")
        else:
            print(f"   - Result: {result}")
        
        print("\n" + "=" * 60)
    
    def run_comprehensive_test(self):
        """Run comprehensive test of Chinese PDF processing."""
        print("🧪 Generic Chinese PDF Processing Test Suite")
        print("=" * 60)
        
        # Test 1: MCP Service Availability
        mcp_available = self.test_mcp_service_availability()
        
        # Test 2: API Endpoints
        available_endpoints = self.test_api_endpoints()
        
        # Test 3: Find Chinese PDFs
        chinese_pdfs = self.find_chinese_pdfs()
        
        if not chinese_pdfs:
            print("\n❌ No Chinese PDFs found for testing")
            return False
        
        # Test 4: Process each Chinese PDF
        results = []
        
        for pdf_path in chinese_pdfs:
            print(f"\n{'='*60}")
            print(f"🔬 Testing PDF: {os.path.basename(pdf_path)}")
            print(f"{'='*60}")
            
            pdf_result = {
                "pdf_path": pdf_path,
                "api_result": None,
                "mcp_result": None
            }
            
            # Test API processing if endpoint is available
            if "/process/pdf-enhanced-multilingual" in available_endpoints:
                pdf_result["api_result"] = self.test_pdf_processing_via_api(pdf_path)
            
            # Test MCP processing if service is available
            if mcp_available:
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
        results_file = f"Results/generic_chinese_pdf_test_results_{timestamp}.json"
        
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
        successful_api = sum(1 for r in results if r.get("api_result") and r["api_result"].get("success"))
        successful_mcp = sum(1 for r in results if r.get("mcp_result") and r["mcp_result"].get("success"))
        
        print(f"📄 Total Chinese PDFs tested: {total_pdfs}")
        print(f"✅ Successful API processing: {successful_api}/{total_pdfs}")
        print(f"✅ Successful MCP processing: {successful_mcp}/{total_pdfs}")
        
        for result in results:
            pdf_name = os.path.basename(result["pdf_path"])
            api_status = "✅" if result.get("api_result") and result["api_result"].get("success") else "❌"
            mcp_status = "✅" if result.get("mcp_result") and result["mcp_result"].get("success") else "❌"
            
            print(f"   - {pdf_name}: API {api_status} | MCP {mcp_status}")
        
        print("\n" + "=" * 60)
        print("🎉 Generic Chinese PDF processing test completed!")
        print("=" * 60)


def main():
    """Main function to run the generic Chinese PDF test."""
    tester = GenericChinesePDFTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
