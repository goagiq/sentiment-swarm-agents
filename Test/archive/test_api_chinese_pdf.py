#!/usr/bin/env python3
"""
Test API endpoints for Chinese PDF processing.
Uses the existing API server on port 8002.
"""

import os
import sys
import json
import requests
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.language_config import LanguageConfigFactory


class APIClientTester:
    """API client tester for Chinese PDF processing."""
    
    def __init__(self):
        self.api_base_url = "http://127.0.0.1:8002"
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
    
    def test_api_availability(self):
        """Test if API server is available."""
        print("🔧 Testing API server availability...")
        
        try:
            # Test basic connectivity to API server
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API server is available and responding")
                return True
            else:
                print(f"⚠️ API server responded with status: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("❌ API server is not available at http://127.0.0.1:8002")
            return False
        except Exception as e:
            print(f"❌ Error testing API server: {e}")
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
    
    def test_pdf_processing_via_api(self, pdf_path):
        """Test PDF processing via API endpoints."""
        print(f"\n🔬 Testing PDF processing via API: {pdf_path}")
        
        try:
            # Use the enhanced multilingual PDF processing endpoint
            url = f"{self.api_base_url}/process-pdf-enhanced-multilingual"
            
            # Prepare the query parameters (not JSON body)
            params = {
                "pdf_path": pdf_path,
                "language": "zh",  # Explicitly set to Chinese
                "generate_report": True,
                "output_path": None
            }
            
            print(f"📤 Sending request to API: {url}")
            print(f"📄 Processing PDF: {pdf_path}")
            print(f"🌍 Language: zh (Chinese)")
            print(f"📊 Generate report: True")
            
            response = requests.post(url, params=params, timeout=300)
            
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
    
    def _display_api_results(self, result):
        """Display API processing results."""
        print("\n" + "=" * 60)
        print("📋 API PROCESSING RESULTS")
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
    
    def run_comprehensive_api_test(self):
        """Run comprehensive test of API endpoints for Chinese PDF processing."""
        print("🧪 API Endpoint Test Suite for Chinese PDF Processing")
        print("=" * 70)
        
        # Test 1: API Server Availability
        api_available = self.test_api_availability()
        
        if not api_available:
            print("\n❌ API server is not available. Cannot proceed with testing.")
            return False
        
        # Test 2: Find Chinese PDFs
        chinese_pdfs = self.find_chinese_pdfs()
        
        if not chinese_pdfs:
            print("\n❌ No Chinese PDFs found for testing")
            return False
        
        # Test 3: Process each Chinese PDF via API
        results = []
        
        for pdf_path in chinese_pdfs:
            print(f"\n{'='*70}")
            print(f"🔬 Testing PDF via API: {os.path.basename(pdf_path)}")
            print(f"{'='*70}")
            
            pdf_result = {
                "pdf_path": pdf_path,
                "api_result": None
            }
            
            # Test API processing
            pdf_result["api_result"] = self.test_pdf_processing_via_api(pdf_path)
            
            results.append(pdf_result)
        
        # Save results
        self._save_test_results(results)
        
        # Display summary
        self._display_test_summary(results)
        
        return True
    
    def _save_test_results(self, results):
        """Save test results to JSON file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"Results/api_chinese_pdf_test_results_{timestamp}.json"
        
        os.makedirs("Results", exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Test results saved to: {results_file}")
    
    def _display_test_summary(self, results):
        """Display test summary."""
        print("\n" + "=" * 70)
        print("📊 API CHINESE PDF TEST SUMMARY")
        print("=" * 70)
        
        total_pdfs = len(results)
        successful_api = sum(1 for r in results if r.get("api_result") and r["api_result"].get("success"))
        
        print(f"📄 Total Chinese PDFs tested: {total_pdfs}")
        print(f"✅ Successful API processing: {successful_api}/{total_pdfs}")
        print(f"🔧 API Server: http://127.0.0.1:8002")
        print(f"🛠️ Endpoint: /process-pdf-enhanced-multilingual")
        
        for result in results:
            pdf_name = os.path.basename(result["pdf_path"])
            api_status = "✅" if (result.get("api_result") and 
                                result["api_result"].get("success")) else "❌"
            
            print(f"   - {pdf_name}: API {api_status}")
            
            # Show additional info for successful results
            if result.get("api_result") and result["api_result"].get("success"):
                api_result = result["api_result"]
                if "text_extraction" in api_result:
                    content_length = api_result["text_extraction"].get("content_length", 0)
                    print(f"     └─ Content length: {content_length} characters")
                if "entity_extraction" in api_result:
                    entities_found = api_result["entity_extraction"].get("entities_found", 0)
                    print(f"     └─ Entities found: {entities_found}")
                if "knowledge_graph" in api_result:
                    nodes = api_result["knowledge_graph"].get("nodes", 0)
                    edges = api_result["knowledge_graph"].get("edges", 0)
                    print(f"     └─ Knowledge graph: {nodes} nodes, {edges} edges")
        
        print("\n" + "=" * 70)
        print("🎉 API Chinese PDF test completed!")
        print("=" * 70)


def main():
    """Main function to run the API test."""
    tester = APIClientTester()
    success = tester.run_comprehensive_api_test()
    
    if success:
        print("\n🎉 All API tests completed successfully!")
        print("✅ All tests used the API endpoints at http://127.0.0.1:8002")
    else:
        print("\n❌ Some API tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
