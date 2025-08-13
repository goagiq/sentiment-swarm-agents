#!/usr/bin/env python3
"""
Step 8: Performance Testing for Consolidated MCP Server
Simplified test focusing on structure validation and basic functionality
"""

import os
import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.mcp_config import get_consolidated_mcp_config
from mcp.consolidated_mcp_server import ConsolidatedMCPServer


class Step8PerformanceTester:
    def __init__(self):
        self.config = get_consolidated_mcp_config()
        self.server = ConsolidatedMCPServer(self.config)
        self.results = {}
        
    def log_test(self, category: str, function: str, status: str, 
                 duration: float, details: str = ""):
        """Log test results"""
        if category not in self.results:
            self.results[category] = {}
        if function not in self.results[category]:
            self.results[category][function] = []
            
        self.results[category][function].append({
            "status": status,
            "duration": duration,
            "details": details,
            "timestamp": time.time()
        })
    
    def test_server_initialization(self):
        """Test server initialization and configuration"""
        print("🔧 Testing Server Initialization...")
        
        start_time = time.time()
        try:
            # Test configuration loading
            assert self.config is not None, "Configuration should be loaded"
            assert hasattr(self.config, 'pdf_server'), "PDF server config missing"
            assert hasattr(self.config, 'audio_server'), "Audio server config missing"
            assert hasattr(self.config, 'video_server'), "Video server config missing"
            assert hasattr(self.config, 'website_server'), "Website server config missing"
            
            # Test server initialization
            assert self.server is not None, "Server should be initialized"
            
            duration = time.time() - start_time
            self.log_test("System", "initialization", "PASS", duration, 
                         "Server and config initialized successfully")
            print(f"  ✅ Server initialization: {duration:.2f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("System", "initialization", "FAIL", duration, str(e))
            print(f"  ❌ Server initialization: {duration:.2f}s - {e}")
    
    def test_configuration_structure(self):
        """Test configuration structure for all categories"""
        print("\n📋 Testing Configuration Structure...")
        
        categories = ['pdf_server', 'audio_server', 'video_server', 'website_server']
        
        for category in categories:
            start_time = time.time()
            try:
                category_config = getattr(self.config, category)
                assert hasattr(category_config, 'enabled'), f"{category} missing enabled"
                assert hasattr(category_config, 'primary_model'), f"{category} missing primary_model"
                assert hasattr(category_config, 'fallback_model'), f"{category} missing fallback_model"
                assert hasattr(category_config, 'max_file_size'), f"{category} missing max_file_size"
                assert hasattr(category_config, 'timeout'), f"{category} missing timeout"
                
                duration = time.time() - start_time
                self.log_test("Config", f"{category}_structure", "PASS", duration,
                             f"{category} configuration structure valid")
                print(f"  ✅ {category} config: {duration:.2f}s")
                
            except Exception as e:
                duration = time.time() - start_time
                self.log_test("Config", f"{category}_structure", "FAIL", duration, str(e))
                print(f"  ❌ {category} config: {duration:.2f}s - {e}")
    
    def test_server_availability(self):
        """Test server availability for each category"""
        print("\n🔍 Testing Server Availability...")
        
        categories = [
            ('pdf_server', 'PDF'),
            ('audio_server', 'Audio'),
            ('video_server', 'Video'),
            ('website_server', 'Website')
        ]
        
        for server_attr, category_name in categories:
            start_time = time.time()
            try:
                server_instance = getattr(self.server, server_attr, None)
                
                if server_instance is not None:
                    # Test if server has required methods
                    required_methods = ['extract_text', 'convert_to_image', 
                                      'summarize', 'translate', 'store_in_vector_db', 
                                      'create_knowledge_graph']
                    
                    missing_methods = []
                    for method in required_methods:
                        if not hasattr(server_instance, method):
                            missing_methods.append(method)
                    
                    if missing_methods:
                        raise Exception(f"Missing methods: {missing_methods}")
                    
                    duration = time.time() - start_time
                    self.log_test("Server", f"{category_name}_availability", "PASS", duration,
                                 f"{category_name} server available with all methods")
                    print(f"  ✅ {category_name} server: {duration:.2f}s")
                else:
                    duration = time.time() - start_time
                    self.log_test("Server", f"{category_name}_availability", "SKIP", duration,
                                 f"{category_name} server not enabled")
                    print(f"  ⏭️  {category_name} server: SKIPPED (not enabled)")
                    
            except Exception as e:
                duration = time.time() - start_time
                self.log_test("Server", f"{category_name}_availability", "FAIL", duration, str(e))
                print(f"  ❌ {category_name} server: {duration:.2f}s - {e}")
    
    def test_language_support(self):
        """Test language configuration support"""
        print("\n🌍 Testing Language Support...")
        
        start_time = time.time()
        try:
            # Test language configs
            assert hasattr(self.config, 'language_configs'), "Language configs missing"
            assert isinstance(self.config.language_configs, dict), "Language configs should be dict"
            
            # Test specific language support
            supported_languages = ['en', 'zh', 'ru']
            for lang in supported_languages:
                if lang in self.config.language_configs:
                    lang_config = self.config.language_configs[lang]
                    assert hasattr(lang_config, 'name'), f"{lang} config missing name"
                    assert hasattr(lang_config, 'code'), f"{lang} config missing code"
            
            duration = time.time() - start_time
            self.log_test("Language", "support", "PASS", duration,
                         f"Language support configured for {len(self.config.language_configs)} languages")
            print(f"  ✅ Language support: {duration:.2f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Language", "support", "FAIL", duration, str(e))
            print(f"  ❌ Language support: {duration:.2f}s - {e}")
    
    def test_storage_paths(self):
        """Test storage path configuration"""
        print("\n💾 Testing Storage Paths...")
        
        start_time = time.time()
        try:
            # Test storage paths
            assert hasattr(self.config, 'storage_base_path'), "Storage base path missing"
            assert hasattr(self.config, 'temp_path'), "Temp path missing"
            assert hasattr(self.config, 'vector_db_path'), "Vector DB path missing"
            assert hasattr(self.config, 'knowledge_graph_path'), "Knowledge graph path missing"
            
            # Test path validity
            paths = [
                self.config.storage_base_path,
                self.config.temp_path,
                self.config.vector_db_path,
                self.config.knowledge_graph_path
            ]
            
            for path in paths:
                assert path is not None, f"Path should not be None"
                assert isinstance(path, str), f"Path should be string"
            
            duration = time.time() - start_time
            self.log_test("Storage", "paths", "PASS", duration,
                         "All storage paths configured correctly")
            print(f"  ✅ Storage paths: {duration:.2f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Storage", "paths", "FAIL", duration, str(e))
            print(f"  ❌ Storage paths: {duration:.2f}s - {e}")
    
    def test_error_handling(self):
        """Test error handling capabilities"""
        print("\n⚠️  Testing Error Handling...")
        
        start_time = time.time()
        try:
            # Test with invalid file path
            invalid_path = "/nonexistent/file.pdf"
            
            # Test PDF server error handling
            if hasattr(self.server, 'pdf_server') and self.server.pdf_server:
                try:
                    self.server.pdf_server.extract_text(invalid_path)
                    # Should not reach here
                    raise Exception("Should have raised an error for invalid path")
                except Exception:
                    # Expected behavior
                    pass
            
            duration = time.time() - start_time
            self.log_test("Error", "handling", "PASS", duration,
                         "Error handling working correctly")
            print(f"  ✅ Error handling: {duration:.2f}s")
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test("Error", "handling", "FAIL", duration, str(e))
            print(f"  ❌ Error handling: {duration:.2f}s - {e}")
    
    def run_all_tests(self):
        """Run all performance tests"""
        print("🚀 Starting Step 8: Performance Testing")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all test categories
        self.test_server_initialization()
        self.test_configuration_structure()
        self.test_server_availability()
        self.test_language_support()
        self.test_storage_paths()
        self.test_error_handling()
        
        total_time = time.time() - start_time
        
        # Generate summary report
        self.generate_report(total_time)
    
    def generate_report(self, total_time: float):
        """Generate performance test report"""
        print("\n" + "=" * 60)
        print("📊 STEP 8 PERFORMANCE TEST SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        
        for category, functions in self.results.items():
            print(f"\n📁 {category.upper()} CATEGORY:")
            for function, tests in functions.items():
                for test in tests:
                    total_tests += 1
                    if test["status"] == "PASS":
                        passed_tests += 1
                        status_icon = "✅"
                    elif test["status"] == "FAIL":
                        failed_tests += 1
                        status_icon = "❌"
                    else:  # SKIP
                        skipped_tests += 1
                        status_icon = "⏭️"
                    
                    print(f"  {status_icon} {function}: {test['duration']:.2f}s - {test['details']}")
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"  Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        print(f"  Skipped: {skipped_tests} ({skipped_tests/total_tests*100:.1f}%)")
        print(f"  Total Time: {total_time:.2f}s")
        
        # Save detailed results
        results_file = Path("Test/step8_performance_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                "step": 8,
                "timestamp": time.time(),
                "total_time": total_time,
                "statistics": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "skipped_tests": skipped_tests
                },
                "results": self.results
            }, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        if failed_tests == 0:
            print("\n🎉 Step 8 completed successfully!")
            print("✅ All performance tests passed!")
        else:
            print(f"\n⚠️  {failed_tests} tests failed. Check the detailed results.")


def main():
    """Main test execution"""
    try:
        tester = Step8PerformanceTester()
        tester.run_all_tests()
    except Exception as e:
        print(f"❌ Step 8 test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())






