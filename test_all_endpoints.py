#!/usr/bin/env python3
"""
Comprehensive test script for MCP Tools and API Endpoints
Tests all endpoints and generates a detailed report
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict


class APITester:
    def __init__(self, base_url: str = "http://127.0.0.1:8003"):
        self.base_url = base_url
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "base_url": base_url,
            "tests": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0
            }
        }
    
    def test_endpoint(self, name: str, method: str, endpoint: str, 
                     data: Dict = None, expected_status: int = 200, 
                     timeout: int = 10) -> Dict:
        """Test a single endpoint"""
        url = f"{self.base_url}{endpoint}"
        test_result = {
            "name": name,
            "method": method,
            "endpoint": endpoint,
            "url": url,
            "status_code": None,
            "response_time": None,
            "success": False,
            "error": None,
            "response": None
        }
        
        try:
            start_time = time.time()
            
            if method.upper() == "GET":
                response = requests.get(url, timeout=timeout)
            elif method.upper() == "POST":
                headers = {"Content-Type": "application/json"}
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            test_result["status_code"] = response.status_code
            test_result["response_time"] = time.time() - start_time
            
            if response.status_code == expected_status:
                test_result["success"] = True
                try:
                    test_result["response"] = response.json()
                except:
                    test_result["response"] = response.text[:500]  # Truncate long responses
            else:
                test_result["error"] = f"Expected status {expected_status}, got {response.status_code}"
                
        except Exception as e:
            test_result["error"] = str(e)
            test_result["success"] = False
        
        return test_result
    
    def run_health_tests(self):
        """Test health and basic endpoints"""
        print("🔍 Testing Health and Basic Endpoints...")
        
        tests = [
            ("Root Endpoint", "GET", "/"),
            ("Health Check", "GET", "/health"),
            ("API Documentation", "GET", "/docs", None, 404),  # Intentionally disabled
            ("OpenAPI Spec", "GET", "/openapi.json", None, 404),  # Intentionally disabled
        ]
        
        for test in tests:
            if len(test) == 3:
                name, method, endpoint = test
                result = self.test_endpoint(name, method, endpoint)
            elif len(test) == 5:
                name, method, endpoint, data, expected_status = test
                result = self.test_endpoint(name, method, endpoint, data, expected_status)
            else:
                continue
                
            self.results["tests"][name] = result
            self.update_summary(result)
            
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  {status} {name}: {result['status_code']} ({result['response_time']:.2f}s)")
    
    def run_core_api_tests(self):
        """Test core API endpoints"""
        print("\n🔍 Testing Core API Endpoints...")
        
        # Test text analysis
        text_data = {
            "content": "This is a great product! I love it!",
            "language": "en"
        }
        result = self.test_endpoint("Text Analysis", "POST", "/analyze/text", text_data)
        self.results["tests"]["Text Analysis"] = result
        self.update_summary(result)
        
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"  {status} Text Analysis: {result['status_code']} ({result['response_time']:.2f}s)")
        
        # Test models endpoint
        result = self.test_endpoint("Get Models", "GET", "/models")
        self.results["tests"]["Get Models"] = result
        self.update_summary(result)
        
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"  {status} Get Models: {result['status_code']} ({result['response_time']:.2f}s)")
    
    def run_advanced_analytics_tests(self):
        """Test advanced analytics endpoints"""
        print("\n🔍 Testing Advanced Analytics Endpoints...")
        
        # Test advanced analytics health
        result = self.test_endpoint("Advanced Analytics Health", "GET", "/advanced-analytics/health")
        self.results["tests"]["Advanced Analytics Health"] = result
        self.update_summary(result)
        
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"  {status} Advanced Analytics Health: {result['status_code']} ({result['response_time']:.2f}s)")
        
        # Test forecasting endpoint
        forecasting_data = {
            "data": [{"date": "2023-01-01", "sales": 100, "temperature": 20}],
            "target_variables": ["sales"],
            "forecast_horizon": 7,
            "model_type": "ensemble",
            "confidence_level": 0.95
        }
        result = self.test_endpoint("Multivariate Forecasting", "POST", "/advanced-analytics/forecasting-test", forecasting_data)
        self.results["tests"]["Multivariate Forecasting"] = result
        self.update_summary(result)
        
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"  {status} Multivariate Forecasting: {result['status_code']} ({result['response_time']:.2f}s)")
    
    def run_business_endpoints_tests(self):
        """Test business intelligence endpoints"""
        print("\n🔍 Testing Business Intelligence Endpoints...")
        
        business_summary_data = {
            "content": "Sample business content for analysis",
            "summary_length": "executive",
            "focus_areas": ["key_insights", "trends", "actions"],
            "include_metrics": True
        }
        
        executive_summary_data = {
            "content_data": "Sample business content for analysis",
            "summary_type": "business"
        }
        
        tests = [
            ("Business Summary", "/business/summary", business_summary_data),
            ("Executive Summary", "/business/executive-summary", executive_summary_data),
        ]
        
        for name, endpoint, data in tests:
            result = self.test_endpoint(name, "POST", endpoint, data)
            self.results["tests"][name] = result
            self.update_summary(result)
            
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  {status} {name}: {result['status_code']} ({result['response_time']:.2f}s)")
    
    def run_search_endpoints_tests(self):
        """Test search and semantic endpoints"""
        print("\n🔍 Testing Search and Semantic Endpoints...")
        
        search_data = {
            "query": "test query",
            "language": "en"
        }
        
        tests = [
            ("Semantic Search", "/semantic/search", search_data),
            ("Knowledge Graph Search", "/search/knowledge-graph", search_data),
        ]
        
        for name, endpoint, data in tests:
            result = self.test_endpoint(name, "POST", endpoint, data)
            self.results["tests"][name] = result
            self.update_summary(result)
            
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  {status} {name}: {result['status_code']} ({result['response_time']:.2f}s)")
    
    def run_analytics_endpoints_tests(self):
        """Test analytics endpoints"""
        print("\n🔍 Testing Analytics Endpoints...")
        
        analytics_data = {
            "data": [{"value": 100, "timestamp": "2023-01-01"}],
            "analysis_type": "trend"
        }
        
        tests = [
            ("Predictive Analytics", "/analytics/predictive", analytics_data),
            ("Scenario Analysis", "/analytics/scenario", analytics_data),
            ("Performance Optimization", "/analytics/performance", {}),
        ]
        
        for name, endpoint, data in tests:
            result = self.test_endpoint(name, "POST", endpoint, data)
            self.results["tests"][name] = result
            self.update_summary(result)
            
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  {status} {name}: {result['status_code']} ({result['response_time']:.2f}s)")
    
    def update_summary(self, result: Dict):
        """Update test summary"""
        self.results["summary"]["total_tests"] += 1
        
        if result["success"]:
            self.results["summary"]["passed"] += 1
        elif result["error"]:
            self.results["summary"]["errors"] += 1
        else:
            self.results["summary"]["failed"] += 1
    
    def run_all_tests(self):
        """Run all test suites"""
        print("🚀 Starting Comprehensive API Testing")
        print("=" * 60)
        
        # Wait for server to be ready
        print("⏳ Waiting for server to be ready...")
        time.sleep(5)
        
        self.run_health_tests()
        self.run_core_api_tests()
        self.run_advanced_analytics_tests()
        self.run_business_endpoints_tests()
        self.run_search_endpoints_tests()
        self.run_analytics_endpoints_tests()
        
        self.print_summary()
        self.save_results()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        summary = self.results["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⚠️ Errors: {summary['errors']}")
        
        success_rate = (summary['passed'] / summary['total_tests']) * 100 if summary['total_tests'] > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        
        print("\n📋 DETAILED RESULTS:")
        for test_name, result in self.results["tests"].items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            print(f"  {status} {test_name}: {result['status_code']} ({result['response_time']:.2f}s)")
            if result["error"]:
                print(f"    Error: {result['error']}")
    
    def save_results(self):
        """Save test results to file"""
        filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to: {filename}")

def main():
    """Main function"""
    print("🔧 MCP Tools and API Endpoints Testing Suite")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get("http://127.0.0.1:8003/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running on http://127.0.0.1:8003")
        else:
            print("⚠️ Server responded with unexpected status code")
    except requests.exceptions.RequestException:
        print("❌ Server is not running on http://127.0.0.1:8003")
        print("Please start the server with: .venv/Scripts/python.exe main.py")
        return
    
    # Run tests
    tester = APITester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
