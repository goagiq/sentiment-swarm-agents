#!/usr/bin/env python3
"""
Phase 4: API-based Testing & Validation of Unified MCP Server

This script tests the unified MCP server functionality through API endpoints
to validate the consolidation was successful.
"""

import json
import time
import sys
import requests
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger


class Phase4APIValidator:
    """API-based validator for Phase 4 MCP tools consolidation."""
    
    def __init__(self):
        self.api_base_url = "http://localhost:8003"
        self.test_results = {
            "test_run": {
                "timestamp": datetime.now().isoformat(),
                "phase": "Phase 4: API-based Testing & Validation",
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "skipped_tests": 0,
                "success_rate": 0.0
            },
            "results": []
        }
        
        # Test data
        self.test_data = {
            "text": (
                "This is a positive test message for sentiment analysis. "
                "We are testing the unified MCP server functionality."
            ),
            "business_text": (
                "Our quarterly earnings increased by 15% compared to last year. "
                "Customer satisfaction scores are at an all-time high."
            )
        }
    
    def test_api_health(self) -> Dict[str, Any]:
        """Test API health endpoint."""
        logger.info("🏥 Testing API Health...")
        
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    "test": "api_health",
                    "status": "PASSED",
                    "result": {
                        "status_code": response.status_code,
                        "data": data
                    }
                }
            else:
                return {
                    "test": "api_health",
                    "status": "FAILED",
                    "result": {
                        "status_code": response.status_code,
                        "error": "Health check failed"
                    }
                }
        except Exception as e:
            return {
                "test": "api_health",
                "status": "FAILED",
                "result": {"error": str(e)}
            }
    
    def test_text_analysis(self) -> Dict[str, Any]:
        """Test text analysis endpoint."""
        logger.info("📝 Testing Text Analysis...")
        
        try:
            payload = {
                "content": self.test_data["text"],
                "language": "en"
            }
            response = requests.post(
                f"{self.api_base_url}/analyze/text",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "test": "text_analysis",
                    "status": "PASSED",
                    "result": {
                        "status_code": response.status_code,
                        "data": data
                    }
                }
            else:
                return {
                    "test": "text_analysis",
                    "status": "FAILED",
                    "result": {
                        "status_code": response.status_code,
                        "error": response.text
                    }
                }
        except Exception as e:
            return {
                "test": "text_analysis",
                "status": "FAILED",
                "result": {"error": str(e)}
            }
    
    def test_business_intelligence(self) -> Dict[str, Any]:
        """Test business intelligence endpoint."""
        logger.info("💼 Testing Business Intelligence...")
        
        try:
            payload = {
                "content_data": self.test_data["business_text"],
                "summary_type": "business",
                "include_metrics": True,
                "include_trends": True
            }
            response = requests.post(
                f"{self.api_base_url}/business/summary",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "test": "business_intelligence",
                    "status": "PASSED",
                    "result": {
                        "status_code": response.status_code,
                        "data": data
                    }
                }
            else:
                return {
                    "test": "business_intelligence",
                    "status": "FAILED",
                    "result": {
                        "status_code": response.status_code,
                        "error": response.text
                    }
                }
        except Exception as e:
            return {
                "test": "business_intelligence",
                "status": "FAILED",
                "result": {"error": str(e)}
            }
    
    def test_agent_status(self) -> Dict[str, Any]:
        """Test agent status endpoint."""
        logger.info("🤖 Testing Agent Status...")
        
        try:
            response = requests.get(
                f"{self.api_base_url}/agents/status",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "test": "agent_status",
                    "status": "PASSED",
                    "result": {
                        "status_code": response.status_code,
                        "data": data
                    }
                }
            else:
                return {
                    "test": "agent_status",
                    "status": "FAILED",
                    "result": {
                        "status_code": response.status_code,
                        "error": response.text
                    }
                }
        except Exception as e:
            return {
                "test": "agent_status",
                "status": "FAILED",
                "result": {"error": str(e)}
            }
    
    def test_models_endpoint(self) -> Dict[str, Any]:
        """Test models endpoint."""
        logger.info("🔧 Testing Models Endpoint...")
        
        try:
            response = requests.get(
                f"{self.api_base_url}/models",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "test": "models_endpoint",
                    "status": "PASSED",
                    "result": {
                        "status_code": response.status_code,
                        "data": data
                    }
                }
            else:
                return {
                    "test": "models_endpoint",
                    "status": "FAILED",
                    "result": {
                        "status_code": response.status_code,
                        "error": response.text
                    }
                }
        except Exception as e:
            return {
                "test": "models_endpoint",
                "status": "FAILED",
                "result": {"error": str(e)}
            }
    
    def test_export_functionality(self) -> Dict[str, Any]:
        """Test export functionality."""
        logger.info("📤 Testing Export Functionality...")
        
        try:
            payload = {
                "data": {
                    "sentiment": "positive",
                    "confidence": 0.85,
                    "analysis_type": "text"
                },
                "export_formats": ["json"],
                "include_metadata": True
            }
            response = requests.post(
                f"{self.api_base_url}/export/analysis-results",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "test": "export_functionality",
                    "status": "PASSED",
                    "result": {
                        "status_code": response.status_code,
                        "data": data
                    }
                }
            else:
                return {
                    "test": "export_functionality",
                    "status": "FAILED",
                    "result": {
                        "status_code": response.status_code,
                        "error": response.text
                    }
                }
        except Exception as e:
            return {
                "test": "export_functionality",
                "status": "FAILED",
                "result": {"error": str(e)}
            }
    
    def test_comprehensive_analysis(self) -> Dict[str, Any]:
        """Test comprehensive analysis endpoint."""
        logger.info("🔍 Testing Comprehensive Analysis...")
        
        try:
            payload = {
                "content_data": {
                    "text": self.test_data["business_text"],
                    "type": "business"
                },
                "analysis_type": "business",
                "include_cross_modal": True,
                "include_insights": True
            }
            response = requests.post(
                f"{self.api_base_url}/analyze/comprehensive",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "test": "comprehensive_analysis",
                    "status": "PASSED",
                    "result": {
                        "status_code": response.status_code,
                        "data": data
                    }
                }
            else:
                return {
                    "test": "comprehensive_analysis",
                    "status": "FAILED",
                    "result": {
                        "status_code": response.status_code,
                        "error": response.text
                    }
                }
        except Exception as e:
            return {
                "test": "comprehensive_analysis",
                "status": "FAILED",
                "result": {"error": str(e)}
            }
    
    def test_performance(self) -> List[Dict[str, Any]]:
        """Test API performance."""
        logger.info("⚡ Testing API Performance...")
        results = []
        
        endpoints = [
            ("/health", "GET", None),
            ("/models", "GET", None),
            ("/agents/status", "GET", None),
            ("/analyze/text", "POST", {
                "content": self.test_data["text"],
                "language": "en"
            })
        ]
        
        for endpoint, method, payload in endpoints:
            logger.info(f"  Testing performance for {endpoint}...")
            start_time = time.time()
            
            try:
                if method == "GET":
                    response = requests.get(
                        f"{self.api_base_url}{endpoint}",
                        timeout=10
                    )
                else:
                    response = requests.post(
                        f"{self.api_base_url}{endpoint}",
                        json=payload,
                        timeout=30
                    )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                results.append({
                    "test": f"performance_{endpoint.replace('/', '_')}",
                    "status": "PASSED" if response.status_code == 200 else "FAILED",
                    "result": {
                        "response_time": response_time,
                        "status_code": response.status_code,
                        "performance_rating": (
                            "excellent" if response_time < 1.0 
                            else "good" if response_time < 3.0 
                            else "needs_improvement"
                        )
                    }
                })
            except Exception as e:
                results.append({
                    "test": f"performance_{endpoint.replace('/', '_')}",
                    "status": "FAILED",
                    "result": {"error": str(e)}
                })
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all API-based tests."""
        logger.info("🚀 Starting Phase 4 API-based Validation...")
        
        all_results = []
        
        # Test core functionality
        all_results.append(self.test_api_health())
        all_results.append(self.test_text_analysis())
        all_results.append(self.test_business_intelligence())
        all_results.append(self.test_agent_status())
        all_results.append(self.test_models_endpoint())
        all_results.append(self.test_export_functionality())
        all_results.append(self.test_comprehensive_analysis())
        
        # Performance tests
        all_results.extend(self.test_performance())
        
        # Calculate statistics
        total_tests = len(all_results)
        passed_tests = len([r for r in all_results if r["status"] == "PASSED"])
        failed_tests = len([r for r in all_results if r["status"] == "FAILED"])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Update test results
        self.test_results["test_run"].update({
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate
        })
        self.test_results["results"] = all_results
        
        return self.test_results
    
    def save_results(self, results: Dict[str, Any]):
        """Save test results to file."""
        results_dir = Path(__file__).parent.parent / "Results"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phase4_api_validation_{timestamp}.json"
        filepath = results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Test results saved to {filepath}")
        return filepath


def main():
    """Main function to run Phase 4 API-based validation."""
    logger.info("🎯 Phase 4: API-based Testing & Validation")
    logger.info("=" * 60)
    
    validator = Phase4APIValidator()
    
    try:
        # Run all tests
        results = validator.run_all_tests()
        
        # Save results
        filepath = validator.save_results(results)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📊 PHASE 4 API VALIDATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {results['test_run']['total_tests']}")
        logger.info(f"Passed: {results['test_run']['passed_tests']}")
        logger.info(f"Failed: {results['test_run']['failed_tests']}")
        logger.info(f"Success Rate: {results['test_run']['success_rate']:.2f}%")
        logger.info(f"Results saved to: {filepath}")
        
        if results['test_run']['success_rate'] >= 80:
            logger.info("✅ Phase 4 API validation PASSED - Ready for Phase 5")
        else:
            logger.warning("⚠️ Phase 4 API validation needs improvement")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error during Phase 4 API validation: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Run the API-based validation
    results = main()
    
    # Exit with appropriate code
    if "error" in results:
        sys.exit(1)
    elif results['test_run']['success_rate'] >= 80:
        sys.exit(0)
    else:
        sys.exit(1)
