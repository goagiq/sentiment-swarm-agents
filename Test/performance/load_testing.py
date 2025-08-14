#!/usr/bin/env python3
"""
Load Testing Script for Sentiment Analysis & Decision Support System
Tests system performance under high load conditions and identifies bottlenecks.
"""

import asyncio
import time
import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any
import aiohttp
import logging
from datetime import datetime
import os
import sys

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from src.core.performance_monitor import PerformanceMonitor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoadTester:
    """Comprehensive load testing for the decision support system."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {
            "test_start": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }
        self.performance_monitor = PerformanceMonitor()
        
    async def test_api_endpoints(
        self, concurrent_users: int = 10, duration: int = 60
    ) -> Dict[str, Any]:
        """Test API endpoints under load."""
        logger.info(
            f"Starting API load test with {concurrent_users} concurrent users "
            f"for {duration} seconds"
        )
        
        endpoints = [
            "/health",
            "/api/v1/analyze",
            "/api/v1/decision-support",
            "/api/v1/knowledge-graph",
            "/api/v1/scenario-analysis"
        ]
        
        test_results = {
            "endpoints": {},
            "concurrent_users": concurrent_users,
            "duration": duration,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": []
        }
        
        async def make_request(session: aiohttp.ClientSession, endpoint: str) -> Dict[str, Any]:
            """Make a single request to an endpoint."""
            start_time = time.time()
            try:
                async with session.get(f"{self.base_url}{endpoint}") as response:
                    response_time = time.time() - start_time
                    return {
                        "endpoint": endpoint,
                        "status": response.status,
                        "response_time": response_time,
                        "success": response.status == 200
                    }
            except Exception as e:
                response_time = time.time() - start_time
                return {
                    "endpoint": endpoint,
                    "status": 0,
                    "response_time": response_time,
                    "success": False,
                    "error": str(e)
                }
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            start_time = time.time()
            
            while time.time() - start_time < duration:
                # Create concurrent requests
                for _ in range(concurrent_users):
                    for endpoint in endpoints:
                        task = asyncio.create_task(make_request(session, endpoint))
                        tasks.append(task)
                
                # Wait a bit before next batch
                await asyncio.sleep(0.1)
            
            # Wait for all tasks to complete
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for response in responses:
                if isinstance(response, dict):
                    test_results["total_requests"] += 1
                    if response["success"]:
                        test_results["successful_requests"] += 1
                    else:
                        test_results["failed_requests"] += 1
                    
                    test_results["response_times"].append(response["response_time"])
                    
                    endpoint = response["endpoint"]
                    if endpoint not in test_results["endpoints"]:
                        test_results["endpoints"][endpoint] = {
                            "requests": 0,
                            "successful": 0,
                            "failed": 0,
                            "response_times": []
                        }
                    
                    test_results["endpoints"][endpoint]["requests"] += 1
                    test_results["endpoints"][endpoint]["response_times"].append(response["response_time"])
                    
                    if response["success"]:
                        test_results["endpoints"][endpoint]["successful"] += 1
                    else:
                        test_results["endpoints"][endpoint]["failed"] += 1
        
        # Calculate statistics
        if test_results["response_times"]:
            test_results["avg_response_time"] = statistics.mean(test_results["response_times"])
            test_results["median_response_time"] = statistics.median(test_results["response_times"])
            test_results["p95_response_time"] = sorted(test_results["response_times"])[int(len(test_results["response_times"]) * 0.95)]
            test_results["p99_response_time"] = sorted(test_results["response_times"])[int(len(test_results["response_times"]) * 0.99)]
            test_results["min_response_time"] = min(test_results["response_times"])
            test_results["max_response_time"] = max(test_results["response_times"])
        
        test_results["requests_per_second"] = test_results["total_requests"] / duration
        test_results["success_rate"] = (test_results["successful_requests"] / test_results["total_requests"]) * 100 if test_results["total_requests"] > 0 else 0
        
        return test_results
    
    def test_database_performance(self, num_queries: int = 1000) -> Dict[str, Any]:
        """Test database performance under load."""
        logger.info(f"Starting database performance test with {num_queries} queries")
        
        from src.core.storage.database_manager import DatabaseManager
        
        db_manager = DatabaseManager()
        test_results = {
            "total_queries": num_queries,
            "query_times": [],
            "successful_queries": 0,
            "failed_queries": 0
        }
        
        # Test different types of queries
        queries = [
            "SELECT COUNT(*) FROM knowledge_graph_entities",
            "SELECT * FROM knowledge_graph_entities LIMIT 100",
            "SELECT * FROM decision_patterns LIMIT 50",
            "SELECT * FROM scenario_analysis LIMIT 25"
        ]
        
        for i in range(num_queries):
            query = queries[i % len(queries)]
            start_time = time.time()
            
            try:
                result = db_manager.execute_query(query)
                query_time = time.time() - start_time
                test_results["query_times"].append(query_time)
                test_results["successful_queries"] += 1
            except Exception as e:
                query_time = time.time() - start_time
                test_results["query_times"].append(query_time)
                test_results["failed_queries"] += 1
                logger.error(f"Query failed: {e}")
        
        # Calculate statistics
        if test_results["query_times"]:
            test_results["avg_query_time"] = statistics.mean(test_results["query_times"])
            test_results["median_query_time"] = statistics.median(test_results["query_times"])
            test_results["p95_query_time"] = sorted(test_results["query_times"])[int(len(test_results["query_times"]) * 0.95)]
            test_results["min_query_time"] = min(test_results["query_times"])
            test_results["max_query_time"] = max(test_results["query_times"])
        
        test_results["queries_per_second"] = test_results["successful_queries"] / sum(test_results["query_times"]) if test_results["query_times"] else 0
        test_results["success_rate"] = (test_results["successful_queries"] / test_results["total_queries"]) * 100
        
        return test_results
    
    def test_memory_usage(self, duration: int = 300) -> Dict[str, Any]:
        """Test memory usage over time."""
        logger.info(f"Starting memory usage test for {duration} seconds")
        
        import psutil
        import threading
        
        memory_data = []
        stop_monitoring = threading.Event()
        
        def monitor_memory():
            process = psutil.Process()
            while not stop_monitoring.is_set():
                memory_info = process.memory_info()
                memory_data.append({
                    "timestamp": time.time(),
                    "rss": memory_info.rss,  # Resident Set Size
                    "vms": memory_info.vms,  # Virtual Memory Size
                    "percent": process.memory_percent()
                })
                time.sleep(1)
        
        # Start memory monitoring
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.start()
        
        # Simulate load during monitoring
        self._simulate_load_during_monitoring(duration)
        
        # Stop monitoring
        stop_monitoring.set()
        monitor_thread.join()
        
        # Calculate statistics
        if memory_data:
            rss_values = [entry["rss"] for entry in memory_data]
            vms_values = [entry["vms"] for entry in memory_data]
            percent_values = [entry["percent"] for entry in memory_data]
            
            test_results = {
                "duration": duration,
                "samples": len(memory_data),
                "rss": {
                    "avg": statistics.mean(rss_values),
                    "max": max(rss_values),
                    "min": min(rss_values),
                    "trend": "increasing" if rss_values[-1] > rss_values[0] else "decreasing"
                },
                "vms": {
                    "avg": statistics.mean(vms_values),
                    "max": max(vms_values),
                    "min": min(vms_values)
                },
                "percent": {
                    "avg": statistics.mean(percent_values),
                    "max": max(percent_values),
                    "min": min(percent_values)
                }
            }
        else:
            test_results = {"error": "No memory data collected"}
        
        return test_results
    
    def _simulate_load_during_monitoring(self, duration: int):
        """Simulate system load during memory monitoring."""
        from src.agents.decision_support_agent import DecisionSupportAgent
        
        agent = DecisionSupportAgent()
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Simulate decision support operations
            try:
                # This is a lightweight operation for testing
                agent._validate_input("Test decision scenario")
            except:
                pass
            time.sleep(0.1)
    
    def test_concurrent_processing(self, num_concurrent: int = 20) -> Dict[str, Any]:
        """Test concurrent processing capabilities."""
        logger.info(f"Starting concurrent processing test with {num_concurrent} concurrent operations")
        
        from src.agents.decision_support_agent import DecisionSupportAgent
        
        test_results = {
            "concurrent_operations": num_concurrent,
            "operation_times": [],
            "successful_operations": 0,
            "failed_operations": 0
        }
        
        def process_decision_scenario(scenario_id: int) -> Dict[str, Any]:
            """Process a single decision scenario."""
            agent = DecisionSupportAgent()
            start_time = time.time()
            
            try:
                # Simulate decision processing
                result = agent._validate_input(f"Test scenario {scenario_id}")
                processing_time = time.time() - start_time
                return {
                    "scenario_id": scenario_id,
                    "success": True,
                    "processing_time": processing_time
                }
            except Exception as e:
                processing_time = time.time() - start_time
                return {
                    "scenario_id": scenario_id,
                    "success": False,
                    "processing_time": processing_time,
                    "error": str(e)
                }
        
        # Execute concurrent operations
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(process_decision_scenario, i) for i in range(num_concurrent)]
            
            for future in as_completed(futures):
                result = future.result()
                test_results["operation_times"].append(result["processing_time"])
                
                if result["success"]:
                    test_results["successful_operations"] += 1
                else:
                    test_results["failed_operations"] += 1
        
        # Calculate statistics
        if test_results["operation_times"]:
            test_results["avg_processing_time"] = statistics.mean(test_results["operation_times"])
            test_results["median_processing_time"] = statistics.median(test_results["operation_times"])
            test_results["p95_processing_time"] = sorted(test_results["operation_times"])[int(len(test_results["operation_times"]) * 0.95)]
            test_results["min_processing_time"] = min(test_results["operation_times"])
            test_results["max_processing_time"] = max(test_results["operation_times"])
        
        test_results["operations_per_second"] = test_results["successful_operations"] / sum(test_results["operation_times"]) if test_results["operation_times"] else 0
        test_results["success_rate"] = (test_results["successful_operations"] / num_concurrent) * 100
        
        return test_results
    
    def test_real_time_processing(self, duration: int = 60) -> Dict[str, Any]:
        """Test real-time processing capabilities."""
        logger.info(f"Starting real-time processing test for {duration} seconds")
        
        from src.core.real_time.pattern_monitor import PatternMonitor
        
        monitor = PatternMonitor()
        test_results = {
            "duration": duration,
            "events_processed": 0,
            "processing_times": [],
            "latency": []
        }
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            event_start = time.time()
            
            try:
                # Simulate real-time event processing
                monitor.process_event({
                    "type": "decision_event",
                    "timestamp": time.time(),
                    "data": {"test": "real_time_processing"}
                })
                
                processing_time = time.time() - event_start
                test_results["processing_times"].append(processing_time)
                test_results["events_processed"] += 1
                
                # Calculate latency
                latency = time.time() - event_start
                test_results["latency"].append(latency)
                
            except Exception as e:
                logger.error(f"Real-time processing error: {e}")
            
            # Simulate event frequency
            time.sleep(0.1)
        
        # Calculate statistics
        if test_results["processing_times"]:
            test_results["avg_processing_time"] = statistics.mean(test_results["processing_times"])
            test_results["avg_latency"] = statistics.mean(test_results["latency"])
            test_results["events_per_second"] = test_results["events_processed"] / duration
            test_results["p95_latency"] = sorted(test_results["latency"])[int(len(test_results["latency"]) * 0.95)]
        
        return test_results
    
    def run_comprehensive_load_test(self) -> Dict[str, Any]:
        """Run all load tests and generate comprehensive report."""
        logger.info("Starting comprehensive load testing")
        
        # Run all tests
        tests = {
            "api_endpoints": asyncio.run(self.test_api_endpoints(concurrent_users=20, duration=120)),
            "database_performance": self.test_database_performance(num_queries=500),
            "memory_usage": self.test_memory_usage(duration=60),
            "concurrent_processing": self.test_concurrent_processing(num_concurrent=30),
            "real_time_processing": self.test_real_time_processing(duration=60)
        }
        
        # Generate summary
        summary = self._generate_summary(tests)
        
        self.results["tests"] = tests
        self.results["summary"] = summary
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _generate_summary(self, tests: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of all test results."""
        summary = {
            "overall_status": "PASS",
            "performance_score": 0,
            "bottlenecks": [],
            "recommendations": []
        }
        
        # Analyze API performance
        api_test = tests.get("api_endpoints", {})
        if api_test.get("avg_response_time", 0) > 1.0:
            summary["bottlenecks"].append("API response time too high")
            summary["overall_status"] = "NEEDS_OPTIMIZATION"
        
        if api_test.get("success_rate", 100) < 95:
            summary["bottlenecks"].append("API success rate below 95%")
            summary["overall_status"] = "NEEDS_OPTIMIZATION"
        
        # Analyze database performance
        db_test = tests.get("database_performance", {})
        if db_test.get("avg_query_time", 0) > 0.1:
            summary["bottlenecks"].append("Database query time too high")
            summary["recommendations"].append("Optimize database queries and indexes")
        
        # Analyze memory usage
        memory_test = tests.get("memory_usage", {})
        if memory_test.get("rss", {}).get("trend") == "increasing":
            summary["bottlenecks"].append("Memory usage increasing over time")
            summary["recommendations"].append("Check for memory leaks")
        
        # Analyze concurrent processing
        concurrent_test = tests.get("concurrent_processing", {})
        if concurrent_test.get("success_rate", 100) < 90:
            summary["bottlenecks"].append("Concurrent processing success rate low")
            summary["recommendations"].append("Review thread safety and resource management")
        
        # Calculate performance score
        score = 0
        if api_test.get("success_rate", 0) >= 95:
            score += 25
        if db_test.get("success_rate", 0) >= 95:
            score += 25
        if memory_test.get("rss", {}).get("trend") != "increasing":
            score += 25
        if concurrent_test.get("success_rate", 0) >= 90:
            score += 25
        
        summary["performance_score"] = score
        
        return summary
    
    def _save_results(self):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"load_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Load test results saved to {filename}")

def main():
    """Main function to run load testing."""
    print("🚀 Starting Comprehensive Load Testing")
    print("=" * 50)
    
    # Initialize load tester
    load_tester = LoadTester()
    
    try:
        # Run comprehensive load test
        results = load_tester.run_comprehensive_load_test()
        
        # Print summary
        print("\n📊 Load Testing Results Summary")
        print("=" * 50)
        
        summary = results["summary"]
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Performance Score: {summary['performance_score']}/100")
        
        if summary["bottlenecks"]:
            print("\n🚨 Identified Bottlenecks:")
            for bottleneck in summary["bottlenecks"]:
                print(f"  - {bottleneck}")
        
        if summary["recommendations"]:
            print("\n💡 Recommendations:")
            for recommendation in summary["recommendations"]:
                print(f"  - {recommendation}")
        
        # Print detailed results
        print("\n📈 Detailed Results:")
        for test_name, test_results in results["tests"].items():
            print(f"\n{test_name.upper()}:")
            if "avg_response_time" in test_results:
                print(f"  Avg Response Time: {test_results['avg_response_time']:.3f}s")
            if "success_rate" in test_results:
                print(f"  Success Rate: {test_results['success_rate']:.1f}%")
            if "avg_processing_time" in test_results:
                print(f"  Avg Processing Time: {test_results['avg_processing_time']:.3f}s")
        
        print(f"\n✅ Load testing completed successfully!")
        print(f"📄 Detailed results saved to load_test_results_*.json")
        
    except Exception as e:
        logger.error(f"Load testing failed: {e}")
        print(f"❌ Load testing failed: {e}")

if __name__ == "__main__":
    main()
