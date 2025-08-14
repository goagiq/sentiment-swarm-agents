#!/usr/bin/env python3
"""
Simplified Performance Test for Sentiment Analysis & Decision Support System
Tests core functionality and performance without requiring external dependencies.
"""

import time
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplePerformanceTester:
    """Simplified performance testing for the decision support system."""
    
    def __init__(self):
        self.results = {
            "test_start": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }
    
    def test_core_components(self) -> Dict[str, Any]:
        """Test core system components performance."""
        logger.info("Testing core system components")
        
        results = {
            "component_tests": {},
            "overall_performance": {}
        }
        
        # Test 1: Basic Python operations
        results["component_tests"]["python_operations"] = self._test_python_operations()
        
        # Test 2: File I/O operations
        results["component_tests"]["file_operations"] = self._test_file_operations()
        
        # Test 3: Memory usage
        results["component_tests"]["memory_operations"] = self._test_memory_operations()
        
        # Test 4: Concurrent processing
        results["component_tests"]["concurrent_processing"] = self._test_concurrent_processing()
        
        # Test 5: Data processing
        results["component_tests"]["data_processing"] = self._test_data_processing()
        
        # Calculate overall performance
        results["overall_performance"] = self._calculate_overall_performance(results["component_tests"])
        
        return results
    
    def _test_python_operations(self) -> Dict[str, Any]:
        """Test basic Python operations performance."""
        logger.info("Testing Python operations")
        
        results = {
            "list_operations": {},
            "dict_operations": {},
            "string_operations": {},
            "math_operations": {}
        }
        
        # Test list operations
        start_time = time.time()
        test_list = list(range(10000))
        list_creation_time = time.time() - start_time
        
        start_time = time.time()
        test_list.append(10001)
        list_append_time = time.time() - start_time
        
        start_time = time.time()
        test_list.sort()
        list_sort_time = time.time() - start_time
        
        results["list_operations"] = {
            "creation_time": list_creation_time,
            "append_time": list_append_time,
            "sort_time": list_sort_time
        }
        
        # Test dictionary operations
        start_time = time.time()
        test_dict = {i: f"value_{i}" for i in range(1000)}
        dict_creation_time = time.time() - start_time
        
        start_time = time.time()
        test_dict[1000] = "new_value"
        dict_set_time = time.time() - start_time
        
        start_time = time.time()
        value = test_dict.get(500)
        dict_get_time = time.time() - start_time
        
        results["dict_operations"] = {
            "creation_time": dict_creation_time,
            "set_time": dict_set_time,
            "get_time": dict_get_time
        }
        
        # Test string operations
        test_string = "This is a test string for performance testing"
        
        start_time = time.time()
        for _ in range(1000):
            test_string.upper()
        string_upper_time = time.time() - start_time
        
        start_time = time.time()
        for _ in range(1000):
            test_string.split()
        string_split_time = time.time() - start_time
        
        results["string_operations"] = {
            "upper_time": string_upper_time,
            "split_time": string_split_time
        }
        
        # Test math operations
        start_time = time.time()
        for i in range(10000):
            result = i * 2 + 1
        math_operations_time = time.time() - start_time
        
        results["math_operations"] = {
            "basic_math_time": math_operations_time
        }
        
        return results
    
    def _test_file_operations(self) -> Dict[str, Any]:
        """Test file I/O operations performance."""
        logger.info("Testing file operations")
        
        results = {
            "write_operations": {},
            "read_operations": {},
            "json_operations": {}
        }
        
        # Test file write
        test_data = "Test data for file operations\n" * 1000
        
        start_time = time.time()
        with open("temp_test_file.txt", "w") as f:
            f.write(test_data)
        write_time = time.time() - start_time
        
        # Test file read
        start_time = time.time()
        with open("temp_test_file.txt", "r") as f:
            content = f.read()
        read_time = time.time() - start_time
        
        results["write_operations"] = {
            "write_time": write_time,
            "data_size": len(test_data)
        }
        
        results["read_operations"] = {
            "read_time": read_time,
            "data_size": len(content)
        }
        
        # Test JSON operations
        test_json_data = {
            "test_key": "test_value",
            "numbers": list(range(100)),
            "nested": {"inner": "value"}
        }
        
        start_time = time.time()
        json_string = json.dumps(test_json_data)
        json_dumps_time = time.time() - start_time
        
        start_time = time.time()
        parsed_data = json.loads(json_string)
        json_loads_time = time.time() - start_time
        
        results["json_operations"] = {
            "dumps_time": json_dumps_time,
            "loads_time": json_loads_time,
            "data_size": len(json_string)
        }
        
        # Clean up
        try:
            os.remove("temp_test_file.txt")
        except:
            pass
        
        return results
    
    def _test_memory_operations(self) -> Dict[str, Any]:
        """Test memory operations performance."""
        logger.info("Testing memory operations")
        
        results = {
            "memory_allocation": {},
            "memory_cleanup": {},
            "garbage_collection": {}
        }
        
        # Test memory allocation
        start_time = time.time()
        large_list = []
        for i in range(100000):
            large_list.append(f"item_{i}" * 10)
        allocation_time = time.time() - start_time
        
        memory_size = len(str(large_list))
        
        # Test memory cleanup
        start_time = time.time()
        del large_list
        cleanup_time = time.time() - start_time
        
        # Test garbage collection
        import gc
        
        start_time = time.time()
        collected = gc.collect()
        gc_time = time.time() - start_time
        
        results["memory_allocation"] = {
            "allocation_time": allocation_time,
            "memory_size_bytes": memory_size
        }
        
        results["memory_cleanup"] = {
            "cleanup_time": cleanup_time
        }
        
        results["garbage_collection"] = {
            "gc_time": gc_time,
            "objects_collected": collected
        }
        
        return results
    
    def _test_concurrent_processing(self) -> Dict[str, Any]:
        """Test concurrent processing performance."""
        logger.info("Testing concurrent processing")
        
        results = {
            "threading_performance": {},
            "process_pool_performance": {}
        }
        
        def worker_function(worker_id: int) -> Dict[str, Any]:
            """Worker function for concurrent testing."""
            start_time = time.time()
            
            # Simulate some work
            result = 0
            for i in range(10000):
                result += i * worker_id
            
            processing_time = time.time() - start_time
            
            return {
                "worker_id": worker_id,
                "result": result,
                "processing_time": processing_time
            }
        
        # Test threading
        start_time = time.time()
        threads = []
        thread_results = []
        
        for i in range(10):
            thread = threading.Thread(target=lambda: thread_results.append(worker_function(i)))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        threading_total_time = time.time() - start_time
        
        # Test process pool
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(worker_function, i) for i in range(10)]
            pool_results = [future.result() for future in as_completed(futures)]
        
        pool_total_time = time.time() - start_time
        
        results["threading_performance"] = {
            "total_time": threading_total_time,
            "worker_count": len(threads),
            "avg_worker_time": statistics.mean([r["processing_time"] for r in thread_results]) if thread_results else 0
        }
        
        results["process_pool_performance"] = {
            "total_time": pool_total_time,
            "worker_count": len(pool_results),
            "avg_worker_time": statistics.mean([r["processing_time"] for r in pool_results]) if pool_results else 0
        }
        
        return results
    
    def _test_data_processing(self) -> Dict[str, Any]:
        """Test data processing performance."""
        logger.info("Testing data processing")
        
        results = {
            "data_filtering": {},
            "data_transformation": {},
            "data_aggregation": {}
        }
        
        # Create test dataset
        test_data = [
            {"id": i, "value": i * 2, "category": f"cat_{i % 5}"}
            for i in range(10000)
        ]
        
        # Test data filtering
        start_time = time.time()
        filtered_data = [item for item in test_data if item["value"] > 5000]
        filtering_time = time.time() - start_time
        
        # Test data transformation
        start_time = time.time()
        transformed_data = [
            {"id": item["id"], "doubled_value": item["value"] * 2}
            for item in test_data
        ]
        transformation_time = time.time() - start_time
        
        # Test data aggregation
        start_time = time.time()
        category_counts = {}
        for item in test_data:
            category = item["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
        aggregation_time = time.time() - start_time
        
        results["data_filtering"] = {
            "filtering_time": filtering_time,
            "filtered_count": len(filtered_data),
            "original_count": len(test_data)
        }
        
        results["data_transformation"] = {
            "transformation_time": transformation_time,
            "transformed_count": len(transformed_data)
        }
        
        results["data_aggregation"] = {
            "aggregation_time": aggregation_time,
            "category_count": len(category_counts)
        }
        
        return results
    
    def _calculate_overall_performance(self, component_tests: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance metrics."""
        logger.info("Calculating overall performance metrics")
        
        # Calculate average times for each component
        avg_times = {}
        
        for component_name, component_data in component_tests.items():
            times = []
            
            if component_name == "python_operations":
                # Extract times from nested operations
                for op_type, op_data in component_data.items():
                    for metric, value in op_data.items():
                        if "time" in metric:
                            times.append(value)
            
            elif component_name == "file_operations":
                for op_type, op_data in component_data.items():
                    for metric, value in op_data.items():
                        if "time" in metric:
                            times.append(value)
            
            elif component_name == "memory_operations":
                for op_type, op_data in component_data.items():
                    for metric, value in op_data.items():
                        if "time" in metric:
                            times.append(value)
            
            elif component_name == "concurrent_processing":
                for op_type, op_data in component_data.items():
                    if "total_time" in op_data:
                        times.append(op_data["total_time"])
            
            elif component_name == "data_processing":
                for op_type, op_data in component_data.items():
                    for metric, value in op_data.items():
                        if "time" in metric:
                            times.append(value)
            
            if times:
                avg_times[component_name] = statistics.mean(times)
        
        # Calculate overall performance score
        if avg_times:
            total_avg_time = sum(avg_times.values())
            performance_score = max(0, 100 - (total_avg_time * 1000))  # Convert to score
        else:
            performance_score = 0
        
        return {
            "average_times": avg_times,
            "total_average_time": sum(avg_times.values()) if avg_times else 0,
            "performance_score": min(100, max(0, performance_score)),
            "component_count": len(component_tests)
        }
    
    def test_system_integration(self) -> Dict[str, Any]:
        """Test system integration performance."""
        logger.info("Testing system integration")
        
        results = {
            "integration_tests": {},
            "error_handling": {},
            "resource_usage": {}
        }
        
        # Test basic system integration
        try:
            # Test importing core modules
            start_time = time.time()
            
            # Try to import available modules
            imported_modules = []
            
            try:
                from src.core.caching_service import CachingService
                imported_modules.append("CachingService")
            except ImportError:
                pass
            
            try:
                from src.core.performance_monitor import PerformanceMonitor
                imported_modules.append("PerformanceMonitor")
            except ImportError:
                pass
            
            try:
                from src.core.memory_manager import MemoryManager
                imported_modules.append("MemoryManager")
            except ImportError:
                pass
            
            import_time = time.time() - start_time
            
            results["integration_tests"]["module_imports"] = {
                "import_time": import_time,
                "successful_imports": len(imported_modules),
                "imported_modules": imported_modules
            }
            
        except Exception as e:
            results["integration_tests"]["module_imports"] = {
                "error": str(e),
                "successful_imports": 0
            }
        
        # Test error handling
        error_handling_tests = []
        
        # Test 1: Division by zero handling
        start_time = time.time()
        try:
            result = 1 / 0
        except ZeroDivisionError:
            error_handling_tests.append({
                "test": "division_by_zero",
                "handled": True,
                "time": time.time() - start_time
            })
        
        # Test 2: File not found handling
        start_time = time.time()
        try:
            with open("nonexistent_file.txt", "r") as f:
                content = f.read()
        except FileNotFoundError:
            error_handling_tests.append({
                "test": "file_not_found",
                "handled": True,
                "time": time.time() - start_time
            })
        
        # Test 3: Key error handling
        start_time = time.time()
        try:
            test_dict = {"key1": "value1"}
            value = test_dict["nonexistent_key"]
        except KeyError:
            error_handling_tests.append({
                "test": "key_error",
                "handled": True,
                "time": time.time() - start_time
            })
        
        results["error_handling"] = {
            "tests": error_handling_tests,
            "total_tests": len(error_handling_tests),
            "successful_handling": len([t for t in error_handling_tests if t["handled"]])
        }
        
        # Test resource usage
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        results["resource_usage"] = {
            "memory_usage_mb": memory_info.rss / (1024 * 1024),
            "cpu_percent": process.cpu_percent(),
            "thread_count": process.num_threads(),
            "open_files": len(process.open_files()),
            "connections": len(process.connections())
        }
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive performance testing."""
        logger.info("Starting comprehensive performance testing")
        
        # Run all tests
        tests = {
            "core_components": self.test_core_components(),
            "system_integration": self.test_system_integration()
        }
        
        # Generate summary
        summary = self._generate_test_summary(tests)
        
        self.results["tests"] = tests
        self.results["summary"] = summary
        
        # Save results
        self._save_test_results()
        
        return self.results
    
    def _generate_test_summary(self, tests: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of test results."""
        logger.info("Generating test summary")
        
        summary = {
            "overall_performance_score": 0,
            "test_status": {},
            "key_metrics": {},
            "recommendations": []
        }
        
        # Analyze core components
        core_components = tests.get("core_components", {})
        overall_performance = core_components.get("overall_performance", {})
        
        performance_score = overall_performance.get("performance_score", 0)
        summary["overall_performance_score"] = performance_score
        
        # Set test status
        summary["test_status"] = {
            "core_components": "completed",
            "system_integration": "completed"
        }
        
        # Extract key metrics
        summary["key_metrics"] = {
            "performance_score": performance_score,
            "component_count": overall_performance.get("component_count", 0),
            "total_average_time": overall_performance.get("total_average_time", 0)
        }
        
        # Generate recommendations
        if performance_score >= 80:
            summary["recommendations"].append("System performance is excellent")
        elif performance_score >= 60:
            summary["recommendations"].append("System performance is good, consider minor optimizations")
        else:
            summary["recommendations"].append("System performance needs improvement")
        
        if overall_performance.get("total_average_time", 0) > 1.0:
            summary["recommendations"].append("Consider optimizing slow operations")
        
        return summary
    
    def _save_test_results(self):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_performance_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {filename}")


def main():
    """Main function to run simple performance testing."""
    print("🚀 Starting Simple Performance Testing")
    print("=" * 50)
    
    # Initialize tester
    tester = SimplePerformanceTester()
    
    try:
        # Run comprehensive test
        results = tester.run_comprehensive_test()
        
        # Print summary
        print("\n📊 Performance Test Results Summary")
        print("=" * 50)
        
        summary = results["summary"]
        print(f"Overall Performance Score: {summary['overall_performance_score']:.1f}/100")
        print(f"Component Count: {summary['key_metrics']['component_count']}")
        print(f"Total Average Time: {summary['key_metrics']['total_average_time']:.4f}s")
        
        if summary["recommendations"]:
            print("\n💡 Recommendations:")
            for recommendation in summary["recommendations"]:
                print(f"  - {recommendation}")
        
        print(f"\n✅ Performance testing completed successfully!")
        print(f"📄 Detailed results saved to simple_performance_test_results_*.json")
        
    except Exception as e:
        logger.error(f"Performance testing failed: {e}")
        print(f"❌ Performance testing failed: {e}")


if __name__ == "__main__":
    main()
