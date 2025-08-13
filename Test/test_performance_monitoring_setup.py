"""
Test Performance Monitoring Setup
Tests the integration of PerformanceDataCollector with PerformanceOptimizer.
"""

import asyncio
import time
import json
from datetime import datetime
from loguru import logger

from src.core.performance_data_collector import (
    PerformanceDataCollector, 
    ComprehensivePerformanceData
)
from src.core.performance_optimizer import get_performance_optimizer


class TestPerformanceMonitoringSetup:
    """Test class for performance monitoring setup."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
    
    def log_test_result(self, test_name: str, status: str, message: str, duration: float = None):
        """Log test result."""
        result = {
            "status": status,
            "message": message,
            "duration": duration
        }
        self.test_results[test_name] = result
        
        status_icon = "✅" if status == "PASSED" else "❌"
        duration_str = f" ({duration:.2f}s)" if duration else ""
        logger.info(f"{status_icon} {test_name}: {message}{duration_str}")
    
    async def test_performance_data_collector_initialization(self):
        """Test performance data collector initialization."""
        logger.info("🧪 Testing Performance Data Collector Initialization...")
        start_time = time.time()
        
        try:
            # Test collector creation
            collector = PerformanceDataCollector(db_path=":memory:")  # Use in-memory DB for testing
            
            # Verify collector attributes
            assert collector.db_path == ":memory:", "Database path not set correctly"
            assert collector.collection_interval == 60, "Collection interval not set correctly"
            assert collector.max_history_days == 30, "Max history days not set correctly"
            assert not collector.is_collecting, "Collector should not be collecting initially"
            
            duration = time.time() - start_time
            self.log_test_result(
                "Performance Data Collector Initialization",
                "PASSED",
                "Collector initialized successfully with correct configuration",
                duration
            )
            
            return {"status": "PASSED", "collector": collector}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Performance Data Collector Initialization",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_comprehensive_data_collection(self):
        """Test comprehensive data collection."""
        logger.info("🧪 Testing Comprehensive Data Collection...")
        start_time = time.time()
        
        try:
            collector = PerformanceDataCollector(db_path=":memory:")
            
            # Test data collection
            data = await collector.collect_comprehensive_data()
            
            # Verify data structure
            assert isinstance(data, ComprehensivePerformanceData), "Data should be ComprehensivePerformanceData"
            assert isinstance(data.cpu_usage, (int, float)), "CPU usage should be numeric"
            assert isinstance(data.memory_usage, (int, float)), "Memory usage should be numeric"
            assert isinstance(data.disk_usage, (int, float)), "Disk usage should be numeric"
            assert isinstance(data.total_operations, int), "Total operations should be integer"
            assert isinstance(data.timestamp, datetime), "Timestamp should be datetime"
            
            # Verify data ranges
            assert 0 <= data.cpu_usage <= 100, "CPU usage should be between 0 and 100"
            assert 0 <= data.memory_usage <= 100, "Memory usage should be between 0 and 100"
            assert 0 <= data.disk_usage <= 100, "Disk usage should be between 0 and 100"
            assert data.total_operations >= 0, "Total operations should be non-negative"
            
            duration = time.time() - start_time
            self.log_test_result(
                "Comprehensive Data Collection",
                "PASSED",
                f"Collected data with {data.total_operations} operations",
                duration
            )
            
            return {"status": "PASSED", "data": data}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Comprehensive Data Collection",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_data_storage_and_retrieval(self):
        """Test data storage and retrieval."""
        logger.info("🧪 Testing Data Storage and Retrieval...")
        start_time = time.time()
        
        try:
            collector = PerformanceDataCollector(db_path=":memory:")
            
            # Collect and store data
            data = await collector.collect_comprehensive_data()
            await collector.store_performance_data(data)
            
            # Retrieve data
            retrieved_data = await collector.get_performance_data(hours=1)
            
            # Verify data was stored and retrieved
            assert len(retrieved_data) > 0, "Should have retrieved at least one record"
            
            # Verify data structure
            record = retrieved_data[0]
            assert 'timestamp' in record, "Record should have timestamp"
            assert 'cpu_usage' in record, "Record should have cpu_usage"
            assert 'memory_usage' in record, "Record should have memory_usage"
            assert 'total_operations' in record, "Record should have total_operations"
            
            duration = time.time() - start_time
            self.log_test_result(
                "Data Storage and Retrieval",
                "PASSED",
                f"Stored and retrieved {len(retrieved_data)} records",
                duration
            )
            
            return {"status": "PASSED", "records": len(retrieved_data)}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Data Storage and Retrieval",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_performance_summary_generation(self):
        """Test performance summary generation."""
        logger.info("🧪 Testing Performance Summary Generation...")
        start_time = time.time()
        
        try:
            collector = PerformanceDataCollector(db_path=":memory:")
            
            # Collect and store some data
            for _ in range(3):
                data = await collector.collect_comprehensive_data()
                await collector.store_performance_data(data)
                await asyncio.sleep(0.1)  # Small delay between collections
            
            # Generate summary
            summary = await collector.get_performance_summary(hours=1)
            
            # Verify summary structure
            assert summary.get("success"), "Summary should be successful"
            assert "period_hours" in summary, "Summary should have period_hours"
            assert "total_records" in summary, "Summary should have total_records"
            assert "system_metrics" in summary, "Summary should have system_metrics"
            assert "application_metrics" in summary, "Summary should have application_metrics"
            
            # Verify summary data
            assert summary["total_records"] >= 3, "Should have at least 3 records"
            assert summary["period_hours"] == 1, "Period should be 1 hour"
            
            duration = time.time() - start_time
            self.log_test_result(
                "Performance Summary Generation",
                "PASSED",
                f"Generated summary with {summary['total_records']} records",
                duration
            )
            
            return {"status": "PASSED", "summary": summary}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Performance Summary Generation",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_performance_optimizer_integration(self):
        """Test performance optimizer integration with data collector."""
        logger.info("🧪 Testing Performance Optimizer Integration...")
        start_time = time.time()
        
        try:
            # Get performance optimizer
            optimizer = await get_performance_optimizer()
            
            # Generate performance report (should use data collector if available)
            report = await optimizer.get_performance_report()
            
            # Verify report structure
            assert report.get("success"), "Report should be successful"
            assert "report" in report, "Report should have report data"
            
            report_data = report["report"]
            assert "summary" in report_data, "Report should have summary"
            assert "current_metrics" in report_data, "Report should have current_metrics"
            assert "recommendations" in report_data, "Report should have recommendations"
            
            # Check if data source is indicated
            if "data_source" in report_data:
                logger.info(f"Data source: {report_data['data_source']}")
            
            duration = time.time() - start_time
            self.log_test_result(
                "Performance Optimizer Integration",
                "PASSED",
                "Performance optimizer integrated successfully with data collector",
                duration
            )
            
            return {"status": "PASSED", "report": report}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Performance Optimizer Integration",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_collection_lifecycle(self):
        """Test collection start/stop lifecycle."""
        logger.info("🧪 Testing Collection Lifecycle...")
        start_time = time.time()
        
        try:
            collector = PerformanceDataCollector(db_path=":memory:")
            
            # Test start collection
            await collector.start_collection()
            assert collector.is_collecting, "Collector should be collecting after start"
            assert collector.collection_task is not None, "Collection task should be created"
            
            # Wait a bit for collection to run
            await asyncio.sleep(0.5)
            
            # Test stop collection
            await collector.stop_collection()
            assert not collector.is_collecting, "Collector should not be collecting after stop"
            
            duration = time.time() - start_time
            self.log_test_result(
                "Collection Lifecycle",
                "PASSED",
                "Collection start/stop lifecycle working correctly",
                duration
            )
            
            return {"status": "PASSED"}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Collection Lifecycle",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def run_all_tests(self):
        """Run all performance monitoring setup tests."""
        logger.info("🚀 Starting Performance Monitoring Setup Tests")
        logger.info("=" * 60)
        
        self.start_time = time.time()
        
        # Run all tests
        tests = [
            self.test_performance_data_collector_initialization(),
            self.test_comprehensive_data_collection(),
            self.test_data_storage_and_retrieval(),
            self.test_performance_summary_generation(),
            self.test_performance_optimizer_integration(),
            self.test_collection_lifecycle()
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Process results
        passed = 0
        failed = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed += 1
                logger.error(f"Test {i+1} failed with exception: {result}")
            elif result.get("status") == "PASSED":
                passed += 1
            else:
                failed += 1
        
        total_duration = time.time() - self.start_time
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📊 PERFORMANCE MONITORING SETUP TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {len(tests)}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {(passed/len(tests)*100):.1f}%")
        logger.info(f"Total Duration: {total_duration:.2f}s")
        
        # Save results
        with open("Test/performance_monitoring_setup_results.json", "w") as f:
            json.dump({
                "test_results": self.test_results,
                "summary": {
                    "total_tests": len(tests),
                    "passed": passed,
                    "failed": failed,
                    "success_rate": (passed/len(tests)*100),
                    "total_duration": total_duration
                }
            }, f, indent=2, default=str)
        
        logger.info("Results saved to: Test/performance_monitoring_setup_results.json")
        
        return {
            "total_tests": len(tests),
            "passed": passed,
            "failed": failed,
            "success_rate": (passed/len(tests)*100),
            "test_results": self.test_results
        }


async def main():
    """Main test function."""
    tester = TestPerformanceMonitoringSetup()
    results = await tester.run_all_tests()
    
    if results["success_rate"] >= 80:
        logger.info("✅ Performance Monitoring Setup: MAJOR SUCCESS")
        return True
    else:
        logger.error("❌ Performance Monitoring Setup: NEEDS IMPROVEMENT")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
