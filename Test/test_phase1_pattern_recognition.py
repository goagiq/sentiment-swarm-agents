"""
Test Phase 1: Pattern Recognition Foundation

This test file verifies the implementation of Phase 1 components:
- Temporal Pattern Recognition Engine
- Seasonal Pattern Detector
- Trend Analysis Engine
- Pattern Storage Service
- Pattern Recognition Agent
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from loguru import logger

# Import pattern recognition components
try:
    from src.core.pattern_recognition.temporal_analyzer import TemporalAnalyzer
    from src.core.pattern_recognition.seasonal_detector import SeasonalDetector
    from src.core.pattern_recognition.trend_engine import TrendEngine
    from src.core.pattern_recognition.pattern_storage import PatternStorage
    from src.agents.pattern_recognition_agent import PatternRecognitionAgent
    from src.config.pattern_recognition_config import get_pattern_recognition_config
    print("✅ All pattern recognition components imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class Phase1PatternRecognitionTest:
    """Test suite for Phase 1 pattern recognition components."""
    
    def __init__(self):
        self.test_results = {}
        self.config = get_pattern_recognition_config()
        
        # Initialize components
        self.temporal_analyzer = TemporalAnalyzer()
        self.seasonal_detector = SeasonalDetector()
        self.trend_engine = TrendEngine()
        self.pattern_storage = PatternStorage()
        self.pattern_agent = PatternRecognitionAgent()
        
        print("🔧 Phase 1 Pattern Recognition Test Suite Initialized")
    
    def generate_test_data(self) -> List[Dict[str, Any]]:
        """Generate test time series data."""
        data = []
        base_date = datetime.now() - timedelta(days=100)
        
        # Generate data with trend and seasonal patterns
        for i in range(100):
            date = base_date + timedelta(days=i)
            
            # Add trend component
            trend = i * 0.1
            
            # Add seasonal component (weekly pattern)
            seasonal = 5 * (i % 7 - 3) / 3
            
            # Add noise
            import random
            noise = random.uniform(-2, 2)
            
            value = 50 + trend + seasonal + noise
            
            data.append({
                "timestamp": date.isoformat(),
                "value": round(value, 2)
            })
        
        return data
    
    async def test_temporal_analyzer(self) -> bool:
        """Test temporal pattern analyzer."""
        print("\n🧪 Testing Temporal Analyzer...")
        
        try:
            # Generate test data
            test_data = self.generate_test_data()
            
            # Test temporal pattern analysis
            result = await self.temporal_analyzer.analyze_temporal_patterns(test_data)
            
            if "error" in result:
                print(f"❌ Temporal analysis failed: {result['error']}")
                return False
            
            # Verify results
            required_keys = ["trend_analysis", "seasonal_patterns", "metadata"]
            for key in required_keys:
                if key not in result:
                    print(f"❌ Missing key in temporal analysis: {key}")
                    return False
            
            print("✅ Temporal analyzer test passed")
            self.test_results["temporal_analyzer"] = "PASS"
            return True
            
        except Exception as e:
            print(f"❌ Temporal analyzer test failed: {e}")
            self.test_results["temporal_analyzer"] = f"FAIL: {e}"
            return False
    
    async def test_seasonal_detector(self) -> bool:
        """Test seasonal pattern detector."""
        print("\n🧪 Testing Seasonal Detector...")
        
        try:
            # Generate test data
            test_data = self.generate_test_data()
            
            # Test seasonal pattern detection
            result = await self.seasonal_detector.detect_seasonal_patterns(test_data)
            
            if "error" in result:
                print(f"❌ Seasonal detection failed: {result['error']}")
                return False
            
            # Verify results
            required_keys = ["autocorrelation_analysis", "fourier_analysis", "metadata"]
            for key in required_keys:
                if key not in result:
                    print(f"❌ Missing key in seasonal detection: {key}")
                    return False
            
            print("✅ Seasonal detector test passed")
            self.test_results["seasonal_detector"] = "PASS"
            return True
            
        except Exception as e:
            print(f"❌ Seasonal detector test failed: {e}")
            self.test_results["seasonal_detector"] = f"FAIL: {e}"
            return False
    
    async def test_trend_engine(self) -> bool:
        """Test trend analysis engine."""
        print("\n🧪 Testing Trend Engine...")
        
        try:
            # Generate test data
            test_data = self.generate_test_data()
            
            # Test trend analysis
            result = await self.trend_engine.analyze_trends(test_data)
            
            if "error" in result:
                print(f"❌ Trend analysis failed: {result['error']}")
                return False
            
            # Verify results
            required_keys = ["linear_trend", "trend_strength", "metadata"]
            for key in required_keys:
                if key not in result:
                    print(f"❌ Missing key in trend analysis: {key}")
                    return False
            
            print("✅ Trend engine test passed")
            self.test_results["trend_engine"] = "PASS"
            return True
            
        except Exception as e:
            print(f"❌ Trend engine test failed: {e}")
            self.test_results["trend_engine"] = f"FAIL: {e}"
            return False
    
    async def test_pattern_storage(self) -> bool:
        """Test pattern storage service."""
        print("\n🧪 Testing Pattern Storage...")
        
        try:
            # Test pattern storage
            test_pattern = {
                "pattern_type": "test",
                "data": [1, 2, 3, 4, 5],
                "metadata": {"test": True}
            }
            
            # Store pattern
            store_result = await self.pattern_storage.store_pattern(
                "test_pattern_001",
                test_pattern,
                "test"
            )
            
            if "error" in store_result:
                print(f"❌ Pattern storage failed: {store_result['error']}")
                return False
            
            # Retrieve pattern
            retrieve_result = await self.pattern_storage.get_pattern("test_pattern_001")
            
            if "error" in retrieve_result:
                print(f"❌ Pattern retrieval failed: {retrieve_result['error']}")
                return False
            
            # Search patterns
            search_result = await self.pattern_storage.search_patterns("test")
            
            if "error" in search_result:
                print(f"❌ Pattern search failed: {search_result['error']}")
                return False
            
            # Get storage summary
            summary_result = await self.pattern_storage.get_storage_summary()
            
            if "error" in summary_result:
                print(f"❌ Storage summary failed: {summary_result['error']}")
                return False
            
            print("✅ Pattern storage test passed")
            self.test_results["pattern_storage"] = "PASS"
            return True
            
        except Exception as e:
            print(f"❌ Pattern storage test failed: {e}")
            self.test_results["pattern_storage"] = f"FAIL: {e}"
            return False
    
    async def test_pattern_recognition_agent(self) -> bool:
        """Test pattern recognition agent."""
        print("\n🧪 Testing Pattern Recognition Agent...")
        
        try:
            # Generate test data
            test_data = self.generate_test_data()
            
            # Convert to JSON string for agent processing
            test_content = json.dumps(test_data)
            
            # Test agent processing
            from src.core.models import AnalysisRequest, DataType
            
            request = AnalysisRequest(
                id="test_request_001",
                data_type=DataType.TIME_SERIES,
                content=test_content,
                language="en"
            )
            
            result = await self.pattern_agent.process(request)
            
            if result.status == "failed":
                print(f"❌ Agent processing failed: {result.metadata.get('error', 'Unknown error')}")
                return False
            
            # Test individual agent methods
            temporal_result = await self.pattern_agent.analyze_temporal_patterns(test_data)
            if "error" in temporal_result:
                print(f"❌ Agent temporal analysis failed: {temporal_result['error']}")
                return False
            
            seasonal_result = await self.pattern_agent.detect_seasonal_patterns(test_data)
            if "error" in seasonal_result:
                print(f"❌ Agent seasonal detection failed: {seasonal_result['error']}")
                return False
            
            trend_result = await self.pattern_agent.analyze_trends(test_data)
            if "error" in trend_result:
                print(f"❌ Agent trend analysis failed: {trend_result['error']}")
                return False
            
            print("✅ Pattern recognition agent test passed")
            self.test_results["pattern_recognition_agent"] = "PASS"
            return True
            
        except Exception as e:
            print(f"❌ Pattern recognition agent test failed: {e}")
            self.test_results["pattern_recognition_agent"] = f"FAIL: {e}"
            return False
    
    async def test_configuration(self) -> bool:
        """Test configuration system."""
        print("\n🧪 Testing Configuration System...")
        
        try:
            # Test configuration access
            config = get_pattern_recognition_config()
            
            # Verify configuration structure
            required_configs = ["temporal", "seasonal", "trend", "storage"]
            for config_name in required_configs:
                if not hasattr(config, config_name):
                    print(f"❌ Missing configuration: {config_name}")
                    return False
            
            # Test configuration values
            if config.temporal.min_data_points < 1:
                print("❌ Invalid temporal configuration")
                return False
            
            if config.seasonal.min_periods < 1:
                print("❌ Invalid seasonal configuration")
                return False
            
            print("✅ Configuration system test passed")
            self.test_results["configuration"] = "PASS"
            return True
            
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            self.test_results["configuration"] = f"FAIL: {e}"
            return False
    
    async def test_integration(self) -> bool:
        """Test integration between components."""
        print("\n🧪 Testing Component Integration...")
        
        try:
            # Generate test data
            test_data = self.generate_test_data()
            
            # Test full pipeline
            # 1. Temporal analysis
            temporal_result = await self.temporal_analyzer.analyze_temporal_patterns(test_data)
            
            # 2. Seasonal detection
            seasonal_result = await self.seasonal_detector.detect_seasonal_patterns(test_data)
            
            # 3. Trend analysis
            trend_result = await self.trend_engine.analyze_trends(test_data)
            
            # 4. Store combined results
            combined_pattern = {
                "temporal_analysis": temporal_result,
                "seasonal_detection": seasonal_result,
                "trend_analysis": trend_result,
                "data_points": len(test_data)
            }
            
            storage_result = await self.pattern_storage.store_pattern(
                "integration_test_001",
                combined_pattern,
                "integration"
            )
            
            if "error" in storage_result:
                print(f"❌ Integration storage failed: {storage_result['error']}")
                return False
            
            print("✅ Component integration test passed")
            self.test_results["integration"] = "PASS"
            return True
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            self.test_results["integration"] = f"FAIL: {e}"
            return False
    
    async def run_all_tests(self) -> Dict[str, str]:
        """Run all Phase 1 tests."""
        print("🚀 Starting Phase 1 Pattern Recognition Tests")
        print("=" * 60)
        
        # Run individual component tests
        await self.test_temporal_analyzer()
        await self.test_seasonal_detector()
        await self.test_trend_engine()
        await self.test_pattern_storage()
        await self.test_pattern_recognition_agent()
        await self.test_configuration()
        await self.test_integration()
        
        # Generate test summary
        print("\n" + "=" * 60)
        print("📊 Phase 1 Test Results Summary")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result == "PASS")
        failed_tests = total_tests - passed_tests
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result == "PASS" else f"❌ FAIL"
            print(f"{test_name:.<30} {status}")
        
        print("-" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests == 0:
            print("\n🎉 All Phase 1 tests passed! Pattern recognition foundation is ready.")
        else:
            print(f"\n⚠️  {failed_tests} tests failed. Please review the implementation.")
        
        return self.test_results


async def main():
    """Main test execution function."""
    try:
        # Initialize test suite
        test_suite = Phase1PatternRecognitionTest()
        
        # Run all tests
        results = await test_suite.run_all_tests()
        
        # Return exit code based on results
        failed_tests = sum(1 for result in results.values() if result != "PASS")
        return 0 if failed_tests == 0 else 1
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    # Run tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
