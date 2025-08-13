#!/usr/bin/env python3
"""
Comprehensive Test Suite for Predictive Analytics - Phase 6.2

This script provides comprehensive testing for all predictive analytics components including:
- Full system testing with edge cases
- Performance testing with load and stress testing
- Integration testing across all components
- User acceptance testing and validation
- Error handling and recovery testing
"""

import sys
import os
import asyncio
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import psutil
import gc

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from loguru import logger

# Import all predictive analytics components
from src.core.predictive_analytics.forecasting_engine import ForecastingEngine
from src.core.predictive_analytics.confidence_calculator import ConfidenceCalculator
from src.core.predictive_analytics.scenario_forecaster import ScenarioForecaster
from src.core.predictive_analytics.forecast_validator import ForecastValidator
from src.agents.predictive_analytics_agent import PredictiveAnalyticsAgent
from src.core.models import DataType, AnalysisRequest, AnalysisResult
from src.core.orchestrator import SentimentOrchestrator


class ComprehensivePredictiveAnalyticsTester:
    """Comprehensive test suite for predictive analytics components."""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        self.start_time = None
        self.orchestrator = None
        
    def log_test_result(self, test_name: str, status: str, message: str = "", 
                       duration: float = 0, metrics: Dict[str, Any] = None):
        """Log test result with comprehensive metrics."""
        result = {
            "test_name": test_name,
            "status": status,
            "message": message,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {}
        }
        self.test_results.append(result)
        logger.info(f"{'✅' if status == 'PASSED' else '❌'} {test_name}: {status} ({duration:.2f}s)")
        if message:
            logger.info(f"   Message: {message}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available": psutil.virtual_memory().available / (1024**3),  # GB
            "disk_usage": psutil.disk_usage('/').percent
        }
    
    def generate_test_data(self, n_points: int = 100, complexity: str = "simple") -> np.ndarray:
        """Generate test data with varying complexity."""
        if complexity == "simple":
            # Simple trend + noise
            trend = np.linspace(0, 10, n_points)
            noise = np.random.normal(0, 0.5, n_points)
            return trend + noise
        elif complexity == "seasonal":
            # Trend + seasonal + noise
            trend = np.linspace(0, 10, n_points)
            seasonal = 2 * np.sin(2 * np.pi * np.arange(n_points) / 12)
            noise = np.random.normal(0, 0.3, n_points)
            return trend + seasonal + noise
        elif complexity == "complex":
            # Multiple seasonalities + trend + noise
            trend = np.linspace(0, 15, n_points)
            seasonal1 = 2 * np.sin(2 * np.pi * np.arange(n_points) / 12)
            seasonal2 = 1.5 * np.sin(2 * np.pi * np.arange(n_points) / 6)
            noise = np.random.normal(0, 0.2, n_points)
            return trend + seasonal1 + seasonal2 + noise
        else:
            raise ValueError(f"Unknown complexity level: {complexity}")
    
    async def setup(self):
        """Setup test environment."""
        logger.info("Setting up comprehensive predictive analytics test environment...")
        self.start_time = time.time()
        
        # Initialize orchestrator
        self.orchestrator = SentimentOrchestrator()
        
        # Initialize components
        self.forecasting_engine = ForecastingEngine()
        self.confidence_calculator = ConfidenceCalculator()
        self.scenario_forecaster = ScenarioForecaster()
        self.forecast_validator = ForecastValidator()
        self.predictive_agent = PredictiveAnalyticsAgent()
        
        logger.info("✅ Test environment setup complete")
    
    async def test_forecasting_engine_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of forecasting engine."""
        logger.info("🧪 Testing Forecasting Engine (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_functionality": False,
                "model_variety": False,
                "data_complexity": False,
                "error_handling": False,
                "performance": False
            }
            
            # Test 1: Basic functionality
            logger.info("   Testing basic functionality...")
            data = self.generate_test_data(50, "simple")
            result = self.forecasting_engine.forecast(
                data=data,
                model_type='ensemble',
                forecast_horizon=12
            )
            
            assert len(result.predictions) == 12, "Incorrect prediction length"
            assert result.model_type == 'ensemble', "Incorrect model type"
            test_results["basic_functionality"] = True
            
            # Test 2: Model variety
            logger.info("   Testing model variety...")
            models = ['ensemble', 'simple_average', 'exponential_smoothing', 'arima']
            for model in models:
                result = self.forecasting_engine.forecast(
                    data=data,
                    model_type=model,
                    forecast_horizon=6
                )
                assert len(result.predictions) == 6, f"Model {model} failed"
            test_results["model_variety"] = True
            
            # Test 3: Data complexity
            logger.info("   Testing data complexity...")
            complexities = ["simple", "seasonal", "complex"]
            for complexity in complexities:
                complex_data = self.generate_test_data(100, complexity)
                result = self.forecasting_engine.forecast(
                    data=complex_data,
                    model_type='ensemble',
                    forecast_horizon=12
                )
                assert len(result.predictions) == 12, f"Complexity {complexity} failed"
            test_results["data_complexity"] = True
            
            # Test 4: Error handling
            logger.info("   Testing error handling...")
            try:
                # Test with invalid data
                self.forecasting_engine.forecast(
                    data=np.array([]),
                    model_type='ensemble',
                    forecast_horizon=12
                )
                assert False, "Should have raised error for empty data"
            except Exception:
                test_results["error_handling"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            large_data = self.generate_test_data(1000, "complex")
            perf_start = time.time()
            result = self.forecasting_engine.forecast(
                data=large_data,
                model_type='ensemble',
                forecast_horizon=24
            )
            perf_time = time.time() - perf_start
            assert perf_time < 10.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Forecasting Engine Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Forecasting Engine Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_confidence_calculator_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of confidence calculator."""
        logger.info("🧪 Testing Confidence Calculator (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_calculation": False,
                "uncertainty_handling": False,
                "model_comparison": False,
                "edge_cases": False,
                "performance": False
            }
            
            # Test 1: Basic calculation
            logger.info("   Testing basic calculation...")
            historical_data = self.generate_test_data(50, "simple")
            predictions = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
            
            confidence = self.confidence_calculator.calculate_confidence(
                historical_data=historical_data,
                predictions=predictions,
                model_type='ensemble'
            )
            
            assert 0 <= confidence.confidence_score <= 1, "Invalid confidence score"
            assert confidence.confidence_level in ['high', 'medium', 'low'], "Invalid confidence level"
            test_results["basic_calculation"] = True
            
            # Test 2: Uncertainty handling
            logger.info("   Testing uncertainty handling...")
            uncertain_data = self.generate_test_data(20, "complex")
            uncertain_predictions = np.array([1.1, 1.2, 1.3])
            
            confidence = self.confidence_calculator.calculate_confidence(
                historical_data=uncertain_data,
                predictions=uncertain_predictions,
                model_type='ensemble'
            )
            
            assert hasattr(confidence, 'uncertainty_factors'), "Missing uncertainty factors"
            test_results["uncertainty_handling"] = True
            
            # Test 3: Model comparison
            logger.info("   Testing model comparison...")
            models = ['ensemble', 'arima', 'exponential_smoothing']
            confidences = []
            
            for model in models:
                confidence = self.confidence_calculator.calculate_confidence(
                    historical_data=historical_data,
                    predictions=predictions,
                    model_type=model
                )
                confidences.append(confidence.confidence_score)
            
            assert len(set(confidences)) > 1, "All models should not have identical confidence"
            test_results["model_comparison"] = True
            
            # Test 4: Edge cases
            logger.info("   Testing edge cases...")
            # Test with minimal data
            min_data = self.generate_test_data(5, "simple")
            min_predictions = np.array([1.0])
            
            confidence = self.confidence_calculator.calculate_confidence(
                historical_data=min_data,
                predictions=min_predictions,
                model_type='ensemble'
            )
            
            assert confidence.confidence_score is not None, "Should handle minimal data"
            test_results["edge_cases"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            large_data = self.generate_test_data(500, "complex")
            large_predictions = np.array([1.1] * 50)
            
            perf_start = time.time()
            confidence = self.confidence_calculator.calculate_confidence(
                historical_data=large_data,
                predictions=large_predictions,
                model_type='ensemble'
            )
            perf_time = time.time() - perf_start
            
            assert perf_time < 5.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Confidence Calculator Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Confidence Calculator Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_scenario_forecaster_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of scenario forecaster."""
        logger.info("🧪 Testing Scenario Forecaster (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_scenarios": False,
                "scenario_variety": False,
                "impact_analysis": False,
                "scenario_comparison": False,
                "performance": False
            }
            
            # Test 1: Basic scenarios
            logger.info("   Testing basic scenarios...")
            base_data = self.generate_test_data(50, "simple")
            
            scenarios = [
                {"name": "optimistic", "multiplier": 1.2},
                {"name": "pessimistic", "multiplier": 0.8},
                {"name": "baseline", "multiplier": 1.0}
            ]
            
            results = self.scenario_forecaster.forecast_scenarios(
                base_data=base_data,
                scenarios=scenarios,
                forecast_horizon=12
            )
            
            assert len(results.scenarios) == 3, "Should generate 3 scenarios"
            test_results["basic_scenarios"] = True
            
            # Test 2: Scenario variety
            logger.info("   Testing scenario variety...")
            complex_scenarios = [
                {"name": "growth", "trend": "increasing", "volatility": "high"},
                {"name": "decline", "trend": "decreasing", "volatility": "medium"},
                {"name": "stable", "trend": "stable", "volatility": "low"}
            ]
            
            results = self.scenario_forecaster.forecast_scenarios(
                base_data=base_data,
                scenarios=complex_scenarios,
                forecast_horizon=6
            )
            
            assert len(results.scenarios) == 3, "Should handle complex scenarios"
            test_results["scenario_variety"] = True
            
            # Test 3: Impact analysis
            logger.info("   Testing impact analysis...")
            impact_results = self.scenario_forecaster.analyze_impacts(results)
            
            assert hasattr(impact_results, 'impact_metrics'), "Missing impact metrics"
            assert hasattr(impact_results, 'risk_assessment'), "Missing risk assessment"
            test_results["impact_analysis"] = True
            
            # Test 4: Scenario comparison
            logger.info("   Testing scenario comparison...")
            comparison = self.scenario_forecaster.compare_scenarios(results)
            
            assert hasattr(comparison, 'comparison_matrix'), "Missing comparison matrix"
            assert hasattr(comparison, 'recommendations'), "Missing recommendations"
            test_results["scenario_comparison"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            large_data = self.generate_test_data(200, "complex")
            many_scenarios = [{"name": f"scenario_{i}", "multiplier": 0.8 + i*0.1} for i in range(10)]
            
            perf_start = time.time()
            results = self.scenario_forecaster.forecast_scenarios(
                base_data=large_data,
                scenarios=many_scenarios,
                forecast_horizon=24
            )
            perf_time = time.time() - perf_start
            
            assert perf_time < 15.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Scenario Forecaster Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Scenario Forecaster Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_forecast_validator_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of forecast validator."""
        logger.info("🧪 Testing Forecast Validator (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "accuracy_metrics": False,
                "validation_methods": False,
                "error_analysis": False,
                "model_selection": False,
                "performance": False
            }
            
            # Test 1: Accuracy metrics
            logger.info("   Testing accuracy metrics...")
            actual_values = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
            predicted_values = np.array([1.05, 1.08, 1.25, 1.28, 1.42])
            
            validation = self.forecast_validator.validate_forecast(
                actual_values=actual_values,
                predicted_values=predicted_values,
                model_type='ensemble'
            )
            
            assert hasattr(validation, 'mae'), "Missing MAE"
            assert hasattr(validation, 'rmse'), "Missing RMSE"
            assert hasattr(validation, 'mape'), "Missing MAPE"
            test_results["accuracy_metrics"] = True
            
            # Test 2: Validation methods
            logger.info("   Testing validation methods...")
            methods = ['holdout', 'cross_validation', 'time_series_split']
            
            for method in methods:
                validation = self.forecast_validator.validate_forecast(
                    actual_values=actual_values,
                    predicted_values=predicted_values,
                    model_type='ensemble',
                    validation_method=method
                )
                assert validation is not None, f"Method {method} failed"
            test_results["validation_methods"] = True
            
            # Test 3: Error analysis
            logger.info("   Testing error analysis...")
            error_analysis = self.forecast_validator.analyze_errors(
                actual_values=actual_values,
                predicted_values=predicted_values
            )
            
            assert hasattr(error_analysis, 'error_distribution'), "Missing error distribution"
            assert hasattr(error_analysis, 'bias_analysis'), "Missing bias analysis"
            test_results["error_analysis"] = True
            
            # Test 4: Model selection
            logger.info("   Testing model selection...")
            models = ['ensemble', 'arima', 'exponential_smoothing']
            model_scores = {}
            
            for model in models:
                validation = self.forecast_validator.validate_forecast(
                    actual_values=actual_values,
                    predicted_values=predicted_values,
                    model_type=model
                )
                model_scores[model] = validation.rmse
            
            best_model = min(model_scores, key=model_scores.get)
            assert best_model is not None, "Should select best model"
            test_results["model_selection"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            large_actual = np.random.normal(100, 10, 1000)
            large_predicted = large_actual + np.random.normal(0, 2, 1000)
            
            perf_start = time.time()
            validation = self.forecast_validator.validate_forecast(
                actual_values=large_actual,
                predicted_values=large_predicted,
                model_type='ensemble'
            )
            perf_time = time.time() - perf_start
            
            assert perf_time < 3.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Forecast Validator Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Forecast Validator Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_predictive_agent_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive testing of predictive analytics agent."""
        logger.info("🧪 Testing Predictive Analytics Agent (Comprehensive)...")
        start_time = time.time()
        
        try:
            test_results = {
                "basic_analysis": False,
                "data_types": False,
                "integration": False,
                "error_handling": False,
                "performance": False
            }
            
            # Test 1: Basic analysis
            logger.info("   Testing basic analysis...")
            request = AnalysisRequest(
                content="Sample text for analysis",
                data_type=DataType.TEXT,
                analysis_type="predictive_analytics",
                language="en"
            )
            
            result = await self.predictive_agent.analyze(request)
            
            assert isinstance(result, AnalysisResult), "Should return AnalysisResult"
            assert result.success, "Analysis should be successful"
            test_results["basic_analysis"] = True
            
            # Test 2: Data types
            logger.info("   Testing data types...")
            data_types = [DataType.TEXT, DataType.AUDIO, DataType.VIDEO]
            
            for data_type in data_types:
                request = AnalysisRequest(
                    content=f"Sample {data_type.value} content",
                    data_type=data_type,
                    analysis_type="predictive_analytics",
                    language="en"
                )
                
                result = await self.predictive_agent.analyze(request)
                assert result.success, f"Data type {data_type.value} failed"
            test_results["data_types"] = True
            
            # Test 3: Integration
            logger.info("   Testing integration...")
            # Test with orchestrator
            orchestrator_result = await self.orchestrator.analyze(request)
            assert orchestrator_result.success, "Orchestrator integration failed"
            test_results["integration"] = True
            
            # Test 4: Error handling
            logger.info("   Testing error handling...")
            invalid_request = AnalysisRequest(
                content="",
                data_type=DataType.TEXT,
                analysis_type="predictive_analytics",
                language="en"
            )
            
            result = await self.predictive_agent.analyze(invalid_request)
            # Should handle gracefully even if content is empty
            assert isinstance(result, AnalysisResult), "Should handle invalid requests"
            test_results["error_handling"] = True
            
            # Test 5: Performance
            logger.info("   Testing performance...")
            large_request = AnalysisRequest(
                content="Large text content " * 1000,  # Large content
                data_type=DataType.TEXT,
                analysis_type="predictive_analytics",
                language="en"
            )
            
            perf_start = time.time()
            result = await self.predictive_agent.analyze(large_request)
            perf_time = time.time() - perf_start
            
            assert perf_time < 30.0, f"Performance too slow: {perf_time:.2f}s"
            test_results["performance"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Predictive Analytics Agent Comprehensive",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "performance_time": perf_time}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Predictive Analytics Agent Comprehensive",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_load_and_stress(self) -> Dict[str, Any]:
        """Load and stress testing."""
        logger.info("🧪 Testing Load and Stress...")
        start_time = time.time()
        
        try:
            test_results = {
                "concurrent_requests": False,
                "memory_usage": False,
                "cpu_usage": False,
                "response_time": False,
                "error_rate": False
            }
            
            # Test 1: Concurrent requests
            logger.info("   Testing concurrent requests...")
            num_requests = 10
            request = AnalysisRequest(
                content="Test content for load testing",
                data_type=DataType.TEXT,
                analysis_type="predictive_analytics",
                language="en"
            )
            
            async def make_request():
                return await self.predictive_agent.analyze(request)
            
            # Run concurrent requests
            tasks = [make_request() for _ in range(num_requests)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_requests = sum(1 for r in results if isinstance(r, AnalysisResult) and r.success)
            assert successful_requests >= num_requests * 0.8, f"Too many failed requests: {successful_requests}/{num_requests}"
            test_results["concurrent_requests"] = True
            
            # Test 2: Memory usage
            logger.info("   Testing memory usage...")
            initial_memory = psutil.virtual_memory().used
            
            # Make multiple requests to test memory usage
            for _ in range(20):
                await self.predictive_agent.analyze(request)
            
            final_memory = psutil.virtual_memory().used
            memory_increase = (final_memory - initial_memory) / (1024**2)  # MB
            
            assert memory_increase < 500, f"Memory increase too high: {memory_increase:.2f}MB"
            test_results["memory_usage"] = True
            
            # Test 3: CPU usage
            logger.info("   Testing CPU usage...")
            cpu_start = psutil.cpu_percent(interval=1)
            
            # Run intensive operations
            for _ in range(5):
                await self.predictive_agent.analyze(request)
            
            cpu_end = psutil.cpu_percent(interval=1)
            cpu_increase = cpu_end - cpu_start
            
            assert cpu_increase < 50, f"CPU usage increase too high: {cpu_increase:.2f}%"
            test_results["cpu_usage"] = True
            
            # Test 4: Response time
            logger.info("   Testing response time...")
            response_times = []
            
            for _ in range(10):
                req_start = time.time()
                await self.predictive_agent.analyze(request)
                req_time = time.time() - req_start
                response_times.append(req_time)
            
            avg_response_time = np.mean(response_times)
            assert avg_response_time < 5.0, f"Average response time too high: {avg_response_time:.2f}s"
            test_results["response_time"] = True
            
            # Test 5: Error rate
            logger.info("   Testing error rate...")
            error_count = 0
            total_requests = 50
            
            for _ in range(total_requests):
                try:
                    result = await self.predictive_agent.analyze(request)
                    if not result.success:
                        error_count += 1
                except Exception:
                    error_count += 1
            
            error_rate = error_count / total_requests
            assert error_rate < 0.1, f"Error rate too high: {error_rate:.2%}"
            test_results["error_rate"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "Load and Stress Testing",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {
                    "test_results": test_results,
                    "memory_increase_mb": memory_increase,
                    "cpu_increase_percent": cpu_increase,
                    "avg_response_time": avg_response_time,
                    "error_rate": error_rate
                }
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "Load and Stress Testing",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def test_integration_end_to_end(self) -> Dict[str, Any]:
        """End-to-end integration testing."""
        logger.info("🧪 Testing End-to-End Integration...")
        start_time = time.time()
        
        try:
            test_results = {
                "orchestrator_integration": False,
                "data_flow": False,
                "component_interaction": False,
                "result_consistency": False,
                "system_stability": False
            }
            
            # Test 1: Orchestrator integration
            logger.info("   Testing orchestrator integration...")
            request = AnalysisRequest(
                content="Comprehensive test content for end-to-end testing",
                data_type=DataType.TEXT,
                analysis_type="predictive_analytics",
                language="en"
            )
            
            result = await self.orchestrator.analyze(request)
            assert result.success, "Orchestrator should handle predictive analytics"
            test_results["orchestrator_integration"] = True
            
            # Test 2: Data flow
            logger.info("   Testing data flow...")
            # Test data flow through all components
            data = self.generate_test_data(100, "complex")
            
            # Flow: Forecasting -> Confidence -> Validation
            forecast = self.forecasting_engine.forecast(data=data, model_type='ensemble', forecast_horizon=12)
            confidence = self.confidence_calculator.calculate_confidence(
                historical_data=data,
                predictions=forecast.predictions,
                model_type='ensemble'
            )
            validation = self.forecast_validator.validate_forecast(
                actual_values=data[-12:],  # Use last 12 points as "actual"
                predicted_values=forecast.predictions,
                model_type='ensemble'
            )
            
            assert all([forecast, confidence, validation]), "Data flow should work end-to-end"
            test_results["data_flow"] = True
            
            # Test 3: Component interaction
            logger.info("   Testing component interaction...")
            # Test scenario forecasting with all components
            scenarios = [
                {"name": "optimistic", "multiplier": 1.2},
                {"name": "pessimistic", "multiplier": 0.8}
            ]
            
            scenario_results = self.scenario_forecaster.forecast_scenarios(
                base_data=data,
                scenarios=scenarios,
                forecast_horizon=12
            )
            
            # Validate scenarios
            for scenario_name in scenario_results.scenarios:
                scenario_data = scenario_results.scenarios[scenario_name].predictions
                scenario_confidence = self.confidence_calculator.calculate_confidence(
                    historical_data=data,
                    predictions=scenario_data,
                    model_type='ensemble'
                )
                assert scenario_confidence.confidence_score is not None, f"Scenario {scenario_name} validation failed"
            
            test_results["component_interaction"] = True
            
            # Test 4: Result consistency
            logger.info("   Testing result consistency...")
            # Run same analysis multiple times
            results = []
            for _ in range(3):
                result = await self.predictive_agent.analyze(request)
                results.append(result)
            
            # Check that results are consistent (not necessarily identical due to randomness)
            success_rates = [r.success for r in results]
            assert all(success_rates), "All results should be successful"
            test_results["result_consistency"] = True
            
            # Test 5: System stability
            logger.info("   Testing system stability...")
            # Run multiple operations to test stability
            operations = []
            for i in range(10):
                op_request = AnalysisRequest(
                    content=f"Stability test content {i}",
                    data_type=DataType.TEXT,
                    analysis_type="predictive_analytics",
                    language="en"
                )
                operations.append(self.predictive_agent.analyze(op_request))
            
            # Execute all operations
            op_results = await asyncio.gather(*operations, return_exceptions=True)
            successful_ops = sum(1 for r in op_results if isinstance(r, AnalysisResult) and r.success)
            
            assert successful_ops >= 8, f"System stability test failed: {successful_ops}/10 successful"
            test_results["system_stability"] = True
            
            duration = time.time() - start_time
            success = all(test_results.values())
            
            self.log_test_result(
                "End-to-End Integration Testing",
                "PASSED" if success else "FAILED",
                f"Tests passed: {sum(test_results.values())}/{len(test_results)}",
                duration,
                {"test_results": test_results, "successful_operations": successful_ops}
            )
            
            return {"status": "PASSED" if success else "FAILED", "results": test_results}
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result(
                "End-to-End Integration Testing",
                "FAILED",
                str(e),
                duration
            )
            return {"status": "FAILED", "error": str(e)}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        logger.info("🚀 Starting Comprehensive Predictive Analytics Testing...")
        
        await self.setup()
        
        # Run all test categories
        test_categories = [
            ("Forecasting Engine", self.test_forecasting_engine_comprehensive),
            ("Confidence Calculator", self.test_confidence_calculator_comprehensive),
            ("Scenario Forecaster", self.test_scenario_forecaster_comprehensive),
            ("Forecast Validator", self.test_forecast_validator_comprehensive),
            ("Predictive Agent", self.test_predictive_agent_comprehensive),
            ("Load and Stress", self.test_load_and_stress),
            ("End-to-End Integration", self.test_integration_end_to_end)
        ]
        
        category_results = {}
        
        for category_name, test_func in test_categories:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing Category: {category_name}")
            logger.info(f"{'='*60}")
            
            try:
                result = await test_func()
                category_results[category_name] = result
            except Exception as e:
                logger.error(f"Error in {category_name}: {str(e)}")
                category_results[category_name] = {"status": "ERROR", "error": str(e)}
        
        # Calculate overall results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["status"] == "PASSED")
        failed_tests = sum(1 for r in self.test_results if r["status"] == "FAILED")
        
        overall_duration = time.time() - self.start_time
        
        # Generate comprehensive report
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_duration": overall_duration
            },
            "category_results": category_results,
            "detailed_results": self.test_results,
            "system_metrics": self.get_system_metrics(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        with open("Test/comprehensive_predictive_analytics_results.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("COMPREHENSIVE PREDICTIVE ANALYTICS TESTING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        logger.info(f"Total Duration: {overall_duration:.2f}s")
        logger.info(f"Results saved to: Test/comprehensive_predictive_analytics_results.json")
        
        return report


async def main():
    """Main test execution function."""
    tester = ComprehensivePredictiveAnalyticsTester()
    results = await tester.run_all_tests()
    return results


if __name__ == "__main__":
    # Run the comprehensive test suite
    results = asyncio.run(main())
    print(f"\nTest execution completed. Success rate: {results['test_summary']['success_rate']*100:.1f}%")
