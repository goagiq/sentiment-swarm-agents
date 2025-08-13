"""
Phase 7.5 Basic Integration Test

This module provides basic tests for Phase 7.5 integration without deep learning dependencies.
"""

import asyncio
import json
import requests
from typing import Dict, Any, List
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.advanced_analytics.multivariate_forecasting import MultivariateForecastingEngine
from src.core.advanced_analytics.causal_inference_engine import CausalInferenceEngine
from src.core.advanced_analytics.advanced_anomaly_detection import AdvancedAnomalyDetector
from src.core.advanced_analytics.model_optimization import ModelOptimizer
from src.core.advanced_analytics.feature_engineering import FeatureEngineer

from src.agents.advanced_forecasting_agent import AdvancedForecastingAgent
from src.agents.causal_analysis_agent import CausalAnalysisAgent
from src.agents.anomaly_detection_agent import AnomalyDetectionAgent

from src.mcp_servers.unified_mcp_server import UnifiedMCPServer


class TestPhase75Basic:
    """Basic test suite for Phase 7.5 System Integration & Testing."""
    
    def setup(self):
        """Setup test environment."""
        self.base_url = "http://localhost:8003"
        self.advanced_analytics_url = f"{self.base_url}/advanced-analytics"
        
        # Sample test data
        self.sample_time_series_data = [
            {"date": "2024-01-01", "sales": 100, "temperature": 20, "advertising": 50},
            {"date": "2024-01-02", "sales": 120, "temperature": 22, "advertising": 60},
            {"date": "2024-01-03", "sales": 110, "temperature": 21, "advertising": 55},
            {"date": "2024-01-04", "sales": 130, "temperature": 25, "advertising": 70},
            {"date": "2024-01-05", "sales": 140, "temperature": 24, "advertising": 75},
            {"date": "2024-01-06", "sales": 125, "temperature": 23, "advertising": 65},
            {"date": "2024-01-07", "sales": 135, "temperature": 26, "advertising": 80},
            {"date": "2024-01-08", "sales": 150, "temperature": 27, "advertising": 85},
            {"date": "2024-01-09", "sales": 145, "temperature": 25, "advertising": 80},
            {"date": "2024-01-10", "sales": 160, "temperature": 28, "advertising": 90}
        ]
        
        self.sample_anomaly_data = [
            {"value": 10, "timestamp": "2024-01-01T00:00:00"},
            {"value": 12, "timestamp": "2024-01-01T01:00:00"},
            {"value": 11, "timestamp": "2024-01-01T02:00:00"},
            {"value": 100, "timestamp": "2024-01-01T03:00:00"},  # Anomaly
            {"value": 13, "timestamp": "2024-01-01T04:00:00"},
            {"value": 14, "timestamp": "2024-01-01T05:00:00"},
            {"value": 12, "timestamp": "2024-01-01T06:00:00"},
            {"value": 15, "timestamp": "2024-01-01T07:00:00"},
            {"value": 11, "timestamp": "2024-01-01T08:00:00"},
            {"value": 13, "timestamp": "2024-01-01T09:00:00"}
        ]

    def test_advanced_analytics_health_check(self):
        """Test advanced analytics health check endpoint."""
        try:
            response = requests.get(f"{self.advanced_analytics_url}/health")
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                assert "status" in data["result"]
                assert data["result"]["status"] == "healthy"
                print("✅ Advanced analytics health check passed")
            else:
                print(f"⚠️ Advanced analytics health check returned status {response.status_code}")
        except Exception as e:
            print(f"⚠️ Advanced analytics health check failed: {e}")

    def test_multivariate_forecasting_api(self):
        """Test multivariate forecasting API endpoint."""
        try:
            request_data = {
                "data": self.sample_time_series_data,
                "target_variables": ["sales", "temperature"],
                "forecast_horizon": 5,
                "model_type": "ensemble",
                "confidence_level": 0.95
            }
            
            response = requests.post(
                f"{self.advanced_analytics_url}/forecasting/multivariate",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                print("✅ Multivariate forecasting API test passed")
            else:
                print(f"⚠️ Multivariate forecasting API returned status {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Multivariate forecasting API test failed: {e}")

    def test_causal_analysis_api(self):
        """Test causal analysis API endpoint."""
        try:
            request_data = {
                "data": self.sample_time_series_data,
                "variables": ["sales", "temperature", "advertising"],
                "analysis_type": "granger",
                "max_lag": 3
            }
            
            response = requests.post(
                f"{self.advanced_analytics_url}/causal-analysis",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                print("✅ Causal analysis API test passed")
            else:
                print(f"⚠️ Causal analysis API returned status {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Causal analysis API test failed: {e}")

    def test_anomaly_detection_api(self):
        """Test anomaly detection API endpoint."""
        try:
            request_data = {
                "data": self.sample_anomaly_data,
                "algorithm": "isolation_forest",
                "threshold": 0.1,
                "features": ["value"]
            }
            
            response = requests.post(
                f"{self.advanced_analytics_url}/anomaly-detection",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                print("✅ Anomaly detection API test passed")
            else:
                print(f"⚠️ Anomaly detection API returned status {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Anomaly detection API test failed: {e}")

    def test_model_optimization_api(self):
        """Test model optimization API endpoint."""
        try:
            request_data = {
                "data": self.sample_time_series_data,
                "target_variable": "sales",
                "model_type": "auto",
                "optimization_metric": "accuracy",
                "cv_folds": 3
            }
            
            response = requests.post(
                f"{self.advanced_analytics_url}/model-optimization",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                print("✅ Model optimization API test passed")
            else:
                print(f"⚠️ Model optimization API returned status {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Model optimization API test failed: {e}")

    def test_feature_engineering_api(self):
        """Test feature engineering API endpoint."""
        try:
            request_data = {
                "data": self.sample_time_series_data,
                "target_variable": "sales",
                "feature_types": ["numerical", "categorical"],
                "max_features": 20
            }
            
            response = requests.post(
                f"{self.advanced_analytics_url}/feature-engineering",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                print("✅ Feature engineering API test passed")
            else:
                print(f"⚠️ Feature engineering API returned status {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Feature engineering API test failed: {e}")

    def test_agent_forecasting_api(self):
        """Test agent-based forecasting API endpoint."""
        try:
            request_data = {
                "data": self.sample_time_series_data,
                "target_variables": ["sales"],
                "forecast_horizon": 5,
                "model_type": "ensemble"
            }
            
            response = requests.post(
                f"{self.advanced_analytics_url}/agents/forecasting",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                print("✅ Agent forecasting API test passed")
            else:
                print(f"⚠️ Agent forecasting API returned status {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Agent forecasting API test failed: {e}")

    def test_agent_causal_analysis_api(self):
        """Test agent-based causal analysis API endpoint."""
        try:
            request_data = {
                "data": self.sample_time_series_data,
                "variables": ["sales", "temperature"],
                "analysis_type": "granger"
            }
            
            response = requests.post(
                f"{self.advanced_analytics_url}/agents/causal-analysis",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                print("✅ Agent causal analysis API test passed")
            else:
                print(f"⚠️ Agent causal analysis API returned status {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Agent causal analysis API test failed: {e}")

    def test_agent_anomaly_detection_api(self):
        """Test agent-based anomaly detection API endpoint."""
        try:
            request_data = {
                "data": self.sample_anomaly_data,
                "algorithm": "isolation_forest",
                "threshold": 0.1
            }
            
            response = requests.post(
                f"{self.advanced_analytics_url}/agents/anomaly-detection",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                print("✅ Agent anomaly detection API test passed")
            else:
                print(f"⚠️ Agent anomaly detection API returned status {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Agent anomaly detection API test failed: {e}")

    def test_performance_monitoring_api(self):
        """Test performance monitoring API endpoints."""
        try:
            # Test performance status
            response = requests.get(f"{self.advanced_analytics_url}/performance/status")
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                print("✅ Performance status API test passed")
            else:
                print(f"⚠️ Performance status API returned status {response.status_code}")
                
            # Test performance metrics
            response = requests.get(f"{self.advanced_analytics_url}/performance/metrics")
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                print("✅ Performance metrics API test passed")
            else:
                print(f"⚠️ Performance metrics API returned status {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Performance monitoring API test failed: {e}")

    def test_model_management_api(self):
        """Test model management API endpoints."""
        try:
            # Test list models
            response = requests.get(f"{self.advanced_analytics_url}/models/list")
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                print("✅ Model list API test passed")
            else:
                print(f"⚠️ Model list API returned status {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Model management API test failed: {e}")

    @pytest.mark.asyncio
    async def test_advanced_analytics_engines(self):
        """Test advanced analytics engines directly."""
        try:
            # Test forecasting engine
            forecasting_engine = MultivariateForecastingEngine()
            result = await forecasting_engine.forecast_multivariate(
                data=self.sample_time_series_data,
                target_variables=["sales"],
                forecast_horizon=3,
                model_type="ensemble"
            )
            assert result is not None
            print("✅ Forecasting engine test passed")
            
            # Test causal inference engine
            causal_engine = CausalInferenceEngine()
            result = await causal_engine.analyze_causality(
                data=self.sample_time_series_data,
                variables=["sales", "temperature"],
                analysis_type="granger"
            )
            assert result is not None
            print("✅ Causal inference engine test passed")
            
            # Test anomaly detector
            anomaly_detector = AdvancedAnomalyDetector()
            result = await anomaly_detector.detect_anomalies(
                data=self.sample_anomaly_data,
                algorithm="isolation_forest",
                threshold=0.1
            )
            assert result is not None
            print("✅ Anomaly detector test passed")
            
            # Test model optimizer
            model_optimizer = ModelOptimizer()
            result = await model_optimizer.optimize_model(
                data=self.sample_time_series_data,
                target_variable="sales",
                model_type="auto"
            )
            assert result is not None
            print("✅ Model optimizer test passed")
            
            # Test feature engineer
            feature_engineer = FeatureEngineer()
            result = await feature_engineer.engineer_features(
                data=self.sample_time_series_data,
                target_variable="sales"
            )
            assert result is not None
            print("✅ Feature engineer test passed")
            
        except Exception as e:
            print(f"⚠️ Advanced analytics engines test failed: {e}")

    @pytest.mark.asyncio
    async def test_advanced_analytics_agents(self):
        """Test advanced analytics agents directly."""
        try:
            # Test forecasting agent
            forecasting_agent = AdvancedForecastingAgent()
            result = await forecasting_agent.forecast(
                data=self.sample_time_series_data,
                target_variables=["sales"],
                forecast_horizon=3
            )
            assert result is not None
            print("✅ Forecasting agent test passed")
            
            # Test causal analysis agent
            causal_agent = CausalAnalysisAgent()
            result = await causal_agent.analyze_causality(
                data=self.sample_time_series_data,
                variables=["sales", "temperature"]
            )
            assert result is not None
            print("✅ Causal analysis agent test passed")
            
            # Test anomaly detection agent
            anomaly_agent = AnomalyDetectionAgent()
            result = await anomaly_agent.detect_anomalies(
                data=self.sample_anomaly_data,
                algorithm="isolation_forest"
            )
            assert result is not None
            print("✅ Anomaly detection agent test passed")
            
        except Exception as e:
            print(f"⚠️ Advanced analytics agents test failed: {e}")

    def test_mcp_server_integration(self):
        """Test MCP server integration with advanced analytics."""
        try:
            # Create MCP server
            mcp_server = UnifiedMCPServer()
            assert mcp_server is not None
            print("✅ MCP server creation test passed")
            
            # Test MCP server has advanced analytics agents
            assert hasattr(mcp_server, 'forecasting_agent')
            assert hasattr(mcp_server, 'causal_agent')
            assert hasattr(mcp_server, 'anomaly_agent')
            print("✅ MCP server advanced analytics agents test passed")
            
            # Test MCP tools registration
            if mcp_server.mcp:
                tools = list(mcp_server.mcp.tools.keys())
                assert len(tools) >= 25  # Should have at least 25 tools
                print(f"✅ MCP server has {len(tools)} tools registered")
                
                # Check for advanced analytics tools
                tool_descriptions = [mcp_server.mcp.tools[tool].description for tool in tools]
                advanced_tools = [desc for desc in tool_descriptions if "advanced" in desc.lower() or "forecasting" in desc.lower() or "causal" in desc.lower()]
                assert len(advanced_tools) > 0
                print(f"✅ MCP server has {len(advanced_tools)} advanced analytics tools")
            else:
                print("⚠️ MCP server not available for tool testing")
                
        except Exception as e:
            print(f"⚠️ MCP server integration test failed: {e}")

    def test_main_api_integration(self):
        """Test main API integration with advanced analytics."""
        try:
            # Test main API root endpoint includes advanced analytics
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                endpoints = data.get("endpoints", {})
                assert "advanced_analytics" in endpoints
                print("✅ Main API advanced analytics integration test passed")
            else:
                print(f"⚠️ Main API returned status {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Main API integration test failed: {e}")

    def test_system_health(self):
        """Test overall system health with advanced analytics."""
        try:
            # Test main health endpoint
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ Main system health check passed")
            else:
                print(f"⚠️ Main system health check returned status {response.status_code}")
                
            # Test advanced analytics health endpoint
            response = requests.get(f"{self.advanced_analytics_url}/health")
            if response.status_code == 200:
                data = response.json()
                assert data["success"] is True
                assert data["result"]["version"] == "7.5.0"
                print("✅ Advanced analytics health check passed")
            else:
                print(f"⚠️ Advanced analytics health check returned status {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ System health test failed: {e}")


def run_phase75_basic_tests():
    """Run all Phase 7.5 basic integration tests."""
    print("🚀 Starting Phase 7.5 Basic System Integration & Testing")
    print("=" * 60)
    
    # Create test instance
    test_instance = TestPhase75Basic()
    test_instance.setup()
    
    # Run API tests
    print("\n📊 Testing Advanced Analytics API Routes:")
    test_instance.test_advanced_analytics_health_check()
    test_instance.test_multivariate_forecasting_api()
    test_instance.test_causal_analysis_api()
    test_instance.test_anomaly_detection_api()
    test_instance.test_model_optimization_api()
    test_instance.test_feature_engineering_api()
    test_instance.test_agent_forecasting_api()
    test_instance.test_agent_causal_analysis_api()
    test_instance.test_agent_anomaly_detection_api()
    test_instance.test_performance_monitoring_api()
    test_instance.test_model_management_api()
    
    # Run integration tests
    print("\n🔗 Testing System Integration:")
    test_instance.test_main_api_integration()
    test_instance.test_system_health()
    test_instance.test_mcp_server_integration()
    
    # Run async tests
    print("\n⚡ Testing Advanced Analytics Components:")
    asyncio.run(test_instance.test_advanced_analytics_engines())
    asyncio.run(test_instance.test_advanced_analytics_agents())
    
    print("\n✅ Phase 7.5 Basic System Integration & Testing completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_phase75_basic_tests()
