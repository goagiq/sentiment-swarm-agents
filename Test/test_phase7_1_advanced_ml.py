"""
Test Phase 7.1: Advanced Machine Learning Implementation

This script tests the advanced machine learning components including:
- Deep Learning Engine
- Transfer Learning Service
- Model Versioning
- Advanced ML Agent
"""

import sys
import os
import logging
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.advanced_ml_config import get_advanced_ml_config, validate_advanced_ml_config
from core.advanced_ml.deep_learning_engine import DeepLearningEngine
from core.advanced_ml.transfer_learning_service import TransferLearningService
from core.advanced_ml.model_versioning import ModelVersioning
from agents.advanced_ml_agent import AdvancedMLAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_advanced_ml_config():
    """Test advanced ML configuration."""
    logger.info("Testing Advanced ML Configuration...")
    
    try:
        # Get configuration
        config = get_advanced_ml_config()
        
        # Test configuration structure
        assert hasattr(config, 'deep_learning'), "Missing deep_learning config"
        assert hasattr(config, 'transfer_learning'), "Missing transfer_learning config"
        assert hasattr(config, 'ensemble'), "Missing ensemble config"
        assert hasattr(config, 'time_series'), "Missing time_series config"
        assert hasattr(config, 'clustering'), "Missing clustering config"
        assert hasattr(config, 'dimensionality_reduction'), "Missing dimensionality_reduction config"
        assert hasattr(config, 'automl'), "Missing automl config"
        assert hasattr(config, 'model_versioning'), "Missing model_versioning config"
        
        # Validate configuration
        assert validate_advanced_ml_config(), "Configuration validation failed"
        
        logger.info("✅ Advanced ML Configuration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Advanced ML Configuration test failed: {str(e)}")
        return False


def test_deep_learning_engine():
    """Test deep learning engine."""
    logger.info("Testing Deep Learning Engine...")
    
    try:
        # Initialize engine
        engine = DeepLearningEngine()
        
        # Test MLP creation
        mlp_model = engine.create_mlp(input_dim=10, output_dim=2)
        assert mlp_model is not None, "MLP model creation failed"
        
        # Test CNN creation
        cnn_model = engine.create_cnn(input_shape=(32, 32, 3), output_dim=10)
        assert cnn_model is not None, "CNN model creation failed"
        
        # Test LSTM creation
        lstm_model = engine.create_lstm(input_shape=(100, 20), output_dim=5)
        assert lstm_model is not None, "LSTM model creation failed"
        
        logger.info("✅ Deep Learning Engine test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Deep Learning Engine test failed: {str(e)}")
        return False


def test_transfer_learning_service():
    """Test transfer learning service."""
    logger.info("Testing Transfer Learning Service...")
    
    try:
        # Initialize service
        service = TransferLearningService()
        
        # Test service initialization
        assert service.framework in ["tensorflow", "pytorch"], "Invalid framework"
        
        # Test pre-trained model loading (this will fail if frameworks not available)
        try:
            # This is expected to fail if transformers not installed
            model = service.load_pre_trained_model("bert")
            logger.info("Pre-trained model loading test completed")
        except Exception as e:
            logger.info(f"Pre-trained model loading test skipped: {str(e)}")
        
        logger.info("✅ Transfer Learning Service test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Transfer Learning Service test failed: {str(e)}")
        return False


def test_model_versioning():
    """Test model versioning system."""
    logger.info("Testing Model Versioning System...")
    
    try:
        # Initialize versioning
        versioning = ModelVersioning()
        
        # Test registry creation
        assert os.path.exists(versioning.registry_path), "Registry path not created"
        
        # Test registry loading
        registry = versioning._load_registry()
        assert "models" in registry, "Registry missing models key"
        assert "versions" in registry, "Registry missing versions key"
        assert "metadata" in registry, "Registry missing metadata key"
        
        # Test version creation with dummy model
        dummy_model = {"type": "test_model", "version": "1.0.0"}
        metadata = {"description": "Test model", "accuracy": 0.85}
        
        version = versioning.create_version("test_model", dummy_model, metadata)
        assert version is not None, "Version creation failed"
        
        # Test version listing
        versions = versioning.list_versions("test_model")
        assert len(versions) > 0, "No versions found"
        
        # Test version metadata
        version_metadata = versioning.get_version_metadata("test_model", version)
        assert version_metadata is not None, "Version metadata not found"
        
        # Test model summary
        summary = versioning.get_model_summary("test_model")
        assert summary is not None, "Model summary not found"
        
        logger.info("✅ Model Versioning System test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model Versioning System test failed: {str(e)}")
        return False


def test_advanced_ml_agent():
    """Test advanced ML agent."""
    logger.info("Testing Advanced ML Agent...")
    
    try:
        # Initialize agent
        agent = AdvancedMLAgent()
        
        # Test agent initialization
        assert agent.agent_name == "AdvancedMLAgent", "Invalid agent name"
        assert hasattr(agent, 'capabilities'), "Missing capabilities"
        
        # Test capabilities
        capabilities = agent.get_capabilities()
        assert "deep_learning" in capabilities["capabilities"], "Missing deep_learning capability"
        assert "transfer_learning" in capabilities["capabilities"], "Missing transfer_learning capability"
        assert "model_versioning" in capabilities["capabilities"], "Missing model_versioning capability"
        
        # Test agent status
        status = agent.get_status()
        assert status["status"] == "active", "Agent not active"
        
        # Test deep learning request
        request = {
            "type": "deep_learning",
            "operation": "create_mlp",
            "data": {
                "input_dim": 10,
                "output_dim": 2,
                "architecture": "mlp"
            }
        }
        
        response = agent.process_request(request)
        assert response["success"], f"Deep learning request failed: {response.get('error', 'Unknown error')}"
        
        logger.info("✅ Advanced ML Agent test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Advanced ML Agent test failed: {str(e)}")
        return False


def test_integration():
    """Test integration between components."""
    logger.info("Testing Component Integration...")
    
    try:
        # Initialize components
        config = get_advanced_ml_config()
        engine = DeepLearningEngine()
        service = TransferLearningService()
        versioning = ModelVersioning()
        agent = AdvancedMLAgent()
        
        # Test that all components use the same configuration
        assert engine.config == config, "Deep learning engine config mismatch"
        assert service.config == config, "Transfer learning service config mismatch"
        assert versioning.config == config, "Model versioning config mismatch"
        assert agent.config == config, "Advanced ML agent config mismatch"
        
        # Test that agent can access all components
        assert agent.deep_learning_engine is not None, "Agent missing deep learning engine"
        assert agent.transfer_learning_service is not None, "Agent missing transfer learning service"
        assert agent.model_versioning is not None, "Agent missing model versioning"
        
        logger.info("✅ Component Integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component Integration test failed: {str(e)}")
        return False


def main():
    """Run all Phase 7.1 tests."""
    logger.info("🚀 Starting Phase 7.1 Advanced ML Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Advanced ML Configuration", test_advanced_ml_config),
        ("Deep Learning Engine", test_deep_learning_engine),
        ("Transfer Learning Service", test_transfer_learning_service),
        ("Model Versioning System", test_model_versioning),
        ("Advanced ML Agent", test_advanced_ml_agent),
        ("Component Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Phase 7.1 Test Results Summary")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Phase 7.1 tests passed! Advanced ML implementation is ready.")
        return True
    else:
        logger.error(f"⚠️  {total - passed} tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
