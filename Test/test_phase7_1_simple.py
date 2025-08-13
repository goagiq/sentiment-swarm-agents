"""
Simple Test Phase 7.1: Advanced Machine Learning Implementation

This script tests the basic structure and configuration of Phase 7.1
without requiring deep learning frameworks.
"""

import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.advanced_ml_config import get_advanced_ml_config, validate_advanced_ml_config
from core.advanced_ml.model_versioning import ModelVersioning

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


def test_directory_structure():
    """Test that all required directories exist."""
    logger.info("Testing Directory Structure...")
    
    try:
        # Check advanced ML directories
        required_dirs = [
            "src/core/advanced_ml",
            "src/models/neural_networks",
            "src/models/ensemble_models",
            "src/models/time_series_models"
        ]
        
        for dir_path in required_dirs:
            assert os.path.exists(dir_path), f"Directory {dir_path} does not exist"
        
        # Check that __init__.py files exist
        init_files = [
            "src/core/advanced_ml/__init__.py",
            "src/models/neural_networks/__init__.py",
            "src/models/ensemble_models/__init__.py",
            "src/models/time_series_models/__init__.py"
        ]
        
        for init_file in init_files:
            assert os.path.exists(init_file), f"Init file {init_file} does not exist"
        
        logger.info("✅ Directory Structure test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Directory Structure test failed: {str(e)}")
        return False


def test_config_files():
    """Test that configuration files exist and are valid."""
    logger.info("Testing Configuration Files...")
    
    try:
        # Check that advanced ML config exists
        config_file = "src/config/advanced_ml_config.py"
        assert os.path.exists(config_file), f"Config file {config_file} does not exist"
        
        # Check that agent file exists
        agent_file = "src/agents/advanced_ml_agent.py"
        assert os.path.exists(agent_file), f"Agent file {agent_file} does not exist"
        
        logger.info("✅ Configuration Files test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration Files test failed: {str(e)}")
        return False


def main():
    """Run all Phase 7.1 simple tests."""
    logger.info("🚀 Starting Phase 7.1 Simple Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Advanced ML Configuration", test_advanced_ml_config),
        ("Model Versioning System", test_model_versioning),
        ("Directory Structure", test_directory_structure),
        ("Configuration Files", test_config_files)
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
    logger.info("📊 Phase 7.1 Simple Test Results Summary")
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
        logger.info("🎉 All Phase 7.1 simple tests passed! Basic structure is ready.")
        logger.info("Note: Deep learning frameworks need to be installed for full functionality.")
        return True
    else:
        logger.error(f"⚠️  {total - passed} tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
