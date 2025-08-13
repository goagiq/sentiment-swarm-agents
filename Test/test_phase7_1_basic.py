"""
Basic Test Phase 7.1: Advanced Machine Learning Implementation

This script tests only the basic structure and configuration of Phase 7.1
without importing any problematic modules.
"""

import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            logger.info(f"✅ Directory exists: {dir_path}")
        
        # Check that __init__.py files exist
        init_files = [
            "src/core/advanced_ml/__init__.py",
            "src/models/neural_networks/__init__.py",
            "src/models/ensemble_models/__init__.py",
            "src/models/time_series_models/__init__.py"
        ]
        
        for init_file in init_files:
            assert os.path.exists(init_file), f"Init file {init_file} does not exist"
            logger.info(f"✅ Init file exists: {init_file}")
        
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
        logger.info(f"✅ Config file exists: {config_file}")
        
        # Check that agent file exists
        agent_file = "src/agents/advanced_ml_agent.py"
        assert os.path.exists(agent_file), f"Agent file {agent_file} does not exist"
        logger.info(f"✅ Agent file exists: {agent_file}")
        
        # Check that core ML files exist
        core_files = [
            "src/core/advanced_ml/deep_learning_engine.py",
            "src/core/advanced_ml/transfer_learning_service.py",
            "src/core/advanced_ml/model_versioning.py",
            "src/core/advanced_ml/ensemble_methods.py",
            "src/core/advanced_ml/time_series_models.py",
            "src/core/advanced_ml/clustering_algorithms.py",
            "src/core/advanced_ml/dimensionality_reduction.py",
            "src/core/advanced_ml/automl_pipeline.py"
        ]
        
        for core_file in core_files:
            assert os.path.exists(core_file), f"Core file {core_file} does not exist"
            logger.info(f"✅ Core file exists: {core_file}")
        
        logger.info("✅ Configuration Files test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration Files test failed: {str(e)}")
        return False


def test_config_import():
    """Test that configuration can be imported."""
    logger.info("Testing Configuration Import...")
    
    try:
        # Import configuration
        from config.advanced_ml_config import get_advanced_ml_config, validate_advanced_ml_config
        
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
        
        logger.info("✅ Configuration structure is valid")
        
        # Validate configuration
        assert validate_advanced_ml_config(), "Configuration validation failed"
        logger.info("✅ Configuration validation passed")
        
        logger.info("✅ Configuration Import test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration Import test failed: {str(e)}")
        return False


def test_model_versioning_import():
    """Test that model versioning can be imported."""
    logger.info("Testing Model Versioning Import...")
    
    try:
        # Import model versioning
        from core.advanced_ml.model_versioning import ModelVersioning
        
        # Initialize versioning
        versioning = ModelVersioning()
        
        # Test basic functionality
        assert hasattr(versioning, 'registry_path'), "Missing registry_path"
        assert hasattr(versioning, 'create_version'), "Missing create_version method"
        assert hasattr(versioning, 'load_version'), "Missing load_version method"
        assert hasattr(versioning, 'list_versions'), "Missing list_versions method"
        
        logger.info("✅ Model Versioning Import test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model Versioning Import test failed: {str(e)}")
        return False


def main():
    """Run all Phase 7.1 basic tests."""
    logger.info("🚀 Starting Phase 7.1 Basic Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Configuration Files", test_config_files),
        ("Configuration Import", test_config_import),
        ("Model Versioning Import", test_model_versioning_import)
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
    logger.info("📊 Phase 7.1 Basic Test Results Summary")
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
        logger.info("🎉 All Phase 7.1 basic tests passed! Structure is ready.")
        logger.info("Note: Deep learning frameworks need to be installed for full functionality.")
        return True
    else:
        logger.error(f"⚠️  {total - passed} tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
