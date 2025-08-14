"""
Test script for Phase 9: Enhanced AutoML Pipeline

This script tests the enhanced AutoML pipeline with comprehensive capabilities
including model selection, hyperparameter optimization, feature engineering,
and ensemble methods.
"""

import sys
import os
import logging
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from src.core.advanced_ml.automl_pipeline import EnhancedAutoMLPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_enhanced_automl_classification():
    """Test enhanced AutoML pipeline for classification tasks."""
    logger.info("Testing Enhanced AutoML Pipeline - Classification")
    
    try:
        # Generate synthetic classification dataset
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=15,
            n_redundant=5, n_classes=3, random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize enhanced AutoML pipeline
        automl = EnhancedAutoMLPipeline()
        
        # Run AutoML pipeline
        start_time = time.time()
        results = automl.run_automl(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task_type='classification',
            max_models=5,
            cv_folds=3,
            optimization_strategy='grid_search',
            ensemble_method='voting'
        )
        end_time = time.time()
        
        # Validate results
        assert results is not None, "AutoML results should not be None"
        assert 'best_model' in results, "Results should contain best_model"
        assert 'best_score' in results, "Results should contain best_score"
        assert 'models_tested' in results, "Results should contain models_tested"
        assert 'training_time' in results, "Results should contain training_time"
        
        # Test predictions
        best_model = results['best_model']
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Classification Test Results:")
        logger.info(f"  Best Score: {results['best_score']:.4f}")
        logger.info(f"  Models Tested: {results['models_tested']}")
        logger.info(f"  Training Time: {results['training_time']:.2f} seconds")
        logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"  Total Time: {end_time - start_time:.2f} seconds")
        
        # Test model summary
        summary = automl.get_model_summary(results)
        assert summary is not None, "Model summary should not be None"
        assert 'best_model_type' in summary, "Summary should contain best_model_type"
        assert 'best_score' in summary, "Summary should contain best_score"
        
        logger.info(f"  Best Model Type: {summary['best_model_type']}")
        logger.info(f"  Ensemble Method: {results['ensemble_method']}")
        
        # Test model saving and loading
        model_path = "test_classification_model.pkl"
        save_success = automl.save_model(best_model, model_path)
        assert save_success, "Model saving should succeed"
        
        loaded_model = automl.load_model(model_path)
        assert loaded_model is not None, "Model loading should succeed"
        
        # Test loaded model predictions
        y_pred_loaded = loaded_model.predict(X_test)
        loaded_accuracy = accuracy_score(y_test, y_pred_loaded)
        assert abs(test_accuracy - loaded_accuracy) < 1e-6, "Loaded model should give same predictions"
        
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
        
        logger.info("✅ Classification test passed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Classification test failed: {str(e)}")
        return False


def test_enhanced_automl_regression():
    """Test enhanced AutoML pipeline for regression tasks."""
    logger.info("Testing Enhanced AutoML Pipeline - Regression")
    
    try:
        # Generate synthetic regression dataset
        X, y = make_regression(
            n_samples=1000, n_features=20, n_informative=15,
            noise=0.1, random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize enhanced AutoML pipeline
        automl = EnhancedAutoMLPipeline()
        
        # Run AutoML pipeline
        start_time = time.time()
        results = automl.run_automl(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task_type='regression',
            max_models=5,
            cv_folds=3,
            optimization_strategy='random_search',
            ensemble_method='stacking'
        )
        end_time = time.time()
        
        # Validate results
        assert results is not None, "AutoML results should not be None"
        assert 'best_model' in results, "Results should contain best_model"
        assert 'best_score' in results, "Results should contain best_score"
        assert 'models_tested' in results, "Results should contain models_tested"
        assert 'training_time' in results, "Results should contain training_time"
        
        # Test predictions
        best_model = results['best_model']
        y_pred = best_model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Regression Test Results:")
        logger.info(f"  Best Score: {results['best_score']:.4f}")
        logger.info(f"  Models Tested: {results['models_tested']}")
        logger.info(f"  Training Time: {results['training_time']:.2f} seconds")
        logger.info(f"  Test R² Score: {test_r2:.4f}")
        logger.info(f"  Total Time: {end_time - start_time:.2f} seconds")
        
        # Test model summary
        summary = automl.get_model_summary(results)
        assert summary is not None, "Model summary should not be None"
        assert 'best_model_type' in summary, "Summary should contain best_model_type"
        assert 'best_score' in summary, "Summary should contain best_score"
        
        logger.info(f"  Best Model Type: {summary['best_model_type']}")
        logger.info(f"  Ensemble Method: {results['ensemble_method']}")
        
        # Test feature importance
        if results.get('metadata', {}).get('feature_importance'):
            feature_importance = results['metadata']['feature_importance']
            logger.info(f"  Feature Importance Available: {len(feature_importance.get('scores', []))} features")
        
        # Test model saving and loading
        model_path = "test_regression_model.pkl"
        save_success = automl.save_model(best_model, model_path)
        assert save_success, "Model saving should succeed"
        
        loaded_model = automl.load_model(model_path)
        assert loaded_model is not None, "Model loading should succeed"
        
        # Test loaded model predictions
        y_pred_loaded = loaded_model.predict(X_test)
        loaded_r2 = r2_score(y_test, y_pred_loaded)
        assert abs(test_r2 - loaded_r2) < 1e-6, "Loaded model should give same predictions"
        
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
        
        logger.info("✅ Regression test passed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Regression test failed: {str(e)}")
        return False


def test_enhanced_automl_auto_detection():
    """Test enhanced AutoML pipeline with automatic task detection."""
    logger.info("Testing Enhanced AutoML Pipeline - Auto Detection")
    
    try:
        # Test classification auto-detection
        X_clf, y_clf = make_classification(
            n_samples=500, n_features=10, n_informative=8,
            n_redundant=2, n_classes=2, random_state=42
        )
        
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
            X_clf, y_clf, test_size=0.2, random_state=42
        )
        
        automl = EnhancedAutoMLPipeline()
        
        # Run with auto task detection
        results_clf = automl.run_automl(
            X_train=X_train_clf,
            y_train=y_train_clf,
            X_test=X_test_clf,
            y_test=y_test_clf,
            task_type='auto',
            max_models=3,
            cv_folds=3,
            ensemble_method='blending'
        )
        
        assert results_clf['task_type'] == 'classification', "Should detect classification task"
        logger.info(f"✅ Auto-detected task type: {results_clf['task_type']}")
        
        # Test regression auto-detection
        X_reg, y_reg = make_regression(
            n_samples=500, n_features=10, n_informative=8,
            noise=0.1, random_state=42
        )
        
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        
        results_reg = automl.run_automl(
            X_train=X_train_reg,
            y_train=y_train_reg,
            X_test=X_test_reg,
            y_test=y_test_reg,
            task_type='auto',
            max_models=3,
            cv_folds=3,
            ensemble_method='blending'
        )
        
        assert results_reg['task_type'] == 'regression', "Should detect regression task"
        logger.info(f"✅ Auto-detected task type: {results_reg['task_type']}")
        
        logger.info("✅ Auto detection test passed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Auto detection test failed: {str(e)}")
        return False


def test_enhanced_automl_hyperparameter_optimization():
    """Test enhanced hyperparameter optimization capabilities."""
    logger.info("Testing Enhanced Hyperparameter Optimization")
    
    try:
        # Generate synthetic dataset
        X, y = make_classification(
            n_samples=500, n_features=10, n_informative=8,
            n_redundant=2, n_classes=2, random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        automl = EnhancedAutoMLPipeline()
        
        # Test grid search optimization
        from sklearn.ensemble import RandomForestClassifier
        
        param_space = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }
        
        model = RandomForestClassifier()
        
        # Test grid search
        grid_results = automl.optimize_hyperparameters(
            model=model,
            X_train=X_train,
            y_train=y_train,
            param_space=param_space,
            optimization_strategy='grid_search',
            cv_folds=3
        )
        
        assert grid_results is not None, "Grid search results should not be None"
        assert 'optimized_model' in grid_results, "Should contain optimized model"
        assert 'best_params' in grid_results, "Should contain best parameters"
        assert 'best_score' in grid_results, "Should contain best score"
        
        logger.info(f"  Grid Search Best Score: {grid_results['best_score']:.4f}")
        logger.info(f"  Grid Search Best Params: {grid_results['best_params']}")
        
        # Test random search optimization
        random_results = automl.optimize_hyperparameters(
            model=model,
            X_train=X_train,
            y_train=y_train,
            param_space=param_space,
            optimization_strategy='random_search',
            cv_folds=3
        )
        
        assert random_results is not None, "Random search results should not be None"
        assert 'optimized_model' in random_results, "Should contain optimized model"
        assert 'best_params' in random_results, "Should contain best parameters"
        assert 'best_score' in random_results, "Should contain best score"
        
        logger.info(f"  Random Search Best Score: {random_results['best_score']:.4f}")
        logger.info(f"  Random Search Best Params: {random_results['best_params']}")
        
        logger.info("✅ Hyperparameter optimization test passed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Hyperparameter optimization test failed: {str(e)}")
        return False


def test_enhanced_automl_feature_engineering():
    """Test enhanced feature engineering capabilities."""
    logger.info("Testing Enhanced Feature Engineering")
    
    try:
        # Generate dataset with many features
        X, y = make_classification(
            n_samples=1000, n_features=50, n_informative=20,
            n_redundant=20, n_repeated=10, n_classes=2, random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        automl = EnhancedAutoMLPipeline()
        
        # Run AutoML with feature engineering
        results = automl.run_automl(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task_type='classification',
            max_models=3,
            cv_folds=3,
            ensemble_method='voting'
        )
        
        # Check if feature engineering was applied
        if results.get('feature_selector') is not None:
            feature_selector = results['feature_selector']
            original_features = X_train.shape[1]
            selected_features = feature_selector.get_support().sum()
            
            logger.info(f"  Original Features: {original_features}")
            logger.info(f"  Selected Features: {selected_features}")
            logger.info(f"  Feature Reduction: {((original_features - selected_features) / original_features * 100):.1f}%")
            
            # Check feature importance
            if results.get('metadata', {}).get('feature_importance'):
                feature_importance = results['metadata']['feature_importance']
                logger.info(f"  Feature Importance Scores Available: {len(feature_importance.get('scores', []))}")
        
        logger.info("✅ Feature engineering test passed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature engineering test failed: {str(e)}")
        return False


def main():
    """Run all Phase 9 Enhanced AutoML tests."""
    logger.info("🚀 Starting Phase 9: Enhanced AutoML Pipeline Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Enhanced AutoML Classification", test_enhanced_automl_classification),
        ("Enhanced AutoML Regression", test_enhanced_automl_regression),
        ("Enhanced AutoML Auto Detection", test_enhanced_automl_auto_detection),
        ("Enhanced Hyperparameter Optimization", test_enhanced_automl_hyperparameter_optimization),
        ("Enhanced Feature Engineering", test_enhanced_automl_feature_engineering)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {str(e)}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Phase 9 Enhanced AutoML tests passed successfully!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
