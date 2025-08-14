"""
Enhanced AutoML Pipeline

This module provides comprehensive automated machine learning capabilities including
model selection, hyperparameter optimization, feature engineering, ensemble methods,
and automated deployment.
"""

import logging
import time
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.core.error_handling_service import ErrorHandlingService
from src.config.advanced_ml_config import get_advanced_ml_config

logger = logging.getLogger(__name__)
error_handler = ErrorHandlingService()


class EnhancedAutoMLPipeline:
    """Enhanced automated machine learning pipeline with comprehensive capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_advanced_ml_config()
        self.models_tested = []
        self.best_model = None
        self.best_score = 0.0
        self.best_params = {}
        self.feature_importance = {}
        self.training_history = []
        self.model_metadata = {}
        
        # Initialize algorithm registry
        self._initialize_algorithm_registry()
        
        # Initialize preprocessing components
        self._initialize_preprocessors()
        
        logger.info("Initialized Enhanced AutoML Pipeline")
    
    def _initialize_algorithm_registry(self):
        """Initialize the algorithm registry with supported models."""
        self.classification_models = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'logistic_regression': LogisticRegression,
            'svm': SVC,
            'knn': KNeighborsClassifier,
            'decision_tree': DecisionTreeClassifier,
            'naive_bayes': GaussianNB,
            'neural_network': MLPClassifier
        }
        
        self.regression_models = {
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor,
            'linear_regression': LinearRegression,
            'ridge': Ridge,
            'lasso': Lasso,
            'svm': SVR,
            'knn': KNeighborsRegressor,
            'decision_tree': DecisionTreeRegressor,
            'neural_network': MLPRegressor
        }
        
        # Hyperparameter spaces for optimization
        self.hyperparameter_spaces = {
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
    
    def _initialize_preprocessors(self):
        """Initialize preprocessing components."""
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        self.imputers = {
            'mean': SimpleImputer(strategy='mean'),
            'median': SimpleImputer(strategy='median'),
            'most_frequent': SimpleImputer(strategy='most_frequent')
        }
        
        self.feature_selectors = {
            'kbest': SelectKBest,
            'rfe': RFE,
            'pca': PCA
        }
    
    def run_automl(self, X_train: Union[np.ndarray, pd.DataFrame], 
                   y_train: Union[np.ndarray, pd.Series],
                   X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
                   y_test: Optional[Union[np.ndarray, pd.Series]] = None,
                   task_type: str = 'auto',
                   max_models: int = 10,
                   cv_folds: int = 5,
                   optimization_strategy: str = 'grid_search',
                   ensemble_method: str = 'voting') -> Dict[str, Any]:
        """
        Run comprehensive automated machine learning pipeline.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features (optional)
            y_test: Test targets (optional)
            task_type: 'classification' or 'regression' or 'auto'
            max_models: Maximum number of models to test
            cv_folds: Number of cross-validation folds
            optimization_strategy: 'grid_search' or 'random_search'
            ensemble_method: 'voting', 'stacking', or 'blending'
        
        Returns:
            Dictionary containing results and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting Enhanced AutoML pipeline for {task_type} task")
            
            # Determine task type automatically if not specified
            if task_type == 'auto':
                task_type = self._detect_task_type(y_train)
            
            # Data preprocessing
            X_train_processed, X_test_processed, preprocessor = self._preprocess_data(
                X_train, X_test, task_type
            )
            
            # Feature engineering
            X_train_engineered, X_test_engineered, feature_selector = self._engineer_features(
                X_train_processed, y_train, X_test_processed, task_type
            )
            
            # Model selection and optimization
            models_results = self._select_and_optimize_models(
                X_train_engineered, y_train, task_type, max_models, cv_folds, optimization_strategy
            )
            
            # Ensemble creation
            ensemble_model = self._create_ensemble(
                models_results, X_train_engineered, y_train, ensemble_method
            )
            
            # Final evaluation - use the same feature set as the ensemble was trained on
            final_results = self._evaluate_final_model(
                ensemble_model, X_train_engineered, y_train, X_test_engineered, y_test, task_type
            )
            
            # Compile results
            results = {
                'best_model': ensemble_model,
                'best_score': final_results['score'],
                'models_tested': len(models_results),
                'training_time': time.time() - start_time,
                'task_type': task_type,
                'preprocessor': preprocessor,
                'feature_selector': feature_selector,
                'models_results': models_results,
                'ensemble_method': ensemble_method,
                'final_results': final_results,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'config': self.config,
                    'feature_importance': self.feature_importance,
                    'training_history': self.training_history
                }
            }
            
            logger.info(f"AutoML pipeline completed in {results['training_time']:.2f} seconds")
            return results
            
        except Exception as e:
            # Create a simple error context for non-agent components
            from src.core.error_handling_service import ErrorContext
            context = ErrorContext(agent_id="automl_pipeline", operation="run_automl")
            error_handler.handle_error(e, context)
            return {}
    
    def _detect_task_type(self, y: Union[np.ndarray, pd.Series]) -> str:
        """Automatically detect if the task is classification or regression."""
        if hasattr(y, 'dtype'):
            if y.dtype in ['object', 'string', 'category'] or len(np.unique(y)) < 20:
                return 'classification'
            else:
                return 'regression'
        else:
            if len(np.unique(y)) < 20:
                return 'classification'
            else:
                return 'regression'
    
    def _preprocess_data(self, X_train: Union[np.ndarray, pd.DataFrame],
                        X_test: Optional[Union[np.ndarray, pd.DataFrame]],
                        task_type: str) -> Tuple[np.ndarray, Optional[np.ndarray], Any]:
        """Preprocess the data including scaling and imputation."""
        logger.info("Preprocessing data")
        
        # Convert to numpy arrays if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if X_test is not None and isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        
        # Handle missing values
        imputer = self.imputers['median']
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test) if X_test is not None else None
        
        # Scale features
        scaler = self.scalers['standard']
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed) if X_test_imputed is not None else None
        
        # Create preprocessor pipeline
        preprocessor = Pipeline([
            ('imputer', imputer),
            ('scaler', scaler)
        ])
        
        return X_train_scaled, X_test_scaled, preprocessor
    
    def _engineer_features(self, X_train: np.ndarray, y_train: Union[np.ndarray, pd.Series],
                          X_test: Optional[np.ndarray], task_type: str) -> Tuple[np.ndarray, Optional[np.ndarray], Any]:
        """Engineer features using automated feature selection."""
        logger.info("Engineering features")
        
        # Feature selection
        if X_train.shape[1] > 10:  # Only apply feature selection if we have many features
            if task_type == 'classification':
                selector = SelectKBest(score_func=f_classif, k=min(10, X_train.shape[1]))
            else:
                selector = SelectKBest(score_func=f_regression, k=min(10, X_train.shape[1]))
            
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test) if X_test is not None else None
            
            # Store feature importance
            if hasattr(selector, 'scores_'):
                self.feature_importance = {
                    'scores': selector.scores_,
                    'pvalues': selector.pvalues_,
                    'selected_features': selector.get_support()
                }
            
            return X_train_selected, X_test_selected, selector
        else:
            return X_train, X_test, None
    
    def _select_and_optimize_models(self, X_train: np.ndarray, y_train: Union[np.ndarray, pd.Series],
                                  task_type: str, max_models: int, cv_folds: int,
                                  optimization_strategy: str) -> List[Dict[str, Any]]:
        """Select and optimize models using the specified strategy."""
        logger.info(f"Selecting and optimizing {max_models} models")
        
        models = self.classification_models if task_type == 'classification' else self.regression_models
        results = []
        
        for i, (model_name, model_class) in enumerate(models.items()):
            if i >= max_models:
                break
            
            try:
                logger.info(f"Testing {model_name}")
                
                # Get hyperparameter space
                param_space = self.hyperparameter_spaces.get(model_name, {})
                
                if param_space:
                    # Optimize hyperparameters
                    if optimization_strategy == 'grid_search':
                        search = GridSearchCV(
                            model_class(), param_space, cv=cv_folds, 
                            scoring=self._get_scoring_metric(task_type), n_jobs=-1
                        )
                    else:  # random_search
                        search = RandomizedSearchCV(
                            model_class(), param_space, cv=cv_folds,
                            scoring=self._get_scoring_metric(task_type), n_jobs=-1, n_iter=20
                        )
                    
                    search.fit(X_train, y_train)
                    best_model = search.best_estimator_
                    best_score = search.best_score_
                    best_params = search.best_params_
                else:
                    # Use default parameters
                    model = model_class()
                    scores = cross_val_score(model, X_train, y_train, cv=cv_folds, 
                                           scoring=self._get_scoring_metric(task_type))
                    best_model = model.fit(X_train, y_train)
                    best_score = scores.mean()
                    best_params = {}
                
                results.append({
                    'model_name': model_name,
                    'model': best_model,
                    'score': best_score,
                    'params': best_params,
                    'cv_scores': scores if 'scores' in locals() else None
                })
                
                self.models_tested.append(model_name)
                self.training_history.append({
                    'model': model_name,
                    'score': best_score,
                    'params': best_params,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.warning(f"Failed to optimize {model_name}: {str(e)}")
                continue
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def _get_scoring_metric(self, task_type: str) -> str:
        """Get appropriate scoring metric for the task type."""
        if task_type == 'classification':
            return 'accuracy'
        else:
            return 'r2'
    
    def _create_ensemble(self, models_results: List[Dict[str, Any]], 
                        X_train: np.ndarray, y_train: Union[np.ndarray, pd.Series],
                        ensemble_method: str) -> Any:
        """Create ensemble model using the specified method."""
        logger.info(f"Creating ensemble using {ensemble_method}")
        
        if ensemble_method == 'voting':
            from sklearn.ensemble import VotingClassifier, VotingRegressor
            
            models = [(result['model_name'], result['model']) for result in models_results[:3]]
            
            if self._detect_task_type(y_train) == 'classification':
                ensemble = VotingClassifier(estimators=models, voting='soft')
            else:
                ensemble = VotingRegressor(estimators=models)
        
        elif ensemble_method == 'stacking':
            from sklearn.ensemble import StackingClassifier, StackingRegressor
            
            # For stacking, we need to ensure all models use the same feature set
            # Use only the best model to avoid feature dimension issues
            best_model_result = models_results[0]
            models = [(best_model_result['model_name'], best_model_result['model'])]
            
            if self._detect_task_type(y_train) == 'classification':
                # Check if the model supports predict_proba
                stack_method = 'predict'
                if hasattr(best_model_result['model'], 'predict_proba'):
                    try:
                        # Test if predict_proba works
                        best_model_result['model'].predict_proba(X_train[:10])
                        stack_method = 'predict_proba'
                    except:
                        stack_method = 'predict'
                
                ensemble = StackingClassifier(
                    estimators=models,
                    final_estimator=LogisticRegression(),
                    cv=5,
                    stack_method=stack_method
                )
            else:
                ensemble = StackingRegressor(
                    estimators=models,
                    final_estimator=LinearRegression(),
                    cv=5
                )
        
        else:  # blending - simple average
            # For blending, we'll use the best model
            ensemble = models_results[0]['model']
        
        # Fit ensemble
        ensemble.fit(X_train, y_train)
        
        return ensemble
    
    def _evaluate_final_model(self, model: Any, X_train: np.ndarray, y_train: Union[np.ndarray, pd.Series],
                             X_test: Optional[np.ndarray], y_test: Optional[Union[np.ndarray, pd.Series]],
                             task_type: str) -> Dict[str, Any]:
        """Evaluate the final model and return comprehensive metrics."""
        logger.info("Evaluating final model")
        
        # Train score
        train_score = self._calculate_score(model, X_train, y_train, task_type)
        
        # Test score (if available)
        test_score = None
        if X_test is not None and y_test is not None:
            test_score = self._calculate_score(model, X_test, y_test, task_type)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                   scoring=self._get_scoring_metric(task_type))
        
        results = {
            'train_score': train_score,
            'test_score': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'score': test_score if test_score is not None else train_score
        }
        
        # Store best model info
        if results['score'] > self.best_score:
            self.best_score = results['score']
            self.best_model = model
        
        return results
    
    def _calculate_score(self, model: Any, X: np.ndarray, y: Union[np.ndarray, pd.Series], 
                        task_type: str) -> float:
        """Calculate appropriate score for the model."""
        y_pred = model.predict(X)
        
        if task_type == 'classification':
            return accuracy_score(y, y_pred)
        else:
            return r2_score(y, y_pred)
    
    def get_best_model(self, automl_result: Dict[str, Any]) -> Any:
        """Get the best model from AutoML results."""
        try:
            return automl_result.get('best_model', self.best_model)
        except Exception as e:
            from src.core.error_handling_service import ErrorContext
            context = ErrorContext(agent_id="automl_pipeline", operation="get_best_model")
            error_handler.handle_error(e, context)
            return None
    
    def optimize_hyperparameters(self, model: Any, X_train: np.ndarray,
                               y_train: np.ndarray,
                               param_space: Dict[str, Any],
                               optimization_strategy: str = 'grid_search',
                               cv_folds: int = 5) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific model."""
        try:
            logger.info("Optimizing hyperparameters")
            
            if optimization_strategy == 'grid_search':
                search = GridSearchCV(model, param_space, cv=cv_folds, n_jobs=-1)
            else:
                search = RandomizedSearchCV(model, param_space, cv=cv_folds, n_jobs=-1, n_iter=20)
            
            search.fit(X_train, y_train)
            
            return {
                'optimized_model': search.best_estimator_,
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_
            }
            
        except Exception as e:
            from src.core.error_handling_service import ErrorContext
            context = ErrorContext(agent_id="automl_pipeline", operation="optimize_hyperparameters")
            error_handler.handle_error(e, context)
            return {}
    
    def save_model(self, model: Any, filepath: str) -> bool:
        """Save the trained model to disk."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            from src.core.error_handling_service import ErrorContext
            context = ErrorContext(agent_id="automl_pipeline", operation="save_model")
            error_handler.handle_error(e, context)
            return False
    
    def load_model(self, filepath: str) -> Any:
        """Load a trained model from disk."""
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            from src.core.error_handling_service import ErrorContext
            context = ErrorContext(agent_id="automl_pipeline", operation="load_model")
            error_handler.handle_error(e, context)
            return None
    
    def get_model_summary(self, automl_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get a comprehensive summary of the AutoML results."""
        try:
            return {
                'best_model_type': type(automl_result.get('best_model')).__name__,
                'best_score': automl_result.get('best_score', 0),
                'models_tested': automl_result.get('models_tested', 0),
                'training_time': automl_result.get('training_time', 0),
                'task_type': automl_result.get('task_type', 'unknown'),
                'ensemble_method': automl_result.get('ensemble_method', 'none'),
                'feature_importance': automl_result.get('metadata', {}).get('feature_importance', {}),
                'models_ranking': [
                    {
                        'model': result['model_name'],
                        'score': result['score'],
                        'params': result['params']
                    }
                    for result in automl_result.get('models_results', [])
                ]
            }
        except Exception as e:
            from src.core.error_handling_service import ErrorContext
            context = ErrorContext(agent_id="automl_pipeline", operation="get_model_summary")
            error_handler.handle_error(e, context)
            return {}


# Backward compatibility
AutoMLPipeline = EnhancedAutoMLPipeline
