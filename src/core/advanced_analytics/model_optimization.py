"""
Model Optimization

Advanced model optimization capabilities including hyperparameter tuning,
feature engineering, and model selection for improved predictive performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import warnings

# Conditional imports
try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Model optimization features limited.")

try:
    from scipy.stats import uniform, randint
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Random search limited.")

# Local imports
from ..error_handler import ErrorHandler
from ..caching_service import CachingService
from ..performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Data class for optimization results"""
    best_model: Any
    best_params: Dict[str, Any]
    best_score: float
    optimization_method: str
    cv_scores: List[float]
    feature_importance: Optional[Dict[str, float]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ModelComparisonResult:
    """Data class for model comparison results"""
    model_results: Dict[str, OptimizationResult]
    comparison_metrics: pd.DataFrame
    best_model_name: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ModelOptimizer:
    """
    Advanced model optimization for hyperparameter tuning and model selection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the model optimizer"""
        self.config = config or {}
        self.error_handler = ErrorHandler()
        self.caching_service = CachingService()
        self.performance_monitor = PerformanceMonitor()
        
        # Configuration
        self.cv_folds = self.config.get('cv_folds', 5)
        self.scoring = self.config.get('scoring', 'neg_mean_squared_error')
        self.n_iter = self.config.get('n_iter', 100)
        
        # Storage
        self.optimization_history = []
        self.best_models = {}
        
        logger.info("ModelOptimizer initialized")
    
    def optimize_hyperparameters(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List[Any]],
        method: str = 'grid_search',
        cv: Optional[int] = None
    ) -> OptimizationResult:
        """
        Optimize hyperparameters using grid search or random search
        
        Args:
            model: Base model to optimize
            X: Feature matrix
            y: Target variable
            param_grid: Parameter grid for optimization
            method: Optimization method ('grid_search', 'random_search')
            cv: Number of cross-validation folds
            
        Returns:
            OptimizationResult object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for hyperparameter optimization")
        
        try:
            with self.performance_monitor.track_operation("hyperparameter_optimization"):
                if cv is None:
                    cv = self.cv_folds
                
                if method == 'grid_search':
                    # Grid search optimization
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid,
                        cv=cv,
                        scoring=self.scoring,
                        n_jobs=-1,
                        verbose=0
                    )
                    
                    grid_search.fit(X, y)
                    
                    result = OptimizationResult(
                        best_model=grid_search.best_estimator_,
                        best_params=grid_search.best_params_,
                        best_score=grid_search.best_score_,
                        optimization_method="grid_search",
                        cv_scores=grid_search.cv_results_['mean_test_score'].tolist()
                    )
                    
                elif method == 'random_search':
                    # Random search optimization
                    if not SCIPY_AVAILABLE:
                        raise ImportError("scipy required for random search")
                    
                    random_search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_grid,
                        n_iter=self.n_iter,
                        cv=cv,
                        scoring=self.scoring,
                        n_jobs=-1,
                        verbose=0,
                        random_state=42
                    )
                    
                    random_search.fit(X, y)
                    
                    result = OptimizationResult(
                        best_model=random_search.best_estimator_,
                        best_params=random_search.best_params_,
                        best_score=random_search.best_score_,
                        optimization_method="random_search",
                        cv_scores=random_search.cv_results_['mean_test_score'].tolist()
                    )
                    
                else:
                    raise ValueError(f"Unsupported optimization method: {method}")
                
                # Add feature importance if available
                if hasattr(result.best_model, 'feature_importances_'):
                    result.feature_importance = dict(zip(X.columns, result.best_model.feature_importances_))
                elif hasattr(result.best_model, 'coef_'):
                    result.feature_importance = dict(zip(X.columns, abs(result.best_model.coef_)))
                
                # Store in history
                self.optimization_history.append(result)
                
                logger.info(f"Completed {method} optimization with best score: {result.best_score}")
                return result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in hyperparameter optimization: {str(e)}", e)
            raise
    
    def compare_models(
        self,
        models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        cv: Optional[int] = None
    ) -> ModelComparisonResult:
        """
        Compare multiple models using cross-validation
        
        Args:
            models: Dictionary of models to compare
            X: Feature matrix
            y: Target variable
            cv: Number of cross-validation folds
            
        Returns:
            ModelComparisonResult object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for model comparison")
        
        try:
            with self.performance_monitor.track_operation("model_comparison"):
                if cv is None:
                    cv = self.cv_folds
                
                model_results = {}
                comparison_data = []
                
                for model_name, model in models.items():
                    try:
                        # Perform cross-validation
                        cv_scores = cross_val_score(
                            model, X, y, cv=cv, scoring=self.scoring, n_jobs=-1
                        )
                        
                        # Calculate metrics
                        mean_score = cv_scores.mean()
                        std_score = cv_scores.std()
                        
                        # Create optimization result
                        result = OptimizationResult(
                            best_model=model,
                            best_params={},
                            best_score=mean_score,
                            optimization_method="cross_validation",
                            cv_scores=cv_scores.tolist()
                        )
                        
                        model_results[model_name] = result
                        
                        # Add to comparison data
                        comparison_data.append({
                            'model': model_name,
                            'mean_score': mean_score,
                            'std_score': std_score,
                            'min_score': cv_scores.min(),
                            'max_score': cv_scores.max()
                        })
                        
                    except Exception as e:
                        logger.warning(f"Model {model_name} failed: {str(e)}")
                        continue
                
                # Create comparison DataFrame
                comparison_df = pd.DataFrame(comparison_data)
                
                # Find best model
                best_model_name = comparison_df.loc[
                    comparison_df['mean_score'].idxmax(), 'model'
                ]
                
                result = ModelComparisonResult(
                    model_results=model_results,
                    comparison_metrics=comparison_df,
                    best_model_name=best_model_name
                )
                
                logger.info(f"Completed model comparison. Best model: {best_model_name}")
                return result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in model comparison: {str(e)}", e)
            raise
    
    def optimize_feature_selection(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'recursive',
        max_features: Optional[int] = None
    ) -> OptimizationResult:
        """
        Optimize feature selection
        
        Args:
            model: Base model
            X: Feature matrix
            y: Target variable
            method: Feature selection method
            max_features: Maximum number of features to select
            
        Returns:
            OptimizationResult object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for feature selection")
        
        try:
            with self.performance_monitor.track_operation("feature_selection_optimization"):
                if max_features is None:
                    max_features = min(20, X.shape[1])
                
                if method == 'recursive':
                    from sklearn.feature_selection import RFECV
                    
                    # Recursive feature elimination with cross-validation
                    rfecv = RFECV(
                        estimator=model,
                        step=1,
                        cv=self.cv_folds,
                        scoring=self.scoring,
                        min_features_to_select=1,
                        n_jobs=-1
                    )
                    
                    rfecv.fit(X, y)
                    
                    # Get selected features
                    selected_features = X.columns[rfecv.support_].tolist()
                    X_selected = X[selected_features]
                    
                    # Train model on selected features
                    model_selected = model.__class__(**model.get_params())
                    cv_scores = cross_val_score(
                        model_selected, X_selected, y, cv=self.cv_folds, scoring=self.scoring
                    )
                    
                    result = OptimizationResult(
                        best_model=model_selected,
                        best_params={'selected_features': selected_features},
                        best_score=cv_scores.mean(),
                        optimization_method="recursive_feature_selection",
                        cv_scores=cv_scores.tolist()
                    )
                    
                elif method == 'mutual_info':
                    from sklearn.feature_selection import SelectKBest, mutual_info_regression
                    
                    # Mutual information feature selection
                    selector = SelectKBest(score_func=mutual_info_regression, k=max_features)
                    X_selected = selector.fit_transform(X, y)
                    
                    # Get selected feature names
                    selected_features = X.columns[selector.get_support()].tolist()
                    
                    # Train model on selected features
                    model_selected = model.__class__(**model.get_params())
                    cv_scores = cross_val_score(
                        model_selected, X_selected, y, cv=self.cv_folds, scoring=self.scoring
                    )
                    
                    result = OptimizationResult(
                        best_model=model_selected,
                        best_params={'selected_features': selected_features},
                        best_score=cv_scores.mean(),
                        optimization_method="mutual_info_feature_selection",
                        cv_scores=cv_scores.tolist()
                    )
                    
                else:
                    raise ValueError(f"Unsupported feature selection method: {method}")
                
                # Store in history
                self.optimization_history.append(result)
                
                logger.info(f"Completed feature selection optimization with {len(result.best_params.get('selected_features', []))} features")
                return result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in feature selection optimization: {str(e)}", e)
            raise
    
    def optimize_ensemble(
        self,
        base_models: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        ensemble_method: str = 'voting'
    ) -> OptimizationResult:
        """
        Optimize ensemble model
        
        Args:
            base_models: Dictionary of base models
            X: Feature matrix
            y: Target variable
            ensemble_method: Ensemble method ('voting', 'stacking')
            
        Returns:
            OptimizationResult object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for ensemble optimization")
        
        try:
            with self.performance_monitor.track_operation("ensemble_optimization"):
                if ensemble_method == 'voting':
                    from sklearn.ensemble import VotingRegressor
                    
                    # Create voting ensemble
                    estimators = [(name, model) for name, model in base_models.items()]
                    ensemble = VotingRegressor(estimators=estimators)
                    
                    # Evaluate ensemble
                    cv_scores = cross_val_score(
                        ensemble, X, y, cv=self.cv_folds, scoring=self.scoring
                    )
                    
                    result = OptimizationResult(
                        best_model=ensemble,
                        best_params={'ensemble_method': 'voting', 'base_models': list(base_models.keys())},
                        best_score=cv_scores.mean(),
                        optimization_method="ensemble_voting",
                        cv_scores=cv_scores.tolist()
                    )
                    
                elif ensemble_method == 'stacking':
                    from sklearn.ensemble import StackingRegressor
                    
                    # Create stacking ensemble
                    estimators = [(name, model) for name, model in base_models.items()]
                    final_estimator = LinearRegression()
                    
                    ensemble = StackingRegressor(
                        estimators=estimators,
                        final_estimator=final_estimator,
                        cv=self.cv_folds
                    )
                    
                    # Evaluate ensemble
                    cv_scores = cross_val_score(
                        ensemble, X, y, cv=self.cv_folds, scoring=self.scoring
                    )
                    
                    result = OptimizationResult(
                        best_model=ensemble,
                        best_params={'ensemble_method': 'stacking', 'base_models': list(base_models.keys())},
                        best_score=cv_scores.mean(),
                        optimization_method="ensemble_stacking",
                        cv_scores=cv_scores.tolist()
                    )
                    
                else:
                    raise ValueError(f"Unsupported ensemble method: {ensemble_method}")
                
                # Store in history
                self.optimization_history.append(result)
                
                logger.info(f"Completed ensemble optimization with {len(base_models)} base models")
                return result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in ensemble optimization: {str(e)}", e)
            raise
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results"""
        if not self.optimization_history:
            return {"message": "No optimization results found"}
        
        summary = {
            'total_optimizations': len(self.optimization_history),
            'methods_used': list(set(result.optimization_method for result in self.optimization_history)),
            'best_scores': [],
            'average_scores': []
        }
        
        for result in self.optimization_history:
            summary['best_scores'].append(result.best_score)
            if result.cv_scores:
                summary['average_scores'].append(np.mean(result.cv_scores))
        
        if summary['best_scores']:
            summary['score_statistics'] = {
                'best_score': max(summary['best_scores']),
                'worst_score': min(summary['best_scores']),
                'mean_score': np.mean(summary['best_scores']),
                'std_score': np.std(summary['best_scores'])
            }
        
        return summary
    
    def export_optimization_results(self, filepath: str) -> None:
        """Export optimization results"""
        try:
            import json
            from datetime import datetime
            
            export_data = {
                'optimization_history': [
                    {
                        'optimization_method': result.optimization_method,
                        'best_score': result.best_score,
                        'best_params': result.best_params,
                        'timestamp': result.timestamp.isoformat() if result.timestamp else None,
                        'cv_scores_mean': np.mean(result.cv_scores) if result.cv_scores else None,
                        'cv_scores_std': np.std(result.cv_scores) if result.cv_scores else None
                    }
                    for result in self.optimization_history
                ],
                'config': self.config
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Optimization results exported to {filepath}")
            
        except Exception as e:
            self.error_handler.handle_error(f"Error exporting optimization results: {str(e)}", e)
