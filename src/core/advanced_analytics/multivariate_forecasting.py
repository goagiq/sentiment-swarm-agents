"""
Multivariate Forecasting Engine

Advanced multivariate forecasting capabilities for handling multiple correlated variables
with sophisticated time series analysis and causal inference integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

# Conditional imports for advanced ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.base import clone
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some multivariate forecasting "
                 "features will be limited.")

try:
    from statsmodels.tsa.vector_ar.var_model import VAR
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. VAR models will not be supported.")

# Local imports
from ..error_handling_service import ErrorHandlingService
from ..caching_service import CachingService
from ..performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class MultivariateForecastResult:
    """Result container for multivariate forecasting"""
    predictions: pd.DataFrame
    confidence_intervals: Dict[str, Tuple[pd.Series, pd.Series]]
    model_performance: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    causal_relationships: Optional[Dict[str, List[str]]] = None
    forecast_horizon: int = 12
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class MultivariateForecastingEngine:
    """
    Advanced multivariate forecasting engine supporting multiple correlated variables
    with sophisticated time series analysis and causal inference.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the multivariate forecasting engine"""
        self.config = config or {}
        self.error_handler = ErrorHandlingService()
        self.caching_service = CachingService()
        self.performance_monitor = PerformanceMonitor()
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Performance tracking
        self.forecast_history = []
        self.model_performance_history = []
        
        logger.info("MultivariateForecastingEngine initialized")
    
    def prepare_multivariate_data(
        self, 
        data: pd.DataFrame, 
        target_columns: List[str],
        feature_columns: Optional[List[str]] = None,
        lag_features: bool = True,
        max_lags: int = 12
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare multivariate data for forecasting with feature engineering
        
        Args:
            data: Input DataFrame with time series data
            target_columns: Columns to forecast
            feature_columns: Additional feature columns
            lag_features: Whether to create lag features
            max_lags: Maximum number of lag features to create
            
        Returns:
            Tuple of (features, targets) DataFrames
        """
        try:
            with self.performance_monitor.track_operation("multivariate_data_preparation"):
                # Ensure data is sorted by time
                if 'timestamp' in data.columns:
                    data = data.sort_values('timestamp')
                elif data.index.dtype == 'datetime64[ns]':
                    data = data.sort_index()
                
                # Create feature DataFrame
                features = pd.DataFrame(index=data.index)
                
                # Add original features
                if feature_columns:
                    features = pd.concat([features, data[feature_columns]], axis=1)
                
                # Create lag features for target variables
                if lag_features:
                    for col in target_columns:
                        for lag in range(1, max_lags + 1):
                            features[f'{col}_lag_{lag}'] = data[col].shift(lag)
                
                # Create rolling statistics
                for col in target_columns:
                    features[f'{col}_rolling_mean_7'] = data[col].rolling(7).mean()
                    features[f'{col}_rolling_std_7'] = data[col].rolling(7).std()
                    features[f'{col}_rolling_mean_30'] = data[col].rolling(30).mean()
                    features[f'{col}_rolling_std_30'] = data[col].rolling(30).std()
                
                # Create seasonal features
                if data.index.dtype == 'datetime64[ns]':
                    features['day_of_week'] = data.index.dayofweek
                    features['month'] = data.index.month
                    features['quarter'] = data.index.quarter
                    features['year'] = data.index.year
                
                # Handle missing values
                features = features.fillna(method='ffill').fillna(method='bfill')
                
                # Create targets
                targets = data[target_columns].copy()
                
                # Remove rows with NaN values
                valid_mask = ~(features.isna().any(axis=1) | targets.isna().any(axis=1))
                features = features[valid_mask]
                targets = targets[valid_mask]
                
                logger.info(f"Prepared multivariate data: {features.shape[0]} samples, {features.shape[1]} features")
                return features, targets
                
        except Exception as e:
            self.error_handler.handle_error(f"Error preparing multivariate data: {str(e)}", e)
            raise
    
    def detect_causal_relationships(
        self, 
        data: pd.DataFrame, 
        target_columns: List[str],
        significance_level: float = 0.05
    ) -> Dict[str, List[str]]:
        """
        Detect causal relationships between variables using Granger causality
        
        Args:
            data: Input DataFrame
            target_columns: Target columns to analyze
            significance_level: Significance level for causality tests
            
        Returns:
            Dictionary mapping target columns to their causal predictors
        """
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available for causal relationship detection")
            return {col: [] for col in target_columns}
        
        try:
            causal_relationships = {}
            
            for target in target_columns:
                causal_predictors = []
                
                for predictor in data.columns:
                    if predictor != target:
                        # Perform Granger causality test
                        try:
                            # Prepare data for Granger causality test
                            test_data = data[[target, predictor]].dropna()
                            if len(test_data) < 20:  # Need sufficient data
                                continue
                            
                            # Perform Granger causality test
                            from statsmodels.tsa.stattools import grangercausalitytests
                            gc_res = grangercausalitytests(test_data, maxlag=5, verbose=False)
                            
                            # Check if predictor Granger-causes target
                            min_p_value = min([gc_res[i+1][0]['ssr_chi2test'][1] for i in range(5)])
                            
                            if min_p_value < significance_level:
                                causal_predictors.append(predictor)
                                
                        except Exception as e:
                            logger.debug(f"Error in Granger causality test for {predictor} -> {target}: {str(e)}")
                            continue
                
                causal_relationships[target] = causal_predictors
                logger.info(f"Detected {len(causal_predictors)} causal predictors for {target}")
            
            return causal_relationships
            
        except Exception as e:
            self.error_handler.handle_error(f"Error detecting causal relationships: {str(e)}", e)
            return {col: [] for col in target_columns}
    
    def train_multivariate_model(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        model_type: str = 'ensemble',
        validation_split: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train multivariate forecasting model
        
        Args:
            features: Feature DataFrame
            targets: Target DataFrame
            model_type: Type of model to train
            validation_split: Fraction of data for validation
            random_state: Random seed
            
        Returns:
            Dictionary containing trained models and metadata
        """
        try:
            with self.performance_monitor.track_operation("multivariate_model_training"):
                # Split data
                split_idx = int(len(features) * (1 - validation_split))
                X_train, X_val = features.iloc[:split_idx], features.iloc[split_idx:]
                y_train, y_val = targets.iloc[:split_idx], targets.iloc[split_idx:]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                models = {}
                feature_importance = {}
                
                if model_type == 'ensemble' and SKLEARN_AVAILABLE:
                    # Train ensemble of models
                    model_configs = [
                        ('random_forest', RandomForestRegressor(n_estimators=100, random_state=random_state)),
                        ('gradient_boosting', GradientBoostingRegressor(n_estimators=100, random_state=random_state)),
                        ('ridge', Ridge(alpha=1.0)),
                        ('lasso', Lasso(alpha=0.1))
                    ]
                    
                    for name, model in model_configs:
                        # Train model for each target
                        target_models = {}
                        target_importance = {}
                        
                        for target_col in targets.columns:
                            model_instance = clone(model)
                            model_instance.fit(X_train_scaled, y_train[target_col])
                            target_models[target_col] = model_instance
                            
                            # Store feature importance if available
                            if hasattr(model_instance, 'feature_importances_'):
                                target_importance[target_col] = dict(zip(features.columns, model_instance.feature_importances_))
                            elif hasattr(model_instance, 'coef_'):
                                target_importance[target_col] = dict(zip(features.columns, abs(model_instance.coef_)))
                        
                        models[name] = target_models
                        feature_importance[name] = target_importance
                
                elif model_type == 'var' and STATSMODELS_AVAILABLE:
                    # Train VAR model
                    var_data = targets.copy()
                    var_model = VAR(var_data)
                    var_result = var_model.fit(maxlags=12)
                    models['var'] = var_result
                
                # Store models and scalers
                self.models[model_type] = models
                self.scalers[model_type] = scaler
                self.feature_importance[model_type] = feature_importance
                
                # Evaluate models
                performance = self._evaluate_models(X_val_scaled, y_val, models, model_type)
                
                logger.info(f"Trained {model_type} multivariate model with performance: {performance}")
                return {
                    'models': models,
                    'scaler': scaler,
                    'feature_importance': feature_importance,
                    'performance': performance
                }
                
        except Exception as e:
            self.error_handler.handle_error(f"Error training multivariate model: {str(e)}", e)
            raise
    
    def forecast_multivariate(
        self,
        data: pd.DataFrame,
        target_columns: List[str],
        forecast_horizon: int = 12,
        model_type: str = 'ensemble',
        include_confidence_intervals: bool = True,
        confidence_level: float = 0.95
    ) -> MultivariateForecastResult:
        """
        Generate multivariate forecasts
        
        Args:
            data: Historical data
            target_columns: Columns to forecast
            forecast_horizon: Number of periods to forecast
            model_type: Type of model to use
            include_confidence_intervals: Whether to include confidence intervals
            confidence_level: Confidence level for intervals
            
        Returns:
            MultivariateForecastResult object
        """
        try:
            with self.performance_monitor.track_operation("multivariate_forecasting"):
                # Prepare data
                features, targets = self.prepare_multivariate_data(data, target_columns)
                
                # Detect causal relationships
                causal_relationships = self.detect_causal_relationships(data, target_columns)
                
                # Train model if not already trained
                if model_type not in self.models:
                    self.train_multivariate_model(features, targets, model_type)
                
                # Generate forecasts
                if model_type == 'ensemble':
                    predictions = self._ensemble_forecast(
                        features, forecast_horizon, include_confidence_intervals, confidence_level
                    )
                elif model_type == 'var':
                    predictions = self._var_forecast(
                        targets, forecast_horizon, include_confidence_intervals, confidence_level
                    )
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                
                # Calculate confidence intervals
                confidence_intervals = {}
                if include_confidence_intervals:
                    confidence_intervals = self._calculate_confidence_intervals(
                        predictions, confidence_level
                    )
                
                # Calculate performance metrics
                performance = self._calculate_performance_metrics(targets, predictions)
                
                # Create result
                result = MultivariateForecastResult(
                    predictions=predictions,
                    confidence_intervals=confidence_intervals,
                    model_performance=performance,
                    feature_importance=self.feature_importance.get(model_type),
                    causal_relationships=causal_relationships,
                    forecast_horizon=forecast_horizon
                )
                
                # Store in history
                self.forecast_history.append(result)
                
                logger.info(f"Generated multivariate forecast for {len(target_columns)} variables")
                return result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in multivariate forecasting: {str(e)}", e)
            raise
    
    def _ensemble_forecast(
        self,
        features: pd.DataFrame,
        forecast_horizon: int,
        include_confidence_intervals: bool,
        confidence_level: float
    ) -> pd.DataFrame:
        """Generate ensemble forecasts"""
        predictions = {}
        
        # Use last available features for forecasting
        last_features = features.iloc[-1:].copy()
        
        for step in range(forecast_horizon):
            step_predictions = {}
            
            for model_name, target_models in self.models['ensemble'].items():
                for target_col, model in target_models.items():
                    # Scale features
                    scaled_features = self.scalers['ensemble'].transform(last_features)
                    
                    # Make prediction
                    pred = model.predict(scaled_features)[0]
                    step_predictions[f"{target_col}_{model_name}"] = pred
            
            # Aggregate predictions
            for target_col in set([col.split('_')[0] for col in step_predictions.keys()]):
                target_preds = [v for k, v in step_predictions.items() if k.startswith(target_col)]
                predictions[f"{target_col}_step_{step+1}"] = np.mean(target_preds)
            
            # Update features for next step (simplified approach)
            # In a real implementation, you'd update lag features based on predictions
        
        return pd.DataFrame([predictions])
    
    def _var_forecast(
        self,
        targets: pd.DataFrame,
        forecast_horizon: int,
        include_confidence_intervals: bool,
        confidence_level: float
    ) -> pd.DataFrame:
        """Generate VAR forecasts"""
        var_model = self.models['var']
        forecast = var_model.forecast(targets.values, steps=forecast_horizon)
        return pd.DataFrame(forecast, columns=targets.columns)
    
    def _calculate_confidence_intervals(
        self,
        predictions: pd.DataFrame,
        confidence_level: float
    ) -> Dict[str, Tuple[pd.Series, pd.Series]]:
        """Calculate confidence intervals for predictions"""
        confidence_intervals = {}
        
        for col in predictions.columns:
            # Simple confidence interval calculation
            # In a real implementation, you'd use more sophisticated methods
            mean_pred = predictions[col].mean()
            std_pred = predictions[col].std()
            
            # Assuming normal distribution
            z_score = 1.96  # For 95% confidence
            lower_bound = mean_pred - z_score * std_pred
            upper_bound = mean_pred + z_score * std_pred
            
            confidence_intervals[col] = (pd.Series([lower_bound]), pd.Series([upper_bound]))
        
        return confidence_intervals
    
    def _evaluate_models(
        self,
        X_val: np.ndarray,
        y_val: pd.DataFrame,
        models: Dict[str, Any],
        model_type: str
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        performance = {}
        
        if model_type == 'ensemble':
            for model_name, target_models in models.items():
                for target_col, model in target_models.items():
                    y_pred = model.predict(X_val)
                    mse = mean_squared_error(y_val[target_col], y_pred)
                    mae = mean_absolute_error(y_val[target_col], y_pred)
                    r2 = r2_score(y_val[target_col], y_pred)
                    
                    performance[f"{target_col}_{model_name}"] = {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2
                    }
        
        return performance
    
    def _calculate_performance_metrics(
        self,
        actual: pd.DataFrame,
        predicted: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate performance metrics"""
        metrics = {}
        
        for col in actual.columns:
            if col in predicted.columns:
                mse = mean_squared_error(actual[col], predicted[col])
                mae = mean_absolute_error(actual[col], predicted[col])
                r2 = r2_score(actual[col], predicted[col])
                
                metrics[col] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
        
        return metrics
    
    def get_forecast_history(self) -> List[MultivariateForecastResult]:
        """Get forecast history"""
        return self.forecast_history
    
    def get_model_performance_history(self) -> List[Dict[str, Any]]:
        """Get model performance history"""
        return self.model_performance_history
    
    def export_model(self, filepath: str) -> None:
        """Export trained models to file"""
        try:
            import joblib
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_importance': self.feature_importance,
                'config': self.config
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Models exported to {filepath}")
        except Exception as e:
            self.error_handler.handle_error(f"Error exporting models: {str(e)}", e)
    
    def load_model(self, filepath: str) -> None:
        """Load trained models from file"""
        try:
            import joblib
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_importance = model_data['feature_importance']
            self.config = model_data.get('config', {})
            logger.info(f"Models loaded from {filepath}")
        except Exception as e:
            self.error_handler.handle_error(f"Error loading models: {str(e)}", e)
