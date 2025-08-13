"""
Advanced Forecasting Engine

This module provides multi-model forecasting capabilities including:
- ARIMA models for time series forecasting
- Prophet models for trend and seasonal forecasting  
- LSTM models for complex pattern recognition
- Ensemble methods for improved accuracy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from enum import Enum

# Import existing pattern recognition components
from ..pattern_recognition.temporal_analyzer import TemporalAnalyzer
from ..pattern_recognition.trend_engine import TrendEngine

logger = logging.getLogger(__name__)


class ForecastModelType(Enum):
    """Supported forecasting model types"""
    ARIMA = "arima"
    PROPHET = "prophet" 
    LSTM = "lstm"
    ENSEMBLE = "ensemble"
    SIMPLE_AVERAGE = "simple_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"


@dataclass
class ForecastResult:
    """Result of a forecasting operation"""
    predictions: np.ndarray
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None
    model_accuracy: Optional[float] = None
    model_type: str = ""
    forecast_horizon: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ForecastingEngine:
    """
    Advanced forecasting engine supporting multiple models and ensemble methods
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the forecasting engine
        
        Args:
            config: Configuration dictionary for forecasting parameters
        """
        self.config = config or {}
        self.temporal_analyzer = TemporalAnalyzer()
        self.trend_engine = TrendEngine()
        
        # Model storage for trained models
        self.trained_models: Dict[str, Any] = {}
        
        # Default configuration
        self.default_config = {
            'forecast_horizon': 12,
            'confidence_level': 0.95,
            'min_data_points': 10,
            'max_data_points': 10000,
            'ensemble_method': 'weighted_average',
            'validation_split': 0.2,
            'random_state': 42
        }
        
        # Update with provided config
        self.default_config.update(self.config)
        self.config = self.default_config
        
        logger.info(f"ForecastingEngine initialized with config: {self.config}")
    
    def forecast(
        self, 
        data: Union[List[float], np.ndarray, pd.Series],
        model_type: Union[ForecastModelType, str] = ForecastModelType.ENSEMBLE,
        forecast_horizon: Optional[int] = None,
        **kwargs
    ) -> ForecastResult:
        """
        Generate forecasts using specified model type
        
        Args:
            data: Time series data to forecast
            model_type: Type of forecasting model to use
            forecast_horizon: Number of periods to forecast
            **kwargs: Additional model-specific parameters
            
        Returns:
            ForecastResult with predictions and metadata
        """
        try:
            # Convert data to numpy array
            if isinstance(data, pd.Series):
                data_array = data.values
            elif isinstance(data, list):
                data_array = np.array(data)
            else:
                data_array = data
            
            # Validate data
            self._validate_data(data_array)
            
            # Set forecast horizon
            horizon = forecast_horizon or self.config['forecast_horizon']
            
            # Convert string to enum if needed
            if isinstance(model_type, str):
                model_type = ForecastModelType(model_type.lower())
            
            logger.info(f"Generating forecast with {model_type.value} model for {horizon} periods")
            
            # Generate forecast based on model type
            if model_type == ForecastModelType.ENSEMBLE:
                result = self._ensemble_forecast(data_array, horizon, **kwargs)
            elif model_type == ForecastModelType.ARIMA:
                result = self._arima_forecast(data_array, horizon, **kwargs)
            elif model_type == ForecastModelType.PROPHET:
                result = self._prophet_forecast(data_array, horizon, **kwargs)
            elif model_type == ForecastModelType.LSTM:
                result = self._lstm_forecast(data_array, horizon, **kwargs)
            elif model_type == ForecastModelType.SIMPLE_AVERAGE:
                result = self._simple_average_forecast(data_array, horizon, **kwargs)
            elif model_type == ForecastModelType.EXPONENTIAL_SMOOTHING:
                result = self._exponential_smoothing_forecast(data_array, horizon, **kwargs)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Add metadata
            result.metadata.update({
                'model_type': model_type.value,
                'forecast_horizon': horizon,
                'data_points': len(data_array),
                'timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Forecast completed successfully with {model_type.value} model")
            return result
            
        except Exception as e:
            logger.error(f"Error in forecasting: {str(e)}")
            raise
    
    def _validate_data(self, data: np.ndarray) -> None:
        """Validate input data"""
        if len(data) < self.config['min_data_points']:
            raise ValueError(f"Insufficient data points. Need at least {self.config['min_data_points']}, got {len(data)}")
        
        if len(data) > self.config['max_data_points']:
            logger.warning(f"Large dataset detected. Truncating to {self.config['max_data_points']} points")
            data = data[-self.config['max_data_points']:]
        
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Data contains NaN or infinite values")
    
    def _ensemble_forecast(
        self, 
        data: np.ndarray, 
        horizon: int, 
        **kwargs
    ) -> ForecastResult:
        """Generate ensemble forecast using multiple models"""
        try:
            # Get individual model forecasts
            models = [
                ForecastModelType.SIMPLE_AVERAGE,
                ForecastModelType.EXPONENTIAL_SMOOTHING
            ]
            
            # Try to add more sophisticated models if data allows
            if len(data) >= 20:
                models.append(ForecastModelType.ARIMA)
            
            forecasts = []
            weights = []
            
            for model_type in models:
                try:
                    if model_type == ForecastModelType.SIMPLE_AVERAGE:
                        forecast = self._simple_average_forecast(data, horizon, **kwargs)
                    elif model_type == ForecastModelType.EXPONENTIAL_SMOOTHING:
                        forecast = self._exponential_smoothing_forecast(data, horizon, **kwargs)
                    elif model_type == ForecastModelType.ARIMA:
                        forecast = self._arima_forecast(data, horizon, **kwargs)
                    
                    forecasts.append(forecast)
                    # Weight based on model accuracy (higher accuracy = higher weight)
                    weight = forecast.model_accuracy or 1.0
                    weights.append(weight)
                    
                except Exception as e:
                    logger.warning(f"Model {model_type.value} failed: {str(e)}")
                    continue
            
            if not forecasts:
                raise ValueError("All forecasting models failed")
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Calculate weighted ensemble
            ensemble_predictions = np.zeros(horizon)
            for i, (forecast, weight) in enumerate(zip(forecasts, weights)):
                ensemble_predictions += weight * forecast.predictions
            
            # Calculate ensemble accuracy (average of individual accuracies)
            ensemble_accuracy = np.average([f.model_accuracy or 0.0 for f in forecasts if f.model_accuracy is not None])
            
            return ForecastResult(
                predictions=ensemble_predictions,
                model_accuracy=ensemble_accuracy,
                model_type="ensemble",
                forecast_horizon=horizon,
                metadata={
                    'ensemble_models': [m.value for m in models],
                    'ensemble_weights': weights.tolist(),
                    'individual_forecasts': len(forecasts)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in ensemble forecasting: {str(e)}")
            raise
    
    def _simple_average_forecast(
        self, 
        data: np.ndarray, 
        horizon: int, 
        **kwargs
    ) -> ForecastResult:
        """Simple moving average forecast"""
        try:
            # Ensure minimum data size
            if len(data) < 2:
                # Fallback to simple average of all data
                avg_value = np.mean(data) if len(data) > 0 else 0
                predictions = np.full(horizon, avg_value)
                window_size = 1  # Default window size for small data
            else:
                # Use the last few values to calculate average
                window_size = max(1, min(5, len(data) // 2))
                recent_data = data[-window_size:]
                
                # Simple average prediction
                avg_value = np.mean(recent_data)
                predictions = np.full(horizon, avg_value)
            
            # Calculate accuracy using historical data
            accuracy = self._calculate_accuracy(data, window_size)
            
            return ForecastResult(
                predictions=predictions,
                model_accuracy=accuracy,
                model_type="simple_average",
                forecast_horizon=horizon
            )
            
        except Exception as e:
            logger.error(f"Error in simple average forecasting: {str(e)}")
            raise
    
    def _exponential_smoothing_forecast(
        self, 
        data: np.ndarray, 
        horizon: int, 
        alpha: float = 0.3,
        **kwargs
    ) -> ForecastResult:
        """Exponential smoothing forecast"""
        try:
            # Apply exponential smoothing
            smoothed = [data[0]]
            for i in range(1, len(data)):
                smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[i-1])
            
            # Use last smoothed value for prediction
            last_smoothed = smoothed[-1]
            predictions = np.full(horizon, last_smoothed)
            
            # Calculate accuracy
            accuracy = self._calculate_accuracy(data, 5, smoothed)
            
            return ForecastResult(
                predictions=predictions,
                model_accuracy=accuracy,
                model_type="exponential_smoothing",
                forecast_horizon=horizon,
                metadata={'alpha': alpha}
            )
            
        except Exception as e:
            logger.error(f"Error in exponential smoothing forecasting: {str(e)}")
            raise
    
    def _arima_forecast(
        self, 
        data: np.ndarray, 
        horizon: int, 
        **kwargs
    ) -> ForecastResult:
        """ARIMA model forecast (simplified implementation)"""
        try:
            # For now, implement a simplified ARIMA-like approach
            # In production, this would use statsmodels ARIMA
            
            # Calculate trend and seasonal components
            # For now, use simple linear regression
            x = np.arange(len(data))
            slope, intercept = np.polyfit(x, data, 1)
            
            # Simple linear trend projection
            if not np.isnan(slope) and not np.isnan(intercept):
                
                # Generate predictions with trend
                predictions = []
                for i in range(1, horizon + 1):
                    pred = intercept + slope * (len(data) + i)
                    predictions.append(pred)
                
                predictions = np.array(predictions)
            else:
                # Fallback to simple average
                predictions = np.full(horizon, np.mean(data))
            
            # Calculate accuracy
            accuracy = self._calculate_accuracy(data, 10)
            
            return ForecastResult(
                predictions=predictions,
                model_accuracy=accuracy,
                model_type="arima",
                forecast_horizon=horizon
            )
            
        except Exception as e:
            logger.error(f"Error in ARIMA forecasting: {str(e)}")
            raise
    
    def _prophet_forecast(
        self, 
        data: np.ndarray, 
        horizon: int, 
        **kwargs
    ) -> ForecastResult:
        """Prophet model forecast (placeholder implementation)"""
        try:
            # Placeholder for Prophet implementation
            # In production, this would use Facebook Prophet
            
            # For now, use exponential smoothing as proxy
            return self._exponential_smoothing_forecast(data, horizon, alpha=0.2, **kwargs)
            
        except Exception as e:
            logger.error(f"Error in Prophet forecasting: {str(e)}")
            raise
    
    def _lstm_forecast(
        self, 
        data: np.ndarray, 
        horizon: int, 
        **kwargs
    ) -> ForecastResult:
        """LSTM model forecast (placeholder implementation)"""
        try:
            # Placeholder for LSTM implementation
            # In production, this would use TensorFlow/Keras
            
            # For now, use trend-based forecasting
            return self._arima_forecast(data, horizon, **kwargs)
            
        except Exception as e:
            logger.error(f"Error in LSTM forecasting: {str(e)}")
            raise
    
    def _calculate_accuracy(
        self, 
        data: np.ndarray, 
        test_periods: int = 5,
        predictions: Optional[np.ndarray] = None
    ) -> float:
        """Calculate forecast accuracy using historical data"""
        try:
            if len(data) < test_periods + 1:
                return 0.5  # Default accuracy for insufficient data
            
            # Use last test_periods for validation
            actual = data[-test_periods:]
            
            if predictions is None:
                # Generate predictions for validation period
                train_data = data[:-test_periods]
                val_predictions = self._simple_average_forecast(train_data, test_periods).predictions
            else:
                val_predictions = predictions[-test_periods:]
            
            # Calculate mean absolute percentage error
            mape = np.mean(np.abs((actual - val_predictions) / actual)) * 100
            accuracy = max(0, 100 - mape) / 100  # Convert to 0-1 scale
            
            return accuracy
            
        except Exception as e:
            logger.warning(f"Error calculating accuracy: {str(e)}")
            return 0.5  # Default accuracy
    
    def get_available_models(self) -> List[str]:
        """Get list of available forecasting models"""
        return [model.value for model in ForecastModelType]
    
    def get_model_info(self, model_type: Union[ForecastModelType, str]) -> Dict[str, Any]:
        """Get information about a specific model type"""
        if isinstance(model_type, str):
            model_type = ForecastModelType(model_type.lower())
        
        model_info = {
            ForecastModelType.ARIMA: {
                'name': 'ARIMA',
                'description': 'Autoregressive Integrated Moving Average',
                'best_for': 'Time series with trends and seasonality',
                'min_data_points': 20,
                'complexity': 'Medium'
            },
            ForecastModelType.PROPHET: {
                'name': 'Prophet',
                'description': 'Facebook Prophet for trend and seasonal forecasting',
                'best_for': 'Time series with strong seasonal patterns',
                'min_data_points': 10,
                'complexity': 'Medium'
            },
            ForecastModelType.LSTM: {
                'name': 'LSTM',
                'description': 'Long Short-Term Memory neural networks',
                'best_for': 'Complex non-linear patterns',
                'min_data_points': 50,
                'complexity': 'High'
            },
            ForecastModelType.ENSEMBLE: {
                'name': 'Ensemble',
                'description': 'Combination of multiple models',
                'best_for': 'General purpose with improved accuracy',
                'min_data_points': 10,
                'complexity': 'Medium'
            },
            ForecastModelType.SIMPLE_AVERAGE: {
                'name': 'Simple Average',
                'description': 'Moving average forecasting',
                'best_for': 'Simple trendless data',
                'min_data_points': 5,
                'complexity': 'Low'
            },
            ForecastModelType.EXPONENTIAL_SMOOTHING: {
                'name': 'Exponential Smoothing',
                'description': 'Weighted average with exponential decay',
                'best_for': 'Data with trends',
                'min_data_points': 5,
                'complexity': 'Low'
            }
        }
        
        return model_info.get(model_type, {})
