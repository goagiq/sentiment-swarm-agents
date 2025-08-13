"""
Confidence Interval Calculator

This module provides confidence interval calculation and uncertainty
quantification for forecasting results.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """Confidence interval result"""
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    confidence_level: float
    method: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ConfidenceCalculator:
    """
    Calculate confidence intervals and uncertainty measures for forecasts
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the confidence calculator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'default_confidence_level': 0.95,
            'bootstrap_samples': 1000,
            'prediction_interval_method': 'parametric',
            'uncertainty_quantification': True
        }
        
        # Update with provided config
        self.default_config.update(self.config)
        self.config = self.default_config
        
        logger.info(f"ConfidenceCalculator initialized with config: {self.config}")
    
    def calculate_confidence_intervals(
        self,
        predictions: np.ndarray,
        historical_data: np.ndarray,
        confidence_level: Optional[float] = None,
        method: str = "parametric"
    ) -> ConfidenceInterval:
        """
        Calculate confidence intervals for forecast predictions
        
        Args:
            predictions: Forecast predictions
            historical_data: Historical data used for forecasting
            confidence_level: Confidence level (0-1)
            method: Method for calculating intervals
            
        Returns:
            ConfidenceInterval with lower and upper bounds
        """
        try:
            confidence_level = confidence_level or self.config['default_confidence_level']
            
            if method == "parametric":
                return self._parametric_confidence_intervals(
                    predictions, historical_data, confidence_level
                )
            elif method == "bootstrap":
                return self._bootstrap_confidence_intervals(
                    predictions, historical_data, confidence_level
                )
            elif method == "empirical":
                return self._empirical_confidence_intervals(
                    predictions, historical_data, confidence_level
                )
            else:
                raise ValueError(f"Unsupported confidence interval method: {method}")
                
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {str(e)}")
            raise
    
    def _parametric_confidence_intervals(
        self,
        predictions: np.ndarray,
        historical_data: np.ndarray,
        confidence_level: float
    ) -> ConfidenceInterval:
        """Calculate parametric confidence intervals"""
        try:
            # Calculate prediction errors from historical data
            errors = self._calculate_prediction_errors(historical_data)
            
            if len(errors) == 0:
                # Fallback to simple percentage-based intervals
                return self._simple_confidence_intervals(
                    predictions, confidence_level
                )
            
            # Calculate standard error of prediction
            std_error = np.std(errors)
            
            # Calculate z-score for confidence level
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            
            # Calculate confidence intervals
            margin_of_error = z_score * std_error
            lower_bound = predictions - margin_of_error
            upper_bound = predictions + margin_of_error
            
            return ConfidenceInterval(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=confidence_level,
                method="parametric",
                metadata={
                    'std_error': std_error,
                    'z_score': z_score,
                    'margin_of_error': margin_of_error,
                    'error_count': len(errors)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in parametric confidence intervals: {str(e)}")
            raise
    
    def _bootstrap_confidence_intervals(
        self,
        predictions: np.ndarray,
        historical_data: np.ndarray,
        confidence_level: float
    ) -> ConfidenceInterval:
        """Calculate bootstrap confidence intervals"""
        try:
            # Calculate prediction errors
            errors = self._calculate_prediction_errors(historical_data)
            
            if len(errors) == 0:
                return self._simple_confidence_intervals(
                    predictions, confidence_level
                )
            
            # Bootstrap sampling of errors
            n_samples = self.config['bootstrap_samples']
            bootstrap_errors = np.random.choice(
                errors, 
                size=(n_samples, len(predictions)), 
                replace=True
            )
            
            # Calculate bootstrap predictions
            bootstrap_predictions = predictions + bootstrap_errors
            
            # Calculate percentiles for confidence intervals
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
            
            return ConfidenceInterval(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=confidence_level,
                method="bootstrap",
                metadata={
                    'bootstrap_samples': n_samples,
                    'error_count': len(errors),
                    'lower_percentile': lower_percentile,
                    'upper_percentile': upper_percentile
                }
            )
            
        except Exception as e:
            logger.error(f"Error in bootstrap confidence intervals: {str(e)}")
            raise
    
    def _empirical_confidence_intervals(
        self,
        predictions: np.ndarray,
        historical_data: np.ndarray,
        confidence_level: float
    ) -> ConfidenceInterval:
        """Calculate empirical confidence intervals based on historical errors"""
        try:
            # Calculate prediction errors
            errors = self._calculate_prediction_errors(historical_data)
            
            if len(errors) == 0:
                return self._simple_confidence_intervals(
                    predictions, confidence_level
                )
            
            # Calculate error percentiles
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            error_lower = np.percentile(errors, lower_percentile)
            error_upper = np.percentile(errors, upper_percentile)
            
            # Apply error bounds to predictions
            lower_bound = predictions + error_lower
            upper_bound = predictions + error_upper
            
            return ConfidenceInterval(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=confidence_level,
                method="empirical",
                metadata={
                    'error_lower': error_lower,
                    'error_upper': error_upper,
                    'error_count': len(errors),
                    'lower_percentile': lower_percentile,
                    'upper_percentile': upper_percentile
                }
            )
            
        except Exception as e:
            logger.error(f"Error in empirical confidence intervals: {str(e)}")
            raise
    
    def _simple_confidence_intervals(
        self,
        predictions: np.ndarray,
        confidence_level: float
    ) -> ConfidenceInterval:
        """Simple percentage-based confidence intervals"""
        try:
            # Use a simple percentage of the prediction value
            percentage = (1 - confidence_level) * 100
            
            margin = predictions * (percentage / 100)
            lower_bound = predictions - margin
            upper_bound = predictions + margin
            
            return ConfidenceInterval(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence_level=confidence_level,
                method="simple_percentage",
                metadata={
                    'percentage_margin': percentage,
                    'fallback_method': True
                }
            )
            
        except Exception as e:
            logger.error(f"Error in simple confidence intervals: {str(e)}")
            raise
    
    def _calculate_prediction_errors(self, historical_data: np.ndarray) -> np.ndarray:
        """Calculate prediction errors from historical data"""
        try:
            if len(historical_data) < 3:
                return np.array([])
            
            # Use simple moving average as baseline prediction
            window_size = max(1, min(5, len(historical_data) // 2))
            errors = []
            
            for i in range(window_size, len(historical_data)):
                # Predict using previous window
                window_data = historical_data[i-window_size:i]
                prediction = np.mean(window_data)
                actual = historical_data[i]
                error = actual - prediction
                errors.append(error)
            
            return np.array(errors)
            
        except Exception as e:
            logger.warning(f"Error calculating prediction errors: {str(e)}")
            return np.array([])
    
    def calculate_uncertainty_metrics(
        self,
        predictions: np.ndarray,
        historical_data: np.ndarray
    ) -> Dict[str, float]:
        """Calculate various uncertainty metrics"""
        try:
            metrics = {}
            
            # Calculate prediction errors
            errors = self._calculate_prediction_errors(historical_data)
            
            if len(errors) > 0:
                # Standard deviation of errors
                metrics['error_std'] = float(np.std(errors))
                
                # Mean absolute error
                metrics['mae'] = float(np.mean(np.abs(errors)))
                
                # Root mean square error
                metrics['rmse'] = float(np.sqrt(np.mean(errors**2)))
                
                # Coefficient of variation
                if np.mean(predictions) != 0:
                    metrics['cv'] = float(metrics['error_std'] / np.mean(predictions))
                
                # Prediction interval width (95%)
                if len(predictions) > 0:
                    interval_width = np.percentile(errors, 97.5) - np.percentile(errors, 2.5)
                    metrics['prediction_interval_width'] = float(interval_width)
            
            # Add basic statistics
            metrics['prediction_mean'] = float(np.mean(predictions))
            metrics['prediction_std'] = float(np.std(predictions))
            metrics['prediction_range'] = float(np.max(predictions) - np.min(predictions))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating uncertainty metrics: {str(e)}")
            return {}
    
    def get_confidence_level_info(self, confidence_level: float) -> Dict[str, Any]:
        """Get information about a confidence level"""
        return {
            'confidence_level': confidence_level,
            'alpha': 1 - confidence_level,
            'z_score_parametric': stats.norm.ppf((1 + confidence_level) / 2),
            'description': f"{confidence_level*100:.1f}% confidence interval"
        }
    
    def validate_confidence_level(self, confidence_level: float) -> bool:
        """Validate that confidence level is in valid range"""
        return 0 < confidence_level < 1
