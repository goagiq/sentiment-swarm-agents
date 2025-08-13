"""
Forecast Validator

This module provides model accuracy assessment and validation
capabilities for forecasting models.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of forecast validation"""
    accuracy_metrics: Dict[str, float]
    error_analysis: Dict[str, Any]
    model_performance: str  # 'excellent', 'good', 'fair', 'poor'
    recommendations: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CrossValidationResult:
    """Result of cross-validation"""
    fold_results: List[ValidationResult]
    average_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    overall_performance: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ForecastValidator:
    """
    Validate and assess forecasting model performance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the forecast validator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'validation_split': 0.2,
            'cross_validation_folds': 5,
            'performance_thresholds': {
                'excellent': 0.9,
                'good': 0.8,
                'fair': 0.7,
                'poor': 0.6
            },
            'min_test_size': 5,
            'max_test_size': 100
        }
        
        # Update with provided config
        self.default_config.update(self.config)
        self.config = self.default_config
        
        logger.info(f"ForecastValidator initialized with config: {self.config}")
    
    def validate_forecast(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        validation_type: str = "holdout"
    ) -> ValidationResult:
        """
        Validate forecast accuracy using actual vs predicted values
        
        Args:
            actual: Actual observed values
            predicted: Predicted values
            validation_type: Type of validation ('holdout', 'cross_validation')
            
        Returns:
            ValidationResult with accuracy metrics and analysis
        """
        try:
            # Validate inputs
            self._validate_inputs(actual, predicted)
            
            if validation_type == "holdout":
                return self._holdout_validation(actual, predicted)
            elif validation_type == "cross_validation":
                return self._cross_validation(actual, predicted)
            else:
                raise ValueError(f"Unsupported validation type: {validation_type}")
                
        except Exception as e:
            logger.error(f"Error in forecast validation: {str(e)}")
            raise
    
    def _holdout_validation(
        self,
        actual: np.ndarray,
        predicted: np.ndarray
    ) -> ValidationResult:
        """Perform holdout validation"""
        try:
            # Calculate accuracy metrics
            accuracy_metrics = self._calculate_accuracy_metrics(actual, predicted)
            
            # Perform error analysis
            error_analysis = self._analyze_errors(actual, predicted)
            
            # Assess model performance
            performance = self._assess_performance(accuracy_metrics)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                accuracy_metrics, error_analysis, performance
            )
            
            return ValidationResult(
                accuracy_metrics=accuracy_metrics,
                error_analysis=error_analysis,
                model_performance=performance,
                recommendations=recommendations,
                metadata={
                    'validation_type': 'holdout',
                    'data_points': len(actual),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error in holdout validation: {str(e)}")
            raise
    
    def _cross_validation(
        self,
        actual: np.ndarray,
        predicted: np.ndarray
    ) -> CrossValidationResult:
        """Perform cross-validation"""
        try:
            n_folds = self.config['cross_validation_folds']
            fold_size = len(actual) // n_folds
            
            fold_results = []
            all_metrics = []
            
            for i in range(n_folds):
                start_idx = i * fold_size
                end_idx = start_idx + fold_size if i < n_folds - 1 else len(actual)
                
                # Extract fold data
                fold_actual = actual[start_idx:end_idx]
                fold_predicted = predicted[start_idx:end_idx]
                
                # Validate fold
                fold_result = self._holdout_validation(fold_actual, fold_predicted)
                fold_results.append(fold_result)
                all_metrics.append(fold_result.accuracy_metrics)
            
            # Calculate average metrics across folds
            average_metrics = self._calculate_average_metrics(all_metrics)
            std_metrics = self._calculate_std_metrics(all_metrics)
            
            # Assess overall performance
            overall_performance = self._assess_performance(average_metrics)
            
            return CrossValidationResult(
                fold_results=fold_results,
                average_metrics=average_metrics,
                std_metrics=std_metrics,
                overall_performance=overall_performance,
                metadata={
                    'validation_type': 'cross_validation',
                    'n_folds': n_folds,
                    'fold_size': fold_size,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise
    
    def _calculate_accuracy_metrics(
        self,
        actual: np.ndarray,
        predicted: np.ndarray
    ) -> Dict[str, float]:
        """Calculate various accuracy metrics"""
        try:
            metrics = {}
            
            # Mean Absolute Error (MAE)
            mae = np.mean(np.abs(actual - predicted))
            metrics['mae'] = float(mae)
            
            # Mean Absolute Percentage Error (MAPE)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            metrics['mape'] = float(mape)
            
            # Root Mean Square Error (RMSE)
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            metrics['rmse'] = float(rmse)
            
            # Mean Squared Error (MSE)
            mse = np.mean((actual - predicted) ** 2)
            metrics['mse'] = float(mse)
            
            # R-squared (Coefficient of Determination)
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            metrics['r_squared'] = float(r_squared)
            
            # Accuracy (1 - MAPE/100)
            accuracy = max(0, 1 - mape / 100)
            metrics['accuracy'] = float(accuracy)
            
            # Symmetric Mean Absolute Percentage Error (SMAPE)
            smape = 2 * np.mean(np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))) * 100
            metrics['smape'] = float(smape)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {str(e)}")
            return {}
    
    def _analyze_errors(
        self,
        actual: np.ndarray,
        predicted: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction errors"""
        try:
            errors = actual - predicted
            
            analysis = {
                'error_statistics': {
                    'mean': float(np.mean(errors)),
                    'std': float(np.std(errors)),
                    'min': float(np.min(errors)),
                    'max': float(np.max(errors)),
                    'median': float(np.median(errors))
                },
                'error_distribution': {
                    'positive_errors': int(np.sum(errors > 0)),
                    'negative_errors': int(np.sum(errors < 0)),
                    'zero_errors': int(np.sum(errors == 0))
                },
                'bias_analysis': {
                    'is_biased': abs(np.mean(errors)) > 0.1 * np.std(actual),
                    'bias_direction': 'positive' if np.mean(errors) > 0 else 'negative',
                    'bias_magnitude': float(abs(np.mean(errors)))
                },
                'outlier_analysis': {
                    'outlier_threshold': float(2 * np.std(errors)),
                    'outlier_count': int(np.sum(np.abs(errors) > 2 * np.std(errors)))
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing errors: {str(e)}")
            return {}
    
    def _assess_performance(self, accuracy_metrics: Dict[str, float]) -> str:
        """Assess overall model performance"""
        try:
            thresholds = self.config['performance_thresholds']
            accuracy = accuracy_metrics.get('accuracy', 0)
            
            if accuracy >= thresholds['excellent']:
                return 'excellent'
            elif accuracy >= thresholds['good']:
                return 'good'
            elif accuracy >= thresholds['fair']:
                return 'fair'
            else:
                return 'poor'
                
        except Exception as e:
            logger.error(f"Error assessing performance: {str(e)}")
            return 'unknown'
    
    def _generate_recommendations(
        self,
        accuracy_metrics: Dict[str, float],
        error_analysis: Dict[str, Any],
        performance: str
    ) -> List[str]:
        """Generate recommendations for model improvement"""
        try:
            recommendations = []
            
            # Performance-based recommendations
            if performance == 'poor':
                recommendations.append("Model performance is poor. Consider using different algorithms or features.")
                recommendations.append("Check data quality and preprocessing steps.")
            elif performance == 'fair':
                recommendations.append("Model performance is fair. Consider hyperparameter tuning.")
                recommendations.append("Explore additional features or data sources.")
            elif performance == 'good':
                recommendations.append("Model performance is good. Minor improvements may be possible.")
            elif performance == 'excellent':
                recommendations.append("Model performance is excellent. Consider monitoring for drift.")
            
            # Error analysis recommendations
            bias_analysis = error_analysis.get('bias_analysis', {})
            if bias_analysis.get('is_biased', False):
                direction = bias_analysis.get('bias_direction', 'unknown')
                recommendations.append(f"Model shows {direction} bias. Consider bias correction techniques.")
            
            outlier_analysis = error_analysis.get('outlier_analysis', {})
            outlier_count = outlier_analysis.get('outlier_count', 0)
            if outlier_count > len(accuracy_metrics) * 0.1:  # More than 10% outliers
                recommendations.append("High number of outliers detected. Consider robust modeling techniques.")
            
            # Accuracy metric recommendations
            mape = accuracy_metrics.get('mape', 0)
            if mape > 20:
                recommendations.append("High MAPE indicates poor relative accuracy. Consider log transformations.")
            
            r_squared = accuracy_metrics.get('r_squared', 0)
            if r_squared < 0.5:
                recommendations.append("Low R-squared suggests poor fit. Consider feature engineering.")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ["Unable to generate recommendations due to error"]
    
    def _calculate_average_metrics(
        self,
        all_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate average metrics across folds"""
        try:
            if not all_metrics:
                return {}
            
            avg_metrics = {}
            metric_names = all_metrics[0].keys()
            
            for metric in metric_names:
                values = [metrics.get(metric, 0) for metrics in all_metrics]
                avg_metrics[metric] = float(np.mean(values))
            
            return avg_metrics
            
        except Exception as e:
            logger.error(f"Error calculating average metrics: {str(e)}")
            return {}
    
    def _calculate_std_metrics(
        self,
        all_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate standard deviation of metrics across folds"""
        try:
            if not all_metrics:
                return {}
            
            std_metrics = {}
            metric_names = all_metrics[0].keys()
            
            for metric in metric_names:
                values = [metrics.get(metric, 0) for metrics in all_metrics]
                std_metrics[metric] = float(np.std(values))
            
            return std_metrics
            
        except Exception as e:
            logger.error(f"Error calculating std metrics: {str(e)}")
            return {}
    
    def _validate_inputs(self, actual: np.ndarray, predicted: np.ndarray) -> None:
        """Validate input arrays"""
        if len(actual) != len(predicted):
            raise ValueError("Actual and predicted arrays must have the same length")
        
        if len(actual) < self.config['min_test_size']:
            raise ValueError(f"Insufficient data points. Need at least {self.config['min_test_size']}")
        
        if np.any(np.isnan(actual)) or np.any(np.isnan(predicted)):
            raise ValueError("Input arrays contain NaN values")
        
        if np.any(np.isinf(actual)) or np.any(np.isinf(predicted)):
            raise ValueError("Input arrays contain infinite values")
    
    def get_performance_summary(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """Get a summary of validation results"""
        try:
            return {
                'performance_level': validation_result.model_performance,
                'key_metrics': {
                    'accuracy': validation_result.accuracy_metrics.get('accuracy', 0),
                    'mape': validation_result.accuracy_metrics.get('mape', 0),
                    'r_squared': validation_result.accuracy_metrics.get('r_squared', 0)
                },
                'recommendations_count': len(validation_result.recommendations),
                'has_bias': validation_result.error_analysis.get('bias_analysis', {}).get('is_biased', False),
                'outlier_percentage': validation_result.error_analysis.get('outlier_analysis', {}).get('outlier_count', 0) / 
                                    validation_result.metadata.get('data_points', 1) * 100
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {}
