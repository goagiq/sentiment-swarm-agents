"""
Confidence Intervals Calculator

Advanced confidence interval calculation for probabilistic forecasting
and uncertainty quantification in multivariate predictions.
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
    from scipy import stats
    from scipy.stats import norm, t
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Some confidence interval methods limited.")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Bootstrap methods limited.")

# Local imports
from ..error_handler import ErrorHandler
from ..caching_service import CachingService
from ..performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceInterval:
    """Data class for confidence interval"""
    lower_bound: float
    upper_bound: float
    confidence_level: float
    method: str
    mean: float
    std: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def width(self) -> float:
        """Calculate confidence interval width"""
        return self.upper_bound - self.lower_bound
    
    @property
    def margin_of_error(self) -> float:
        """Calculate margin of error"""
        return self.width / 2


@dataclass
class ConfidenceIntervalResult:
    """Result container for confidence interval calculations"""
    intervals: Dict[str, ConfidenceInterval]
    prediction_intervals: Dict[str, Tuple[pd.Series, pd.Series]]
    uncertainty_metrics: Dict[str, float]
    coverage_analysis: Dict[str, float]
    method_used: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ConfidenceIntervalCalculator:
    """
    Advanced confidence interval calculator for probabilistic forecasting
    and uncertainty quantification.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the confidence interval calculator"""
        self.config = config or {}
        self.error_handler = ErrorHandler()
        self.caching_service = CachingService()
        self.performance_monitor = PerformanceMonitor()
        
        # Configuration
        self.default_confidence_level = self.config.get('default_confidence_level', 0.95)
        self.bootstrap_samples = self.config.get('bootstrap_samples', 1000)
        self.methods = self.config.get('methods', ['parametric', 'bootstrap', 'empirical'])
        
        # Storage
        self.interval_history = []
        
        logger.info("ConfidenceIntervalCalculator initialized")
    
    def calculate_parametric_intervals(
        self,
        data: pd.Series,
        confidence_level: Optional[float] = None,
        distribution: str = 'normal'
    ) -> ConfidenceInterval:
        """
        Calculate parametric confidence intervals
        
        Args:
            data: Input data series
            confidence_level: Confidence level (default: 0.95)
            distribution: Assumed distribution ('normal', 't')
            
        Returns:
            ConfidenceInterval object
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for parametric confidence intervals")
        
        try:
            with self.performance_monitor.track_operation("parametric_confidence_intervals"):
                if confidence_level is None:
                    confidence_level = self.default_confidence_level
                
                # Calculate basic statistics
                mean_val = data.mean()
                std_val = data.std()
                n = len(data)
                
                if distribution == 'normal':
                    # Normal distribution confidence interval
                    z_score = norm.ppf((1 + confidence_level) / 2)
                    margin_of_error = z_score * (std_val / np.sqrt(n))
                    
                elif distribution == 't':
                    # t-distribution confidence interval
                    t_score = t.ppf((1 + confidence_level) / 2, df=n-1)
                    margin_of_error = t_score * (std_val / np.sqrt(n))
                    
                else:
                    raise ValueError(f"Unsupported distribution: {distribution}")
                
                lower_bound = mean_val - margin_of_error
                upper_bound = mean_val + margin_of_error
                
                interval = ConfidenceInterval(
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    confidence_level=confidence_level,
                    method=f"parametric_{distribution}",
                    mean=mean_val,
                    std=std_val
                )
                
                logger.info(f"Calculated parametric confidence interval: {interval}")
                return interval
                
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating parametric intervals: {str(e)}", e)
            raise
    
    def calculate_bootstrap_intervals(
        self,
        data: pd.Series,
        confidence_level: Optional[float] = None,
        n_bootstrap: Optional[int] = None
    ) -> ConfidenceInterval:
        """
        Calculate bootstrap confidence intervals
        
        Args:
            data: Input data series
            confidence_level: Confidence level
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            ConfidenceInterval object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for bootstrap confidence intervals")
        
        try:
            with self.performance_monitor.track_operation("bootstrap_confidence_intervals"):
                if confidence_level is None:
                    confidence_level = self.default_confidence_level
                
                if n_bootstrap is None:
                    n_bootstrap = self.bootstrap_samples
                
                # Generate bootstrap samples
                bootstrap_means = []
                for _ in range(n_bootstrap):
                    bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                    bootstrap_means.append(bootstrap_sample.mean())
                
                bootstrap_means = np.array(bootstrap_means)
                
                # Calculate confidence interval
                alpha = 1 - confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                lower_bound = np.percentile(bootstrap_means, lower_percentile)
                upper_bound = np.percentile(bootstrap_means, upper_percentile)
                mean_val = np.mean(bootstrap_means)
                std_val = np.std(bootstrap_means)
                
                interval = ConfidenceInterval(
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    confidence_level=confidence_level,
                    method="bootstrap",
                    mean=mean_val,
                    std=std_val
                )
                
                logger.info(f"Calculated bootstrap confidence interval: {interval}")
                return interval
                
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating bootstrap intervals: {str(e)}", e)
            raise
    
    def calculate_empirical_intervals(
        self,
        data: pd.Series,
        confidence_level: Optional[float] = None
    ) -> ConfidenceInterval:
        """
        Calculate empirical confidence intervals using quantiles
        
        Args:
            data: Input data series
            confidence_level: Confidence level
            
        Returns:
            ConfidenceInterval object
        """
        try:
            with self.performance_monitor.track_operation("empirical_confidence_intervals"):
                if confidence_level is None:
                    confidence_level = self.default_confidence_level
                
                # Calculate empirical quantiles
                alpha = 1 - confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                lower_bound = np.percentile(data, lower_percentile)
                upper_bound = np.percentile(data, upper_percentile)
                mean_val = data.mean()
                std_val = data.std()
                
                interval = ConfidenceInterval(
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    confidence_level=confidence_level,
                    method="empirical",
                    mean=mean_val,
                    std=std_val
                )
                
                logger.info(f"Calculated empirical confidence interval: {interval}")
                return interval
                
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating empirical intervals: {str(e)}", e)
            raise
    
    def calculate_multivariate_intervals(
        self,
        data: pd.DataFrame,
        confidence_level: Optional[float] = None,
        method: str = 'parametric'
    ) -> ConfidenceIntervalResult:
        """
        Calculate confidence intervals for multiple variables
        
        Args:
            data: Input DataFrame
            confidence_level: Confidence level
            method: Method to use ('parametric', 'bootstrap', 'empirical')
            
        Returns:
            ConfidenceIntervalResult object
        """
        try:
            with self.performance_monitor.track_operation("multivariate_confidence_intervals"):
                if confidence_level is None:
                    confidence_level = self.default_confidence_level
                
                intervals = {}
                prediction_intervals = {}
                uncertainty_metrics = {}
                
                for column in data.columns:
                    if data[column].dtype in [np.number, 'float64', 'int64']:
                        # Calculate confidence interval for each column
                        if method == 'parametric':
                            interval = self.calculate_parametric_intervals(
                                data[column], confidence_level
                            )
                        elif method == 'bootstrap':
                            interval = self.calculate_bootstrap_intervals(
                                data[column], confidence_level
                            )
                        elif method == 'empirical':
                            interval = self.calculate_empirical_intervals(
                                data[column], confidence_level
                            )
                        else:
                            raise ValueError(f"Unsupported method: {method}")
                        
                        intervals[column] = interval
                        
                        # Calculate prediction intervals
                        prediction_intervals[column] = self._calculate_prediction_interval(
                            data[column], confidence_level, method
                        )
                        
                        # Calculate uncertainty metrics
                        uncertainty_metrics[column] = self._calculate_uncertainty_metrics(
                            data[column], interval
                        )
                
                # Calculate coverage analysis
                coverage_analysis = self._analyze_coverage(data, intervals)
                
                result = ConfidenceIntervalResult(
                    intervals=intervals,
                    prediction_intervals=prediction_intervals,
                    uncertainty_metrics=uncertainty_metrics,
                    coverage_analysis=coverage_analysis,
                    method_used=method
                )
                
                # Store in history
                self.interval_history.append(result)
                
                logger.info(f"Calculated multivariate confidence intervals for {len(intervals)} variables")
                return result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating multivariate intervals: {str(e)}", e)
            raise
    
    def _calculate_prediction_interval(
        self,
        data: pd.Series,
        confidence_level: float,
        method: str
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate prediction intervals"""
        try:
            if method == 'parametric':
                # Parametric prediction interval
                mean_val = data.mean()
                std_val = data.std()
                n = len(data)
                
                if SCIPY_AVAILABLE:
                    t_score = t.ppf((1 + confidence_level) / 2, df=n-1)
                    margin = t_score * std_val * np.sqrt(1 + 1/n)
                else:
                    # Fallback to normal distribution
                    z_score = 1.96  # For 95% confidence
                    margin = z_score * std_val
                
                lower_bound = mean_val - margin
                upper_bound = mean_val + margin
                
            elif method == 'bootstrap':
                # Bootstrap prediction interval
                bootstrap_samples = []
                for _ in range(self.bootstrap_samples):
                    sample = np.random.choice(data, size=len(data), replace=True)
                    bootstrap_samples.append(sample.mean())
                
                alpha = 1 - confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                lower_bound = np.percentile(bootstrap_samples, lower_percentile)
                upper_bound = np.percentile(bootstrap_samples, upper_percentile)
                
            else:
                # Empirical prediction interval
                alpha = 1 - confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                lower_bound = np.percentile(data, lower_percentile)
                upper_bound = np.percentile(data, upper_percentile)
            
            return pd.Series([lower_bound]), pd.Series([upper_bound])
            
        except Exception as e:
            logger.warning(f"Error calculating prediction interval: {str(e)}")
            return pd.Series([data.min()]), pd.Series([data.max()])
    
    def _calculate_uncertainty_metrics(
        self,
        data: pd.Series,
        interval: ConfidenceInterval
    ) -> float:
        """Calculate uncertainty metrics"""
        try:
            # Coefficient of variation
            cv = interval.std / abs(interval.mean) if interval.mean != 0 else 0
            
            # Relative interval width
            relative_width = interval.width / abs(interval.mean) if interval.mean != 0 else 0
            
            # Combined uncertainty metric
            uncertainty = (cv + relative_width) / 2
            
            return uncertainty
            
        except Exception as e:
            logger.warning(f"Error calculating uncertainty metrics: {str(e)}")
            return 0.0
    
    def _analyze_coverage(
        self,
        data: pd.DataFrame,
        intervals: Dict[str, ConfidenceInterval]
    ) -> Dict[str, float]:
        """Analyze coverage of confidence intervals"""
        coverage_analysis = {}
        
        for column, interval in intervals.items():
            if column in data.columns:
                # Calculate how many data points fall within the interval
                within_interval = (
                    (data[column] >= interval.lower_bound) & 
                    (data[column] <= interval.upper_bound)
                )
                
                coverage = within_interval.mean()
                coverage_analysis[column] = coverage
        
        return coverage_analysis
    
    def calculate_forecast_intervals(
        self,
        historical_data: pd.DataFrame,
        forecast_values: pd.DataFrame,
        confidence_level: Optional[float] = None,
        method: str = 'parametric'
    ) -> Dict[str, Tuple[pd.Series, pd.Series]]:
        """
        Calculate confidence intervals for forecast values
        
        Args:
            historical_data: Historical data used for forecasting
            forecast_values: Forecasted values
            confidence_level: Confidence level
            method: Method to use
            
        Returns:
            Dictionary of forecast intervals
        """
        try:
            with self.performance_monitor.track_operation("forecast_confidence_intervals"):
                if confidence_level is None:
                    confidence_level = self.default_confidence_level
                
                forecast_intervals = {}
                
                for column in forecast_values.columns:
                    if column in historical_data.columns:
                        # Calculate forecast uncertainty based on historical errors
                        if hasattr(forecast_values, 'index') and len(forecast_values) > 0:
                            # Use historical data to estimate forecast uncertainty
                            historical_std = historical_data[column].std()
                            
                            # Calculate forecast intervals
                            forecast_mean = forecast_values[column].values
                            
                            if method == 'parametric' and SCIPY_AVAILABLE:
                                z_score = norm.ppf((1 + confidence_level) / 2)
                                margin = z_score * historical_std
                            else:
                                # Fallback to simple method
                                margin = 1.96 * historical_std
                            
                            lower_bound = forecast_mean - margin
                            upper_bound = forecast_mean + margin
                            
                            forecast_intervals[column] = (
                                pd.Series(lower_bound, index=forecast_values.index),
                                pd.Series(upper_bound, index=forecast_values.index)
                            )
                
                logger.info(f"Calculated forecast intervals for {len(forecast_intervals)} variables")
                return forecast_intervals
                
        except Exception as e:
            self.error_handler.handle_error(f"Error calculating forecast intervals: {str(e)}", e)
            raise
    
    def get_interval_summary(self) -> Dict[str, Any]:
        """Get summary of confidence interval calculations"""
        if not self.interval_history:
            return {"message": "No confidence interval calculations found"}
        
        summary = {
            'total_calculations': len(self.interval_history),
            'methods_used': list(set(result.method_used for result in self.interval_history)),
            'average_coverage': {},
            'average_uncertainty': {}
        }
        
        # Calculate average coverage and uncertainty across all calculations
        all_coverage = []
        all_uncertainty = []
        
        for result in self.interval_history:
            all_coverage.extend(result.coverage_analysis.values())
            all_uncertainty.extend(result.uncertainty_metrics.values())
        
        if all_coverage:
            summary['average_coverage'] = {
                'mean': np.mean(all_coverage),
                'std': np.std(all_coverage),
                'min': np.min(all_coverage),
                'max': np.max(all_coverage)
            }
        
        if all_uncertainty:
            summary['average_uncertainty'] = {
                'mean': np.mean(all_uncertainty),
                'std': np.std(all_uncertainty),
                'min': np.min(all_uncertainty),
                'max': np.max(all_uncertainty)
            }
        
        return summary
    
    def export_intervals(self, filepath: str) -> None:
        """Export confidence interval results"""
        try:
            import json
            from datetime import datetime
            
            export_data = {
                'interval_history': [
                    {
                        'method_used': result.method_used,
                        'timestamp': result.timestamp.isoformat() if result.timestamp else None,
                        'intervals_count': len(result.intervals),
                        'coverage_analysis': result.coverage_analysis,
                        'uncertainty_metrics': result.uncertainty_metrics
                    }
                    for result in self.interval_history
                ],
                'config': self.config
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Confidence intervals exported to {filepath}")
            
        except Exception as e:
            self.error_handler.handle_error(f"Error exporting intervals: {str(e)}", e)
