"""
Advanced Anomaly Detection

Sophisticated anomaly detection capabilities using multiple algorithms
for identifying outliers and unusual patterns in multivariate data.
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
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.covariance import EllipticEnvelope
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Anomaly detection features limited.")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Statistical anomaly detection limited.")

# Local imports
from ..error_handler import ErrorHandler
from ..caching_service import CachingService
from ..performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """Data class for anomaly detection results"""
    index: int
    value: float
    score: float
    method: str
    severity: str
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class AnomalyDetectionResult:
    """Result container for anomaly detection"""
    anomalies: List[Anomaly]
    anomaly_scores: pd.Series
    threshold: float
    method_used: str
    performance_metrics: Dict[str, float]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class AdvancedAnomalyDetector:
    """
    Advanced anomaly detection using multiple algorithms and methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the advanced anomaly detector"""
        self.config = config or {}
        self.error_handler = ErrorHandler()
        self.caching_service = CachingService()
        self.performance_monitor = PerformanceMonitor()
        
        # Configuration
        self.contamination = self.config.get('contamination', 0.1)
        self.threshold_method = self.config.get('threshold_method', 'percentile')
        self.severity_thresholds = self.config.get('severity_thresholds', {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        })
        
        # Storage
        self.detection_history = []
        self.models = {}
        
        logger.info("AdvancedAnomalyDetector initialized")
    
    def detect_isolation_forest(
        self,
        data: pd.DataFrame,
        contamination: Optional[float] = None
    ) -> AnomalyDetectionResult:
        """
        Detect anomalies using Isolation Forest
        
        Args:
            data: Input data
            contamination: Expected fraction of anomalies
            
        Returns:
            AnomalyDetectionResult object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for Isolation Forest")
        
        try:
            with self.performance_monitor.track_operation("isolation_forest_detection"):
                if contamination is None:
                    contamination = self.contamination
                
                # Prepare data
                X = self._prepare_data(data)
                
                # Train Isolation Forest
                iso_forest = IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_estimators=100
                )
                
                # Fit and predict
                iso_forest.fit(X)
                scores = iso_forest.score_samples(X)
                predictions = iso_forest.predict(X)
                
                # Convert scores to anomaly scores (higher = more anomalous)
                anomaly_scores = -scores
                
                # Determine threshold
                threshold = self._calculate_threshold(anomaly_scores, contamination)
                
                # Find anomalies
                anomalies = self._find_anomalies(
                    data, anomaly_scores, threshold, "isolation_forest"
                )
                
                # Calculate performance metrics
                performance_metrics = self._calculate_performance_metrics(
                    anomaly_scores, threshold
                )
                
                result = AnomalyDetectionResult(
                    anomalies=anomalies,
                    anomaly_scores=pd.Series(anomaly_scores, index=data.index),
                    threshold=threshold,
                    method_used="isolation_forest",
                    performance_metrics=performance_metrics
                )
                
                # Store model
                self.models['isolation_forest'] = iso_forest
                
                logger.info(f"Detected {len(anomalies)} anomalies using Isolation Forest")
                return result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in Isolation Forest detection: {str(e)}", e)
            raise
    
    def detect_one_class_svm(
        self,
        data: pd.DataFrame,
        kernel: str = 'rbf',
        nu: float = 0.1
    ) -> AnomalyDetectionResult:
        """
        Detect anomalies using One-Class SVM
        
        Args:
            data: Input data
            kernel: SVM kernel type
            nu: Upper bound on fraction of training errors
            
        Returns:
            AnomalyDetectionResult object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for One-Class SVM")
        
        try:
            with self.performance_monitor.track_operation("one_class_svm_detection"):
                # Prepare data
                X = self._prepare_data(data)
                
                # Train One-Class SVM
                oc_svm = OneClassSVM(
                    kernel=kernel,
                    nu=nu,
                    gamma='scale'
                )
                
                # Fit and predict
                oc_svm.fit(X)
                scores = oc_svm.score_samples(X)
                predictions = oc_svm.predict(X)
                
                # Convert scores to anomaly scores
                anomaly_scores = -scores
                
                # Determine threshold
                threshold = self._calculate_threshold(anomaly_scores, nu)
                
                # Find anomalies
                anomalies = self._find_anomalies(
                    data, anomaly_scores, threshold, "one_class_svm"
                )
                
                # Calculate performance metrics
                performance_metrics = self._calculate_performance_metrics(
                    anomaly_scores, threshold
                )
                
                result = AnomalyDetectionResult(
                    anomalies=anomalies,
                    anomaly_scores=pd.Series(anomaly_scores, index=data.index),
                    threshold=threshold,
                    method_used="one_class_svm",
                    performance_metrics=performance_metrics
                )
                
                # Store model
                self.models['one_class_svm'] = oc_svm
                
                logger.info(f"Detected {len(anomalies)} anomalies using One-Class SVM")
                return result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in One-Class SVM detection: {str(e)}", e)
            raise
    
    def detect_local_outlier_factor(
        self,
        data: pd.DataFrame,
        n_neighbors: int = 20,
        contamination: Optional[float] = None
    ) -> AnomalyDetectionResult:
        """
        Detect anomalies using Local Outlier Factor
        
        Args:
            data: Input data
            n_neighbors: Number of neighbors
            contamination: Expected fraction of anomalies
            
        Returns:
            AnomalyDetectionResult object
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for Local Outlier Factor")
        
        try:
            with self.performance_monitor.track_operation("lof_detection"):
                if contamination is None:
                    contamination = self.contamination
                
                # Prepare data
                X = self._prepare_data(data)
                
                # Train Local Outlier Factor
                lof = LocalOutlierFactor(
                    n_neighbors=n_neighbors,
                    contamination=contamination,
                    novelty=False
                )
                
                # Fit and predict
                predictions = lof.fit_predict(X)
                scores = lof.negative_outlier_factor_
                
                # Convert scores to anomaly scores
                anomaly_scores = -scores
                
                # Determine threshold
                threshold = self._calculate_threshold(anomaly_scores, contamination)
                
                # Find anomalies
                anomalies = self._find_anomalies(
                    data, anomaly_scores, threshold, "local_outlier_factor"
                )
                
                # Calculate performance metrics
                performance_metrics = self._calculate_performance_metrics(
                    anomaly_scores, threshold
                )
                
                result = AnomalyDetectionResult(
                    anomalies=anomalies,
                    anomaly_scores=pd.Series(anomaly_scores, index=data.index),
                    threshold=threshold,
                    method_used="local_outlier_factor",
                    performance_metrics=performance_metrics
                )
                
                # Store model
                self.models['local_outlier_factor'] = lof
                
                logger.info(f"Detected {len(anomalies)} anomalies using Local Outlier Factor")
                return result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in LOF detection: {str(e)}", e)
            raise
    
    def detect_statistical_outliers(
        self,
        data: pd.DataFrame,
        method: str = 'zscore',
        threshold: float = 3.0
    ) -> AnomalyDetectionResult:
        """
        Detect anomalies using statistical methods
        
        Args:
            data: Input data
            method: Statistical method ('zscore', 'iqr', 'modified_zscore')
            threshold: Detection threshold
            
        Returns:
            AnomalyDetectionResult object
        """
        try:
            with self.performance_monitor.track_operation("statistical_anomaly_detection"):
                anomalies = []
                anomaly_scores = pd.Series(0.0, index=data.index)
                
                for column in data.columns:
                    if data[column].dtype in [np.number, 'float64', 'int64']:
                        column_anomalies, column_scores = self._detect_column_anomalies(
                            data[column], method, threshold
                        )
                        anomalies.extend(column_anomalies)
                        anomaly_scores = anomaly_scores.add(column_scores, fill_value=0)
                
                # Normalize scores
                if len(anomaly_scores) > 0:
                    anomaly_scores = anomaly_scores / len(data.columns)
                
                # Calculate performance metrics
                performance_metrics = self._calculate_performance_metrics(
                    anomaly_scores, threshold
                )
                
                result = AnomalyDetectionResult(
                    anomalies=anomalies,
                    anomaly_scores=anomaly_scores,
                    threshold=threshold,
                    method_used=f"statistical_{method}",
                    performance_metrics=performance_metrics
                )
                
                logger.info(f"Detected {len(anomalies)} anomalies using statistical method")
                return result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in statistical detection: {str(e)}", e)
            raise
    
    def detect_multivariate_anomalies(
        self,
        data: pd.DataFrame,
        methods: Optional[List[str]] = None,
        ensemble: bool = True
    ) -> Dict[str, AnomalyDetectionResult]:
        """
        Detect anomalies using multiple methods
        
        Args:
            data: Input data
            methods: List of methods to use
            ensemble: Whether to create ensemble result
            
        Returns:
            Dictionary of results for each method
        """
        try:
            with self.performance_monitor.track_operation("multivariate_anomaly_detection"):
                if methods is None:
                    methods = ['isolation_forest', 'one_class_svm', 'statistical']
                
                results = {}
                
                for method in methods:
                    try:
                        if method == 'isolation_forest':
                            results[method] = self.detect_isolation_forest(data)
                        elif method == 'one_class_svm':
                            results[method] = self.detect_one_class_svm(data)
                        elif method == 'local_outlier_factor':
                            results[method] = self.detect_local_outlier_factor(data)
                        elif method == 'statistical':
                            results[method] = self.detect_statistical_outliers(data)
                        else:
                            logger.warning(f"Unknown method: {method}")
                            continue
                            
                    except Exception as e:
                        logger.warning(f"Method {method} failed: {str(e)}")
                        continue
                
                # Create ensemble result if requested
                if ensemble and len(results) > 1:
                    ensemble_result = self._create_ensemble_result(results)
                    results['ensemble'] = ensemble_result
                
                # Store in history
                self.detection_history.extend(results.values())
                
                logger.info(f"Completed multivariate anomaly detection with {len(results)} methods")
                return results
                
        except Exception as e:
            self.error_handler.handle_error(f"Error in multivariate detection: {str(e)}", e)
            raise
    
    def _prepare_data(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare data for anomaly detection"""
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Handle missing values
        numeric_data = numeric_data.fillna(numeric_data.mean())
        
        return numeric_data.values
    
    def _detect_column_anomalies(
        self,
        series: pd.Series,
        method: str,
        threshold: float
    ) -> Tuple[List[Anomaly], pd.Series]:
        """Detect anomalies in a single column"""
        anomalies = []
        scores = pd.Series(0.0, index=series.index)
        
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(series.dropna()))
            anomaly_indices = np.where(z_scores > threshold)[0]
            scores = pd.Series(z_scores, index=series.index)
            
        elif method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            anomaly_mask = (series < lower_bound) | (series > upper_bound)
            anomaly_indices = np.where(anomaly_mask)[0]
            scores = pd.Series(
                np.abs(series - series.median()) / IQR,
                index=series.index
            )
            
        elif method == 'modified_zscore':
            median = series.median()
            mad = np.median(np.abs(series - median))
            modified_z_scores = 0.6745 * (series - median) / mad
            anomaly_indices = np.where(np.abs(modified_z_scores) > threshold)[0]
            scores = pd.Series(np.abs(modified_z_scores), index=series.index)
            
        else:
            return anomalies, scores
        
        # Create anomaly objects
        for idx in anomaly_indices:
            if idx < len(series):
                value = series.iloc[idx]
                score = scores.iloc[idx]
                severity = self._determine_severity(score)
                
                anomaly = Anomaly(
                    index=idx,
                    value=value,
                    score=score,
                    method=f"statistical_{method}",
                    severity=severity
                )
                anomalies.append(anomaly)
        
        return anomalies, scores
    
    def _calculate_threshold(
        self,
        scores: np.ndarray,
        contamination: float
    ) -> float:
        """Calculate threshold for anomaly detection"""
        if self.threshold_method == 'percentile':
            threshold = np.percentile(scores, (1 - contamination) * 100)
        else:
            threshold = np.mean(scores) + 2 * np.std(scores)
        
        return threshold
    
    def _find_anomalies(
        self,
        data: pd.DataFrame,
        scores: np.ndarray,
        threshold: float,
        method: str
    ) -> List[Anomaly]:
        """Find anomalies based on scores and threshold"""
        anomalies = []
        
        for i, score in enumerate(scores):
            if score > threshold and i < len(data):
                # Get the value from the first numeric column
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    value = data[numeric_cols[0]].iloc[i]
                else:
                    value = 0.0
                
                severity = self._determine_severity(score)
                
                anomaly = Anomaly(
                    index=i,
                    value=value,
                    score=score,
                    method=method,
                    severity=severity
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _determine_severity(self, score: float) -> str:
        """Determine anomaly severity based on score"""
        if score > self.severity_thresholds['high']:
            return 'high'
        elif score > self.severity_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_performance_metrics(
        self,
        scores: pd.Series,
        threshold: float
    ) -> Dict[str, float]:
        """Calculate performance metrics for anomaly detection"""
        try:
            anomaly_count = (scores > threshold).sum()
            total_count = len(scores)
            
            metrics = {
                'anomaly_rate': anomaly_count / total_count if total_count > 0 else 0,
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'max_score': scores.max(),
                'min_score': scores.min(),
                'threshold': threshold
            }
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def _create_ensemble_result(
        self,
        results: Dict[str, AnomalyDetectionResult]
    ) -> AnomalyDetectionResult:
        """Create ensemble result from multiple methods"""
        try:
            # Combine scores from all methods
            all_scores = []
            for result in results.values():
                all_scores.append(result.anomaly_scores)
            
            # Calculate ensemble scores
            ensemble_scores = pd.concat(all_scores, axis=1).mean(axis=1)
            
            # Calculate ensemble threshold
            ensemble_threshold = np.mean([result.threshold for result in results.values()])
            
            # Find ensemble anomalies
            ensemble_anomalies = self._find_anomalies(
                pd.DataFrame(ensemble_scores),  # Dummy DataFrame
                ensemble_scores.values,
                ensemble_threshold,
                "ensemble"
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                ensemble_scores, ensemble_threshold
            )
            
            ensemble_result = AnomalyDetectionResult(
                anomalies=ensemble_anomalies,
                anomaly_scores=ensemble_scores,
                threshold=ensemble_threshold,
                method_used="ensemble",
                performance_metrics=performance_metrics
            )
            
            return ensemble_result
            
        except Exception as e:
            logger.warning(f"Error creating ensemble result: {str(e)}")
            # Return first result as fallback
            return list(results.values())[0]
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of anomaly detection results"""
        if not self.detection_history:
            return {"message": "No anomaly detection results found"}
        
        summary = {
            'total_detections': len(self.detection_history),
            'methods_used': list(set(result.method_used for result in self.detection_history)),
            'total_anomalies': sum(len(result.anomalies) for result in self.detection_history),
            'average_anomaly_rate': {},
            'severity_distribution': {}
        }
        
        # Calculate average anomaly rates
        anomaly_rates = [result.performance_metrics.get('anomaly_rate', 0) 
                        for result in self.detection_history]
        if anomaly_rates:
            summary['average_anomaly_rate'] = {
                'mean': np.mean(anomaly_rates),
                'std': np.std(anomaly_rates),
                'min': np.min(anomaly_rates),
                'max': np.max(anomaly_rates)
            }
        
        # Calculate severity distribution
        all_severities = []
        for result in self.detection_history:
            all_severities.extend([anomaly.severity for anomaly in result.anomalies])
        
        if all_severities:
            severity_counts = {}
            for severity in set(all_severities):
                severity_counts[severity] = all_severities.count(severity)
            summary['severity_distribution'] = severity_counts
        
        return summary
    
    def export_detection_results(self, filepath: str) -> None:
        """Export anomaly detection results"""
        try:
            import json
            from datetime import datetime
            
            export_data = {
                'detection_history': [
                    {
                        'method_used': result.method_used,
                        'timestamp': result.timestamp.isoformat() if result.timestamp else None,
                        'anomaly_count': len(result.anomalies),
                        'threshold': result.threshold,
                        'performance_metrics': result.performance_metrics
                    }
                    for result in self.detection_history
                ],
                'config': self.config
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Anomaly detection results exported to {filepath}")
            
        except Exception as e:
            self.error_handler.handle_error(f"Error exporting detection results: {str(e)}", e)
