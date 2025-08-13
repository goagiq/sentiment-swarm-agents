"""
Performance Monitoring

Advanced performance monitoring capabilities for tracking model performance,
system metrics, and real-time analytics in production environments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import warnings
import time
import threading
from collections import deque

# Conditional imports
try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Performance metrics limited.")

# Local imports
from ..error_handler import ErrorHandler
from ..caching_service import CachingService
from ..performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Data class for performance metrics"""
    name: str
    value: float
    timestamp: datetime
    metric_type: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PerformanceAlert:
    """Data class for performance alerts"""
    alert_id: str
    metric_name: str
    threshold: float
    current_value: float
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PerformanceReport:
    """Data class for performance reports"""
    report_id: str
    metrics: Dict[str, List[PerformanceMetric]]
    alerts: List[PerformanceAlert]
    summary: Dict[str, Any]
    timestamp: datetime
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class AdvancedPerformanceMonitor:
    """
    Advanced performance monitoring for real-time analytics and model tracking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the advanced performance monitor"""
        self.config = config or {}
        self.error_handler = ErrorHandler()
        self.caching_service = CachingService()
        self.performance_monitor = PerformanceMonitor()
        
        # Configuration
        self.monitoring_interval = self.config.get('monitoring_interval', 60)  # seconds
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'response_time': 5.0,
            'error_rate': 0.05,
            'memory_usage': 0.8,
            'cpu_usage': 0.8
        })
        self.metric_history_size = self.config.get('metric_history_size', 1000)
        
        # Storage
        self.metrics_history = {}
        self.alerts_history = []
        self.reports_history = []
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info("AdvancedPerformanceMonitor initialized")
    
    def start_monitoring(self) -> None:
        """Start continuous performance monitoring"""
        try:
            if not self.monitoring_active:
                self.monitoring_active = True
                self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
                self.monitoring_thread.daemon = True
                self.monitoring_thread.start()
                logger.info("Performance monitoring started")
        except Exception as e:
            self.error_handler.handle_error(f"Error starting monitoring: {str(e)}", e)
    
    def stop_monitoring(self) -> None:
        """Stop continuous performance monitoring"""
        try:
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5)
            logger.info("Performance monitoring stopped")
        except Exception as e:
            self.error_handler.handle_error(f"Error stopping monitoring: {str(e)}", e)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check for alerts
                self._check_alerts()
                
                # Generate report if needed
                if len(self.reports_history) == 0 or \
                   (datetime.now() - self.reports_history[-1].timestamp).seconds > 3600:
                    self._generate_performance_report()
                
                # Wait for next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> None:
        """Collect system performance metrics"""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available / (1024**3)  # GB
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free / (1024**3)  # GB
            
            # Network metrics
            network = psutil.net_io_counters()
            bytes_sent = network.bytes_sent
            bytes_recv = network.bytes_recv
            
            # Store metrics
            metrics = {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_percent': memory_percent,
                'memory_available_gb': memory_available,
                'disk_percent': disk_percent,
                'disk_free_gb': disk_free,
                'network_bytes_sent': bytes_sent,
                'network_bytes_recv': bytes_recv
            }
            
            for name, value in metrics.items():
                self._store_metric(name, value, 'system')
                
        except ImportError:
            logger.warning("psutil not available. System metrics collection disabled.")
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    def _store_metric(self, name: str, value: float, metric_type: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store a performance metric"""
        try:
            metric = PerformanceMetric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                metric_type=metric_type,
                metadata=metadata
            )
            
            if name not in self.metrics_history:
                self.metrics_history[name] = deque(maxlen=self.metric_history_size)
            
            self.metrics_history[name].append(metric)
            
        except Exception as e:
            logger.error(f"Error storing metric {name}: {str(e)}")
    
    def _check_alerts(self) -> None:
        """Check for performance alerts"""
        try:
            for metric_name, threshold in self.alert_thresholds.items():
                if metric_name in self.metrics_history and self.metrics_history[metric_name]:
                    latest_metric = self.metrics_history[metric_name][-1]
                    
                    if latest_metric.value > threshold:
                        # Create alert
                        alert = PerformanceAlert(
                            alert_id=f"{metric_name}_{int(time.time())}",
                            metric_name=metric_name,
                            threshold=threshold,
                            current_value=latest_metric.value,
                            severity='high' if latest_metric.value > threshold * 1.5 else 'medium',
                            message=f"{metric_name} exceeded threshold: {latest_metric.value:.2f} > {threshold:.2f}",
                            timestamp=datetime.now()
                        )
                        
                        self.alerts_history.append(alert)
                        logger.warning(f"Performance alert: {alert.message}")
                        
        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")
    
    def _generate_performance_report(self) -> None:
        """Generate performance report"""
        try:
            report_id = f"report_{int(time.time())}"
            
            # Calculate summary statistics
            summary = {}
            for metric_name, metrics in self.metrics_history.items():
                if metrics:
                    values = [m.value for m in metrics]
                    summary[metric_name] = {
                        'current': values[-1],
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
            
            # Get active alerts
            active_alerts = [alert for alert in self.alerts_history if not alert.resolved]
            
            report = PerformanceReport(
                report_id=report_id,
                metrics=self.metrics_history.copy(),
                alerts=active_alerts,
                summary=summary,
                timestamp=datetime.now()
            )
            
            self.reports_history.append(report)
            logger.info(f"Generated performance report: {report_id}")
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
    
    def track_model_performance(
        self,
        model_name: str,
        predictions: np.ndarray,
        actual_values: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Track model performance metrics
        
        Args:
            model_name: Name of the model
            predictions: Model predictions
            actual_values: Actual values
            metadata: Additional metadata
            
        Returns:
            Dictionary of performance metrics
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Model performance tracking limited.")
            return {}
        
        try:
            with self.performance_monitor.track_operation("model_performance_tracking"):
                metrics = {}
                
                # Calculate basic metrics
                mse = mean_squared_error(actual_values, predictions)
                mae = mean_absolute_error(actual_values, predictions)
                r2 = r2_score(actual_values, predictions)
                
                # Calculate additional metrics
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
                
                metrics = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': rmse,
                    'mape': mape
                }
                
                # Store metrics
                for metric_name, value in metrics.items():
                    self._store_metric(
                        f"{model_name}_{metric_name}",
                        value,
                        'model_performance',
                        metadata
                    )
                
                logger.info(f"Tracked performance for model {model_name}: R² = {r2:.4f}")
                return metrics
                
        except Exception as e:
            self.error_handler.handle_error(f"Error tracking model performance: {str(e)}", e)
            return {}
    
    def track_prediction_latency(
        self,
        model_name: str,
        latency_seconds: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track prediction latency
        
        Args:
            model_name: Name of the model
            latency_seconds: Prediction latency in seconds
            metadata: Additional metadata
        """
        try:
            self._store_metric(
                f"{model_name}_latency",
                latency_seconds,
                'prediction_latency',
                metadata
            )
            
            logger.debug(f"Tracked latency for {model_name}: {latency_seconds:.4f}s")
            
        except Exception as e:
            logger.error(f"Error tracking prediction latency: {str(e)}")
    
    def track_data_quality(
        self,
        dataset_name: str,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Track data quality metrics
        
        Args:
            dataset_name: Name of the dataset
            data: Input data
            metadata: Additional metadata
            
        Returns:
            Dictionary of quality metrics
        """
        try:
            with self.performance_monitor.track_operation("data_quality_tracking"):
                quality_metrics = {}
                
                # Missing data percentage
                missing_percentage = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
                quality_metrics['missing_percentage'] = missing_percentage
                
                # Duplicate rows percentage
                duplicate_percentage = (data.duplicated().sum() / data.shape[0]) * 100
                quality_metrics['duplicate_percentage'] = duplicate_percentage
                
                # Data types distribution
                numeric_columns = data.select_dtypes(include=[np.number]).shape[1]
                categorical_columns = data.select_dtypes(include=['object']).shape[1]
                quality_metrics['numeric_columns'] = numeric_columns
                quality_metrics['categorical_columns'] = categorical_columns
                
                # Memory usage
                memory_usage = data.memory_usage(deep=True).sum() / (1024**2)  # MB
                quality_metrics['memory_usage_mb'] = memory_usage
                
                # Store metrics
                for metric_name, value in quality_metrics.items():
                    self._store_metric(
                        f"{dataset_name}_{metric_name}",
                        value,
                        'data_quality',
                        metadata
                    )
                
                logger.info(f"Tracked data quality for {dataset_name}")
                return quality_metrics
                
        except Exception as e:
            self.error_handler.handle_error(f"Error tracking data quality: {str(e)}", e)
            return {}
    
    def get_performance_summary(
        self,
        metric_names: Optional[List[str]] = None,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get performance summary
        
        Args:
            metric_names: Specific metrics to include
            time_window: Time window for analysis
            
        Returns:
            Performance summary
        """
        try:
            summary = {
                'total_metrics': len(self.metrics_history),
                'total_alerts': len(self.alerts_history),
                'total_reports': len(self.reports_history),
                'active_alerts': len([a for a in self.alerts_history if not a.resolved]),
                'metric_summaries': {}
            }
            
            # Filter by time window
            if time_window:
                cutoff_time = datetime.now() - time_window
            else:
                cutoff_time = None
            
            # Calculate summaries for each metric
            for metric_name, metrics in self.metrics_history.items():
                if metric_names and metric_name not in metric_names:
                    continue
                
                if cutoff_time:
                    filtered_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
                else:
                    filtered_metrics = list(metrics)
                
                if filtered_metrics:
                    values = [m.value for m in filtered_metrics]
                    summary['metric_summaries'][metric_name] = {
                        'count': len(values),
                        'current': values[-1],
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'latest_timestamp': filtered_metrics[-1].timestamp
                    }
            
            return summary
            
        except Exception as e:
            self.error_handler.handle_error(f"Error getting performance summary: {str(e)}", e)
            return {}
    
    def get_metric_history(
        self,
        metric_name: str,
        time_window: Optional[timedelta] = None
    ) -> List[PerformanceMetric]:
        """
        Get metric history
        
        Args:
            metric_name: Name of the metric
            time_window: Time window for filtering
            
        Returns:
            List of performance metrics
        """
        try:
            if metric_name not in self.metrics_history:
                return []
            
            metrics = list(self.metrics_history[metric_name])
            
            if time_window:
                cutoff_time = datetime.now() - time_window
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metric history: {str(e)}")
            return []
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve a performance alert
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            True if alert was resolved, False otherwise
        """
        try:
            for alert in self.alerts_history:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    logger.info(f"Resolved alert: {alert_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert: {str(e)}")
            return False
    
    def export_performance_data(self, filepath: str) -> None:
        """Export performance data"""
        try:
            import json
            from datetime import datetime
            
            export_data = {
                'metrics_history': {
                    metric_name: [
                        {
                            'name': metric.name,
                            'value': metric.value,
                            'timestamp': metric.timestamp.isoformat(),
                            'metric_type': metric.metric_type,
                            'metadata': metric.metadata
                        }
                        for metric in metrics
                    ]
                    for metric_name, metrics in self.metrics_history.items()
                },
                'alerts_history': [
                    {
                        'alert_id': alert.alert_id,
                        'metric_name': alert.metric_name,
                        'threshold': alert.threshold,
                        'current_value': alert.current_value,
                        'severity': alert.severity,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'resolved': alert.resolved
                    }
                    for alert in self.alerts_history
                ],
                'reports_history': [
                    {
                        'report_id': report.report_id,
                        'timestamp': report.timestamp.isoformat(),
                        'summary': report.summary
                    }
                    for report in self.reports_history
                ],
                'config': self.config
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Performance data exported to {filepath}")
            
        except Exception as e:
            self.error_handler.handle_error(f"Error exporting performance data: {str(e)}", e)
