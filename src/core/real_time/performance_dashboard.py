"""
Performance Dashboard

This module provides live metrics display and performance monitoring
capabilities for real-time systems.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Represents a performance metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardConfig:
    """Configuration for the performance dashboard"""
    update_interval: float = 1.0  # seconds
    max_metrics_history: int = 1000
    enable_auto_scaling: bool = True
    alert_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    categories: List[str] = field(default_factory=lambda: [
        "system", "performance", "memory", "network", "custom"
    ])


class PerformanceDashboard:
    """
    Real-time performance dashboard that displays live metrics and
    provides performance monitoring capabilities.
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """
        Initialize the performance dashboard
        
        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        self.metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.max_metrics_history)
        )
        self.current_metrics: Dict[str, Metric] = {}
        self.is_monitoring = False
        self.monitoring_task = None
        self.metric_callbacks: List[Callable[[Metric], None]] = []
        self.alert_callbacks: List[Callable[[str, Metric], None]] = []
        
        # Performance tracking
        self.start_time = datetime.now()
        self.update_count = 0
        self.last_update = datetime.now()
        
        logger.info("PerformanceDashboard initialized")
    
    async def start_monitoring(self):
        """Start the performance monitoring process"""
        if self.is_monitoring:
            logger.warning("Dashboard monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance dashboard monitoring started")
    
    async def stop_monitoring(self):
        """Stop the performance monitoring process"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance dashboard monitoring stopped")
    
    def add_metric(self, name: str, value: float, unit: str = "", 
                   category: str = "general", metadata: Optional[Dict[str, Any]] = None):
        """
        Add a new metric to the dashboard
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            category: Metric category
            metadata: Additional metadata
        """
        timestamp = datetime.now()
        metric = Metric(
            name=name,
            value=value,
            unit=unit,
            timestamp=timestamp,
            category=category,
            metadata=metadata or {}
        )
        
        # Store in current metrics
        self.current_metrics[name] = metric
        
        # Add to history
        self.metrics_history[name].append(metric)
        
        # Check for alerts
        self._check_alert_thresholds(name, metric)
        
        # Call callbacks
        for callback in self.metric_callbacks:
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"Error in metric callback: {str(e)}")
    
    def add_metric_callback(self, callback: Callable[[Metric], None]):
        """
        Add a callback function to be called when metrics are updated
        
        Args:
            callback: Function to call with Metric
        """
        self.metric_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[str, Metric], None]):
        """
        Add a callback function to be called when alerts are triggered
        
        Args:
            callback: Function to call with alert message and Metric
        """
        self.alert_callbacks.append(callback)
    
    def set_alert_threshold(self, metric_name: str, threshold_type: str, value: float):
        """
        Set an alert threshold for a metric
        
        Args:
            metric_name: Name of the metric
            threshold_type: Type of threshold (min, max, critical_min, critical_max)
            value: Threshold value
        """
        if metric_name not in self.config.alert_thresholds:
            self.config.alert_thresholds[metric_name] = {}
        
        self.config.alert_thresholds[metric_name][threshold_type] = value
        logger.info(f"Set {threshold_type} threshold for {metric_name}: {value}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Update dashboard statistics
                self._update_dashboard_stats()
                
                self.update_count += 1
                self.last_update = datetime.now()
                
                await asyncio.sleep(self.config.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dashboard monitoring loop: {str(e)}")
                await asyncio.sleep(1.0)
    
    def _update_system_metrics(self):
        """Update system-level metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.add_metric("cpu_usage", cpu_percent, "%", "system")
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.add_metric("memory_usage", memory.percent, "%", "memory")
            self.add_metric("memory_available", memory.available / (1024**3), "GB", "memory")
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.add_metric("disk_usage", disk.percent, "%", "system")
            self.add_metric("disk_free", disk.free / (1024**3), "GB", "system")
            
            # Network I/O
            network = psutil.net_io_counters()
            self.add_metric("network_bytes_sent", network.bytes_sent / (1024**2), "MB", "network")
            self.add_metric("network_bytes_recv", network.bytes_recv / (1024**2), "MB", "network")
            
        except ImportError:
            # psutil not available, use mock data
            self.add_metric("cpu_usage", 50.0, "%", "system")
            self.add_metric("memory_usage", 60.0, "%", "memory")
            self.add_metric("disk_usage", 70.0, "%", "system")
    
    def _update_dashboard_stats(self):
        """Update dashboard performance statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        updates_per_second = self.update_count / uptime if uptime > 0 else 0
        
        self.add_metric("dashboard_uptime", uptime / 3600, "hours", "performance")
        self.add_metric("updates_per_second", updates_per_second, "updates/s", "performance")
        self.add_metric("active_metrics", len(self.current_metrics), "count", "performance")
    
    def _check_alert_thresholds(self, metric_name: str, metric: Metric):
        """Check if metric exceeds alert thresholds"""
        if metric_name not in self.config.alert_thresholds:
            return
        
        thresholds = self.config.alert_thresholds[metric_name]
        
        for threshold_type, threshold_value in thresholds.items():
            triggered = False
            alert_message = ""
            
            if threshold_type == "min" and metric.value < threshold_value:
                triggered = True
                alert_message = f"{metric_name} below minimum threshold: {metric.value} {metric.unit} < {threshold_value}"
            elif threshold_type == "max" and metric.value > threshold_value:
                triggered = True
                alert_message = f"{metric_name} above maximum threshold: {metric.value} {metric.unit} > {threshold_value}"
            elif threshold_type == "critical_min" and metric.value < threshold_value:
                triggered = True
                alert_message = f"CRITICAL: {metric_name} below critical minimum: {metric.value} {metric.unit} < {threshold_value}"
            elif threshold_type == "critical_max" and metric.value > threshold_value:
                triggered = True
                alert_message = f"CRITICAL: {metric_name} above critical maximum: {metric.value} {metric.unit} > {threshold_value}"
            
            if triggered:
                for callback in self.alert_callbacks:
                    try:
                        callback(alert_message, metric)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {str(e)}")
                
                logger.warning(f"Alert triggered: {alert_message}")
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """
        Get current value of a metric
        
        Args:
            name: Metric name
            
        Returns:
            Current metric value or None if not found
        """
        return self.current_metrics.get(name)
    
    def get_metric_history(self, name: str, limit: int = 100) -> List[Metric]:
        """
        Get historical values for a metric
        
        Args:
            name: Metric name
            limit: Maximum number of historical values to return
            
        Returns:
            List of historical metric values
        """
        if name not in self.metrics_history:
            return []
        
        history = list(self.metrics_history[name])
        return history[-limit:] if limit > 0 else history
    
    def get_metrics_by_category(self, category: str) -> Dict[str, Metric]:
        """
        Get all current metrics in a specific category
        
        Args:
            category: Metric category
            
        Returns:
            Dictionary of metrics in the category
        """
        return {
            name: metric for name, metric in self.current_metrics.items()
            if metric.category == category
        }
    
    def get_metric_statistics(self, name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get statistical summary for a metric over a time window
        
        Args:
            name: Metric name
            window_minutes: Time window in minutes
            
        Returns:
            Dictionary with statistical summary
        """
        if name not in self.metrics_history:
            return {}
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [
            m for m in self.metrics_history[name]
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        values = [m.value for m in recent_metrics]
        
        return {
            'count': len(values),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'window_minutes': window_minutes
        }
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get overall dashboard summary"""
        categories = defaultdict(int)
        for metric in self.current_metrics.values():
            categories[metric.category] += 1
        
        return {
            'total_metrics': len(self.current_metrics),
            'categories': dict(categories),
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'update_count': self.update_count,
            'last_update': self.last_update.isoformat(),
            'is_monitoring': self.is_monitoring
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        report = {
            'dashboard_summary': self.get_dashboard_summary(),
            'system_metrics': {},
            'performance_metrics': {},
            'memory_metrics': {},
            'network_metrics': {}
        }
        
        # Add current metrics by category
        for category in ['system', 'performance', 'memory', 'network']:
            category_metrics = self.get_metrics_by_category(category)
            report[f'{category}_metrics'] = {
                name: {
                    'value': metric.value,
                    'unit': metric.unit,
                    'timestamp': metric.timestamp.isoformat()
                }
                for name, metric in category_metrics.items()
            }
        
        return report
    
    def clear_metrics(self, category: Optional[str] = None):
        """
        Clear metrics history
        
        Args:
            category: Optional category to clear (if None, clears all)
        """
        if category:
            for name in list(self.metrics_history.keys()):
                if self.current_metrics.get(name, {}).category == category:
                    del self.metrics_history[name]
                    if name in self.current_metrics:
                        del self.current_metrics[name]
            logger.info(f"Cleared metrics for category: {category}")
        else:
            self.metrics_history.clear()
            self.current_metrics.clear()
            logger.info("Cleared all metrics")
    
    def export_metrics(self, format: str = "json") -> str:
        """
        Export metrics data
        
        Args:
            format: Export format (json, csv)
            
        Returns:
            Exported data as string
        """
        if format.lower() == "json":
            import json
            data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    name: {
                        'value': metric.value,
                        'unit': metric.unit,
                        'category': metric.category,
                        'timestamp': metric.timestamp.isoformat(),
                        'metadata': metric.metadata
                    }
                    for name, metric in self.current_metrics.items()
                }
            }
            return json.dumps(data, indent=2)
        
        elif format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['name', 'value', 'unit', 'category', 'timestamp'])
            
            for name, metric in self.current_metrics.items():
                writer.writerow([
                    name, metric.value, metric.unit, metric.category,
                    metric.timestamp.isoformat()
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
