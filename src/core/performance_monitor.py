"""
Performance monitoring service for multilingual content processing.
Tracks processing times, memory usage, error rates, and performance metrics.
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Represents a performance metric."""
    operation: str
    language: str
    duration: float
    timestamp: datetime
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryMetric:
    """Represents a memory usage metric."""
    memory_mb: float
    memory_percentage: float
    timestamp: datetime
    language: str = "general"


@dataclass
class ErrorMetric:
    """Represents an error metric."""
    operation: str
    language: str
    error_type: str
    error_message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Performance monitoring system for multilingual processing."""
    
    def __init__(self, 
                 max_history_size: int = 1000,
                 enable_real_time_monitoring: bool = True,
                 alert_threshold_seconds: float = 10.0):
        self.max_history_size = max_history_size
        self.enable_real_time_monitoring = enable_real_time_monitoring
        self.alert_threshold_seconds = alert_threshold_seconds
        
        # Performance metrics storage
        self.performance_metrics: deque = deque(maxlen=max_history_size)
        self.memory_metrics: deque = deque(maxlen=max_history_size)
        self.error_metrics: deque = deque(maxlen=max_history_size)
        
        # Real-time statistics
        self.real_time_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "peak_processing_time": 0.0,
            "current_memory_mb": 0.0,
            "peak_memory_mb": 0.0,
            "error_rate": 0.0
        }
        
        # Language-specific statistics
        self.language_stats = defaultdict(lambda: {
            "operations": 0,
            "successful": 0,
            "failed": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "peak_time": 0.0,
            "error_rate": 0.0
        })
        
        # Operation-specific statistics
        self.operation_stats = defaultdict(lambda: {
            "count": 0,
            "successful": 0,
            "failed": 0,
            "total_time": 0.0,
            "average_time": 0.0,
            "peak_time": 0.0,
            "error_rate": 0.0
        })
        
        # Performance alerts
        self.alerts = []
        self.max_alerts = 100
        
        # Start monitoring if enabled
        if self.enable_real_time_monitoring:
            try:
                asyncio.create_task(self._monitor_performance())
            except RuntimeError:
                # No running event loop, will start monitoring when needed
                pass
    
    async def track_processing_time(self, operation: str, language: str = "en"):
        """Track processing time for operations."""
        start_time = time.time()
        
        async def end_tracking(success: bool = True, error: str = None, metadata: Dict[str, Any] = None):
            end_time = time.time()
            duration = end_time - start_time
            
            # Create performance metric
            metric = PerformanceMetric(
                operation=operation,
                language=language,
                duration=duration,
                timestamp=datetime.now(),
                success=success,
                error=error,
                metadata=metadata or {}
            )
            
            # Store metric
            self.performance_metrics.append(metric)
            
            # Update real-time statistics
            self._update_real_time_stats(metric)
            
            # Update language-specific statistics
            self._update_language_stats(metric)
            
            # Update operation-specific statistics
            self._update_operation_stats(metric)
            
            # Check for performance alerts
            if duration > self.alert_threshold_seconds:
                await self._create_performance_alert(metric)
            
            return metric
        
        return end_tracking
    
    async def track_memory_usage(self, memory_mb: float, language: str = "en"):
        """Track memory usage."""
        try:
            # Create memory metric
            metric = MemoryMetric(
                memory_mb=memory_mb,
                memory_percentage=(memory_mb / 1024) * 100,  # Assuming 1GB max
                timestamp=datetime.now(),
                language=language
            )
            
            # Store metric
            self.memory_metrics.append(metric)
            
            # Update real-time statistics
            self.real_time_stats["current_memory_mb"] = memory_mb
            self.real_time_stats["peak_memory_mb"] = max(
                self.real_time_stats["peak_memory_mb"], 
                memory_mb
            )
            
        except Exception as e:
            logger.error(f"Error tracking memory usage: {e}")
    
    async def track_error(self, operation: str, language: str, error_type: str, error_message: str, metadata: Dict[str, Any] = None):
        """Track error occurrences."""
        try:
            # Create error metric
            metric = ErrorMetric(
                operation=operation,
                language=language,
                error_type=error_type,
                error_message=error_message,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            # Store metric
            self.error_metrics.append(metric)
            
            # Update error statistics
            self.real_time_stats["failed_operations"] += 1
            self._update_error_rate()
            
        except Exception as e:
            logger.error(f"Error tracking error: {e}")
    
    def get_performance_report(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            
            # Filter metrics by time window
            recent_performance = [
                m for m in self.performance_metrics 
                if m.timestamp >= cutoff_time
            ]
            recent_memory = [
                m for m in self.memory_metrics 
                if m.timestamp >= cutoff_time
            ]
            recent_errors = [
                m for m in self.error_metrics 
                if m.timestamp >= cutoff_time
            ]
            
            # Calculate statistics
            if recent_performance:
                processing_times = [m.duration for m in recent_performance]
                avg_processing_time = statistics.mean(processing_times)
                median_processing_time = statistics.median(processing_times)
                min_processing_time = min(processing_times)
                max_processing_time = max(processing_times)
            else:
                avg_processing_time = median_processing_time = min_processing_time = max_processing_time = 0.0
            
            if recent_memory:
                memory_usage = [m.memory_mb for m in recent_memory]
                avg_memory = statistics.mean(memory_usage)
                peak_memory = max(memory_usage)
            else:
                avg_memory = peak_memory = 0.0
            
            # Calculate success rate
            total_operations = len(recent_performance)
            successful_operations = len([m for m in recent_performance if m.success])
            success_rate = (successful_operations / total_operations * 100) if total_operations > 0 else 0.0
            
            # Language breakdown
            language_breakdown = {}
            for lang in set(m.language for m in recent_performance):
                lang_metrics = [m for m in recent_performance if m.language == lang]
                if lang_metrics:
                    lang_times = [m.duration for m in lang_metrics]
                    lang_success = len([m for m in lang_metrics if m.success])
                    
                    language_breakdown[lang] = {
                        "operations": len(lang_metrics),
                        "success_rate": (lang_success / len(lang_metrics)) * 100,
                        "average_time": statistics.mean(lang_times),
                        "peak_time": max(lang_times),
                        "total_time": sum(lang_times)
                    }
            
            # Operation breakdown
            operation_breakdown = {}
            for op in set(m.operation for m in recent_performance):
                op_metrics = [m for m in recent_performance if m.operation == op]
                if op_metrics:
                    op_times = [m.duration for m in op_metrics]
                    op_success = len([m for m in op_metrics if m.success])
                    
                    operation_breakdown[op] = {
                        "count": len(op_metrics),
                        "success_rate": (op_success / len(op_metrics)) * 100,
                        "average_time": statistics.mean(op_times),
                        "peak_time": max(op_times),
                        "total_time": sum(op_times)
                    }
            
            return {
                "time_window_minutes": time_window_minutes,
                "summary": {
                    "total_operations": total_operations,
                    "successful_operations": successful_operations,
                    "failed_operations": total_operations - successful_operations,
                    "success_rate": success_rate,
                    "average_processing_time": avg_processing_time,
                    "median_processing_time": median_processing_time,
                    "min_processing_time": min_processing_time,
                    "max_processing_time": max_processing_time,
                    "average_memory_mb": avg_memory,
                    "peak_memory_mb": peak_memory,
                    "error_count": len(recent_errors)
                },
                "language_breakdown": language_breakdown,
                "operation_breakdown": operation_breakdown,
                "recent_alerts": self.alerts[-10:],  # Last 10 alerts
                "real_time_stats": self.real_time_stats,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": str(e)}
    
    def get_language_performance(self, language: str, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance statistics for a specific language."""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            
            # Filter metrics for the language
            lang_performance = [
                m for m in self.performance_metrics 
                if m.language == language and m.timestamp >= cutoff_time
            ]
            
            if not lang_performance:
                return {
                    "language": language,
                    "message": "No performance data available for this language in the specified time window"
                }
            
            # Calculate statistics
            processing_times = [m.duration for m in lang_performance]
            successful_operations = len([m for m in lang_performance if m.success])
            
            return {
                "language": language,
                "time_window_minutes": time_window_minutes,
                "total_operations": len(lang_performance),
                "successful_operations": successful_operations,
                "failed_operations": len(lang_performance) - successful_operations,
                "success_rate": (successful_operations / len(lang_performance)) * 100,
                "average_processing_time": statistics.mean(processing_times),
                "median_processing_time": statistics.median(processing_times),
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times),
                "total_processing_time": sum(processing_times),
                "language_stats": self.language_stats[language]
            }
            
        except Exception as e:
            logger.error(f"Error getting language performance for {language}: {e}")
            return {"error": str(e)}
    
    def get_operation_performance(self, operation: str, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance statistics for a specific operation."""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            
            # Filter metrics for the operation
            op_performance = [
                m for m in self.performance_metrics 
                if m.operation == operation and m.timestamp >= cutoff_time
            ]
            
            if not op_performance:
                return {
                    "operation": operation,
                    "message": "No performance data available for this operation in the specified time window"
                }
            
            # Calculate statistics
            processing_times = [m.duration for m in op_performance]
            successful_operations = len([m for m in op_performance if m.success])
            
            return {
                "operation": operation,
                "time_window_minutes": time_window_minutes,
                "total_operations": len(op_performance),
                "successful_operations": successful_operations,
                "failed_operations": len(op_performance) - successful_operations,
                "success_rate": (successful_operations / len(op_performance)) * 100,
                "average_processing_time": statistics.mean(processing_times),
                "median_processing_time": statistics.median(processing_times),
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times),
                "total_processing_time": sum(processing_times),
                "operation_stats": self.operation_stats[operation]
            }
            
        except Exception as e:
            logger.error(f"Error getting operation performance for {operation}: {e}")
            return {"error": str(e)}
    
    def get_alerts(self, max_alerts: int = 50) -> List[Dict[str, Any]]:
        """Get recent performance alerts."""
        return self.alerts[-max_alerts:]
    
    def clear_history(self):
        """Clear all performance history."""
        self.performance_metrics.clear()
        self.memory_metrics.clear()
        self.error_metrics.clear()
        self.alerts.clear()
        
        # Reset statistics
        self.real_time_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "peak_processing_time": 0.0,
            "current_memory_mb": 0.0,
            "peak_memory_mb": 0.0,
            "error_rate": 0.0
        }
        
        self.language_stats.clear()
        self.operation_stats.clear()
        
        logger.info("Performance history cleared")
    
    def _update_real_time_stats(self, metric: PerformanceMetric):
        """Update real-time statistics with a new metric."""
        self.real_time_stats["total_operations"] += 1
        self.real_time_stats["total_processing_time"] += metric.duration
        
        if metric.success:
            self.real_time_stats["successful_operations"] += 1
        else:
            self.real_time_stats["failed_operations"] += 1
        
        # Update average processing time
        total_ops = self.real_time_stats["total_operations"]
        if total_ops > 0:
            self.real_time_stats["average_processing_time"] = (
                self.real_time_stats["total_processing_time"] / total_ops
            )
        
        # Update peak processing time
        self.real_time_stats["peak_processing_time"] = max(
            self.real_time_stats["peak_processing_time"],
            metric.duration
        )
        
        # Update error rate
        self._update_error_rate()
    
    def _update_language_stats(self, metric: PerformanceMetric):
        """Update language-specific statistics."""
        lang_stats = self.language_stats[metric.language]
        lang_stats["operations"] += 1
        lang_stats["total_time"] += metric.duration
        
        if metric.success:
            lang_stats["successful"] += 1
        else:
            lang_stats["failed"] += 1
        
        # Update averages
        if lang_stats["operations"] > 0:
            lang_stats["average_time"] = lang_stats["total_time"] / lang_stats["operations"]
            lang_stats["error_rate"] = (lang_stats["failed"] / lang_stats["operations"]) * 100
        
        # Update peak time
        lang_stats["peak_time"] = max(lang_stats["peak_time"], metric.duration)
    
    def _update_operation_stats(self, metric: PerformanceMetric):
        """Update operation-specific statistics."""
        op_stats = self.operation_stats[metric.operation]
        op_stats["count"] += 1
        op_stats["total_time"] += metric.duration
        
        if metric.success:
            op_stats["successful"] += 1
        else:
            op_stats["failed"] += 1
        
        # Update averages
        if op_stats["count"] > 0:
            op_stats["average_time"] = op_stats["total_time"] / op_stats["count"]
            op_stats["error_rate"] = (op_stats["failed"] / op_stats["count"]) * 100
        
        # Update peak time
        op_stats["peak_time"] = max(op_stats["peak_time"], metric.duration)
    
    def _update_error_rate(self):
        """Update overall error rate."""
        total_ops = self.real_time_stats["total_operations"]
        failed_ops = self.real_time_stats["failed_operations"]
        
        if total_ops > 0:
            self.real_time_stats["error_rate"] = (failed_ops / total_ops) * 100
    
    async def _create_performance_alert(self, metric: PerformanceMetric):
        """Create a performance alert."""
        try:
            alert = {
                "timestamp": metric.timestamp.isoformat(),
                "operation": metric.operation,
                "language": metric.language,
                "duration": metric.duration,
                "threshold": self.alert_threshold_seconds,
                "message": f"Operation '{metric.operation}' for language '{metric.language}' took {metric.duration:.2f}s (threshold: {self.alert_threshold_seconds}s)",
                "severity": "warning" if metric.duration < self.alert_threshold_seconds * 2 else "critical"
            }
            
            self.alerts.append(alert)
            
            # Keep alerts list manageable
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]
            
            logger.warning(f"Performance alert: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Error creating performance alert: {e}")
    
    async def _monitor_performance(self):
        """Background performance monitoring task."""
        while True:
            try:
                # Update memory usage if available
                # This would integrate with the actual memory manager
                # For now, just log the monitoring
                logger.debug("Performance monitoring check completed")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(120)  # Wait longer on error


# Global performance monitor instance
_global_performance_monitor: Optional[PerformanceMonitor] = None


def get_global_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    return _global_performance_monitor


def set_global_performance_monitor(monitor: PerformanceMonitor):
    """Set global performance monitor instance."""
    global _global_performance_monitor
    _global_performance_monitor = monitor
