"""
Application Performance Monitoring System

This module provides comprehensive application monitoring including:
- Performance metrics tracking
- Error tracking and analysis
- User analytics
- Custom metrics collection
- Alerting rules and notifications
"""

import time
import statistics
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import psutil
import threading
from collections import defaultdict, deque

from loguru import logger

from src.core.error_handler import with_error_handling


@dataclass
class PerformanceMetric:
    """Represents a performance metric."""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    category: str  # cpu, memory, disk, network, custom
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorRecord:
    """Represents an error record."""
    error_id: str
    error_type: str
    error_message: str
    stack_trace: str
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    severity: str = "error"  # error, warning, info
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserAnalytics:
    """Represents user analytics data."""
    user_id: str
    action: str
    timestamp: datetime
    session_id: Optional[str] = None
    page_url: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Represents an alert rule."""
    rule_id: str
    rule_name: str
    metric_name: str
    condition: str  # >, <, >=, <=, ==, !=
    threshold: float
    duration: int  # seconds to check
    severity: str  # low, medium, high, critical
    notification_channels: List[str] = field(default_factory=list)
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class ApplicationMonitor:
    """
    Comprehensive application performance monitoring system.
    """
    
    def __init__(self):
        self.performance_metrics: List[PerformanceMetric] = []
        self.error_records: List[ErrorRecord] = []
        self.user_analytics: List[UserAnalytics] = []
        self.custom_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_handlers: Dict[str, Callable] = {}
        
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = 30  # seconds
        
        # Initialize default alert handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default alert handlers."""
        self.alert_handlers["log"] = self._log_alert
        self.alert_handlers["email"] = self._email_alert
        self.alert_handlers["slack"] = self._slack_alert
    
    @with_error_handling("start_monitoring")
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Application monitoring started")
    
    @with_error_handling("stop_monitoring")
    def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self.monitoring_active:
            logger.warning("Monitoring is not active")
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Application monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._collect_system_metrics()
                self._check_alert_rules()
                self._cleanup_old_data()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    @with_error_handling("collect_system_metrics")
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.performance_metrics.append(PerformanceMetric(
            metric_name="cpu_usage",
            value=cpu_percent,
            unit="percent",
            timestamp=datetime.now(),
            category="system"
        ))
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.performance_metrics.append(PerformanceMetric(
            metric_name="memory_usage",
            value=memory.percent,
            unit="percent",
            timestamp=datetime.now(),
            category="system"
        ))
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.performance_metrics.append(PerformanceMetric(
            metric_name="disk_usage",
            value=disk_percent,
            unit="percent",
            timestamp=datetime.now(),
            category="system"
        ))
        
        # Network I/O
        network = psutil.net_io_counters()
        self.performance_metrics.append(PerformanceMetric(
            metric_name="network_bytes_sent",
            value=network.bytes_sent,
            unit="bytes",
            timestamp=datetime.now(),
            category="system"
        ))
        self.performance_metrics.append(PerformanceMetric(
            metric_name="network_bytes_recv",
            value=network.bytes_recv,
            unit="bytes",
            timestamp=datetime.now(),
            category="system"
        ))
    
    @with_error_handling("record_error")
    def record_error(self, error_type: str, error_message: str, 
                    stack_trace: str = "", user_id: Optional[str] = None,
                    session_id: Optional[str] = None, severity: str = "error",
                    metadata: Optional[Dict[str, Any]] = None):
        """Record an error."""
        error_id = f"error_{int(time.time())}_{hash(error_message) % 10000}"
        
        error_record = ErrorRecord(
            error_id=error_id,
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            severity=severity,
            metadata=metadata or {}
        )
        
        self.error_records.append(error_record)
        logger.error(f"Error recorded: {error_type} - {error_message}")
    
    @with_error_handling("record_user_action")
    def record_user_action(self, user_id: str, action: str,
                          session_id: Optional[str] = None,
                          page_url: Optional[str] = None,
                          user_agent: Optional[str] = None,
                          ip_address: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None):
        """Record a user action."""
        user_analytics = UserAnalytics(
            user_id=user_id,
            action=action,
            timestamp=datetime.now(),
            session_id=session_id,
            page_url=page_url,
            user_agent=user_agent,
            ip_address=ip_address,
            metadata=metadata or {}
        )
        
        self.user_analytics.append(user_analytics)
        logger.debug(f"User action recorded: {user_id} - {action}")
    
    @with_error_handling("add_alert_rule")
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Alert rule added: {rule.rule_name}")
    
    def _check_alert_rules(self):
        """Check all alert rules and trigger alerts if needed."""
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Get recent metrics for this rule
            recent_metrics = [
                m for m in self.performance_metrics
                if m.metric_name == rule.metric_name
                and m.timestamp > datetime.now() - timedelta(
                    seconds=rule.duration
                )
            ]
            
            if not recent_metrics:
                continue
            
            # Calculate average value
            avg_value = statistics.mean([m.value for m in recent_metrics])
            
            # Check condition
            should_alert = False
            if rule.condition == ">":
                should_alert = avg_value > rule.threshold
            elif rule.condition == "<":
                should_alert = avg_value < rule.threshold
            elif rule.condition == ">=":
                should_alert = avg_value >= rule.threshold
            elif rule.condition == "<=":
                should_alert = avg_value <= rule.threshold
            elif rule.condition == "==":
                should_alert = avg_value == rule.threshold
            elif rule.condition == "!=":
                should_alert = avg_value != rule.threshold
            
            if should_alert:
                # Use asyncio.run for async call in sync context
                import asyncio
                try:
                    asyncio.run(self._trigger_alert(rule, avg_value))
                except RuntimeError:
                    # If already in event loop, create task
                    loop = asyncio.get_event_loop()
                    loop.create_task(self._trigger_alert(rule, avg_value))
    
    @with_error_handling("trigger_alert")
    async def _trigger_alert(self, rule: AlertRule, current_value: float):
        """Trigger an alert."""
        alert_id = f"alert_{int(time.time())}_{hash(rule.rule_name) % 10000}"
        
        alert_data = {
            "alert_id": alert_id,
            "rule_id": rule.rule_id,
            "rule_name": rule.rule_name,
            "metric_name": rule.metric_name,
            "current_value": current_value,
            "threshold": rule.threshold,
            "condition": rule.condition,
            "severity": rule.severity,
            "timestamp": datetime.now(),
            "notification_channels": rule.notification_channels
        }
        
        self.active_alerts[alert_id] = alert_data
        
        # Send notifications
        for channel in rule.notification_channels:
            if channel in self.alert_handlers:
                try:
                    await self.alert_handlers[channel](alert_data)
                except Exception as e:
                    logger.error(f"Error sending alert via {channel}: {e}")
        
        logger.warning(
            f"Alert triggered: {rule.rule_name} - "
            f"{rule.metric_name} {rule.condition} {rule.threshold}"
        )
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Clean up performance metrics
        self.performance_metrics = [
            m for m in self.performance_metrics
            if m.timestamp > cutoff_time
        ]
        
        # Clean up error records
        self.error_records = [
            e for e in self.error_records
            if e.timestamp > cutoff_time
        ]
        
        # Clean up user analytics
        self.user_analytics = [
            u for u in self.user_analytics
            if u.timestamp > cutoff_time
        ]
    
    @with_error_handling("get_performance_summary")
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of current performance metrics."""
        if not self.performance_metrics:
            return {"error": "No performance metrics available"}
        
        # Get recent metrics (last hour)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_metrics = [
            m for m in self.performance_metrics
            if m.timestamp > recent_cutoff
        ]
        
        summary = {
            "timestamp": datetime.now(),
            "metrics": {},
            "alerts": {
                "active": len(self.active_alerts),
                "total_rules": len(self.alert_rules)
            },
            "errors": {
                "total": len(self.error_records),
                "recent": len([
                    e for e in self.error_records 
                    if e.timestamp > recent_cutoff
                ])
            },
            "users": {
                "total_actions": len(self.user_analytics),
                "recent_actions": len([
                    u for u in self.user_analytics 
                    if u.timestamp > recent_cutoff
                ])
            }
        }
        
        # Calculate metrics by category
        for category in ["system", "custom"]:
            category_metrics = [
                m for m in recent_metrics if m.category == category
            ]
            if category_metrics:
                summary["metrics"][category] = {}
                for metric in category_metrics:
                    if metric.metric_name not in summary["metrics"][category]:
                        summary["metrics"][category][metric.metric_name] = {
                            "current": metric.value,
                            "unit": metric.unit,
                            "count": 1
                        }
                    else:
                        summary["metrics"][category][metric.metric_name]["count"] += 1
        
        return summary
    
    @with_error_handling("get_error_analysis")
    async def get_error_analysis(self) -> Dict[str, Any]:
        """Get error analysis and statistics."""
        if not self.error_records:
            return {"error": "No error records available"}
        
        # Get recent errors (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_errors = [
            e for e in self.error_records
            if e.timestamp > recent_cutoff
        ]
        
        # Group by error type
        error_types = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for error in recent_errors:
            error_types[error.error_type] += 1
            severity_counts[error.severity] += 1
        
        analysis = {
            "timestamp": datetime.now(),
            "total_errors": len(recent_errors),
            "error_types": dict(error_types),
            "severity_distribution": dict(severity_counts),
            "top_errors": sorted(
                error_types.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
        
        return analysis
    
    @with_error_handling("get_user_analytics")
    async def get_user_analytics(self) -> Dict[str, Any]:
        """Get user analytics summary."""
        if not self.user_analytics:
            return {"error": "No user analytics available"}
        
        # Get recent analytics (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_analytics = [
            u for u in self.user_analytics
            if u.timestamp > recent_cutoff
        ]
        
        # Group by action
        action_counts = defaultdict(int)
        user_counts = defaultdict(int)
        
        for analytics in recent_analytics:
            action_counts[analytics.action] += 1
            user_counts[analytics.user_id] += 1
        
        analytics_summary = {
            "timestamp": datetime.now(),
            "total_actions": len(recent_analytics),
            "unique_users": len(user_counts),
            "action_distribution": dict(action_counts),
            "top_actions": sorted(
                action_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            "top_users": sorted(
                user_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
        
        return analytics_summary
    
    # Default alert handlers
    async def _log_alert(self, alert_data: Dict[str, Any]):
        """Log alert to console."""
        logger.warning(f"ALERT: {alert_data['rule_name']} - "
                      f"{alert_data['metric_name']} = {alert_data['current_value']}")
    
    async def _email_alert(self, alert_data: Dict[str, Any]):
        """Send alert via email (placeholder)."""
        logger.info(f"Email alert would be sent: {alert_data['rule_name']}")
    
    async def _slack_alert(self, alert_data: Dict[str, Any]):
        """Send alert via Slack (placeholder)."""
        logger.info(f"Slack alert would be sent: {alert_data['rule_name']}")


# Global instance
application_monitor = ApplicationMonitor()
