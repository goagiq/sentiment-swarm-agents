"""
Infrastructure Monitoring System

This module provides comprehensive infrastructure monitoring including:
- Server monitoring (CPU, memory, disk, network)
- Database monitoring (connections, queries, performance)
- Network monitoring (bandwidth, latency, connectivity)
- Log aggregation and analysis
- Infrastructure alerts and notifications
"""

import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import psutil
import threading
import subprocess
import socket
from collections import defaultdict, deque

from loguru import logger

from src.core.error_handler import with_error_handling


@dataclass
class ServerMetric:
    """Represents a server performance metric."""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    server_id: str
    category: str  # cpu, memory, disk, network
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatabaseMetric:
    """Represents a database performance metric."""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    database_id: str
    connection_id: Optional[str] = None
    query_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkMetric:
    """Represents a network performance metric."""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    source_host: str
    target_host: str
    protocol: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogEntry:
    """Represents a log entry."""
    log_id: str
    timestamp: datetime
    level: str  # debug, info, warning, error, critical
    source: str
    message: str
    host: str
    service: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InfrastructureAlert:
    """Represents an infrastructure alert."""
    alert_id: str
    alert_type: str  # server, database, network, log
    severity: str  # low, medium, high, critical
    message: str
    timestamp: datetime
    source_id: str
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class InfrastructureMonitor:
    """
    Comprehensive infrastructure monitoring system.
    """
    
    def __init__(self):
        self.server_metrics: List[ServerMetric] = []
        self.database_metrics: List[DatabaseMetric] = []
        self.network_metrics: List[NetworkMetric] = []
        self.log_entries: List[LogEntry] = []
        self.infrastructure_alerts: List[InfrastructureAlert] = []
        
        # Monitoring configuration
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = 30  # seconds
        
        # Server configuration
        self.servers = {
            "localhost": {
                "host": "localhost",
                "port": 22,
                "enabled": True
            }
        }
        
        # Database configuration
        self.databases = {
            "default": {
                "host": "localhost",
                "port": 5432,
                "type": "postgresql",
                "enabled": True
            }
        }
        
        # Network targets
        self.network_targets = [
            "8.8.8.8",  # Google DNS
            "1.1.1.1",  # Cloudflare DNS
            "localhost"
        ]
        
        # Alert thresholds
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "network_latency": 100.0,  # ms
            "database_connections": 100
        }
    
    @with_error_handling("start_monitoring")
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            logger.warning("Infrastructure monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Infrastructure monitoring started")
    
    @with_error_handling("stop_monitoring")
    def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self.monitoring_active:
            logger.warning("Infrastructure monitoring is not active")
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Infrastructure monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._monitor_servers()
                self._monitor_databases()
                self._monitor_network()
                self._aggregate_logs()
                self._check_infrastructure_alerts()
                self._cleanup_old_data()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in infrastructure monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    @with_error_handling("monitor_servers")
    def _monitor_servers(self):
        """Monitor server performance metrics."""
        for server_id, server_config in self.servers.items():
            if not server_config.get("enabled", True):
                continue
            
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.server_metrics.append(ServerMetric(
                    metric_name="cpu_usage",
                    value=cpu_percent,
                    unit="percent",
                    timestamp=datetime.now(),
                    server_id=server_id,
                    category="cpu"
                ))
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.server_metrics.append(ServerMetric(
                    metric_name="memory_usage",
                    value=memory.percent,
                    unit="percent",
                    timestamp=datetime.now(),
                    server_id=server_id,
                    category="memory"
                ))
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.server_metrics.append(ServerMetric(
                    metric_name="disk_usage",
                    value=disk_percent,
                    unit="percent",
                    timestamp=datetime.now(),
                    server_id=server_id,
                    category="disk"
                ))
                
                # Network I/O
                network = psutil.net_io_counters()
                self.server_metrics.append(ServerMetric(
                    metric_name="network_bytes_sent",
                    value=network.bytes_sent,
                    unit="bytes",
                    timestamp=datetime.now(),
                    server_id=server_id,
                    category="network"
                ))
                self.server_metrics.append(ServerMetric(
                    metric_name="network_bytes_recv",
                    value=network.bytes_recv,
                    unit="bytes",
                    timestamp=datetime.now(),
                    server_id=server_id,
                    category="network"
                ))
                
            except Exception as e:
                logger.error(f"Error monitoring server {server_id}: {e}")
    
    @with_error_handling("monitor_databases")
    def _monitor_databases(self):
        """Monitor database performance metrics."""
        for db_id, db_config in self.databases.items():
            if not db_config.get("enabled", True):
                continue
            
            try:
                # Simulate database metrics (in real implementation, 
                # connect to actual databases)
                import random
                
                # Connection count
                connection_count = random.randint(10, 50)
                self.database_metrics.append(DatabaseMetric(
                    metric_name="connection_count",
                    value=connection_count,
                    unit="connections",
                    timestamp=datetime.now(),
                    database_id=db_id
                ))
                
                # Query response time
                query_time = random.uniform(0.1, 2.0)
                self.database_metrics.append(DatabaseMetric(
                    metric_name="query_response_time",
                    value=query_time,
                    unit="seconds",
                    timestamp=datetime.now(),
                    database_id=db_id,
                    query_type="select"
                ))
                
                # Active transactions
                active_transactions = random.randint(0, 20)
                self.database_metrics.append(DatabaseMetric(
                    metric_name="active_transactions",
                    value=active_transactions,
                    unit="transactions",
                    timestamp=datetime.now(),
                    database_id=db_id
                ))
                
            except Exception as e:
                logger.error(f"Error monitoring database {db_id}: {e}")
    
    @with_error_handling("monitor_network")
    def _monitor_network(self):
        """Monitor network connectivity and performance."""
        for target in self.network_targets:
            try:
                # Ping test
                start_time = time.time()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((target, 80))
                sock.close()
                
                if result == 0:
                    latency = (time.time() - start_time) * 1000  # Convert to ms
                    self.network_metrics.append(NetworkMetric(
                        metric_name="ping_latency",
                        value=latency,
                        unit="milliseconds",
                        timestamp=datetime.now(),
                        source_host="localhost",
                        target_host=target,
                        protocol="tcp"
                    ))
                else:
                    # Connection failed
                    self.network_metrics.append(NetworkMetric(
                        metric_name="connection_status",
                        value=0,
                        unit="status",
                        timestamp=datetime.now(),
                        source_host="localhost",
                        target_host=target,
                        protocol="tcp"
                    ))
                
            except Exception as e:
                logger.error(f"Error monitoring network target {target}: {e}")
    
    @with_error_handling("aggregate_logs")
    def _aggregate_logs(self):
        """Aggregate and analyze log entries."""
        try:
            # Simulate log aggregation (in real implementation, 
            # read from actual log files)
            import random
            
            log_levels = ["debug", "info", "warning", "error", "critical"]
            services = ["web", "api", "database", "cache", "monitoring"]
            
            # Generate some sample log entries
            for _ in range(random.randint(1, 5)):
                log_entry = LogEntry(
                    log_id=f"log_{int(time.time())}_{random.randint(1000, 9999)}",
                    timestamp=datetime.now(),
                    level=random.choice(log_levels),
                    source="system",
                    message=f"Sample log message {random.randint(1, 100)}",
                    host="localhost",
                    service=random.choice(services)
                )
                self.log_entries.append(log_entry)
                
        except Exception as e:
            logger.error(f"Error aggregating logs: {e}")
    
    @with_error_handling("check_infrastructure_alerts")
    def _check_infrastructure_alerts(self):
        """Check for infrastructure alerts based on thresholds."""
        try:
            # Check server metrics
            recent_server_metrics = [
                m for m in self.server_metrics
                if m.timestamp > datetime.now() - timedelta(minutes=5)
            ]
            
            for metric in recent_server_metrics:
                threshold_key = metric.metric_name
                if threshold_key in self.alert_thresholds:
                    threshold = self.alert_thresholds[threshold_key]
                    
                    if metric.value > threshold:
                        self._trigger_infrastructure_alert(
                            alert_type="server",
                            severity="high",
                            message=f"{metric.metric_name} exceeded threshold: "
                                   f"{metric.value} > {threshold}",
                            source_id=metric.server_id,
                            metric_name=metric.metric_name,
                            current_value=metric.value,
                            threshold=threshold
                        )
            
            # Check database metrics
            recent_db_metrics = [
                m for m in self.database_metrics
                if m.timestamp > datetime.now() - timedelta(minutes=5)
            ]
            
            for metric in recent_db_metrics:
                if metric.metric_name == "connection_count":
                    threshold = self.alert_thresholds.get("database_connections", 100)
                    if metric.value > threshold:
                        self._trigger_infrastructure_alert(
                            alert_type="database",
                            severity="medium",
                            message=f"Database connections exceeded threshold: "
                                   f"{metric.value} > {threshold}",
                            source_id=metric.database_id,
                            metric_name=metric.metric_name,
                            current_value=metric.value,
                            threshold=threshold
                        )
            
            # Check network metrics
            recent_network_metrics = [
                m for m in self.network_metrics
                if m.timestamp > datetime.now() - timedelta(minutes=5)
            ]
            
            for metric in recent_network_metrics:
                if metric.metric_name == "ping_latency":
                    threshold = self.alert_thresholds.get("network_latency", 100)
                    if metric.value > threshold:
                        self._trigger_infrastructure_alert(
                            alert_type="network",
                            severity="medium",
                            message=f"Network latency exceeded threshold: "
                                   f"{metric.value}ms > {threshold}ms",
                            source_id=metric.target_host,
                            metric_name=metric.metric_name,
                            current_value=metric.value,
                            threshold=threshold
                        )
                        
        except Exception as e:
            logger.error(f"Error checking infrastructure alerts: {e}")
    
    @with_error_handling("trigger_infrastructure_alert")
    def _trigger_infrastructure_alert(self, alert_type: str, severity: str,
                                    message: str, source_id: str,
                                    metric_name: Optional[str] = None,
                                    current_value: Optional[float] = None,
                                    threshold: Optional[float] = None):
        """Trigger an infrastructure alert."""
        alert_id = f"infra_alert_{int(time.time())}_{hash(message) % 10000}"
        
        alert = InfrastructureAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            source_id=source_id,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold
        )
        
        self.infrastructure_alerts.append(alert)
        logger.warning(f"Infrastructure alert: {alert_type} - {message}")
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Clean up server metrics
        self.server_metrics = [
            m for m in self.server_metrics
            if m.timestamp > cutoff_time
        ]
        
        # Clean up database metrics
        self.database_metrics = [
            m for m in self.database_metrics
            if m.timestamp > cutoff_time
        ]
        
        # Clean up network metrics
        self.network_metrics = [
            m for m in self.network_metrics
            if m.timestamp > cutoff_time
        ]
        
        # Clean up log entries
        self.log_entries = [
            l for l in self.log_entries
            if l.timestamp > cutoff_time
        ]
        
        # Clean up alerts (keep for 7 days)
        alert_cutoff = datetime.now() - timedelta(days=7)
        self.infrastructure_alerts = [
            a for a in self.infrastructure_alerts
            if a.timestamp > alert_cutoff
        ]
    
    @with_error_handling("get_infrastructure_summary")
    async def get_infrastructure_summary(self) -> Dict[str, Any]:
        """Get a summary of infrastructure status."""
        # Get recent metrics (last hour)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        
        recent_server_metrics = [
            m for m in self.server_metrics
            if m.timestamp > recent_cutoff
        ]
        recent_db_metrics = [
            m for m in self.database_metrics
            if m.timestamp > recent_cutoff
        ]
        recent_network_metrics = [
            m for m in self.network_metrics
            if m.timestamp > recent_cutoff
        ]
        recent_logs = [
            l for l in self.log_entries
            if l.timestamp > recent_cutoff
        ]
        
        summary = {
            "timestamp": datetime.now(),
            "servers": {
                "total_metrics": len(recent_server_metrics),
                "servers_monitored": len(self.servers),
                "alerts": len([
                    a for a in self.infrastructure_alerts
                    if a.alert_type == "server" and a.timestamp > recent_cutoff
                ])
            },
            "databases": {
                "total_metrics": len(recent_db_metrics),
                "databases_monitored": len(self.databases),
                "alerts": len([
                    a for a in self.infrastructure_alerts
                    if a.alert_type == "database" and a.timestamp > recent_cutoff
                ])
            },
            "network": {
                "total_metrics": len(recent_network_metrics),
                "targets_monitored": len(self.network_targets),
                "alerts": len([
                    a for a in self.infrastructure_alerts
                    if a.alert_type == "network" and a.timestamp > recent_cutoff
                ])
            },
            "logs": {
                "total_entries": len(recent_logs),
                "error_count": len([
                    l for l in recent_logs if l.level in ["error", "critical"]
                ])
            },
            "alerts": {
                "total": len(self.infrastructure_alerts),
                "recent": len([
                    a for a in self.infrastructure_alerts
                    if a.timestamp > recent_cutoff
                ]),
                "by_severity": {}
            }
        }
        
        # Count alerts by severity
        for alert in self.infrastructure_alerts:
            if alert.timestamp > recent_cutoff:
                if alert.severity not in summary["alerts"]["by_severity"]:
                    summary["alerts"]["by_severity"][alert.severity] = 0
                summary["alerts"]["by_severity"][alert.severity] += 1
        
        return summary
    
    @with_error_handling("get_server_status")
    async def get_server_status(self, server_id: str) -> Dict[str, Any]:
        """Get detailed status for a specific server."""
        server_metrics = [
            m for m in self.server_metrics
            if m.server_id == server_id
        ]
        
        if not server_metrics:
            return {"error": f"No metrics found for server {server_id}"}
        
        # Get latest metrics for each category
        latest_metrics = {}
        for category in ["cpu", "memory", "disk", "network"]:
            category_metrics = [
                m for m in server_metrics
                if m.category == category
            ]
            if category_metrics:
                latest_metrics[category] = max(
                    category_metrics, key=lambda x: x.timestamp
                )
        
        status = {
            "server_id": server_id,
            "timestamp": datetime.now(),
            "metrics": latest_metrics,
            "alerts": [
                a for a in self.infrastructure_alerts
                if a.alert_type == "server" and a.source_id == server_id
            ]
        }
        
        return status
    
    @with_error_handling("get_database_status")
    async def get_database_status(self, database_id: str) -> Dict[str, Any]:
        """Get detailed status for a specific database."""
        db_metrics = [
            m for m in self.database_metrics
            if m.database_id == database_id
        ]
        
        if not db_metrics:
            return {"error": f"No metrics found for database {database_id}"}
        
        # Get latest metrics
        latest_metrics = {}
        for metric_name in ["connection_count", "query_response_time", 
                           "active_transactions"]:
            name_metrics = [
                m for m in db_metrics
                if m.metric_name == metric_name
            ]
            if name_metrics:
                latest_metrics[metric_name] = max(
                    name_metrics, key=lambda x: x.timestamp
                )
        
        status = {
            "database_id": database_id,
            "timestamp": datetime.now(),
            "metrics": latest_metrics,
            "alerts": [
                a for a in self.infrastructure_alerts
                if a.alert_type == "database" and a.source_id == database_id
            ]
        }
        
        return status
    
    @with_error_handling("get_network_status")
    async def get_network_status(self) -> Dict[str, Any]:
        """Get network connectivity status."""
        recent_network_metrics = [
            m for m in self.network_metrics
            if m.timestamp > datetime.now() - timedelta(minutes=5)
        ]
        
        # Group by target
        target_status = {}
        for target in self.network_targets:
            target_metrics = [
                m for m in recent_network_metrics
                if m.target_host == target
            ]
            
            if target_metrics:
                latest_latency = None
                latest_status = None
                
                for metric in target_metrics:
                    if metric.metric_name == "ping_latency":
                        if (latest_latency is None or 
                            metric.timestamp > latest_latency.timestamp):
                            latest_latency = metric
                    elif metric.metric_name == "connection_status":
                        if (latest_status is None or 
                            metric.timestamp > latest_status.timestamp):
                            latest_status = metric
                
                target_status[target] = {
                    "latency": latest_latency.value if latest_latency else None,
                    "status": "up" if latest_status and latest_status.value == 0 else "down",
                    "last_check": max(
                        [m.timestamp for m in target_metrics]
                    ) if target_metrics else None
                }
            else:
                target_status[target] = {
                    "latency": None,
                    "status": "unknown",
                    "last_check": None
                }
        
        status = {
            "timestamp": datetime.now(),
            "targets": target_status,
            "alerts": [
                a for a in self.infrastructure_alerts
                if a.alert_type == "network" and 
                a.timestamp > datetime.now() - timedelta(minutes=5)
            ]
        }
        
        return status


# Global instance
infrastructure_monitor = InfrastructureMonitor()
