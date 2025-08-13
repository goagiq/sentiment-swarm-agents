"""
System Health Monitor

Monitors overall system health, resource usage, and component status.
"""

import psutil
import time
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a specific component"""
    name: str
    status: HealthStatus
    last_check: datetime
    response_time: float
    error_count: int
    details: Dict[str, Any]


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, float]
    active_connections: int
    timestamp: datetime


class SystemHealthMonitor:
    """Monitors overall system health and provides real-time status updates."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.components: Dict[str, ComponentHealth] = {}
        self.metrics_history: List[SystemMetrics] = []
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'cpu_warning': 70.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0
        })
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.check_interval = self.config.get('check_interval', 30)
        
    def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.monitoring_active:
            logger.warning("Health monitoring is already active")
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
        
    def stop_monitoring(self):
        """Stop continuous health monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self.check_system_health()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
                
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        try:
            metrics = self._get_system_metrics()
            self.metrics_history.append(metrics)
            
            # Keep only recent history (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.metrics_history = [
                m for m in self.metrics_history 
                if m.timestamp > cutoff_time
            ]
            
            self._check_components()
            health_report = self._generate_health_report(metrics)
            alerts = self._check_alerts(metrics)
            
            if alerts:
                self._send_alerts(alerts)
                
            return health_report
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {"status": HealthStatus.UNKNOWN, "error": str(e)}
            
    def _get_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            network_io = psutil.net_io_counters()
            network_data = {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv,
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv
            }
            active_connections = len(psutil.net_connections())
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_io=network_data,
                active_connections=active_connections,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            raise
            
    def _check_components(self):
        """Check health of registered components"""
        for component_name in self.components:
            try:
                self._check_component_health(component_name)
            except Exception as e:
                logger.error(f"Error checking component {component_name}: {e}")
                
    def _check_component_health(self, component_name: str):
        """Check health of a specific component"""
        start_time = time.time()
        
        try:
            component = self.components[component_name]
            response_time = time.time() - start_time
            
            self.components[component_name] = ComponentHealth(
                name=component_name,
                status=HealthStatus.HEALTHY,
                last_check=datetime.now(),
                response_time=response_time,
                error_count=0,
                details={"response_time": response_time}
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            error_count = self.components[component_name].error_count + 1
            
            self.components[component_name] = ComponentHealth(
                name=component_name,
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time=response_time,
                error_count=error_count,
                details={"error": str(e), "error_count": error_count}
            )
            
    def register_component(self, component_name: str):
        """Register a component for health monitoring"""
        self.components[component_name] = ComponentHealth(
            name=component_name,
            status=HealthStatus.UNKNOWN,
            last_check=datetime.now(),
            response_time=0.0,
            error_count=0,
            details={}
        )
        logger.info(f"Registered component: {component_name}")
        
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary"""
        if not self.metrics_history:
            return {"status": "unknown", "message": "No metrics available"}
            
        latest_metrics = self.metrics_history[-1]
        return self._generate_health_report(latest_metrics)
        
    def _generate_health_report(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        overall_status = self._determine_overall_status(metrics)
        
        component_summary = {
            'total_components': len(self.components),
            'healthy_components': len([c for c in self.components.values() 
                                     if c.status == HealthStatus.HEALTHY]),
            'warning_components': len([c for c in self.components.values() 
                                     if c.status == HealthStatus.WARNING]),
            'critical_components': len([c for c in self.components.values() 
                                      if c.status == HealthStatus.CRITICAL])
        }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status.value,
            'system_metrics': asdict(metrics),
            'component_summary': component_summary,
            'components': {name: asdict(comp) for name, comp in self.components.items()},
            'alerts': self._get_active_alerts(metrics)
        }
        
    def _determine_overall_status(self, metrics: SystemMetrics) -> HealthStatus:
        """Determine overall system health status"""
        if (metrics.cpu_percent >= self.alert_thresholds.get('cpu_critical', 90.0) or
            metrics.memory_percent >= self.alert_thresholds.get('memory_critical', 95.0)):
            return HealthStatus.CRITICAL
            
        if (metrics.cpu_percent >= self.alert_thresholds.get('cpu_warning', 70.0) or
            metrics.memory_percent >= self.alert_thresholds.get('memory_warning', 80.0)):
            return HealthStatus.WARNING
            
        critical_components = [c for c in self.components.values() 
                             if c.status == HealthStatus.CRITICAL]
        if critical_components:
            return HealthStatus.CRITICAL
            
        warning_components = [c for c in self.components.values() 
                            if c.status == HealthStatus.WARNING]
        if warning_components:
            return HealthStatus.WARNING
            
        return HealthStatus.HEALTHY
        
    def _check_alerts(self, metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """Check for alert conditions"""
        alerts = []
        
        if metrics.cpu_percent >= self.alert_thresholds['cpu_critical']:
            alerts.append({
                'type': 'critical',
                'component': 'cpu',
                'message': f'CPU usage critical: {metrics.cpu_percent:.1f}%',
                'value': metrics.cpu_percent,
                'threshold': self.alert_thresholds['cpu_critical']
            })
        elif metrics.cpu_percent >= self.alert_thresholds['cpu_warning']:
            alerts.append({
                'type': 'warning',
                'component': 'cpu',
                'message': f'CPU usage high: {metrics.cpu_percent:.1f}%',
                'value': metrics.cpu_percent,
                'threshold': self.alert_thresholds['cpu_warning']
            })
            
        if metrics.memory_percent >= self.alert_thresholds['memory_critical']:
            alerts.append({
                'type': 'critical',
                'component': 'memory',
                'message': f'Memory usage critical: {metrics.memory_percent:.1f}%',
                'value': metrics.memory_percent,
                'threshold': self.alert_thresholds['memory_critical']
            })
        elif metrics.memory_percent >= self.alert_thresholds['memory_warning']:
            alerts.append({
                'type': 'warning',
                'component': 'memory',
                'message': f'Memory usage high: {metrics.memory_percent:.1f}%',
                'value': metrics.memory_percent,
                'threshold': self.alert_thresholds['memory_warning']
            })
            
        return alerts
        
    def _get_active_alerts(self, metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """Get currently active alerts"""
        return self._check_alerts(metrics)
        
    def _send_alerts(self, alerts: List[Dict[str, Any]]):
        """Send alerts"""
        for alert in alerts:
            logger.warning(f"ALERT: {alert['message']}")
