#!/usr/bin/env python3
"""
Performance Monitoring Dashboard
Real-time monitoring and alerting for the multilingual sentiment analysis system
"""

import asyncio
import json
import time
import psutil
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.performance_monitor import get_global_performance_monitor
from core.memory_manager import get_global_memory_manager
from core.advanced_caching_service import get_global_cache
from core.parallel_processor import get_global_parallel_processor
from config.dynamic_config_manager import dynamic_config_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_processes: int

@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    timestamp: str
    processing_time: float
    throughput: float
    error_rate: float
    cache_hit_rate: float
    memory_usage: float
    queue_length: int
    active_workers: int

@dataclass
class Alert:
    """Alert information"""
    timestamp: str
    severity: str
    category: str
    message: str
    details: Dict[str, Any]

class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""
    
    def __init__(self):
        self.monitor = get_global_performance_monitor()
        self.memory_manager = get_global_memory_manager()
        self.cache = get_global_cache()
        self.processor = get_global_parallel_processor()
        
        # Alert thresholds
        self.alert_thresholds = {
            'memory_usage': 85.0,  # 85%
            'cpu_usage': 90.0,     # 90%
            'error_rate': 5.0,     # 5%
            'cache_hit_rate': 70.0, # 70%
            'processing_time': 5000, # 5 seconds
            'queue_length': 100     # 100 items
        }
        
        # Alert history
        self.alerts: List[Alert] = []
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Redis connection for distributed monitoring
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_available = True
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_available = False
    
    async def get_system_metrics(self) -> SystemMetrics:
        """Get system-level metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Active processes
            active_processes = len(psutil.pids())
            
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                active_processes=active_processes
            )
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_io={},
                active_processes=0
            )
    
    async def get_application_metrics(self) -> ApplicationMetrics:
        """Get application-specific metrics"""
        try:
            # Performance monitor metrics
            perf_metrics = self.monitor.get_metrics()
            
            # Memory manager metrics
            memory_status = self.memory_manager.get_status()
            
            # Cache metrics
            cache_status = self.cache.get_status()
            
            # Processor metrics
            processor_status = self.processor.get_status()
            
            return ApplicationMetrics(
                timestamp=datetime.now().isoformat(),
                processing_time=perf_metrics.get('avg_processing_time', 0.0),
                throughput=perf_metrics.get('throughput', 0.0),
                error_rate=perf_metrics.get('error_rate', 0.0),
                cache_hit_rate=cache_status.get('hit_rate', 0.0) * 100,
                memory_usage=memory_status.get('memory_usage_ratio', 0.0) * 100,
                queue_length=processor_status.get('queue_length', 0),
                active_workers=processor_status.get('active_workers', 0)
            )
        except Exception as e:
            logger.error(f"Error getting application metrics: {e}")
            return ApplicationMetrics(
                timestamp=datetime.now().isoformat(),
                processing_time=0.0,
                throughput=0.0,
                error_rate=0.0,
                cache_hit_rate=0.0,
                memory_usage=0.0,
                queue_length=0,
                active_workers=0
            )
    
    async def check_alerts(self, system_metrics: SystemMetrics, app_metrics: ApplicationMetrics) -> List[Alert]:
        """Check for alert conditions"""
        alerts = []
        timestamp = datetime.now().isoformat()
        
        # Memory usage alert
        if system_metrics.memory_usage > self.alert_thresholds['memory_usage']:
            alerts.append(Alert(
                timestamp=timestamp,
                severity='WARNING',
                category='MEMORY',
                message=f"High memory usage: {system_metrics.memory_usage:.1f}%",
                details={'usage': system_metrics.memory_usage, 'threshold': self.alert_thresholds['memory_usage']}
            ))
        
        # CPU usage alert
        if system_metrics.cpu_usage > self.alert_thresholds['cpu_usage']:
            alerts.append(Alert(
                timestamp=timestamp,
                severity='WARNING',
                category='CPU',
                message=f"High CPU usage: {system_metrics.cpu_usage:.1f}%",
                details={'usage': system_metrics.cpu_usage, 'threshold': self.alert_thresholds['cpu_usage']}
            ))
        
        # Error rate alert
        if app_metrics.error_rate > self.alert_thresholds['error_rate']:
            alerts.append(Alert(
                timestamp=timestamp,
                severity='CRITICAL',
                category='ERROR_RATE',
                message=f"High error rate: {app_metrics.error_rate:.1f}%",
                details={'rate': app_metrics.error_rate, 'threshold': self.alert_thresholds['error_rate']}
            ))
        
        # Cache hit rate alert
        if app_metrics.cache_hit_rate < self.alert_thresholds['cache_hit_rate']:
            alerts.append(Alert(
                timestamp=timestamp,
                severity='WARNING',
                category='CACHE',
                message=f"Low cache hit rate: {app_metrics.cache_hit_rate:.1f}%",
                details={'hit_rate': app_metrics.cache_hit_rate, 'threshold': self.alert_thresholds['cache_hit_rate']}
            ))
        
        # Processing time alert
        if app_metrics.processing_time > self.alert_thresholds['processing_time']:
            alerts.append(Alert(
                timestamp=timestamp,
                severity='WARNING',
                category='PERFORMANCE',
                message=f"High processing time: {app_metrics.processing_time:.1f}ms",
                details={'time': app_metrics.processing_time, 'threshold': self.alert_thresholds['processing_time']}
            ))
        
        # Queue length alert
        if app_metrics.queue_length > self.alert_thresholds['queue_length']:
            alerts.append(Alert(
                timestamp=timestamp,
                severity='WARNING',
                category='QUEUE',
                message=f"High queue length: {app_metrics.queue_length}",
                details={'length': app_metrics.queue_length, 'threshold': self.alert_thresholds['queue_length']}
            ))
        
        return alerts
    
    async def store_metrics(self, system_metrics: SystemMetrics, app_metrics: ApplicationMetrics):
        """Store metrics for historical analysis"""
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'system': asdict(system_metrics),
            'application': asdict(app_metrics)
        }
        
        # Store in memory (keep last 1000 entries)
        self.metrics_history.append(metrics_data)
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)
        
        # Store in Redis if available
        if self.redis_available:
            try:
                self.redis_client.lpush('performance_metrics', json.dumps(metrics_data))
                self.redis_client.ltrim('performance_metrics', 0, 999)  # Keep last 1000
            except Exception as e:
                logger.warning(f"Failed to store metrics in Redis: {e}")
    
    async def generate_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        try:
            # Get current metrics
            system_metrics = await self.get_system_metrics()
            app_metrics = await self.get_application_metrics()
            
            # Calculate trends (last 10 minutes)
            recent_metrics = [m for m in self.metrics_history 
                            if datetime.fromisoformat(m['timestamp']) > datetime.now() - timedelta(minutes=10)]
            
            trends = {}
            if recent_metrics:
                # Calculate averages
                avg_processing_time = sum(m['application']['processing_time'] for m in recent_metrics) / len(recent_metrics)
                avg_throughput = sum(m['application']['throughput'] for m in recent_metrics) / len(recent_metrics)
                avg_error_rate = sum(m['application']['error_rate'] for m in recent_metrics) / len(recent_metrics)
                
                trends = {
                    'avg_processing_time': avg_processing_time,
                    'avg_throughput': avg_throughput,
                    'avg_error_rate': avg_error_rate,
                    'data_points': len(recent_metrics)
                }
            
            # Generate recommendations
            recommendations = []
            
            if app_metrics.cache_hit_rate < 80:
                recommendations.append("Consider increasing cache size or optimizing cache keys")
            
            if app_metrics.error_rate > 2:
                recommendations.append("Investigate error patterns and implement error handling improvements")
            
            if app_metrics.queue_length > 50:
                recommendations.append("Consider increasing worker count or optimizing processing")
            
            if system_metrics.memory_usage > 80:
                recommendations.append("Monitor memory usage and consider memory optimization")
            
            return {
                'timestamp': datetime.now().isoformat(),
                'current_metrics': {
                    'system': asdict(system_metrics),
                    'application': asdict(app_metrics)
                },
                'trends': trends,
                'alerts': [asdict(alert) for alert in self.alerts[-10:]],  # Last 10 alerts
                'recommendations': recommendations,
                'summary': {
                    'status': 'HEALTHY' if not self.alerts else 'WARNING',
                    'active_alerts': len([a for a in self.alerts if a.severity in ['WARNING', 'CRITICAL']]),
                    'uptime': self._calculate_uptime()
                }
            }
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {'error': str(e)}
    
    def _calculate_uptime(self) -> str:
        """Calculate system uptime"""
        try:
            uptime_seconds = time.time() - psutil.boot_time()
            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
        except:
            return "Unknown"
    
    async def display_dashboard(self):
        """Display real-time dashboard"""
        print("\n" + "="*80)
        print("🚀 MULTILINGUAL SENTIMENT ANALYSIS - PERFORMANCE DASHBOARD")
        print("="*80)
        
        while True:
            try:
                # Clear screen (works on most terminals)
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Get metrics
                system_metrics = await self.get_system_metrics()
                app_metrics = await self.get_application_metrics()
                
                # Check for alerts
                new_alerts = await self.check_alerts(system_metrics, app_metrics)
                self.alerts.extend(new_alerts)
                
                # Store metrics
                await self.store_metrics(system_metrics, app_metrics)
                
                # Display current time
                print(f"\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("="*80)
                
                # System Metrics
                print("\n🖥️  SYSTEM METRICS:")
                print(f"   CPU Usage:     {system_metrics.cpu_usage:6.1f}%")
                print(f"   Memory Usage:  {system_metrics.memory_usage:6.1f}%")
                print(f"   Disk Usage:    {system_metrics.disk_usage:6.1f}%")
                print(f"   Active Procs:  {system_metrics.active_processes:6d}")
                
                # Application Metrics
                print("\n⚡ APPLICATION METRICS:")
                print(f"   Processing Time: {app_metrics.processing_time:8.1f}ms")
                print(f"   Throughput:      {app_metrics.throughput:8.1f} ops/sec")
                print(f"   Error Rate:      {app_metrics.error_rate:8.1f}%")
                print(f"   Cache Hit Rate:  {app_metrics.cache_hit_rate:8.1f}%")
                print(f"   Memory Usage:    {app_metrics.memory_usage:8.1f}%")
                print(f"   Queue Length:    {app_metrics.queue_length:8d}")
                print(f"   Active Workers:  {app_metrics.active_workers:8d}")
                
                # Alerts
                if new_alerts:
                    print("\n🚨 ACTIVE ALERTS:")
                    for alert in new_alerts:
                        severity_icon = "🔴" if alert.severity == "CRITICAL" else "🟡"
                        print(f"   {severity_icon} {alert.severity}: {alert.message}")
                
                # Status Summary
                status = "🟢 HEALTHY" if not new_alerts else "🟡 WARNING" if not any(a.severity == "CRITICAL" for a in new_alerts) else "🔴 CRITICAL"
                print(f"\n📊 STATUS: {status}")
                
                # Auto-refresh every 5 seconds
                print(f"\n🔄 Auto-refreshing in 5 seconds... (Press Ctrl+C to exit)")
                await asyncio.sleep(5)
                
            except KeyboardInterrupt:
                print("\n\n👋 Dashboard stopped by user")
                break
            except Exception as e:
                logger.error(f"Dashboard error: {e}")
                await asyncio.sleep(5)
    
    async def export_metrics(self, filename: str):
        """Export metrics to JSON file"""
        try:
            report = await self.generate_report()
            
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Metrics exported to {filename}")
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")

async def main():
    """Main function"""
    dashboard = PerformanceDashboard()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "report":
            # Generate and display report
            report = await dashboard.generate_report()
            print(json.dumps(report, indent=2))
        
        elif command == "export":
            # Export metrics
            filename = sys.argv[2] if len(sys.argv) > 2 else f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            await dashboard.export_metrics(filename)
        
        elif command == "alerts":
            # Show recent alerts
            for alert in dashboard.alerts[-10:]:
                print(f"[{alert.timestamp}] {alert.severity}: {alert.message}")
        
        else:
            print("Usage:")
            print("  python performance_monitoring_dashboard.py          # Start dashboard")
            print("  python performance_monitoring_dashboard.py report   # Generate report")
            print("  python performance_monitoring_dashboard.py export   # Export metrics")
            print("  python performance_monitoring_dashboard.py alerts   # Show alerts")
    else:
        # Start interactive dashboard
        await dashboard.display_dashboard()

if __name__ == "__main__":
    asyncio.run(main())
