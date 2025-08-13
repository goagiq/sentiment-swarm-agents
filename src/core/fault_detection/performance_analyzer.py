"""
Performance Analyzer

Analyzes system performance metrics and identifies bottlenecks.
Provides detailed performance insights and optimization recommendations.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import statistics
import psutil

logger = logging.getLogger(__name__)


class PerformanceLevel(Enum):
    """Performance level classifications"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    level: PerformanceLevel
    threshold: float


@dataclass
class PerformanceSnapshot:
    """Complete performance snapshot"""
    timestamp: datetime
    cpu_metrics: Dict[str, PerformanceMetric]
    memory_metrics: Dict[str, PerformanceMetric]
    disk_metrics: Dict[str, PerformanceMetric]
    network_metrics: Dict[str, PerformanceMetric]
    process_metrics: Dict[str, PerformanceMetric]
    overall_score: float


class PerformanceAnalyzer:
    """Analyzes system performance and identifies optimization opportunities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.performance_history: List[PerformanceSnapshot] = []
        self.analysis_active = False
        self.analysis_thread: Optional[threading.Thread] = None
        self.analysis_interval = self.config.get('analysis_interval', 60)
        
        # Performance thresholds
        self.thresholds = self.config.get('thresholds', {
            'cpu_excellent': 20.0,
            'cpu_good': 40.0,
            'cpu_fair': 60.0,
            'cpu_poor': 80.0,
            'memory_excellent': 30.0,
            'memory_good': 50.0,
            'memory_fair': 70.0,
            'memory_poor': 85.0,
            'disk_excellent': 50.0,
            'disk_good': 70.0,
            'disk_fair': 85.0,
            'disk_poor': 95.0
        })
        
    def start_analysis(self):
        """Start continuous performance analysis"""
        if self.analysis_active:
            logger.warning("Performance analysis is already active")
            return
            
        self.analysis_active = True
        self.analysis_thread = threading.Thread(
            target=self._analysis_loop, daemon=True
        )
        self.analysis_thread.start()
        logger.info("Performance analysis started")
        
    def stop_analysis(self):
        """Stop continuous performance analysis"""
        self.analysis_active = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5)
        logger.info("Performance analysis stopped")
        
    def _analysis_loop(self):
        """Main analysis loop"""
        while self.analysis_active:
            try:
                snapshot = self.analyze_performance()
                self.performance_history.append(snapshot)
                
                # Keep only recent history (last 7 days)
                cutoff_time = datetime.now() - timedelta(days=7)
                self.performance_history = [
                    s for s in self.performance_history 
                    if s.timestamp > cutoff_time
                ]
                
                time.sleep(self.analysis_interval)
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                time.sleep(10)
                
    def analyze_performance(self) -> PerformanceSnapshot:
        """Perform comprehensive performance analysis"""
        try:
            timestamp = datetime.now()
            
            # Collect CPU metrics
            cpu_metrics = self._analyze_cpu_performance()
            
            # Collect memory metrics
            memory_metrics = self._analyze_memory_performance()
            
            # Collect disk metrics
            disk_metrics = self._analyze_disk_performance()
            
            # Collect network metrics
            network_metrics = self._analyze_network_performance()
            
            # Collect process metrics
            process_metrics = self._analyze_process_performance()
            
            # Calculate overall performance score
            overall_score = self._calculate_overall_score(
                cpu_metrics, memory_metrics, disk_metrics, 
                network_metrics, process_metrics
            )
            
            return PerformanceSnapshot(
                timestamp=timestamp,
                cpu_metrics=cpu_metrics,
                memory_metrics=memory_metrics,
                disk_metrics=disk_metrics,
                network_metrics=network_metrics,
                process_metrics=process_metrics,
                overall_score=overall_score
            )
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
            raise
            
    def _analyze_cpu_performance(self) -> Dict[str, PerformanceMetric]:
        """Analyze CPU performance metrics"""
        metrics = {}
        
        try:
            # CPU usage percentage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_level = self._classify_performance(
                cpu_percent, 'cpu', reverse=True
            )
            metrics['cpu_usage'] = PerformanceMetric(
                name='CPU Usage',
                value=cpu_percent,
                unit='%',
                timestamp=datetime.now(),
                level=cpu_level,
                threshold=self.thresholds.get('cpu_poor', 80.0)
            )
            
            # CPU frequency
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                freq_percent = (cpu_freq.current / cpu_freq.max) * 100
                freq_level = self._classify_performance(freq_percent, 'cpu')
                metrics['cpu_frequency'] = PerformanceMetric(
                    name='CPU Frequency',
                    value=freq_percent,
                    unit='%',
                    timestamp=datetime.now(),
                    level=freq_level,
                    threshold=80.0
                )
                
            # CPU load average (if available)
            try:
                load_avg = psutil.getloadavg()
                if load_avg:
                    avg_load = statistics.mean(load_avg)
                    load_level = self._classify_performance(
                        avg_load, 'cpu', reverse=True
                    )
                    metrics['cpu_load'] = PerformanceMetric(
                        name='CPU Load Average',
                        value=avg_load,
                        unit='',
                        timestamp=datetime.now(),
                        level=load_level,
                        threshold=2.0
                    )
            except AttributeError:
                pass  # load average not available on Windows
                
        except Exception as e:
            logger.error(f"Error analyzing CPU performance: {e}")
            
        return metrics
        
    def _analyze_memory_performance(self) -> Dict[str, PerformanceMetric]:
        """Analyze memory performance metrics"""
        metrics = {}
        
        try:
            memory = psutil.virtual_memory()
            
            # Memory usage percentage
            memory_percent = memory.percent
            memory_level = self._classify_performance(
                memory_percent, 'memory', reverse=True
            )
            metrics['memory_usage'] = PerformanceMetric(
                name='Memory Usage',
                value=memory_percent,
                unit='%',
                timestamp=datetime.now(),
                level=memory_level,
                threshold=self.thresholds.get('memory_poor', 85.0)
            )
            
            # Available memory
            available_gb = memory.available / (1024**3)
            available_percent = (memory.available / memory.total) * 100
            available_level = self._classify_performance(available_percent, 'memory')
            metrics['memory_available'] = PerformanceMetric(
                name='Available Memory',
                value=available_gb,
                unit='GB',
                timestamp=datetime.now(),
                level=available_level,
                threshold=2.0  # 2GB minimum
            )
            
            # Memory swap usage
            swap = psutil.swap_memory()
            swap_percent = swap.percent
            swap_level = self._classify_performance(
                swap_percent, 'memory', reverse=True
            )
            metrics['swap_usage'] = PerformanceMetric(
                name='Swap Usage',
                value=swap_percent,
                unit='%',
                timestamp=datetime.now(),
                level=swap_level,
                threshold=50.0
            )
            
        except Exception as e:
            logger.error(f"Error analyzing memory performance: {e}")
            
        return metrics
        
    def _analyze_disk_performance(self) -> Dict[str, PerformanceMetric]:
        """Analyze disk performance metrics"""
        metrics = {}
        
        try:
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_level = self._classify_performance(
                disk_percent, 'disk', reverse=True
            )
            metrics['disk_usage'] = PerformanceMetric(
                name='Disk Usage',
                value=disk_percent,
                unit='%',
                timestamp=datetime.now(),
                level=disk_level,
                threshold=self.thresholds.get('disk_poor', 95.0)
            )
            
            # Disk I/O (if available)
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    # Calculate I/O rate (simplified)
                    io_rate = (disk_io.read_bytes + disk_io.write_bytes) / (1024**2)
                    io_level = self._classify_performance(io_rate, 'disk')
                    metrics['disk_io'] = PerformanceMetric(
                        name='Disk I/O',
                        value=io_rate,
                        unit='MB/s',
                        timestamp=datetime.now(),
                        level=io_level,
                        threshold=100.0
                    )
            except Exception:
                pass  # Disk I/O not available
                
        except Exception as e:
            logger.error(f"Error analyzing disk performance: {e}")
            
        return metrics
        
    def _analyze_network_performance(self) -> Dict[str, PerformanceMetric]:
        """Analyze network performance metrics"""
        metrics = {}
        
        try:
            network_io = psutil.net_io_counters()
            
            # Network bytes sent/received
            total_bytes = network_io.bytes_sent + network_io.bytes_recv
            total_mb = total_bytes / (1024**2)
            
            # Simple network activity metric
            network_level = self._classify_performance(total_mb, 'network')
            metrics['network_activity'] = PerformanceMetric(
                name='Network Activity',
                value=total_mb,
                unit='MB',
                timestamp=datetime.now(),
                level=network_level,
                threshold=1000.0
            )
            
            # Network connections
            connections = len(psutil.net_connections())
            conn_level = self._classify_performance(connections, 'network')
            metrics['network_connections'] = PerformanceMetric(
                name='Network Connections',
                value=connections,
                unit='',
                timestamp=datetime.now(),
                level=conn_level,
                threshold=1000
            )
            
        except Exception as e:
            logger.error(f"Error analyzing network performance: {e}")
            
        return metrics
        
    def _analyze_process_performance(self) -> Dict[str, PerformanceMetric]:
        """Analyze process performance metrics"""
        metrics = {}
        
        try:
            # Number of processes
            process_count = len(psutil.pids())
            proc_level = self._classify_performance(process_count, 'process')
            metrics['process_count'] = PerformanceMetric(
                name='Process Count',
                value=process_count,
                unit='',
                timestamp=datetime.now(),
                level=proc_level,
                threshold=500
            )
            
            # Top processes by CPU usage
            try:
                top_processes = psutil.process_iter(
                    ['pid', 'name', 'cpu_percent', 'memory_percent']
                )
                cpu_processes = sorted(
                    top_processes, 
                    key=lambda x: x.info['cpu_percent'], 
                    reverse=True
                )[:5]
                
                if cpu_processes:
                    avg_cpu = statistics.mean(
                        [p.info['cpu_percent'] for p in cpu_processes]
                    )
                    cpu_proc_level = self._classify_performance(
                        avg_cpu, 'process', reverse=True
                    )
                    metrics['top_processes_cpu'] = PerformanceMetric(
                        name='Top Processes CPU',
                        value=avg_cpu,
                        unit='%',
                        timestamp=datetime.now(),
                        level=cpu_proc_level,
                        threshold=50.0
                    )
            except Exception:
                pass
                
        except Exception as e:
            logger.error(f"Error analyzing process performance: {e}")
            
        return metrics
        
    def _classify_performance(
        self, value: float, metric_type: str, reverse: bool = False
    ) -> PerformanceLevel:
        """Classify performance level based on value and thresholds"""
        thresholds = self.thresholds
        
        if metric_type == 'cpu':
            if reverse:
                if value <= thresholds.get('cpu_excellent', 20.0):
                    return PerformanceLevel.EXCELLENT
                elif value <= thresholds.get('cpu_good', 40.0):
                    return PerformanceLevel.GOOD
                elif value <= thresholds.get('cpu_fair', 60.0):
                    return PerformanceLevel.FAIR
                elif value <= thresholds.get('cpu_poor', 80.0):
                    return PerformanceLevel.POOR
                else:
                    return PerformanceLevel.CRITICAL
            else:
                if value >= 80.0:
                    return PerformanceLevel.EXCELLENT
                elif value >= 60.0:
                    return PerformanceLevel.GOOD
                elif value >= 40.0:
                    return PerformanceLevel.FAIR
                elif value >= 20.0:
                    return PerformanceLevel.POOR
                else:
                    return PerformanceLevel.CRITICAL
                    
        elif metric_type == 'memory':
            if reverse:
                if value <= thresholds.get('memory_excellent', 30.0):
                    return PerformanceLevel.EXCELLENT
                elif value <= thresholds.get('memory_good', 50.0):
                    return PerformanceLevel.GOOD
                elif value <= thresholds.get('memory_fair', 70.0):
                    return PerformanceLevel.FAIR
                elif value <= thresholds.get('memory_poor', 85.0):
                    return PerformanceLevel.POOR
                else:
                    return PerformanceLevel.CRITICAL
            else:
                if value >= 70.0:
                    return PerformanceLevel.EXCELLENT
                elif value >= 50.0:
                    return PerformanceLevel.GOOD
                elif value >= 30.0:
                    return PerformanceLevel.FAIR
                elif value >= 15.0:
                    return PerformanceLevel.POOR
                else:
                    return PerformanceLevel.CRITICAL
                    
        else:
            # Default classification
            if value >= 80.0:
                return PerformanceLevel.EXCELLENT
            elif value >= 60.0:
                return PerformanceLevel.GOOD
            elif value >= 40.0:
                return PerformanceLevel.FAIR
            elif value >= 20.0:
                return PerformanceLevel.POOR
            else:
                return PerformanceLevel.CRITICAL
                
    def _calculate_overall_score(
        self, cpu_metrics: Dict[str, PerformanceMetric],
        memory_metrics: Dict[str, PerformanceMetric],
        disk_metrics: Dict[str, PerformanceMetric],
        network_metrics: Dict[str, PerformanceMetric],
        process_metrics: Dict[str, PerformanceMetric]
    ) -> float:
        """Calculate overall performance score (0-100)"""
        try:
            all_metrics = []
            all_metrics.extend(cpu_metrics.values())
            all_metrics.extend(memory_metrics.values())
            all_metrics.extend(disk_metrics.values())
            all_metrics.extend(network_metrics.values())
            all_metrics.extend(process_metrics.values())
            
            if not all_metrics:
                return 0.0
                
            # Convert performance levels to scores
            level_scores = {
                PerformanceLevel.EXCELLENT: 100.0,
                PerformanceLevel.GOOD: 80.0,
                PerformanceLevel.FAIR: 60.0,
                PerformanceLevel.POOR: 40.0,
                PerformanceLevel.CRITICAL: 20.0
            }
            
            scores = [level_scores[metric.level] for metric in all_metrics]
            return statistics.mean(scores)
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 0.0
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        if not self.performance_history:
            return {"status": "unknown", "message": "No performance data available"}
            
        latest_snapshot = self.performance_history[-1]
        
        return {
            'timestamp': latest_snapshot.timestamp.isoformat(),
            'overall_score': latest_snapshot.overall_score,
            'cpu_metrics': {name: asdict(metric) for name, metric in latest_snapshot.cpu_metrics.items()},
            'memory_metrics': {name: asdict(metric) for name, metric in latest_snapshot.memory_metrics.items()},
            'disk_metrics': {name: asdict(metric) for name, metric in latest_snapshot.disk_metrics.items()},
            'network_metrics': {name: asdict(metric) for name, metric in latest_snapshot.network_metrics.items()},
            'process_metrics': {name: asdict(metric) for name, metric in latest_snapshot.process_metrics.items()}
        }
        
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        if not self.performance_history:
            return bottlenecks
            
        latest_snapshot = self.performance_history[-1]
        
        # Check CPU bottlenecks
        for name, metric in latest_snapshot.cpu_metrics.items():
            if metric.level in [PerformanceLevel.POOR, PerformanceLevel.CRITICAL]:
                bottlenecks.append({
                    'component': 'CPU',
                    'metric': name,
                    'value': metric.value,
                    'unit': metric.unit,
                    'level': metric.level.value,
                    'recommendation': self._get_bottleneck_recommendation('CPU', name)
                })
                
        # Check memory bottlenecks
        for name, metric in latest_snapshot.memory_metrics.items():
            if metric.level in [PerformanceLevel.POOR, PerformanceLevel.CRITICAL]:
                bottlenecks.append({
                    'component': 'Memory',
                    'metric': name,
                    'value': metric.value,
                    'unit': metric.unit,
                    'level': metric.level.value,
                    'recommendation': self._get_bottleneck_recommendation('Memory', name)
                })
                
        # Check disk bottlenecks
        for name, metric in latest_snapshot.disk_metrics.items():
            if metric.level in [PerformanceLevel.POOR, PerformanceLevel.CRITICAL]:
                bottlenecks.append({
                    'component': 'Disk',
                    'metric': name,
                    'value': metric.value,
                    'unit': metric.unit,
                    'level': metric.level.value,
                    'recommendation': self._get_bottleneck_recommendation('Disk', name)
                })
                
        return bottlenecks
        
    def _get_bottleneck_recommendation(self, component: str, metric: str) -> str:
        """Get recommendation for addressing bottleneck"""
        recommendations = {
            'CPU': {
                'CPU Usage': 'Consider optimizing CPU-intensive processes or upgrading CPU',
                'CPU Frequency': 'Check power management settings and thermal throttling',
                'CPU Load Average': 'Reduce system load or add more CPU cores'
            },
            'Memory': {
                'Memory Usage': 'Close unnecessary applications or add more RAM',
                'Available Memory': 'Free up memory by closing applications',
                'Swap Usage': 'Reduce memory pressure or increase swap space'
            },
            'Disk': {
                'Disk Usage': 'Free up disk space or expand storage',
                'Disk I/O': 'Optimize I/O operations or upgrade to SSD'
            }
        }
        
        return recommendations.get(component, {}).get(metric, 'Monitor and investigate further')
        
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_snapshots = [
            s for s in self.performance_history 
            if s.timestamp > cutoff_time
        ]
        
        if not recent_snapshots:
            return {"message": "No performance data available for specified period"}
            
        # Calculate trends
        scores = [s.overall_score for s in recent_snapshots]
        timestamps = [s.timestamp.isoformat() for s in recent_snapshots]
        
        return {
            'period_hours': hours,
            'data_points': len(recent_snapshots),
            'average_score': statistics.mean(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'trend': 'improving' if scores[-1] > scores[0] else 'declining' if scores[-1] < scores[0] else 'stable',
            'scores': scores,
            'timestamps': timestamps
        }
