"""
Performance Optimizer for the Sentiment Analysis System.
Handles system optimization, caching strategies, and performance monitoring.
"""

import asyncio
import time
import psutil
import gc
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger

from src.core.models import AnalysisRequest, AnalysisResult
from src.core.caching_service import CachingService


@dataclass
class PerformanceMetrics:
    """Performance metrics data class."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    response_time: float
    throughput: float
    cache_hit_rate: float
    error_rate: float
    timestamp: datetime


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation data class."""
    category: str
    priority: str
    description: str
    impact: str
    implementation: str
    estimated_improvement: float


class PerformanceOptimizer:
    """Performance optimizer for the sentiment analysis system."""
    
    def __init__(self):
        self.caching_service = CachingService()
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_history: List[OptimizationRecommendation] = []
        self.max_history_size = 1000
        self.monitoring_interval = 60  # seconds
        self.is_monitoring = False
        
    async def start_monitoring(self):
        """Start performance monitoring."""
        if self.is_monitoring:
            logger.warning("Performance monitoring already running")
            return
        
        self.is_monitoring = True
        logger.info("Starting performance monitoring")
        
        while self.is_monitoring:
            try:
                metrics = await self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep history size manageable
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]
                
                # Check for optimization opportunities
                recommendations = await self.analyze_performance()
                if recommendations:
                    self.optimization_history.extend(recommendations)
                    logger.info(f"Generated {len(recommendations)} optimization recommendations")
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        logger.info("Stopped performance monitoring")
    
    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Application metrics
            cache_stats = await self.caching_service.get_stats()
            cache_hit_rate = cache_stats.get('hit_rate', 0.0)
            
            # Calculate response time and throughput from recent history
            recent_metrics = self.metrics_history[-10:] if self.metrics_history else []
            avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0.0
            throughput = len(recent_metrics) / 60.0 if recent_metrics else 0.0  # requests per minute
            
            # Error rate calculation
            error_rate = 0.0  # This would be calculated from actual error tracking
            
            return PerformanceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                response_time=avg_response_time,
                throughput=throughput,
                cache_hit_rate=cache_hit_rate,
                error_rate=error_rate,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return PerformanceMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                response_time=0.0,
                throughput=0.0,
                cache_hit_rate=0.0,
                error_rate=0.0,
                timestamp=datetime.now()
            )
    
    async def analyze_performance(self) -> List[OptimizationRecommendation]:
        """Analyze performance and generate optimization recommendations."""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        if len(recent_metrics) < 5:
            return recommendations
        
        # Calculate averages
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        
        # CPU optimization recommendations
        if avg_cpu > 80:
            recommendations.append(OptimizationRecommendation(
                category="CPU",
                priority="High",
                description="High CPU usage detected",
                impact="System may become unresponsive",
                implementation="Consider scaling horizontally or optimizing algorithms",
                estimated_improvement=20.0
            ))
        
        # Memory optimization recommendations
        if avg_memory > 85:
            recommendations.append(OptimizationRecommendation(
                category="Memory",
                priority="High",
                description="High memory usage detected",
                impact="Risk of out-of-memory errors",
                implementation="Implement memory cleanup and garbage collection",
                estimated_improvement=15.0
            ))
        
        # Cache optimization recommendations
        if avg_cache_hit_rate < 0.7:
            recommendations.append(OptimizationRecommendation(
                category="Cache",
                priority="Medium",
                description="Low cache hit rate",
                impact="Increased response times",
                implementation="Optimize cache keys and increase cache size",
                estimated_improvement=25.0
            ))
        
        # Response time optimization
        if avg_response_time > 2.0:
            recommendations.append(OptimizationRecommendation(
                category="Response Time",
                priority="High",
                description="High response times detected",
                impact="Poor user experience",
                implementation="Optimize algorithms and implement async processing",
                estimated_improvement=30.0
            ))
        
        return recommendations
    
    async def optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache configuration based on usage patterns."""
        try:
            cache_stats = await self.caching_service.get_stats()
            
            # Analyze cache patterns
            total_requests = cache_stats.get('total_requests', 0)
            cache_hits = cache_stats.get('cache_hits', 0)
            cache_misses = cache_stats.get('cache_misses', 0)
            
            if total_requests == 0:
                return {"success": False, "message": "No cache data available"}
            
            hit_rate = cache_hits / total_requests
            
            optimizations = {
                "current_hit_rate": hit_rate,
                "recommendations": []
            }
            
            # Low hit rate optimization
            if hit_rate < 0.6:
                optimizations["recommendations"].append({
                    "type": "increase_cache_size",
                    "description": "Increase cache size to improve hit rate",
                    "estimated_improvement": "15-25%"
                })
            
            # Cache key optimization
            if cache_misses > cache_hits:
                optimizations["recommendations"].append({
                    "type": "optimize_cache_keys",
                    "description": "Review and optimize cache key generation",
                    "estimated_improvement": "10-20%"
                })
            
            return {"success": True, "optimizations": optimizations}
            
        except Exception as e:
            logger.error(f"Error optimizing cache: {e}")
            return {"success": False, "error": str(e)}
    
    async def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage."""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            # Get memory info
            memory = psutil.virtual_memory()
            
            optimizations = {
                "garbage_collected": collected,
                "memory_usage": memory.percent,
                "available_memory": memory.available / (1024**3),  # GB
                "recommendations": []
            }
            
            # High memory usage recommendations
            if memory.percent > 80:
                optimizations["recommendations"].append({
                    "type": "memory_cleanup",
                    "description": "High memory usage detected",
                    "action": "Consider restarting heavy processes"
                })
            
            return {"success": True, "optimizations": optimizations}
            
        except Exception as e:
            logger.error(f"Error optimizing memory: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            # Try to get data from performance data collector first
            try:
                from src.core.performance_data_collector import get_performance_data_collector
                collector = await get_performance_data_collector()
                summary = await collector.get_performance_summary(hours=24)
                
                if summary.get("success"):
                    # Use comprehensive data from collector
                    report = {
                        "summary": {
                            "total_measurements": summary.get("total_records", 0),
                            "monitoring_duration": "24 hours",
                            "current_status": "Healthy" if summary.get("application_metrics", {}).get("error_rate", 0) < 5 else "Needs Attention"
                        },
                        "current_metrics": {
                            "cpu_usage": summary.get("system_metrics", {}).get("avg_cpu_usage", 0.0),
                            "memory_usage": summary.get("system_metrics", {}).get("avg_memory_usage", 0.0),
                            "response_time": summary.get("application_metrics", {}).get("avg_response_time", 0.0),
                            "cache_hit_rate": 0.0  # Would be available from cache stats
                        },
                        "trends": {
                            "cpu_trend": "Stable",  # Would calculate from historical data
                            "memory_trend": "Stable",
                            "response_time_trend": "Stable"
                        },
                        "recommendations": [],
                        "optimization_history": len(self.optimization_history),
                        "data_source": "performance_data_collector"
                    }
                    
                    # Generate recommendations based on comprehensive data
                    recommendations = await self.analyze_performance()
                    report["recommendations"] = [
                        {
                            "category": r.category,
                            "priority": r.priority,
                            "description": r.description,
                            "impact": r.impact,
                            "implementation": r.implementation,
                            "estimated_improvement": r.estimated_improvement
                        }
                        for r in recommendations
                    ]
                    
                    return {"success": True, "report": report}
                    
            except ImportError:
                logger.debug("Performance data collector not available, using local metrics")
            
            # Fallback to local metrics if collector is not available
            if not self.metrics_history:
                return {"success": False, "message": "No performance data available"}
            
            recent_metrics = self.metrics_history[-24:]  # Last 24 measurements
            
            # Calculate trends
            cpu_trend = self._calculate_trend([m.cpu_usage for m in recent_metrics])
            memory_trend = self._calculate_trend([m.memory_usage for m in recent_metrics])
            response_time_trend = self._calculate_trend([m.response_time for m in recent_metrics])
            
            # Generate recommendations
            recommendations = await self.analyze_performance()
            
            report = {
                "summary": {
                    "total_measurements": len(self.metrics_history),
                    "monitoring_duration": self._calculate_duration(),
                    "current_status": "Healthy" if not recommendations else "Needs Attention"
                },
                "current_metrics": {
                    "cpu_usage": recent_metrics[-1].cpu_usage if recent_metrics else 0.0,
                    "memory_usage": recent_metrics[-1].memory_usage if recent_metrics else 0.0,
                    "response_time": recent_metrics[-1].response_time if recent_metrics else 0.0,
                    "cache_hit_rate": recent_metrics[-1].cache_hit_rate if recent_metrics else 0.0
                },
                "trends": {
                    "cpu_trend": cpu_trend,
                    "memory_trend": memory_trend,
                    "response_time_trend": response_time_trend
                },
                "recommendations": [
                    {
                        "category": r.category,
                        "priority": r.priority,
                        "description": r.description,
                        "impact": r.impact,
                        "implementation": r.implementation,
                        "estimated_improvement": r.estimated_improvement
                    }
                    for r in recommendations
                ],
                "optimization_history": len(self.optimization_history),
                "data_source": "local_metrics"
            }
            
            return {"success": True, "report": report}
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values."""
        if len(values) < 2:
            return "Stable"
        
        # Simple linear trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        if not first_half or not second_half:
            return "Stable"
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg * 1.1:
            return "Increasing"
        elif second_avg < first_avg * 0.9:
            return "Decreasing"
        else:
            return "Stable"
    
    def _calculate_duration(self) -> str:
        """Calculate monitoring duration."""
        if not self.metrics_history:
            return "0 minutes"
        
        duration = self.metrics_history[-1].timestamp - self.metrics_history[0].timestamp
        hours = duration.total_seconds() / 3600
        
        if hours < 1:
            return f"{int(duration.total_seconds() / 60)} minutes"
        elif hours < 24:
            return f"{hours:.1f} hours"
        else:
            days = hours / 24
            return f"{days:.1f} days"
    
    async def apply_optimizations(self, optimization_type: str) -> Dict[str, Any]:
        """Apply specific optimizations."""
        try:
            if optimization_type == "cache":
                return await self.optimize_cache()
            elif optimization_type == "memory":
                return await self.optimize_memory()
            elif optimization_type == "all":
                cache_result = await self.optimize_cache()
                memory_result = await self.optimize_memory()
                return {
                    "success": True,
                    "cache_optimization": cache_result,
                    "memory_optimization": memory_result
                }
            else:
                return {"success": False, "error": f"Unknown optimization type: {optimization_type}"}
                
        except Exception as e:
            logger.error(f"Error applying optimizations: {e}")
            return {"success": False, "error": str(e)}


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()


async def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance."""
    return performance_optimizer
