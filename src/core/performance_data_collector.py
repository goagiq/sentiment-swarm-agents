"""
Performance Data Collector for the Sentiment Analysis System.
Integrates PerformanceOptimizer with PerformanceMonitor to provide comprehensive 
performance data collection and storage.
"""

import asyncio
import json
import sqlite3
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

from src.core.performance_optimizer import PerformanceOptimizer
from src.core.performance_monitor import PerformanceMonitor


@dataclass
class ComprehensivePerformanceData:
    """Comprehensive performance data combining system and application metrics."""
    # System metrics
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_usage: float
    
    # Application metrics
    total_operations: int
    successful_operations: int
    failed_operations: int
    average_response_time: float
    peak_response_time: float
    cache_hit_rate: float
    error_rate: float
    
    # Language-specific metrics
    language_metrics: Dict[str, Dict[str, Any]]
    
    # Operation-specific metrics
    operation_metrics: Dict[str, Dict[str, Any]]
    
    # Timestamp
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class PerformanceDataCollector:
    """Collects and stores comprehensive performance data."""
    
    def __init__(self, 
                 db_path: str = "data/performance_data.db",
                 collection_interval: int = 60,
                 max_history_days: int = 30):
        self.db_path = db_path
        self.collection_interval = collection_interval
        self.max_history_days = max_history_days
        
        # Initialize components
        self.performance_optimizer = PerformanceOptimizer()
        self.performance_monitor = PerformanceMonitor()
        
        # Ensure data directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Collection state
        self.is_collecting = False
        self.collection_task: Optional[asyncio.Task] = None
        
        logger.info(f"Performance Data Collector initialized with DB: {self.db_path}")
    
    def _init_database(self):
        """Initialize the performance data database."""
        try:
            # Create a persistent connection for in-memory database
            if self.db_path == ":memory:":
                self.db_connection = sqlite3.connect(self.db_path)
            else:
                self.db_connection = None
            
            conn = self.db_connection if self.db_connection else sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create comprehensive performance data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    network_usage REAL,
                    total_operations INTEGER,
                    successful_operations INTEGER,
                    failed_operations INTEGER,
                    average_response_time REAL,
                    peak_response_time REAL,
                    cache_hit_rate REAL,
                    error_rate REAL,
                    language_metrics TEXT,
                    operation_metrics TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for timestamp queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON performance_data(timestamp)
            """)
            
            # Create optimization recommendations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    category TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    description TEXT NOT NULL,
                    impact TEXT NOT NULL,
                    implementation TEXT NOT NULL,
                    estimated_improvement REAL,
                    applied BOOLEAN DEFAULT FALSE,
                    applied_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            
            if not self.db_connection:
                conn.close()
                
            logger.info("Performance data database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    async def start_collection(self):
        """Start performance data collection."""
        if self.is_collecting:
            logger.warning("Performance data collection already running")
            return
        
        self.is_collecting = True
        logger.info("Starting performance data collection")
        
        # Start the collection task
        self.collection_task = asyncio.create_task(self._collection_loop())
    
    async def stop_collection(self):
        """Stop performance data collection."""
        self.is_collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped performance data collection")
    
    async def _collection_loop(self):
        """Main collection loop."""
        while self.is_collecting:
            try:
                # Collect comprehensive performance data
                performance_data = await self.collect_comprehensive_data()
                
                # Store the data
                await self.store_performance_data(performance_data)
                
                # Generate and store optimization recommendations
                recommendations = await self.performance_optimizer.analyze_performance()
                await self.store_optimization_recommendations(recommendations)
                
                # Clean up old data
                await self.cleanup_old_data()
                
                # Wait for next collection interval
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in performance data collection: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def collect_comprehensive_data(self) -> ComprehensivePerformanceData:
        """Collect comprehensive performance data from all sources."""
        try:
            # Get system metrics from performance optimizer
            system_metrics = await self.performance_optimizer.collect_metrics()
            
            # Get application metrics from performance monitor
            app_stats = self.performance_monitor.real_time_stats
            lang_stats = dict(self.performance_monitor.language_stats)
            op_stats = dict(self.performance_monitor.operation_stats)
            
            # Calculate network usage (placeholder - would need actual network monitoring)
            network_usage = 0.0  # This would be implemented with actual network monitoring
            
            # Create comprehensive data
            comprehensive_data = ComprehensivePerformanceData(
                # System metrics
                cpu_usage=system_metrics.cpu_usage,
                memory_usage=system_metrics.memory_usage,
                disk_usage=system_metrics.disk_usage,
                network_usage=network_usage,
                
                # Application metrics
                total_operations=app_stats.get('total_operations', 0),
                successful_operations=app_stats.get('successful_operations', 0),
                failed_operations=app_stats.get('failed_operations', 0),
                average_response_time=app_stats.get('average_processing_time', 0.0),
                peak_response_time=app_stats.get('peak_processing_time', 0.0),
                cache_hit_rate=system_metrics.cache_hit_rate,
                error_rate=app_stats.get('error_rate', 0.0),
                
                # Language and operation metrics
                language_metrics=lang_stats,
                operation_metrics=op_stats,
                
                # Timestamp
                timestamp=datetime.now()
            )
            
            logger.debug(f"Collected comprehensive performance data: {comprehensive_data.total_operations} operations")
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Error collecting comprehensive data: {e}")
            # Return default data on error
            return ComprehensivePerformanceData(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_usage=0.0,
                total_operations=0,
                successful_operations=0,
                failed_operations=0,
                average_response_time=0.0,
                peak_response_time=0.0,
                cache_hit_rate=0.0,
                error_rate=0.0,
                language_metrics={},
                operation_metrics={},
                timestamp=datetime.now()
            )
    
    async def store_performance_data(self, data: ComprehensivePerformanceData):
        """Store performance data in the database."""
        try:
            conn = self.db_connection if self.db_connection else sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO performance_data (
                    timestamp, cpu_usage, memory_usage, disk_usage, network_usage,
                    total_operations, successful_operations, failed_operations,
                    average_response_time, peak_response_time, cache_hit_rate, error_rate,
                    language_metrics, operation_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.timestamp.isoformat(),
                data.cpu_usage,
                data.memory_usage,
                data.disk_usage,
                data.network_usage,
                data.total_operations,
                data.successful_operations,
                data.failed_operations,
                data.average_response_time,
                data.peak_response_time,
                data.cache_hit_rate,
                data.error_rate,
                json.dumps(data.language_metrics),
                json.dumps(data.operation_metrics)
            ))
            
            conn.commit()
            
            if not self.db_connection:
                conn.close()
                
            logger.debug(f"Stored performance data for {data.timestamp}")
            
        except Exception as e:
            logger.error(f"Error storing performance data: {e}")
    
    async def store_optimization_recommendations(self, recommendations: List):
        """Store optimization recommendations in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for recommendation in recommendations:
                    cursor.execute("""
                        INSERT INTO optimization_recommendations (
                            timestamp, category, priority, description, impact,
                            implementation, estimated_improvement
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        datetime.now().isoformat(),
                        recommendation.category,
                        recommendation.priority,
                        recommendation.description,
                        recommendation.impact,
                        recommendation.implementation,
                        recommendation.estimated_improvement
                    ))
                
                conn.commit()
                if recommendations:
                    logger.debug(f"Stored {len(recommendations)} optimization recommendations")
                
        except Exception as e:
            logger.error(f"Error storing optimization recommendations: {e}")
    
    async def get_performance_data(self, 
                                 hours: int = 24,
                                 include_language_metrics: bool = True,
                                 include_operation_metrics: bool = True) -> List[Dict[str, Any]]:
        """Retrieve performance data from the database."""
        try:
            from datetime import timedelta
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            conn = self.db_connection if self.db_connection else sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM performance_data 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC
            """, (cutoff_time.isoformat(),))
            
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries
            data = []
            for row in rows:
                row_dict = {
                    'id': row[0],
                    'timestamp': row[1],
                    'cpu_usage': row[2],
                    'memory_usage': row[3],
                    'disk_usage': row[4],
                    'network_usage': row[5],
                    'total_operations': row[6],
                    'successful_operations': row[7],
                    'failed_operations': row[8],
                    'average_response_time': row[9],
                    'peak_response_time': row[10],
                    'cache_hit_rate': row[11],
                    'error_rate': row[12]
                }
                
                if include_language_metrics:
                    row_dict['language_metrics'] = json.loads(row[13]) if row[13] else {}
                
                if include_operation_metrics:
                    row_dict['operation_metrics'] = json.loads(row[14]) if row[14] else {}
                
                data.append(row_dict)
            
            if not self.db_connection:
                conn.close()
                
            logger.debug(f"Retrieved {len(data)} performance data records")
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving performance data: {e}")
            return []
    
    async def get_optimization_recommendations(self, 
                                             hours: int = 24,
                                             applied_only: bool = False) -> List[Dict[str, Any]]:
        """Retrieve optimization recommendations from the database."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if applied_only:
                    cursor.execute("""
                        SELECT * FROM optimization_recommendations 
                        WHERE timestamp >= ? AND applied = TRUE
                        ORDER BY timestamp DESC
                    """, (cutoff_time.isoformat(),))
                else:
                    cursor.execute("""
                        SELECT * FROM optimization_recommendations 
                        WHERE timestamp >= ?
                        ORDER BY timestamp DESC
                    """, (cutoff_time.isoformat(),))
                
                rows = cursor.fetchall()
                
                recommendations = []
                for row in rows:
                    recommendations.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'category': row[2],
                        'priority': row[3],
                        'description': row[4],
                        'impact': row[5],
                        'implementation': row[6],
                        'estimated_improvement': row[7],
                        'applied': bool(row[8]),
                        'applied_at': row[9]
                    })
                
                logger.debug(f"Retrieved {len(recommendations)} optimization recommendations")
                return recommendations
                
        except Exception as e:
            logger.error(f"Error retrieving optimization recommendations: {e}")
            return []
    
    async def cleanup_old_data(self):
        """Clean up old performance data."""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.max_history_days)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old performance data
                cursor.execute("""
                    DELETE FROM performance_data 
                    WHERE timestamp < ?
                """, (cutoff_time.isoformat(),))
                
                performance_deleted = cursor.rowcount
                
                # Delete old optimization recommendations
                cursor.execute("""
                    DELETE FROM optimization_recommendations 
                    WHERE timestamp < ?
                """, (cutoff_time.isoformat(),))
                
                recommendations_deleted = cursor.rowcount
                
                conn.commit()
                
                if performance_deleted > 0 or recommendations_deleted > 0:
                    logger.info(f"Cleaned up {performance_deleted} performance records and {recommendations_deleted} recommendations")
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    async def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get a summary of performance data."""
        try:
            data = await self.get_performance_data(hours, include_language_metrics=False, include_operation_metrics=False)
            
            if not data:
                return {"success": False, "message": "No performance data available"}
            
            # Calculate summary statistics
            cpu_values = [d['cpu_usage'] for d in data if d['cpu_usage'] is not None]
            memory_values = [d['memory_usage'] for d in data if d['memory_usage'] is not None]
            response_times = [d['average_response_time'] for d in data if d['average_response_time'] is not None]
            
            summary = {
                "success": True,
                "period_hours": hours,
                "total_records": len(data),
                "system_metrics": {
                    "avg_cpu_usage": sum(cpu_values) / len(cpu_values) if cpu_values else 0.0,
                    "max_cpu_usage": max(cpu_values) if cpu_values else 0.0,
                    "avg_memory_usage": sum(memory_values) / len(memory_values) if memory_values else 0.0,
                    "max_memory_usage": max(memory_values) if memory_values else 0.0
                },
                "application_metrics": {
                    "avg_response_time": sum(response_times) / len(response_times) if response_times else 0.0,
                    "max_response_time": max(response_times) if response_times else 0.0,
                    "total_operations": sum(d['total_operations'] for d in data),
                    "successful_operations": sum(d['successful_operations'] for d in data),
                    "failed_operations": sum(d['failed_operations'] for d in data)
                }
            }
            
            # Calculate error rate
            total_ops = summary["application_metrics"]["total_operations"]
            failed_ops = summary["application_metrics"]["failed_operations"]
            summary["application_metrics"]["error_rate"] = (failed_ops / total_ops * 100) if total_ops > 0 else 0.0
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {"success": False, "error": str(e)}


# Global performance data collector instance
performance_data_collector = PerformanceDataCollector()


async def get_performance_data_collector() -> PerformanceDataCollector:
    """Get the global performance data collector instance."""
    return performance_data_collector
