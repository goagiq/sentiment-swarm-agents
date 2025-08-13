"""
Real-Time Data Stream Processor

Enhanced data stream processor for real-time analytics dashboard with:
- Live data streaming capabilities
- Real-time data validation
- Stream processing optimization
- Integration with existing stream processor
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque, defaultdict
import numpy as np

from loguru import logger

# Import existing stream processor
from ..real_time.stream_processor import DataStreamProcessor as BaseStreamProcessor
from ..real_time.stream_processor import DataPoint, StreamConfig

from ...config.real_time_analytics_config import get_real_time_analytics_config


@dataclass
class RealTimeDataPoint(DataPoint):
    """Enhanced data point for real-time analytics."""
    quality_score: float = 1.0
    processing_latency: float = 0.0
    source_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamMetrics:
    """Metrics for stream processing performance."""
    total_processed: int = 0
    total_filtered: int = 0
    total_aggregated: int = 0
    processing_errors: int = 0
    avg_latency: float = 0.0
    throughput: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)


class EnhancedDataStreamProcessor(BaseStreamProcessor):
    """
    Enhanced data stream processor for real-time analytics dashboard.
    Extends the base stream processor with real-time analytics capabilities.
    """
    
    def __init__(self, config: Optional[StreamConfig] = None):
        """Initialize the enhanced data stream processor."""
        super().__init__(config)
        
        # Get real-time analytics configuration
        self.rt_config = get_real_time_analytics_config()
        
        # Enhanced features
        self.quality_validators: Dict[str, Callable] = {}
        self.real_time_analytics: Dict[str, Callable] = {}
        self.stream_metrics = StreamMetrics()
        self.alert_conditions: Dict[str, Callable] = {}
        
        # Real-time processing queues
        self.high_priority_queue = deque(maxlen=100)
        self.normal_priority_queue = deque(maxlen=1000)
        self.low_priority_queue = deque(maxlen=5000)
        
        # Performance monitoring
        self.performance_monitor = {
            'processing_times': deque(maxlen=1000),
            'queue_sizes': deque(maxlen=100),
            'error_counts': defaultdict(int)
        }
        
        logger.info("EnhancedDataStreamProcessor initialized")
    
    def add_quality_validator(self, name: str, validator: Callable) -> None:
        """Add a data quality validator."""
        self.quality_validators[name] = validator
        logger.info(f"Added quality validator: {name}")
    
    def add_real_time_analytics(self, name: str, analyzer: Callable) -> None:
        """Add a real-time analytics processor."""
        self.real_time_analytics[name] = analyzer
        logger.info(f"Added real-time analytics: {name}")
    
    def add_alert_condition(self, name: str, condition: Callable) -> None:
        """Add an alert condition."""
        self.alert_conditions[name] = condition
        logger.info(f"Added alert condition: {name}")
    
    async def add_data_point_enhanced(
        self, 
        value: Union[float, int, str], 
        source: str = "default",
        priority: str = "normal",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a data point with enhanced features."""
        try:
            # Create enhanced data point
            data_point = RealTimeDataPoint(
                value=value,
                timestamp=datetime.now(),
                source=source,
                metadata=metadata or {},
                quality_score=1.0,
                processing_latency=0.0
            )
            
            # Validate data quality
            data_point.quality_score = await self._validate_data_quality(data_point)
            
            # Add to appropriate priority queue
            if priority == "high":
                self.high_priority_queue.append(data_point)
            elif priority == "low":
                self.low_priority_queue.append(data_point)
            else:
                self.normal_priority_queue.append(data_point)
            
            # Update metrics
            self.stream_metrics.total_processed += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding enhanced data point: {str(e)}")
            self.stream_metrics.processing_errors += 1
            return False
    
    async def _validate_data_quality(self, data_point: RealTimeDataPoint) -> float:
        """Validate data quality using registered validators."""
        quality_score = 1.0
        
        for name, validator in self.quality_validators.items():
            try:
                score = await validator(data_point)
                quality_score = min(quality_score, score)
            except Exception as e:
                logger.warning(f"Quality validator {name} failed: {str(e)}")
        
        return quality_score
    
    async def _process_real_time_analytics(self, data_point: RealTimeDataPoint) -> Dict[str, Any]:
        """Process real-time analytics on data point."""
        results = {}
        
        for name, analyzer in self.real_time_analytics.items():
            try:
                result = await analyzer(data_point)
                results[name] = result
            except Exception as e:
                logger.warning(f"Real-time analytics {name} failed: {str(e)}")
        
        return results
    
    async def _check_alert_conditions(self, data_point: RealTimeDataPoint) -> List[Dict[str, Any]]:
        """Check alert conditions for data point."""
        alerts = []
        
        for name, condition in self.alert_conditions.items():
            try:
                if await condition(data_point):
                    alerts.append({
                        "condition": name,
                        "timestamp": data_point.timestamp,
                        "source": data_point.source,
                        "value": data_point.value,
                        "severity": "medium"
                    })
            except Exception as e:
                logger.warning(f"Alert condition {name} failed: {str(e)}")
        
        return alerts
    
    async def _enhanced_processing_loop(self):
        """Enhanced processing loop with real-time analytics."""
        while self.is_processing:
            try:
                # Process high priority queue first
                if self.high_priority_queue:
                    data_point = self.high_priority_queue.popleft()
                    await self._process_enhanced_data_point(data_point, "high")
                
                # Process normal priority queue
                elif self.normal_priority_queue:
                    data_point = self.normal_priority_queue.popleft()
                    await self._process_enhanced_data_point(data_point, "normal")
                
                # Process low priority queue
                elif self.low_priority_queue:
                    data_point = self.low_priority_queue.popleft()
                    await self._process_enhanced_data_point(data_point, "low")
                
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Sleep based on configuration
                await asyncio.sleep(self.config.processing_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in enhanced processing loop: {str(e)}")
                self.stream_metrics.processing_errors += 1
                await asyncio.sleep(1.0)
    
    async def _process_enhanced_data_point(
        self, 
        data_point: RealTimeDataPoint, 
        priority: str
    ) -> None:
        """Process a single enhanced data point."""
        start_time = datetime.now()
        
        try:
            # Process real-time analytics
            analytics_results = await self._process_real_time_analytics(data_point)
            
            # Check alert conditions
            alerts = await self._check_alert_conditions(data_point)
            
            # Update data point with results
            data_point.processing_latency = (datetime.now() - start_time).total_seconds()
            data_point.metadata.update({
                "analytics_results": analytics_results,
                "alerts": alerts,
                "priority": priority
            })
            
            # Route to consumers
            await self._route_to_consumers(data_point)
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.performance_monitor['processing_times'].append(processing_time)
            
        except Exception as e:
            logger.error(f"Error processing enhanced data point: {str(e)}")
            self.stream_metrics.processing_errors += 1
    
    async def _update_performance_metrics(self) -> None:
        """Update performance monitoring metrics."""
        # Calculate average processing time
        if self.performance_monitor['processing_times']:
            avg_time = np.mean(self.performance_monitor['processing_times'])
            self.stream_metrics.avg_latency = avg_time
        
        # Calculate throughput
        total_time = (datetime.now() - self.stream_metrics.start_time).total_seconds()
        if total_time > 0:
            self.stream_metrics.throughput = self.stream_metrics.total_processed / total_time
        
        # Update queue sizes
        queue_sizes = {
            'high_priority': len(self.high_priority_queue),
            'normal_priority': len(self.normal_priority_queue),
            'low_priority': len(self.low_priority_queue)
        }
        self.performance_monitor['queue_sizes'].append(queue_sizes)
    
    def get_stream_metrics(self) -> Dict[str, Any]:
        """Get current stream metrics."""
        return {
            'total_processed': self.stream_metrics.total_processed,
            'total_filtered': self.stream_metrics.total_filtered,
            'total_aggregated': self.stream_metrics.total_aggregated,
            'processing_errors': self.stream_metrics.processing_errors,
            'avg_latency': self.stream_metrics.avg_latency,
            'throughput': self.stream_metrics.throughput,
            'queue_sizes': {
                'high_priority': len(self.high_priority_queue),
                'normal_priority': len(self.normal_priority_queue),
                'low_priority': len(self.low_priority_queue)
            },
            'uptime': (datetime.now() - self.stream_metrics.start_time).total_seconds()
        }
    
    async def start_enhanced_processing(self) -> None:
        """Start enhanced processing with real-time analytics."""
        await self.start_processing()
        
        # Start enhanced processing loop
        self.enhanced_processing_task = asyncio.create_task(self._enhanced_processing_loop())
        logger.info("Enhanced data stream processing started")
    
    async def stop_enhanced_processing(self) -> None:
        """Stop enhanced processing."""
        await self.stop_processing()
        
        if hasattr(self, 'enhanced_processing_task'):
            self.enhanced_processing_task.cancel()
            try:
                await self.enhanced_processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Enhanced data stream processing stopped")


# Factory function for creating enhanced stream processor
def create_enhanced_stream_processor() -> EnhancedDataStreamProcessor:
    """Create an enhanced data stream processor with default configuration."""
    config = get_real_time_analytics_config()
    
    stream_config = StreamConfig(
        buffer_size=config.stream_processing.buffer_size,
        batch_size=config.stream_processing.batch_size,
        processing_interval=config.stream_processing.processing_interval,
        enable_batching=config.stream_processing.enable_batching,
        enable_filtering=config.stream_processing.enable_filtering,
        enable_aggregation=config.stream_processing.enable_aggregation,
        max_processing_latency=config.stream_processing.max_latency
    )
    
    return EnhancedDataStreamProcessor(stream_config)
