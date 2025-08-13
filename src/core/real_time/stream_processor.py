"""
Data Stream Processor

This module provides real-time data ingestion and processing capabilities
for continuous data streams.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DataPoint:
    """Represents a single data point in a stream"""
    value: Union[float, int, str]
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamConfig:
    """Configuration for data stream processing"""
    buffer_size: int = 1000
    batch_size: int = 100
    processing_interval: float = 0.1  # seconds
    enable_batching: bool = True
    enable_filtering: bool = True
    enable_aggregation: bool = True
    max_processing_latency: float = 1.0  # seconds


class DataStreamProcessor:
    """
    Real-time data stream processor that handles continuous data ingestion,
    processing, and routing to various consumers.
    """
    
    def __init__(self, config: Optional[StreamConfig] = None):
        """
        Initialize the data stream processor
        
        Args:
            config: Stream processing configuration
        """
        self.config = config or StreamConfig()
        self.input_buffer = deque(maxlen=self.config.buffer_size)
        self.output_buffers: Dict[str, deque] = {}
        self.processors: Dict[str, Callable] = {}
        self.consumers: Dict[str, List[Callable]] = defaultdict(list)
        self.filters: Dict[str, Callable] = {}
        self.aggregators: Dict[str, Callable] = {}
        
        self.is_processing = False
        self.processing_task = None
        self.stats = {
            'total_processed': 0,
            'total_filtered': 0,
            'total_aggregated': 0,
            'processing_errors': 0,
            'start_time': datetime.now()
        }
        
        logger.info("DataStreamProcessor initialized")
    
    async def start_processing(self):
        """Start the data stream processing"""
        if self.is_processing:
            logger.warning("Stream processing is already running")
            return
        
        self.is_processing = True
        self.processing_task = asyncio.create_task(self._processing_loop())
        logger.info("Data stream processing started")
    
    async def stop_processing(self):
        """Stop the data stream processing"""
        if not self.is_processing:
            return
        
        self.is_processing = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Data stream processing stopped")
    
    def add_data_point(self, value: Union[float, int, str], 
                       source: str = "default",
                       timestamp: Optional[datetime] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """
        Add a data point to the input stream
        
        Args:
            value: Data value
            source: Data source identifier
            timestamp: Optional timestamp (defaults to current time)
            metadata: Optional metadata
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        data_point = DataPoint(
            value=value,
            timestamp=timestamp,
            source=source,
            metadata=metadata or {}
        )
        
        self.input_buffer.append(data_point)
        
        # Process immediately if buffer is getting full
        if len(self.input_buffer) >= self.config.buffer_size * 0.8:
            asyncio.create_task(self._process_batch())
    
    def register_processor(self, name: str, processor_func: Callable[[DataPoint], Any]):
        """
        Register a data processor function
        
        Args:
            name: Processor name
            processor_func: Function to process data points
        """
        self.processors[name] = processor_func
        logger.info(f"Registered processor: {name}")
    
    def register_consumer(self, stream_name: str, consumer_func: Callable[[Any], None]):
        """
        Register a data consumer function
        
        Args:
            stream_name: Name of the output stream
            consumer_func: Function to consume processed data
        """
        self.consumers[stream_name].append(consumer_func)
        if stream_name not in self.output_buffers:
            self.output_buffers[stream_name] = deque(maxlen=self.config.buffer_size)
        logger.info(f"Registered consumer for stream: {stream_name}")
    
    def register_filter(self, name: str, filter_func: Callable[[DataPoint], bool]):
        """
        Register a data filter function
        
        Args:
            name: Filter name
            filter_func: Function that returns True to keep data point, False to filter out
        """
        self.filters[name] = filter_func
        logger.info(f"Registered filter: {name}")
    
    def register_aggregator(self, name: str, 
                           aggregator_func: Callable[[List[DataPoint]], Any]):
        """
        Register a data aggregator function
        
        Args:
            name: Aggregator name
            aggregator_func: Function to aggregate multiple data points
        """
        self.aggregators[name] = aggregator_func
        logger.info(f"Registered aggregator: {name}")
    
    async def _processing_loop(self):
        """Main processing loop"""
        while self.is_processing:
            try:
                if len(self.input_buffer) > 0:
                    if self.config.enable_batching and len(self.input_buffer) >= self.config.batch_size:
                        await self._process_batch()
                    else:
                        await self._process_single()
                
                await asyncio.sleep(self.config.processing_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                self.stats['processing_errors'] += 1
                await asyncio.sleep(1.0)
    
    async def _process_single(self):
        """Process a single data point"""
        if len(self.input_buffer) == 0:
            return
        
        data_point = self.input_buffer.popleft()
        await self._process_data_point(data_point)
    
    async def _process_batch(self):
        """Process a batch of data points"""
        if len(self.input_buffer) == 0:
            return
        
        batch = []
        batch_size = min(self.config.batch_size, len(self.input_buffer))
        
        for _ in range(batch_size):
            if len(self.input_buffer) > 0:
                batch.append(self.input_buffer.popleft())
        
        # Process each data point in the batch
        for data_point in batch:
            await self._process_data_point(data_point)
        
        # Apply aggregators if enabled
        if self.config.enable_aggregation and self.aggregators:
            await self._apply_aggregators(batch)
    
    async def _process_data_point(self, data_point: DataPoint):
        """Process a single data point through the pipeline"""
        try:
            # Apply filters
            if self.config.enable_filtering and self.filters:
                for filter_name, filter_func in self.filters.items():
                    try:
                        if not filter_func(data_point):
                            self.stats['total_filtered'] += 1
                            return
                    except Exception as e:
                        logger.error(f"Error in filter {filter_name}: {str(e)}")
            
            # Apply processors
            processed_data = data_point
            for processor_name, processor_func in self.processors.items():
                try:
                    processed_data = processor_func(processed_data)
                except Exception as e:
                    logger.error(f"Error in processor {processor_name}: {str(e)}")
                    self.stats['processing_errors'] += 1
                    return
            
            # Route to consumers
            await self._route_to_consumers(processed_data)
            
            self.stats['total_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing data point: {str(e)}")
            self.stats['processing_errors'] += 1
    
    async def _apply_aggregators(self, batch: List[DataPoint]):
        """Apply aggregators to a batch of data points"""
        for aggregator_name, aggregator_func in self.aggregators.items():
            try:
                aggregated_result = aggregator_func(batch)
                await self._route_to_consumers(aggregated_result, f"{aggregator_name}_output")
                self.stats['total_aggregated'] += 1
            except Exception as e:
                logger.error(f"Error in aggregator {aggregator_name}: {str(e)}")
                self.stats['processing_errors'] += 1
    
    async def _route_to_consumers(self, data: Any, stream_name: str = "default"):
        """Route processed data to consumers"""
        if stream_name in self.output_buffers:
            self.output_buffers[stream_name].append(data)
        
        if stream_name in self.consumers:
            for consumer_func in self.consumers[stream_name]:
                try:
                    # Run consumer in background to avoid blocking
                    asyncio.create_task(self._run_consumer(consumer_func, data))
                except Exception as e:
                    logger.error(f"Error routing to consumer: {str(e)}")
    
    async def _run_consumer(self, consumer_func: Callable, data: Any):
        """Run a consumer function asynchronously"""
        try:
            if asyncio.iscoroutinefunction(consumer_func):
                await consumer_func(data)
            else:
                consumer_func(data)
        except Exception as e:
            logger.error(f"Error in consumer: {str(e)}")
    
    def get_stream_data(self, stream_name: str, limit: int = 100) -> List[Any]:
        """
        Get data from a specific output stream
        
        Args:
            stream_name: Name of the output stream
            limit: Maximum number of data points to return
            
        Returns:
            List of data points from the stream
        """
        if stream_name not in self.output_buffers:
            return []
        
        data = list(self.output_buffers[stream_name])
        return data[-limit:] if limit > 0 else data
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        
        return {
            'uptime_seconds': uptime,
            'total_processed': self.stats['total_processed'],
            'total_filtered': self.stats['total_filtered'],
            'total_aggregated': self.stats['total_aggregated'],
            'processing_errors': self.stats['processing_errors'],
            'input_buffer_size': len(self.input_buffer),
            'output_streams': {
                name: len(buffer) for name, buffer in self.output_buffers.items()
            },
            'active_processors': len(self.processors),
            'active_filters': len(self.filters),
            'active_aggregators': len(self.aggregators),
            'is_processing': self.is_processing
        }
    
    def clear_buffers(self, stream_name: Optional[str] = None):
        """
        Clear data buffers
        
        Args:
            stream_name: Optional specific stream to clear (if None, clears all)
        """
        if stream_name:
            if stream_name in self.output_buffers:
                self.output_buffers[stream_name].clear()
            logger.info(f"Cleared buffer for stream: {stream_name}")
        else:
            self.input_buffer.clear()
            for buffer in self.output_buffers.values():
                buffer.clear()
            logger.info("Cleared all buffers")
    
    def add_numeric_processor(self, name: str, operation: str = "identity"):
        """
        Add a numeric data processor
        
        Args:
            name: Processor name
            operation: Operation to perform (identity, scale, normalize, etc.)
        """
        def numeric_processor(data_point: DataPoint) -> DataPoint:
            if isinstance(data_point.value, (int, float)):
                if operation == "scale":
                    # Scale by a factor (can be configured via metadata)
                    factor = data_point.metadata.get('scale_factor', 1.0)
                    new_value = data_point.value * factor
                elif operation == "normalize":
                    # Simple normalization (0-1 range)
                    new_value = max(0, min(1, data_point.value))
                elif operation == "log":
                    # Natural logarithm
                    new_value = np.log(max(1e-10, data_point.value))
                else:
                    # Identity operation
                    new_value = data_point.value
                
                return DataPoint(
                    value=new_value,
                    timestamp=data_point.timestamp,
                    source=data_point.source,
                    metadata=data_point.metadata
                )
            return data_point
        
        self.register_processor(name, numeric_processor)
    
    def add_time_window_filter(self, name: str, window_seconds: int = 60):
        """
        Add a time window filter
        
        Args:
            name: Filter name
            window_seconds: Time window in seconds
        """
        def time_filter(data_point: DataPoint) -> bool:
            cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
            return data_point.timestamp >= cutoff_time
        
        self.register_filter(name, time_filter)
    
    def add_threshold_filter(self, name: str, min_value: Optional[float] = None, 
                           max_value: Optional[float] = None):
        """
        Add a threshold filter
        
        Args:
            name: Filter name
            min_value: Minimum threshold
            max_value: Maximum threshold
        """
        def threshold_filter(data_point: DataPoint) -> bool:
            if not isinstance(data_point.value, (int, float)):
                return True
            
            if min_value is not None and data_point.value < min_value:
                return False
            if max_value is not None and data_point.value > max_value:
                return False
            return True
        
        self.register_filter(name, threshold_filter)
    
    def add_average_aggregator(self, name: str):
        """Add an average aggregator"""
        def average_aggregator(batch: List[DataPoint]) -> Dict[str, Any]:
            numeric_values = [
                dp.value for dp in batch 
                if isinstance(dp.value, (int, float))
            ]
            
            if not numeric_values:
                return {'average': 0, 'count': 0}
            
            return {
                'average': float(np.mean(numeric_values)),
                'count': len(numeric_values),
                'timestamp': batch[-1].timestamp.isoformat()
            }
        
        self.register_aggregator(name, average_aggregator)
