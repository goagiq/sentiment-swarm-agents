"""
Stream Analytics

Real-time stream analytics for analytics dashboard with:
- Real-time aggregations
- Window functions
- Pattern detection
- Anomaly detection
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import numpy as np

from loguru import logger

from .data_stream_processor import RealTimeDataPoint
from ...config.real_time_analytics_config import get_real_time_analytics_config


@dataclass
class AnalyticsWindow:
    """Represents a time window for analytics."""
    start_time: datetime
    end_time: datetime
    data_points: List[RealTimeDataPoint] = field(default_factory=list)
    aggregations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamAnalyticsResult:
    """Result of stream analytics processing."""
    window_id: str
    timestamp: datetime
    aggregations: Dict[str, Any]
    patterns: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamAnalytics:
    """
    Real-time stream analytics processor for continuous data analysis.
    """
    
    def __init__(self):
        """Initialize the stream analytics processor."""
        self.config = get_real_time_analytics_config()
        
        # Analytics windows
        self.windows: Dict[str, AnalyticsWindow] = {}
        self.window_configs: Dict[str, Dict[str, Any]] = {}
        
        # Analytics functions
        self.aggregation_functions: Dict[str, Callable] = {}
        self.pattern_detectors: Dict[str, Callable] = {}
        self.anomaly_detectors: Dict[str, Callable] = {}
        
        # Results storage
        self.analytics_results: deque = deque(maxlen=1000)
        self.metrics = {
            'total_processed': 0,
            'windows_created': 0,
            'analytics_computed': 0,
            'errors': 0
        }
        
        # Initialize default analytics functions
        self._initialize_default_functions()
        
        logger.info("StreamAnalytics initialized")
    
    def _initialize_default_functions(self) -> None:
        """Initialize default analytics functions."""
        # Default aggregation functions
        self.aggregation_functions.update({
            'count': lambda data: len(data),
            'sum': lambda data: sum(d.value for d in data if isinstance(d.value, (int, float))),
            'mean': lambda data: np.mean([d.value for d in data if isinstance(d.value, (int, float))]),
            'min': lambda data: min(d.value for d in data if isinstance(d.value, (int, float))),
            'max': lambda data: max(d.value for d in data if isinstance(d.value, (int, float))),
            'std': lambda data: np.std([d.value for d in data if isinstance(d.value, (int, float))])
        })
        
        # Default pattern detectors
        self.pattern_detectors.update({
            'trend': self._detect_trend,
            'seasonality': self._detect_seasonality,
            'spike': self._detect_spike
        })
        
        # Default anomaly detectors
        self.anomaly_detectors.update({
            'zscore': self._zscore_anomaly,
            'iqr': self._iqr_anomaly,
            'isolation_forest': self._isolation_forest_anomaly
        })
    
    def add_window(self, window_id: str, duration: timedelta, 
                   config: Optional[Dict[str, Any]] = None) -> None:
        """Add a new analytics window."""
        self.window_configs[window_id] = {
            'duration': duration,
            'config': config or {}
        }
        logger.info(f"Added analytics window: {window_id}")
    
    def add_aggregation_function(self, name: str, func: Callable) -> None:
        """Add a custom aggregation function."""
        self.aggregation_functions[name] = func
        logger.info(f"Added aggregation function: {name}")
    
    def add_pattern_detector(self, name: str, detector: Callable) -> None:
        """Add a custom pattern detector."""
        self.pattern_detectors[name] = detector
        logger.info(f"Added pattern detector: {name}")
    
    def add_anomaly_detector(self, name: str, detector: Callable) -> None:
        """Add a custom anomaly detector."""
        self.anomaly_detectors[name] = detector
        logger.info(f"Added anomaly detector: {name}")
    
    async def process_data_point(self, data_point: RealTimeDataPoint) -> List[StreamAnalyticsResult]:
        """Process a data point through all analytics windows."""
        results = []
        
        try:
            # Add data point to relevant windows
            for window_id, config in self.window_configs.items():
                window = await self._get_or_create_window(window_id, data_point.timestamp)
                
                if window and self._is_data_point_in_window(data_point, window):
                    window.data_points.append(data_point)
                    results.append(await self._compute_window_analytics(window_id, window))
            
            self.metrics['total_processed'] += 1
            return results
            
        except Exception as e:
            logger.error(f"Error processing data point: {str(e)}")
            self.metrics['errors'] += 1
            return []
    
    async def _get_or_create_window(self, window_id: str, timestamp: datetime) -> Optional[AnalyticsWindow]:
        """Get or create a window for the given timestamp."""
        config = self.window_configs.get(window_id)
        if not config:
            return None
        
        duration = config['duration']
        window_start = timestamp - duration
        
        # Check if we need to create a new window
        if window_id not in self.windows or self.windows[window_id].end_time <= timestamp:
            # Create new window
            new_window = AnalyticsWindow(
                start_time=window_start,
                end_time=timestamp
            )
            self.windows[window_id] = new_window
            self.metrics['windows_created'] += 1
            
            # Clean up old data points
            await self._cleanup_old_data(window_id, new_window)
        
        return self.windows[window_id]
    
    def _is_data_point_in_window(self, data_point: RealTimeDataPoint, window: AnalyticsWindow) -> bool:
        """Check if data point is within the window."""
        return window.start_time <= data_point.timestamp <= window.end_time
    
    async def _cleanup_old_data(self, window_id: str, current_window: AnalyticsWindow) -> None:
        """Clean up old data points from the window."""
        if window_id in self.windows:
            old_window = self.windows[window_id]
            # Keep only data points that are still relevant
            old_window.data_points = [
                dp for dp in old_window.data_points
                if dp.timestamp >= current_window.start_time
            ]
    
    async def _compute_window_analytics(self, window_id: str, window: AnalyticsWindow) -> StreamAnalyticsResult:
        """Compute analytics for a window."""
        try:
            # Compute aggregations
            aggregations = await self._compute_aggregations(window.data_points)
            
            # Detect patterns
            patterns = await self._detect_patterns(window.data_points)
            
            # Detect anomalies
            anomalies = await self._detect_anomalies(window.data_points)
            
            # Create result
            result = StreamAnalyticsResult(
                window_id=window_id,
                timestamp=datetime.now(),
                aggregations=aggregations,
                patterns=patterns,
                anomalies=anomalies,
                metadata={
                    'window_start': window.start_time.isoformat(),
                    'window_end': window.end_time.isoformat(),
                    'data_points_count': len(window.data_points)
                }
            )
            
            # Store result
            self.analytics_results.append(result)
            self.metrics['analytics_computed'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error computing window analytics: {str(e)}")
            self.metrics['errors'] += 1
            return StreamAnalyticsResult(
                window_id=window_id,
                timestamp=datetime.now(),
                aggregations={},
                patterns=[],
                anomalies=[],
                metadata={'error': str(e)}
            )
    
    async def _compute_aggregations(self, data_points: List[RealTimeDataPoint]) -> Dict[str, Any]:
        """Compute aggregations on data points."""
        aggregations = {}
        
        for name, func in self.aggregation_functions.items():
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(data_points)
                else:
                    result = func(data_points)
                aggregations[name] = result
            except Exception as e:
                logger.warning(f"Aggregation function {name} failed: {str(e)}")
                aggregations[name] = None
        
        return aggregations
    
    async def _detect_patterns(self, data_points: List[RealTimeDataPoint]) -> List[Dict[str, Any]]:
        """Detect patterns in data points."""
        patterns = []
        
        for name, detector in self.pattern_detectors.items():
            try:
                if asyncio.iscoroutinefunction(detector):
                    detected_patterns = await detector(data_points)
                else:
                    detected_patterns = detector(data_points)
                
                if detected_patterns:
                    patterns.extend(detected_patterns)
            except Exception as e:
                logger.warning(f"Pattern detector {name} failed: {str(e)}")
        
        return patterns
    
    async def _detect_anomalies(self, data_points: List[RealTimeDataPoint]) -> List[Dict[str, Any]]:
        """Detect anomalies in data points."""
        anomalies = []
        
        for name, detector in self.anomaly_detectors.items():
            try:
                if asyncio.iscoroutinefunction(detector):
                    detected_anomalies = await detector(data_points)
                else:
                    detected_anomalies = detector(data_points)
                
                if detected_anomalies:
                    anomalies.extend(detected_anomalies)
            except Exception as e:
                logger.warning(f"Anomaly detector {name} failed: {str(e)}")
        
        return anomalies
    
    # Default pattern detection methods
    def _detect_trend(self, data_points: List[RealTimeDataPoint]) -> List[Dict[str, Any]]:
        """Detect trend patterns in data."""
        if len(data_points) < 3:
            return []
        
        values = [dp.value for dp in data_points if isinstance(dp.value, (int, float))]
        if len(values) < 3:
            return []
        
        # Simple linear trend detection
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) > 0.1:  # Threshold for trend detection
            return [{
                'type': 'trend',
                'direction': 'increasing' if slope > 0 else 'decreasing',
                'strength': abs(slope),
                'confidence': 0.8
            }]
        
        return []
    
    def _detect_seasonality(self, data_points: List[RealTimeDataPoint]) -> List[Dict[str, Any]]:
        """Detect seasonality patterns in data."""
        # Simple seasonality detection (placeholder)
        return []
    
    def _detect_spike(self, data_points: List[RealTimeDataPoint]) -> List[Dict[str, Any]]:
        """Detect spike patterns in data."""
        if len(data_points) < 5:
            return []
        
        values = [dp.value for dp in data_points if isinstance(dp.value, (int, float))]
        if len(values) < 5:
            return []
        
        # Detect spikes using moving average
        window_size = 3
        spikes = []
        
        for i in range(window_size, len(values)):
            window = values[i-window_size:i]
            current = values[i]
            window_mean = np.mean(window)
            window_std = np.std(window)
            
            if window_std > 0 and abs(current - window_mean) > 2 * window_std:
                spikes.append({
                    'type': 'spike',
                    'position': i,
                    'value': current,
                    'threshold': window_mean + 2 * window_std,
                    'confidence': 0.9
                })
        
        return spikes
    
    # Default anomaly detection methods
    def _zscore_anomaly(self, data_points: List[RealTimeDataPoint]) -> List[Dict[str, Any]]:
        """Detect anomalies using Z-score method."""
        if len(data_points) < 10:
            return []
        
        values = [dp.value for dp in data_points if isinstance(dp.value, (int, float))]
        if len(values) < 10:
            return []
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return []
        
        anomalies = []
        for i, value in enumerate(values):
            z_score = abs((value - mean) / std)
            if z_score > 3:  # 3-sigma rule
                anomalies.append({
                    'type': 'zscore_anomaly',
                    'position': i,
                    'value': value,
                    'z_score': z_score,
                    'threshold': 3,
                    'confidence': min(z_score / 5, 1.0)
                })
        
        return anomalies
    
    def _iqr_anomaly(self, data_points: List[RealTimeDataPoint]) -> List[Dict[str, Any]]:
        """Detect anomalies using IQR method."""
        if len(data_points) < 10:
            return []
        
        values = [dp.value for dp in data_points if isinstance(dp.value, (int, float))]
        if len(values) < 10:
            return []
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return []
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        anomalies = []
        for i, value in enumerate(values):
            if value < lower_bound or value > upper_bound:
                anomalies.append({
                    'type': 'iqr_anomaly',
                    'position': i,
                    'value': value,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'confidence': 0.8
                })
        
        return anomalies
    
    def _isolation_forest_anomaly(self, data_points: List[RealTimeDataPoint]) -> List[Dict[str, Any]]:
        """Detect anomalies using isolation forest (simplified)."""
        # Simplified isolation forest implementation
        # In a real implementation, you would use scikit-learn's IsolationForest
        return []
    
    def get_analytics_results(self, limit: Optional[int] = None) -> List[StreamAnalyticsResult]:
        """Get recent analytics results."""
        if limit is None:
            return list(self.analytics_results)
        return list(self.analytics_results)[-limit:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get analytics metrics."""
        return {
            'total_processed': self.metrics['total_processed'],
            'windows_created': self.metrics['windows_created'],
            'analytics_computed': self.metrics['analytics_computed'],
            'errors': self.metrics['errors'],
            'active_windows': len(self.windows),
            'results_stored': len(self.analytics_results)
        }


# Factory function for creating stream analytics
def create_stream_analytics() -> StreamAnalytics:
    """Create a stream analytics processor with default configuration."""
    return StreamAnalytics()
