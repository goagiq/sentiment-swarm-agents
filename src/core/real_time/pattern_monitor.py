"""
Real-Time Pattern Monitor

This module provides continuous pattern detection and monitoring capabilities
for real-time data streams.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PatternEvent:
    """Represents a detected pattern event"""
    pattern_id: str
    pattern_type: str
    confidence: float
    timestamp: datetime
    data_points: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, critical


@dataclass
class MonitoringConfig:
    """Configuration for real-time pattern monitoring"""
    window_size: int = 100
    update_interval: float = 1.0  # seconds
    pattern_threshold: float = 0.7
    max_history: int = 1000
    enable_anomaly_detection: bool = True
    enable_trend_detection: bool = True
    enable_seasonal_detection: bool = True


class RealTimePatternMonitor:
    """
    Real-time pattern monitoring system that continuously analyzes data streams
    for patterns, anomalies, and trends.
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        """
        Initialize the real-time pattern monitor
        
        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitoringConfig()
        self.data_buffer = deque(maxlen=self.config.max_history)
        self.pattern_history = deque(maxlen=self.config.max_history)
        self.is_monitoring = False
        self.monitoring_task = None
        self.callbacks: List[Callable[[PatternEvent], None]] = []
        
        # Pattern detection state
        self.last_update = datetime.now()
        self.pattern_counters = {
            'anomalies': 0,
            'trends': 0,
            'seasonal': 0,
            'spikes': 0
        }
        
        logger.info("RealTimePatternMonitor initialized")
    
    async def start_monitoring(self):
        """Start the real-time monitoring process"""
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Real-time pattern monitoring started")
    
    async def stop_monitoring(self):
        """Stop the real-time monitoring process"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Real-time pattern monitoring stopped")
    
    def add_data_point(self, value: float, timestamp: Optional[datetime] = None):
        """
        Add a new data point to the monitoring buffer
        
        Args:
            value: The data value
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.data_buffer.append((timestamp, value))
        
        # Trigger immediate analysis if buffer is full enough
        if len(self.data_buffer) >= self.config.window_size:
            self._analyze_patterns()
    
    def add_callback(self, callback: Callable[[PatternEvent], None]):
        """
        Add a callback function to be called when patterns are detected
        
        Args:
            callback: Function to call with PatternEvent
        """
        self.callbacks.append(callback)
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Analyze patterns at regular intervals
                if len(self.data_buffer) >= self.config.window_size:
                    self._analyze_patterns()
                
                await asyncio.sleep(self.config.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(1.0)
    
    def _analyze_patterns(self):
        """Analyze current data buffer for patterns"""
        if len(self.data_buffer) < self.config.window_size:
            return
        
        # Extract recent data
        recent_data = list(self.data_buffer)[-self.config.window_size:]
        values = [point[1] for point in recent_data]
        timestamps = [point[0] for point in recent_data]
        
        # Detect different types of patterns
        if self.config.enable_anomaly_detection:
            self._detect_anomalies(values, timestamps)
        
        if self.config.enable_trend_detection:
            self._detect_trends(values, timestamps)
        
        if self.config.enable_seasonal_detection:
            self._detect_seasonal_patterns(values, timestamps)
        
        self._detect_spikes(values, timestamps)
    
    def _detect_anomalies(self, values: List[float], timestamps: List[datetime]):
        """Detect anomalies in the data"""
        if len(values) < 10:
            return
        
        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        # Detect outliers using z-score
        z_scores = np.abs((values_array - mean) / std)
        anomaly_indices = np.where(z_scores > 2.5)[0]
        
        for idx in anomaly_indices:
            event = PatternEvent(
                pattern_id=f"anomaly_{self.pattern_counters['anomalies']}",
                pattern_type="anomaly",
                confidence=min(z_scores[idx] / 3.0, 1.0),
                timestamp=timestamps[idx],
                data_points=[values[idx]],
                metadata={
                    'z_score': float(z_scores[idx]),
                    'mean': float(mean),
                    'std': float(std)
                },
                severity="warning" if z_scores[idx] > 3.0 else "info"
            )
            
            self._trigger_pattern_event(event)
            self.pattern_counters['anomalies'] += 1
    
    def _detect_trends(self, values: List[float], timestamps: List[datetime]):
        """Detect trends in the data"""
        if len(values) < 20:
            return
        
        values_array = np.array(values)
        x = np.arange(len(values_array))
        
        # Linear regression
        slope, intercept = np.polyfit(x, values_array, 1)
        r_squared = self._calculate_r_squared(values_array, slope, intercept)
        
        # Determine trend strength
        trend_strength = abs(slope) * r_squared
        
        if trend_strength > 0.1:  # Threshold for trend detection
            trend_direction = "increasing" if slope > 0 else "decreasing"
            
            event = PatternEvent(
                pattern_id=f"trend_{self.pattern_counters['trends']}",
                pattern_type="trend",
                confidence=min(trend_strength, 1.0),
                timestamp=timestamps[-1],
                data_points=values,
                metadata={
                    'slope': float(slope),
                    'r_squared': float(r_squared),
                    'direction': trend_direction,
                    'strength': float(trend_strength)
                },
                severity="info"
            )
            
            self._trigger_pattern_event(event)
            self.pattern_counters['trends'] += 1
    
    def _detect_seasonal_patterns(self, values: List[float], timestamps: List[datetime]):
        """Detect seasonal patterns in the data"""
        if len(values) < 50:
            return
        
        values_array = np.array(values)
        
        # Simple seasonal detection using autocorrelation
        autocorr = self._calculate_autocorrelation(values_array)
        
        # Find peaks in autocorrelation (potential seasonal periods)
        peaks = self._find_peaks(autocorr)
        
        if len(peaks) > 0:
            # Use the strongest seasonal pattern
            best_peak = max(peaks, key=lambda x: autocorr[x])
            seasonal_period = best_peak + 1
            
            if seasonal_period > 1 and autocorr[best_peak] > 0.3:
                event = PatternEvent(
                    pattern_id=f"seasonal_{self.pattern_counters['seasonal']}",
                    pattern_type="seasonal",
                    confidence=float(autocorr[best_peak]),
                    timestamp=timestamps[-1],
                    data_points=values,
                    metadata={
                        'period': int(seasonal_period),
                        'autocorrelation': float(autocorr[best_peak])
                    },
                    severity="info"
                )
                
                self._trigger_pattern_event(event)
                self.pattern_counters['seasonal'] += 1
    
    def _detect_spikes(self, values: List[float], timestamps: List[datetime]):
        """Detect sudden spikes in the data"""
        if len(values) < 5:
            return
        
        values_array = np.array(values)
        
        # Calculate rate of change
        rate_of_change = np.diff(values_array)
        
        # Detect sudden changes
        threshold = np.std(rate_of_change) * 2.0
        
        spike_indices = np.where(np.abs(rate_of_change) > threshold)[0]
        
        for idx in spike_indices:
            if idx + 1 < len(values):
                event = PatternEvent(
                    pattern_id=f"spike_{self.pattern_counters['spikes']}",
                    pattern_type="spike",
                    confidence=min(np.abs(rate_of_change[idx]) / (threshold * 2), 1.0),
                    timestamp=timestamps[idx + 1],
                    data_points=[values[idx], values[idx + 1]],
                    metadata={
                        'rate_of_change': float(rate_of_change[idx]),
                        'threshold': float(threshold)
                    },
                    severity="warning"
                )
                
                self._trigger_pattern_event(event)
                self.pattern_counters['spikes'] += 1
    
    def _calculate_r_squared(self, y: np.ndarray, slope: float, intercept: float) -> float:
        """Calculate R-squared value for trend detection"""
        y_pred = slope * np.arange(len(y)) + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def _calculate_autocorrelation(self, data: np.ndarray, max_lag: int = 20) -> np.ndarray:
        """Calculate autocorrelation for seasonal pattern detection"""
        n = len(data)
        max_lag = min(max_lag, n // 2)
        
        autocorr = np.zeros(max_lag)
        for lag in range(1, max_lag + 1):
            if lag < n:
                autocorr[lag - 1] = np.corrcoef(data[:-lag], data[lag:])[0, 1]
        
        return autocorr
    
    def _find_peaks(self, data: np.ndarray, threshold: float = 0.1) -> List[int]:
        """Find peaks in the data above threshold"""
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > threshold and data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append(i)
        return peaks
    
    def _trigger_pattern_event(self, event: PatternEvent):
        """Trigger pattern event callbacks"""
        self.pattern_history.append(event)
        
        # Call all registered callbacks
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in pattern event callback: {str(e)}")
        
        logger.info(f"Pattern detected: {event.pattern_type} (confidence: {event.confidence:.2f})")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            'total_patterns': sum(self.pattern_counters.values()),
            'pattern_counts': dict(self.pattern_counters),
            'buffer_size': len(self.data_buffer),
            'is_monitoring': self.is_monitoring,
            'last_update': self.last_update.isoformat() if self.last_update else None
        }
    
    def get_recent_patterns(self, limit: int = 10) -> List[PatternEvent]:
        """Get recent pattern events"""
        return list(self.pattern_history)[-limit:]
    
    def clear_history(self):
        """Clear pattern history"""
        self.pattern_history.clear()
        self.pattern_counters = {k: 0 for k in self.pattern_counters}
