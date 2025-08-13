"""
Error Predictor

Uses machine learning and statistical analysis to predict potential system errors.
Provides proactive error detection and early warning capabilities.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import statistics
import json
import random

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorType(Enum):
    """Types of errors that can be predicted"""
    SYSTEM_CRASH = "system_crash"
    MEMORY_LEAK = "memory_leak"
    DISK_FULL = "disk_full"
    NETWORK_TIMEOUT = "network_timeout"
    PROCESS_HANG = "process_hang"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PERFORMANCE_DEGRADATION = "performance_degradation"


@dataclass
class ErrorPrediction:
    """Prediction of a potential error"""
    error_type: ErrorType
    severity: ErrorSeverity
    probability: float
    predicted_time: datetime
    confidence: float
    indicators: List[str]
    description: str


@dataclass
class SystemState:
    """System state snapshot for error prediction"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_errors: int
    process_count: int
    error_count: int
    response_time: float


class ErrorPredictor:
    """Predicts potential system errors using ML and statistical analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.prediction_active = False
        self.prediction_thread: Optional[threading.Thread] = None
        self.prediction_interval = self.config.get('prediction_interval', 300)
        
        # Historical data for prediction
        self.system_history: List[SystemState] = []
        self.error_history: List[Dict[str, Any]] = []
        self.predictions: List[ErrorPrediction] = []
        
        # Prediction thresholds
        self.thresholds = self.config.get('thresholds', {
            'cpu_critical': 90.0,
            'memory_critical': 95.0,
            'disk_critical': 98.0,
            'network_error_threshold': 10,
            'response_time_threshold': 5.0
        })
        
        # ML model parameters (simplified)
        self.model_weights = {
            'cpu_weight': 0.3,
            'memory_weight': 0.25,
            'disk_weight': 0.2,
            'network_weight': 0.15,
            'process_weight': 0.1
        }
        
    def start_prediction(self):
        """Start continuous error prediction"""
        if self.prediction_active:
            logger.warning("Error prediction is already active")
            return
            
        self.prediction_active = True
        self.prediction_thread = threading.Thread(
            target=self._prediction_loop, daemon=True
        )
        self.prediction_thread.start()
        logger.info("Error prediction started")
        
    def stop_prediction(self):
        """Stop continuous error prediction"""
        self.prediction_active = False
        if self.prediction_thread:
            self.prediction_thread.join(timeout=5)
        logger.info("Error prediction stopped")
        
    def _prediction_loop(self):
        """Main prediction loop"""
        while self.prediction_active:
            try:
                self.predict_errors()
                time.sleep(self.prediction_interval)
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                time.sleep(30)
                
    def predict_errors(self) -> List[ErrorPrediction]:
        """Perform comprehensive error prediction"""
        try:
            # Collect current system state
            current_state = self._collect_system_state()
            self.system_history.append(current_state)
            
            # Keep only recent history (last 7 days)
            cutoff_time = datetime.now() - timedelta(days=7)
            self.system_history = [
                s for s in self.system_history 
                if s.timestamp > cutoff_time
            ]
            
            # Generate predictions
            predictions = []
            
            # Predict system crash
            crash_prediction = self._predict_system_crash(current_state)
            if crash_prediction:
                predictions.append(crash_prediction)
                
            # Predict memory leak
            memory_prediction = self._predict_memory_leak(current_state)
            if memory_prediction:
                predictions.append(memory_prediction)
                
            # Predict disk full
            disk_prediction = self._predict_disk_full(current_state)
            if disk_prediction:
                predictions.append(disk_prediction)
                
            # Predict network timeout
            network_prediction = self._predict_network_timeout(current_state)
            if network_prediction:
                predictions.append(network_prediction)
                
            # Predict process hang
            process_prediction = self._predict_process_hang(current_state)
            if process_prediction:
                predictions.append(process_prediction)
                
            # Update predictions list
            self.predictions = predictions
            
            # Log high-probability predictions
            for prediction in predictions:
                if prediction.probability > 0.7:
                    logger.warning(
                        f"High probability error predicted: {prediction.error_type.value} "
                        f"(probability: {prediction.probability:.2f})"
                    )
                    
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting errors: {e}")
            return []
            
    def _collect_system_state(self) -> SystemState:
        """Collect current system state for prediction"""
        try:
            import psutil
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            
            # Network errors (simplified - would need actual network monitoring)
            network_errors = self._estimate_network_errors()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Error count (simplified - would need actual error tracking)
            error_count = self._estimate_error_count()
            
            # Response time (simplified - would need actual response time monitoring)
            response_time = self._estimate_response_time()
            
            return SystemState(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_errors=network_errors,
                process_count=process_count,
                error_count=error_count,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"Error collecting system state: {e}")
            # Return default state
            return SystemState(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_errors=0,
                process_count=0,
                error_count=0,
                response_time=0.0
            )
            
    def _estimate_network_errors(self) -> int:
        """Estimate network errors (placeholder)"""
        # In a real implementation, this would check actual network error logs
        # For now, return a random small number
        return random.randint(0, 5)
        
    def _estimate_error_count(self) -> int:
        """Estimate current error count (placeholder)"""
        # In a real implementation, this would check actual error logs
        # For now, return a random small number
        return random.randint(0, 3)
        
    def _estimate_response_time(self) -> float:
        """Estimate system response time (placeholder)"""
        # In a real implementation, this would measure actual response times
        # For now, return a random value
        return random.uniform(0.1, 2.0)
        
    def _predict_system_crash(self, state: SystemState) -> Optional[ErrorPrediction]:
        """Predict system crash based on current state"""
        try:
            # Calculate crash probability based on multiple factors
            cpu_factor = state.cpu_usage / 100.0
            memory_factor = state.memory_usage / 100.0
            process_factor = min(state.process_count / 1000.0, 1.0)
            error_factor = min(state.error_count / 10.0, 1.0)
            
            # Weighted probability calculation
            probability = (
                cpu_factor * self.model_weights['cpu_weight'] +
                memory_factor * self.model_weights['memory_weight'] +
                process_factor * self.model_weights['process_weight'] +
                error_factor * 0.2
            )
            
            # Apply historical trend analysis
            if len(self.system_history) > 10:
                recent_states = self.system_history[-10:]
                trend_factor = self._calculate_trend_factor(recent_states)
                probability *= trend_factor
                
            # Determine severity
            if probability > 0.8:
                severity = ErrorSeverity.CRITICAL
            elif probability > 0.6:
                severity = ErrorSeverity.HIGH
            elif probability > 0.4:
                severity = ErrorSeverity.MEDIUM
            else:
                severity = ErrorSeverity.LOW
                
            # Only return prediction if probability is significant
            if probability > 0.3:
                indicators = []
                if state.cpu_usage > 80:
                    indicators.append("High CPU usage")
                if state.memory_usage > 90:
                    indicators.append("High memory usage")
                if state.process_count > 800:
                    indicators.append("High process count")
                if state.error_count > 5:
                    indicators.append("High error rate")
                    
                return ErrorPrediction(
                    error_type=ErrorType.SYSTEM_CRASH,
                    severity=severity,
                    probability=probability,
                    predicted_time=datetime.now() + timedelta(hours=1),
                    confidence=min(probability * 1.2, 1.0),
                    indicators=indicators,
                    description="System crash predicted based on resource usage patterns"
                )
                
        except Exception as e:
            logger.error(f"Error predicting system crash: {e}")
            
        return None
        
    def _predict_memory_leak(self, state: SystemState) -> Optional[ErrorPrediction]:
        """Predict memory leak based on current state"""
        try:
            # Check for memory leak patterns
            if len(self.system_history) < 5:
                return None
                
            recent_states = self.system_history[-5:]
            memory_trend = [s.memory_usage for s in recent_states]
            
            # Check if memory usage is consistently increasing
            if len(memory_trend) >= 3:
                trend = self._calculate_trend(memory_trend)
                
                if trend > 0.5:  # Memory usage increasing significantly
                    probability = min(trend * 0.8, 0.9)
                    
                    if probability > 0.4:
                        severity = ErrorSeverity.HIGH if probability > 0.7 else ErrorSeverity.MEDIUM
                        
                        return ErrorPrediction(
                            error_type=ErrorType.MEMORY_LEAK,
                            severity=severity,
                            probability=probability,
                            predicted_time=datetime.now() + timedelta(hours=2),
                            confidence=probability * 0.9,
                            indicators=["Consistently increasing memory usage"],
                            description="Memory leak detected based on usage trend"
                        )
                        
        except Exception as e:
            logger.error(f"Error predicting memory leak: {e}")
            
        return None
        
    def _predict_disk_full(self, state: SystemState) -> Optional[ErrorPrediction]:
        """Predict disk full error based on current state"""
        try:
            # Calculate probability based on disk usage and growth rate
            current_usage = state.disk_usage / 100.0
            
            if len(self.system_history) >= 3:
                recent_states = self.system_history[-3:]
                disk_trend = [s.disk_usage for s in recent_states]
                growth_rate = self._calculate_trend(disk_trend)
                
                # Predict when disk will be full
                if growth_rate > 0 and current_usage > 0.7:
                    remaining_space = 1.0 - current_usage
                    hours_to_full = remaining_space / (growth_rate / 100.0)
                    
                    if hours_to_full < 24:  # Within 24 hours
                        probability = 1.0 - (hours_to_full / 24.0)
                        
                        if probability > 0.5:
                            severity = ErrorSeverity.CRITICAL if probability > 0.8 else ErrorSeverity.HIGH
                            
                            return ErrorPrediction(
                                error_type=ErrorType.DISK_FULL,
                                severity=severity,
                                probability=probability,
                                predicted_time=datetime.now() + timedelta(hours=hours_to_full),
                                confidence=probability * 0.95,
                                indicators=[f"Disk usage at {state.disk_usage:.1f}%"],
                                description=f"Disk predicted to be full in {hours_to_full:.1f} hours"
                            )
                            
        except Exception as e:
            logger.error(f"Error predicting disk full: {e}")
            
        return None
        
    def _predict_network_timeout(self, state: SystemState) -> Optional[ErrorPrediction]:
        """Predict network timeout based on current state"""
        try:
            # Calculate probability based on network errors and response time
            error_factor = min(state.network_errors / 10.0, 1.0)
            response_factor = min(state.response_time / 5.0, 1.0)
            
            probability = (error_factor * 0.6 + response_factor * 0.4)
            
            if probability > 0.4:
                severity = ErrorSeverity.MEDIUM if probability > 0.6 else ErrorSeverity.LOW
                
                return ErrorPrediction(
                    error_type=ErrorType.NETWORK_TIMEOUT,
                    severity=severity,
                    probability=probability,
                    predicted_time=datetime.now() + timedelta(minutes=30),
                    confidence=probability * 0.8,
                    indicators=[
                        f"Network errors: {state.network_errors}",
                        f"Response time: {state.response_time:.2f}s"
                    ],
                    description="Network timeout predicted based on error rate and response time"
                )
                
        except Exception as e:
            logger.error(f"Error predicting network timeout: {e}")
            
        return None
        
    def _predict_process_hang(self, state: SystemState) -> Optional[ErrorPrediction]:
        """Predict process hang based on current state"""
        try:
            # Calculate probability based on process count and response time
            process_factor = min(state.process_count / 1000.0, 1.0)
            response_factor = min(state.response_time / 5.0, 1.0)
            
            probability = (process_factor * 0.5 + response_factor * 0.5)
            
            if probability > 0.5:
                severity = ErrorSeverity.HIGH if probability > 0.7 else ErrorSeverity.MEDIUM
                
                return ErrorPrediction(
                    error_type=ErrorType.PROCESS_HANG,
                    severity=severity,
                    probability=probability,
                    predicted_time=datetime.now() + timedelta(minutes=15),
                    confidence=probability * 0.85,
                    indicators=[
                        f"Process count: {state.process_count}",
                        f"Response time: {state.response_time:.2f}s"
                    ],
                    description="Process hang predicted based on high process count and slow response"
                )
                
        except Exception as e:
            logger.error(f"Error predicting process hang: {e}")
            
        return None
        
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend of values (positive = increasing, negative = decreasing)"""
        if len(values) < 2:
            return 0.0
            
        try:
            # Simple linear trend calculation
            x_values = list(range(len(values)))
            y_values = values
            
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
            
        except Exception:
            return 0.0
            
    def _calculate_trend_factor(self, states: List[SystemState]) -> float:
        """Calculate trend factor for prediction adjustment"""
        if len(states) < 2:
            return 1.0
            
        try:
            # Calculate trend in resource usage
            cpu_trend = self._calculate_trend([s.cpu_usage for s in states])
            memory_trend = self._calculate_trend([s.memory_usage for s in states])
            
            # Combine trends
            combined_trend = (cpu_trend + memory_trend) / 2.0
            
            # Convert to factor (1.0 = no change, >1.0 = worsening, <1.0 = improving)
            if combined_trend > 0:
                return 1.0 + min(combined_trend / 100.0, 0.5)
            else:
                return max(1.0 + combined_trend / 100.0, 0.5)
                
        except Exception:
            return 1.0
            
    def get_predictions(self) -> List[Dict[str, Any]]:
        """Get current error predictions"""
        return [asdict(prediction) for prediction in self.predictions]
        
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of current predictions"""
        if not self.predictions:
            return {"message": "No error predictions available"}
            
        critical_predictions = [p for p in self.predictions if p.severity == ErrorSeverity.CRITICAL]
        high_predictions = [p for p in self.predictions if p.severity == ErrorSeverity.HIGH]
        
        return {
            'total_predictions': len(self.predictions),
            'critical_predictions': len(critical_predictions),
            'high_predictions': len(high_predictions),
            'average_probability': statistics.mean([p.probability for p in self.predictions]),
            'predictions': self.get_predictions()
        }
        
    def record_actual_error(self, error_type: str, timestamp: datetime):
        """Record an actual error for model improvement"""
        self.error_history.append({
            'error_type': error_type,
            'timestamp': timestamp,
            'predicted': any(
                p.error_type.value == error_type and 
                abs((p.predicted_time - timestamp).total_seconds()) < 3600
                for p in self.predictions
            )
        })
        
        # Keep only recent error history
        cutoff_time = datetime.now() - timedelta(days=30)
        self.error_history = [
            e for e in self.error_history 
            if e['timestamp'] > cutoff_time
        ]
        
    def get_prediction_accuracy(self) -> Dict[str, Any]:
        """Get prediction accuracy statistics"""
        if not self.error_history:
            return {"message": "No error history available"}
            
        total_errors = len(self.error_history)
        predicted_errors = sum(1 for e in self.error_history if e['predicted'])
        
        accuracy = predicted_errors / total_errors if total_errors > 0 else 0.0
        
        return {
            'total_errors': total_errors,
            'predicted_errors': predicted_errors,
            'accuracy': accuracy,
            'prediction_rate': accuracy * 100
        }
