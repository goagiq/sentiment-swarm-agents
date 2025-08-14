"""
Automated Decision Monitoring System

This module provides automated monitoring and alerting for decision outcomes,
tracking decision performance, and generating insights for improvement.
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import statistics

from loguru import logger

from src.core.error_handler import with_error_handling


@dataclass
class DecisionRecord:
    """Represents a decision record for monitoring."""
    decision_id: str
    decision_type: str
    scenario_id: str
    parameters: Dict[str, Any]
    predicted_outcome: Dict[str, Any]
    actual_outcome: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    risk_score: float = 0.0
    impact_score: float = 0.0
    status: str = "pending"  # pending, in_progress, completed, failed
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completion_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringAlert:
    """Represents a monitoring alert."""
    alert_id: str
    alert_type: str  # performance, accuracy, risk, threshold
    severity: str  # low, medium, high, critical
    message: str
    decision_id: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Represents a performance metric for decision monitoring."""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    decision_id: str
    category: str  # accuracy, speed, efficiency, cost
    metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionMonitor:
    """
    Automated decision monitoring system that tracks decision outcomes
    and generates alerts for performance issues.
    """
    
    def __init__(self):
        self.decision_records: Dict[str, DecisionRecord] = {}
        self.alerts: List[MonitoringAlert] = []
        self.performance_metrics: List[PerformanceMetric] = []
        
        # Monitoring thresholds
        self.thresholds = {
            "accuracy": 0.8,
            "confidence": 0.7,
            "response_time": 5.0,  # seconds
            "risk_score": 0.3,
            "failure_rate": 0.1
        }
        
        # Alert handlers
        self.alert_handlers: Dict[str, Callable] = {}
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = {}
        
        logger.info("DecisionMonitor initialized")
    
    @with_error_handling("decision_recording")
    def record_decision(
        self,
        decision_type: str,
        input_data: Dict[str, Any],
        decision_output: Dict[str, Any],
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a decision for monitoring.
        
        Args:
            decision_type: Type of decision
            input_data: Input data for the decision
            decision_output: Output/result of the decision
            user_id: Optional user ID
            metadata: Optional metadata
            
        Returns:
            Decision ID
        """
        try:
            decision_id = f"decision_{int(datetime.now().timestamp())}_{hash(str(input_data)) % 10000}"
            
            decision_record = DecisionRecord(
                decision_id=decision_id,
                decision_type=decision_type,
                scenario_id=f"scenario_{decision_id}",
                parameters=input_data,
                predicted_outcome=decision_output,
                confidence_score=decision_output.get("confidence", 0.0),
                risk_score=decision_output.get("risk", 0.0),
                impact_score=decision_output.get("impact", 0.0),
                status="completed",
                completion_time=datetime.now(),
                metadata=metadata or {}
            )
            
            if user_id:
                decision_record.metadata["user_id"] = user_id
            
            self.decision_records[decision_id] = decision_record
            
            # Record performance metrics
            self._record_performance_metrics(decision_record)
            
            logger.info(f"Recorded decision {decision_id}: {decision_type}")
            return decision_id
            
        except Exception as e:
            logger.error(f"Error recording decision: {e}")
            raise

    @with_error_handling("decision_outcome_recording")
    def record_decision_outcome(
        self,
        decision_id: str,
        actual_outcome: Dict[str, Any],
        success: bool = True,
        feedback: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record the actual outcome of a decision.
        
        Args:
            decision_id: ID of the decision
            actual_outcome: Actual outcome data
            success: Whether the decision was successful
            feedback: Optional feedback data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if decision_id not in self.decision_records:
                logger.warning(f"Decision {decision_id} not found for outcome recording")
                return False
            
            decision_record = self.decision_records[decision_id]
            decision_record.actual_outcome = actual_outcome
            decision_record.updated_at = datetime.now()
            
            if feedback:
                decision_record.metadata["feedback"] = feedback
            
            # Update status based on success
            if success:
                decision_record.status = "completed"
            else:
                decision_record.status = "failed"
            
            # Record performance metrics for the outcome
            self._record_outcome_metrics(decision_record, actual_outcome, success)
            
            logger.info(f"Recorded outcome for decision {decision_id}: success={success}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording decision outcome: {e}")
            return False

    def _record_performance_metrics(self, decision_record: DecisionRecord):
        """Record performance metrics for a decision."""
        try:
            # Record response time (simulated)
            response_time = 0.5  # Simulated response time
            self.performance_metrics.append(PerformanceMetric(
                metric_name="response_time",
                value=response_time,
                unit="seconds",
                timestamp=decision_record.created_at,
                decision_id=decision_record.decision_id,
                category="speed"
            ))
            
            # Record confidence score
            self.performance_metrics.append(PerformanceMetric(
                metric_name="confidence_score",
                value=decision_record.confidence_score,
                unit="percentage",
                timestamp=decision_record.created_at,
                decision_id=decision_record.decision_id,
                category="accuracy"
            ))
            
        except Exception as e:
            logger.error(f"Error recording performance metrics: {e}")

    def _record_outcome_metrics(self, decision_record: DecisionRecord, actual_outcome: Dict[str, Any], success: bool):
        """Record metrics based on decision outcome."""
        try:
            # Record success/failure metric
            success_value = 1.0 if success else 0.0
            self.performance_metrics.append(PerformanceMetric(
                metric_name="success_rate",
                value=success_value,
                unit="boolean",
                timestamp=decision_record.updated_at,
                decision_id=decision_record.decision_id,
                category="accuracy"
            ))
            
        except Exception as e:
            logger.error(f"Error recording outcome metrics: {e}")

    @with_error_handling("decision_analysis")
    def analyze_decisions(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze decisions for the specified time period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Analysis results
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_decisions = [
                d for d in self.decision_records.values()
                if d.created_at > cutoff_time
            ]
            
            if not recent_decisions:
                return {"error": f"No decisions found in the last {hours} hours"}
            
            # Calculate basic statistics
            total_decisions = len(recent_decisions)
            successful_decisions = len([d for d in recent_decisions if d.status == "completed"])
            success_rate = successful_decisions / total_decisions if total_decisions > 0 else 0
            
            # Calculate average confidence
            confidence_scores = [d.confidence_score for d in recent_decisions]
            avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
            
            # Calculate average risk
            risk_scores = [d.risk_score for d in recent_decisions]
            avg_risk = statistics.mean(risk_scores) if risk_scores else 0
            
            analysis = {
                "period_hours": hours,
                "total_decisions": total_decisions,
                "successful_decisions": successful_decisions,
                "success_rate": success_rate,
                "average_confidence": avg_confidence,
                "average_risk": avg_risk,
                "decision_types": {},
                "alerts": len([a for a in self.alerts if a.timestamp > cutoff_time])
            }
            
            # Group by decision type
            for decision in recent_decisions:
                decision_type = decision.decision_type
                if decision_type not in analysis["decision_types"]:
                    analysis["decision_types"][decision_type] = {
                        "count": 0,
                        "success_count": 0,
                        "avg_confidence": 0.0
                    }
                
                analysis["decision_types"][decision_type]["count"] += 1
                if decision.status == "completed":
                    analysis["decision_types"][decision_type]["success_count"] += 1
                
                # Update average confidence
                current_avg = analysis["decision_types"][decision_type]["avg_confidence"]
                count = analysis["decision_types"][decision_type]["count"]
                analysis["decision_types"][decision_type]["avg_confidence"] = (
                    (current_avg * (count - 1) + decision.confidence_score) / count
                )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing decisions: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    @property
    def decision_history(self) -> List[DecisionRecord]:
        """Get all decision history."""
        return list(self.decision_records.values())

    @property
    def monitoring_active(self) -> bool:
        """Check if monitoring is active."""
        return True  # Always active for now
    
    # Removed duplicate async record_decision method - using sync version above
    
    @with_error_handling("outcome_update")
    async def update_decision_outcome(
        self,
        decision_id: str,
        actual_outcome: Dict[str, Any],
        status: str = "completed"
    ) -> bool:
        """
        Update a decision with its actual outcome.
        
        Args:
            decision_id: Decision ID to update
            actual_outcome: Actual outcome data
            status: New status
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if decision_id not in self.decision_records:
                logger.warning(f"Decision {decision_id} not found for outcome update")
                return False
            
            decision_record = self.decision_records[decision_id]
            decision_record.actual_outcome = actual_outcome
            decision_record.status = status
            decision_record.completion_time = datetime.now()
            decision_record.updated_at = datetime.now()
            
            # Calculate performance metrics
            await self._calculate_performance_metrics(decision_record)
            
            # Check for performance alerts
            await self._check_performance_alerts(decision_record)
            
            logger.info(f"Updated outcome for decision {decision_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating outcome for decision {decision_id}: {e}")
            return False
    
    async def _calculate_performance_metrics(self, decision_record: DecisionRecord):
        """Calculate performance metrics for a decision."""
        try:
            if not decision_record.actual_outcome:
                return
            
            # Calculate accuracy
            predicted = decision_record.predicted_outcome
            actual = decision_record.actual_outcome
            
            # Simple accuracy calculation (can be enhanced)
            accuracy = 0.0
            if "success" in predicted and "success" in actual:
                accuracy = 1.0 if predicted["success"] == actual["success"] else 0.0
            
            # Calculate response time
            response_time = (decision_record.completion_time - decision_record.created_at).total_seconds()
            
            # Record metrics
            accuracy_metric = PerformanceMetric(
                metric_name="accuracy",
                value=accuracy,
                unit="percentage",
                timestamp=datetime.now(),
                decision_id=decision_record.decision_id,
                category="accuracy"
            )
            
            response_time_metric = PerformanceMetric(
                metric_name="response_time",
                value=response_time,
                unit="seconds",
                timestamp=datetime.now(),
                decision_id=decision_record.decision_id,
                category="speed"
            )
            
            self.performance_metrics.extend([accuracy_metric, response_time_metric])
            
            # Update performance history
            if "accuracy" not in self.performance_history:
                self.performance_history["accuracy"] = []
            if "response_time" not in self.performance_history:
                self.performance_history["response_time"] = []
            
            self.performance_history["accuracy"].append(accuracy)
            self.performance_history["response_time"].append(response_time)
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
    
    async def _check_immediate_alerts(self, decision_record: DecisionRecord):
        """Check for immediate alerts when a decision is recorded."""
        try:
            # Check confidence threshold
            if decision_record.confidence_score < self.thresholds["confidence"]:
                alert = MonitoringAlert(
                    alert_id=f"conf_{decision_record.decision_id}",
                    alert_type="confidence",
                    severity="medium",
                    message=f"Low confidence decision: {decision_record.confidence_score:.2f}",
                    decision_id=decision_record.decision_id,
                    metric_name="confidence",
                    current_value=decision_record.confidence_score,
                    threshold_value=self.thresholds["confidence"]
                )
                self.alerts.append(alert)
            
            # Check risk threshold
            if decision_record.risk_score > self.thresholds["risk_score"]:
                alert = MonitoringAlert(
                    alert_id=f"risk_{decision_record.decision_id}",
                    alert_type="risk",
                    severity="high",
                    message=f"High risk decision: {decision_record.risk_score:.2f}",
                    decision_id=decision_record.decision_id,
                    metric_name="risk_score",
                    current_value=decision_record.risk_score,
                    threshold_value=self.thresholds["risk_score"]
                )
                self.alerts.append(alert)
                
        except Exception as e:
            logger.error(f"Error checking immediate alerts: {e}")
    
    async def _check_performance_alerts(self, decision_record: DecisionRecord):
        """Check for performance alerts after outcome update."""
        try:
            # Get recent performance metrics
            recent_metrics = [m for m in self.performance_metrics 
                            if m.decision_id == decision_record.decision_id]
            
            for metric in recent_metrics:
                if metric.metric_name == "accuracy":
                    if metric.value < self.thresholds["accuracy"]:
                        alert = MonitoringAlert(
                            alert_id=f"acc_{decision_record.decision_id}",
                            alert_type="accuracy",
                            severity="high",
                            message=f"Low accuracy: {metric.value:.2f}",
                            decision_id=decision_record.decision_id,
                            metric_name="accuracy",
                            current_value=metric.value,
                            threshold_value=self.thresholds["accuracy"]
                        )
                        self.alerts.append(alert)
                
                elif metric.metric_name == "response_time":
                    if metric.value > self.thresholds["response_time"]:
                        alert = MonitoringAlert(
                            alert_id=f"time_{decision_record.decision_id}",
                            alert_type="performance",
                            severity="medium",
                            message=f"Slow response time: {metric.value:.2f}s",
                            decision_id=decision_record.decision_id,
                            metric_name="response_time",
                            current_value=metric.value,
                            threshold_value=self.thresholds["response_time"]
                        )
                        self.alerts.append(alert)
                        
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    async def get_performance_summary(
        self,
        time_window: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """
        Get performance summary for the specified time window.
        
        Args:
            time_window: Time window for analysis
            
        Returns:
            Performance summary dictionary
        """
        try:
            cutoff_time = datetime.now() - time_window
            
            # Filter metrics by time window
            recent_metrics = [
                m for m in self.performance_metrics 
                if m.timestamp >= cutoff_time
            ]
            
            # Group metrics by category
            metrics_by_category = {}
            for metric in recent_metrics:
                if metric.category not in metrics_by_category:
                    metrics_by_category[metric.category] = []
                metrics_by_category[metric.category].append(metric.value)
            
            # Calculate summary statistics
            summary = {
                "time_window": str(time_window),
                "total_decisions": len(recent_metrics),
                "categories": {}
            }
            
            for category, values in metrics_by_category.items():
                if values:
                    summary["categories"][category] = {
                        "count": len(values),
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                        "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {}
    
    async def get_active_alerts(
        self,
        severity: Optional[str] = None
    ) -> List[MonitoringAlert]:
        """
        Get active (unresolved) alerts.
        
        Args:
            severity: Filter by severity level
            
        Returns:
            List of active alerts
        """
        try:
            active_alerts = [alert for alert in self.alerts if not alert.resolved]
            
            if severity:
                active_alerts = [alert for alert in active_alerts if alert.severity == severity]
            
            return active_alerts
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID to acknowledge
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    alert.updated_at = datetime.now()
                    logger.info(f"Acknowledged alert {alert_id}")
                    return True
            
            logger.warning(f"Alert {alert_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert ID to resolve
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.updated_at = datetime.now()
                    logger.info(f"Resolved alert {alert_id}")
                    return True
            
            logger.warning(f"Alert {alert_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    async def set_threshold(
        self,
        metric_name: str,
        threshold_value: float
    ) -> bool:
        """
        Set a monitoring threshold.
        
        Args:
            metric_name: Name of the metric
            threshold_value: Threshold value
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.thresholds[metric_name] = threshold_value
            logger.info(f"Set threshold for {metric_name}: {threshold_value}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting threshold for {metric_name}: {e}")
            return False
    
    async def register_alert_handler(
        self,
        alert_type: str,
        handler: Callable
    ) -> bool:
        """
        Register an alert handler.
        
        Args:
            alert_type: Type of alert to handle
            handler: Handler function
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.alert_handlers[alert_type] = handler
            logger.info(f"Registered alert handler for {alert_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering alert handler for {alert_type}: {e}")
            return False
    
    async def export_monitoring_data(
        self,
        format: str = "json"
    ) -> str:
        """
        Export monitoring data.
        
        Args:
            format: Export format (json, csv)
            
        Returns:
            Exported data as string
        """
        try:
            if format.lower() == "json":
                data = {
                    "decision_records": {
                        k: v.__dict__ for k, v in self.decision_records.items()
                    },
                    "alerts": [alert.__dict__ for alert in self.alerts],
                    "performance_metrics": [metric.__dict__ for metric in self.performance_metrics],
                    "thresholds": self.thresholds,
                    "export_timestamp": datetime.now().isoformat()
                }
                return json.dumps(data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting monitoring data: {e}")
            return ""


# Global decision monitor instance
decision_monitor = DecisionMonitor()


async def get_decision_monitor() -> DecisionMonitor:
    """Get the global decision monitor instance."""
    return decision_monitor
