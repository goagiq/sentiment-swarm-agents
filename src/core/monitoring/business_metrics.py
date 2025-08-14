"""
Business Metrics Monitoring System

This module provides comprehensive business metrics monitoring including:
- Decision accuracy tracking
- User engagement monitoring
- Feature usage analytics
- System performance metrics
- Business intelligence dashboards
"""

import time
import statistics
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque

from loguru import logger

from src.core.error_handler import with_error_handling


@dataclass
class DecisionAccuracyMetric:
    """Represents a decision accuracy metric."""
    decision_id: str
    decision_type: str
    predicted_outcome: Dict[str, Any]
    actual_outcome: Dict[str, Any]
    accuracy_score: float
    confidence_score: float
    timestamp: datetime
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserEngagementMetric:
    """Represents a user engagement metric."""
    user_id: str
    session_id: str
    action_type: str
    duration: float
    timestamp: datetime
    page_url: Optional[str] = None
    feature_used: Optional[str] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureUsageMetric:
    """Represents a feature usage metric."""
    feature_name: str
    user_id: str
    usage_count: int
    timestamp: datetime
    session_id: Optional[str] = None
    success_rate: float = 1.0
    average_duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemPerformanceMetric:
    """Represents a system performance metric."""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    category: str  # response_time, throughput, error_rate, availability
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BusinessAlert:
    """Represents a business alert."""
    alert_id: str
    alert_type: str  # decision_accuracy, user_engagement, feature_usage, performance
    severity: str  # low, medium, high, critical
    message: str
    timestamp: datetime
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BusinessMetricsMonitor:
    """
    Comprehensive business metrics monitoring system.
    """
    
    def __init__(self):
        self.decision_accuracy_metrics: List[DecisionAccuracyMetric] = []
        self.user_engagement_metrics: List[UserEngagementMetric] = []
        self.feature_usage_metrics: List[FeatureUsageMetric] = []
        self.system_performance_metrics: List[SystemPerformanceMetric] = []
        self.business_alerts: List[BusinessAlert] = []
        
        # Monitoring configuration
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = 60  # seconds
        
        # Business thresholds
        self.business_thresholds = {
            "decision_accuracy": 0.85,  # 85% accuracy
            "user_engagement_rate": 0.70,  # 70% engagement
            "feature_adoption_rate": 0.50,  # 50% adoption
            "response_time": 2.0,  # 2 seconds
            "error_rate": 0.05,  # 5% error rate
            "user_satisfaction": 4.0  # 4.0/5.0 rating
        }
        
        # Feature tracking
        self.features = {
            "sentiment_analysis": {"enabled": True, "tracking": True},
            "decision_support": {"enabled": True, "tracking": True},
            "knowledge_graph": {"enabled": True, "tracking": True},
            "report_generation": {"enabled": True, "tracking": True},
            "api_access": {"enabled": True, "tracking": True}
        }
    
    @with_error_handling("start_monitoring")
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.monitoring_active:
            logger.warning("Business metrics monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Business metrics monitoring started")
    
    @with_error_handling("stop_monitoring")
    def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self.monitoring_active:
            logger.warning("Business metrics monitoring is not active")
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Business metrics monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._analyze_decision_accuracy()
                self._analyze_user_engagement()
                self._analyze_feature_usage()
                self._analyze_system_performance()
                self._check_business_alerts()
                self._cleanup_old_data()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in business monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    @with_error_handling("record_decision_accuracy")
    def record_decision_accuracy(self, decision_id: str, decision_type: str,
                               predicted_outcome: Dict[str, Any],
                               actual_outcome: Dict[str, Any],
                               accuracy_score: float, confidence_score: float,
                               user_id: Optional[str] = None,
                               metadata: Optional[Dict[str, Any]] = None):
        """Record a decision accuracy metric."""
        metric = DecisionAccuracyMetric(
            decision_id=decision_id,
            decision_type=decision_type,
            predicted_outcome=predicted_outcome,
            actual_outcome=actual_outcome,
            accuracy_score=accuracy_score,
            confidence_score=confidence_score,
            timestamp=datetime.now(),
            user_id=user_id,
            metadata=metadata or {}
        )
        
        self.decision_accuracy_metrics.append(metric)
        logger.debug(f"Decision accuracy recorded: {decision_id} - {accuracy_score}")
    
    @with_error_handling("record_user_engagement")
    def record_user_engagement(self, user_id: str, session_id: str,
                             action_type: str, duration: float,
                             page_url: Optional[str] = None,
                             feature_used: Optional[str] = None,
                             success: bool = True,
                             metadata: Optional[Dict[str, Any]] = None):
        """Record a user engagement metric."""
        metric = UserEngagementMetric(
            user_id=user_id,
            session_id=session_id,
            action_type=action_type,
            duration=duration,
            timestamp=datetime.now(),
            page_url=page_url,
            feature_used=feature_used,
            success=success,
            metadata=metadata or {}
        )
        
        self.user_engagement_metrics.append(metric)
        logger.debug(f"User engagement recorded: {user_id} - {action_type}")
    
    @with_error_handling("record_feature_usage")
    def record_feature_usage(self, feature_name: str, user_id: str,
                           usage_count: int = 1,
                           session_id: Optional[str] = None,
                           success_rate: float = 1.0,
                           average_duration: Optional[float] = None,
                           metadata: Optional[Dict[str, Any]] = None):
        """Record a feature usage metric."""
        metric = FeatureUsageMetric(
            feature_name=feature_name,
            user_id=user_id,
            usage_count=usage_count,
            timestamp=datetime.now(),
            session_id=session_id,
            success_rate=success_rate,
            average_duration=average_duration,
            metadata=metadata or {}
        )
        
        self.feature_usage_metrics.append(metric)
        logger.debug(f"Feature usage recorded: {feature_name} - {user_id}")
    
    @with_error_handling("record_system_performance")
    def record_system_performance(self, metric_name: str, value: float,
                                unit: str, category: str,
                                user_id: Optional[str] = None,
                                metadata: Optional[Dict[str, Any]] = None):
        """Record a system performance metric."""
        metric = SystemPerformanceMetric(
            metric_name=metric_name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            category=category,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        self.system_performance_metrics.append(metric)
        logger.debug(f"System performance recorded: {metric_name} - {value}")
    
    @with_error_handling("analyze_decision_accuracy")
    def _analyze_decision_accuracy(self):
        """Analyze decision accuracy trends."""
        if not self.decision_accuracy_metrics:
            return
        
        # Get recent decisions (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_decisions = [
            d for d in self.decision_accuracy_metrics
            if d.timestamp > recent_cutoff
        ]
        
        if not recent_decisions:
            return
        
        # Calculate average accuracy
        avg_accuracy = statistics.mean([d.accuracy_score for d in recent_decisions])
        threshold = self.business_thresholds.get("decision_accuracy", 0.85)
        
        if avg_accuracy < threshold:
            self._trigger_business_alert(
                alert_type="decision_accuracy",
                severity="high",
                message=f"Decision accuracy below threshold: {avg_accuracy:.2f} < {threshold}",
                metric_name="decision_accuracy",
                current_value=avg_accuracy,
                threshold=threshold
            )
    
    @with_error_handling("analyze_user_engagement")
    def _analyze_user_engagement(self):
        """Analyze user engagement trends."""
        if not self.user_engagement_metrics:
            return
        
        # Get recent engagement (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_engagement = [
            e for e in self.user_engagement_metrics
            if e.timestamp > recent_cutoff
        ]
        
        if not recent_engagement:
            return
        
        # Calculate engagement rate (successful actions / total actions)
        total_actions = len(recent_engagement)
        successful_actions = len([e for e in recent_engagement if e.success])
        engagement_rate = successful_actions / total_actions if total_actions > 0 else 0
        
        threshold = self.business_thresholds.get("user_engagement_rate", 0.70)
        
        if engagement_rate < threshold:
            self._trigger_business_alert(
                alert_type="user_engagement",
                severity="medium",
                message=f"User engagement below threshold: {engagement_rate:.2f} < {threshold}",
                metric_name="user_engagement_rate",
                current_value=engagement_rate,
                threshold=threshold
            )
    
    @with_error_handling("analyze_feature_usage")
    def _analyze_feature_usage(self):
        """Analyze feature usage trends."""
        if not self.feature_usage_metrics:
            return
        
        # Get recent usage (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_usage = [
            f for f in self.feature_usage_metrics
            if f.timestamp > recent_cutoff
        ]
        
        if not recent_usage:
            return
        
        # Calculate adoption rate for each feature
        for feature_name in self.features:
            if not self.features[feature_name].get("tracking", True):
                continue
            
            feature_usage = [
                f for f in recent_usage
                if f.feature_name == feature_name
            ]
            
            if feature_usage:
                total_usage = sum(f.usage_count for f in feature_usage)
                avg_success_rate = statistics.mean([f.success_rate for f in feature_usage])
                
                # Simple adoption rate calculation
                adoption_rate = min(total_usage / 100, 1.0)  # Normalize to 0-1
                
                threshold = self.business_thresholds.get("feature_adoption_rate", 0.50)
                
                if adoption_rate < threshold:
                    self._trigger_business_alert(
                        alert_type="feature_usage",
                        severity="medium",
                        message=f"Feature adoption below threshold for {feature_name}: "
                               f"{adoption_rate:.2f} < {threshold}",
                        metric_name=f"feature_adoption_{feature_name}",
                        current_value=adoption_rate,
                        threshold=threshold
                    )
    
    @with_error_handling("analyze_system_performance")
    def _analyze_system_performance(self):
        """Analyze system performance trends."""
        if not self.system_performance_metrics:
            return
        
        # Get recent performance metrics (last hour)
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_performance = [
            p for p in self.system_performance_metrics
            if p.timestamp > recent_cutoff
        ]
        
        if not recent_performance:
            return
        
        # Check response time
        response_time_metrics = [
            p for p in recent_performance
            if p.metric_name == "response_time"
        ]
        
        if response_time_metrics:
            avg_response_time = statistics.mean([p.value for p in response_time_metrics])
            threshold = self.business_thresholds.get("response_time", 2.0)
            
            if avg_response_time > threshold:
                self._trigger_business_alert(
                    alert_type="performance",
                    severity="high",
                    message=f"Response time above threshold: {avg_response_time:.2f}s > {threshold}s",
                    metric_name="response_time",
                    current_value=avg_response_time,
                    threshold=threshold
                )
        
        # Check error rate
        error_rate_metrics = [
            p for p in recent_performance
            if p.metric_name == "error_rate"
        ]
        
        if error_rate_metrics:
            avg_error_rate = statistics.mean([p.value for p in error_rate_metrics])
            threshold = self.business_thresholds.get("error_rate", 0.05)
            
            if avg_error_rate > threshold:
                self._trigger_business_alert(
                    alert_type="performance",
                    severity="critical",
                    message=f"Error rate above threshold: {avg_error_rate:.2f} > {threshold}",
                    metric_name="error_rate",
                    current_value=avg_error_rate,
                    threshold=threshold
                )
    
    @with_error_handling("trigger_business_alert")
    def _trigger_business_alert(self, alert_type: str, severity: str,
                              message: str, metric_name: Optional[str] = None,
                              current_value: Optional[float] = None,
                              threshold: Optional[float] = None):
        """Trigger a business alert."""
        alert_id = f"business_alert_{int(time.time())}_{hash(message) % 10000}"
        
        alert = BusinessAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold
        )
        
        self.business_alerts.append(alert)
        logger.warning(f"Business alert: {alert_type} - {message}")
    
    @with_error_handling("check_business_alerts")
    def _check_business_alerts(self):
        """Check for business alerts based on thresholds."""
        # This method is called by the monitoring loop
        # Individual analysis methods trigger alerts as needed
        pass
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data."""
        cutoff_time = datetime.now() - timedelta(days=30)  # Keep 30 days
        
        # Clean up decision accuracy metrics
        self.decision_accuracy_metrics = [
            d for d in self.decision_accuracy_metrics
            if d.timestamp > cutoff_time
        ]
        
        # Clean up user engagement metrics
        self.user_engagement_metrics = [
            e for e in self.user_engagement_metrics
            if e.timestamp > cutoff_time
        ]
        
        # Clean up feature usage metrics
        self.feature_usage_metrics = [
            f for f in self.feature_usage_metrics
            if f.timestamp > cutoff_time
        ]
        
        # Clean up system performance metrics
        self.system_performance_metrics = [
            p for p in self.system_performance_metrics
            if p.timestamp > cutoff_time
        ]
        
        # Clean up alerts (keep for 90 days)
        alert_cutoff = datetime.now() - timedelta(days=90)
        self.business_alerts = [
            a for a in self.business_alerts
            if a.timestamp > alert_cutoff
        ]
    
    @with_error_handling("get_business_summary")
    async def get_business_summary(self) -> Dict[str, Any]:
        """Get a summary of business metrics."""
        # Get recent data (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        
        recent_decisions = [
            d for d in self.decision_accuracy_metrics
            if d.timestamp > recent_cutoff
        ]
        recent_engagement = [
            e for e in self.user_engagement_metrics
            if e.timestamp > recent_cutoff
        ]
        recent_usage = [
            f for f in self.feature_usage_metrics
            if f.timestamp > recent_cutoff
        ]
        recent_performance = [
            p for p in self.system_performance_metrics
            if p.timestamp > recent_cutoff
        ]
        
        summary = {
            "timestamp": datetime.now(),
            "decision_accuracy": {
                "total_decisions": len(recent_decisions),
                "average_accuracy": statistics.mean([d.accuracy_score for d in recent_decisions]) if recent_decisions else 0,
                "average_confidence": statistics.mean([d.confidence_score for d in recent_decisions]) if recent_decisions else 0
            },
            "user_engagement": {
                "total_actions": len(recent_engagement),
                "unique_users": len(set(e.user_id for e in recent_engagement)),
                "success_rate": len([e for e in recent_engagement if e.success]) / len(recent_engagement) if recent_engagement else 0
            },
            "feature_usage": {
                "total_usage": sum(f.usage_count for f in recent_usage),
                "features_used": len(set(f.feature_name for f in recent_usage)),
                "average_success_rate": statistics.mean([f.success_rate for f in recent_usage]) if recent_usage else 0
            },
            "system_performance": {
                "total_metrics": len(recent_performance),
                "categories": list(set(p.category for p in recent_performance))
            },
            "alerts": {
                "total": len(self.business_alerts),
                "recent": len([a for a in self.business_alerts if a.timestamp > recent_cutoff]),
                "by_severity": {}
            }
        }
        
        # Count alerts by severity
        for alert in self.business_alerts:
            if alert.timestamp > recent_cutoff:
                if alert.severity not in summary["alerts"]["by_severity"]:
                    summary["alerts"]["by_severity"][alert.severity] = 0
                summary["alerts"]["by_severity"][alert.severity] += 1
        
        return summary
    
    @with_error_handling("get_decision_accuracy_report")
    async def get_decision_accuracy_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get detailed decision accuracy report."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_decisions = [
            d for d in self.decision_accuracy_metrics
            if d.timestamp > cutoff_time
        ]
        
        if not recent_decisions:
            return {"error": f"No decision data available for the last {hours} hours"}
        
        # Group by decision type
        decision_types = defaultdict(list)
        for decision in recent_decisions:
            decision_types[decision.decision_type].append(decision)
        
        report = {
            "period_hours": hours,
            "total_decisions": len(recent_decisions),
            "overall_accuracy": statistics.mean([d.accuracy_score for d in recent_decisions]),
            "overall_confidence": statistics.mean([d.confidence_score for d in recent_decisions]),
            "by_decision_type": {}
        }
        
        for decision_type, decisions in decision_types.items():
            report["by_decision_type"][decision_type] = {
                "count": len(decisions),
                "average_accuracy": statistics.mean([d.accuracy_score for d in decisions]),
                "average_confidence": statistics.mean([d.confidence_score for d in decisions]),
                "accuracy_trend": sorted([d.accuracy_score for d in decisions])
            }
        
        return report
    
    @with_error_handling("get_user_engagement_report")
    async def get_user_engagement_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get detailed user engagement report."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_engagement = [
            e for e in self.user_engagement_metrics
            if e.timestamp > cutoff_time
        ]
        
        if not recent_engagement:
            return {"error": f"No engagement data available for the last {hours} hours"}
        
        # Group by action type
        action_types = defaultdict(list)
        for engagement in recent_engagement:
            action_types[engagement.action_type].append(engagement)
        
        report = {
            "period_hours": hours,
            "total_actions": len(recent_engagement),
            "unique_users": len(set(e.user_id for e in recent_engagement)),
            "overall_success_rate": len([e for e in recent_engagement if e.success]) / len(recent_engagement),
            "average_duration": statistics.mean([e.duration for e in recent_engagement]),
            "by_action_type": {}
        }
        
        for action_type, actions in action_types.items():
            report["by_action_type"][action_type] = {
                "count": len(actions),
                "success_rate": len([a for a in actions if a.success]) / len(actions),
                "average_duration": statistics.mean([a.duration for a in actions]),
                "unique_users": len(set(a.user_id for a in actions))
            }
        
        return report


# Global instance
business_metrics_monitor = BusinessMetricsMonitor()
