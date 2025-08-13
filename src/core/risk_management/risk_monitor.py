"""
Risk Monitor Module

This module provides continuous risk tracking and monitoring capabilities
for ongoing risk assessment and alerting.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
import asyncio
import threading
import time

from .risk_identifier import Risk
from .risk_quantifier import RiskImpact
from .mitigation_planner import MitigationPlan

logger = logging.getLogger(__name__)


@dataclass
class RiskAlert:
    """Risk alert data structure."""
    alert_id: str
    risk_id: str
    alert_type: str  # 'new_risk', 'escalation', 'mitigation_needed', 'threshold_exceeded'
    severity: str
    message: str
    timestamp: datetime
    risk_data: Dict[str, Any]
    action_required: bool
    assigned_to: Optional[str] = None


@dataclass
class RiskTrend:
    """Risk trend data structure."""
    risk_id: str
    trend_period: str  # 'daily', 'weekly', 'monthly'
    start_date: datetime
    end_date: datetime
    risk_scores: List[float]
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0.0 to 1.0
    confidence_level: float


class RiskMonitor:
    """
    Continuous risk monitoring system that tracks risk trends,
    generates alerts, and provides real-time risk assessment.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the risk monitor with configuration."""
        self.config = config or {}
        self.alert_thresholds = self._load_alert_thresholds()
        self.monitoring_rules = self._load_monitoring_rules()
        self.risk_history = {}  # risk_id -> List[Risk]
        self.active_alerts = {}  # alert_id -> RiskAlert
        self.monitoring_active = False
        self.monitoring_thread = None
        self.alert_callbacks = []
        self.logger = logging.getLogger(__name__)
        
    def _load_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load alert thresholds for different risk categories."""
        return {
            'data_quality': {
                'new_risk_threshold': 0.3,
                'escalation_threshold': 0.7,
                'critical_threshold': 0.9
            },
            'performance': {
                'new_risk_threshold': 0.4,
                'escalation_threshold': 0.8,
                'critical_threshold': 0.95
            },
            'security': {
                'new_risk_threshold': 0.2,
                'escalation_threshold': 0.6,
                'critical_threshold': 0.8
            },
            'business': {
                'new_risk_threshold': 0.5,
                'escalation_threshold': 0.8,
                'critical_threshold': 0.9
            }
        }
    
    def _load_monitoring_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load monitoring rules and parameters."""
        return {
            'monitoring_interval': 300,  # 5 minutes
            'trend_analysis_period': 7,  # days
            'alert_retention_days': 30,
            'max_active_alerts': 100,
            'auto_escalation_enabled': True,
            'trend_analysis_enabled': True
        }
    
    def add_risk_to_history(self, risk: Risk) -> None:
        """Add a risk to the monitoring history."""
        if risk.id not in self.risk_history:
            self.risk_history[risk.id] = []
        
        self.risk_history[risk.id].append(risk)
        
        # Keep only recent history (last 90 days)
        cutoff_date = datetime.now() - timedelta(days=90)
        self.risk_history[risk.id] = [
            r for r in self.risk_history[risk.id] 
            if r.detected_at > cutoff_date
        ]
        
        self.logger.debug(f"Added risk {risk.id} to monitoring history")
    
    def calculate_risk_score(self, risk: Risk) -> float:
        """Calculate a normalized risk score for monitoring."""
        # Base score from probability and impact
        base_score = risk.probability * risk.impact
        
        # Adjust based on severity
        severity_multiplier = {
            'low': 0.5,
            'medium': 1.0,
            'high': 1.5,
            'critical': 2.0
        }
        
        adjusted_score = base_score * severity_multiplier.get(risk.severity, 1.0)
        
        # Normalize to 0-1 range
        return min(adjusted_score, 1.0)
    
    def check_alert_conditions(self, risk: Risk, 
                             risk_impact: Optional[RiskImpact] = None) -> List[RiskAlert]:
        """Check if risk meets alert conditions."""
        alerts = []
        risk_score = self.calculate_risk_score(risk)
        thresholds = self.alert_thresholds.get(risk.category, {})
        
        # Check for new risk alert
        if risk_score >= thresholds.get('new_risk_threshold', 0.3):
            alerts.append(RiskAlert(
                alert_id=f"new_{risk.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                risk_id=risk.id,
                alert_type='new_risk',
                severity=risk.severity,
                message=f"New {risk.category} risk detected: {risk.description}",
                timestamp=datetime.now(),
                risk_data={
                    'score': risk_score,
                    'probability': risk.probability,
                    'impact': risk.impact,
                    'category': risk.category
                },
                action_required=True
            ))
        
        # Check for escalation
        if risk_score >= thresholds.get('escalation_threshold', 0.7):
            alerts.append(RiskAlert(
                alert_id=f"escalation_{risk.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                risk_id=risk.id,
                alert_type='escalation',
                severity='high',
                message=f"Risk escalation: {risk.description}",
                timestamp=datetime.now(),
                risk_data={
                    'score': risk_score,
                    'previous_score': self._get_previous_risk_score(risk.id),
                    'escalation_factor': risk_score / max(self._get_previous_risk_score(risk.id), 0.1)
                },
                action_required=True
            ))
        
        # Check for critical threshold
        if risk_score >= thresholds.get('critical_threshold', 0.9):
            alerts.append(RiskAlert(
                alert_id=f"critical_{risk.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                risk_id=risk.id,
                alert_type='threshold_exceeded',
                severity='critical',
                message=f"Critical risk threshold exceeded: {risk.description}",
                timestamp=datetime.now(),
                risk_data={
                    'score': risk_score,
                    'threshold': thresholds.get('critical_threshold', 0.9)
                },
                action_required=True
            ))
        
        return alerts
    
    def _get_previous_risk_score(self, risk_id: str) -> float:
        """Get the previous risk score for comparison."""
        if risk_id not in self.risk_history or len(self.risk_history[risk_id]) < 2:
            return 0.0
        
        # Get the second most recent risk score
        previous_risk = self.risk_history[risk_id][-2]
        return self.calculate_risk_score(previous_risk)
    
    def analyze_risk_trend(self, risk_id: str, 
                          period_days: int = 7) -> Optional[RiskTrend]:
        """Analyze risk trend over a specified period."""
        if risk_id not in self.risk_history:
            return None
        
        risk_history = self.risk_history[risk_id]
        cutoff_date = datetime.now() - timedelta(days=period_days)
        
        # Filter risks within the period
        recent_risks = [r for r in risk_history if r.detected_at > cutoff_date]
        
        if len(recent_risks) < 2:
            return None
        
        # Calculate risk scores over time
        risk_scores = [self.calculate_risk_score(r) for r in recent_risks]
        
        # Calculate trend direction and strength
        if len(risk_scores) >= 2:
            trend_direction, trend_strength = self._calculate_trend(risk_scores)
        else:
            trend_direction = 'stable'
            trend_strength = 0.0
        
        # Calculate confidence level based on data points
        confidence_level = min(len(risk_scores) / 10.0, 1.0)
        
        return RiskTrend(
            risk_id=risk_id,
            trend_period=f"{period_days}_days",
            start_date=recent_risks[0].detected_at,
            end_date=recent_risks[-1].detected_at,
            risk_scores=risk_scores,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            confidence_level=confidence_level
        )
    
    def _calculate_trend(self, scores: List[float]) -> tuple[str, float]:
        """Calculate trend direction and strength from a list of scores."""
        if len(scores) < 2:
            return 'stable', 0.0
        
        # Calculate linear regression slope
        x = np.arange(len(scores))
        y = np.array(scores)
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine trend direction
        if abs(slope) < 0.01:
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        # Calculate trend strength (normalized)
        trend_strength = min(abs(slope) * 10, 1.0)
        
        return direction, trend_strength
    
    def add_alert_callback(self, callback: Callable[[RiskAlert], None]) -> None:
        """Add a callback function for risk alerts."""
        self.alert_callbacks.append(callback)
        self.logger.info(f"Added alert callback: {callback.__name__}")
    
    def _trigger_alert_callbacks(self, alert: RiskAlert) -> None:
        """Trigger all registered alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback {callback.__name__}: {e}")
    
    def process_risk_update(self, risk: Risk, 
                          risk_impact: Optional[RiskImpact] = None) -> List[RiskAlert]:
        """Process a risk update and generate alerts."""
        # Add to history
        self.add_risk_to_history(risk)
        
        # Check for alerts
        alerts = self.check_alert_conditions(risk, risk_impact)
        
        # Store active alerts
        for alert in alerts:
            self.active_alerts[alert.alert_id] = alert
            self._trigger_alert_callbacks(alert)
        
        # Clean up old alerts
        self._cleanup_old_alerts()
        
        self.logger.info(f"Processed risk update for {risk.id}, generated {len(alerts)} alerts")
        return alerts
    
    def _cleanup_old_alerts(self) -> None:
        """Clean up old alerts based on retention policy."""
        retention_days = self.monitoring_rules.get('alert_retention_days', 30)
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        alerts_to_remove = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.timestamp < cutoff_date
        ]
        
        for alert_id in alerts_to_remove:
            del self.active_alerts[alert_id]
        
        if alerts_to_remove:
            self.logger.debug(f"Cleaned up {len(alerts_to_remove)} old alerts")
    
    def get_active_alerts(self, 
                         severity_filter: Optional[str] = None,
                         category_filter: Optional[str] = None) -> List[RiskAlert]:
        """Get active alerts with optional filtering."""
        alerts = list(self.active_alerts.values())
        
        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]
        
        if category_filter:
            alerts = [a for a in alerts if a.risk_data.get('category') == category_filter]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get a summary of monitored risks."""
        total_risks = sum(len(risks) for risks in self.risk_history.values())
        active_alerts_count = len(self.active_alerts)
        
        # Risk distribution by category
        category_distribution = {}
        for risks in self.risk_history.values():
            for risk in risks:
                category = risk.category
                category_distribution[category] = category_distribution.get(category, 0) + 1
        
        # Alert distribution by severity
        severity_distribution = {}
        for alert in self.active_alerts.values():
            severity = alert.severity
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
        
        return {
            'total_risks_monitored': total_risks,
            'active_alerts': active_alerts_count,
            'category_distribution': category_distribution,
            'severity_distribution': severity_distribution,
            'monitoring_active': self.monitoring_active
        }
    
    def start_monitoring(self) -> None:
        """Start continuous risk monitoring."""
        if self.monitoring_active:
            self.logger.warning("Risk monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Started continuous risk monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop continuous risk monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Stopped continuous risk monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop for continuous risk assessment."""
        interval = self.monitoring_rules.get('monitoring_interval', 300)
        
        while self.monitoring_active:
            try:
                # Perform periodic risk assessments
                self._perform_periodic_assessment()
                
                # Analyze trends for monitored risks
                if self.monitoring_rules.get('trend_analysis_enabled', True):
                    self._analyze_all_trends()
                
                # Wait for next interval
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _perform_periodic_assessment(self) -> None:
        """Perform periodic risk assessment for monitored risks."""
        # This would typically involve checking current system state
        # and comparing with historical data to detect new risks
        self.logger.debug("Performing periodic risk assessment")
    
    def _analyze_all_trends(self) -> None:
        """Analyze trends for all monitored risks."""
        for risk_id in self.risk_history.keys():
            trend = self.analyze_risk_trend(risk_id)
            if trend and trend.trend_strength > 0.5:
                self.logger.info(f"Strong trend detected for risk {risk_id}: "
                               f"{trend.trend_direction} (strength: {trend.trend_strength:.2f})")
