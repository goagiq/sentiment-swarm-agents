"""
Alert System

This module provides automated notification capabilities for real-time
monitoring events and pattern detection.
"""

import asyncio
import logging
import smtplib
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

logger = logging.getLogger(__name__)


@dataclass
class AlertRule:
    """Defines an alert rule for pattern detection"""
    rule_id: str
    pattern_type: str
    severity_threshold: str  # info, warning, critical
    conditions: Dict[str, Any]
    actions: List[str]  # email, webhook, log, etc.
    enabled: bool = True
    cooldown_minutes: int = 5


@dataclass
class Alert:
    """Represents an alert that has been triggered"""
    alert_id: str
    rule_id: str
    pattern_type: str
    severity: str
    message: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


@dataclass
class AlertConfig:
    """Configuration for the alert system"""
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = field(default_factory=list)
    
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    
    log_enabled: bool = True
    alert_history_size: int = 1000


class AlertSystem:
    """
    Alert system that provides automated notifications for monitoring events
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        """
        Initialize the alert system
        
        Args:
            config: Alert system configuration
        """
        self.config = config or AlertConfig()
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.last_alert_times: Dict[str, datetime] = {}
        
        logger.info("AlertSystem initialized")
    
    def add_rule(self, rule: AlertRule):
        """
        Add an alert rule
        
        Args:
            rule: Alert rule to add
        """
        self.rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.rule_id}")
    
    def remove_rule(self, rule_id: str):
        """
        Remove an alert rule
        
        Args:
            rule_id: ID of the rule to remove
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
    
    def add_callback(self, callback: Callable[[Alert], None]):
        """
        Add a callback function to be called when alerts are triggered
        
        Args:
            callback: Function to call with Alert
        """
        self.alert_callbacks.append(callback)
    
    async def process_pattern_event(self, pattern_event: Any):
        """
        Process a pattern event and trigger alerts if conditions are met
        
        Args:
            pattern_event: Pattern event from the monitor
        """
        # Check each rule against the pattern event
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            if self._should_trigger_alert(rule, pattern_event):
                await self._trigger_alert(rule, pattern_event)
    
    def _should_trigger_alert(self, rule: AlertRule, pattern_event: Any) -> bool:
        """
        Check if an alert should be triggered based on the rule and event
        
        Args:
            rule: Alert rule to check
            pattern_event: Pattern event to evaluate
            
        Returns:
            True if alert should be triggered
        """
        # Check pattern type
        if rule.pattern_type != pattern_event.pattern_type:
            return False
        
        # Check severity threshold
        severity_levels = {"info": 0, "warning": 1, "critical": 2}
        event_severity = severity_levels.get(pattern_event.severity, 0)
        rule_severity = severity_levels.get(rule.severity_threshold, 0)
        
        if event_severity < rule_severity:
            return False
        
        # Check cooldown
        if rule.rule_id in self.last_alert_times:
            time_since_last = datetime.now() - self.last_alert_times[rule.rule_id]
            if time_since_last < timedelta(minutes=rule.cooldown_minutes):
                return False
        
        # Check custom conditions
        if not self._evaluate_conditions(rule.conditions, pattern_event):
            return False
        
        return True
    
    def _evaluate_conditions(self, conditions: Dict[str, Any], pattern_event: Any) -> bool:
        """
        Evaluate custom conditions against a pattern event
        
        Args:
            conditions: Conditions to evaluate
            pattern_event: Pattern event to check
            
        Returns:
            True if all conditions are met
        """
        for condition_type, condition_value in conditions.items():
            if condition_type == "confidence_min":
                if pattern_event.confidence < condition_value:
                    return False
            elif condition_type == "confidence_max":
                if pattern_event.confidence > condition_value:
                    return False
            elif condition_type == "data_points_min":
                if len(pattern_event.data_points) < condition_value:
                    return False
            elif condition_type == "metadata_contains":
                if not self._check_metadata_contains(pattern_event.metadata, condition_value):
                    return False
        
        return True
    
    def _check_metadata_contains(self, metadata: Dict[str, Any], required_items: Dict[str, Any]) -> bool:
        """
        Check if metadata contains required items
        
        Args:
            metadata: Event metadata
            required_items: Required metadata items
            
        Returns:
            True if all required items are present
        """
        for key, value in required_items.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
    
    async def _trigger_alert(self, rule: AlertRule, pattern_event: Any):
        """
        Trigger an alert based on a rule and pattern event
        
        Args:
            rule: Alert rule that was triggered
            pattern_event: Pattern event that triggered the alert
        """
        # Create alert
        alert = Alert(
            alert_id=f"alert_{len(self.alerts)}_{datetime.now().timestamp()}",
            rule_id=rule.rule_id,
            pattern_type=pattern_event.pattern_type,
            severity=pattern_event.severity,
            message=self._generate_alert_message(rule, pattern_event),
            timestamp=datetime.now(),
            data={
                'pattern_id': pattern_event.pattern_id,
                'confidence': pattern_event.confidence,
                'metadata': pattern_event.metadata
            }
        )
        
        # Add to alert history
        self.alerts.append(alert)
        if len(self.alerts) > self.config.alert_history_size:
            self.alerts.pop(0)
        
        # Update last alert time
        self.last_alert_times[rule.rule_id] = datetime.now()
        
        # Execute actions
        for action in rule.actions:
            try:
                if action == "email" and self.config.email_enabled:
                    await self._send_email_alert(alert)
                elif action == "webhook" and self.config.webhook_enabled:
                    await self._send_webhook_alert(alert)
                elif action == "slack" and self.config.slack_enabled:
                    await self._send_slack_alert(alert)
                elif action == "log" and self.config.log_enabled:
                    self._log_alert(alert)
            except Exception as e:
                logger.error(f"Error executing alert action {action}: {str(e)}")
        
        # Call callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}")
        
        logger.info(f"Alert triggered: {alert.alert_id} - {alert.message}")
    
    def _generate_alert_message(self, rule: AlertRule, pattern_event: Any) -> str:
        """
        Generate alert message from rule and pattern event
        
        Args:
            rule: Alert rule
            pattern_event: Pattern event
            
        Returns:
            Generated alert message
        """
        return (
            f"Pattern Alert: {pattern_event.pattern_type.upper()} detected\n"
            f"Confidence: {pattern_event.confidence:.2f}\n"
            f"Severity: {pattern_event.severity}\n"
            f"Pattern ID: {pattern_event.pattern_id}\n"
            f"Timestamp: {pattern_event.timestamp}"
        )
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        if not self.config.email_username or not self.config.email_password:
            logger.warning("Email credentials not configured")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.email_username
            msg['To'] = ", ".join(self.config.email_recipients)
            msg['Subject'] = f"Alert: {alert.pattern_type.upper()} - {alert.severity}"
            
            body = alert.message
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config.email_smtp_server, self.config.email_smtp_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            
            text = msg.as_string()
            server.sendmail(self.config.email_username, self.config.email_recipients, text)
            server.quit()
            
            logger.info(f"Email alert sent: {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
    
    async def _send_webhook_alert(self, alert: Alert):
        """Send webhook alert"""
        try:
            payload = {
                'alert_id': alert.alert_id,
                'rule_id': alert.rule_id,
                'pattern_type': alert.pattern_type,
                'severity': alert.severity,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'data': alert.data
            }
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                headers=self.config.webhook_headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Webhook alert sent: {alert.alert_id}")
            else:
                logger.warning(f"Webhook alert failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending webhook alert: {str(e)}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        try:
            payload = {
                'text': f"🚨 *{alert.pattern_type.upper()} Alert*\n{alert.message}",
                'attachments': [{
                    'fields': [
                        {'title': 'Severity', 'value': alert.severity, 'short': True},
                        {'title': 'Confidence', 'value': f"{alert.data.get('confidence', 0):.2f}", 'short': True}
                    ]
                }]
            }
            
            response = requests.post(
                self.config.slack_webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent: {alert.alert_id}")
            else:
                logger.warning(f"Slack alert failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending Slack alert: {str(e)}")
    
    def _log_alert(self, alert: Alert):
        """Log alert to system logs"""
        log_level = logging.WARNING if alert.severity in ["warning", "critical"] else logging.INFO
        logger.log(log_level, f"ALERT: {alert.message}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """
        Acknowledge an alert
        
        Args:
            alert_id: ID of the alert to acknowledge
            acknowledged_by: User who acknowledged the alert
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                break
    
    def get_alerts(self, 
                   severity: Optional[str] = None,
                   pattern_type: Optional[str] = None,
                   acknowledged: Optional[bool] = None,
                   limit: int = 50) -> List[Alert]:
        """
        Get alerts with optional filtering
        
        Args:
            severity: Filter by severity
            pattern_type: Filter by pattern type
            acknowledged: Filter by acknowledgment status
            limit: Maximum number of alerts to return
            
        Returns:
            List of filtered alerts
        """
        filtered_alerts = self.alerts
        
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
        
        if pattern_type:
            filtered_alerts = [a for a in filtered_alerts if a.pattern_type == pattern_type]
        
        if acknowledged is not None:
            filtered_alerts = [a for a in filtered_alerts if a.acknowledged == acknowledged]
        
        return filtered_alerts[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics"""
        total_alerts = len(self.alerts)
        unacknowledged = len([a for a in self.alerts if not a.acknowledged])
        
        severity_counts = {}
        pattern_type_counts = {}
        
        for alert in self.alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
            pattern_type_counts[alert.pattern_type] = pattern_type_counts.get(alert.pattern_type, 0) + 1
        
        return {
            'total_alerts': total_alerts,
            'unacknowledged_alerts': unacknowledged,
            'severity_counts': severity_counts,
            'pattern_type_counts': pattern_type_counts,
            'active_rules': len([r for r in self.rules.values() if r.enabled])
        }
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts.clear()
        logger.info("All alerts cleared")
