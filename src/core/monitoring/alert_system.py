"""
Alert System for Decision Monitoring

This module provides alert generation, notification, and management
for the automated decision monitoring system.
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from loguru import logger

from src.core.error_handler import with_error_handling


@dataclass
class AlertNotification:
    """Represents an alert notification."""
    notification_id: str
    alert_id: str
    notification_type: str  # email, slack, webhook, sms
    recipient: str
    message: str
    status: str = "pending"  # pending, sent, failed
    sent_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Represents an alert rule for automated monitoring."""
    rule_id: str
    rule_name: str
    condition: str  # metric > threshold, metric < threshold, etc.
    metric_name: str
    threshold_value: float
    severity: str  # low, medium, high, critical
    notification_channels: List[str] = field(default_factory=list)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertSystem:
    """
    Alert system for generating and managing notifications
    based on decision monitoring events.
    """
    
    def __init__(self):
        self.notifications: List[AlertNotification] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        self.notification_handlers: Dict[str, Callable] = {}
        
        # Default notification channels
        self.default_channels = {
            "email": self._send_email_notification,
            "slack": self._send_slack_notification,
            "webhook": self._send_webhook_notification,
            "sms": self._send_sms_notification
        }
        
        # Initialize default handlers
        self._initialize_default_handlers()
        
        logger.info("AlertSystem initialized")
    
    def _initialize_default_handlers(self):
        """Initialize default notification handlers."""
        for channel, handler in self.default_channels.items():
            self.notification_handlers[channel] = handler
    
    @with_error_handling("alert_generation")
    async def generate_alert(
        self,
        alert_id: str,
        alert_type: str,
        severity: str,
        message: str,
        decision_id: str,
        metric_name: str,
        current_value: float,
        threshold_value: float,
        notification_channels: List[str] = None
    ) -> bool:
        """
        Generate an alert and send notifications.
        
        Args:
            alert_id: Unique alert identifier
            alert_type: Type of alert
            severity: Alert severity
            message: Alert message
            decision_id: Associated decision ID
            metric_name: Metric that triggered the alert
            current_value: Current metric value
            threshold_value: Threshold value
            notification_channels: Channels to send notifications to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if notification_channels is None:
                notification_channels = ["email"]  # Default to email
            
            # Create notifications for each channel
            for channel in notification_channels:
                if channel in self.notification_handlers:
                    notification = AlertNotification(
                        notification_id=f"{alert_id}_{channel}",
                        alert_id=alert_id,
                        notification_type=channel,
                        recipient=self._get_recipient_for_channel(channel),
                        message=self._format_alert_message(
                            message, metric_name, current_value, threshold_value
                        )
                    )
                    
                    self.notifications.append(notification)
                    
                    # Send notification
                    await self._send_notification(notification)
                else:
                    logger.warning(f"Unknown notification channel: {channel}")
            
            logger.info(f"Generated alert {alert_id} with {len(notification_channels)} notifications")
            return True
            
        except Exception as e:
            logger.error(f"Error generating alert {alert_id}: {e}")
            return False
    
    async def _send_notification(self, notification: AlertNotification) -> bool:
        """Send a notification using the appropriate handler."""
        try:
            handler = self.notification_handlers.get(notification.notification_type)
            if handler:
                success = await handler(notification)
                if success:
                    notification.status = "sent"
                    notification.sent_at = datetime.now()
                else:
                    notification.status = "failed"
                
                logger.info(f"Notification {notification.notification_id} status: {notification.status}")
                return success
            else:
                logger.error(f"No handler for notification type: {notification.notification_type}")
                notification.status = "failed"
                return False
                
        except Exception as e:
            logger.error(f"Error sending notification {notification.notification_id}: {e}")
            notification.status = "failed"
            return False
    
    def _format_alert_message(
        self,
        message: str,
        metric_name: str,
        current_value: float,
        threshold_value: float
    ) -> str:
        """Format alert message with metric details."""
        return f"""
ALERT: {message}

Metric: {metric_name}
Current Value: {current_value:.2f}
Threshold: {threshold_value:.2f}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This is an automated alert from the Decision Monitoring System.
        """.strip()
    
    def _get_recipient_for_channel(self, channel: str) -> str:
        """Get recipient for a notification channel."""
        recipients = {
            "email": "admin@company.com",
            "slack": "#alerts",
            "webhook": "https://hooks.slack.com/services/...",
            "sms": "+1234567890"
        }
        return recipients.get(channel, "default")
    
    # Default notification handlers (placeholders)
    async def _send_email_notification(self, notification: AlertNotification) -> bool:
        """Send email notification (placeholder implementation)."""
        try:
            # In a real implementation, this would use an email service
            logger.info(f"Email notification sent to {notification.recipient}: {notification.message[:100]}...")
            await asyncio.sleep(0.1)  # Simulate network delay
            return True
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
    async def _send_slack_notification(self, notification: AlertNotification) -> bool:
        """Send Slack notification (placeholder implementation)."""
        try:
            # In a real implementation, this would use Slack API
            logger.info(f"Slack notification sent to {notification.recipient}: {notification.message[:100]}...")
            await asyncio.sleep(0.1)  # Simulate network delay
            return True
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False
    
    async def _send_webhook_notification(self, notification: AlertNotification) -> bool:
        """Send webhook notification (placeholder implementation)."""
        try:
            # In a real implementation, this would make an HTTP POST request
            logger.info(f"Webhook notification sent to {notification.recipient}: {notification.message[:100]}...")
            await asyncio.sleep(0.1)  # Simulate network delay
            return True
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False
    
    async def _send_sms_notification(self, notification: AlertNotification) -> bool:
        """Send SMS notification (placeholder implementation)."""
        try:
            # In a real implementation, this would use an SMS service
            logger.info(f"SMS notification sent to {notification.recipient}: {notification.message[:100]}...")
            await asyncio.sleep(0.1)  # Simulate network delay
            return True
        except Exception as e:
            logger.error(f"Error sending SMS notification: {e}")
            return False
    
    @with_error_handling("alert_rule_creation")
    async def create_alert_rule(
        self,
        rule_name: str,
        condition: str,
        metric_name: str,
        threshold_value: float,
        severity: str,
        notification_channels: List[str] = None
    ) -> AlertRule:
        """
        Create a new alert rule.
        
        Args:
            rule_name: Name of the rule
            condition: Condition string (e.g., ">", "<", "==")
            metric_name: Metric to monitor
            threshold_value: Threshold value
            severity: Alert severity
            notification_channels: Channels to notify
            
        Returns:
            AlertRule object
        """
        try:
            rule_id = f"rule_{len(self.alert_rules) + 1}"
            
            if notification_channels is None:
                notification_channels = ["email"]
            
            alert_rule = AlertRule(
                rule_id=rule_id,
                rule_name=rule_name,
                condition=condition,
                metric_name=metric_name,
                threshold_value=threshold_value,
                severity=severity,
                notification_channels=notification_channels
            )
            
            self.alert_rules[rule_id] = alert_rule
            
            logger.info(f"Created alert rule {rule_id}: {rule_name}")
            return alert_rule
            
        except Exception as e:
            logger.error(f"Error creating alert rule: {e}")
            raise
    
    @with_error_handling("alert_rule_evaluation")
    async def evaluate_alert_rules(
        self,
        metric_name: str,
        current_value: float
    ) -> List[AlertRule]:
        """
        Evaluate alert rules for a given metric.
        
        Args:
            metric_name: Name of the metric
            current_value: Current metric value
            
        Returns:
            List of triggered alert rules
        """
        try:
            triggered_rules = []
            
            for rule in self.alert_rules.values():
                if (rule.enabled and 
                    rule.metric_name == metric_name and 
                    self._evaluate_condition(current_value, rule.condition, rule.threshold_value)):
                    
                    triggered_rules.append(rule)
                    
                    # Generate alert for triggered rule
                    await self.generate_alert(
                        alert_id=f"auto_{rule.rule_id}_{datetime.now().timestamp()}",
                        alert_type="rule_triggered",
                        severity=rule.severity,
                        message=f"Alert rule '{rule.rule_name}' triggered",
                        decision_id="auto",
                        metric_name=metric_name,
                        current_value=current_value,
                        threshold_value=rule.threshold_value,
                        notification_channels=rule.notification_channels
                    )
            
            return triggered_rules
            
        except Exception as e:
            logger.error(f"Error evaluating alert rules: {e}")
            return []
    
    def _evaluate_condition(
        self,
        current_value: float,
        condition: str,
        threshold_value: float
    ) -> bool:
        """Evaluate a condition against current and threshold values."""
        try:
            if condition == ">":
                return current_value > threshold_value
            elif condition == "<":
                return current_value < threshold_value
            elif condition == ">=":
                return current_value >= threshold_value
            elif condition == "<=":
                return current_value <= threshold_value
            elif condition == "==":
                return abs(current_value - threshold_value) < 0.001
            elif condition == "!=":
                return abs(current_value - threshold_value) >= 0.001
            else:
                logger.warning(f"Unknown condition: {condition}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
    
    async def get_notification_history(
        self,
        time_window: timedelta = timedelta(days=7)
    ) -> List[AlertNotification]:
        """
        Get notification history for the specified time window.
        
        Args:
            time_window: Time window for history
            
        Returns:
            List of notifications
        """
        try:
            cutoff_time = datetime.now() - time_window
            
            recent_notifications = [
                n for n in self.notifications 
                if n.created_at >= cutoff_time
            ]
            
            return recent_notifications
            
        except Exception as e:
            logger.error(f"Error getting notification history: {e}")
            return []
    
    async def get_notification_stats(
        self,
        time_window: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """
        Get notification statistics for the specified time window.
        
        Args:
            time_window: Time window for statistics
            
        Returns:
            Statistics dictionary
        """
        try:
            notifications = await self.get_notification_history(time_window)
            
            stats = {
                "total_notifications": len(notifications),
                "sent_notifications": len([n for n in notifications if n.status == "sent"]),
                "failed_notifications": len([n for n in notifications if n.status == "failed"]),
                "pending_notifications": len([n for n in notifications if n.status == "pending"]),
                "by_channel": {},
                "by_severity": {}
            }
            
            # Group by channel
            for notification in notifications:
                channel = notification.notification_type
                if channel not in stats["by_channel"]:
                    stats["by_channel"][channel] = {
                        "total": 0,
                        "sent": 0,
                        "failed": 0
                    }
                
                stats["by_channel"][channel]["total"] += 1
                if notification.status == "sent":
                    stats["by_channel"][channel]["sent"] += 1
                elif notification.status == "failed":
                    stats["by_channel"][channel]["failed"] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting notification stats: {e}")
            return {}
    
    async def register_notification_handler(
        self,
        channel: str,
        handler: Callable
    ) -> bool:
        """
        Register a custom notification handler.
        
        Args:
            channel: Notification channel
            handler: Handler function
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.notification_handlers[channel] = handler
            logger.info(f"Registered custom notification handler for {channel}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering notification handler for {channel}: {e}")
            return False

    @with_error_handling("alert_creation")
    def create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        source: str,
        alert_id: Optional[str] = None
    ) -> str:
        """
        Create a simple alert without complex parameters.
        
        Args:
            alert_type: Type of alert
            severity: Alert severity
            message: Alert message
            source: Source of the alert
            alert_id: Optional alert ID
            
        Returns:
            Alert ID
        """
        try:
            if alert_id is None:
                alert_id = f"alert_{int(datetime.now().timestamp())}_{hash(message) % 10000}"
            
            # Create a simple alert notification
            notification = AlertNotification(
                notification_id=f"notif_{alert_id}",
                alert_id=alert_id,
                notification_type="log",
                recipient="system",
                message=message,
                status="pending",
                metadata={
                    "alert_type": alert_type,
                    "severity": severity,
                    "source": source
                }
            )
            
            self.notifications.append(notification)
            logger.info(f"Created alert {alert_id}: {message}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            raise

    @with_error_handling("get_active_alerts")
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        Get all active alerts.
        
        Returns:
            List of active alert dictionaries
        """
        try:
            active_alerts = []
            
            # Get recent notifications (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_notifications = [
                n for n in self.notifications
                if n.created_at > recent_cutoff
            ]
            
            for notification in recent_notifications:
                alert_data = {
                    "alert_id": notification.alert_id,
                    "alert_type": notification.metadata.get("alert_type", "unknown"),
                    "severity": notification.metadata.get("severity", "medium"),
                    "message": notification.message,
                    "source": notification.metadata.get("source", "unknown"),
                    "timestamp": notification.created_at.isoformat(),
                    "status": notification.status
                }
                active_alerts.append(alert_data)
            
            return active_alerts
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []

    @with_error_handling("acknowledge_alert")
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for notification in self.notifications:
                if notification.alert_id == alert_id:
                    notification.status = "acknowledged"
                    notification.metadata["acknowledged_at"] = datetime.now().isoformat()
                    logger.info(f"Alert {alert_id} acknowledged")
                    return True
            
            logger.warning(f"Alert {alert_id} not found for acknowledgment")
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False

    @with_error_handling("resolve_alert")
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for notification in self.notifications:
                if notification.alert_id == alert_id:
                    notification.status = "resolved"
                    notification.metadata["resolved_at"] = datetime.now().isoformat()
                    logger.info(f"Alert {alert_id} resolved")
                    return True
            
            logger.warning(f"Alert {alert_id} not found for resolution")
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False

    @property
    def active_alerts(self) -> List[AlertNotification]:
        """Get active alerts (notifications with pending status)."""
        return [n for n in self.notifications if n.status == "pending"]

    @property
    def alert_history(self) -> List[AlertNotification]:
        """Get all alert history."""
        return self.notifications


# Global alert system instance
alert_system = AlertSystem()


async def get_alert_system() -> AlertSystem:
    """Get the global alert system instance."""
    return alert_system
