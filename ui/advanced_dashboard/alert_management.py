"""
Alert Management

Alert management system for real-time analytics dashboard with:
- Custom alert rules
- Multi-channel notifications
- Alert severity levels
- Alert history and management
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any


class AlertManagement:
    """
    Alert management component for real-time analytics.
    """
    
    def __init__(self):
        """Initialize the alert management system."""
        self.alerts = []
        self.alert_rules = {}
        self.notification_channels = {
            'dashboard': True,
            'email': False,
            'webhook': False
        }
        self.severity_levels = {
            'critical': {'color': '#d62728', 'priority': 1},
            'high': {'color': '#ff7f0e', 'priority': 2},
            'medium': {'color': '#ffdc00', 'priority': 3},
            'low': {'color': '#2ca02c', 'priority': 4}
        }
    
    def create_alert_rule(self, name: str, condition: str, severity: str,
                         channels: List[str], config: Optional[Dict[str, Any]] = None) -> str:
        """Create a new alert rule."""
        rule_id = f"rule_{len(self.alert_rules) + 1}"
        
        rule = {
            'id': rule_id,
            'name': name,
            'condition': condition,
            'severity': severity,
            'channels': channels,
            'config': config or {},
            'created_at': datetime.now().isoformat(),
            'enabled': True,
            'trigger_count': 0
        }
        
        self.alert_rules[rule_id] = rule
        return rule_id
    
    def trigger_alert(self, rule_id: str, message: str, data: Optional[Dict[str, Any]] = None) -> str:
        """Trigger an alert based on a rule."""
        if rule_id not in self.alert_rules:
            return None
        
        rule = self.alert_rules[rule_id]
        if not rule['enabled']:
            return None
        
        alert_id = f"alert_{len(self.alerts) + 1}"
        
        alert = {
            'id': alert_id,
            'rule_id': rule_id,
            'rule_name': rule['name'],
            'message': message,
            'severity': rule['severity'],
            'data': data or {},
            'timestamp': datetime.now().isoformat(),
            'status': 'active',
            'acknowledged': False,
            'acknowledged_by': None,
            'acknowledged_at': None
        }
        
        self.alerts.append(alert)
        rule['trigger_count'] += 1
        
        # Send notifications
        self._send_notifications(alert)
        
        return alert_id
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                alert['acknowledged_by'] = user_id
                alert['acknowledged_at'] = datetime.now().isoformat()
                return True
        return False
    
    def resolve_alert(self, alert_id: str, user_id: str, resolution_notes: str = "") -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['status'] = 'resolved'
                alert['resolved_by'] = user_id
                alert['resolved_at'] = datetime.now().isoformat()
                alert['resolution_notes'] = resolution_notes
                return True
        return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return [alert for alert in self.alerts if alert['status'] == 'active']
    
    def get_alerts_by_severity(self, severity: str) -> List[Dict[str, Any]]:
        """Get alerts by severity level."""
        return [alert for alert in self.alerts if alert['severity'] == severity]
    
    def get_alert_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get alert history for the specified number of days."""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) >= cutoff_date
        ]
    
    def _send_notifications(self, alert: Dict[str, Any]) -> None:
        """Send notifications for an alert."""
        rule = self.alert_rules.get(alert['rule_id'])
        if not rule:
            return
        
        for channel in rule['channels']:
            if channel in self.notification_channels and self.notification_channels[channel]:
                if channel == 'dashboard':
                    # Dashboard notifications are handled by the UI
                    pass
                elif channel == 'email':
                    self._send_email_notification(alert)
                elif channel == 'webhook':
                    self._send_webhook_notification(alert)
    
    def _send_email_notification(self, alert: Dict[str, Any]) -> None:
        """Send email notification."""
        # This would integrate with an email service
        pass
    
    def _send_webhook_notification(self, alert: Dict[str, Any]) -> None:
        """Send webhook notification."""
        # This would integrate with webhook services
        pass
    
    def render_alert_manager(self):
        """Render the alert management interface."""
        st.markdown("## 🚨 Alert Management")
        
        # Create alert rules
        with st.expander("Create Alert Rule", expanded=False):
            self._render_create_alert_rule_form()
        
        # Alert rules list
        st.markdown("### 📋 Alert Rules")
        self._render_alert_rules_list()
        
        # Active alerts
        st.markdown("### ⚠️ Active Alerts")
        self._render_active_alerts()
        
        # Alert history
        st.markdown("### 📊 Alert History")
        self._render_alert_history()
        
        # Notification settings
        st.markdown("### 🔔 Notification Settings")
        self._render_notification_settings()
    
    def _render_create_alert_rule_form(self):
        """Render the create alert rule form."""
        with st.form("create_alert_rule"):
            rule_name = st.text_input("Rule Name")
            
            condition_type = st.selectbox(
                "Condition Type",
                ["threshold", "anomaly", "trend", "custom"]
            )
            
            if condition_type == "threshold":
                metric = st.selectbox("Metric", ["CPU Usage", "Memory Usage", "Response Time"])
                operator = st.selectbox("Operator", [">", "<", ">=", "<=", "=="])
                threshold_value = st.number_input("Threshold Value", value=80.0)
                condition = f"{metric} {operator} {threshold_value}"
            
            elif condition_type == "anomaly":
                metric = st.selectbox("Metric", ["CPU Usage", "Memory Usage", "Response Time"])
                sensitivity = st.slider("Sensitivity", 1, 10, 5)
                condition = f"Anomaly detected in {metric} (sensitivity: {sensitivity})"
            
            elif condition_type == "trend":
                metric = st.selectbox("Metric", ["CPU Usage", "Memory Usage", "Response Time"])
                trend_direction = st.selectbox("Trend Direction", ["increasing", "decreasing"])
                time_window = st.selectbox("Time Window", ["5m", "15m", "1h", "6h", "24h"])
                condition = f"{metric} {trend_direction} trend over {time_window}"
            
            else:  # custom
                condition = st.text_area("Custom Condition")
            
            severity = st.selectbox(
                "Severity Level",
                list(self.severity_levels.keys())
            )
            
            channels = st.multiselect(
                "Notification Channels",
                list(self.notification_channels.keys()),
                default=["dashboard"]
            )
            
            if st.form_submit_button("Create Alert Rule"):
                if rule_name and condition:
                    rule_id = self.create_alert_rule(rule_name, condition, severity, channels)
                    st.success(f"Alert rule created with ID: {rule_id}")
                else:
                    st.error("Please enter rule name and condition")
    
    def _render_alert_rules_list(self):
        """Render the alert rules list."""
        if not self.alert_rules:
            st.info("No alert rules created yet")
            return
        
        for rule_id, rule in self.alert_rules.items():
            with st.expander(f"{rule['name']} ({rule['severity']})"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Condition:** {rule['condition']}")
                    st.write(f"**Channels:** {', '.join(rule['channels'])}")
                
                with col2:
                    st.write(f"**Trigger Count:** {rule['trigger_count']}")
                    st.write(f"**Created:** {rule['created_at'][:10]}")
                
                with col3:
                    enabled = st.checkbox("Enabled", value=rule['enabled'], key=f"enabled_{rule_id}")
                    if enabled != rule['enabled']:
                        rule['enabled'] = enabled
                    
                    if st.button("Test", key=f"test_{rule_id}"):
                        alert_id = self.trigger_alert(rule_id, "Test alert triggered")
                        if alert_id:
                            st.success(f"Test alert triggered: {alert_id}")
    
    def _render_active_alerts(self):
        """Render active alerts."""
        active_alerts = self.get_active_alerts()
        
        if not active_alerts:
            st.info("No active alerts")
            return
        
        for alert in active_alerts:
            severity_config = self.severity_levels[alert['severity']]
            
            st.markdown(f"""
            <div style="
                background-color: {severity_config['color']}20;
                border-left: 4px solid {severity_config['color']};
                padding: 1rem;
                margin: 0.5rem 0;
                border-radius: 4px;
            ">
                <strong>{alert['rule_name']}</strong> ({alert['severity'].upper()})<br>
                {alert['message']}<br>
                <small>Time: {alert['timestamp'][:16]} | ID: {alert['id']}</small>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Acknowledge", key=f"ack_{alert['id']}"):
                    if self.acknowledge_alert(alert['id'], "current_user"):
                        st.success("Alert acknowledged")
                        st.rerun()
            
            with col2:
                if st.button("Resolve", key=f"resolve_{alert['id']}"):
                    resolution_notes = st.text_input("Resolution Notes", key=f"notes_{alert['id']}")
                    if resolution_notes:
                        if self.resolve_alert(alert['id'], "current_user", resolution_notes):
                            st.success("Alert resolved")
                            st.rerun()
    
    def _render_alert_history(self):
        """Render alert history."""
        days = st.slider("Show history for (days)", 1, 30, 7)
        history = self.get_alert_history(days)
        
        if not history:
            st.info(f"No alerts in the last {days} days")
            return
        
        # Group by severity
        severity_counts = {}
        for alert in history:
            severity = alert['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Display summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Critical", severity_counts.get('critical', 0))
        
        with col2:
            st.metric("High", severity_counts.get('high', 0))
        
        with col3:
            st.metric("Medium", severity_counts.get('medium', 0))
        
        with col4:
            st.metric("Low", severity_counts.get('low', 0))
        
        # Show recent alerts
        st.markdown("#### Recent Alerts")
        for alert in history[-10:]:  # Show last 10 alerts
            status_icon = "✅" if alert['status'] == 'resolved' else "⚠️"
            st.write(f"{status_icon} {alert['rule_name']} - {alert['message'][:50]}... ({alert['timestamp'][:16]})")
    
    def _render_notification_settings(self):
        """Render notification settings."""
        st.markdown("#### Notification Channels")
        
        for channel, enabled in self.notification_channels.items():
            new_enabled = st.checkbox(
                f"Enable {channel.title()} notifications",
                value=enabled,
                key=f"notif_{channel}"
            )
            if new_enabled != enabled:
                self.notification_channels[channel] = new_enabled
        
        st.markdown("#### Alert Severity Levels")
        
        for severity, config in self.severity_levels.items():
            st.color_picker(
                f"{severity.title()} Alert Color",
                value=config['color'],
                key=f"color_{severity}"
            )


# Factory function for creating alert management
def create_alert_management() -> AlertManagement:
    """Create an alert management system."""
    return AlertManagement()
