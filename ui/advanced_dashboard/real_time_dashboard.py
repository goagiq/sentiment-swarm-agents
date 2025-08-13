"""
Real-Time Analytics Dashboard

Advanced real-time dashboard with live data streaming,
interactive visualizations, and real-time analytics.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime
from typing import Dict, Optional, Any

# API configuration
API_BASE_URL = "http://localhost:8003"


class RealTimeDashboard:
    """
    Real-time analytics dashboard with live data streaming and interactive features.
    """
    
    def __init__(self):
        """Initialize the real-time dashboard."""
        self.api_base_url = API_BASE_URL
        self.dashboard_config = self._load_dashboard_config()
        self.active_widgets = {}
        self.data_streams = {}
        
    def _load_dashboard_config(self) -> Dict[str, Any]:
        """Load dashboard configuration."""
        return {
            'refresh_interval': 5.0,  # seconds
            'max_data_points': 1000,
            'enable_auto_refresh': True,
            'widget_layout': 'grid',
            'theme': 'plotly_white'
        }
    
    def render_dashboard(self):
        """Render the main real-time dashboard."""
        st.set_page_config(
            page_title="Real-Time Analytics Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for dashboard styling
        st.markdown(self._get_dashboard_css(), unsafe_allow_html=True)
        
        # Dashboard header
        st.markdown("""
        <div class="dashboard-header">
            <h1>🚀 Real-Time Analytics Dashboard</h1>
            <p>Live data streaming and interactive analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar controls
        self._render_sidebar_controls()
        
        # Main dashboard content
        self._render_dashboard_content()
        
        # Auto-refresh functionality
        if self.dashboard_config['enable_auto_refresh']:
            self._setup_auto_refresh()
    
    def _get_dashboard_css(self) -> str:
        """Get custom CSS for dashboard styling."""
        return """
        <style>
        .dashboard-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        
        .dashboard-header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: bold;
        }
        
        .dashboard-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #667eea;
            margin-bottom: 1rem;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .chart-container {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        
        .alert-card {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        .alert-critical {
            background: #f8d7da;
            border-color: #f5c6cb;
        }
        
        .alert-high {
            background: #fff3cd;
            border-color: #ffeaa7;
        }
        
        .alert-medium {
            background: #d1ecf1;
            border-color: #bee5eb;
        }
        
        .alert-low {
            background: #d4edda;
            border-color: #c3e6cb;
        }
        </style>
        """
    
    def _render_sidebar_controls(self):
        """Render sidebar controls for dashboard configuration."""
        with st.sidebar:
            st.markdown("## ⚙️ Dashboard Controls")
            
            # Auto-refresh toggle
            auto_refresh = st.checkbox(
                "🔄 Auto Refresh",
                value=self.dashboard_config['enable_auto_refresh'],
                help="Enable automatic dashboard refresh"
            )
            self.dashboard_config['enable_auto_refresh'] = auto_refresh
            
            # Refresh interval
            refresh_interval = st.slider(
                "Refresh Interval (seconds)",
                min_value=1,
                max_value=60,
                value=int(self.dashboard_config['refresh_interval']),
                help="How often to refresh the dashboard"
            )
            self.dashboard_config['refresh_interval'] = float(refresh_interval)
            
            # Data source selection
            st.markdown("### 📊 Data Sources")
            data_sources = st.multiselect(
                "Select Data Sources",
                ["System Metrics", "User Activity", "Performance Data", "Custom Streams"],
                default=["System Metrics", "Performance Data"]
            )
            
            # Chart type selection
            st.markdown("### 📈 Chart Types")
            chart_types = st.multiselect(
                "Select Chart Types",
                ["Line Chart", "Bar Chart", "Scatter Plot", "Heatmap", "Gauge"],
                default=["Line Chart", "Bar Chart"]
            )
            
            # Manual refresh button
            if st.button("🔄 Refresh Now"):
                st.rerun()
            
            # Dashboard export
            if st.button("📥 Export Dashboard"):
                self._export_dashboard()
    
    def _render_dashboard_content(self):
        """Render the main dashboard content."""
        # System overview metrics
        self._render_system_overview()
        
        # Real-time charts
        self._render_real_time_charts()
        
        # Alerts and notifications
        self._render_alerts()
        
        # Performance metrics
        self._render_performance_metrics()
        
        # Data streams
        self._render_data_streams()
    
    def _render_system_overview(self):
        """Render system overview metrics."""
        st.markdown("## 📊 System Overview")
        
        # Get system metrics
        metrics = self._get_system_metrics()
        
        if "error" not in metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                self._render_metric_card(
                    "CPU Usage",
                    f"{metrics.get('cpu_usage', 0):.1f}%",
                    f"{metrics.get('cpu_usage', 0) - 50:.1f}%" if metrics.get('cpu_usage', 0) > 50 else None
                )
            
            with col2:
                self._render_metric_card(
                    "Memory Usage",
                    f"{metrics.get('memory_usage', 0):.1f}%",
                    f"{metrics.get('memory_usage', 0) - 70:.1f}%" if metrics.get('memory_usage', 0) > 70 else None
                )
            
            with col3:
                self._render_metric_card(
                    "Response Time",
                    f"{metrics.get('avg_response_time', 0):.2f}s",
                    f"{metrics.get('avg_response_time', 0) - 1.0:.2f}s" if metrics.get('avg_response_time', 0) > 1.0 else None
                )
            
            with col4:
                self._render_metric_card(
                    "Active Requests",
                    str(metrics.get('active_requests', 0)),
                    None
                )
        else:
            st.error(f"Error loading system metrics: {metrics['error']}")
    
    def _render_metric_card(self, label: str, value: str, delta: Optional[str] = None):
        """Render a metric card."""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {f'<div style="color: {"red" if delta and float(delta.replace("%", "").replace("s", "")) > 0 else "green"}; font-size: 0.8rem;">{delta}</div>' if delta else ''}
        </div>
        """, unsafe_allow_html=True)
    
    def _render_real_time_charts(self):
        """Render real-time charts."""
        st.markdown("## 📈 Real-Time Charts")
        
        # Create tabs for different chart types
        tab1, tab2, tab3 = st.tabs(["Performance Metrics", "User Activity", "System Health"])
        
        with tab1:
            self._render_performance_chart()
        
        with tab2:
            self._render_user_activity_chart()
        
        with tab3:
            self._render_system_health_chart()
    
    def _render_performance_chart(self):
        """Render performance monitoring chart."""
        performance_data = self._get_performance_data()
        
        if "error" not in performance_data and performance_data.get('data'):
            df = pd.DataFrame(performance_data['data'])
            
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('CPU Usage', 'Memory Usage', 'Response Time'),
                vertical_spacing=0.1
            )
            
            # CPU Usage
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['cpu_usage'],
                    mode='lines+markers',
                    name='CPU %',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
            
            # Memory Usage
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['memory_usage'],
                    mode='lines+markers',
                    name='Memory %',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
            
            # Response Time
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['response_time'],
                    mode='lines+markers',
                    name='Response Time (ms)',
                    line=dict(color='green', width=2)
                ),
                row=3, col=1
            )
            
            fig.update_layout(
                title="Real-Time Performance Monitoring",
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No performance data available")
    
    def _render_user_activity_chart(self):
        """Render user activity chart."""
        # Placeholder for user activity data
        st.info("User activity chart - Data source not configured")
    
    def _render_system_health_chart(self):
        """Render system health chart."""
        # Placeholder for system health data
        st.info("System health chart - Data source not configured")
    
    def _render_alerts(self):
        """Render alerts and notifications."""
        st.markdown("## 🚨 Alerts & Notifications")
        
        alerts = self._get_alerts()
        
        if "error" not in alerts and alerts.get('alerts'):
            for alert in alerts['alerts']:
                severity = alert.get('severity', 'medium')
                alert_class = f"alert-{severity}"
                
                st.markdown(f"""
                <div class="alert-card {alert_class}">
                    <strong>{alert.get('title', 'Alert')}</strong><br>
                    {alert.get('message', 'No message')}<br>
                    <small>Severity: {severity.upper()} | Time: {alert.get('timestamp', 'Unknown')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No active alerts")
    
    def _render_performance_metrics(self):
        """Render detailed performance metrics."""
        st.markdown("## ⚡ Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### System Performance")
            # Add detailed system performance metrics here
            st.info("Detailed system performance metrics")
        
        with col2:
            st.markdown("### Application Performance")
            # Add detailed application performance metrics here
            st.info("Detailed application performance metrics")
    
    def _render_data_streams(self):
        """Render data streams information."""
        st.markdown("## 🌊 Data Streams")
        
        # Get stream metrics
        stream_metrics = self._get_stream_metrics()
        
        if "error" not in stream_metrics:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Processed", stream_metrics.get('total_processed', 0))
            
            with col2:
                st.metric("Throughput", f"{stream_metrics.get('throughput', 0):.2f} msg/s")
            
            with col3:
                st.metric("Avg Latency", f"{stream_metrics.get('avg_latency', 0):.3f}s")
        else:
            st.warning("Stream metrics not available")
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics from API."""
        try:
            response = requests.get(
                f"{self.api_base_url}/monitoring/metrics",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Request Error: {str(e)}"}
    
    def _get_performance_data(self, time_range: str = "1h") -> Dict[str, Any]:
        """Get performance data from API."""
        try:
            response = requests.post(
                f"{self.api_base_url}/monitoring/performance",
                json={"time_range": time_range},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Request Error: {str(e)}"}
    
    def _get_alerts(self) -> Dict[str, Any]:
        """Get alerts from API."""
        try:
            response = requests.get(
                f"{self.api_base_url}/monitoring/alerts",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Request Error: {str(e)}"}
    
    def _get_stream_metrics(self) -> Dict[str, Any]:
        """Get stream metrics from API."""
        try:
            response = requests.get(
                f"{self.api_base_url}/analytics/stream/metrics",
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"API Error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Request Error: {str(e)}"}
    
    def _setup_auto_refresh(self):
        """Setup auto-refresh functionality."""
        # This would be implemented with JavaScript for true auto-refresh
        # For now, we'll use Streamlit's rerun functionality
        pass
    
    def _export_dashboard(self):
        """Export dashboard configuration and data."""
        # Export dashboard configuration
        dashboard_config = {
            'timestamp': datetime.now().isoformat(),
            'config': self.dashboard_config,
            'widgets': list(self.active_widgets.keys())
        }
        
        st.download_button(
            label="📥 Download Dashboard Config",
            data=json.dumps(dashboard_config, indent=2),
            file_name=f"dashboard_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


# Main function to run the dashboard
def main():
    """Main function to run the real-time dashboard."""
    dashboard = RealTimeDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    main()
