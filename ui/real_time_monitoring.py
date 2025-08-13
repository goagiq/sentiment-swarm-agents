"""
Real-Time Monitoring Dashboard

Provides real-time monitoring capabilities including:
- Live metrics display
- System performance monitoring
- Alert management
- Performance dashboards
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time
from typing import Dict, Any, List, Optional

# API configuration
API_BASE_URL = "http://localhost:8003"


def get_system_metrics() -> Dict[str, Any]:
    """Get real-time system metrics from API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/monitoring/metrics",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def get_performance_data(time_range: str = "1h") -> Dict[str, Any]:
    """Get performance data from API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/monitoring/performance",
            json={"time_range": time_range},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def get_alerts() -> Dict[str, Any]:
    """Get current alerts from API."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/monitoring/alerts",
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def create_performance_chart(performance_data: List[Dict]) -> go.Figure:
    """Create a performance monitoring chart."""
    df = pd.DataFrame(performance_data)
    
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
        title="System Performance Monitoring",
        height=600,
        showlegend=True
    )
    
    return fig


def create_alert_dashboard(alerts: List[Dict]) -> go.Figure:
    """Create an alert dashboard visualization."""
    if not alerts:
        # Create empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No active alerts",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title="Active Alerts")
        return fig
    
    # Group alerts by severity
    severity_counts = {}
    for alert in alerts:
        severity = alert.get('severity', 'Unknown')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(severity_counts.keys()),
            y=list(severity_counts.values()),
            marker_color=['red', 'orange', 'yellow', 'green']
        )
    ])
    
    fig.update_layout(
        title="Active Alerts by Severity",
        xaxis_title="Severity",
        yaxis_title="Count"
    )
    
    return fig


def display_system_overview():
    """Display system overview metrics."""
    st.markdown("## 📊 System Overview")
    
    # Auto-refresh every 30 seconds
    if st.button("🔄 Refresh Now"):
        st.rerun()
    
    # Get current metrics
    metrics = get_system_metrics()
    
    if "error" not in metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_usage = metrics.get('cpu_usage', 0)
            st.metric(
                "CPU Usage", 
                f"{cpu_usage:.1f}%",
                delta=f"{cpu_usage - 50:.1f}%" if cpu_usage > 50 else None
            )
        
        with col2:
            memory_usage = metrics.get('memory_usage', 0)
            st.metric(
                "Memory Usage", 
                f"{memory_usage:.1f}%",
                delta=f"{memory_usage - 70:.1f}%" if memory_usage > 70 else None
            )
        
        with col3:
            response_time = metrics.get('avg_response_time', 0)
            st.metric(
                "Avg Response Time", 
                f"{response_time:.2f}s",
                delta=f"{response_time - 1.0:.2f}s" if response_time > 1.0 else None
            )
        
        with col4:
            active_requests = metrics.get('active_requests', 0)
            st.metric(
                "Active Requests", 
                active_requests,
                delta=active_requests - 10 if active_requests > 10 else None
            )
        
        # System status
        system_status = metrics.get('system_status', 'Unknown')
        if system_status == 'healthy':
            st.success("✅ System Status: Healthy")
        elif system_status == 'warning':
            st.warning("⚠️ System Status: Warning")
        elif system_status == 'critical':
            st.error("🚨 System Status: Critical")
        else:
            st.info(f"ℹ️ System Status: {system_status}")
    else:
        st.error(f"Error fetching metrics: {metrics['error']}")


def display_performance_monitoring():
    """Display performance monitoring section."""
    st.markdown("## ⚡ Performance Monitoring")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Time range selection
        time_range = st.selectbox(
            "Time Range:",
            ["15m", "30m", "1h", "6h", "24h"],
            index=2
        )
        
        if st.button("Update Performance Data", type="primary"):
            with st.spinner("Fetching performance data..."):
                performance_data = get_performance_data(time_range)
                
                if "error" not in performance_data:
                    # Create and display chart
                    data = performance_data.get("performance_data", [])
                    if data:
                        fig = create_performance_chart(data)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No performance data available for the selected time range.")
                else:
                    st.error(f"Error fetching performance data: {performance_data['error']}")
    
    with col2:
        st.markdown("### 📈 Performance Metrics")
        st.markdown("""
        **Key Metrics:**
        - **CPU Usage**: System processor utilization
        - **Memory Usage**: RAM consumption
        - **Response Time**: API response latency
        - **Throughput**: Requests per second
        - **Error Rate**: Failed request percentage
        """)
        
        st.markdown("### 🎯 Thresholds")
        st.markdown("""
        **Alert Thresholds:**
        - **CPU**: > 80% (Warning), > 90% (Critical)
        - **Memory**: > 85% (Warning), > 95% (Critical)
        - **Response Time**: > 2s (Warning), > 5s (Critical)
        - **Error Rate**: > 5% (Warning), > 10% (Critical)
        """)


def display_alert_management():
    """Display alert management section."""
    st.markdown("## 🚨 Alert Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get current alerts
        alerts_result = get_alerts()
        
        if "error" not in alerts_result:
            alerts = alerts_result.get("alerts", [])
            
            if alerts:
                # Create alert dashboard
                fig = create_alert_dashboard(alerts)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display alert details
                st.markdown("### 📋 Alert Details")
                
                for alert in alerts:
                    severity = alert.get('severity', 'Unknown')
                    if severity == 'critical':
                        st.error(f"🚨 **{alert.get('title', 'Alert')}**")
                    elif severity == 'warning':
                        st.warning(f"⚠️ **{alert.get('title', 'Alert')}**")
                    elif severity == 'info':
                        st.info(f"ℹ️ **{alert.get('title', 'Alert')}**")
                    else:
                        st.write(f"📌 **{alert.get('title', 'Alert')}**")
                    
                    st.write(f"**Description**: {alert.get('description', 'No description')}")
                    st.write(f"**Time**: {alert.get('timestamp', 'Unknown')}")
                    st.write(f"**Source**: {alert.get('source', 'Unknown')}")
                    st.write("---")
            else:
                st.success("✅ No active alerts")
        else:
            st.error(f"Error fetching alerts: {alerts_result['error']}")
    
    with col2:
        st.markdown("### 🎛️ Alert Configuration")
        st.markdown("""
        **Alert Types:**
        - **System Alerts**: Hardware/software issues
        - **Performance Alerts**: Performance degradation
        - **Security Alerts**: Security threats
        - **Business Alerts**: Business rule violations
        """)
        
        st.markdown("### 📧 Notification Channels")
        st.markdown("""
        **Channels:**
        - **Email**: Critical alerts
        - **SMS**: Emergency alerts
        - **Slack**: Team notifications
        - **Dashboard**: Real-time display
        """)


def display_agent_monitoring():
    """Display agent monitoring section."""
    st.markdown("## 🤖 Agent Monitoring")
    
    # Get agent status
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=10)
        if response.status_code == 200:
            status = response.json()
            agents = status.get("agents", [])
            
            if agents:
                # Create agent status table
                agent_data = []
                for agent in agents:
                    agent_data.append({
                        "Agent ID": agent["agent_id"],
                        "Type": agent["agent_type"],
                        "Status": agent["status"],
                        "Load": f"{agent['current_load']}/{agent['max_capacity']}",
                        "Last Heartbeat": agent["last_heartbeat"]
                    })
                
                st.dataframe(agent_data, use_container_width=True)
                
                # Agent performance chart
                st.markdown("### 📊 Agent Performance")
                
                # Create sample agent performance data
                agent_performance = []
                for agent in agents:
                    agent_performance.append({
                        "Agent": agent["agent_type"],
                        "Load %": (agent["current_load"] / agent["max_capacity"]) * 100,
                        "Status": 1 if agent["status"] == "active" else 0
                    })
                
                df = pd.DataFrame(agent_performance)
                
                fig = px.bar(
                    df, 
                    x="Agent", 
                    y="Load %",
                    color="Status",
                    title="Agent Load Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No agent information available")
        else:
            st.error("Unable to retrieve agent status")
    except Exception as e:
        st.error(f"Error fetching agent status: {e}")


def main():
    """Main real-time monitoring dashboard."""
    st.set_page_config(
        page_title="Real-Time Monitoring",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown('<h1 class="main-header">⚡ Real-Time Monitoring Dashboard</h1>', unsafe_allow_html=True)
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview", 
        "⚡ Performance", 
        "🚨 Alerts", 
        "🤖 Agents"
    ])
    
    with tab1:
        display_system_overview()
    
    with tab2:
        display_performance_monitoring()
    
    with tab3:
        display_alert_management()
    
    with tab4:
        display_agent_monitoring()


if __name__ == "__main__":
    main()
