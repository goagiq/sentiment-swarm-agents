"""
Predictive Analytics Dashboard

Provides advanced predictive analytics capabilities including:
- Forecasting visualizations
- Trend analysis
- Scenario modeling
- Performance metrics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List, Optional

# API configuration
API_BASE_URL = "http://localhost:8003"


def get_forecast_data(metric: str, periods: int = 12) -> Dict[str, Any]:
    """Get forecast data from API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predictive/forecast",
            json={
                "metric": metric,
                "forecast_periods": periods,
                "model_type": "ensemble"
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def get_trend_analysis(data_type: str, time_range: str = "30d") -> Dict[str, Any]:
    """Get trend analysis data from API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predictive/trends",
            json={
                "data_type": data_type,
                "time_range": time_range
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def get_scenario_data(scenario_name: str) -> Dict[str, Any]:
    """Get scenario analysis data from API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/scenario/analyze",
            json={
                "scenario_name": scenario_name
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Request Error: {str(e)}"}


def create_forecast_chart(historical_data: List[Dict], forecast_data: List[Dict]) -> go.Figure:
    """Create a forecast visualization chart."""
    # Prepare data
    hist_df = pd.DataFrame(historical_data)
    forecast_df = pd.DataFrame(forecast_data)
    
    # Create figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=hist_df['timestamp'],
        y=hist_df['value'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Add forecast data
    fig.add_trace(go.Scatter(
        x=forecast_df['timestamp'],
        y=forecast_df['forecast'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Add confidence intervals if available
    if 'lower_bound' in forecast_df.columns and 'upper_bound' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['timestamp'],
            y=forecast_df['upper_bound'],
            mode='lines',
            name='Upper Bound',
            line=dict(color='red', width=1, dash='dot'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['timestamp'],
            y=forecast_df['lower_bound'],
            mode='lines',
            fill='tonexty',
            name='Confidence Interval',
            line=dict(color='red', width=1, dash='dot'),
            fillcolor='rgba(255,0,0,0.1)'
        ))
    
    # Update layout
    fig.update_layout(
        title="Forecast Analysis",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_trend_chart(trend_data: List[Dict]) -> go.Figure:
    """Create a trend analysis chart."""
    df = pd.DataFrame(trend_data)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Trend Analysis', 'Seasonal Decomposition'),
        vertical_spacing=0.1
    )
    
    # Main trend
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['value'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Trend line
    if 'trend' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['trend'],
                mode='lines',
                name='Trend',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
    
    # Seasonal component
    if 'seasonal' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['seasonal'],
                mode='lines',
                name='Seasonal',
                line=dict(color='green')
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        title="Trend Analysis",
        height=600,
        showlegend=True
    )
    
    return fig


def create_scenario_comparison(scenarios: List[Dict]) -> go.Figure:
    """Create a scenario comparison chart."""
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, scenario in enumerate(scenarios):
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=scenario['timeline'],
            y=scenario['values'],
            mode='lines+markers',
            name=scenario['name'],
            line=dict(color=color, width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Scenario Comparison",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified'
    )
    
    return fig


def display_forecasting_section():
    """Display the forecasting section."""
    st.markdown("## 📈 Forecasting Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Metric selection
        metric = st.selectbox(
            "Select Metric to Forecast:",
            ["sentiment_score", "request_volume", "response_time", "accuracy", "user_satisfaction"]
        )
        
        # Forecast periods
        periods = st.slider("Forecast Periods:", 1, 24, 12)
        
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Generating forecast..."):
                forecast_result = get_forecast_data(metric, periods)
                
                if "error" not in forecast_result:
                    # Create and display chart
                    fig = create_forecast_chart(
                        forecast_result.get("historical_data", []),
                        forecast_result.get("forecast_data", [])
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display metrics
                    metrics = forecast_result.get("metrics", {})
                    if metrics:
                        st.markdown("### 📊 Forecast Metrics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
                        
                        with col2:
                            st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                        
                        with col3:
                            st.metric("Confidence", f"{metrics.get('confidence', 0):.2f}%")
                else:
                    st.error(f"Error generating forecast: {forecast_result['error']}")
    
    with col2:
        st.markdown("### 📋 Forecast Options")
        st.markdown("""
        **Available Metrics:**
        - **Sentiment Score**: Overall sentiment trends
        - **Request Volume**: System usage patterns
        - **Response Time**: Performance trends
        - **Accuracy**: Model accuracy over time
        - **User Satisfaction**: User feedback trends
        """)
        
        st.markdown("### 🔧 Model Settings")
        model_type = st.selectbox(
            "Model Type:",
            ["ensemble", "arima", "prophet", "lstm"]
        )
        
        confidence_level = st.slider("Confidence Level:", 0.8, 0.99, 0.95)


def display_trend_analysis_section():
    """Display the trend analysis section."""
    st.markdown("## 📊 Trend Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Data type selection
        data_type = st.selectbox(
            "Select Data Type:",
            ["sentiment", "performance", "usage", "quality"]
        )
        
        # Time range selection
        time_range = st.selectbox(
            "Time Range:",
            ["7d", "30d", "90d", "180d", "365d"]
        )
        
        if st.button("Analyze Trends", type="primary"):
            with st.spinner("Analyzing trends..."):
                trend_result = get_trend_analysis(data_type, time_range)
                
                if "error" not in trend_result:
                    # Create and display chart
                    fig = create_trend_chart(trend_result.get("trend_data", []))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display insights
                    insights = trend_result.get("insights", [])
                    if insights:
                        st.markdown("### 💡 Trend Insights")
                        for insight in insights:
                            st.info(insight)
                else:
                    st.error(f"Error analyzing trends: {trend_result['error']}")
    
    with col2:
        st.markdown("### 📈 Analysis Types")
        st.markdown("""
        **Trend Analysis:**
        - **Linear Trend**: Overall direction
        - **Seasonal Patterns**: Cyclical behavior
        - **Anomaly Detection**: Outlier identification
        - **Change Points**: Significant shifts
        """)
        
        st.markdown("### 📊 Statistical Measures")
        st.markdown("""
        - **Trend Strength**: How strong the trend is
        - **Seasonality**: Cyclical patterns
        - **Volatility**: Data variability
        - **Correlation**: Relationships between variables
        """)


def display_scenario_analysis_section():
    """Display the scenario analysis section."""
    st.markdown("## 🎯 Scenario Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Scenario selection
        scenario_name = st.selectbox(
            "Select Scenario:",
            ["baseline", "optimistic", "pessimistic", "market_expansion", "cost_reduction"]
        )
        
        if st.button("Run Scenario Analysis", type="primary"):
            with st.spinner("Running scenario analysis..."):
                scenario_result = get_scenario_data(scenario_name)
                
                if "error" not in scenario_result:
                    # Create and display chart
                    scenarios = scenario_result.get("scenarios", [])
                    if scenarios:
                        fig = create_scenario_comparison(scenarios)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display scenario details
                    details = scenario_result.get("details", {})
                    if details:
                        st.markdown("### 📋 Scenario Details")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Risk Score", f"{details.get('risk_score', 0):.2f}")
                            st.metric("Expected Value", f"{details.get('expected_value', 0):.2f}")
                        
                        with col2:
                            st.metric("Confidence", f"{details.get('confidence', 0):.2f}%")
                            st.metric("Time Horizon", f"{details.get('time_horizon', 0)} months")
                        
                        # Assumptions
                        assumptions = details.get("assumptions", [])
                        if assumptions:
                            st.markdown("### 📝 Key Assumptions")
                            for assumption in assumptions:
                                st.write(f"• {assumption}")
                else:
                    st.error(f"Error running scenario: {scenario_result['error']}")
    
    with col2:
        st.markdown("### 🎲 Scenario Types")
        st.markdown("""
        **Available Scenarios:**
        - **Baseline**: Current trajectory
        - **Optimistic**: Best-case scenario
        - **Pessimistic**: Worst-case scenario
        - **Market Expansion**: Growth opportunities
        - **Cost Reduction**: Efficiency improvements
        """)
        
        st.markdown("### 📊 Impact Metrics")
        st.markdown("""
        - **Financial Impact**: Revenue/cost changes
        - **Operational Impact**: Process improvements
        - **Risk Assessment**: Potential risks
        - **Timeline**: Implementation schedule
        """)


def display_performance_metrics():
    """Display performance metrics dashboard."""
    st.markdown("## ⚡ Performance Metrics")
    
    # Create sample data for demonstration
    metrics_data = {
        "System Performance": {
            "Response Time": 0.85,
            "Throughput": 1250,
            "Error Rate": 0.02,
            "Availability": 99.8
        },
        "Model Performance": {
            "Accuracy": 94.5,
            "Precision": 92.3,
            "Recall": 89.7,
            "F1 Score": 90.9
        },
        "User Metrics": {
            "Active Users": 1250,
            "Session Duration": 8.5,
            "Bounce Rate": 0.15,
            "Satisfaction": 4.6
        }
    }
    
    # Display metrics in cards
    for category, metrics in metrics_data.items():
        st.markdown(f"### {category}")
        cols = st.columns(len(metrics))
        
        for i, (metric_name, value) in enumerate(metrics.items()):
            with cols[i]:
                if isinstance(value, float):
                    if value < 1:
                        st.metric(metric_name, f"{value:.3f}")
                    else:
                        st.metric(metric_name, f"{value:.1f}")
                else:
                    st.metric(metric_name, value)


def main():
    """Main predictive analytics dashboard."""
    st.set_page_config(
        page_title="Predictive Analytics Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown('<h1 class="main-header">📈 Predictive Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Forecasting", 
        "📊 Trend Analysis", 
        "🎯 Scenarios", 
        "⚡ Performance"
    ])
    
    with tab1:
        display_forecasting_section()
    
    with tab2:
        display_trend_analysis_section()
    
    with tab3:
        display_scenario_analysis_section()
    
    with tab4:
        display_performance_metrics()


if __name__ == "__main__":
    main()
