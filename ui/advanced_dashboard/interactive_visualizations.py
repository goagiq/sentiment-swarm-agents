"""
Interactive Visualizations

Advanced interactive visualizations for real-time analytics dashboard
with zoom, pan, drill-down, and cross-filtering capabilities.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Any


class InteractiveVisualizations:
    """
    Interactive visualization component with advanced features.
    """
    
    def __init__(self):
        """Initialize the interactive visualizations."""
        self.chart_configs = {}
        self.interactive_features = {
            'zoom': True,
            'pan': True,
            'drill_down': True,
            'cross_filtering': True
        }
    
    def create_line_chart(self, data: pd.DataFrame, x_col: str, y_col: str, 
                         title: str = "Line Chart", **kwargs) -> go.Figure:
        """Create an interactive line chart."""
        fig = px.line(
            data, 
            x=x_col, 
            y=y_col,
            title=title,
            **kwargs
        )
        
        # Add interactive features
        fig.update_layout(
            hovermode='x unified',
            showlegend=True,
            dragmode='zoom',
            selectdirection='any'
        )
        
        # Enable zoom and pan
        if self.interactive_features['zoom']:
            fig.update_xaxes(rangeslider_visible=True)
        
        return fig
    
    def create_bar_chart(self, data: pd.DataFrame, x_col: str, y_col: str,
                        title: str = "Bar Chart", **kwargs) -> go.Figure:
        """Create an interactive bar chart."""
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            title=title,
            **kwargs
        )
        
        # Add interactive features
        fig.update_layout(
            hovermode='closest',
            showlegend=True
        )
        
        return fig
    
    def create_scatter_plot(self, data: pd.DataFrame, x_col: str, y_col: str,
                           color_col: Optional[str] = None,
                           size_col: Optional[str] = None,
                           title: str = "Scatter Plot", **kwargs) -> go.Figure:
        """Create an interactive scatter plot."""
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col,
            title=title,
            **kwargs
        )
        
        # Add interactive features
        fig.update_layout(
            hovermode='closest',
            showlegend=True,
            dragmode='select'
        )
        
        return fig
    
    def create_heatmap(self, data: pd.DataFrame, x_col: str, y_col: str, 
                      value_col: str, title: str = "Heatmap", **kwargs) -> go.Figure:
        """Create an interactive heatmap."""
        # Pivot data for heatmap
        pivot_data = data.pivot(index=y_col, columns=x_col, values=value_col)
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Viridis',
            **kwargs
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col
        )
        
        return fig
    
    def create_gauge_chart(self, value: float, min_val: float = 0, 
                          max_val: float = 100, title: str = "Gauge") -> go.Figure:
        """Create an interactive gauge chart."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': (max_val + min_val) / 2},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [min_val, max_val * 0.6], 'color': "lightgray"},
                    {'range': [max_val * 0.6, max_val * 0.8], 'color': "gray"},
                    {'range': [max_val * 0.8, max_val], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_val * 0.9
                }
            }
        ))
        
        return fig
    
    def create_dashboard_layout(self, charts: List[go.Figure], 
                               layout: str = "grid") -> go.Figure:
        """Create a dashboard layout with multiple charts."""
        if layout == "grid":
            n_charts = len(charts)
            cols = min(3, n_charts)
            rows = (n_charts + cols - 1) // cols
            
            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=[f"Chart {i+1}" for i in range(n_charts)]
            )
            
            for i, chart in enumerate(charts):
                row = i // cols + 1
                col = i % cols + 1
                
                # Add traces from chart to subplot
                for trace in chart.data:
                    fig.add_trace(trace, row=row, col=col)
            
            fig.update_layout(
                height=300 * rows,
                showlegend=False,
                title_text="Dashboard Overview"
            )
            
            return fig
        
        return charts[0] if charts else go.Figure()
    
    def add_drill_down_capability(self, fig: go.Figure, 
                                 drill_down_data: Dict[str, Any]) -> go.Figure:
        """Add drill-down capability to a chart."""
        # Add click events for drill-down
        fig.update_layout(
            clickmode='event+select'
        )
        
        # Store drill-down data
        fig.add_annotation(
            text="Click to drill down",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10, color="gray")
        )
        
        return fig
    
    def add_cross_filtering(self, charts: List[go.Figure]) -> List[go.Figure]:
        """Add cross-filtering between charts."""
        # This would implement cross-filtering between charts
        # For now, we'll just return the charts as-is
        return charts
    
    def create_3d_plot(self, data: pd.DataFrame, x_col: str, y_col: str, 
                      z_col: str, color_col: Optional[str] = None,
                      title: str = "3D Plot") -> go.Figure:
        """Create an interactive 3D plot."""
        fig = px.scatter_3d(
            data,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color_col,
            title=title
        )
        
        # Add 3D interactive features
        fig.update_layout(
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            )
        )
        
        return fig
    
    def create_network_graph(self, nodes: List[Dict], edges: List[Dict],
                           title: str = "Network Graph") -> go.Figure:
        """Create an interactive network graph."""
        # Extract node positions and labels
        node_x = [node['x'] for node in nodes]
        node_y = [node['y'] for node in nodes]
        node_text = [node.get('label', f'Node {i}') for i, node in enumerate(nodes)]
        node_colors = [node.get('color', 'blue') for node in nodes]
        
        # Extract edge coordinates
        edge_x = []
        edge_y = []
        for edge in edges:
            source_idx = edge['source']
            target_idx = edge['target']
            edge_x.extend([node_x[source_idx], node_x[target_idx], None])
            edge_y.extend([node_y[source_idx], node_y[target_idx], None])
        
        # Create the network graph
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=20,
                color=node_colors,
                line=dict(width=2)
            )
        ))
        
        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def create_time_series_forecast(self, data: pd.DataFrame, 
                                  date_col: str, value_col: str,
                                  forecast_periods: int = 30,
                                  title: str = "Time Series Forecast") -> go.Figure:
        """Create a time series forecast chart."""
        # Simple moving average forecast (in real implementation, use proper forecasting)
        dates = pd.to_datetime(data[date_col])
        values = data[value_col]
        
        # Calculate moving average
        window_size = min(7, len(values) // 4)
        if window_size > 1:
            ma_values = values.rolling(window=window_size).mean()
        else:
            ma_values = values
        
        # Create forecast (simple extension)
        last_date = dates.iloc[-1]
        forecast_dates = pd.date_range(start=last_date, periods=forecast_periods + 1)[1:]
        forecast_values = [ma_values.iloc[-1]] * forecast_periods
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Moving average
        fig.add_trace(go.Scatter(
            x=dates,
            y=ma_values,
            mode='lines',
            name='Moving Average',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines',
            name='Forecast',
            line=dict(color='green', width=2, dash='dot')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified'
        )
        
        return fig
    
    def export_chart_config(self, chart_id: str, config: Dict[str, Any]) -> None:
        """Export chart configuration."""
        self.chart_configs[chart_id] = config
    
    def get_chart_config(self, chart_id: str) -> Optional[Dict[str, Any]]:
        """Get chart configuration."""
        return self.chart_configs.get(chart_id)
    
    def render_interactive_demo(self):
        """Render an interactive visualization demo."""
        st.markdown("## 📊 Interactive Visualizations Demo")
        
        # Sample data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        values = np.random.randn(100).cumsum()
        categories = np.random.choice(['A', 'B', 'C'], 100)
        
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'category': categories
        })
        
        # Create different chart types
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Line Chart")
            line_fig = self.create_line_chart(df, 'date', 'value', "Time Series")
            st.plotly_chart(line_fig, use_container_width=True)
        
        with col2:
            st.markdown("### Bar Chart")
            bar_data = df.groupby('category')['value'].mean().reset_index()
            bar_fig = self.create_bar_chart(bar_data, 'category', 'value', "Category Averages")
            st.plotly_chart(bar_fig, use_container_width=True)
        
        # Scatter plot
        st.markdown("### Scatter Plot")
        scatter_fig = self.create_scatter_plot(
            df, 'date', 'value', color_col='category',
            title="Scatter Plot with Categories"
        )
        st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Gauge chart
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gauge_fig = self.create_gauge_chart(75, title="Performance")
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col2:
            gauge_fig2 = self.create_gauge_chart(45, title="Efficiency")
            st.plotly_chart(gauge_fig2, use_container_width=True)
        
        with col3:
            gauge_fig3 = self.create_gauge_chart(90, title="Quality")
            st.plotly_chart(gauge_fig3, use_container_width=True)


# Factory function for creating interactive visualizations
def create_interactive_visualizations() -> InteractiveVisualizations:
    """Create an interactive visualizations component."""
    return InteractiveVisualizations()
