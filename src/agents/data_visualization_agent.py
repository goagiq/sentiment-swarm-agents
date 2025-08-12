"""
Data Visualization Agent for creating interactive charts and visualizations.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from loguru import logger

from src.agents.base_agent import StrandsBaseAgent
from src.config.business_intelligence_config import bi_config
from src.core.models import AnalysisRequest, AnalysisResult, DataType, SentimentResult
from src.core.error_handler import with_error_handling


class VisualizationGenerator:
    """Generate interactive data visualizations."""
    
    def __init__(self):
        self.config = bi_config.dashboard
        self.cache = {}
    
    @with_error_handling("visualization_generation")
    async def generate_visualizations(self, data: str, chart_types: List[str] = None, interactive: bool = True) -> Dict[str, Any]:
        """Generate interactive data visualizations."""
        try:
            logger.info(f"Generating visualizations for data with {len(data)} characters")
            
            if chart_types is None:
                chart_types = self.config.default_chart_types
            
            visualizations = {}
            
            for chart_type in chart_types:
                if chart_type == "trend":
                    visualizations["trend"] = await self._create_trend_chart(data)
                elif chart_type == "distribution":
                    visualizations["distribution"] = await self._create_distribution_chart(data)
                elif chart_type == "correlation":
                    visualizations["correlation"] = await self._create_correlation_chart(data)
                elif chart_type == "pie":
                    visualizations["pie"] = await self._create_pie_chart(data)
                elif chart_type == "bar":
                    visualizations["bar"] = await self._create_bar_chart(data)
                elif chart_type == "scatter":
                    visualizations["scatter"] = await self._create_scatter_chart(data)
            
            result = {
                "visualizations": visualizations,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "chart_types": chart_types,
                    "interactive": interactive,
                    "total_charts": len(visualizations)
                }
            }
            
            logger.info(f"Generated {len(visualizations)} visualizations successfully")
            return result
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return {"error": str(e)}
    
    async def _create_trend_chart(self, data: str) -> Dict[str, Any]:
        """Create trend analysis chart."""
        # Sample trend data (replace with actual data processing)
        dates = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]
        values = [10, 15, 13, 18, 20]
        
        fig = px.line(
            x=dates,
            y=values,
            title="Trend Analysis",
            labels={"x": "Time Period", "y": "Value"}
        )
        
        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Value",
            hovermode="x unified"
        )
        
        return fig.to_dict()
    
    async def _create_distribution_chart(self, data: str) -> Dict[str, Any]:
        """Create distribution chart."""
        # Sample distribution data
        categories = ["Category A", "Category B", "Category C", "Category D"]
        values = [25, 30, 20, 25]
        
        fig = px.bar(
            x=categories,
            y=values,
            title="Distribution Analysis",
            labels={"x": "Categories", "y": "Count"}
        )
        
        fig.update_layout(
            xaxis_title="Categories",
            yaxis_title="Count"
        )
        
        return fig.to_dict()
    
    async def _create_correlation_chart(self, data: str) -> Dict[str, Any]:
        """Create correlation chart."""
        # Sample correlation data
        x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y_values = [2, 4, 5, 4, 5, 7, 8, 9, 10, 12]
        
        fig = px.scatter(
            x=x_values,
            y=y_values,
            title="Correlation Analysis",
            labels={"x": "Variable X", "y": "Variable Y"}
        )
        
        # Add trend line
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=[1.2 * x + 0.5 for x in x_values],
                mode="lines",
                name="Trend Line"
            )
        )
        
        fig.update_layout(
            xaxis_title="Variable X",
            yaxis_title="Variable Y"
        )
        
        return fig.to_dict()
    
    async def _create_pie_chart(self, data: str) -> Dict[str, Any]:
        """Create pie chart."""
        # Sample pie chart data
        labels = ["Positive", "Neutral", "Negative"]
        values = [60, 25, 15]
        
        fig = px.pie(
            values=values,
            names=labels,
            title="Sentiment Distribution"
        )
        
        return fig.to_dict()
    
    async def _create_bar_chart(self, data: str) -> Dict[str, Any]:
        """Create bar chart."""
        # Sample bar chart data
        categories = ["Metric 1", "Metric 2", "Metric 3", "Metric 4"]
        values = [85, 92, 78, 88]
        
        fig = px.bar(
            x=categories,
            y=values,
            title="Performance Metrics",
            labels={"x": "Metrics", "y": "Score"}
        )
        
        fig.update_layout(
            xaxis_title="Metrics",
            yaxis_title="Score"
        )
        
        return fig.to_dict()
    
    async def _create_scatter_chart(self, data: str) -> Dict[str, Any]:
        """Create scatter chart."""
        # Sample scatter data
        x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y_values = [2, 4, 5, 4, 5, 7, 8, 9, 10, 12]
        categories = ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]
        
        fig = px.scatter(
            x=x_values,
            y=y_values,
            color=categories,
            title="Scatter Analysis",
            labels={"x": "X Axis", "y": "Y Axis"}
        )
        
        fig.update_layout(
            xaxis_title="X Axis",
            yaxis_title="Y Axis"
        )
        
        return fig.to_dict()


class DataVisualizationAgent(StrandsBaseAgent):
    """
    Data Visualization Agent for creating interactive charts and visualizations.
    
    Supports:
    - Interactive data visualizations
    - Multiple chart types (trend, distribution, correlation, pie, bar, scatter)
    - Export capabilities
    - Customizable themes and layouts
    """
    
    def __init__(self, model_name: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_name or "mistral-small3.1:latest", **kwargs)
        
        # Initialize visualization components
        self.visualization_generator = VisualizationGenerator()
        
        # Set metadata
        self.metadata["agent_type"] = "data_visualization"
        self.metadata["capabilities"] = [
            "interactive_visualizations",
            "multiple_chart_types",
            "data_analysis",
            "export_capabilities"
        ]
        self.metadata["supported_chart_types"] = [
            "trend", "distribution", "correlation", "pie", "bar", "scatter"
        ]
        self.metadata["supported_formats"] = ["json", "html", "png", "svg"]
        
        logger.info("DataVisualizationAgent initialized successfully")
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        # Data visualization agent can process text and general requests
        return request.data_type in [DataType.TEXT, DataType.GENERAL]
    
    @with_error_handling("data_visualization_processing")
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process data visualization requests."""
        try:
            logger.info(f"Processing data visualization request: {request.data_type}")
            
            start_time = datetime.now()
            
            # Extract visualization parameters from metadata
            chart_types = request.metadata.get("chart_types", ["trend", "distribution", "correlation"])
            interactive = request.metadata.get("interactive", True)
            
            # Generate visualizations
            result = await self.visualization_generator.generate_visualizations(
                request.content,
                chart_types,
                interactive
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                status="completed",
                sentiment=SentimentResult(label="neutral", confidence=0.5, reasoning="Visualizations generated successfully"),
                extracted_text="",
                metadata=result,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Data visualization processing failed: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                status="failed",
                sentiment=SentimentResult(label="neutral", confidence=0.0, reasoning=f"Processing failed: {str(e)}"),
                metadata={"error": str(e)},
                processing_time=0.0
            )
    
    async def generate_visualizations(self, data: str, chart_types: List[str] = None, interactive: bool = True) -> Dict[str, Any]:
        """Generate interactive data visualizations."""
        return await self.visualization_generator.generate_visualizations(data, chart_types, interactive)
    
    async def create_trend_chart(self, data: str) -> Dict[str, Any]:
        """Create trend analysis chart."""
        return await self.visualization_generator._create_trend_chart(data)
    
    async def create_distribution_chart(self, data: str) -> Dict[str, Any]:
        """Create distribution chart."""
        return await self.visualization_generator._create_distribution_chart(data)
    
    async def create_correlation_chart(self, data: str) -> Dict[str, Any]:
        """Create correlation chart."""
        return await self.visualization_generator._create_correlation_chart(data)
