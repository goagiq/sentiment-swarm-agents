"""
Trend Analysis Engine

This module provides trend analysis capabilities including:
- Trend direction and strength analysis
- Trend pattern classification
- Trend forecasting
- Trend comparison and correlation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
from loguru import logger

from src.core.error_handler import with_error_handling


class TrendEngine:
    """
    Analyzes trends in data including direction, strength, and patterns.
    """
    
    def __init__(self):
        self.trend_cache = {}
        self.analysis_config = {
            "min_data_points": 5,
            "trend_threshold": 0.1,
            "confidence_threshold": 0.7,
            "forecast_periods": 10,
            "smoothing_window": 3
        }
        
        logger.info("TrendEngine initialized successfully")
    
    @with_error_handling("trend_analysis")
    async def analyze_trends(
        self, 
        data: List[Dict[str, Any]], 
        time_field: str = "timestamp",
        value_field: str = "value"
    ) -> Dict[str, Any]:
        """
        Analyze trends in the provided data.
        
        Args:
            data: List of data points with timestamps
            time_field: Field name containing timestamp
            value_field: Field name containing values to analyze
            
        Returns:
            Dictionary containing trend analysis results
        """
        try:
            logger.info(f"Analyzing trends for {len(data)} data points")
            
            # Convert to pandas DataFrame
            df = self._prepare_dataframe(data, time_field, value_field)
            
            if len(df) < self.analysis_config["min_data_points"]:
                return {
                    "error": f"Insufficient data for trend analysis. Need at least {self.analysis_config['min_data_points']} points"
                }
            
            # Perform trend analysis
            results = {
                "linear_trend": await self._analyze_linear_trend(df),
                "moving_average_trend": await self._analyze_moving_average_trend(df),
                "trend_strength": await self._calculate_trend_strength(df),
                "trend_patterns": await self._identify_trend_patterns(df),
                "trend_forecast": await self._forecast_trend(df),
                "trend_comparison": await self._compare_trends(df),
                "metadata": {
                    "total_data_points": len(df),
                    "time_range": {
                        "start": df[time_field].min().isoformat(),
                        "end": df[time_field].max().isoformat()
                    },
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info("Trend analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {"error": str(e)}
    
    def _prepare_dataframe(self, data: List[Dict[str, Any]], time_field: str, value_field: str) -> pd.DataFrame:
        """Prepare data for analysis by converting to pandas DataFrame."""
        df = pd.DataFrame(data)
        
        # Convert timestamp field to datetime
        if time_field in df.columns:
            df[time_field] = pd.to_datetime(df[time_field])
            df = df.sort_values(time_field)
        
        return df
    
    async def _analyze_linear_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze linear trend in the data."""
        try:
            values = df.iloc[:, 1].values  # Assuming second column is the value
            
            if len(values) < 2:
                return {"error": "Insufficient data for linear trend analysis"}
            
            # Calculate linear trend
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            
            # Calculate trend statistics
            trend_values = slope * x + intercept
            residuals = values - trend_values
            r_squared = 1 - (np.sum(residuals**2) / np.sum((values - np.mean(values))**2))
            
            # Determine trend direction
            if slope > self.analysis_config["trend_threshold"]:
                trend_direction = "increasing"
            elif slope < -self.analysis_config["trend_threshold"]:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
            
            return {
                "trend_direction": trend_direction,
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_squared),
                "trend_strength": abs(slope) / (np.std(values) + 1e-8),
                "confidence": min(0.95, r_squared),
                "trend_type": "linear"
            }
            
        except Exception as e:
            logger.error(f"Linear trend analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_moving_average_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend using moving averages."""
        try:
            values = df.iloc[:, 1].values
            
            if len(values) < self.analysis_config["smoothing_window"] * 2:
                return {"error": "Insufficient data for moving average analysis"}
            
            # Calculate moving averages
            short_ma = self._moving_average(values, self.analysis_config["smoothing_window"])
            long_ma = self._moving_average(values, self.analysis_config["smoothing_window"] * 2)
            
            # Analyze moving average crossover
            ma_crossover = short_ma - long_ma
            
            # Determine trend based on moving average relationship
            recent_crossover = ma_crossover[-10:] if len(ma_crossover) >= 10 else ma_crossover
            avg_crossover = np.mean(recent_crossover)
            
            if avg_crossover > 0:
                trend_direction = "increasing"
            elif avg_crossover < 0:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
            
            return {
                "trend_direction": trend_direction,
                "short_ma": short_ma.tolist(),
                "long_ma": long_ma.tolist(),
                "ma_crossover": ma_crossover.tolist(),
                "crossover_strength": abs(avg_crossover) / (np.std(values) + 1e-8),
                "trend_type": "moving_average"
            }
            
        except Exception as e:
            logger.error(f"Moving average trend analysis failed: {e}")
            return {"error": str(e)}
    
    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average with specified window size."""
        if window >= len(data):
            return np.full_like(data, np.mean(data))
        
        # Pad the data for edge handling
        padded = np.pad(data, (window//2, window//2), mode='edge')
        
        # Calculate moving average
        result = np.convolve(padded, np.ones(window)/window, mode='valid')
        
        return result
    
    async def _calculate_trend_strength(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate the overall strength of the trend."""
        try:
            values = df.iloc[:, 1].values
            
            # Calculate trend strength using multiple methods
            linear_trend = await self._analyze_linear_trend(df)
            ma_trend = await self._analyze_moving_average_trend(df)
            
            # Combine trend strength indicators
            linear_strength = linear_trend.get("trend_strength", 0.0)
            ma_strength = ma_trend.get("crossover_strength", 0.0)
            
            # Calculate overall trend strength
            overall_strength = (linear_strength + ma_strength) / 2
            
            return {
                "overall_strength": float(overall_strength),
                "linear_strength": float(linear_strength),
                "ma_strength": float(ma_strength),
                "strength_category": "strong" if overall_strength > 0.5 else "moderate" if overall_strength > 0.2 else "weak",
                "confidence": min(0.9, overall_strength)
            }
            
        except Exception as e:
            logger.error(f"Trend strength calculation failed: {e}")
            return {"error": str(e)}
    
    async def _identify_trend_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify specific trend patterns in the data."""
        try:
            values = df.iloc[:, 1].values
            patterns = []
            
            # Detect trend reversals
            reversals = self._detect_trend_reversals(values)
            if reversals:
                patterns.append({
                    "pattern_type": "trend_reversal",
                    "count": len(reversals),
                    "positions": reversals,
                    "significance": "high" if len(reversals) > 2 else "medium"
                })
            
            # Detect trend acceleration/deceleration
            acceleration = self._detect_trend_acceleration(values)
            if acceleration:
                patterns.append({
                    "pattern_type": "trend_acceleration",
                    "direction": acceleration["direction"],
                    "strength": acceleration["strength"],
                    "significance": "high" if acceleration["strength"] > 0.5 else "medium"
                })
            
            # Detect trend consolidation
            consolidation = self._detect_trend_consolidation(values)
            if consolidation:
                patterns.append({
                    "pattern_type": "trend_consolidation",
                    "duration": consolidation["duration"],
                    "strength": consolidation["strength"],
                    "significance": "medium"
                })
            
            return {
                "trend_patterns": patterns,
                "total_patterns": len(patterns),
                "pattern_types": [p["pattern_type"] for p in patterns]
            }
            
        except Exception as e:
            logger.error(f"Trend pattern identification failed: {e}")
            return {"error": str(e)}
    
    def _detect_trend_reversals(self, values: np.ndarray) -> List[int]:
        """Detect trend reversals in the data."""
        reversals = []
        
        if len(values) < 3:
            return reversals
        
        # Simple reversal detection using local minima/maxima
        for i in range(1, len(values) - 1):
            # Local maximum
            if values[i] > values[i-1] and values[i] > values[i+1]:
                reversals.append(i)
            # Local minimum
            elif values[i] < values[i-1] and values[i] < values[i+1]:
                reversals.append(i)
        
        return reversals
    
    def _detect_trend_acceleration(self, values: np.ndarray) -> Dict[str, Any]:
        """Detect trend acceleration or deceleration."""
        if len(values) < 4:
            return {}
        
        # Calculate second derivative (acceleration)
        first_diff = np.diff(values)
        second_diff = np.diff(first_diff)
        
        # Calculate average acceleration
        avg_acceleration = np.mean(second_diff)
        
        if abs(avg_acceleration) > 0.1:  # Threshold for significant acceleration
            return {
                "direction": "accelerating" if avg_acceleration > 0 else "decelerating",
                "strength": abs(avg_acceleration) / (np.std(values) + 1e-8)
            }
        
        return {}
    
    def _detect_trend_consolidation(self, values: np.ndarray) -> Dict[str, Any]:
        """Detect trend consolidation (sideways movement)."""
        if len(values) < 10:
            return {}
        
        # Calculate trend consistency
        recent_values = values[-10:]
        trend_consistency = 1 - (np.std(recent_values) / (np.mean(recent_values) + 1e-8))
        
        if trend_consistency > 0.8:  # High consistency indicates consolidation
            return {
                "duration": 10,
                "strength": trend_consistency
            }
        
        return {}
    
    async def _forecast_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Forecast future trend based on current patterns."""
        try:
            values = df.iloc[:, 1].values
            
            if len(values) < 5:
                return {"error": "Insufficient data for trend forecasting"}
            
            # Simple linear trend forecasting
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            
            # Forecast future values
            future_x = np.arange(len(values), len(values) + self.analysis_config["forecast_periods"])
            forecast_values = slope * future_x + intercept
            
            # Calculate forecast confidence
            forecast_confidence = min(0.9, abs(slope) / (np.std(values) + 1e-8))
            
            return {
                "forecast_values": forecast_values.tolist(),
                "forecast_periods": self.analysis_config["forecast_periods"],
                "forecast_direction": "increasing" if slope > 0 else "decreasing",
                "forecast_confidence": float(forecast_confidence),
                "forecast_method": "linear_extrapolation"
            }
            
        except Exception as e:
            logger.error(f"Trend forecasting failed: {e}")
            return {"error": str(e)}
    
    async def _compare_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare trends across different time periods."""
        try:
            values = df.iloc[:, 1].values
            
            if len(values) < 10:
                return {"error": "Insufficient data for trend comparison"}
            
            # Split data into periods for comparison
            mid_point = len(values) // 2
            first_half = values[:mid_point]
            second_half = values[mid_point:]
            
            # Analyze trends in each half
            first_trend = await self._analyze_linear_trend(pd.DataFrame({
                "timestamp": range(len(first_half)),
                "value": first_half
            }))
            
            second_trend = await self._analyze_linear_trend(pd.DataFrame({
                "timestamp": range(len(second_half)),
                "value": second_half
            }))
            
            # Compare trends
            trend_change = second_trend.get("slope", 0) - first_trend.get("slope", 0)
            
            return {
                "first_period_trend": first_trend,
                "second_period_trend": second_trend,
                "trend_change": float(trend_change),
                "trend_acceleration": trend_change > 0,
                "comparison_method": "period_split"
            }
            
        except Exception as e:
            logger.error(f"Trend comparison failed: {e}")
            return {"error": str(e)}
    
    async def get_trend_summary(self) -> Dict[str, Any]:
        """Get a summary of all trend analyses."""
        try:
            return {
                "total_analyses": len(self.trend_cache),
                "trend_types": list(set(analysis.get("trend_type", "unknown") for analysis in self.trend_cache.values())),
                "analysis_status": "active",
                "last_analysis": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Trend summary generation failed: {e}")
            return {"error": str(e)}
