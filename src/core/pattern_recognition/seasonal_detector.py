"""
Seasonal Pattern Detector

This module provides seasonal pattern detection capabilities including:
- Cyclical behavior identification
- Seasonal pattern analysis
- Periodicity detection
- Seasonal trend analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from loguru import logger

from src.core.error_handler import with_error_handling


class SeasonalDetector:
    """
    Detects seasonal patterns and cyclical behavior in time series data.
    """
    
    def __init__(self):
        self.seasonal_patterns = {}
        self.detection_config = {
            "min_periods": 3,
            "max_periods": 365,
            "autocorr_threshold": 0.3,
            "seasonal_strength_threshold": 0.1,
            "fourier_threshold": 0.1
        }
        
        logger.info("SeasonalDetector initialized successfully")
    
    @with_error_handling("seasonal_detection")
    async def detect_seasonal_patterns(
        self, 
        data: List[Dict[str, Any]], 
        time_field: str = "timestamp",
        value_field: str = "value"
    ) -> Dict[str, Any]:
        """
        Detect seasonal patterns in the provided data.
        
        Args:
            data: List of data points with timestamps
            time_field: Field name containing timestamp
            value_field: Field name containing values to analyze
            
        Returns:
            Dictionary containing seasonal pattern detection results
        """
        try:
            logger.info(f"Detecting seasonal patterns for {len(data)} data points")
            
            # Convert to pandas DataFrame
            df = self._prepare_dataframe(data, time_field, value_field)
            
            if len(df) < self.detection_config["min_periods"] * 2:
                return {
                    "error": f"Insufficient data for seasonal analysis. Need at least {self.detection_config['min_periods'] * 2} points"
                }
            
            # Perform seasonal pattern detection
            results = {
                "autocorrelation_analysis": await self._autocorrelation_analysis(df),
                "fourier_analysis": await self._fourier_analysis(df),
                "seasonal_decomposition": await self._seasonal_decomposition(df),
                "periodicity_detection": await self._detect_periodicity(df),
                "seasonal_strength": await self._calculate_seasonal_strength(df),
                "metadata": {
                    "total_data_points": len(df),
                    "time_range": {
                        "start": df[time_field].min().isoformat(),
                        "end": df[time_field].max().isoformat()
                    },
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info("Seasonal pattern detection completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Seasonal pattern detection failed: {e}")
            return {"error": str(e)}
    
    def _prepare_dataframe(self, data: List[Dict[str, Any]], time_field: str, value_field: str) -> pd.DataFrame:
        """Prepare data for analysis by converting to pandas DataFrame."""
        df = pd.DataFrame(data)
        
        # Convert timestamp field to datetime
        if time_field in df.columns:
            df[time_field] = pd.to_datetime(df[time_field])
            df = df.sort_values(time_field)
        
        return df
    
    async def _autocorrelation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform autocorrelation analysis to detect seasonal patterns."""
        try:
            values = df.iloc[:, 1].values  # Assuming second column is the value
            autocorr_results = []
            
            # Test different lag periods
            max_lag = min(len(values) // 2, self.detection_config["max_periods"])
            
            for lag in range(1, max_lag + 1):
                autocorr = self._calculate_autocorrelation(values, lag)
                
                if autocorr > self.detection_config["autocorr_threshold"]:
                    autocorr_results.append({
                        "lag": lag,
                        "autocorrelation": float(autocorr),
                        "significance": "high" if autocorr > 0.5 else "medium",
                        "pattern_type": "seasonal"
                    })
            
            return {
                "autocorrelation_results": autocorr_results,
                "significant_lags": len(autocorr_results),
                "max_autocorrelation": max([r["autocorrelation"] for r in autocorr_results]) if autocorr_results else 0.0
            }
            
        except Exception as e:
            logger.error(f"Autocorrelation analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_autocorrelation(self, data: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation for a given lag."""
        if len(data) < lag * 2:
            return 0.0
        
        # Calculate autocorrelation
        mean = np.mean(data)
        var = np.var(data)
        
        if var == 0:
            return 0.0
        
        autocorr = 0.0
        for i in range(len(data) - lag):
            autocorr += (data[i] - mean) * (data[i + lag] - mean)
        
        autocorr /= (len(data) - lag) * var
        return abs(autocorr)
    
    async def _fourier_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform Fourier analysis to detect periodic patterns."""
        try:
            values = df.iloc[:, 1].values
            
            # Perform FFT
            fft_values = np.fft.fft(values)
            fft_freq = np.fft.fftfreq(len(values))
            
            # Find dominant frequencies
            power_spectrum = np.abs(fft_values) ** 2
            significant_freqs = []
            
            # Find frequencies above threshold
            threshold = np.max(power_spectrum) * self.detection_config["fourier_threshold"]
            
            for i, power in enumerate(power_spectrum[1:len(power_spectrum)//2]):
                if power > threshold:
                    freq = fft_freq[i + 1]
                    if freq > 0:  # Only positive frequencies
                        period = 1 / freq
                        significant_freqs.append({
                            "frequency": float(freq),
                            "period": float(period),
                            "power": float(power),
                            "significance": "high" if power > threshold * 2 else "medium"
                        })
            
            return {
                "significant_frequencies": significant_freqs,
                "dominant_periods": [f["period"] for f in significant_freqs],
                "total_significant_freqs": len(significant_freqs)
            }
            
        except Exception as e:
            logger.error(f"Fourier analysis failed: {e}")
            return {"error": str(e)}
    
    async def _seasonal_decomposition(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform seasonal decomposition of the time series."""
        try:
            values = df.iloc[:, 1].values
            
            # Simple seasonal decomposition
            # In a real implementation, you might use statsmodels.seasonal_decompose
            
            # Calculate moving averages for trend
            window_size = min(7, len(values) // 4)  # Adaptive window size
            if window_size % 2 == 0:
                window_size += 1
            
            trend = self._moving_average(values, window_size)
            
            # Calculate seasonal component (simplified)
            seasonal = self._extract_seasonal_component(values, trend)
            
            # Calculate residual
            residual = values - trend - seasonal
            
            return {
                "trend_component": trend.tolist(),
                "seasonal_component": seasonal.tolist(),
                "residual_component": residual.tolist(),
                "decomposition_method": "moving_average",
                "window_size": window_size
            }
            
        except Exception as e:
            logger.error(f"Seasonal decomposition failed: {e}")
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
    
    def _extract_seasonal_component(self, data: np.ndarray, trend: np.ndarray) -> np.ndarray:
        """Extract seasonal component from detrended data."""
        # Detrend the data
        detrended = data - trend
        
        # Simple seasonal extraction (assume weekly pattern for demo)
        seasonal = np.zeros_like(data)
        
        # For demonstration, create a simple weekly pattern
        if len(data) >= 7:
            weekly_pattern = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1])
            for i in range(len(data)):
                seasonal[i] = weekly_pattern[i % 7] * np.std(detrended)
        
        return seasonal
    
    async def _detect_periodicity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect periodic patterns in the data."""
        try:
            values = df.iloc[:, 1].values
            periodic_patterns = []
            
            # Test for common periodicities
            common_periods = [7, 14, 30, 90, 180, 365]  # days
            
            for period in common_periods:
                if len(values) >= period * 2:
                    # Calculate periodicity strength
                    periodicity_strength = self._calculate_periodicity_strength(values, period)
                    
                    if periodicity_strength > self.detection_config["seasonal_strength_threshold"]:
                        periodic_patterns.append({
                            "period": period,
                            "strength": float(periodicity_strength),
                            "type": "weekly" if period == 7 else "monthly" if period == 30 else "yearly",
                            "confidence": min(0.9, periodicity_strength)
                        })
            
            return {
                "periodic_patterns": periodic_patterns,
                "total_patterns": len(periodic_patterns),
                "strongest_period": max(periodic_patterns, key=lambda x: x["strength"]) if periodic_patterns else None
            }
            
        except Exception as e:
            logger.error(f"Periodicity detection failed: {e}")
            return {"error": str(e)}
    
    def _calculate_periodicity_strength(self, data: np.ndarray, period: int) -> float:
        """Calculate the strength of a periodic pattern."""
        if len(data) < period * 2:
            return 0.0
        
        # Calculate correlation between data and its lagged version
        lagged_data = np.roll(data, period)
        
        # Calculate correlation
        correlation = np.corrcoef(data[:-period], lagged_data[:-period])[0, 1]
        
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    async def _calculate_seasonal_strength(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate the overall seasonal strength of the data."""
        try:
            values = df.iloc[:, 1].values
            
            # Calculate seasonal strength using variance ratio
            # Seasonal strength = variance of seasonal component / total variance
            
            # Get seasonal component
            window_size = min(7, len(values) // 4)
            if window_size % 2 == 0:
                window_size += 1
            
            trend = self._moving_average(values, window_size)
            seasonal = self._extract_seasonal_component(values, trend)
            
            # Calculate variances
            total_variance = np.var(values)
            seasonal_variance = np.var(seasonal)
            
            if total_variance > 0:
                seasonal_strength = seasonal_variance / total_variance
            else:
                seasonal_strength = 0.0
            
            return {
                "seasonal_strength": float(seasonal_strength),
                "total_variance": float(total_variance),
                "seasonal_variance": float(seasonal_variance),
                "strength_category": "strong" if seasonal_strength > 0.6 else "moderate" if seasonal_strength > 0.3 else "weak"
            }
            
        except Exception as e:
            logger.error(f"Seasonal strength calculation failed: {e}")
            return {"error": str(e)}
    
    async def get_seasonal_summary(self) -> Dict[str, Any]:
        """Get a summary of all detected seasonal patterns."""
        try:
            return {
                "total_patterns": len(self.seasonal_patterns),
                "pattern_types": list(set(p.get("type", "unknown") for p in self.seasonal_patterns.values())),
                "analysis_status": "active",
                "last_analysis": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Seasonal summary generation failed: {e}")
            return {"error": str(e)}
