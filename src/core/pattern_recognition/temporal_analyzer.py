"""
Temporal Pattern Analyzer

This module provides time-based pattern analysis capabilities including:
- Historical pattern detection
- Time series analysis
- Temporal relationship tracking
- Content evolution patterns
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from loguru import logger

from src.core.models import AnalysisRequest, AnalysisResult
from src.core.error_handler import with_error_handling


class TemporalAnalyzer:
    """
    Analyzes temporal patterns in data including historical trends,
    seasonal patterns, and time-based relationships.
    """
    
    def __init__(self):
        self.pattern_cache = {}
        self.temporal_data = defaultdict(list)
        self.analysis_config = {
            "min_data_points": 10,
            "trend_threshold": 0.1,
            "seasonal_periods": [7, 30, 90, 365],  # days
            "confidence_threshold": 0.7
        }
        
        logger.info("TemporalAnalyzer initialized successfully")
    
    @with_error_handling("temporal_analysis")
    async def analyze_temporal_patterns(
        self, 
        data: List[Dict[str, Any]], 
        time_field: str = "timestamp",
        value_field: str = "value"
    ) -> Dict[str, Any]:
        """
        Analyze temporal patterns in the provided data.
        
        Args:
            data: List of data points with timestamps
            time_field: Field name containing timestamp
            value_field: Field name containing values to analyze
            
        Returns:
            Dictionary containing temporal pattern analysis results
        """
        try:
            logger.info(f"Analyzing temporal patterns for {len(data)} data points")
            
            # Convert to pandas DataFrame for analysis
            df = self._prepare_dataframe(data, time_field, value_field)
            
            if len(df) < self.analysis_config["min_data_points"]:
                return {
                    "error": f"Insufficient data points. Need at least {self.analysis_config['min_data_points']}"
                }
            
            # Perform various temporal analyses
            results = {
                "trend_analysis": await self._analyze_trends(df),
                "seasonal_patterns": await self._detect_seasonal_patterns(df),
                "temporal_relationships": await self._analyze_temporal_relationships(df),
                "content_evolution": await self._analyze_content_evolution(df),
                "cross_language_patterns": await self._analyze_cross_language_patterns(df),
                "metadata": {
                    "total_data_points": len(df),
                    "time_range": {
                        "start": df[time_field].min().isoformat(),
                        "end": df[time_field].max().isoformat()
                    },
                    "analysis_timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info("Temporal pattern analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Temporal pattern analysis failed: {e}")
            return {"error": str(e)}
    
    def _prepare_dataframe(self, data: List[Dict[str, Any]], time_field: str, value_field: str) -> pd.DataFrame:
        """Prepare data for analysis by converting to pandas DataFrame."""
        df = pd.DataFrame(data)
        
        # Convert timestamp field to datetime
        if time_field in df.columns:
            df[time_field] = pd.to_datetime(df[time_field])
            df = df.sort_values(time_field)
        
        return df
    
    async def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in the data."""
        try:
            # Calculate trend direction and strength
            if len(df) < 2:
                return {"error": "Insufficient data for trend analysis"}
            
            # Simple linear trend analysis
            x = np.arange(len(df))
            y = df.iloc[:, 1].values  # Assuming second column is the value
            
            # Calculate trend
            slope, intercept = np.polyfit(x, y, 1)
            trend_strength = abs(slope) / (np.std(y) + 1e-8)
            
            # Determine trend direction
            if slope > self.analysis_config["trend_threshold"]:
                trend_direction = "increasing"
            elif slope < -self.analysis_config["trend_threshold"]:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
            
            return {
                "trend_direction": trend_direction,
                "trend_strength": float(trend_strength),
                "slope": float(slope),
                "confidence": min(0.95, trend_strength),
                "trend_type": "linear"
            }
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {"error": str(e)}
    
    async def _detect_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect seasonal patterns in the data."""
        try:
            seasonal_patterns = []
            
            for period in self.analysis_config["seasonal_periods"]:
                if len(df) >= period * 2:
                    # Calculate autocorrelation for seasonal detection
                    autocorr = self._calculate_autocorrelation(df.iloc[:, 1].values, period)
                    
                    if autocorr > 0.3:  # Threshold for seasonal pattern
                        seasonal_patterns.append({
                            "period_days": period,
                            "strength": float(autocorr),
                            "pattern_type": "seasonal",
                            "confidence": min(0.9, autocorr)
                        })
            
            return {
                "seasonal_patterns": seasonal_patterns,
                "total_patterns": len(seasonal_patterns),
                "analysis_method": "autocorrelation"
            }
            
        except Exception as e:
            logger.error(f"Seasonal pattern detection failed: {e}")
            return {"error": str(e)}
    
    def _calculate_autocorrelation(self, data: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation for seasonal pattern detection."""
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
    
    async def _analyze_temporal_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal relationships between entities."""
        try:
            # This would analyze relationships between different entities over time
            # For now, return a basic structure
            return {
                "entity_relationships": [],
                "relationship_strength": 0.0,
                "analysis_method": "temporal_correlation",
                "notes": "Entity relationship analysis requires entity extraction data"
            }
            
        except Exception as e:
            logger.error(f"Temporal relationship analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_content_evolution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how content evolves over time."""
        try:
            # This would analyze how content themes, topics, or sentiment evolve
            return {
                "content_evolution": {
                    "theme_changes": [],
                    "topic_shifts": [],
                    "sentiment_evolution": []
                },
                "evolution_strength": 0.0,
                "analysis_method": "content_tracking"
            }
            
        except Exception as e:
            logger.error(f"Content evolution analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_cross_language_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns across different languages."""
        try:
            # This would analyze patterns that occur across multiple languages
            return {
                "cross_language_patterns": [],
                "language_correlation": 0.0,
                "analysis_method": "multilingual_correlation"
            }
            
        except Exception as e:
            logger.error(f"Cross-language pattern analysis failed: {e}")
            return {"error": str(e)}
    
    async def track_entity_relationships(
        self, 
        entities: List[Dict[str, Any]], 
        timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Track relationships between entities over time.
        
        Args:
            entities: List of entities with their properties
            timestamp: When these entities were observed
            
        Returns:
            Dictionary containing relationship tracking results
        """
        try:
            # Store temporal data
            self.temporal_data[timestamp.isoformat()] = entities
            
            # Analyze relationships
            relationships = await self._extract_entity_relationships(entities)
            
            return {
                "timestamp": timestamp.isoformat(),
                "entities_count": len(entities),
                "relationships": relationships,
                "tracking_status": "active"
            }
            
        except Exception as e:
            logger.error(f"Entity relationship tracking failed: {e}")
            return {"error": str(e)}
    
    async def _extract_entity_relationships(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities."""
        relationships = []
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Calculate relationship strength based on entity properties
                relationship_strength = self._calculate_relationship_strength(entity1, entity2)
                
                if relationship_strength > 0.1:  # Threshold for meaningful relationship
                    relationships.append({
                        "entity1": entity1.get("id", f"entity_{i}"),
                        "entity2": entity2.get("id", f"entity_{j}"),
                        "relationship_type": "temporal_cooccurrence",
                        "strength": relationship_strength,
                        "confidence": min(0.9, relationship_strength)
                    })
        
        return relationships
    
    def _calculate_relationship_strength(self, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> float:
        """Calculate the strength of relationship between two entities."""
        # Simple relationship strength calculation
        # In a real implementation, this would be more sophisticated
        
        # Check for common properties
        common_properties = 0
        total_properties = 0
        
        for key in entity1.keys():
            if key in entity2:
                if entity1[key] == entity2[key]:
                    common_properties += 1
                total_properties += 1
        
        if total_properties == 0:
            return 0.0
        
        return common_properties / total_properties
    
    async def get_temporal_summary(self) -> Dict[str, Any]:
        """Get a summary of all temporal data and patterns."""
        try:
            total_observations = len(self.temporal_data)
            total_entities = sum(len(entities) for entities in self.temporal_data.values())
            
            return {
                "total_observations": total_observations,
                "total_entities": total_entities,
                "time_range": {
                    "start": min(self.temporal_data.keys()) if self.temporal_data else None,
                    "end": max(self.temporal_data.keys()) if self.temporal_data else None
                },
                "pattern_cache_size": len(self.pattern_cache),
                "analysis_status": "active"
            }
            
        except Exception as e:
            logger.error(f"Temporal summary generation failed: {e}")
            return {"error": str(e)}
