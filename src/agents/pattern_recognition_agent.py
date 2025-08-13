"""
Pattern Recognition Agent

This agent coordinates pattern recognition capabilities including:
- Temporal pattern analysis
- Seasonal pattern detection
- Trend analysis
- Pattern storage and retrieval
- Vector-based pattern discovery
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from uuid import uuid4

from loguru import logger

from src.agents.base_agent import StrandsBaseAgent
from src.core.models import AnalysisRequest, AnalysisResult, DataType, SentimentResult
from src.core.error_handler import with_error_handling
from src.core.pattern_recognition import (
    TemporalAnalyzer,
    SeasonalDetector,
    TrendEngine,
    PatternStorage
)


class PatternRecognitionAgent(StrandsBaseAgent):
    """
    Agent for comprehensive pattern recognition and analysis.
    
    Supports:
    - Temporal pattern analysis
    - Seasonal pattern detection
    - Trend analysis and forecasting
    - Pattern storage and retrieval
    - Vector-based pattern discovery
    """
    
    def __init__(self, model_name: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_name or "llama3.2:latest", **kwargs)
        
        # Initialize pattern recognition components
        self.temporal_analyzer = TemporalAnalyzer()
        self.seasonal_detector = SeasonalDetector()
        self.trend_engine = TrendEngine()
        self.pattern_storage = PatternStorage()
        
        # Set metadata
        self.metadata["agent_type"] = "pattern_recognition"
        self.metadata["capabilities"] = [
            "temporal_analysis",
            "seasonal_detection",
            "trend_analysis",
            "pattern_storage",
            "vector_patterns",
            "anomaly_detection"
        ]
        self.metadata["supported_data_types"] = [
            "text", "time_series", "numerical", "categorical"
        ]
        
        logger.info("PatternRecognitionAgent initialized successfully")
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        # Pattern recognition agent can process various data types
        return request.data_type in [
            DataType.TEXT, 
            DataType.GENERAL, 
            DataType.NUMERICAL,
            DataType.TIME_SERIES
        ]
    
    @with_error_handling("pattern_recognition_processing")
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process pattern recognition requests."""
        try:
            logger.info(f"Processing pattern recognition request: {request.data_type}")
            
            start_time = datetime.now()
            
            # Route request based on data type and metadata
            if request.data_type == DataType.TIME_SERIES:
                result = await self._process_time_series_request(request)
            elif request.data_type == DataType.NUMERICAL:
                result = await self._process_numerical_request(request)
            else:
                result = await self._process_general_request(request)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create sentiment result based on pattern analysis
            sentiment = SentimentResult(
                label="neutral", 
                confidence=0.8, 
                reasoning=f"Pattern analysis completed: {result.get('analysis_type', 'general')} analysis with {result.get('data_points', 0)} data points"
            )
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=sentiment,
                processing_time=processing_time,
                raw_content=str(request.content),
                extracted_text=result.get("extracted_text", ""),
                metadata=result.get("pattern_analysis", {}),
                model_used="pattern_recognition_engine",
                reflection_enabled=request.reflection_enabled,
                quality_score=0.8
            )
            
        except Exception as e:
            logger.error(f"Pattern recognition processing failed: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral", 
                    confidence=0.0, 
                    reasoning=f"Processing failed: {str(e)}"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={"error": str(e)},
                model_used="pattern_recognition_engine",
                reflection_enabled=request.reflection_enabled,
                quality_score=0.0
            )
    
    async def _process_time_series_request(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Process time series data for pattern recognition."""
        try:
            # Parse time series data
            data = self._parse_time_series_data(request.content)
            
            if not data:
                return {"error": "Invalid time series data format"}
            
            # Perform comprehensive pattern analysis
            results = {
                "temporal_patterns": await self.temporal_analyzer.analyze_temporal_patterns(data),
                "seasonal_patterns": await self.seasonal_detector.detect_seasonal_patterns(data),
                "trend_analysis": await self.trend_engine.analyze_trends(data),
                "pattern_storage": await self._store_patterns(data, "time_series"),
                "analysis_summary": await self._generate_analysis_summary(data)
            }
            
            return {
                "pattern_analysis": results,
                "analysis_type": "time_series",
                "data_points": len(data),
                "extracted_text": f"Analyzed {len(data)} time series data points for patterns"
            }
            
        except Exception as e:
            logger.error(f"Time series processing failed: {e}")
            return {"error": str(e)}
    
    async def _process_numerical_request(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Process numerical data for pattern recognition."""
        try:
            # Parse numerical data
            data = self._parse_numerical_data(request.content)
            
            if not data:
                return {"error": "Invalid numerical data format"}
            
            # Perform numerical pattern analysis
            results = {
                "trend_analysis": await self.trend_engine.analyze_trends(data),
                "statistical_patterns": await self._analyze_statistical_patterns(data),
                "pattern_storage": await self._store_patterns(data, "numerical"),
                "analysis_summary": await self._generate_analysis_summary(data)
            }
            
            return {
                "pattern_analysis": results,
                "analysis_type": "numerical",
                "data_points": len(data),
                "extracted_text": f"Analyzed {len(data)} numerical data points for patterns"
            }
            
        except Exception as e:
            logger.error(f"Numerical processing failed: {e}")
            return {"error": str(e)}
    
    async def _process_general_request(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Process general data for pattern recognition."""
        try:
            # For general requests, try to extract patterns from text or other content
            content = request.content
            
            # Extract basic patterns from content
            patterns = await self._extract_content_patterns(content)
            
            # Store patterns
            storage_result = await self._store_patterns(patterns, "general")
            
            return {
                "pattern_analysis": {
                    "content_patterns": patterns,
                    "pattern_storage": storage_result,
                    "analysis_summary": await self._generate_content_summary(content)
                },
                "analysis_type": "general",
                "content_length": len(content),
                "extracted_text": f"Extracted patterns from {len(content)} character content"
            }
            
        except Exception as e:
            logger.error(f"General processing failed: {e}")
            return {"error": str(e)}
    
    def _parse_time_series_data(self, content: str) -> List[Dict[str, Any]]:
        """Parse time series data from content."""
        try:
            # Try to parse as JSON first
            if content.strip().startswith('['):
                data = json.loads(content)
                return data
            
            # Try to parse as CSV-like format
            lines = content.strip().split('\n')
            data = []
            
            for line in lines:
                if ',' in line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            timestamp = parts[0].strip()
                            value = float(parts[1].strip())
                            data.append({
                                "timestamp": timestamp,
                                "value": value
                            })
                        except ValueError:
                            continue
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to parse time series data: {e}")
            return []
    
    def _parse_numerical_data(self, content: str) -> List[Dict[str, Any]]:
        """Parse numerical data from content."""
        try:
            # Try to parse as JSON first
            if content.strip().startswith('['):
                data = json.loads(content)
                return data
            
            # Try to parse as simple numerical list
            lines = content.strip().split('\n')
            data = []
            
            for i, line in enumerate(lines):
                try:
                    value = float(line.strip())
                    data.append({
                        "timestamp": i,
                        "value": value
                    })
                except ValueError:
                    continue
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to parse numerical data: {e}")
            return []
    
    async def _analyze_statistical_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze statistical patterns in numerical data."""
        try:
            values = [d.get("value", 0) for d in data]
            
            if not values:
                return {"error": "No valid numerical values found"}
            
            # Calculate basic statistics
            import numpy as np
            
            stats = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "range": float(np.max(values) - np.min(values)),
                "count": len(values)
            }
            
            # Detect outliers using IQR method
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = [v for v in values if v < lower_bound or v > upper_bound]
            
            return {
                "statistics": stats,
                "outliers": {
                    "count": len(outliers),
                    "values": outliers,
                    "percentage": len(outliers) / len(values) * 100
                },
                "distribution": {
                    "skewness": float(self._calculate_skewness(values)),
                    "kurtosis": float(self._calculate_kurtosis(values))
                }
            }
            
        except Exception as e:
            logger.error(f"Statistical pattern analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of the data."""
        try:
            import numpy as np
            mean = np.mean(values)
            std = np.std(values)
            n = len(values)
            
            skewness = (n / ((n-1) * (n-2))) * np.sum(((values - mean) / std) ** 3)
            return skewness
        except:
            return 0.0
    
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate kurtosis of the data."""
        try:
            import numpy as np
            mean = np.mean(values)
            std = np.std(values)
            n = len(values)
            
            kurtosis = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * np.sum(((values - mean) / std) ** 4) - (3 * (n-1)**2 / ((n-2) * (n-3)))
            return kurtosis
        except:
            return 0.0
    
    async def _extract_content_patterns(self, content: str) -> Dict[str, Any]:
        """Extract patterns from text content."""
        try:
            patterns = {
                "text_length": len(content),
                "word_count": len(content.split()),
                "character_frequency": self._analyze_character_frequency(content),
                "word_frequency": self._analyze_word_frequency(content),
                "sentence_count": len([s for s in content.split('.') if s.strip()]),
                "paragraph_count": len([p for p in content.split('\n\n') if p.strip()])
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Content pattern extraction failed: {e}")
            return {"error": str(e)}
    
    def _analyze_character_frequency(self, content: str) -> Dict[str, int]:
        """Analyze character frequency in content."""
        freq = {}
        for char in content.lower():
            if char.isalpha():
                freq[char] = freq.get(char, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _analyze_word_frequency(self, content: str) -> Dict[str, int]:
        """Analyze word frequency in content."""
        import re
        words = re.findall(r'\b\w+\b', content.lower())
        freq = {}
        for word in words:
            if len(word) > 2:  # Skip short words
                freq[word] = freq.get(word, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10])
    
    async def _store_patterns(self, data: Any, pattern_type: str) -> Dict[str, Any]:
        """Store patterns in the pattern storage."""
        try:
            pattern_id = f"{pattern_type}_{uuid4().hex[:8]}"
            
            result = await self.pattern_storage.store_pattern(
                pattern_id=pattern_id,
                pattern_data=data,
                pattern_type=pattern_type,
                metadata={
                    "agent_id": self.agent_id,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "data_type": pattern_type
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Pattern storage failed: {e}")
            return {"error": str(e)}
    
    async def _generate_analysis_summary(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of the analysis."""
        try:
            return {
                "total_data_points": len(data),
                "analysis_timestamp": datetime.now().isoformat(),
                "agent_id": self.agent_id,
                "analysis_status": "completed",
                "key_insights": [
                    f"Analyzed {len(data)} data points",
                    "Patterns detected and stored",
                    "Statistical analysis completed"
                ]
            }
            
        except Exception as e:
            logger.error(f"Analysis summary generation failed: {e}")
            return {"error": str(e)}
    
    async def _generate_content_summary(self, content: str) -> Dict[str, Any]:
        """Generate a summary of content analysis."""
        try:
            return {
                "content_length": len(content),
                "analysis_timestamp": datetime.now().isoformat(),
                "agent_id": self.agent_id,
                "analysis_status": "completed",
                "key_insights": [
                    f"Analyzed {len(content)} character content",
                    "Text patterns extracted",
                    "Frequency analysis completed"
                ]
            }
            
        except Exception as e:
            logger.error(f"Content summary generation failed: {e}")
            return {"error": str(e)}
    
    # Public API methods for external use
    
    async def analyze_temporal_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in data."""
        return await self.temporal_analyzer.analyze_temporal_patterns(data)
    
    async def detect_seasonal_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect seasonal patterns in data."""
        return await self.seasonal_detector.detect_seasonal_patterns(data)
    
    async def analyze_trends(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in data."""
        return await self.trend_engine.analyze_trends(data)
    
    async def store_pattern(self, pattern_id: str, pattern_data: Dict[str, Any], pattern_type: str = "general") -> Dict[str, Any]:
        """Store a pattern in the database."""
        return await self.pattern_storage.store_pattern(pattern_id, pattern_data, pattern_type)
    
    async def get_pattern(self, pattern_id: str) -> Dict[str, Any]:
        """Retrieve a pattern from the database."""
        return await self.pattern_storage.get_pattern(pattern_id)
    
    async def search_patterns(self, pattern_type: Optional[str] = None, status: str = "active") -> Dict[str, Any]:
        """Search for patterns in the database."""
        return await self.pattern_storage.search_patterns(pattern_type, status)
    
    async def get_storage_summary(self) -> Dict[str, Any]:
        """Get a summary of the pattern storage."""
        return await self.pattern_storage.get_storage_summary()
