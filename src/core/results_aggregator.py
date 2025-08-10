"""
Results aggregator for sentiment analysis results.
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import statistics

from loguru import logger

from src.core.models import AnalysisResult, SentimentLabel


class ResultsAggregator:
    """Aggregates and analyzes sentiment analysis results."""
    
    def __init__(self):
        self.results: List[AnalysisResult] = []
    
    def add_result(self, result: AnalysisResult):
        """Add a result to the aggregator."""
        self.results.append(result)
        logger.debug(f"Added result {result.request_id} to aggregator")
    
    def add_results(self, results: List[AnalysisResult]):
        """Add multiple results to the aggregator."""
        self.results.extend(results)
        logger.info(f"Added {len(results)} results to aggregator")
    
    def clear_results(self):
        """Clear all results from the aggregator."""
        self.results.clear()
        logger.info("Cleared all results from aggregator")
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the results."""
        if not self.results:
            return {"error": "No results available"}
        
        stats = {
            "total_results": len(self.results),
            "sentiment_distribution": {},
            "confidence_stats": {
                "min": float('inf'),
                "max": float('-inf'),
                "avg": 0.0,
                "median": 0.0
            },
            "processing_time_stats": {
                "min": float('inf'),
                "max": float('-inf'),
                "avg": 0.0,
                "median": 0.0
            },
            "quality_score_stats": {
                "min": float('inf'),
                "max": float('-inf'),
                "avg": 0.0,
                "median": 0.0
            },
            "data_type_distribution": {},
            "model_distribution": {},
            "agent_distribution": {},
            "language_distribution": {},
            "reflection_enabled_count": 0,
            "timestamp_range": {
                "earliest": None,
                "latest": None
            }
        }
        
        # Process each result
        confidences = []
        processing_times = []
        quality_scores = []
        
        for result in self.results:
            # Sentiment distribution
            sentiment = result.sentiment.label.value
            stats["sentiment_distribution"][sentiment] = \
                stats["sentiment_distribution"].get(sentiment, 0) + 1
            
            # Confidence stats
            if result.sentiment.confidence > 0:
                confidences.append(result.sentiment.confidence)
                stats["confidence_stats"]["min"] = min(
                    stats["confidence_stats"]["min"], 
                    result.sentiment.confidence
                )
                stats["confidence_stats"]["max"] = max(
                    stats["confidence_stats"]["max"], 
                    result.sentiment.confidence
                )
            
            # Processing time stats
            if result.processing_time > 0:
                processing_times.append(result.processing_time)
                stats["processing_time_stats"]["min"] = min(
                    stats["processing_time_stats"]["min"], 
                    result.processing_time
                )
                stats["processing_time_stats"]["max"] = max(
                    stats["processing_time_stats"]["max"], 
                    result.processing_time
                )
            
            # Quality score stats
            if result.quality_score and result.quality_score > 0:
                quality_scores.append(result.quality_score)
                stats["quality_score_stats"]["min"] = min(
                    stats["quality_score_stats"]["min"], 
                    result.quality_score
                )
                stats["quality_score_stats"]["max"] = max(
                    stats["quality_score_stats"]["max"], 
                    result.quality_score
                )
            
            # Data type distribution
            data_type = result.data_type.value
            stats["data_type_distribution"][data_type] = \
                stats["data_type_distribution"].get(data_type, 0) + 1
            
            # Model distribution
            if result.model_used:
                model = result.model_used
                stats["model_distribution"][model] = \
                    stats["model_distribution"].get(model, 0) + 1
            
            # Agent distribution
            agent_id = result.metadata.get("agent_id", "unknown")
            stats["agent_distribution"][agent_id] = \
                stats["agent_distribution"].get(agent_id, 0) + 1
            
            # Language distribution
            language = getattr(result, 'language', 'en')
            stats["language_distribution"][language] = \
                stats["language_distribution"].get(language, 0) + 1
            
            # Reflection enabled count
            if result.reflection_enabled:
                stats["reflection_enabled_count"] += 1
        
        # Calculate statistics
        if confidences:
            stats["confidence_stats"]["avg"] = statistics.mean(confidences)
            stats["confidence_stats"]["median"] = statistics.median(confidences)
            stats["confidence_stats"]["min"] = stats["confidence_stats"]["min"] if stats["confidence_stats"]["min"] != float('inf') else 0.0
            stats["confidence_stats"]["max"] = stats["confidence_stats"]["max"] if stats["confidence_stats"]["max"] != float('-inf') else 0.0
        
        if processing_times:
            stats["processing_time_stats"]["avg"] = statistics.mean(processing_times)
            stats["processing_time_stats"]["median"] = statistics.median(processing_times)
            stats["processing_time_stats"]["min"] = stats["processing_time_stats"]["min"] if stats["processing_time_stats"]["min"] != float('inf') else 0.0
            stats["processing_time_stats"]["max"] = stats["processing_time_stats"]["max"] if stats["processing_time_stats"]["max"] != float('-inf') else 0.0
        
        if quality_scores:
            stats["quality_score_stats"]["avg"] = statistics.mean(quality_scores)
            stats["quality_score_stats"]["median"] = statistics.median(quality_scores)
            stats["quality_score_stats"]["min"] = stats["quality_score_stats"]["min"] if stats["quality_score_stats"]["min"] != float('inf') else 0.0
            stats["quality_score_stats"]["max"] = stats["quality_score_stats"]["max"] if stats["quality_score_stats"]["max"] != float('-inf') else 0.0
        
        return stats
    
    def get_sentiment_analysis(self) -> Dict[str, Any]:
        """Get detailed sentiment analysis."""
        if not self.results:
            return {"error": "No results available"}
        
        analysis = {
            "total_results": len(self.results),
            "sentiment_breakdown": {},
            "confidence_by_sentiment": {},
            "quality_by_sentiment": {},
            "processing_time_by_sentiment": {},
            "sentiment_trends": {}
        }
        
        # Group results by sentiment
        sentiment_groups = {}
        for result in self.results:
            sentiment = result.sentiment.label.value
            if sentiment not in sentiment_groups:
                sentiment_groups[sentiment] = []
            sentiment_groups[sentiment].append(result)
        
        # Analyze each sentiment group
        for sentiment, results in sentiment_groups.items():
            analysis["sentiment_breakdown"][sentiment] = {
                "count": len(results),
                "percentage": len(results) / len(self.results) * 100
            }
            
            # Confidence analysis for this sentiment
            confidences = [r.sentiment.confidence for r in results if r.sentiment.confidence > 0]
            if confidences:
                analysis["confidence_by_sentiment"][sentiment] = {
                    "min": min(confidences),
                    "max": max(confidences),
                    "avg": statistics.mean(confidences),
                    "median": statistics.median(confidences)
                }
            
            # Quality analysis for this sentiment
            quality_scores = [r.quality_score for r in results if r.quality_score and r.quality_score > 0]
            if quality_scores:
                analysis["quality_by_sentiment"][sentiment] = {
                    "min": min(quality_scores),
                    "max": max(quality_scores),
                    "avg": statistics.mean(quality_scores),
                    "median": statistics.median(quality_scores)
                }
            
            # Processing time analysis for this sentiment
            processing_times = [r.processing_time for r in results if r.processing_time > 0]
            if processing_times:
                analysis["processing_time_by_sentiment"][sentiment] = {
                    "min": min(processing_times),
                    "max": max(processing_times),
                    "avg": statistics.mean(processing_times),
                    "median": statistics.median(processing_times)
                }
        
        return analysis
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """Get performance analysis of the system."""
        if not self.results:
            return {"error": "No results available"}
        
        performance = {
            "total_results": len(self.results),
            "overall_performance": {},
            "performance_by_agent": {},
            "performance_by_model": {},
            "performance_by_data_type": {},
            "efficiency_metrics": {}
        }
        
        # Overall performance
        processing_times = [r.processing_time for r in self.results if r.processing_time > 0]
        quality_scores = [r.quality_score for r in self.results if r.quality_score and r.quality_score > 0]
        
        if processing_times:
            performance["overall_performance"]["processing_time"] = {
                "min": min(processing_times),
                "max": max(processing_times),
                "avg": statistics.mean(processing_times),
                "median": statistics.median(processing_times),
                "total": sum(processing_times)
            }
        
        if quality_scores:
            performance["overall_performance"]["quality_score"] = {
                "min": min(quality_scores),
                "max": max(quality_scores),
                "avg": statistics.mean(quality_scores),
                "median": statistics.median(quality_scores)
            }
        
        # Performance by agent
        agent_groups = {}
        for result in self.results:
            agent_id = result.metadata.get("agent_id", "unknown")
            if agent_id not in agent_groups:
                agent_groups[agent_id] = []
            agent_groups[agent_id].append(result)
        
        for agent_id, results in agent_groups.items():
            agent_times = [r.processing_time for r in results if r.processing_time > 0]
            agent_quality = [r.quality_score for r in results if r.quality_score and r.quality_score > 0]
            
            performance["performance_by_agent"][agent_id] = {
                "total_results": len(results),
                "avg_processing_time": statistics.mean(agent_times) if agent_times else 0,
                "avg_quality_score": statistics.mean(agent_quality) if agent_quality else 0
            }
        
        # Performance by model
        model_groups = {}
        for result in self.results:
            model = result.model_used or "unknown"
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append(result)
        
        for model, results in model_groups.items():
            model_times = [r.processing_time for r in results if r.processing_time > 0]
            model_quality = [r.quality_score for r in results if r.quality_score and r.quality_score > 0]
            
            performance["performance_by_model"][model] = {
                "total_results": len(results),
                "avg_processing_time": statistics.mean(model_times) if model_times else 0,
                "avg_quality_score": statistics.mean(model_quality) if model_quality else 0
            }
        
        # Efficiency metrics
        if processing_times and quality_scores:
            # Quality per time unit (higher is better)
            avg_time = statistics.mean(processing_times)
            avg_quality = statistics.mean(quality_scores)
            if avg_time > 0:
                performance["efficiency_metrics"]["quality_per_time"] = avg_quality / avg_time
            
            # Results per time unit
            total_time = sum(processing_times)
            if total_time > 0:
                performance["efficiency_metrics"]["results_per_time"] = len(self.results) / total_time
        
        return performance
    
    def get_trend_analysis(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get trend analysis over time."""
        if not self.results:
            return {"error": "No results available"}
        
        # Group results by time windows
        now = datetime.now()
        time_windows = {}
        
        for result in self.results:
            # Use current timestamp if not available
            timestamp = getattr(result, 'timestamp', now)
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except:
                    timestamp = now
            
            # Group by hour
            hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
            if hour_key not in time_windows:
                time_windows[hour_key] = []
            time_windows[hour_key].append(result)
        
        # Sort time windows
        sorted_windows = sorted(time_windows.keys())
        
        trends = {
            "time_windows": [],
            "sentiment_trends": {},
            "performance_trends": {},
            "volume_trends": []
        }
        
        for window in sorted_windows:
            results = time_windows[window]
            
            # Count by sentiment
            sentiment_counts = {}
            for result in results:
                sentiment = result.sentiment.label.value
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            # Average processing time and quality
            processing_times = [r.processing_time for r in results if r.processing_time > 0]
            quality_scores = [r.quality_score for r in results if r.quality_score and r.quality_score > 0]
            
            window_data = {
                "timestamp": window.isoformat(),
                "total_results": len(results),
                "sentiment_distribution": sentiment_counts,
                "avg_processing_time": statistics.mean(processing_times) if processing_times else 0,
                "avg_quality_score": statistics.mean(quality_scores) if quality_scores else 0
            }
            
            trends["time_windows"].append(window_data)
            trends["volume_trends"].append(len(results))
        
        # Calculate trends
        if len(trends["volume_trends"]) > 1:
            trends["volume_trend"] = "increasing" if trends["volume_trends"][-1] > trends["volume_trends"][0] else "decreasing"
        
        return trends
    
    def export_results(self, filepath: str, format: str = "json") -> bool:
        """Export results to a file."""
        try:
            filepath = Path(filepath)
            
            if format.lower() == "json":
                # Export as JSON
                export_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_results": len(self.results),
                    "results": [result.dict() for result in self.results],
                    "summary": self.get_basic_stats()
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                logger.info(f"Exported {len(self.results)} results to {filepath}")
                return True
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report."""
        if not self.results:
            return {"error": "No results available"}
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "summary": self.get_basic_stats(),
            "sentiment_analysis": self.get_sentiment_analysis(),
            "performance_analysis": self.get_performance_analysis(),
            "trend_analysis": self.get_trend_analysis(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if not self.results:
            return recommendations
        
        # Analyze confidence scores
        low_confidence_count = sum(
            1 for r in self.results 
            if r.sentiment.confidence < 0.7
        )
        if low_confidence_count > len(self.results) * 0.2:  # More than 20%
            recommendations.append(
                "Consider improving model quality or enabling reflection for better confidence scores"
            )
        
        # Analyze processing times
        avg_processing_time = statistics.mean([
            r.processing_time for r in self.results 
            if r.processing_time > 0
        ])
        if avg_processing_time > 5.0:  # More than 5 seconds
            recommendations.append(
                "Processing times are high - consider optimizing models or using faster hardware"
            )
        
        # Analyze quality scores
        low_quality_count = sum(
            1 for r in self.results 
            if r.quality_score and r.quality_score < 0.7
        )
        if low_quality_count > len(self.results) * 0.15:  # More than 15%
            recommendations.append(
                "Quality scores are low - consider enabling reflection or improving model selection"
            )
        
        # Analyze reflection usage
        reflection_enabled_count = sum(
            1 for r in self.results 
            if r.reflection_enabled
        )
        if reflection_enabled_count < len(self.results) * 0.5:  # Less than 50%
            recommendations.append(
                "Consider enabling reflection for more results to improve quality"
            )
        
        if not recommendations:
            recommendations.append("System is performing well - no immediate improvements needed")
        
        return recommendations


# Global instance
results_aggregator = ResultsAggregator()
