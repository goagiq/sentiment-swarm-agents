"""
Enhanced Scenario Analysis

Provides advanced scenario analysis capabilities including:
- Real-time data integration
- Historical pattern analysis
- Multi-modal scenario inputs
- Dynamic scenario adaptation
- Enhanced confidence scoring
- External data source integration
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics

from loguru import logger

from src.core.models import DataType, AnalysisRequest
from src.core.error_handler import with_error_handling
from src.core.unified_mcp_client import call_unified_mcp_tool
from src.core.multi_modal_integration_engine import MultiModalIntegrationEngine


@dataclass
class RealTimeDataPoint:
    """Represents a real-time data point."""
    source: str  # market_data, social_media, iot_sensor, news
    data_type: str
    value: Any
    timestamp: datetime
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoricalPattern:
    """Represents a historical pattern for scenario analysis."""
    pattern_type: str
    entity: str
    frequency: float
    success_rate: float
    impact_score: float
    time_period: timedelta
    evidence_count: int
    last_occurrence: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedScenario:
    """Enhanced scenario with real-time data and historical patterns."""
    scenario_id: str
    name: str
    description: str
    scenario_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    real_time_data: List[RealTimeDataPoint] = field(default_factory=list)
    historical_patterns: List[HistoricalPattern] = field(default_factory=list)
    multi_modal_inputs: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    risk_score: float = 0.0
    impact_score: float = 0.0
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ScenarioOutcome:
    """Represents a predicted scenario outcome."""
    scenario_id: str
    outcome_type: str  # success, failure, partial_success
    probability: float
    confidence: float
    impact_metrics: Dict[str, float] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    success_factors: List[str] = field(default_factory=list)
    timeline: timedelta = timedelta(days=30)
    evidence: Dict[str, Any] = field(default_factory=dict)


class EnhancedScenarioAnalysis:
    """
    Enhanced scenario analysis with real-time data integration and multi-modal inputs.
    """
    
    def __init__(self):
        self.multi_modal_engine = MultiModalIntegrationEngine()
        self.real_time_data_sources = {
            "market_data": self._fetch_market_data,
            "social_media": self._fetch_social_media_data,
            "news": self._fetch_news_data,
            "iot_sensors": self._fetch_iot_data
        }
        
        self.historical_pattern_cache = {}
        self.scenario_cache = {}
        self.data_refresh_interval = 300  # 5 minutes
        
        # Configuration for different data sources
        self.data_source_config = {
            "market_data": {
                "refresh_interval": 60,  # 1 minute
                "confidence_threshold": 0.7,
                "max_data_points": 1000
            },
            "social_media": {
                "refresh_interval": 300,  # 5 minutes
                "confidence_threshold": 0.6,
                "max_data_points": 500
            },
            "news": {
                "refresh_interval": 600,  # 10 minutes
                "confidence_threshold": 0.8,
                "max_data_points": 200
            },
            "iot_sensors": {
                "refresh_interval": 30,  # 30 seconds
                "confidence_threshold": 0.9,
                "max_data_points": 2000
            }
        }
        
        logger.info("EnhancedScenarioAnalysis initialized successfully")
    
    @with_error_handling("enhanced_scenario_analysis")
    async def create_enhanced_scenario(
        self,
        name: str,
        description: str,
        scenario_type: str,
        parameters: Dict[str, Any],
        multi_modal_inputs: Optional[List[AnalysisRequest]] = None,
        real_time_sources: Optional[List[str]] = None
    ) -> EnhancedScenario:
        """
        Create an enhanced scenario with real-time data and multi-modal inputs.
        
        Args:
            name: Scenario name
            description: Scenario description
            scenario_type: Type of scenario
            parameters: Scenario parameters
            multi_modal_inputs: Multi-modal analysis requests
            real_time_sources: List of real-time data sources to include
            
        Returns:
            Enhanced scenario with integrated data
        """
        try:
            scenario_id = f"scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(name)}"
            
            # Fetch real-time data
            real_time_data = []
            if real_time_sources:
                for source in real_time_sources:
                    if source in self.real_time_data_sources:
                        data_points = await self.real_time_data_sources[source]()
                        real_time_data.extend(data_points)
            
            # Analyze multi-modal inputs
            multi_modal_context = None
            if multi_modal_inputs:
                multi_modal_context = await self.multi_modal_engine.build_unified_context(
                    multi_modal_inputs
                )
            
            # Find relevant historical patterns
            historical_patterns = await self._find_relevant_historical_patterns(
                scenario_type, parameters, multi_modal_context
            )
            
            # Calculate initial confidence and risk scores
            confidence_score = await self._calculate_scenario_confidence(
                real_time_data, historical_patterns, multi_modal_context
            )
            
            risk_score = await self._calculate_scenario_risk(
                real_time_data, historical_patterns, multi_modal_context
            )
            
            impact_score = await self._calculate_scenario_impact(
                parameters, historical_patterns, multi_modal_context
            )
            
            scenario = EnhancedScenario(
                scenario_id=scenario_id,
                name=name,
                description=description,
                scenario_type=scenario_type,
                parameters=parameters,
                real_time_data=real_time_data,
                historical_patterns=historical_patterns,
                multi_modal_inputs=multi_modal_context.__dict__ if multi_modal_context else {},
                confidence_score=confidence_score,
                risk_score=risk_score,
                impact_score=impact_score
            )
            
            # Cache the scenario
            self.scenario_cache[scenario_id] = scenario
            
            logger.info(f"Created enhanced scenario: {scenario_id}")
            return scenario
            
        except Exception as e:
            logger.error(f"Error creating enhanced scenario: {e}")
            raise
    
    @with_error_handling("scenario_outcome_prediction")
    async def predict_scenario_outcomes(
        self,
        scenario: EnhancedScenario,
        time_horizon: timedelta = timedelta(days=30)
    ) -> List[ScenarioOutcome]:
        """
        Predict outcomes for an enhanced scenario.
        
        Args:
            scenario: Enhanced scenario to analyze
            time_horizon: Time horizon for prediction
            
        Returns:
            List of predicted outcomes with probabilities
        """
        try:
            outcomes = []
            
            # Success outcome
            success_probability = await self._calculate_success_probability(scenario)
            success_outcome = ScenarioOutcome(
                scenario_id=scenario.scenario_id,
                outcome_type="success",
                probability=success_probability,
                confidence=scenario.confidence_score,
                impact_metrics=await self._calculate_impact_metrics(scenario, "success"),
                success_factors=await self._identify_success_factors(scenario),
                timeline=time_horizon,
                evidence={"confidence_score": scenario.confidence_score}
            )
            outcomes.append(success_outcome)
            
            # Failure outcome
            failure_probability = await self._calculate_failure_probability(scenario)
            failure_outcome = ScenarioOutcome(
                scenario_id=scenario.scenario_id,
                outcome_type="failure",
                probability=failure_probability,
                confidence=scenario.confidence_score,
                impact_metrics=await self._calculate_impact_metrics(scenario, "failure"),
                risk_factors=await self._identify_risk_factors(scenario),
                timeline=time_horizon,
                evidence={"risk_score": scenario.risk_score}
            )
            outcomes.append(failure_outcome)
            
            # Partial success outcome
            partial_success_probability = 1.0 - success_probability - failure_probability
            if partial_success_probability > 0.1:  # Only include if significant
                partial_outcome = ScenarioOutcome(
                    scenario_id=scenario.scenario_id,
                    outcome_type="partial_success",
                    probability=partial_success_probability,
                    confidence=scenario.confidence_score * 0.8,
                    impact_metrics=await self._calculate_impact_metrics(scenario, "partial"),
                    timeline=time_horizon,
                    evidence={"impact_score": scenario.impact_score}
                )
                outcomes.append(partial_outcome)
            
            return outcomes
            
        except Exception as e:
            logger.error(f"Error predicting scenario outcomes: {e}")
            raise
    
    @with_error_handling("scenario_adaptation")
    async def adapt_scenario(
        self,
        scenario: EnhancedScenario,
        new_data: Dict[str, Any]
    ) -> EnhancedScenario:
        """
        Adapt a scenario based on new data.
        
        Args:
            scenario: Scenario to adapt
            new_data: New data to incorporate
            
        Returns:
            Adapted scenario
        """
        try:
            # Create adaptation record
            adaptation_record = {
                "timestamp": datetime.now(),
                "trigger": new_data.get("trigger", "manual"),
                "changes": new_data.get("changes", {}),
                "reason": new_data.get("reason", "data_update")
            }
            
            # Update scenario parameters
            if "parameters" in new_data:
                scenario.parameters.update(new_data["parameters"])
            
            # Update real-time data
            if "real_time_data" in new_data:
                scenario.real_time_data.extend(new_data["real_time_data"])
                # Keep only recent data points
                cutoff_time = datetime.now() - timedelta(hours=24)
                scenario.real_time_data = [
                    dp for dp in scenario.real_time_data
                    if dp.timestamp > cutoff_time
                ]
            
            # Recalculate scores
            scenario.confidence_score = await self._calculate_scenario_confidence(
                scenario.real_time_data,
                scenario.historical_patterns,
                scenario.multi_modal_inputs
            )
            
            scenario.risk_score = await self._calculate_scenario_risk(
                scenario.real_time_data,
                scenario.historical_patterns,
                scenario.multi_modal_inputs
            )
            
            scenario.impact_score = await self._calculate_scenario_impact(
                scenario.parameters,
                scenario.historical_patterns,
                scenario.multi_modal_inputs
            )
            
            # Update timestamps
            scenario.updated_at = datetime.now()
            scenario.adaptation_history.append(adaptation_record)
            
            # Update cache
            self.scenario_cache[scenario.scenario_id] = scenario
            
            logger.info(f"Adapted scenario: {scenario.scenario_id}")
            return scenario
            
        except Exception as e:
            logger.error(f"Error adapting scenario: {e}")
            raise
    
    @with_error_handling("real_time_data_fetch")
    async def _fetch_market_data(self) -> List[RealTimeDataPoint]:
        """Fetch real-time market data using external API connectors."""
        try:
            from src.core.integration.market_data_connector import get_market_data_connector
            
            connector = await get_market_data_connector()
            async with connector:
                market_data = await connector.fetch_real_time_data()
            
            data_points = []
            for data_point in market_data:
                real_time_point = RealTimeDataPoint(
                    source="market_data",
                    data_type="stock_price",
                    value={
                        "symbol": data_point.symbol,
                        "price": data_point.price,
                        "change": data_point.change,
                        "volume": data_point.volume,
                        "change_percent": data_point.change_percent
                    },
                    timestamp=data_point.timestamp,
                    confidence=data_point.confidence,
                    metadata=data_point.metadata
                )
                data_points.append(real_time_point)
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error fetching real market data: {e}")
            # Fallback to simulated data
            return await self._fetch_simulated_market_data()
    
    async def _fetch_simulated_market_data(self) -> List[RealTimeDataPoint]:
        """Fetch simulated market data as fallback."""
        try:
            data_points = []
            
            # Simulate market data
            market_indicators = ["stock_price", "volume", "volatility", "sentiment"]
            for indicator in market_indicators:
                data_point = RealTimeDataPoint(
                    source="market_data",
                    data_type=indicator,
                    value=0.5 + (hash(indicator) % 100) / 100,  # Simulated value
                    timestamp=datetime.now(),
                    confidence=0.5,  # Lower confidence for simulated data
                    metadata={"indicator": indicator, "simulated": True, "fallback": True}
                )
                data_points.append(data_point)
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error fetching simulated market data: {e}")
            return []
    
    @with_error_handling("social_media_data_fetch")
    async def _fetch_social_media_data(self) -> List[RealTimeDataPoint]:
        """Fetch real-time social media data using external API connectors."""
        try:
            from src.core.integration.social_media_connector import get_social_media_connector
            
            connector = await get_social_media_connector()
            async with connector:
                social_posts = await connector.fetch_real_time_data()
            
            data_points = []
            for post in social_posts:
                real_time_point = RealTimeDataPoint(
                    source="social_media",
                    data_type="post",
                    value={
                        "platform": post.platform,
                        "content": post.content,
                        "sentiment_score": post.sentiment_score,
                        "engagement": post.engagement
                    },
                    timestamp=post.timestamp,
                    confidence=post.confidence,
                    metadata=post.metadata
                )
                data_points.append(real_time_point)
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error fetching real social media data: {e}")
            # Fallback to simulated data
            return await self._fetch_simulated_social_media_data()
    
    async def _fetch_simulated_social_media_data(self) -> List[RealTimeDataPoint]:
        """Fetch simulated social media data as fallback."""
        try:
            data_points = []
            
            # Simulate social media data
            social_metrics = ["sentiment", "engagement", "reach", "trending_topics"]
            for metric in social_metrics:
                data_point = RealTimeDataPoint(
                    source="social_media",
                    data_type=metric,
                    value=0.3 + (hash(metric) % 100) / 100,  # Simulated value
                    timestamp=datetime.now(),
                    confidence=0.3,  # Lower confidence for simulated data
                    metadata={"metric": metric, "simulated": True, "fallback": True}
                )
                data_points.append(data_point)
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error fetching simulated social media data: {e}")
            return []
    
    @with_error_handling("news_data_fetch")
    async def _fetch_news_data(self) -> List[RealTimeDataPoint]:
        """Fetch real-time news data."""
        try:
            # This would integrate with news APIs
            # For now, return simulated data
            data_points = []
            
            # Simulate news data
            news_metrics = ["sentiment", "relevance", "urgency", "impact"]
            for metric in news_metrics:
                data_point = RealTimeDataPoint(
                    source="news",
                    data_type=metric,
                    value=0.4 + (hash(metric) % 100) / 100,  # Simulated value
                    timestamp=datetime.now(),
                    confidence=0.8,
                    metadata={"metric": metric}
                )
                data_points.append(data_point)
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
            return []
    
    @with_error_handling("iot_data_fetch")
    async def _fetch_iot_data(self) -> List[RealTimeDataPoint]:
        """Fetch real-time IoT sensor data."""
        try:
            # This would integrate with IoT sensor networks
            # For now, return simulated data
            data_points = []
            
            # Simulate IoT data
            iot_metrics = ["temperature", "humidity", "pressure", "motion"]
            for metric in iot_metrics:
                data_point = RealTimeDataPoint(
                    source="iot_sensors",
                    data_type=metric,
                    value=20 + (hash(metric) % 50),  # Simulated value
                    timestamp=datetime.now(),
                    confidence=0.9,
                    metadata={"sensor_type": metric}
                )
                data_points.append(data_point)
            
            return data_points
            
        except Exception as e:
            logger.error(f"Error fetching IoT data: {e}")
            return []
    
    async def _find_relevant_historical_patterns(
        self,
        scenario_type: str,
        parameters: Dict[str, Any],
        multi_modal_context: Optional[Any] = None
    ) -> List[HistoricalPattern]:
        """Find historical patterns relevant to the scenario."""
        try:
            patterns = []
            
            # Query knowledge graph for historical patterns
            if multi_modal_context and hasattr(multi_modal_context, 'unified_entities'):
                for entity in multi_modal_context.unified_entities:
                    entity_name = entity.get("name", "")
                    if entity_name:
                        # Query for historical patterns related to this entity
                        pattern_result = await call_unified_mcp_tool(
                            "analyze_decision_patterns",
                            {
                                "entity_name": entity_name,
                                "pattern_type": "historical_patterns",
                                "time_window": "1y"
                            }
                        )
                        
                        if pattern_result.get("success") and pattern_result.get("patterns"):
                            for pattern_data in pattern_result["patterns"]:
                                pattern = HistoricalPattern(
                                    pattern_type=pattern_data.get("type", "unknown"),
                                    entity=entity_name,
                                    frequency=pattern_data.get("frequency", 0.0),
                                    success_rate=pattern_data.get("success_rate", 0.0),
                                    impact_score=pattern_data.get("impact_score", 0.0),
                                    time_period=timedelta(days=pattern_data.get("days", 30)),
                                    evidence_count=pattern_data.get("evidence_count", 0),
                                    last_occurrence=datetime.fromisoformat(
                                        pattern_data.get("last_occurrence", datetime.now().isoformat())
                                    ),
                                    metadata=pattern_data.get("metadata", {})
                                )
                                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error finding historical patterns: {e}")
            return []
    
    async def _calculate_scenario_confidence(
        self,
        real_time_data: List[RealTimeDataPoint],
        historical_patterns: List[HistoricalPattern],
        multi_modal_context: Optional[Any] = None
    ) -> float:
        """Calculate confidence score for a scenario."""
        try:
            confidence_factors = []
            
            # Real-time data confidence
            if real_time_data:
                avg_data_confidence = statistics.mean([dp.confidence for dp in real_time_data])
                confidence_factors.append(avg_data_confidence * 0.3)
            
            # Historical pattern confidence
            if historical_patterns:
                avg_pattern_confidence = statistics.mean([
                    pattern.success_rate for pattern in historical_patterns
                ])
                confidence_factors.append(avg_pattern_confidence * 0.4)
            
            # Multi-modal context confidence
            if multi_modal_context and hasattr(multi_modal_context, 'overall_confidence'):
                confidence_factors.append(multi_modal_context.overall_confidence * 0.3)
            
            # Calculate weighted average
            if confidence_factors:
                return sum(confidence_factors) / len(confidence_factors)
            else:
                return 0.5  # Default confidence
            
        except Exception as e:
            logger.error(f"Error calculating scenario confidence: {e}")
            return 0.5
    
    async def _calculate_scenario_risk(
        self,
        real_time_data: List[RealTimeDataPoint],
        historical_patterns: List[HistoricalPattern],
        multi_modal_context: Optional[Any] = None
    ) -> float:
        """Calculate risk score for a scenario."""
        try:
            risk_factors = []
            
            # Real-time data risk indicators
            if real_time_data:
                risk_indicators = [
                    dp.value for dp in real_time_data
                    if dp.data_type in ["volatility", "risk_score", "uncertainty"]
                ]
                if risk_indicators:
                    avg_risk = statistics.mean(risk_indicators)
                    risk_factors.append(avg_risk * 0.4)
            
            # Historical pattern risk
            if historical_patterns:
                failure_rates = [1.0 - pattern.success_rate for pattern in historical_patterns]
                if failure_rates:
                    avg_failure_rate = statistics.mean(failure_rates)
                    risk_factors.append(avg_failure_rate * 0.3)
            
            # Multi-modal context risk
            if multi_modal_context and hasattr(multi_modal_context, 'decision_factors'):
                risk_factors_count = sum(
                    1 for factor in multi_modal_context.decision_factors
                    if factor.get("type") == "risk"
                )
                if risk_factors_count > 0:
                    risk_factors.append(min(risk_factors_count * 0.1, 0.3))
            
            # Calculate weighted average
            if risk_factors:
                return min(sum(risk_factors), 1.0)
            else:
                return 0.3  # Default risk
            
        except Exception as e:
            logger.error(f"Error calculating scenario risk: {e}")
            return 0.3
    
    async def _calculate_scenario_impact(
        self,
        parameters: Dict[str, Any],
        historical_patterns: List[HistoricalPattern],
        multi_modal_context: Optional[Any] = None
    ) -> float:
        """Calculate impact score for a scenario."""
        try:
            impact_factors = []
            
            # Parameter-based impact
            if "impact_scope" in parameters:
                impact_factors.append(parameters["impact_scope"] * 0.3)
            
            if "resource_requirement" in parameters:
                impact_factors.append(parameters["resource_requirement"] * 0.2)
            
            # Historical pattern impact
            if historical_patterns:
                avg_impact = statistics.mean([
                    pattern.impact_score for pattern in historical_patterns
                ])
                impact_factors.append(avg_impact * 0.3)
            
            # Multi-modal context impact
            if multi_modal_context and hasattr(multi_modal_context, 'decision_factors'):
                high_impact_factors = sum(
                    1 for factor in multi_modal_context.decision_factors
                    if factor.get("impact") == "high"
                )
                if high_impact_factors > 0:
                    impact_factors.append(min(high_impact_factors * 0.1, 0.2))
            
            # Calculate weighted average
            if impact_factors:
                return min(sum(impact_factors), 1.0)
            else:
                return 0.5  # Default impact
            
        except Exception as e:
            logger.error(f"Error calculating scenario impact: {e}")
            return 0.5
    
    async def _calculate_success_probability(self, scenario: EnhancedScenario) -> float:
        """Calculate success probability for a scenario."""
        try:
            # Base probability from confidence
            base_probability = scenario.confidence_score * 0.6
            
            # Adjust based on risk
            risk_adjustment = (1.0 - scenario.risk_score) * 0.3
            
            # Adjust based on historical patterns
            historical_adjustment = 0.0
            if scenario.historical_patterns:
                avg_success_rate = statistics.mean([
                    pattern.success_rate for pattern in scenario.historical_patterns
                ])
                historical_adjustment = avg_success_rate * 0.1
            
            final_probability = base_probability + risk_adjustment + historical_adjustment
            return min(max(final_probability, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating success probability: {e}")
            return 0.5
    
    async def _calculate_failure_probability(self, scenario: EnhancedScenario) -> float:
        """Calculate failure probability for a scenario."""
        try:
            # Base probability from risk
            base_probability = scenario.risk_score * 0.7
            
            # Adjust based on confidence
            confidence_adjustment = (1.0 - scenario.confidence_score) * 0.2
            
            # Adjust based on historical patterns
            historical_adjustment = 0.0
            if scenario.historical_patterns:
                avg_failure_rate = statistics.mean([
                    1.0 - pattern.success_rate for pattern in scenario.historical_patterns
                ])
                historical_adjustment = avg_failure_rate * 0.1
            
            final_probability = base_probability + confidence_adjustment + historical_adjustment
            return min(max(final_probability, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating failure probability: {e}")
            return 0.3
    
    async def _calculate_impact_metrics(
        self, 
        scenario: EnhancedScenario, 
        outcome_type: str
    ) -> Dict[str, float]:
        """Calculate impact metrics for a scenario outcome."""
        try:
            base_impact = scenario.impact_score
            
            if outcome_type == "success":
                return {
                    "financial_impact": base_impact * 1.2,
                    "operational_impact": base_impact * 0.8,
                    "strategic_impact": base_impact * 1.0,
                    "risk_reduction": base_impact * 0.6
                }
            elif outcome_type == "failure":
                return {
                    "financial_impact": -base_impact * 1.5,
                    "operational_impact": -base_impact * 1.0,
                    "strategic_impact": -base_impact * 1.2,
                    "risk_increase": base_impact * 1.0
                }
            else:  # partial_success
                return {
                    "financial_impact": base_impact * 0.3,
                    "operational_impact": base_impact * 0.5,
                    "strategic_impact": base_impact * 0.4,
                    "risk_change": base_impact * 0.2
                }
            
        except Exception as e:
            logger.error(f"Error calculating impact metrics: {e}")
            return {"financial_impact": 0.0, "operational_impact": 0.0}
    
    async def _identify_success_factors(self, scenario: EnhancedScenario) -> List[str]:
        """Identify factors that contribute to scenario success."""
        try:
            success_factors = []
            
            # High confidence factors
            if scenario.confidence_score > 0.8:
                success_factors.append("High confidence in scenario parameters")
            
            # Strong historical patterns
            strong_patterns = [
                pattern for pattern in scenario.historical_patterns
                if pattern.success_rate > 0.7
            ]
            if strong_patterns:
                success_factors.append(f"Strong historical success patterns ({len(strong_patterns)} patterns)")
            
            # Real-time data support
            if scenario.real_time_data:
                positive_data = [
                    dp for dp in scenario.real_time_data
                    if dp.confidence > 0.7 and dp.value > 0.6
                ]
                if positive_data:
                    success_factors.append(f"Supportive real-time data ({len(positive_data)} data points)")
            
            return success_factors
            
        except Exception as e:
            logger.error(f"Error identifying success factors: {e}")
            return []
    
    async def _identify_risk_factors(self, scenario: EnhancedScenario) -> List[str]:
        """Identify factors that contribute to scenario risk."""
        try:
            risk_factors = []
            
            # High risk score
            if scenario.risk_score > 0.7:
                risk_factors.append("High overall risk score")
            
            # Weak historical patterns
            weak_patterns = [
                pattern for pattern in scenario.historical_patterns
                if pattern.success_rate < 0.3
            ]
            if weak_patterns:
                risk_factors.append(f"Poor historical performance patterns ({len(weak_patterns)} patterns)")
            
            # Negative real-time data
            if scenario.real_time_data:
                negative_data = [
                    dp for dp in scenario.real_time_data
                    if dp.confidence > 0.7 and dp.value < 0.4
                ]
                if negative_data:
                    risk_factors.append(f"Concerning real-time data indicators ({len(negative_data)} data points)")
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {e}")
            return []
