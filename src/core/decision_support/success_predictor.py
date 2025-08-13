"""
Success Predictor Component

Provides AI-powered success prediction and outcome analysis for recommendations
based on historical data, contextual factors, and risk assessment.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

from .recommendation_engine import Recommendation

logger = logging.getLogger(__name__)


class SuccessLevel(Enum):
    """Success levels for predictions."""
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    FAILURE = "failure"


class PredictionConfidence(Enum):
    """Confidence levels for predictions."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class OutcomeType(Enum):
    """Types of outcomes that can be predicted."""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    RISK = "risk"
    COMPLIANCE = "compliance"
    REPUTATIONAL = "reputational"


@dataclass
class SuccessFactor:
    """Represents a factor that influences success prediction."""
    name: str
    weight: float
    score: float
    description: str
    impact: str = "positive"  # positive, negative, neutral
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'weight': self.weight,
            'score': self.score,
            'description': self.description,
            'impact': self.impact
        }


@dataclass
class OutcomePrediction:
    """Represents a prediction for a specific outcome type."""
    outcome_type: OutcomeType
    success_probability: float
    confidence_level: PredictionConfidence
    expected_value: Optional[float] = None
    best_case_scenario: Optional[float] = None
    worst_case_scenario: Optional[float] = None
    factors: List[SuccessFactor] = field(default_factory=list)
    timeline_months: int = 12
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'outcome_type': self.outcome_type.value,
            'success_probability': self.success_probability,
            'confidence_level': self.confidence_level.value,
            'expected_value': self.expected_value,
            'best_case_scenario': self.best_case_scenario,
            'worst_case_scenario': self.worst_case_scenario,
            'factors': [factor.to_dict() for factor in self.factors],
            'timeline_months': self.timeline_months
        }


@dataclass
class SuccessPrediction:
    """Complete success prediction for a recommendation."""
    recommendation_id: str
    overall_success_probability: float
    success_level: SuccessLevel
    confidence_level: PredictionConfidence
    outcome_predictions: List[OutcomePrediction] = field(default_factory=list)
    key_success_factors: List[SuccessFactor] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    recommendations_for_success: List[str] = field(default_factory=list)
    timeline_months: int = 12
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'recommendation_id': self.recommendation_id,
            'overall_success_probability': self.overall_success_probability,
            'success_level': self.success_level.value,
            'confidence_level': self.confidence_level.value,
            'outcome_predictions': [pred.to_dict() for pred in self.outcome_predictions],
            'key_success_factors': [factor.to_dict() for factor in self.key_success_factors],
            'risk_factors': self.risk_factors,
            'recommendations_for_success': self.recommendations_for_success,
            'timeline_months': self.timeline_months,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class PredictionContext:
    """Context information for success prediction."""
    historical_success_rates: Dict[str, float] = field(default_factory=dict)
    industry_benchmarks: Dict[str, Any] = field(default_factory=dict)
    organizational_capabilities: Dict[str, float] = field(default_factory=dict)
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    resource_availability: Dict[str, Any] = field(default_factory=dict)
    stakeholder_support: Dict[str, float] = field(default_factory=dict)
    external_factors: Dict[str, Any] = field(default_factory=dict)


class SuccessPredictor:
    """AI-powered success prediction engine."""
    
    def __init__(self):
        self.success_factors = self._load_success_factors()
        self.industry_patterns = self._load_industry_patterns()
        self.historical_data = self._load_historical_data()
        logger.info("Initialized SuccessPredictor")
    
    def _load_success_factors(self) -> Dict[str, Dict[str, Any]]:
        """Load success factors for different recommendation types."""
        return {
            "technology_adoption": {
                "technical_expertise": {"weight": 0.25, "description": "Team technical capabilities"},
                "change_management": {"weight": 0.20, "description": "Change management effectiveness"},
                "user_adoption": {"weight": 0.20, "description": "User acceptance and adoption"},
                "infrastructure": {"weight": 0.15, "description": "Infrastructure readiness"},
                "budget_adequacy": {"weight": 0.10, "description": "Budget sufficiency"},
                "timeline_realism": {"weight": 0.10, "description": "Timeline feasibility"}
            },
            "process_improvement": {
                "process_understanding": {"weight": 0.20, "description": "Understanding of current processes"},
                "stakeholder_buy_in": {"weight": 0.25, "description": "Stakeholder support and commitment"},
                "change_management": {"weight": 0.20, "description": "Change management effectiveness"},
                "resource_availability": {"weight": 0.15, "description": "Resource availability"},
                "measurement_capability": {"weight": 0.10, "description": "Ability to measure improvements"},
                "continuous_improvement": {"weight": 0.10, "description": "Continuous improvement culture"}
            },
            "risk_management": {
                "risk_assessment_quality": {"weight": 0.25, "description": "Quality of risk assessment"},
                "mitigation_strategy": {"weight": 0.25, "description": "Effectiveness of mitigation strategy"},
                "monitoring_capability": {"weight": 0.20, "description": "Risk monitoring capabilities"},
                "organizational_commitment": {"weight": 0.15, "description": "Organizational commitment"},
                "resource_allocation": {"weight": 0.10, "description": "Resource allocation for risk management"},
                "compliance_requirements": {"weight": 0.05, "description": "Compliance requirements"}
            },
            "market_strategy": {
                "market_understanding": {"weight": 0.25, "description": "Market knowledge and insights"},
                "competitive_positioning": {"weight": 0.20, "description": "Competitive positioning"},
                "execution_capability": {"weight": 0.20, "description": "Execution capabilities"},
                "resource_allocation": {"weight": 0.15, "description": "Resource allocation"},
                "timing": {"weight": 0.10, "description": "Market timing"},
                "flexibility": {"weight": 0.10, "description": "Strategy flexibility"}
            }
        }
    
    def _load_industry_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load industry-specific success patterns."""
        return {
            "technology": {
                "success_rate": 0.65,
                "key_factors": ["technical_expertise", "user_adoption", "change_management"],
                "common_failures": ["scope_creep", "user_resistance", "technical_debt"]
            },
            "healthcare": {
                "success_rate": 0.55,
                "key_factors": ["compliance", "stakeholder_buy_in", "change_management"],
                "common_failures": ["regulatory_issues", "staff_resistance", "integration_challenges"]
            },
            "finance": {
                "success_rate": 0.60,
                "key_factors": ["risk_management", "compliance", "change_management"],
                "common_failures": ["regulatory_violations", "system_failures", "data_breaches"]
            },
            "manufacturing": {
                "success_rate": 0.70,
                "key_factors": ["process_understanding", "resource_availability", "measurement_capability"],
                "common_failures": ["process_disruption", "quality_issues", "supply_chain_problems"]
            }
        }
    
    def _load_historical_data(self) -> Dict[str, Any]:
        """Load historical success data for predictions."""
        return {
            "overall_success_rate": 0.62,
            "success_by_category": {
                "technology_adoption": 0.58,
                "process_improvement": 0.65,
                "risk_management": 0.70,
                "market_strategy": 0.55
            },
            "success_by_priority": {
                "critical": 0.75,
                "high": 0.65,
                "medium": 0.60,
                "low": 0.50
            },
            "success_by_effort": {
                "low": 0.70,
                "medium": 0.65,
                "high": 0.55
            }
        }
    
    async def predict_success(
        self,
        recommendation: Recommendation,
        context: PredictionContext
    ) -> SuccessPrediction:
        """Predict the likelihood of success for a recommendation."""
        try:
            logger.info(f"Predicting success for recommendation: {recommendation.title}")
            
            # Get success factors for recommendation type
            recommendation_type = self._get_recommendation_type(recommendation)
            factors = self.success_factors.get(recommendation_type, {})
            
            # Calculate success factors
            success_factors = await self._calculate_success_factors(
                recommendation, factors, context
            )
            
            # Calculate overall success probability
            overall_probability = await self._calculate_overall_probability(
                success_factors, recommendation, context
            )
            
            # Determine success level and confidence
            success_level = self._determine_success_level(overall_probability)
            confidence_level = await self._determine_confidence_level(
                success_factors, context
            )
            
            # Generate outcome predictions
            outcome_predictions = await self._generate_outcome_predictions(
                recommendation, success_factors, context
            )
            
            # Identify key success factors and risk factors
            key_success_factors = self._identify_key_success_factors(success_factors)
            risk_factors = await self._identify_risk_factors(recommendation, context)
            
            # Generate recommendations for success
            success_recommendations = await self._generate_success_recommendations(
                recommendation, success_factors, risk_factors
            )
            
            # Create success prediction
            prediction = SuccessPrediction(
                recommendation_id=recommendation.id,
                overall_success_probability=overall_probability,
                success_level=success_level,
                confidence_level=confidence_level,
                outcome_predictions=outcome_predictions,
                key_success_factors=key_success_factors,
                risk_factors=risk_factors,
                recommendations_for_success=success_recommendations,
                timeline_months=self._estimate_timeline(recommendation)
            )
            
            logger.info(f"Success prediction completed: {overall_probability:.2%} probability")
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting success: {e}")
            return SuccessPrediction(
                recommendation_id=recommendation.id,
                overall_success_probability=0.5,
                success_level=SuccessLevel.MODERATE,
                confidence_level=PredictionConfidence.LOW
            )
    
    def _get_recommendation_type(self, recommendation: Recommendation) -> str:
        """Get recommendation type for factor selection."""
        category_mapping = {
            "technology_adoption": "technology_adoption",
            "process_improvement": "process_improvement",
            "risk_management": "risk_management",
            "market_strategy": "market_strategy"
        }
        
        return category_mapping.get(recommendation.category.value, "process_improvement")
    
    async def _calculate_success_factors(
        self,
        recommendation: Recommendation,
        factors: Dict[str, Dict[str, Any]],
        context: PredictionContext
    ) -> List[SuccessFactor]:
        """Calculate success factors for the recommendation."""
        success_factors = []
        
        for factor_name, factor_config in factors.items():
            # Calculate factor score based on context and recommendation
            score = await self._calculate_factor_score(
                factor_name, recommendation, context
            )
            
            # Determine impact (positive, negative, neutral)
            impact = "positive" if score > 0.6 else "negative" if score < 0.4 else "neutral"
            
            success_factor = SuccessFactor(
                name=factor_name,
                weight=factor_config["weight"],
                score=score,
                description=factor_config["description"],
                impact=impact
            )
            
            success_factors.append(success_factor)
        
        return success_factors
    
    async def _calculate_factor_score(
        self,
        factor_name: str,
        recommendation: Recommendation,
        context: PredictionContext
    ) -> float:
        """Calculate score for a specific success factor."""
        # Base score from historical data
        base_score = 0.6
        
        # Adjust based on recommendation characteristics
        if factor_name == "technical_expertise":
            if recommendation.implementation_effort == "high":
                base_score -= 0.1
            elif recommendation.implementation_effort == "low":
                base_score += 0.1
        
        elif factor_name == "change_management":
            if recommendation.priority.value == "critical":
                base_score += 0.1
            elif recommendation.priority.value == "low":
                base_score -= 0.1
        
        elif factor_name == "budget_adequacy":
            if recommendation.cost_estimate and context.resource_availability.get("budget"):
                budget_ratio = recommendation.cost_estimate / context.resource_availability["budget"]
                if budget_ratio < 0.5:
                    base_score += 0.2
                elif budget_ratio > 1.0:
                    base_score -= 0.2
        
        elif factor_name == "stakeholder_buy_in":
            if context.stakeholder_support:
                avg_support = sum(context.stakeholder_support.values()) / len(context.stakeholder_support)
                base_score = avg_support
        
        # Adjust based on organizational capabilities
        if factor_name in context.organizational_capabilities:
            base_score = (base_score + context.organizational_capabilities[factor_name]) / 2
        
        return max(0.0, min(1.0, base_score))
    
    async def _calculate_overall_probability(
        self,
        success_factors: List[SuccessFactor],
        recommendation: Recommendation,
        context: PredictionContext
    ) -> float:
        """Calculate overall success probability."""
        # Weighted average of success factors
        weighted_sum = sum(factor.weight * factor.score for factor in success_factors)
        total_weight = sum(factor.weight for factor in success_factors)
        
        if total_weight == 0:
            base_probability = 0.5
        else:
            base_probability = weighted_sum / total_weight
        
        # Adjust based on historical data
        historical_rate = self.historical_data["success_by_category"].get(
            self._get_recommendation_type(recommendation), 0.6
        )
        
        # Combine base probability with historical rate
        combined_probability = (base_probability * 0.7) + (historical_rate * 0.3)
        
        # Adjust based on priority and effort
        priority_adjustment = {
            "critical": 0.1,
            "high": 0.05,
            "medium": 0.0,
            "low": -0.05
        }.get(recommendation.priority.value, 0.0)
        
        effort_adjustment = {
            "low": 0.05,
            "medium": 0.0,
            "high": -0.05
        }.get(recommendation.implementation_effort, 0.0)
        
        final_probability = combined_probability + priority_adjustment + effort_adjustment
        
        return max(0.0, min(1.0, final_probability))
    
    def _determine_success_level(self, probability: float) -> SuccessLevel:
        """Determine success level based on probability."""
        if probability >= 0.8:
            return SuccessLevel.EXCELLENT
        elif probability >= 0.7:
            return SuccessLevel.GOOD
        elif probability >= 0.6:
            return SuccessLevel.MODERATE
        elif probability >= 0.4:
            return SuccessLevel.POOR
        else:
            return SuccessLevel.FAILURE
    
    async def _determine_confidence_level(
        self,
        success_factors: List[SuccessFactor],
        context: PredictionContext
    ) -> PredictionConfidence:
        """Determine confidence level for the prediction."""
        # Calculate confidence based on data quality and factor coverage
        factor_coverage = len(success_factors) / 6  # Assuming 6 is ideal number of factors
        
        # Check data availability
        data_quality = 0.0
        if context.historical_success_rates:
            data_quality += 0.3
        if context.organizational_capabilities:
            data_quality += 0.3
        if context.stakeholder_support:
            data_quality += 0.2
        if context.market_conditions:
            data_quality += 0.2
        
        overall_confidence = (factor_coverage + data_quality) / 2
        
        if overall_confidence >= 0.7:
            return PredictionConfidence.HIGH
        elif overall_confidence >= 0.5:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW
    
    async def _generate_outcome_predictions(
        self,
        recommendation: Recommendation,
        success_factors: List[SuccessFactor],
        context: PredictionContext
    ) -> List[OutcomePrediction]:
        """Generate predictions for different outcome types."""
        predictions = []
        
        # Financial outcome prediction
        financial_prediction = await self._predict_financial_outcome(
            recommendation, success_factors, context
        )
        predictions.append(financial_prediction)
        
        # Operational outcome prediction
        operational_prediction = await self._predict_operational_outcome(
            recommendation, success_factors, context
        )
        predictions.append(operational_prediction)
        
        # Strategic outcome prediction
        strategic_prediction = await self._predict_strategic_outcome(
            recommendation, success_factors, context
        )
        predictions.append(strategic_prediction)
        
        return predictions
    
    async def _predict_financial_outcome(
        self,
        recommendation: Recommendation,
        success_factors: List[SuccessFactor],
        context: PredictionContext
    ) -> OutcomePrediction:
        """Predict financial outcome."""
        # Calculate financial success probability
        financial_factors = [f for f in success_factors if "budget" in f.name or "cost" in f.name]
        if financial_factors:
            financial_probability = sum(f.score for f in financial_factors) / len(financial_factors)
        else:
            financial_probability = 0.6
        
        # Estimate financial impact
        base_impact = recommendation.cost_estimate or 10000
        expected_value = base_impact * 0.15  # Assume 15% ROI
        best_case = expected_value * 1.5
        worst_case = expected_value * 0.5
        
        return OutcomePrediction(
            outcome_type=OutcomeType.FINANCIAL,
            success_probability=financial_probability,
            confidence_level=PredictionConfidence.MEDIUM,
            expected_value=expected_value,
            best_case_scenario=best_case,
            worst_case_scenario=worst_case,
            factors=financial_factors,
            timeline_months=12
        )
    
    async def _predict_operational_outcome(
        self,
        recommendation: Recommendation,
        success_factors: List[SuccessFactor],
        context: PredictionContext
    ) -> OutcomePrediction:
        """Predict operational outcome."""
        # Calculate operational success probability
        operational_factors = [f for f in success_factors if "process" in f.name or "efficiency" in f.name]
        if operational_factors:
            operational_probability = sum(f.score for f in operational_factors) / len(operational_factors)
        else:
            operational_probability = 0.65
        
        return OutcomePrediction(
            outcome_type=OutcomeType.OPERATIONAL,
            success_probability=operational_probability,
            confidence_level=PredictionConfidence.MEDIUM,
            factors=operational_factors,
            timeline_months=6
        )
    
    async def _predict_strategic_outcome(
        self,
        recommendation: Recommendation,
        success_factors: List[SuccessFactor],
        context: PredictionContext
    ) -> OutcomePrediction:
        """Predict strategic outcome."""
        # Calculate strategic success probability
        strategic_factors = [f for f in success_factors if "strategy" in f.name or "market" in f.name]
        if strategic_factors:
            strategic_probability = sum(f.score for f in strategic_factors) / len(strategic_factors)
        else:
            strategic_probability = 0.55
        
        return OutcomePrediction(
            outcome_type=OutcomeType.STRATEGIC,
            success_probability=strategic_probability,
            confidence_level=PredictionConfidence.LOW,
            factors=strategic_factors,
            timeline_months=18
        )
    
    def _identify_key_success_factors(self, success_factors: List[SuccessFactor]) -> List[SuccessFactor]:
        """Identify key success factors (top 3 by weight)."""
        sorted_factors = sorted(success_factors, key=lambda x: x.weight, reverse=True)
        return sorted_factors[:3]
    
    async def _identify_risk_factors(
        self,
        recommendation: Recommendation,
        context: PredictionContext
    ) -> List[str]:
        """Identify risk factors that could impact success."""
        risks = []
        
        # Add recommendation-specific risks
        if recommendation.implementation_effort == "high":
            risks.append("High implementation complexity")
        
        if recommendation.priority.value == "critical":
            risks.append("Critical priority increases pressure")
        
        if recommendation.cost_estimate and context.resource_availability.get("budget"):
            if recommendation.cost_estimate > context.resource_availability["budget"]:
                risks.append("Budget constraints")
        
        # Add context-specific risks
        if context.market_conditions.get("volatility", 0) > 0.7:
            risks.append("Market volatility")
        
        if not context.stakeholder_support:
            risks.append("Limited stakeholder support")
        
        return risks
    
    async def _generate_success_recommendations(
        self,
        recommendation: Recommendation,
        success_factors: List[SuccessFactor],
        risk_factors: List[str]
    ) -> List[str]:
        """Generate recommendations to improve success probability."""
        recommendations = []
        
        # Address low-scoring success factors
        low_scoring_factors = [f for f in success_factors if f.score < 0.5]
        for factor in low_scoring_factors:
            if factor.name == "technical_expertise":
                recommendations.append("Invest in technical training and expertise")
            elif factor.name == "change_management":
                recommendations.append("Develop comprehensive change management plan")
            elif factor.name == "stakeholder_buy_in":
                recommendations.append("Increase stakeholder engagement and communication")
            elif factor.name == "budget_adequacy":
                recommendations.append("Secure additional budget or reduce scope")
        
        # Address risk factors
        for risk in risk_factors:
            if "budget" in risk.lower():
                recommendations.append("Implement cost control measures")
            elif "stakeholder" in risk.lower():
                recommendations.append("Strengthen stakeholder relationships")
            elif "complexity" in risk.lower():
                recommendations.append("Break down implementation into smaller phases")
        
        return recommendations
    
    def _estimate_timeline(self, recommendation: Recommendation) -> int:
        """Estimate timeline for success measurement."""
        timeline_mapping = {
            "1-3 months": 3,
            "3-6 months": 6,
            "6-12 months": 12
        }
        
        return timeline_mapping.get(recommendation.time_to_implement, 12)
    
    async def get_prediction_report(self, prediction: SuccessPrediction) -> Dict[str, Any]:
        """Generate a comprehensive prediction report."""
        report = {
            "prediction_summary": {
                "success_probability": f"{prediction.overall_success_probability:.1%}",
                "success_level": prediction.success_level.value,
                "confidence_level": prediction.confidence_level.value,
                "timeline_months": prediction.timeline_months
            },
            "outcome_predictions": [
                {
                    "outcome_type": pred.outcome_type.value,
                    "success_probability": f"{pred.success_probability:.1%}",
                    "confidence_level": pred.confidence_level.value,
                    "timeline_months": pred.timeline_months
                }
                for pred in prediction.outcome_predictions
            ],
            "key_success_factors": [
                {
                    "name": factor.name,
                    "weight": factor.weight,
                    "score": f"{factor.score:.2f}",
                    "impact": factor.impact
                }
                for factor in prediction.key_success_factors
            ],
            "risk_factors": prediction.risk_factors,
            "recommendations_for_success": prediction.recommendations_for_success
        }
        
        return report
