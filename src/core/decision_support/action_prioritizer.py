"""
Action Prioritizer Component

Provides intelligent prioritization and ranking of recommendations
based on multiple factors including impact, effort, risk, and urgency.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid
from datetime import datetime

from .recommendation_engine import Recommendation, RecommendationPriority

logger = logging.getLogger(__name__)


class PriorityMethod(Enum):
    """Methods for calculating priority scores."""
    WEIGHTED_SUM = "weighted_sum"
    MULTIPLICATIVE = "multiplicative"
    HIERARCHICAL = "hierarchical"
    MACHINE_LEARNING = "machine_learning"


class PriorityFactor(Enum):
    """Factors considered in priority calculation."""
    IMPACT = "impact"
    EFFORT = "effort"
    URGENCY = "urgency"
    RISK = "risk"
    RESOURCE_AVAILABILITY = "resource_availability"
    STAKEHOLDER_ALIGNMENT = "stakeholder_alignment"
    STRATEGIC_ALIGNMENT = "strategic_alignment"


@dataclass
class PriorityWeights:
    """Weights for different priority factors."""
    impact_weight: float = 0.25
    effort_weight: float = 0.20
    urgency_weight: float = 0.20
    risk_weight: float = 0.15
    resource_availability_weight: float = 0.10
    stakeholder_alignment_weight: float = 0.05
    strategic_alignment_weight: float = 0.05
    
    def validate(self) -> bool:
        """Validate that weights sum to 1.0."""
        total = sum([
            self.impact_weight,
            self.effort_weight,
            self.urgency_weight,
            self.risk_weight,
            self.resource_availability_weight,
            self.stakeholder_alignment_weight,
            self.strategic_alignment_weight
        ])
        return abs(total - 1.0) < 0.01


@dataclass
class PrioritizedAction:
    """Represents a prioritized action with detailed scoring."""
    recommendation: Recommendation
    priority_score: float = 0.0
    factor_scores: Dict[str, float] = field(default_factory=dict)
    rank: int = 0
    justification: str = ""
    dependencies: List[str] = field(default_factory=list)
    estimated_start_date: Optional[datetime] = None
    estimated_completion_date: Optional[datetime] = None
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'recommendation': self.recommendation.to_dict(),
            'priority_score': self.priority_score,
            'factor_scores': self.factor_scores,
            'rank': self.rank,
            'justification': self.justification,
            'dependencies': self.dependencies,
            'estimated_start_date': self.estimated_start_date.isoformat() if self.estimated_start_date else None,
            'estimated_completion_date': self.estimated_completion_date.isoformat() if self.estimated_completion_date else None,
            'resource_requirements': self.resource_requirements
        }


@dataclass
class PrioritizationContext:
    """Context information for prioritization."""
    available_resources: Dict[str, Any] = field(default_factory=dict)
    time_constraints: Dict[str, Any] = field(default_factory=dict)
    stakeholder_preferences: Dict[str, Any] = field(default_factory=dict)
    strategic_goals: List[str] = field(default_factory=list)
    risk_tolerance: str = "medium"
    budget_constraints: Optional[float] = None
    team_capacity: Dict[str, int] = field(default_factory=dict)


class ActionPrioritizer:
    """Intelligent action prioritization engine."""
    
    def __init__(self, method: PriorityMethod = PriorityMethod.WEIGHTED_SUM):
        self.method = method
        self.default_weights = PriorityWeights()
        self.default_weights.validate()
        logger.info(f"Initialized ActionPrioritizer with method: {method.value}")
    
    async def prioritize_actions(
        self,
        recommendations: List[Recommendation],
        context: PrioritizationContext,
        custom_weights: Optional[PriorityWeights] = None
    ) -> List[PrioritizedAction]:
        """Prioritize a list of recommendations based on context."""
        try:
            logger.info(f"Prioritizing {len(recommendations)} recommendations")
            
            weights = custom_weights or self.default_weights
            if not weights.validate():
                raise ValueError("Priority weights must sum to 1.0")
            
            prioritized_actions = []
            
            for recommendation in recommendations:
                # Calculate factor scores
                factor_scores = await self._calculate_factor_scores(
                    recommendation, context
                )
                
                # Calculate overall priority score
                priority_score = await self._calculate_priority_score(
                    factor_scores, weights
                )
                
                # Create prioritized action
                prioritized_action = PrioritizedAction(
                    recommendation=recommendation,
                    priority_score=priority_score,
                    factor_scores=factor_scores,
                    justification=await self._generate_justification(
                        recommendation, factor_scores
                    ),
                    dependencies=await self._identify_dependencies(recommendation),
                    estimated_start_date=await self._estimate_start_date(
                        recommendation, context
                    ),
                    estimated_completion_date=await self._estimate_completion_date(
                        recommendation, context
                    ),
                    resource_requirements=await self._estimate_resources(
                        recommendation, context
                    )
                )
                
                prioritized_actions.append(prioritized_action)
            
            # Sort by priority score and assign ranks
            sorted_actions = sorted(
                prioritized_actions,
                key=lambda x: x.priority_score,
                reverse=True
            )
            
            for i, action in enumerate(sorted_actions):
                action.rank = i + 1
            
            logger.info(f"Successfully prioritized {len(sorted_actions)} actions")
            return sorted_actions
            
        except Exception as e:
            logger.error(f"Error prioritizing actions: {e}")
            return []
    
    async def _calculate_factor_scores(
        self,
        recommendation: Recommendation,
        context: PrioritizationContext
    ) -> Dict[str, float]:
        """Calculate individual factor scores for a recommendation."""
        scores = {}
        
        # Impact score (0-1)
        scores['impact'] = await self._calculate_impact_score(recommendation)
        
        # Effort score (0-1, inverted so lower effort = higher score)
        scores['effort'] = await self._calculate_effort_score(recommendation)
        
        # Urgency score (0-1)
        scores['urgency'] = await self._calculate_urgency_score(recommendation)
        
        # Risk score (0-1, inverted so lower risk = higher score)
        scores['risk'] = await self._calculate_risk_score(recommendation)
        
        # Resource availability score (0-1)
        scores['resource_availability'] = await self._calculate_resource_score(
            recommendation, context
        )
        
        # Stakeholder alignment score (0-1)
        scores['stakeholder_alignment'] = await self._calculate_stakeholder_score(
            recommendation, context
        )
        
        # Strategic alignment score (0-1)
        scores['strategic_alignment'] = await self._calculate_strategic_score(
            recommendation, context
        )
        
        return scores
    
    async def _calculate_impact_score(self, recommendation: Recommendation) -> float:
        """Calculate impact score based on expected impact."""
        if not recommendation.expected_impact:
            return 0.5
        
        impact_values = recommendation.expected_impact.values()
        high_impact_count = sum(1 for impact in impact_values if impact in ['high', 'improved', 'enhanced'])
        total_impacts = len(impact_values)
        
        return high_impact_count / total_impacts if total_impacts > 0 else 0.5
    
    async def _calculate_effort_score(self, recommendation: Recommendation) -> float:
        """Calculate effort score (inverted)."""
        effort_scores = {
            'low': 1.0,
            'medium': 0.6,
            'high': 0.3
        }
        return effort_scores.get(recommendation.implementation_effort, 0.6)
    
    async def _calculate_urgency_score(self, recommendation: Recommendation) -> float:
        """Calculate urgency score based on priority and time constraints."""
        priority_scores = {
            RecommendationPriority.CRITICAL: 1.0,
            RecommendationPriority.HIGH: 0.8,
            RecommendationPriority.MEDIUM: 0.6,
            RecommendationPriority.LOW: 0.4
        }
        return priority_scores.get(recommendation.priority, 0.6)
    
    async def _calculate_risk_score(self, recommendation: Recommendation) -> float:
        """Calculate risk score (inverted)."""
        risk_scores = {
            'low': 1.0,
            'medium': 0.7,
            'high': 0.4
        }
        return risk_scores.get(recommendation.risk_level, 0.7)
    
    async def _calculate_resource_score(
        self,
        recommendation: Recommendation,
        context: PrioritizationContext
    ) -> float:
        """Calculate resource availability score."""
        # Simple implementation - in production would check actual resource availability
        if context.budget_constraints and recommendation.cost_estimate:
            if recommendation.cost_estimate <= context.budget_constraints:
                return 1.0
            else:
                return 0.3
        
        return 0.7  # Default assumption
    
    async def _calculate_stakeholder_score(
        self,
        recommendation: Recommendation,
        context: PrioritizationContext
    ) -> float:
        """Calculate stakeholder alignment score."""
        # Simple implementation - in production would check actual stakeholder preferences
        if context.stakeholder_preferences:
            # Check if recommendation aligns with stakeholder preferences
            return 0.8  # Default assumption
        return 0.6
    
    async def _calculate_strategic_score(
        self,
        recommendation: Recommendation,
        context: PrioritizationContext
    ) -> float:
        """Calculate strategic alignment score."""
        if not context.strategic_goals:
            return 0.6
        
        # Check if recommendation type aligns with strategic goals
        if recommendation.recommendation_type.value in ['strategic', 'market_strategy']:
            return 0.9
        elif recommendation.recommendation_type.value in ['operational', 'tactical']:
            return 0.7
        else:
            return 0.5
    
    async def _calculate_priority_score(
        self,
        factor_scores: Dict[str, float],
        weights: PriorityWeights
    ) -> float:
        """Calculate overall priority score using weighted sum method."""
        score = (
            factor_scores['impact'] * weights.impact_weight +
            factor_scores['effort'] * weights.effort_weight +
            factor_scores['urgency'] * weights.urgency_weight +
            factor_scores['risk'] * weights.risk_weight +
            factor_scores['resource_availability'] * weights.resource_availability_weight +
            factor_scores['stakeholder_alignment'] * weights.stakeholder_alignment_weight +
            factor_scores['strategic_alignment'] * weights.strategic_alignment_weight
        )
        
        return min(score, 1.0)
    
    async def _generate_justification(
        self,
        recommendation: Recommendation,
        factor_scores: Dict[str, float]
    ) -> str:
        """Generate justification for the priority score."""
        top_factors = sorted(
            factor_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        justification_parts = []
        for factor, score in top_factors:
            if score > 0.8:
                justification_parts.append(f"High {factor.replace('_', ' ')}")
            elif score > 0.6:
                justification_parts.append(f"Good {factor.replace('_', ' ')}")
        
        if justification_parts:
            return f"Prioritized due to: {', '.join(justification_parts)}"
        else:
            return "Standard priority based on balanced factors"
    
    async def _identify_dependencies(self, recommendation: Recommendation) -> List[str]:
        """Identify dependencies for the recommendation."""
        dependencies = []
        
        # Add prerequisites as dependencies
        dependencies.extend(recommendation.prerequisites)
        
        # Add category-specific dependencies
        if recommendation.category.value == 'technology_adoption':
            dependencies.append("IT infrastructure assessment")
        elif recommendation.category.value == 'process_improvement':
            dependencies.append("Current process documentation")
        
        return dependencies
    
    async def _estimate_start_date(
        self,
        recommendation: Recommendation,
        context: PrioritizationContext
    ) -> Optional[datetime]:
        """Estimate start date based on dependencies and resources."""
        # Simple implementation - in production would consider actual scheduling
        base_delay_days = {
            'low': 7,
            'medium': 14,
            'high': 30
        }
        
        delay_days = base_delay_days.get(recommendation.implementation_effort, 14)
        return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    async def _estimate_completion_date(
        self,
        recommendation: Recommendation,
        context: PrioritizationContext
    ) -> Optional[datetime]:
        """Estimate completion date based on implementation time."""
        # Simple implementation - in production would use more sophisticated estimation
        time_mapping = {
            '1-3 months': 60,
            '3-6 months': 135,
            '6-12 months': 270
        }
        
        days = time_mapping.get(recommendation.time_to_implement, 90)
        start_date = await self._estimate_start_date(recommendation, context)
        if start_date:
            from datetime import timedelta
            return start_date + timedelta(days=days)
        
        return None
    
    async def _estimate_resources(
        self,
        recommendation: Recommendation,
        context: PrioritizationContext
    ) -> Dict[str, Any]:
        """Estimate resource requirements."""
        resources = {
            'budget': recommendation.cost_estimate,
            'team_size': {
                'low': 1,
                'medium': 3,
                'high': 5
            }.get(recommendation.implementation_effort, 3),
            'skills_required': self._get_required_skills(recommendation),
            'tools_needed': self._get_required_tools(recommendation)
        }
        
        return resources
    
    def _get_required_skills(self, recommendation: Recommendation) -> List[str]:
        """Get required skills for implementation."""
        skill_mapping = {
            'technology_adoption': ['technical_implementation', 'change_management'],
            'process_improvement': ['process_analysis', 'project_management'],
            'risk_management': ['risk_assessment', 'compliance_knowledge'],
            'market_strategy': ['market_analysis', 'strategic_planning']
        }
        
        return skill_mapping.get(recommendation.category.value, ['project_management'])
    
    def _get_required_tools(self, recommendation: Recommendation) -> List[str]:
        """Get required tools for implementation."""
        tool_mapping = {
            'technology_adoption': ['development_environment', 'testing_tools'],
            'process_improvement': ['process_mapping_tools', 'analytics_platform'],
            'risk_management': ['risk_assessment_framework', 'monitoring_tools'],
            'market_strategy': ['market_research_tools', 'analytics_platform']
        }
        
        return tool_mapping.get(recommendation.category.value, ['project_management_tools'])
    
    async def get_prioritization_report(
        self,
        prioritized_actions: List[PrioritizedAction]
    ) -> Dict[str, Any]:
        """Generate a comprehensive prioritization report."""
        if not prioritized_actions:
            return {"error": "No actions to report on"}
        
        report = {
            "summary": {
                "total_actions": len(prioritized_actions),
                "high_priority_count": len([a for a in prioritized_actions if a.priority_score > 0.8]),
                "medium_priority_count": len([a for a in prioritized_actions if 0.6 <= a.priority_score <= 0.8]),
                "low_priority_count": len([a for a in prioritized_actions if a.priority_score < 0.6])
            },
            "top_priorities": [
                {
                    "rank": action.rank,
                    "title": action.recommendation.title,
                    "priority_score": action.priority_score,
                    "category": action.recommendation.category.value,
                    "justification": action.justification
                }
                for action in prioritized_actions[:5]
            ],
            "factor_analysis": self._analyze_factors(prioritized_actions),
            "resource_requirements": self._analyze_resources(prioritized_actions),
            "timeline": self._analyze_timeline(prioritized_actions)
        }
        
        return report
    
    def _analyze_factors(self, actions: List[PrioritizedAction]) -> Dict[str, Any]:
        """Analyze factor scores across all actions."""
        factor_averages = {}
        for factor in ['impact', 'effort', 'urgency', 'risk', 'resource_availability', 'stakeholder_alignment', 'strategic_alignment']:
            scores = [action.factor_scores.get(factor, 0) for action in actions]
            factor_averages[factor] = sum(scores) / len(scores) if scores else 0
        
        return factor_averages
    
    def _analyze_resources(self, actions: List[PrioritizedAction]) -> Dict[str, Any]:
        """Analyze resource requirements across all actions."""
        total_budget = sum(
            action.resource_requirements.get('budget', 0) or 0
            for action in actions
        )
        
        total_team_size = sum(
            action.resource_requirements.get('team_size', 0)
            for action in actions
        )
        
        return {
            "total_budget": total_budget,
            "total_team_size": total_team_size,
            "budget_distribution": self._get_budget_distribution(actions)
        }
    
    def _get_budget_distribution(self, actions: List[PrioritizedAction]) -> Dict[str, float]:
        """Get budget distribution by priority level."""
        high_priority_budget = sum(
            action.resource_requirements.get('budget', 0) or 0
            for action in actions if action.priority_score > 0.8
        )
        
        medium_priority_budget = sum(
            action.resource_requirements.get('budget', 0) or 0
            for action in actions if 0.6 <= action.priority_score <= 0.8
        )
        
        low_priority_budget = sum(
            action.resource_requirements.get('budget', 0) or 0
            for action in actions if action.priority_score < 0.6
        )
        
        total_budget = high_priority_budget + medium_priority_budget + low_priority_budget
        
        if total_budget > 0:
            return {
                "high_priority": high_priority_budget / total_budget,
                "medium_priority": medium_priority_budget / total_budget,
                "low_priority": low_priority_budget / total_budget
            }
        else:
            return {"high_priority": 0, "medium_priority": 0, "low_priority": 0}
    
    def _analyze_timeline(self, actions: List[PrioritizedAction]) -> Dict[str, Any]:
        """Analyze timeline across all actions."""
        completion_dates = [
            action.estimated_completion_date
            for action in actions
            if action.estimated_completion_date
        ]
        
        if completion_dates:
            earliest_completion = min(completion_dates)
            latest_completion = max(completion_dates)
            
            return {
                "earliest_completion": earliest_completion.isoformat(),
                "latest_completion": latest_completion.isoformat(),
                "average_duration_days": self._calculate_average_duration(actions)
            }
        else:
            return {"error": "No completion dates available"}
    
    def _calculate_average_duration(self, actions: List[PrioritizedAction]) -> float:
        """Calculate average duration in days."""
        durations = []
        for action in actions:
            if action.estimated_start_date and action.estimated_completion_date:
                duration = (action.estimated_completion_date - action.estimated_start_date).days
                durations.append(duration)
        
        return sum(durations) / len(durations) if durations else 0
