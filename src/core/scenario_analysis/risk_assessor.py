"""
Risk Assessor Component

Provides scenario risk evaluation and scoring capabilities.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class RiskCategory(Enum):
    """Categories of risks that can be assessed."""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    COMPLIANCE = "compliance"
    TECHNICAL = "technical"
    MARKET = "market"
    REGULATORY = "regulatory"
    REPUTATIONAL = "reputational"


class RiskLevel(Enum):
    """Risk levels for assessment."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


@dataclass
class RiskFactor:
    """Represents a specific risk factor."""
    name: str
    category: RiskCategory
    probability: float  # 0.0 to 1.0
    impact: float  # 0.0 to 1.0
    risk_score: float = 0.0
    description: str = ""
    mitigation_strategies: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    
    def __post_init__(self):
        """Calculate risk score after initialization."""
        self.risk_score = self.probability * self.impact
        self.risk_level = self._determine_risk_level()
    
    def _determine_risk_level(self) -> RiskLevel:
        """Determine risk level based on risk score."""
        if self.risk_score < 0.1:
            return RiskLevel.VERY_LOW
        elif self.risk_score < 0.25:
            return RiskLevel.LOW
        elif self.risk_score < 0.5:
            return RiskLevel.MEDIUM
        elif self.risk_score < 0.75:
            return RiskLevel.HIGH
        elif self.risk_score < 0.9:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.CRITICAL


@dataclass
class RiskAssessment:
    """Complete risk assessment result."""
    scenario_id: str
    assessment_timestamp: datetime = field(default_factory=datetime.now)
    risk_factors: Dict[str, RiskFactor] = field(default_factory=dict)
    category_risks: Dict[RiskCategory, List[RiskFactor]] = field(default_factory=dict)
    overall_risk_score: float = 0.0
    overall_risk_level: RiskLevel = RiskLevel.LOW
    high_risk_factors: List[RiskFactor] = field(default_factory=list)
    mitigation_priorities: List[str] = field(default_factory=list)
    risk_trends: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment to dictionary for serialization."""
        return {
            'scenario_id': self.scenario_id,
            'assessment_timestamp': self.assessment_timestamp.isoformat(),
            'risk_factors': {
                name: {
                    'name': factor.name,
                    'category': factor.category.value,
                    'probability': factor.probability,
                    'impact': factor.impact,
                    'risk_score': factor.risk_score,
                    'description': factor.description,
                    'mitigation_strategies': factor.mitigation_strategies,
                    'risk_level': factor.risk_level.value
                }
                for name, factor in self.risk_factors.items()
            },
            'overall_risk_score': self.overall_risk_score,
            'overall_risk_level': self.overall_risk_level.value,
            'high_risk_factors': [
                {
                    'name': factor.name,
                    'category': factor.category.value,
                    'risk_score': factor.risk_score,
                    'risk_level': factor.risk_level.value
                }
                for factor in self.high_risk_factors
            ],
            'mitigation_priorities': self.mitigation_priorities,
            'risk_trends': self.risk_trends
        }


class RiskAssessor:
    """
    Scenario risk evaluation and scoring system.
    
    Provides capabilities for:
    - Identifying and quantifying risk factors
    - Calculating risk scores and levels
    - Generating mitigation strategies
    - Risk trend analysis
    """
    
    def __init__(self):
        self.risk_models: Dict[str, Any] = {}
        self.historical_data: Dict[str, Any] = {}
        self.risk_thresholds: Dict[RiskLevel, float] = {
            RiskLevel.VERY_LOW: 0.1,
            RiskLevel.LOW: 0.25,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.75,
            RiskLevel.VERY_HIGH: 0.9,
            RiskLevel.CRITICAL: 1.0
        }
    
    def assess_risk(self, scenario: 'Scenario',
                   baseline_data: Optional[Dict[str, Any]] = None) -> RiskAssessment:
        """
        Assess risks for a scenario.
        
        Args:
            scenario: Scenario to assess
            baseline_data: Optional baseline data for comparison
            
        Returns:
            Risk assessment result
        """
        logger.info(f"Starting risk assessment for scenario: {scenario.name}")
        
        # Initialize assessment
        assessment = RiskAssessment(scenario_id=scenario.id)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(scenario, baseline_data)
        assessment.risk_factors = risk_factors
        
        # Categorize risks
        assessment.category_risks = self._categorize_risks(risk_factors)
        
        # Calculate overall risk score
        assessment.overall_risk_score = self._calculate_overall_risk_score(risk_factors)
        assessment.overall_risk_level = self._determine_overall_risk_level(
            assessment.overall_risk_score
        )
        
        # Identify high-risk factors
        assessment.high_risk_factors = self._identify_high_risk_factors(risk_factors)
        
        # Generate mitigation priorities
        assessment.mitigation_priorities = self._generate_mitigation_priorities(assessment)
        
        # Analyze risk trends
        assessment.risk_trends = self._analyze_risk_trends(scenario, risk_factors)
        
        logger.info(f"Completed risk assessment for scenario: {scenario.name}")
        return assessment
    
    def _identify_risk_factors(self, scenario: 'Scenario',
                              baseline_data: Optional[Dict[str, Any]] = None) -> Dict[str, RiskFactor]:
        """Identify risk factors for a scenario."""
        risk_factors = {}
        
        # Analyze each parameter for potential risks
        for param_name, parameter in scenario.parameters.items():
            risk_factor = self._analyze_parameter_risk(parameter, baseline_data)
            if risk_factor:
                risk_factors[param_name] = risk_factor
        
        # Add scenario-level risks
        scenario_risks = self._analyze_scenario_risks(scenario)
        risk_factors.update(scenario_risks)
        
        return risk_factors
    
    def _analyze_parameter_risk(self, parameter: 'ScenarioParameter',
                               baseline_data: Optional[Dict[str, Any]] = None) -> Optional[RiskFactor]:
        """Analyze risk for a specific parameter."""
        try:
            # Determine risk category based on parameter type
            category = self._determine_parameter_risk_category(parameter)
            
            # Calculate probability and impact
            probability = self._calculate_risk_probability(parameter, baseline_data)
            impact = self._calculate_risk_impact(parameter, baseline_data)
            
            # Generate description
            description = self._generate_risk_description(parameter, probability, impact)
            
            # Generate mitigation strategies
            mitigation_strategies = self._generate_mitigation_strategies(parameter, category)
            
            risk_factor = RiskFactor(
                name=f"Parameter Risk: {parameter.name}",
                category=category,
                probability=probability,
                impact=impact,
                description=description,
                mitigation_strategies=mitigation_strategies
            )
            
            return risk_factor
            
        except Exception as e:
            logger.warning(f"Error analyzing risk for parameter {parameter.name}: {e}")
            return None
    
    def _determine_parameter_risk_category(self, parameter: 'ScenarioParameter') -> RiskCategory:
        """Determine risk category for a parameter."""
        # This is a simplified mapping - in practice, you might have more sophisticated logic
        if parameter.parameter_type.value in ['numerical', 'percentage']:
            return RiskCategory.FINANCIAL
        elif parameter.parameter_type.value == 'time_series':
            return RiskCategory.OPERATIONAL
        elif parameter.parameter_type.value == 'categorical':
            return RiskCategory.STRATEGIC
        else:
            return RiskCategory.TECHNICAL
    
    def _calculate_risk_probability(self, parameter: 'ScenarioParameter',
                                  baseline_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate probability of risk occurrence."""
        # Simplified probability calculation
        # In practice, this would use historical data, expert judgment, or statistical models
        
        # Base probability on parameter type and value changes
        base_probability = 0.1
        
        # Adjust based on parameter type
        if parameter.parameter_type.value == 'numerical':
            if parameter.min_value is not None and parameter.max_value is not None:
                # Higher probability if value is outside expected range
                if not (parameter.min_value <= parameter.scenario_value <= parameter.max_value):
                    base_probability += 0.3
            else:
                # Higher probability for larger changes
                if isinstance(parameter.current_value, (int, float)) and isinstance(parameter.scenario_value, (int, float)):
                    change_ratio = abs(parameter.scenario_value - parameter.current_value) / max(abs(parameter.current_value), 1)
                    base_probability += min(change_ratio * 0.5, 0.4)
        
        # Adjust based on confidence level
        base_probability += (1 - parameter.confidence_level) * 0.2
        
        return min(base_probability, 1.0)
    
    def _calculate_risk_impact(self, parameter: 'ScenarioParameter',
                              baseline_data: Optional[Dict[str, Any]] = None) -> float:
        """Calculate potential impact of risk."""
        # Simplified impact calculation
        # In practice, this would use impact models, expert judgment, or historical data
        
        base_impact = 0.2
        
        # Adjust based on parameter type
        if parameter.parameter_type.value == 'numerical':
            if isinstance(parameter.current_value, (int, float)) and isinstance(parameter.scenario_value, (int, float)):
                if parameter.current_value != 0:
                    change_ratio = abs(parameter.scenario_value - parameter.current_value) / abs(parameter.current_value)
                    base_impact += min(change_ratio * 0.3, 0.6)
        
        # Adjust based on parameter importance (could be determined by metadata)
        if parameter.description and any(keyword in parameter.description.lower() 
                                        for keyword in ['critical', 'important', 'key']):
            base_impact += 0.2
        
        return min(base_impact, 1.0)
    
    def _generate_risk_description(self, parameter: 'ScenarioParameter',
                                  probability: float, impact: float) -> str:
        """Generate description for risk factor."""
        risk_level = "low"
        if probability * impact > 0.5:
            risk_level = "high"
        elif probability * impact > 0.25:
            risk_level = "medium"
        
        return (f"{risk_level.capitalize()} risk associated with parameter '{parameter.name}' "
                f"(probability: {probability:.2f}, impact: {impact:.2f})")
    
    def _generate_mitigation_strategies(self, parameter: 'ScenarioParameter',
                                       category: RiskCategory) -> List[str]:
        """Generate mitigation strategies for a risk factor."""
        strategies = []
        
        # General strategies
        strategies.append("Monitor parameter changes closely")
        strategies.append("Establish early warning systems")
        
        # Category-specific strategies
        if category == RiskCategory.FINANCIAL:
            strategies.append("Implement financial controls and limits")
            strategies.append("Diversify financial exposures")
        elif category == RiskCategory.OPERATIONAL:
            strategies.append("Develop contingency plans")
            strategies.append("Cross-train personnel")
        elif category == RiskCategory.STRATEGIC:
            strategies.append("Conduct regular strategic reviews")
            strategies.append("Maintain strategic flexibility")
        
        # Parameter-specific strategies
        if parameter.parameter_type.value == 'numerical':
            strategies.append("Set parameter bounds and alerts")
        
        return strategies
    
    def _analyze_scenario_risks(self, scenario: 'Scenario') -> Dict[str, RiskFactor]:
        """Analyze scenario-level risks."""
        scenario_risks = {}
        
        # Risk based on scenario type
        if scenario.scenario_type.value == 'stress_test':
            scenario_risks['stress_test'] = RiskFactor(
                name="Stress Test Scenario Risk",
                category=RiskCategory.STRATEGIC,
                probability=0.8,
                impact=0.9,
                description="High probability of extreme conditions in stress test scenario",
                mitigation_strategies=[
                    "Ensure robust stress testing methodology",
                    "Validate stress test assumptions",
                    "Prepare for extreme outcomes"
                ]
            )
        
        # Risk based on parameter count
        if len(scenario.parameters) > 10:
            scenario_risks['complexity'] = RiskFactor(
                name="Scenario Complexity Risk",
                category=RiskCategory.OPERATIONAL,
                probability=0.6,
                impact=0.4,
                description="High complexity scenario with many parameters",
                mitigation_strategies=[
                    "Simplify scenario structure",
                    "Focus on key parameters",
                    "Use scenario templates"
                ]
            )
        
        return scenario_risks
    
    def _categorize_risks(self, risk_factors: Dict[str, RiskFactor]) -> Dict[RiskCategory, List[RiskFactor]]:
        """Categorize risk factors by category."""
        categorized = {category: [] for category in RiskCategory}
        
        for factor in risk_factors.values():
            categorized[factor.category].append(factor)
        
        return categorized
    
    def _calculate_overall_risk_score(self, risk_factors: Dict[str, RiskFactor]) -> float:
        """Calculate overall risk score."""
        if not risk_factors:
            return 0.0
        
        # Weighted average of risk scores
        total_score = sum(factor.risk_score for factor in risk_factors.values())
        return total_score / len(risk_factors)
    
    def _determine_overall_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine overall risk level."""
        if risk_score < 0.1:
            return RiskLevel.VERY_LOW
        elif risk_score < 0.25:
            return RiskLevel.LOW
        elif risk_score < 0.5:
            return RiskLevel.MEDIUM
        elif risk_score < 0.75:
            return RiskLevel.HIGH
        elif risk_score < 0.9:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _identify_high_risk_factors(self, risk_factors: Dict[str, RiskFactor]) -> List[RiskFactor]:
        """Identify high-risk factors that need immediate attention."""
        high_risk_factors = []
        
        for factor in risk_factors.values():
            if factor.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.CRITICAL]:
                high_risk_factors.append(factor)
        
        # Sort by risk score (highest first)
        high_risk_factors.sort(key=lambda x: x.risk_score, reverse=True)
        
        return high_risk_factors
    
    def _generate_mitigation_priorities(self, assessment: RiskAssessment) -> List[str]:
        """Generate prioritized mitigation strategies."""
        priorities = []
        
        # Critical risks first
        critical_risks = [f for f in assessment.high_risk_factors 
                         if f.risk_level == RiskLevel.CRITICAL]
        for risk in critical_risks:
            priorities.extend(risk.mitigation_strategies[:2])  # Top 2 strategies
        
        # High risks next
        high_risks = [f for f in assessment.high_risk_factors 
                     if f.risk_level == RiskLevel.HIGH]
        for risk in high_risks:
            priorities.extend(risk.mitigation_strategies[:1])  # Top strategy
        
        # Remove duplicates while preserving order
        seen = set()
        unique_priorities = []
        for priority in priorities:
            if priority not in seen:
                seen.add(priority)
                unique_priorities.append(priority)
        
        return unique_priorities
    
    def _analyze_risk_trends(self, scenario: 'Scenario',
                           risk_factors: Dict[str, RiskFactor]) -> Dict[str, str]:
        """Analyze risk trends."""
        trends = {}
        
        # Analyze by category
        category_risks = self._categorize_risks(risk_factors)
        for category, factors in category_risks.items():
            if factors:
                avg_score = np.mean([f.risk_score for f in factors])
                if avg_score > 0.7:
                    trends[category.value] = "increasing"
                elif avg_score > 0.4:
                    trends[category.value] = "stable"
                else:
                    trends[category.value] = "decreasing"
        
        # Overall trend
        if risk_factors:
            overall_score = self._calculate_overall_risk_score(risk_factors)
            if overall_score > 0.7:
                trends['overall'] = "increasing"
            elif overall_score > 0.4:
                trends['overall'] = "stable"
            else:
                trends['overall'] = "decreasing"
        
        return trends
    
    def compare_risk_assessments(self, assessments: List[RiskAssessment]) -> Dict[str, Any]:
        """
        Compare risk assessments across multiple scenarios.
        
        Args:
            assessments: List of risk assessments to compare
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(assessments)} risk assessments")
        
        comparison = {
            'scenario_rankings': [],
            'highest_risk_scenarios': [],
            'risk_distribution': {},
            'common_risks': []
        }
        
        # Rank scenarios by risk
        scenario_rankings = [(a.scenario_id, a.overall_risk_score) for a in assessments]
        scenario_rankings.sort(key=lambda x: x[1], reverse=True)
        comparison['scenario_rankings'] = scenario_rankings
        
        # Identify highest risk scenarios
        high_risk_threshold = 0.7
        comparison['highest_risk_scenarios'] = [
            a.scenario_id for a in assessments 
            if a.overall_risk_score > high_risk_threshold
        ]
        
        # Analyze risk distribution
        risk_levels = [a.overall_risk_level.value for a in assessments]
        comparison['risk_distribution'] = {
            level: risk_levels.count(level) for level in set(risk_levels)
        }
        
        # Identify common risks
        all_risk_factors = []
        for assessment in assessments:
            all_risk_factors.extend(assessment.risk_factors.values())
        
        # Find risks that appear in multiple scenarios
        risk_names = [f.name for f in all_risk_factors]
        common_risks = [name for name in set(risk_names) 
                       if risk_names.count(name) > 1]
        comparison['common_risks'] = common_risks
        
        return comparison
