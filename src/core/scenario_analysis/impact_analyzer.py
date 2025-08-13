"""
Impact Analyzer Component

Provides scenario outcome prediction and impact quantification capabilities.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ImpactType(Enum):
    """Types of impacts that can be analyzed."""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    RISK = "risk"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"


class ImpactDirection(Enum):
    """Direction of impact."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class ImpactMetric:
    """Represents a specific impact metric."""
    name: str
    value: float
    unit: str
    baseline_value: float
    change_percentage: float
    impact_direction: ImpactDirection
    confidence_level: float = 0.95
    description: str = ""


@dataclass
class ImpactCategory:
    """Represents a category of impacts."""
    category_type: ImpactType
    metrics: List[ImpactMetric] = field(default_factory=list)
    total_impact_score: float = 0.0
    risk_level: str = "low"
    priority: str = "medium"


@dataclass
class ImpactAnalysis:
    """Complete impact analysis result."""
    scenario_id: str
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    impact_categories: Dict[ImpactType, ImpactCategory] = field(default_factory=dict)
    overall_impact_score: float = 0.0
    overall_risk_level: str = "low"
    confidence_level: float = 0.95
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary for serialization."""
        return {
            'scenario_id': self.scenario_id,
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'impact_categories': {
                category_type.value: {
                    'category_type': category_type.value,
                    'metrics': [
                        {
                            'name': metric.name,
                            'value': metric.value,
                            'unit': metric.unit,
                            'baseline_value': metric.baseline_value,
                            'change_percentage': metric.change_percentage,
                            'impact_direction': metric.impact_direction.value,
                            'confidence_level': metric.confidence_level,
                            'description': metric.description
                        }
                        for metric in category.metrics
                    ],
                    'total_impact_score': category.total_impact_score,
                    'risk_level': category.risk_level,
                    'priority': category.priority
                }
                for category_type, category in self.impact_categories.items()
            },
            'overall_impact_score': self.overall_impact_score,
            'overall_risk_level': self.overall_risk_level,
            'confidence_level': self.confidence_level,
            'assumptions': self.assumptions,
            'limitations': self.limitations,
            'recommendations': self.recommendations
        }


class ImpactAnalyzer:
    """
    Scenario outcome prediction and impact quantification system.
    
    Provides capabilities for:
    - Quantifying scenario impacts across multiple dimensions
    - Calculating confidence intervals for impact predictions
    - Generating impact-based recommendations
    - Risk assessment integration
    """
    
    def __init__(self):
        self.impact_models: Dict[str, Any] = {}
        self.baseline_data: Dict[str, Any] = {}
        self.impact_weights: Dict[ImpactType, float] = {
            ImpactType.FINANCIAL: 0.3,
            ImpactType.OPERATIONAL: 0.25,
            ImpactType.STRATEGIC: 0.2,
            ImpactType.RISK: 0.15,
            ImpactType.PERFORMANCE: 0.1
        }
    
    def analyze_impact(self, scenario: 'Scenario', 
                      baseline_data: Optional[Dict[str, Any]] = None,
                      custom_weights: Optional[Dict[ImpactType, float]] = None) -> ImpactAnalysis:
        """
        Analyze the impact of a scenario.
        
        Args:
            scenario: Scenario to analyze
            baseline_data: Optional baseline data for comparison
            custom_weights: Optional custom weights for impact categories
            
        Returns:
            Impact analysis result
        """
        logger.info(f"Starting impact analysis for scenario: {scenario.name}")
        
        # Use provided baseline data or default
        if baseline_data:
            self.baseline_data = baseline_data
        
        # Use custom weights if provided
        weights = custom_weights or self.impact_weights
        
        # Initialize analysis
        analysis = ImpactAnalysis(scenario_id=scenario.id)
        
        # Analyze each impact category
        for impact_type in ImpactType:
            category_analysis = self._analyze_category(
                scenario, impact_type, weights.get(impact_type, 0.1)
            )
            analysis.impact_categories[impact_type] = category_analysis
        
        # Calculate overall impact score
        analysis.overall_impact_score = self._calculate_overall_score(
            analysis.impact_categories, weights
        )
        
        # Determine overall risk level
        analysis.overall_risk_level = self._determine_risk_level(
            analysis.overall_impact_score
        )
        
        # Generate recommendations
        analysis.recommendations = self._generate_recommendations(analysis)
        
        # Add assumptions and limitations
        analysis.assumptions = self._identify_assumptions(scenario)
        analysis.limitations = self._identify_limitations(scenario)
        
        logger.info(f"Completed impact analysis for scenario: {scenario.name}")
        return analysis
    
    def _analyze_category(self, scenario: 'Scenario', 
                         impact_type: ImpactType, weight: float) -> ImpactCategory:
        """Analyze impact for a specific category."""
        category = ImpactCategory(category_type=impact_type)
        
        # Get baseline metrics for this category
        baseline_metrics = self.baseline_data.get(impact_type.value, {})
        
        # Analyze each parameter's impact
        for param_name, parameter in scenario.parameters.items():
            impact_metric = self._calculate_parameter_impact(
                parameter, baseline_metrics.get(param_name, {})
            )
            if impact_metric:
                category.metrics.append(impact_metric)
        
        # Calculate category impact score
        if category.metrics:
            category.total_impact_score = np.mean([
                abs(metric.change_percentage) for metric in category.metrics
            ]) * weight
        
        # Determine category risk level
        category.risk_level = self._determine_category_risk_level(category)
        
        # Determine category priority
        category.priority = self._determine_category_priority(category)
        
        return category
    
    def _calculate_parameter_impact(self, parameter: 'ScenarioParameter',
                                  baseline_data: Dict[str, Any]) -> Optional[ImpactMetric]:
        """Calculate impact for a specific parameter."""
        try:
            # Get baseline value
            baseline_value = baseline_data.get('value', parameter.current_value)
            
            # Calculate change
            if isinstance(parameter.scenario_value, (int, float)) and isinstance(baseline_value, (int, float)):
                if baseline_value != 0:
                    change_percentage = ((parameter.scenario_value - baseline_value) / baseline_value) * 100
                else:
                    change_percentage = 0.0
            else:
                # For non-numerical parameters, use a simple comparison
                change_percentage = 0.0 if parameter.scenario_value == baseline_value else 10.0
            
            # Determine impact direction
            if change_percentage > 0:
                impact_direction = ImpactDirection.POSITIVE
            elif change_percentage < 0:
                impact_direction = ImpactDirection.NEGATIVE
            else:
                impact_direction = ImpactDirection.NEUTRAL
            
            # Create metric
            metric = ImpactMetric(
                name=parameter.name,
                value=parameter.scenario_value,
                unit=parameter.unit or "units",
                baseline_value=baseline_value,
                change_percentage=change_percentage,
                impact_direction=impact_direction,
                confidence_level=parameter.confidence_level,
                description=parameter.description
            )
            
            return metric
            
        except Exception as e:
            logger.warning(f"Error calculating impact for parameter {parameter.name}: {e}")
            return None
    
    def _calculate_overall_score(self, categories: Dict[ImpactType, ImpactCategory],
                               weights: Dict[ImpactType, float]) -> float:
        """Calculate overall impact score."""
        total_score = 0.0
        total_weight = 0.0
        
        for impact_type, category in categories.items():
            weight = weights.get(impact_type, 0.1)
            total_score += category.total_impact_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_risk_level(self, impact_score: float) -> str:
        """Determine risk level based on impact score."""
        if impact_score < 5.0:
            return "low"
        elif impact_score < 15.0:
            return "medium"
        elif impact_score < 30.0:
            return "high"
        else:
            return "critical"
    
    def _determine_category_risk_level(self, category: ImpactCategory) -> str:
        """Determine risk level for a category."""
        if not category.metrics:
            return "low"
        
        # Calculate average change percentage
        avg_change = np.mean([abs(metric.change_percentage) for metric in category.metrics])
        
        if avg_change < 5.0:
            return "low"
        elif avg_change < 15.0:
            return "medium"
        elif avg_change < 30.0:
            return "high"
        else:
            return "critical"
    
    def _determine_category_priority(self, category: ImpactCategory) -> str:
        """Determine priority for a category."""
        if category.risk_level == "critical":
            return "high"
        elif category.risk_level == "high":
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self, analysis: ImpactAnalysis) -> List[str]:
        """Generate recommendations based on impact analysis."""
        recommendations = []
        
        # High-risk recommendations
        if analysis.overall_risk_level in ["high", "critical"]:
            recommendations.append(
                "Consider implementing risk mitigation strategies immediately"
            )
            recommendations.append(
                "Monitor key metrics closely and establish early warning systems"
            )
        
        # Category-specific recommendations
        for impact_type, category in analysis.impact_categories.items():
            if category.risk_level == "high":
                recommendations.append(
                    f"Focus on {impact_type.value} impact mitigation strategies"
                )
            
            if category.priority == "high":
                recommendations.append(
                    f"Prioritize {impact_type.value} category in planning"
                )
        
        # Positive impact recommendations
        positive_impacts = []
        for category in analysis.impact_categories.values():
            for metric in category.metrics:
                if metric.impact_direction == ImpactDirection.POSITIVE:
                    positive_impacts.append(metric.name)
        
        if positive_impacts:
            recommendations.append(
                f"Leverage positive impacts in: {', '.join(positive_impacts[:3])}"
            )
        
        return recommendations
    
    def _identify_assumptions(self, scenario: 'Scenario') -> List[str]:
        """Identify assumptions made in the analysis."""
        assumptions = [
            "Parameter changes are independent of each other",
            "Linear relationship between parameter changes and impacts",
            "Historical patterns continue to apply"
        ]
        
        # Add scenario-specific assumptions
        if scenario.scenario_type.value in ["optimistic", "pessimistic"]:
            assumptions.append(
                f"All parameters move in {scenario.scenario_type.value} direction"
            )
        
        return assumptions
    
    def _identify_limitations(self, scenario: 'Scenario') -> List[str]:
        """Identify limitations of the analysis."""
        limitations = [
            "Analysis based on simplified models",
            "Does not account for complex interdependencies",
            "Limited historical data for some parameters"
        ]
        
        # Add scenario-specific limitations
        if len(scenario.parameters) < 3:
            limitations.append("Limited parameter coverage may underestimate impacts")
        
        return limitations
    
    def compare_scenarios(self, scenarios: List['Scenario'],
                         baseline_data: Optional[Dict[str, Any]] = None) -> Dict[str, ImpactAnalysis]:
        """
        Compare impacts across multiple scenarios.
        
        Args:
            scenarios: List of scenarios to compare
            baseline_data: Optional baseline data for comparison
            
        Returns:
            Dictionary mapping scenario IDs to impact analyses
        """
        logger.info(f"Starting scenario comparison for {len(scenarios)} scenarios")
        
        results = {}
        for scenario in scenarios:
            analysis = self.analyze_impact(scenario, baseline_data)
            results[scenario.id] = analysis
        
        return results
    
    def sensitivity_analysis(self, scenario: 'Scenario',
                           parameter_name: str,
                           range_values: List[float],
                           baseline_data: Optional[Dict[str, Any]] = None) -> Dict[float, ImpactAnalysis]:
        """
        Perform sensitivity analysis for a parameter.
        
        Args:
            scenario: Base scenario
            parameter_name: Parameter to analyze
            range_values: Range of values to test
            baseline_data: Optional baseline data
            
        Returns:
            Dictionary mapping parameter values to impact analyses
        """
        logger.info(f"Starting sensitivity analysis for parameter: {parameter_name}")
        
        results = {}
        original_value = None
        
        # Store original parameter value
        if parameter_name in scenario.parameters:
            original_value = scenario.parameters[parameter_name].scenario_value
        
        # Test each value in the range
        for value in range_values:
            # Create a copy of the scenario with modified parameter
            test_scenario = self._create_scenario_copy(scenario)
            if parameter_name in test_scenario.parameters:
                test_scenario.parameters[parameter_name].scenario_value = value
            
            # Analyze impact
            analysis = self.analyze_impact(test_scenario, baseline_data)
            results[value] = analysis
        
        # Restore original value
        if original_value is not None and parameter_name in scenario.parameters:
            scenario.parameters[parameter_name].scenario_value = original_value
        
        return results
    
    def _create_scenario_copy(self, scenario: 'Scenario') -> 'Scenario':
        """Create a copy of a scenario for testing."""
        # This is a simplified copy - in a real implementation,
        # you might want to use copy.deepcopy or implement proper cloning
        from .scenario_builder import Scenario, ScenarioParameter, ParameterType
        
        new_scenario = Scenario(
            name=f"{scenario.name} (Copy)",
            description=scenario.description,
            scenario_type=scenario.scenario_type,
            tags=scenario.tags.copy()
        )
        
        # Copy parameters
        for name, param in scenario.parameters.items():
            new_scenario.parameters[name] = ScenarioParameter(
                name=param.name,
                parameter_type=param.parameter_type,
                current_value=param.current_value,
                scenario_value=param.scenario_value,
                min_value=param.min_value,
                max_value=param.max_value,
                description=param.description,
                unit=param.unit,
                confidence_level=param.confidence_level
            )
        
        return new_scenario
