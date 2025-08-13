"""
Scenario Comparator Component

Provides multi-scenario analysis and comparison capabilities.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ComparisonMetric(Enum):
    """Metrics used for scenario comparison."""
    IMPACT_SCORE = "impact_score"
    RISK_SCORE = "risk_score"
    CONFIDENCE_LEVEL = "confidence_level"
    PARAMETER_CHANGE = "parameter_change"
    OVERALL_SCORE = "overall_score"


class ComparisonType(Enum):
    """Types of scenario comparisons."""
    PAIRWISE = "pairwise"
    RANKING = "ranking"
    CLUSTERING = "clustering"
    SENSITIVITY = "sensitivity"


@dataclass
class ComparisonResult:
    """Result of scenario comparison."""
    comparison_id: str
    comparison_type: ComparisonType
    scenarios_compared: List[str]
    comparison_timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[ComparisonMetric, Dict[str, float]] = field(default_factory=dict)
    rankings: Dict[str, int] = field(default_factory=dict)
    similarities: Dict[Tuple[str, str], float] = field(default_factory=dict)
    differences: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert comparison to dictionary for serialization."""
        return {
            'comparison_id': self.comparison_id,
            'comparison_type': self.comparison_type.value,
            'scenarios_compared': self.scenarios_compared,
            'comparison_timestamp': self.comparison_timestamp.isoformat(),
            'metrics': {
                metric.value: values for metric, values in self.metrics.items()
            },
            'rankings': self.rankings,
            'similarities': {
                f"{s1}_{s2}": similarity 
                for (s1, s2), similarity in self.similarities.items()
            },
            'differences': {
                f"{s1}_{s2}": differences 
                for (s1, s2), differences in self.differences.items()
            },
            'recommendations': self.recommendations
        }


@dataclass
class ScenarioRanking:
    """Ranking of scenarios based on specific criteria."""
    scenario_id: str
    rank: int
    score: float
    criteria: str
    percentile: float = 0.0


class ScenarioComparator:
    """
    Multi-scenario analysis and comparison system.
    
    Provides capabilities for:
    - Pairwise scenario comparison
    - Scenario ranking and prioritization
    - Similarity and difference analysis
    - Sensitivity analysis across scenarios
    """
    
    def __init__(self):
        self.comparison_history: List[ComparisonResult] = []
        self.ranking_methods: Dict[str, callable] = {}
        self.similarity_metrics: Dict[str, callable] = {}
        self._initialize_methods()
    
    def _initialize_methods(self):
        """Initialize comparison methods."""
        # Ranking methods
        self.ranking_methods = {
            'impact_based': self._rank_by_impact,
            'risk_based': self._rank_by_risk,
            'composite': self._rank_by_composite_score
        }
        
        # Similarity metrics
        self.similarity_metrics = {
            'parameter_similarity': self._calculate_parameter_similarity,
            'overall_similarity': self._calculate_overall_similarity
        }
    
    def compare_scenarios(self, scenarios: List['Scenario'],
                         comparison_type: ComparisonType = ComparisonType.PAIRWISE,
                         metrics: Optional[List[ComparisonMetric]] = None) -> ComparisonResult:
        """
        Compare multiple scenarios.
        
        Args:
            scenarios: List of scenarios to compare
            comparison_type: Type of comparison to perform
            metrics: Specific metrics to compare
            
        Returns:
            Comparison result
        """
        logger.info(f"Starting {comparison_type.value} comparison of {len(scenarios)} scenarios")
        
        # Default metrics if none specified
        if metrics is None:
            metrics = [ComparisonMetric.IMPACT_SCORE, ComparisonMetric.RISK_SCORE]
        
        # Create comparison result
        comparison_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = ComparisonResult(
            comparison_id=comparison_id,
            comparison_type=comparison_type,
            scenarios_compared=[s.id for s in scenarios]
        )
        
        # Perform comparison based on type
        if comparison_type == ComparisonType.PAIRWISE:
            self._perform_pairwise_comparison(scenarios, result, metrics)
        elif comparison_type == ComparisonType.RANKING:
            self._perform_ranking_comparison(scenarios, result, metrics)
        elif comparison_type == ComparisonType.CLUSTERING:
            self._perform_clustering_comparison(scenarios, result, metrics)
        
        # Generate recommendations
        result.recommendations = self._generate_comparison_recommendations(result)
        
        # Store in history
        self.comparison_history.append(result)
        
        logger.info(f"Completed {comparison_type.value} comparison")
        return result
    
    def _perform_pairwise_comparison(self, scenarios: List['Scenario'],
                                   result: ComparisonResult,
                                   metrics: List[ComparisonMetric]):
        """Perform pairwise comparison of scenarios."""
        # Calculate metrics for each scenario
        for metric in metrics:
            result.metrics[metric] = {}
            for scenario in scenarios:
                score = self._calculate_metric_score(scenario, metric)
                result.metrics[metric][scenario.id] = score
        
        # Calculate similarities and differences
        for i, scenario1 in enumerate(scenarios):
            for j, scenario2 in enumerate(scenarios):
                if i < j:  # Avoid duplicate comparisons
                    similarity = self._calculate_overall_similarity(scenario1, scenario2)
                    differences = self._identify_differences(scenario1, scenario2)
                    
                    result.similarities[(scenario1.id, scenario2.id)] = similarity
                    result.differences[(scenario1.id, scenario2.id)] = differences
    
    def _perform_ranking_comparison(self, scenarios: List['Scenario'],
                                  result: ComparisonResult,
                                  metrics: List[ComparisonMetric]):
        """Perform ranking comparison of scenarios."""
        # Calculate composite scores
        composite_scores = {}
        for scenario in scenarios:
            scores = []
            for metric in metrics:
                score = self._calculate_metric_score(scenario, metric)
                scores.append(score)
            
            # Average the scores (could be weighted)
            composite_scores[scenario.id] = np.mean(scores)
        
        # Create rankings
        sorted_scenarios = sorted(composite_scores.items(), 
                                key=lambda x: x[1], reverse=True)
        
        for rank, (scenario_id, score) in enumerate(sorted_scenarios, 1):
            result.rankings[scenario_id] = rank
            result.metrics[ComparisonMetric.OVERALL_SCORE] = {scenario_id: score}
    
    def _perform_clustering_comparison(self, scenarios: List['Scenario'],
                                     result: ComparisonResult,
                                     metrics: List[ComparisonMetric]):
        """Perform clustering comparison of scenarios."""
        # Calculate similarity matrix
        similarity_matrix = {}
        for i, scenario1 in enumerate(scenarios):
            for j, scenario2 in enumerate(scenarios):
                if i <= j:
                    similarity = self._calculate_overall_similarity(scenario1, scenario2)
                    similarity_matrix[(scenario1.id, scenario2.id)] = similarity
                    if i != j:
                        similarity_matrix[(scenario2.id, scenario1.id)] = similarity
        
        # Simple clustering based on similarity threshold
        clusters = self._cluster_scenarios(scenarios, similarity_matrix)
        
        # Add clustering information to result
        result.metadata = {'clusters': clusters}
    
    def _calculate_metric_score(self, scenario: 'Scenario', 
                              metric: ComparisonMetric) -> float:
        """Calculate score for a specific metric."""
        if metric == ComparisonMetric.IMPACT_SCORE:
            return self._calculate_impact_score(scenario)
        elif metric == ComparisonMetric.RISK_SCORE:
            return self._calculate_risk_score(scenario)
        elif metric == ComparisonMetric.CONFIDENCE_LEVEL:
            return self._calculate_confidence_level(scenario)
        elif metric == ComparisonMetric.PARAMETER_CHANGE:
            return self._calculate_parameter_change_score(scenario)
        else:
            return 0.0
    
    def _calculate_impact_score(self, scenario: 'Scenario') -> float:
        """Calculate impact score for a scenario."""
        if not scenario.parameters:
            return 0.0
        
        # Calculate average parameter change
        changes = []
        for parameter in scenario.parameters.values():
            if isinstance(parameter.current_value, (int, float)) and isinstance(parameter.scenario_value, (int, float)):
                if parameter.current_value != 0:
                    change = abs(parameter.scenario_value - parameter.current_value) / abs(parameter.current_value)
                    changes.append(change)
        
        return np.mean(changes) if changes else 0.0
    
    def _calculate_risk_score(self, scenario: 'Scenario') -> float:
        """Calculate risk score for a scenario."""
        if not scenario.parameters:
            return 0.0
        
        # Simplified risk calculation based on parameter changes and confidence
        risk_scores = []
        for parameter in scenario.parameters.values():
            # Base risk on confidence level (lower confidence = higher risk)
            base_risk = 1 - parameter.confidence_level
            
            # Additional risk based on parameter type
            if parameter.parameter_type.value == 'numerical':
                if parameter.min_value is not None and parameter.max_value is not None:
                    if not (parameter.min_value <= parameter.scenario_value <= parameter.max_value):
                        base_risk += 0.3
            
            risk_scores.append(base_risk)
        
        return np.mean(risk_scores) if risk_scores else 0.0
    
    def _calculate_confidence_level(self, scenario: 'Scenario') -> float:
        """Calculate overall confidence level for a scenario."""
        if not scenario.parameters:
            return 1.0
        
        confidence_levels = [param.confidence_level for param in scenario.parameters.values()]
        return np.mean(confidence_levels)
    
    def _calculate_parameter_change_score(self, scenario: 'Scenario') -> float:
        """Calculate parameter change score."""
        if not scenario.parameters:
            return 0.0
        
        # Count parameters that have changed
        changed_params = 0
        for parameter in scenario.parameters.values():
            if parameter.current_value != parameter.scenario_value:
                changed_params += 1
        
        return changed_params / len(scenario.parameters)
    
    def _calculate_overall_similarity(self, scenario1: 'Scenario', 
                                    scenario2: 'Scenario') -> float:
        """Calculate overall similarity between two scenarios."""
        similarities = []
        
        # Parameter similarity
        param_similarity = self._calculate_parameter_similarity(scenario1, scenario2)
        similarities.append(param_similarity)
        
        # Scenario type similarity
        type_similarity = 1.0 if scenario1.scenario_type == scenario2.scenario_type else 0.0
        similarities.append(type_similarity)
        
        # Tag similarity
        tag_similarity = self._calculate_tag_similarity(scenario1, scenario2)
        similarities.append(tag_similarity)
        
        return np.mean(similarities)
    
    def _calculate_parameter_similarity(self, scenario1: 'Scenario', 
                                      scenario2: 'Scenario') -> float:
        """Calculate parameter similarity between scenarios."""
        if not scenario1.parameters or not scenario2.parameters:
            return 0.0
        
        # Find common parameters
        common_params = set(scenario1.parameters.keys()) & set(scenario2.parameters.keys())
        if not common_params:
            return 0.0
        
        # Calculate similarity for common parameters
        similarities = []
        for param_name in common_params:
            param1 = scenario1.parameters[param_name]
            param2 = scenario2.parameters[param_name]
            
            # Value similarity
            if isinstance(param1.scenario_value, (int, float)) and isinstance(param2.scenario_value, (int, float)):
                if param1.scenario_value == param2.scenario_value:
                    similarities.append(1.0)
                else:
                    # Calculate relative difference
                    max_val = max(abs(param1.scenario_value), abs(param2.scenario_value))
                    if max_val != 0:
                        diff = abs(param1.scenario_value - param2.scenario_value) / max_val
                        similarities.append(1.0 - min(diff, 1.0))
                    else:
                        similarities.append(0.0)
            else:
                # For non-numerical values, exact match
                similarities.append(1.0 if param1.scenario_value == param2.scenario_value else 0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_tag_similarity(self, scenario1: 'Scenario', 
                                scenario2: 'Scenario') -> float:
        """Calculate tag similarity between scenarios."""
        if not scenario1.tags or not scenario2.tags:
            return 0.0
        
        common_tags = set(scenario1.tags) & set(scenario2.tags)
        all_tags = set(scenario1.tags) | set(scenario2.tags)
        
        return len(common_tags) / len(all_tags) if all_tags else 0.0
    
    def _identify_differences(self, scenario1: 'Scenario', 
                            scenario2: 'Scenario') -> List[str]:
        """Identify key differences between scenarios."""
        differences = []
        
        # Parameter differences
        for param_name, param1 in scenario1.parameters.items():
            if param_name in scenario2.parameters:
                param2 = scenario2.parameters[param_name]
                if param1.scenario_value != param2.scenario_value:
                    differences.append(f"Parameter '{param_name}' differs: {param1.scenario_value} vs {param2.scenario_value}")
            else:
                differences.append(f"Parameter '{param_name}' only in scenario 1")
        
        # Parameters only in scenario 2
        for param_name in scenario2.parameters:
            if param_name not in scenario1.parameters:
                differences.append(f"Parameter '{param_name}' only in scenario 2")
        
        # Scenario type differences
        if scenario1.scenario_type != scenario2.scenario_type:
            differences.append(f"Scenario type differs: {scenario1.scenario_type.value} vs {scenario2.scenario_type.value}")
        
        return differences
    
    def _cluster_scenarios(self, scenarios: List['Scenario'],
                          similarity_matrix: Dict[Tuple[str, str], float]) -> List[List[str]]:
        """Cluster scenarios based on similarity."""
        # Simple clustering using similarity threshold
        threshold = 0.7
        clusters = []
        assigned = set()
        
        for scenario in scenarios:
            if scenario.id in assigned:
                continue
            
            # Start new cluster
            cluster = [scenario.id]
            assigned.add(scenario.id)
            
            # Find similar scenarios
            for other_scenario in scenarios:
                if other_scenario.id in assigned:
                    continue
                
                similarity = similarity_matrix.get((scenario.id, other_scenario.id), 0.0)
                if similarity >= threshold:
                    cluster.append(other_scenario.id)
                    assigned.add(other_scenario.id)
            
            clusters.append(cluster)
        
        return clusters
    
    def _generate_comparison_recommendations(self, result: ComparisonResult) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []
        
        # Recommendations based on rankings
        if result.rankings:
            top_scenario = min(result.rankings.items(), key=lambda x: x[1])[0]
            recommendations.append(f"Consider prioritizing scenario {top_scenario} based on rankings")
        
        # Recommendations based on similarities
        if result.similarities:
            high_similarity_pairs = [
                (s1, s2) for (s1, s2), sim in result.similarities.items() 
                if sim > 0.8
            ]
            if high_similarity_pairs:
                recommendations.append("Consider consolidating highly similar scenarios")
        
        # Recommendations based on differences
        if result.differences:
            high_difference_pairs = [
                (s1, s2) for (s1, s2), diffs in result.differences.items() 
                if len(diffs) > 5
            ]
            if high_difference_pairs:
                recommendations.append("Scenarios with many differences may represent distinct options")
        
        return recommendations
    
    def rank_scenarios(self, scenarios: List['Scenario'],
                      method: str = 'composite',
                      criteria: Optional[Dict[str, float]] = None) -> List[ScenarioRanking]:
        """
        Rank scenarios using specified method.
        
        Args:
            scenarios: Scenarios to rank
            method: Ranking method to use
            criteria: Optional criteria weights
            
        Returns:
            List of scenario rankings
        """
        if method not in self.ranking_methods:
            raise ValueError(f"Unknown ranking method: {method}")
        
        rankings = self.ranking_methods[method](scenarios, criteria)
        return rankings
    
    def _rank_by_impact(self, scenarios: List['Scenario'],
                       criteria: Optional[Dict[str, float]] = None) -> List[ScenarioRanking]:
        """Rank scenarios by impact score."""
        scores = [(s.id, self._calculate_impact_score(s)) for s in scenarios]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        rankings = []
        for rank, (scenario_id, score) in enumerate(scores, 1):
            percentile = (len(scores) - rank + 1) / len(scores)
            rankings.append(ScenarioRanking(
                scenario_id=scenario_id,
                rank=rank,
                score=score,
                criteria="impact_score",
                percentile=percentile
            ))
        
        return rankings
    
    def _rank_by_risk(self, scenarios: List['Scenario'],
                     criteria: Optional[Dict[str, float]] = None) -> List[ScenarioRanking]:
        """Rank scenarios by risk score (lower is better)."""
        scores = [(s.id, self._calculate_risk_score(s)) for s in scenarios]
        scores.sort(key=lambda x: x[1])  # Sort by risk (ascending)
        
        rankings = []
        for rank, (scenario_id, score) in enumerate(scores, 1):
            percentile = (len(scores) - rank + 1) / len(scores)
            rankings.append(ScenarioRanking(
                scenario_id=scenario_id,
                rank=rank,
                score=score,
                criteria="risk_score",
                percentile=percentile
            ))
        
        return rankings
    
    def _rank_by_composite_score(self, scenarios: List['Scenario'],
                                criteria: Optional[Dict[str, float]] = None) -> List[ScenarioRanking]:
        """Rank scenarios by composite score."""
        # Default criteria weights
        if criteria is None:
            criteria = {
                'impact': 0.4,
                'risk': 0.3,
                'confidence': 0.3
            }
        
        scores = []
        for scenario in scenarios:
            impact_score = self._calculate_impact_score(scenario)
            risk_score = self._calculate_risk_score(scenario)
            confidence_score = self._calculate_confidence_level(scenario)
            
            # Composite score (lower risk is better, so invert)
            composite_score = (
                criteria.get('impact', 0.4) * impact_score +
                criteria.get('risk', 0.3) * (1 - risk_score) +
                criteria.get('confidence', 0.3) * confidence_score
            )
            
            scores.append((scenario.id, composite_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        rankings = []
        for rank, (scenario_id, score) in enumerate(scores, 1):
            percentile = (len(scores) - rank + 1) / len(scores)
            rankings.append(ScenarioRanking(
                scenario_id=scenario_id,
                rank=rank,
                score=score,
                criteria="composite_score",
                percentile=percentile
            ))
        
        return rankings
