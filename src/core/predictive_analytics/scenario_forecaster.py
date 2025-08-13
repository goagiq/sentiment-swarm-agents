"""
Scenario Forecaster

This module provides what-if scenario analysis and multiple future
scenario generation capabilities.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """Scenario definition and results"""
    name: str
    description: str
    parameters: Dict[str, Any]
    predictions: np.ndarray
    confidence_intervals: Optional[tuple] = None
    probability: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ScenarioComparison:
    """Results of scenario comparison"""
    scenarios: List[Scenario]
    comparison_metrics: Dict[str, Any]
    best_scenario: Optional[str] = None
    worst_scenario: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ScenarioForecaster:
    """
    Generate and analyze multiple future scenarios
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the scenario forecaster
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'default_scenarios': ['baseline', 'optimistic', 'pessimistic'],
            'scenario_probabilities': {
                'baseline': 0.6,
                'optimistic': 0.2,
                'pessimistic': 0.2
            },
            'parameter_variation': 0.2,  # 20% variation
            'max_scenarios': 10,
            'confidence_level': 0.95
        }
        
        # Update with provided config
        self.default_config.update(self.config)
        self.config = self.default_config
        
        logger.info(f"ScenarioForecaster initialized with config: {self.config}")
    
    def generate_scenarios(
        self,
        base_data: np.ndarray,
        scenario_definitions: Optional[Dict[str, Dict[str, Any]]] = None,
        forecast_horizon: int = 12
    ) -> List[Scenario]:
        """
        Generate multiple scenarios based on different parameter assumptions
        
        Args:
            base_data: Base time series data
            scenario_definitions: Dictionary of scenario definitions
            forecast_horizon: Number of periods to forecast
            
        Returns:
            List of Scenario objects
        """
        try:
            if scenario_definitions is None:
                scenario_definitions = self._get_default_scenarios()
            
            scenarios = []
            
            for scenario_name, scenario_config in scenario_definitions.items():
                try:
                    scenario = self._generate_single_scenario(
                        base_data, scenario_name, scenario_config, forecast_horizon
                    )
                    scenarios.append(scenario)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate scenario {scenario_name}: {str(e)}")
                    continue
            
            logger.info(f"Generated {len(scenarios)} scenarios successfully")
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating scenarios: {str(e)}")
            raise
    
    def _generate_single_scenario(
        self,
        base_data: np.ndarray,
        scenario_name: str,
        scenario_config: Dict[str, Any],
        forecast_horizon: int
    ) -> Scenario:
        """Generate a single scenario"""
        try:
            # Apply scenario parameters to base data
            modified_data = self._apply_scenario_parameters(base_data, scenario_config)
            
            # Generate forecast for modified data
            from .forecasting_engine import ForecastingEngine
            forecaster = ForecastingEngine()
            
            forecast_result = forecaster.forecast(
                modified_data,
                model_type='ensemble',
                forecast_horizon=forecast_horizon
            )
            
            # Get scenario probability
            probability = scenario_config.get('probability', 
                                            self.config['scenario_probabilities'].get(scenario_name, 1.0))
            
            # Create scenario object
            scenario = Scenario(
                name=scenario_name,
                description=scenario_config.get('description', f'{scenario_name} scenario'),
                parameters=scenario_config.get('parameters', {}),
                predictions=forecast_result.predictions,
                confidence_intervals=forecast_result.confidence_intervals,
                probability=probability,
                metadata={
                    'model_accuracy': forecast_result.model_accuracy,
                    'model_type': forecast_result.model_type,
                    'forecast_horizon': forecast_horizon,
                    'base_data_points': len(base_data),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            return scenario
            
        except Exception as e:
            logger.error(f"Error generating scenario {scenario_name}: {str(e)}")
            raise
    
    def _apply_scenario_parameters(
        self,
        base_data: np.ndarray,
        scenario_config: Dict[str, Any]
    ) -> np.ndarray:
        """Apply scenario parameters to modify base data"""
        try:
            modified_data = base_data.copy()
            parameters = scenario_config.get('parameters', {})
            
            # Apply trend modification
            if 'trend_multiplier' in parameters:
                trend_mult = parameters['trend_multiplier']
                # Add trend to data
                trend = np.linspace(0, trend_mult, len(modified_data))
                modified_data = modified_data + trend
            
            # Apply volatility modification
            if 'volatility_multiplier' in parameters:
                vol_mult = parameters['volatility_multiplier']
                # Scale the variance
                mean_val = np.mean(modified_data)
                modified_data = mean_val + (modified_data - mean_val) * vol_mult
            
            # Apply level shift
            if 'level_shift' in parameters:
                level_shift = parameters['level_shift']
                modified_data = modified_data + level_shift
            
            # Apply seasonal modification
            if 'seasonal_strength' in parameters:
                seasonal_strength = parameters['seasonal_strength']
                # Add seasonal component
                seasonal = np.sin(2 * np.pi * np.arange(len(modified_data)) / 12) * seasonal_strength
                modified_data = modified_data + seasonal
            
            return modified_data
            
        except Exception as e:
            logger.error(f"Error applying scenario parameters: {str(e)}")
            return base_data
    
    def _get_default_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Get default scenario definitions"""
        return {
            'baseline': {
                'description': 'Baseline scenario with current trends',
                'probability': 0.6,
                'parameters': {
                    'trend_multiplier': 0.0,
                    'volatility_multiplier': 1.0,
                    'level_shift': 0.0,
                    'seasonal_strength': 0.0
                }
            },
            'optimistic': {
                'description': 'Optimistic scenario with positive trends',
                'probability': 0.2,
                'parameters': {
                    'trend_multiplier': 0.1,
                    'volatility_multiplier': 0.8,
                    'level_shift': 0.05,
                    'seasonal_strength': 0.0
                }
            },
            'pessimistic': {
                'description': 'Pessimistic scenario with negative trends',
                'probability': 0.2,
                'parameters': {
                    'trend_multiplier': -0.1,
                    'volatility_multiplier': 1.2,
                    'level_shift': -0.05,
                    'seasonal_strength': 0.0
                }
            }
        }
    
    def compare_scenarios(self, scenarios: List[Scenario]) -> ScenarioComparison:
        """
        Compare multiple scenarios and identify best/worst cases
        
        Args:
            scenarios: List of scenarios to compare
            
        Returns:
            ScenarioComparison with metrics and rankings
        """
        try:
            if not scenarios:
                raise ValueError("No scenarios provided for comparison")
            
            comparison_metrics = {}
            
            # Calculate comparison metrics
            comparison_metrics['scenario_count'] = len(scenarios)
            comparison_metrics['forecast_horizon'] = scenarios[0].predictions.shape[0]
            
            # Calculate expected values (probability-weighted)
            expected_values = []
            for scenario in scenarios:
                expected_value = np.mean(scenario.predictions) * scenario.probability
                expected_values.append(expected_value)
            
            comparison_metrics['expected_values'] = expected_values
            
            # Find best and worst scenarios
            scenario_names = [s.name for s in scenarios]
            best_idx = np.argmax(expected_values)
            worst_idx = np.argmin(expected_values)
            
            best_scenario = scenario_names[best_idx]
            worst_scenario = scenario_names[worst_idx]
            
            # Calculate additional metrics
            comparison_metrics['value_range'] = {
                'min': float(np.min([np.min(s.predictions) for s in scenarios])),
                'max': float(np.max([np.max(s.predictions) for s in scenarios])),
                'range': float(np.max([np.max(s.predictions) for s in scenarios]) - 
                             np.min([np.min(s.predictions) for s in scenarios]))
            }
            
            comparison_metrics['volatility'] = {
                scenario.name: float(np.std(scenario.predictions)) 
                for scenario in scenarios
            }
            
            comparison_metrics['probability_weighted_forecast'] = self._calculate_probability_weighted_forecast(scenarios)
            
            # Create comparison result
            comparison = ScenarioComparison(
                scenarios=scenarios,
                comparison_metrics=comparison_metrics,
                best_scenario=best_scenario,
                worst_scenario=worst_scenario,
                metadata={
                    'comparison_timestamp': datetime.now().isoformat(),
                    'total_probability': sum(s.probability for s in scenarios)
                }
            )
            
            logger.info(f"Scenario comparison completed. Best: {best_scenario}, Worst: {worst_scenario}")
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing scenarios: {str(e)}")
            raise
    
    def _calculate_probability_weighted_forecast(self, scenarios: List[Scenario]) -> np.ndarray:
        """Calculate probability-weighted average forecast across scenarios"""
        try:
            if not scenarios:
                return np.array([])
            
            forecast_length = scenarios[0].predictions.shape[0]
            weighted_forecast = np.zeros(forecast_length)
            total_probability = sum(s.probability for s in scenarios)
            
            for scenario in scenarios:
                weight = scenario.probability / total_probability
                weighted_forecast += weight * scenario.predictions
            
            return weighted_forecast
            
        except Exception as e:
            logger.error(f"Error calculating probability-weighted forecast: {str(e)}")
            return np.array([])
    
    def create_custom_scenario(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        probability: float = 1.0
    ) -> Dict[str, Any]:
        """
        Create a custom scenario definition
        
        Args:
            name: Scenario name
            description: Scenario description
            parameters: Scenario parameters
            probability: Scenario probability
            
        Returns:
            Scenario definition dictionary
        """
        return {
            'description': description,
            'probability': probability,
            'parameters': parameters
        }
    
    def validate_scenario_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate scenario parameters"""
        try:
            valid_parameters = {
                'trend_multiplier', 'volatility_multiplier', 
                'level_shift', 'seasonal_strength'
            }
            
            for param in parameters:
                if param not in valid_parameters:
                    logger.warning(f"Unknown parameter: {param}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating scenario parameters: {str(e)}")
            return False
    
    def get_scenario_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined scenario templates"""
        return {
            'growth_scenario': {
                'description': 'High growth scenario',
                'parameters': {
                    'trend_multiplier': 0.15,
                    'volatility_multiplier': 0.9,
                    'level_shift': 0.1
                }
            },
            'recession_scenario': {
                'description': 'Economic recession scenario',
                'parameters': {
                    'trend_multiplier': -0.2,
                    'volatility_multiplier': 1.5,
                    'level_shift': -0.15
                }
            },
            'stable_scenario': {
                'description': 'Stable growth scenario',
                'parameters': {
                    'trend_multiplier': 0.05,
                    'volatility_multiplier': 0.7,
                    'level_shift': 0.02
                }
            },
            'volatile_scenario': {
                'description': 'High volatility scenario',
                'parameters': {
                    'trend_multiplier': 0.0,
                    'volatility_multiplier': 2.0,
                    'level_shift': 0.0
                }
            }
        }
