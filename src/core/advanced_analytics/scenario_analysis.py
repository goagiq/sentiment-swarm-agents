"""
Scenario Analysis Engine

Advanced scenario analysis capabilities for what-if modeling and impact assessment
in multivariate forecasting and decision support systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import warnings

# Conditional imports
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Scenario analysis features limited.")

# Local imports
from ..error_handling_service import ErrorHandlingService
from ..caching_service import CachingService
from ..performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """Data class for scenario definition"""
    name: str
    description: str
    parameters: Dict[str, Any]
    probability: float = 1.0
    impact_level: str = "medium"
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ScenarioResult:
    """Result container for scenario analysis"""
    scenario: Scenario
    baseline_forecast: pd.DataFrame
    scenario_forecast: pd.DataFrame
    impact_metrics: Dict[str, float]
    sensitivity_analysis: Dict[str, float]
    risk_assessment: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ScenarioAnalysisEngine:
    """
    Advanced scenario analysis engine for what-if modeling and impact assessment.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the scenario analysis engine"""
        self.config = config or {}
        self.error_handler = ErrorHandlingService()
        self.caching_service = CachingService()
        self.performance_monitor = PerformanceMonitor()
        
        # Analysis storage
        self.scenarios = {}
        self.scenario_results = []
        self.baseline_model = None
        
        # Configuration
        self.default_scenarios = self.config.get('default_scenarios', [
            'baseline', 'optimistic', 'pessimistic'
        ])
        self.impact_thresholds = self.config.get('impact_thresholds', {
            'low': 0.05,
            'medium': 0.15,
            'high': 0.30
        })
        
        logger.info("ScenarioAnalysisEngine initialized")
    
    def create_scenario(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        probability: float = 1.0,
        impact_level: str = "medium"
    ) -> Scenario:
        """
        Create a new scenario for analysis
        
        Args:
            name: Scenario name
            description: Scenario description
            parameters: Scenario parameters
            probability: Scenario probability
            impact_level: Expected impact level
            
        Returns:
            Scenario object
        """
        try:
            scenario = Scenario(
                name=name,
                description=description,
                parameters=parameters,
                probability=probability,
                impact_level=impact_level
            )
            
            self.scenarios[name] = scenario
            logger.info(f"Created scenario: {name}")
            return scenario
            
        except Exception as e:
            self.error_handler.handle_error(f"Error creating scenario: {str(e)}", e)
            raise
    
    def create_default_scenarios(self, data: pd.DataFrame) -> Dict[str, Scenario]:
        """
        Create default scenarios (baseline, optimistic, pessimistic)
        
        Args:
            data: Historical data for scenario parameter estimation
            
        Returns:
            Dictionary of default scenarios
        """
        try:
            scenarios = {}
            
            # Baseline scenario
            baseline_params = self._estimate_baseline_parameters(data)
            scenarios['baseline'] = self.create_scenario(
                name="baseline",
                description="Baseline scenario with current trends",
                parameters=baseline_params,
                probability=0.6,
                impact_level="low"
            )
            
            # Optimistic scenario
            optimistic_params = self._estimate_optimistic_parameters(data)
            scenarios['optimistic'] = self.create_scenario(
                name="optimistic",
                description="Optimistic scenario with positive trends",
                parameters=optimistic_params,
                probability=0.2,
                impact_level="medium"
            )
            
            # Pessimistic scenario
            pessimistic_params = self._estimate_pessimistic_parameters(data)
            scenarios['pessimistic'] = self.create_scenario(
                name="pessimistic",
                description="Pessimistic scenario with negative trends",
                parameters=pessimistic_params,
                probability=0.2,
                impact_level="high"
            )
            
            logger.info(f"Created {len(scenarios)} default scenarios")
            return scenarios
            
        except Exception as e:
            self.error_handler.handle_error(f"Error creating default scenarios: {str(e)}", e)
            raise
    
    def analyze_scenario(
        self,
        scenario: Scenario,
        baseline_data: pd.DataFrame,
        forecast_model: Any,
        target_columns: List[str],
        forecast_horizon: int = 12
    ) -> ScenarioResult:
        """
        Analyze a specific scenario
        
        Args:
            scenario: Scenario to analyze
            baseline_data: Historical baseline data
            forecast_model: Forecasting model to use
            target_columns: Target columns for forecasting
            forecast_horizon: Forecast horizon
            
        Returns:
            ScenarioResult object
        """
        try:
            with self.performance_monitor.track_operation("scenario_analysis"):
                # Generate baseline forecast
                baseline_forecast = self._generate_baseline_forecast(
                    baseline_data, forecast_model, target_columns, forecast_horizon
                )
                
                # Apply scenario parameters to data
                scenario_data = self._apply_scenario_parameters(
                    baseline_data, scenario.parameters
                )
                
                # Generate scenario forecast
                scenario_forecast = self._generate_scenario_forecast(
                    scenario_data, forecast_model, target_columns, forecast_horizon
                )
                
                # Calculate impact metrics
                impact_metrics = self._calculate_impact_metrics(
                    baseline_forecast, scenario_forecast
                )
                
                # Perform sensitivity analysis
                sensitivity_analysis = self._perform_sensitivity_analysis(
                    baseline_data, scenario, forecast_model, target_columns
                )
                
                # Assess risks
                risk_assessment = self._assess_risks(
                    scenario, impact_metrics, sensitivity_analysis
                )
                
                # Create result
                result = ScenarioResult(
                    scenario=scenario,
                    baseline_forecast=baseline_forecast,
                    scenario_forecast=scenario_forecast,
                    impact_metrics=impact_metrics,
                    sensitivity_analysis=sensitivity_analysis,
                    risk_assessment=risk_assessment
                )
                
                # Store result
                self.scenario_results.append(result)
                
                logger.info(f"Completed scenario analysis for: {scenario.name}")
                return result
                
        except Exception as e:
            self.error_handler.handle_error(f"Error analyzing scenario: {str(e)}", e)
            raise
    
    def analyze_multiple_scenarios(
        self,
        scenarios: List[Scenario],
        baseline_data: pd.DataFrame,
        forecast_model: Any,
        target_columns: List[str],
        forecast_horizon: int = 12
    ) -> List[ScenarioResult]:
        """
        Analyze multiple scenarios
        
        Args:
            scenarios: List of scenarios to analyze
            baseline_data: Historical baseline data
            forecast_model: Forecasting model to use
            target_columns: Target columns for forecasting
            forecast_horizon: Forecast horizon
            
        Returns:
            List of ScenarioResult objects
        """
        try:
            results = []
            
            for scenario in scenarios:
                result = self.analyze_scenario(
                    scenario, baseline_data, forecast_model, target_columns, forecast_horizon
                )
                results.append(result)
            
            logger.info(f"Completed analysis for {len(scenarios)} scenarios")
            return results
            
        except Exception as e:
            self.error_handler.handle_error(f"Error analyzing multiple scenarios: {str(e)}", e)
            raise
    
    def _estimate_baseline_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Estimate baseline scenario parameters"""
        params = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            params[f"{col}_trend"] = 0.0  # No change from current trend
            params[f"{col}_volatility"] = data[col].std()
            params[f"{col}_growth_rate"] = 0.0
        
        return params
    
    def _estimate_optimistic_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Estimate optimistic scenario parameters"""
        params = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            # Positive trend
            params[f"{col}_trend"] = 0.1
            params[f"{col}_volatility"] = data[col].std() * 0.8  # Lower volatility
            params[f"{col}_growth_rate"] = 0.05  # 5% growth
        
        return params
    
    def _estimate_pessimistic_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Estimate pessimistic scenario parameters"""
        params = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            # Negative trend
            params[f"{col}_trend"] = -0.1
            params[f"{col}_volatility"] = data[col].std() * 1.5  # Higher volatility
            params[f"{col}_growth_rate"] = -0.05  # -5% growth
        
        return params
    
    def _generate_baseline_forecast(
        self,
        data: pd.DataFrame,
        forecast_model: Any,
        target_columns: List[str],
        forecast_horizon: int
    ) -> pd.DataFrame:
        """Generate baseline forecast"""
        try:
            # Use the forecast model to generate baseline forecast
            if hasattr(forecast_model, 'forecast_multivariate'):
                result = forecast_model.forecast_multivariate(
                    data, target_columns, forecast_horizon
                )
                return result.predictions
            else:
                # Fallback to simple forecasting
                return self._simple_forecast(data, target_columns, forecast_horizon)
                
        except Exception as e:
            logger.warning(f"Error in baseline forecast, using simple method: {str(e)}")
            return self._simple_forecast(data, target_columns, forecast_horizon)
    
    def _generate_scenario_forecast(
        self,
        data: pd.DataFrame,
        forecast_model: Any,
        target_columns: List[str],
        forecast_horizon: int
    ) -> pd.DataFrame:
        """Generate scenario forecast"""
        try:
            # Use the forecast model to generate scenario forecast
            if hasattr(forecast_model, 'forecast_multivariate'):
                result = forecast_model.forecast_multivariate(
                    data, target_columns, forecast_horizon
                )
                return result.predictions
            else:
                # Fallback to simple forecasting
                return self._simple_forecast(data, target_columns, forecast_horizon)
                
        except Exception as e:
            logger.warning(f"Error in scenario forecast, using simple method: {str(e)}")
            return self._simple_forecast(data, target_columns, forecast_horizon)
    
    def _simple_forecast(
        self,
        data: pd.DataFrame,
        target_columns: List[str],
        forecast_horizon: int
    ) -> pd.DataFrame:
        """Simple forecasting method as fallback"""
        forecasts = {}
        
        for col in target_columns:
            if col in data.columns:
                # Simple moving average forecast
                last_values = data[col].tail(12).values
                trend = np.polyfit(range(len(last_values)), last_values, 1)[0]
                
                forecast_values = []
                last_value = last_values[-1]
                
                for i in range(forecast_horizon):
                    next_value = last_value + trend * (i + 1)
                    forecast_values.append(next_value)
                
                forecasts[col] = forecast_values
        
        return pd.DataFrame(forecasts)
    
    def _apply_scenario_parameters(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply scenario parameters to data"""
        scenario_data = data.copy()
        
        for param_name, param_value in parameters.items():
            if '_trend' in param_name:
                col_name = param_name.replace('_trend', '')
                if col_name in scenario_data.columns:
                    # Apply trend adjustment
                    scenario_data[col_name] = scenario_data[col_name] * (1 + param_value)
            
            elif '_volatility' in param_name:
                col_name = param_name.replace('_volatility', '')
                if col_name in scenario_data.columns:
                    # Apply volatility adjustment
                    noise = np.random.normal(0, param_value, len(scenario_data))
                    scenario_data[col_name] = scenario_data[col_name] + noise
            
            elif '_growth_rate' in param_name:
                col_name = param_name.replace('_growth_rate', '')
                if col_name in scenario_data.columns:
                    # Apply growth rate adjustment
                    scenario_data[col_name] = scenario_data[col_name] * (1 + param_value)
        
        return scenario_data
    
    def _calculate_impact_metrics(
        self,
        baseline_forecast: pd.DataFrame,
        scenario_forecast: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate impact metrics between baseline and scenario"""
        impact_metrics = {}
        
        for col in baseline_forecast.columns:
            if col in scenario_forecast.columns:
                baseline_values = baseline_forecast[col].values
                scenario_values = scenario_forecast[col].values
                
                # Calculate various impact metrics
                absolute_change = np.mean(scenario_values - baseline_values)
                relative_change = np.mean((scenario_values - baseline_values) / baseline_values)
                max_impact = np.max(np.abs(scenario_values - baseline_values))
                volatility_impact = np.std(scenario_values) - np.std(baseline_values)
                
                impact_metrics[f"{col}_absolute_change"] = absolute_change
                impact_metrics[f"{col}_relative_change"] = relative_change
                impact_metrics[f"{col}_max_impact"] = max_impact
                impact_metrics[f"{col}_volatility_impact"] = volatility_impact
        
        return impact_metrics
    
    def _perform_sensitivity_analysis(
        self,
        baseline_data: pd.DataFrame,
        scenario: Scenario,
        forecast_model: Any,
        target_columns: List[str]
    ) -> Dict[str, float]:
        """Perform sensitivity analysis on scenario parameters"""
        sensitivity_results = {}
        
        # Test parameter sensitivity
        for param_name, param_value in scenario.parameters.items():
            if isinstance(param_value, (int, float)):
                # Test small variations
                variations = [param_value * 0.9, param_value * 1.1]
                
                for variation in variations:
                    test_params = scenario.parameters.copy()
                    test_params[param_name] = variation
                    
                    # Create test scenario
                    test_scenario = Scenario(
                        name=f"{scenario.name}_sensitivity_{param_name}",
                        description=f"Sensitivity test for {param_name}",
                        parameters=test_params
                    )
                    
                    # Analyze test scenario
                    try:
                        result = self.analyze_scenario(
                            test_scenario, baseline_data, forecast_model, target_columns
                        )
                        
                        # Calculate sensitivity
                        baseline_impact = np.mean(list(result.impact_metrics.values()))
                        sensitivity_results[f"{param_name}_sensitivity"] = baseline_impact
                        
                    except Exception as e:
                        logger.debug(f"Sensitivity analysis failed for {param_name}: {str(e)}")
                        continue
        
        return sensitivity_results
    
    def _assess_risks(
        self,
        scenario: Scenario,
        impact_metrics: Dict[str, float],
        sensitivity_analysis: Dict[str, float]
    ) -> Dict[str, Any]:
        """Assess risks associated with the scenario"""
        risk_assessment = {
            'overall_risk_level': 'low',
            'high_impact_metrics': [],
            'sensitive_parameters': [],
            'risk_factors': []
        }
        
        # Assess impact-based risks
        for metric_name, metric_value in impact_metrics.items():
            if abs(metric_value) > self.impact_thresholds['high']:
                risk_assessment['high_impact_metrics'].append(metric_name)
                risk_assessment['risk_factors'].append(f"High impact: {metric_name}")
        
        # Assess sensitivity-based risks
        for param_name, sensitivity_value in sensitivity_analysis.items():
            if abs(sensitivity_value) > 0.1:  # High sensitivity threshold
                risk_assessment['sensitive_parameters'].append(param_name)
                risk_assessment['risk_factors'].append(f"High sensitivity: {param_name}")
        
        # Determine overall risk level
        risk_factors_count = len(risk_assessment['risk_factors'])
        if risk_factors_count > 5:
            risk_assessment['overall_risk_level'] = 'high'
        elif risk_factors_count > 2:
            risk_assessment['overall_risk_level'] = 'medium'
        else:
            risk_assessment['overall_risk_level'] = 'low'
        
        return risk_assessment
    
    def get_scenario_comparison(self, scenario_names: List[str]) -> pd.DataFrame:
        """Compare multiple scenarios"""
        comparison_data = []
        
        for result in self.scenario_results:
            if result.scenario.name in scenario_names:
                comparison_data.append({
                    'scenario': result.scenario.name,
                    'probability': result.scenario.probability,
                    'impact_level': result.scenario.impact_level,
                    'overall_risk': result.risk_assessment['overall_risk_level'],
                    'avg_impact': np.mean(list(result.impact_metrics.values())),
                    'max_impact': np.max(list(result.impact_metrics.values())),
                    'risk_factors_count': len(result.risk_assessment['risk_factors'])
                })
        
        return pd.DataFrame(comparison_data)
    
    def export_scenario_analysis(self, filepath: str) -> None:
        """Export scenario analysis results"""
        try:
            import json
            from datetime import datetime
            
            export_data = {
                'scenarios': [
                    {
                        'name': scenario.name,
                        'description': scenario.description,
                        'parameters': scenario.parameters,
                        'probability': scenario.probability,
                        'impact_level': scenario.impact_level,
                        'timestamp': scenario.timestamp.isoformat() if scenario.timestamp else None
                    }
                    for scenario in self.scenarios.values()
                ],
                'results': [
                    {
                        'scenario_name': result.scenario.name,
                        'impact_metrics': result.impact_metrics,
                        'risk_assessment': result.risk_assessment,
                        'timestamp': result.timestamp.isoformat() if result.timestamp else None
                    }
                    for result in self.scenario_results
                ],
                'config': self.config
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Scenario analysis exported to {filepath}")
            
        except Exception as e:
            self.error_handler.handle_error(f"Error exporting scenario analysis: {str(e)}", e)
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of scenario analysis results"""
        if not self.scenario_results:
            return {"message": "No scenario analysis results found"}
        
        summary = {
            'total_scenarios': len(self.scenarios),
            'total_analyses': len(self.scenario_results),
            'scenario_names': list(self.scenarios.keys()),
            'risk_distribution': {},
            'impact_distribution': {}
        }
        
        # Calculate risk distribution
        risk_levels = [result.risk_assessment['overall_risk_level'] for result in self.scenario_results]
        for risk_level in set(risk_levels):
            summary['risk_distribution'][risk_level] = risk_levels.count(risk_level)
        
        # Calculate impact distribution
        impacts = [np.mean(list(result.impact_metrics.values())) for result in self.scenario_results]
        summary['impact_distribution'] = {
            'mean': np.mean(impacts),
            'std': np.std(impacts),
            'min': np.min(impacts),
            'max': np.max(impacts)
        }
        
        return summary
