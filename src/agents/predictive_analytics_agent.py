"""
Predictive Analytics Agent

Orchestrates all predictive analytics components including forecasting,
confidence calculation, validation, and scenario analysis.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

from .base_agent import StrandsBaseAgent
from ..core.models import DataType, AnalysisRequest, AnalysisResult, SentimentResult
from ..core.predictive_analytics.forecasting_engine import ForecastingEngine
from ..core.predictive_analytics.confidence_calculator import ConfidenceCalculator
from ..core.predictive_analytics.scenario_forecaster import ScenarioForecaster
from ..core.predictive_analytics.forecast_validator import ForecastValidator

logger = logging.getLogger(__name__)


class PredictiveAnalyticsAgent(StrandsBaseAgent):
    """Orchestrates comprehensive predictive analytics capabilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize base agent
        super().__init__()
        
        # Initialize predictive analytics components
        self.forecasting_engine = ForecastingEngine(
            self.config.get('forecasting_engine', {})
        )
        self.confidence_calculator = ConfidenceCalculator(
            self.config.get('confidence_calculator', {})
        )
        self.scenario_forecaster = ScenarioForecaster(
            self.config.get('scenario_forecaster', {})
        )
        self.forecast_validator = ForecastValidator(
            self.config.get('forecast_validator', {})
        )
        
        # Agent state
        self.current_forecast = None
        self.current_confidence_intervals = None
        self.current_scenarios = []
        self.validation_results = []
        
        # Configuration
        self.default_forecast_horizon = self.config.get('default_forecast_horizon', 12)
        self.default_confidence_level = self.config.get('default_confidence_level', 0.95)
        self.max_scenarios = self.config.get('max_scenarios', 5)
        
        # Supported data types
        self.supported_data_types = [DataType.NUMERICAL, DataType.TIME_SERIES]
        
        logger.info("Predictive Analytics Agent initialized")
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        # Predictive analytics agent can process time series and numerical data
        supported_types = ['time_series', 'numerical']
        return request.data_type.value in supported_types
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Process predictive analytics request.
        
        Args:
            request: Processing request with data and parameters
            
        Returns:
            AnalysisResult with analysis results
        """
        try:
            logger.info(f"Processing predictive analytics request: {request.id}")
            
            # Validate request
            if not self._validate_request(request):
                return AnalysisResult(
                    request_id=request.id,
                    data_type=request.data_type,
                    sentiment=SentimentResult(
                        label="uncertain",
                        confidence=0.0,
                        reasoning="Invalid request: Unsupported data type or missing data"
                    ),
                    processing_time=0.0,
                    raw_content=str(request.content),
                    extracted_text="",
                    metadata={"error": "Invalid request: Unsupported data type or missing data"}
                )
            
            # Extract data
            data = self._extract_data(request)
            if data is None:
                return AnalysisResult(
                    request_id=request.id,
                    data_type=request.data_type,
                    sentiment=SentimentResult(
                        label="uncertain",
                        confidence=0.0,
                        reasoning="Failed to extract data from request"
                    ),
                    processing_time=0.0,
                    raw_content=str(request.content),
                    extracted_text="",
                    metadata={"error": "Failed to extract data from request"}
                )
            
            # Determine analysis type
            analysis_type = request.metadata.get('analysis_type', 'forecast')
            
            # Perform analysis based on type
            if analysis_type == 'forecast':
                result = await self._perform_forecasting(data, request.metadata)
            elif analysis_type == 'scenario':
                result = await self._perform_scenario_analysis(data, request.metadata)
            elif analysis_type == 'validation':
                result = await self._perform_validation(data, request.metadata)
            elif analysis_type == 'confidence':
                result = await self._perform_confidence_analysis(data, request.metadata)
            else:
                return AnalysisResult(
                    request_id=request.id,
                    data_type=request.data_type,
                    sentiment=SentimentResult(
                        label="uncertain",
                        confidence=0.0,
                        reasoning=f"Unsupported analysis type: {analysis_type}"
                    ),
                    processing_time=0.0,
                    raw_content=str(request.content),
                    extracted_text="",
                    metadata={"error": f"Unsupported analysis type: {analysis_type}"}
                )
            
            # Create analysis result
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.9,
                    reasoning="Predictive analytics completed successfully"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={
                    'analysis_type': analysis_type,
                    'data_points': len(data),
                    'timestamp': datetime.now().isoformat(),
                    'agent_type': 'predictive_analytics_agent',
                    'result': result
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing predictive analytics request: {str(e)}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="uncertain",
                    confidence=0.0,
                    reasoning=f"Processing error: {str(e)}"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={"error": str(e)}
            )
    
    async def _perform_forecasting(
        self,
        data: np.ndarray,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform forecasting analysis"""
        try:
            # Extract forecasting parameters
            model_type = parameters.get('model_type', 'ensemble')
            forecast_horizon = parameters.get('forecast_horizon', 12)
            confidence_level = parameters.get('confidence_level', 0.95)
            
            # Generate forecast
            forecast_result = self.forecasting_engine.forecast(
                data=data,
                model_type=model_type,
                forecast_horizon=forecast_horizon
            )
            
            # Calculate confidence intervals
            confidence_intervals = self.confidence_calculator.calculate_confidence_intervals(
                predictions=forecast_result.predictions,
                historical_data=data,
                confidence_level=confidence_level
            )
            
            # Calculate uncertainty metrics
            uncertainty_metrics = self.confidence_calculator.calculate_uncertainty_metrics(
                predictions=forecast_result.predictions,
                historical_data=data
            )
            
            return {
                'forecast': {
                    'predictions': forecast_result.predictions.tolist(),
                    'model_type': forecast_result.model_type,
                    'model_accuracy': forecast_result.model_accuracy,
                    'forecast_horizon': forecast_result.forecast_horizon,
                    'metadata': forecast_result.metadata
                },
                'confidence_intervals': {
                    'lower_bound': confidence_intervals.lower_bound.tolist(),
                    'upper_bound': confidence_intervals.upper_bound.tolist(),
                    'confidence_level': confidence_intervals.confidence_level,
                    'method': confidence_intervals.method,
                    'metadata': confidence_intervals.metadata
                },
                'uncertainty_metrics': uncertainty_metrics,
                'analysis_type': 'forecast'
            }
            
        except Exception as e:
            logger.error(f"Error in forecasting: {str(e)}")
            raise
    
    async def _perform_scenario_analysis(
        self,
        data: np.ndarray,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform scenario analysis"""
        try:
            # Extract scenario parameters
            forecast_horizon = parameters.get('forecast_horizon', 12)
            scenario_definitions = parameters.get('scenario_definitions')
            
            # Generate scenarios
            scenarios = self.scenario_forecaster.generate_scenarios(
                base_data=data,
                scenario_definitions=scenario_definitions,
                forecast_horizon=forecast_horizon
            )
            
            # Compare scenarios
            comparison = self.scenario_forecaster.compare_scenarios(scenarios)
            
            # Convert scenarios to serializable format
            scenario_data = []
            for scenario in scenarios:
                scenario_data.append({
                    'name': scenario.name,
                    'description': scenario.description,
                    'predictions': scenario.predictions.tolist(),
                    'probability': scenario.probability,
                    'metadata': scenario.metadata
                })
            
            return {
                'scenarios': scenario_data,
                'comparison': {
                    'best_scenario': comparison.best_scenario,
                    'worst_scenario': comparison.worst_scenario,
                    'comparison_metrics': comparison.comparison_metrics,
                    'metadata': comparison.metadata
                },
                'analysis_type': 'scenario'
            }
            
        except Exception as e:
            logger.error(f"Error in scenario analysis: {str(e)}")
            raise
    
    async def _perform_validation(
        self,
        data: np.ndarray,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform forecast validation"""
        try:
            # Extract validation parameters
            validation_type = parameters.get('validation_type', 'holdout')
            
            # For validation, we need both actual and predicted values
            # If only one array is provided, we'll use a simple split
            if len(data) < 10:
                raise ValueError("Insufficient data for validation")
            
            # Split data for validation (simple approach)
            split_point = len(data) // 2
            actual = data[split_point:]
            predicted = data[:split_point]
            
            # Ensure same length
            min_length = min(len(actual), len(predicted))
            actual = actual[:min_length]
            predicted = predicted[:min_length]
            
            # Perform validation
            validation_result = self.forecast_validator.validate_forecast(
                actual=actual,
                predicted=predicted,
                validation_type=validation_type
            )
            
            # Get performance summary
            performance_summary = self.forecast_validator.get_performance_summary(
                validation_result
            )
            
            return {
                'validation': {
                    'accuracy_metrics': validation_result.accuracy_metrics,
                    'error_analysis': validation_result.error_analysis,
                    'model_performance': validation_result.model_performance,
                    'recommendations': validation_result.recommendations,
                    'metadata': validation_result.metadata
                },
                'performance_summary': performance_summary,
                'analysis_type': 'validation'
            }
            
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            raise
    
    async def _perform_confidence_analysis(
        self,
        data: np.ndarray,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform confidence interval analysis"""
        try:
            # Generate a simple forecast first
            forecast_result = self.forecasting_engine.forecast(
                data=data,
                model_type='ensemble',
                forecast_horizon=12
            )
            
            # Calculate confidence intervals with different methods
            confidence_level = parameters.get('confidence_level', 0.95)
            
            parametric_ci = self.confidence_calculator.calculate_confidence_intervals(
                predictions=forecast_result.predictions,
                historical_data=data,
                confidence_level=confidence_level,
                method='parametric'
            )
            
            bootstrap_ci = self.confidence_calculator.calculate_confidence_intervals(
                predictions=forecast_result.predictions,
                historical_data=data,
                confidence_level=confidence_level,
                method='bootstrap'
            )
            
            # Calculate uncertainty metrics
            uncertainty_metrics = self.confidence_calculator.calculate_uncertainty_metrics(
                predictions=forecast_result.predictions,
                historical_data=data
            )
            
            return {
                'confidence_analysis': {
                    'parametric_intervals': {
                        'lower_bound': parametric_ci.lower_bound.tolist(),
                        'upper_bound': parametric_ci.upper_bound.tolist(),
                        'method': parametric_ci.method,
                        'metadata': parametric_ci.metadata
                    },
                    'bootstrap_intervals': {
                        'lower_bound': bootstrap_ci.lower_bound.tolist(),
                        'upper_bound': bootstrap_ci.upper_bound.tolist(),
                        'method': bootstrap_ci.method,
                        'metadata': bootstrap_ci.metadata
                    },
                    'confidence_level': confidence_level
                },
                'uncertainty_metrics': uncertainty_metrics,
                'forecast_predictions': forecast_result.predictions.tolist(),
                'analysis_type': 'confidence'
            }
            
        except Exception as e:
            logger.error(f"Error in confidence analysis: {str(e)}")
            raise
    
    def _validate_request(self, request: AnalysisRequest) -> bool:
        """Validate analysis request"""
        try:
            # Check data type
            if request.data_type not in self.supported_data_types:
                return False
            
            # Check if content is provided
            if not request.content:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating request: {str(e)}")
            return False
    
    def _extract_data(self, request: AnalysisRequest) -> Optional[np.ndarray]:
        """Extract and convert data from request"""
        try:
            # Handle different content types
            if isinstance(request.content, str):
                # Try to parse JSON string
                import json
                try:
                    content_data = json.loads(request.content)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON content")
                    return None
            elif isinstance(request.content, dict):
                content_data = request.content
            else:
                logger.error(f"Unsupported content type: {type(request.content)}")
                return None
            
            # Extract data from content
            data = content_data.get('time_series_data') or content_data.get('data')
            if data is None:
                logger.error("No time_series_data or data found in request content")
                return None
                
            if isinstance(data, list):
                return np.array(data)
            elif isinstance(data, np.ndarray):
                return data
            else:
                logger.error(f"Unsupported data type: {type(data)}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting data: {str(e)}")
            return None
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities"""
        return {
            'agent_type': self.agent_type,
            'agent_id': self.agent_id,
            'supported_data_types': [dt.value for dt in self.supported_data_types],
            'analysis_types': [
                'forecast',
                'scenario', 
                'validation',
                'confidence'
            ],
            'available_models': self.forecasting_engine.get_available_models(),
            'scenario_templates': list(self.scenario_forecaster.get_scenario_templates().keys()),
            'validation_methods': ['holdout', 'cross_validation'],
            'confidence_methods': ['parametric', 'bootstrap', 'empirical']
        }
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        return self.forecasting_engine.get_model_info(model_type)
    
    def get_scenario_templates(self) -> Dict[str, Any]:
        """Get available scenario templates"""
        return self.scenario_forecaster.get_scenario_templates()
