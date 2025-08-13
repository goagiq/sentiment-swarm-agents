"""
Advanced Forecasting Agent

Advanced forecasting agent for multivariate forecasting, causal inference,
and scenario analysis using Phase 7.2 advanced analytics capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
import json

# Local imports
from .base_agent import BaseAgent
from ..core.models import AnalysisRequest, AnalysisResult, ProcessingStatus
from ..core.advanced_analytics.multivariate_forecasting import MultivariateForecastingEngine
from ..core.advanced_analytics.causal_inference_engine import CausalInferenceEngine
from ..core.advanced_analytics.scenario_analysis import ScenarioAnalysisEngine
from ..core.advanced_analytics.confidence_intervals import ConfidenceIntervalCalculator
from ..core.advanced_analytics.advanced_anomaly_detection import AdvancedAnomalyDetector
from ..core.advanced_analytics.model_optimization import ModelOptimizer
from ..core.advanced_analytics.feature_engineering import FeatureEngineer
from ..core.advanced_analytics.performance_monitoring import AdvancedPerformanceMonitor
from ..config.advanced_analytics_config import get_advanced_analytics_config
from ..core.error_handling_service import ErrorHandlingService
from ..core.caching_service import CachingService

logger = logging.getLogger(__name__)


class AdvancedForecastingAgent(BaseAgent):
    """
    Advanced forecasting agent for sophisticated multivariate forecasting
    and causal analysis using Phase 7.2 capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the advanced forecasting agent"""
        super().__init__()
        
        # Load configuration
        self.analytics_config = get_advanced_analytics_config()
        self.config = config or {}
        
        # Initialize services
        self.error_handler = ErrorHandlingService()
        self.caching_service = CachingService()
        
        # Initialize advanced analytics engines
        self.forecasting_engine = MultivariateForecastingEngine(
            self.analytics_config.multivariate_forecasting.__dict__
        )
        self.causal_engine = CausalInferenceEngine(
            self.analytics_config.causal_inference.__dict__
        )
        self.scenario_engine = ScenarioAnalysisEngine(
            self.analytics_config.scenario_analysis.__dict__
        )
        self.confidence_calculator = ConfidenceIntervalCalculator(
            self.analytics_config.confidence_intervals.__dict__
        )
        self.anomaly_detector = AdvancedAnomalyDetector(
            self.analytics_config.anomaly_detection.__dict__
        )
        self.model_optimizer = ModelOptimizer(
            self.analytics_config.model_optimization.__dict__
        )
        self.feature_engineer = FeatureEngineer(
            self.analytics_config.feature_engineering.__dict__
        )
        self.performance_monitor = AdvancedPerformanceMonitor(
            self.analytics_config.performance_monitoring.__dict__
        )
        
        # Agent metadata
        self.agent_name = "AdvancedForecastingAgent"
        self.agent_version = "7.2.0"
        self.agent_description = "Advanced forecasting agent with multivariate analysis capabilities"
        
        logger.info(f"{self.agent_name} initialized with version {self.agent_version}")
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        try:
            # Check if request contains forecasting-related data
            content = request.content.lower()
            keywords = ['forecast', 'prediction', 'trend', 'time series', 'multivariate', 'causal']
            return any(keyword in content for keyword in keywords)
        except Exception as e:
            logger.error(f"Error checking if agent can process request: {e}")
            return False
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process the analysis request."""
        try:
            # Convert AnalysisRequest to dict format for processing
            request_dict = {
                'content': request.content,
                'type': 'multivariate_forecast',  # Default type
                'metadata': request.metadata or {}
            }
            
            # Process the request
            result = self.process_request(request_dict)
            
            # Convert result to AnalysisResult
            return AnalysisResult(
                id=request.id,
                content=request.content,
                result=result,
                status=ProcessingStatus.COMPLETED,
                processing_time=0.0,
                metadata=result.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return AnalysisResult(
                id=request.id,
                content=request.content,
                result={'error': str(e)},
                status=ProcessingStatus.FAILED,
                processing_time=0.0,
                metadata={}
            )
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process advanced forecasting request
        
        Args:
            request: Request dictionary containing parameters
            
        Returns:
            Response dictionary with results
        """
        try:
            request_type = request.get('type', 'forecast')
            
            if request_type == 'multivariate_forecast':
                return self._handle_multivariate_forecast(request)
            elif request_type == 'causal_analysis':
                return self._handle_causal_analysis(request)
            elif request_type == 'scenario_analysis':
                return self._handle_scenario_analysis(request)
            elif request_type == 'anomaly_detection':
                return self._handle_anomaly_detection(request)
            elif request_type == 'model_optimization':
                return self._handle_model_optimization(request)
            elif request_type == 'feature_engineering':
                return self._handle_feature_engineering(request)
            elif request_type == 'performance_monitoring':
                return self._handle_performance_monitoring(request)
            else:
                return self._handle_comprehensive_analysis(request)
                
        except Exception as e:
            self.error_handler.handle_error(f"Error processing request: {str(e)}", e)
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _handle_multivariate_forecast(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle multivariate forecasting request"""
        try:
            # Extract parameters
            data = pd.DataFrame(request.get('data', []))
            target_columns = request.get('target_columns', [])
            forecast_horizon = request.get('forecast_horizon', 12)
            model_type = request.get('model_type', 'ensemble')
            include_confidence_intervals = request.get('include_confidence_intervals', True)
            
            # Validate input
            if data.empty or not target_columns:
                raise ValueError("Data and target_columns are required")
            
            # Perform multivariate forecasting
            forecast_result = self.forecasting_engine.forecast_multivariate(
                data=data,
                target_columns=target_columns,
                forecast_horizon=forecast_horizon,
                model_type=model_type,
                include_confidence_intervals=include_confidence_intervals
            )
            
            # Prepare response
            response = {
                'success': True,
                'type': 'multivariate_forecast',
                'predictions': forecast_result.predictions.to_dict(),
                'confidence_intervals': {
                    k: {'lower': v[0].tolist(), 'upper': v[1].tolist()}
                    for k, v in forecast_result.confidence_intervals.items()
                },
                'model_performance': forecast_result.model_performance,
                'causal_relationships': forecast_result.causal_relationships,
                'feature_importance': forecast_result.feature_importance,
                'forecast_horizon': forecast_result.forecast_horizon,
                'timestamp': forecast_result.timestamp.isoformat()
            }
            
            return response
            
        except Exception as e:
            self.error_handler.handle_error(f"Error in multivariate forecasting: {str(e)}", e)
            raise
    
    def _handle_causal_analysis(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle causal analysis request"""
        try:
            # Extract parameters
            data = pd.DataFrame(request.get('data', []))
            methods = request.get('methods', ['correlation', 'granger_causality'])
            variables = request.get('variables', None)
            
            # Validate input
            if data.empty:
                raise ValueError("Data is required for causal analysis")
            
            # Perform causal analysis
            causal_result = self.causal_engine.perform_causal_analysis(
                data=data,
                methods=methods,
                variables=variables
            )
            
            # Prepare response
            response = {
                'success': True,
                'type': 'causal_analysis',
                'relationships': [
                    {
                        'cause': rel.cause,
                        'effect': rel.effect,
                        'strength': rel.strength,
                        'confidence': rel.confidence,
                        'method': rel.method,
                        'p_value': rel.p_value,
                        'direction': rel.direction
                    }
                    for rel in causal_result.relationships
                ],
                'causal_graph': causal_result.causal_graph,
                'strength_matrix': causal_result.strength_matrix.to_dict(),
                'confidence_matrix': causal_result.confidence_matrix.to_dict(),
                'analysis_methods': causal_result.analysis_methods,
                'timestamp': causal_result.timestamp.isoformat()
            }
            
            return response
            
        except Exception as e:
            self.error_handler.handle_error(f"Error in causal analysis: {str(e)}", e)
            raise
    
    def _handle_scenario_analysis(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle scenario analysis request"""
        try:
            # Extract parameters
            data = pd.DataFrame(request.get('data', []))
            scenarios = request.get('scenarios', [])
            target_columns = request.get('target_columns', [])
            forecast_horizon = request.get('forecast_horizon', 12)
            
            # Validate input
            if data.empty or not target_columns:
                raise ValueError("Data and target_columns are required")
            
            # Create scenarios if not provided
            if not scenarios:
                scenarios_dict = self.scenario_engine.create_default_scenarios(data)
                scenarios = list(scenarios_dict.values())
            
            # Perform scenario analysis
            scenario_results = self.scenario_engine.analyze_multiple_scenarios(
                scenarios=scenarios,
                baseline_data=data,
                forecast_model=self.forecasting_engine,
                target_columns=target_columns,
                forecast_horizon=forecast_horizon
            )
            
            # Prepare response
            response = {
                'success': True,
                'type': 'scenario_analysis',
                'scenarios': [
                    {
                        'scenario_name': result.scenario.name,
                        'description': result.scenario.description,
                        'impact_metrics': result.impact_metrics,
                        'risk_assessment': result.risk_assessment,
                        'sensitivity_analysis': result.sensitivity_analysis
                    }
                    for result in scenario_results
                ],
                'comparison': self.scenario_engine.get_scenario_comparison(
                    [result.scenario.name for result in scenario_results]
                ).to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            self.error_handler.handle_error(f"Error in scenario analysis: {str(e)}", e)
            raise
    
    def _handle_anomaly_detection(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle anomaly detection request"""
        try:
            # Extract parameters
            data = pd.DataFrame(request.get('data', []))
            methods = request.get('methods', ['isolation_forest', 'statistical'])
            ensemble = request.get('ensemble', True)
            
            # Validate input
            if data.empty:
                raise ValueError("Data is required for anomaly detection")
            
            # Perform anomaly detection
            anomaly_results = self.anomaly_detector.detect_multivariate_anomalies(
                data=data,
                methods=methods,
                ensemble=ensemble
            )
            
            # Prepare response
            response = {
                'success': True,
                'type': 'anomaly_detection',
                'results': {}
            }
            
            for method_name, result in anomaly_results.items():
                response['results'][method_name] = {
                    'anomalies': [
                        {
                            'index': anomaly.index,
                            'value': anomaly.value,
                            'score': anomaly.score,
                            'method': anomaly.method,
                            'severity': anomaly.severity
                        }
                        for anomaly in result.anomalies
                    ],
                    'anomaly_scores': result.anomaly_scores.to_dict(),
                    'threshold': result.threshold,
                    'performance_metrics': result.performance_metrics
                }
            
            response['summary'] = self.anomaly_detector.get_detection_summary()
            response['timestamp'] = datetime.now().isoformat()
            
            return response
            
        except Exception as e:
            self.error_handler.handle_error(f"Error in anomaly detection: {str(e)}", e)
            raise
    
    def _handle_model_optimization(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model optimization request"""
        try:
            # Extract parameters
            data = pd.DataFrame(request.get('data', []))
            target_column = request.get('target_column', '')
            optimization_type = request.get('optimization_type', 'hyperparameter')
            
            # Validate input
            if data.empty or not target_column:
                raise ValueError("Data and target_column are required")
            
            # Prepare data
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            if optimization_type == 'hyperparameter':
                # Hyperparameter optimization
                from sklearn.ensemble import RandomForestRegressor
                
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
                
                model = RandomForestRegressor(random_state=42)
                result = self.model_optimizer.optimize_hyperparameters(
                    model=model,
                    X=X,
                    y=y,
                    param_grid=param_grid,
                    method='grid_search'
                )
                
                response = {
                    'success': True,
                    'type': 'model_optimization',
                    'optimization_type': 'hyperparameter',
                    'best_params': result.best_params,
                    'best_score': result.best_score,
                    'optimization_method': result.optimization_method,
                    'cv_scores': result.cv_scores,
                    'feature_importance': result.feature_importance
                }
                
            elif optimization_type == 'feature_selection':
                # Feature selection optimization
                from sklearn.ensemble import RandomForestRegressor
                
                model = RandomForestRegressor(random_state=42)
                result = self.model_optimizer.optimize_feature_selection(
                    model=model,
                    X=X,
                    y=y,
                    method='recursive'
                )
                
                response = {
                    'success': True,
                    'type': 'model_optimization',
                    'optimization_type': 'feature_selection',
                    'selected_features': result.best_params.get('selected_features', []),
                    'best_score': result.best_score,
                    'optimization_method': result.optimization_method
                }
            
            response['timestamp'] = datetime.now().isoformat()
            return response
            
        except Exception as e:
            self.error_handler.handle_error(f"Error in model optimization: {str(e)}", e)
            raise
    
    def _handle_feature_engineering(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle feature engineering request"""
        try:
            # Extract parameters
            data = pd.DataFrame(request.get('data', []))
            engineering_type = request.get('engineering_type', 'time_series')
            date_column = request.get('date_column', None)
            target_columns = request.get('target_columns', None)
            
            # Validate input
            if data.empty:
                raise ValueError("Data is required for feature engineering")
            
            if engineering_type == 'time_series':
                if not date_column:
                    raise ValueError("date_column is required for time series feature engineering")
                
                result = self.feature_engineer.create_time_series_features(
                    data=data,
                    date_column=date_column,
                    target_columns=target_columns
                )
                
            elif engineering_type == 'statistical':
                result = self.feature_engineer.create_statistical_features(
                    data=data,
                    target_columns=target_columns
                )
                
            elif engineering_type == 'interaction':
                result = self.feature_engineer.create_interaction_features(
                    data=data,
                    feature_columns=target_columns
                )
                
            elif engineering_type == 'scaling':
                method = request.get('scaling_method', 'standard')
                result = self.feature_engineer.apply_feature_scaling(
                    data=data,
                    method=method
                )
                
            else:
                raise ValueError(f"Unsupported engineering type: {engineering_type}")
            
            # Prepare response
            response = {
                'success': True,
                'type': 'feature_engineering',
                'engineering_type': engineering_type,
                'engineered_features': result.engineered_features.to_dict(),
                'feature_names': result.feature_names,
                'feature_types': result.feature_types,
                'transformation_info': result.transformation_info,
                'timestamp': result.timestamp.isoformat()
            }
            
            return response
            
        except Exception as e:
            self.error_handler.handle_error(f"Error in feature engineering: {str(e)}", e)
            raise
    
    def _handle_performance_monitoring(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance monitoring request"""
        try:
            # Extract parameters
            action = request.get('action', 'get_summary')
            
            if action == 'start_monitoring':
                self.performance_monitor.start_monitoring()
                response = {
                    'success': True,
                    'type': 'performance_monitoring',
                    'action': 'start_monitoring',
                    'message': 'Performance monitoring started'
                }
                
            elif action == 'stop_monitoring':
                self.performance_monitor.stop_monitoring()
                response = {
                    'success': True,
                    'type': 'performance_monitoring',
                    'action': 'stop_monitoring',
                    'message': 'Performance monitoring stopped'
                }
                
            elif action == 'get_summary':
                summary = self.performance_monitor.get_performance_summary()
                response = {
                    'success': True,
                    'type': 'performance_monitoring',
                    'action': 'get_summary',
                    'summary': summary
                }
                
            elif action == 'track_model_performance':
                model_name = request.get('model_name', 'unknown_model')
                predictions = np.array(request.get('predictions', []))
                actual_values = np.array(request.get('actual_values', []))
                
                metrics = self.performance_monitor.track_model_performance(
                    model_name=model_name,
                    predictions=predictions,
                    actual_values=actual_values
                )
                
                response = {
                    'success': True,
                    'type': 'performance_monitoring',
                    'action': 'track_model_performance',
                    'model_name': model_name,
                    'metrics': metrics
                }
                
            else:
                raise ValueError(f"Unsupported action: {action}")
            
            response['timestamp'] = datetime.now().isoformat()
            return response
            
        except Exception as e:
            self.error_handler.handle_error(f"Error in performance monitoring: {str(e)}", e)
            raise
    
    def _handle_comprehensive_analysis(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle comprehensive analysis request"""
        try:
            # Extract parameters
            data = pd.DataFrame(request.get('data', []))
            target_columns = request.get('target_columns', [])
            analysis_components = request.get('components', [
                'forecasting', 'causal_analysis', 'anomaly_detection'
            ])
            
            # Validate input
            if data.empty or not target_columns:
                raise ValueError("Data and target_columns are required")
            
            comprehensive_results = {}
            
            # Perform requested analyses
            if 'forecasting' in analysis_components:
                comprehensive_results['forecasting'] = self._handle_multivariate_forecast({
                    'data': data.to_dict(),
                    'target_columns': target_columns,
                    'forecast_horizon': 12
                })
            
            if 'causal_analysis' in analysis_components:
                comprehensive_results['causal_analysis'] = self._handle_causal_analysis({
                    'data': data.to_dict(),
                    'methods': ['correlation', 'granger_causality']
                })
            
            if 'anomaly_detection' in analysis_components:
                comprehensive_results['anomaly_detection'] = self._handle_anomaly_detection({
                    'data': data.to_dict(),
                    'methods': ['isolation_forest', 'statistical']
                })
            
            if 'scenario_analysis' in analysis_components:
                comprehensive_results['scenario_analysis'] = self._handle_scenario_analysis({
                    'data': data.to_dict(),
                    'target_columns': target_columns
                })
            
            # Prepare response
            response = {
                'success': True,
                'type': 'comprehensive_analysis',
                'components': analysis_components,
                'results': comprehensive_results,
                'timestamp': datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            self.error_handler.handle_error(f"Error in comprehensive analysis: {str(e)}", e)
            raise
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'name': self.agent_name,
            'version': self.agent_version,
            'description': self.agent_description,
            'capabilities': [
                'multivariate_forecasting',
                'causal_analysis',
                'scenario_analysis',
                'anomaly_detection',
                'model_optimization',
                'feature_engineering',
                'performance_monitoring'
            ],
            'config': self.analytics_config.__dict__
        }
    
    def export_results(self, filepath: str) -> None:
        """Export agent results"""
        try:
            export_data = {
                'agent_info': self.get_agent_info(),
                'forecasting_history': [
                    {
                        'timestamp': result.timestamp.isoformat(),
                        'forecast_horizon': result.forecast_horizon,
                        'target_columns': list(result.predictions.columns)
                    }
                    for result in self.forecasting_engine.get_forecast_history()
                ],
                'causal_analysis_history': [
                    {
                        'timestamp': result.timestamp.isoformat(),
                        'methods': result.analysis_methods,
                        'relationship_count': len(result.relationships)
                    }
                    for result in self.causal_engine.analysis_history
                ],
                'scenario_analysis_history': [
                    {
                        'timestamp': result.timestamp.isoformat(),
                        'scenario_name': result.scenario.name
                    }
                    for result in self.scenario_engine.scenario_results
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Agent results exported to {filepath}")
            
        except Exception as e:
            self.error_handler.handle_error(f"Error exporting results: {str(e)}", e)
