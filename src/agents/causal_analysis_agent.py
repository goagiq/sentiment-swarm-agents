"""
Causal Analysis Agent

Advanced causal analysis agent for identifying cause-effect relationships
and performing sophisticated causal inference using Phase 7.2 capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import json

# Local imports
from .base_agent import BaseAgent
from ..core.models import AnalysisRequest, AnalysisResult, ProcessingStatus
from ..core.advanced_analytics.causal_inference_engine import CausalInferenceEngine
from ..config.advanced_analytics_config import get_advanced_analytics_config
from ..core.error_handling_service import ErrorHandlingService
from ..core.caching_service import CachingService

logger = logging.getLogger(__name__)


class CausalAnalysisAgent(BaseAgent):
    """
    Advanced causal analysis agent for sophisticated cause-effect relationship
    identification and causal inference using Phase 7.2 capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the causal analysis agent"""
        super().__init__()
        
        # Load configuration
        self.analytics_config = get_advanced_analytics_config()
        self.config = config or {}
        
        # Initialize services
        self.error_handler = ErrorHandlingService()
        self.caching_service = CachingService()
        
        # Initialize causal inference engine
        self.causal_engine = CausalInferenceEngine(
            self.analytics_config.causal_inference.__dict__
        )
        
        # Agent metadata
        self.agent_name = "CausalAnalysisAgent"
        self.agent_version = "7.2.0"
        self.agent_description = "Advanced causal analysis agent for cause-effect relationship identification"
        
        logger.info(f"{self.agent_name} initialized with version {self.agent_version}")
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        try:
            # Check if request contains causal analysis-related data
            content = request.content.lower()
            keywords = ['causal', 'cause', 'effect', 'relationship', 'correlation', 'granger']
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
                'type': 'causal_analysis',  # Default type
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
        Process causal analysis request
        
        Args:
            request: Request dictionary containing parameters
            
        Returns:
            Response dictionary with results
        """
        try:
            request_type = request.get('type', 'causal_analysis')
            
            if request_type == 'granger_causality':
                return self._handle_granger_causality(request)
            elif request_type == 'correlation_analysis':
                return self._handle_correlation_analysis(request)
            elif request_type == 'conditional_independence':
                return self._handle_conditional_independence(request)
            elif request_type == 'causal_paths':
                return self._handle_causal_paths(request)
            elif request_type == 'causal_graph':
                return self._handle_causal_graph(request)
            else:
                return self._handle_comprehensive_causal_analysis(request)
                
        except Exception as e:
            self.error_handler.handle_error(f"Error processing request: {str(e)}", e)
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _handle_granger_causality(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Granger causality analysis request"""
        try:
            # Extract parameters
            data = pd.DataFrame(request.get('data', []))
            variables = request.get('variables', None)
            max_lag = request.get('max_lag', 10)
            
            # Validate input
            if data.empty:
                raise ValueError("Data is required for Granger causality analysis")
            
            # Perform Granger causality analysis
            granger_relationships = self.causal_engine.detect_granger_causality(
                data=data,
                variables=variables,
                max_lag=max_lag
            )
            
            # Prepare response
            response = {
                'success': True,
                'type': 'granger_causality',
                'relationships': [
                    {
                        'cause': rel.cause,
                        'effect': rel.effect,
                        'strength': rel.strength,
                        'confidence': rel.confidence,
                        'p_value': rel.p_value,
                        'method': rel.method
                    }
                    for rel in granger_relationships
                ],
                'analysis_params': {
                    'max_lag': max_lag,
                    'variables': variables or list(data.columns),
                    'significance_level': self.analytics_config.causal_inference.significance_level
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            self.error_handler.handle_error(f"Error in Granger causality analysis: {str(e)}", e)
            raise
    
    def _handle_correlation_analysis(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle correlation analysis request"""
        try:
            # Extract parameters
            data = pd.DataFrame(request.get('data', []))
            variables = request.get('variables', None)
            method = request.get('method', 'pearson')
            
            # Validate input
            if data.empty:
                raise ValueError("Data is required for correlation analysis")
            
            # Perform correlation analysis
            correlation_relationships = self.causal_engine.detect_correlation_causality(
                data=data,
                variables=variables,
                method=method
            )
            
            # Prepare response
            response = {
                'success': True,
                'type': 'correlation_analysis',
                'relationships': [
                    {
                        'cause': rel.cause,
                        'effect': rel.effect,
                        'strength': rel.strength,
                        'confidence': rel.confidence,
                        'direction': rel.direction,
                        'method': rel.method
                    }
                    for rel in correlation_relationships
                ],
                'analysis_params': {
                    'method': method,
                    'variables': variables or list(data.columns),
                    'threshold': self.analytics_config.causal_inference.min_relationship_strength
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            self.error_handler.handle_error(f"Error in correlation analysis: {str(e)}", e)
            raise
    
    def _handle_conditional_independence(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle conditional independence analysis request"""
        try:
            # Extract parameters
            data = pd.DataFrame(request.get('data', []))
            variables = request.get('variables', None)
            conditioning_vars = request.get('conditioning_vars', None)
            
            # Validate input
            if data.empty:
                raise ValueError("Data is required for conditional independence analysis")
            
            # Perform conditional independence analysis
            cond_relationships = self.causal_engine.detect_conditional_independence(
                data=data,
                variables=variables,
                conditioning_vars=conditioning_vars
            )
            
            # Prepare response
            response = {
                'success': True,
                'type': 'conditional_independence',
                'relationships': [
                    {
                        'cause': rel.cause,
                        'effect': rel.effect,
                        'strength': rel.strength,
                        'confidence': rel.confidence,
                        'direction': rel.direction,
                        'method': rel.method
                    }
                    for rel in cond_relationships
                ],
                'analysis_params': {
                    'variables': variables or list(data.columns),
                    'conditioning_vars': conditioning_vars or [],
                    'threshold': self.analytics_config.causal_inference.min_relationship_strength
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            self.error_handler.handle_error(f"Error in conditional independence analysis: {str(e)}", e)
            raise
    
    def _handle_causal_paths(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle causal paths analysis request"""
        try:
            # Extract parameters
            start_variable = request.get('start_variable', '')
            end_variable = request.get('end_variable', '')
            max_path_length = request.get('max_path_length', 3)
            
            # Validate input
            if not start_variable or not end_variable:
                raise ValueError("start_variable and end_variable are required")
            
            # Find causal paths
            causal_paths = self.causal_engine.get_causal_paths(
                start_variable=start_variable,
                end_variable=end_variable,
                max_path_length=max_path_length
            )
            
            # Prepare response
            response = {
                'success': True,
                'type': 'causal_paths',
                'start_variable': start_variable,
                'end_variable': end_variable,
                'max_path_length': max_path_length,
                'causal_paths': causal_paths,
                'path_count': len(causal_paths),
                'timestamp': datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            self.error_handler.handle_error(f"Error in causal paths analysis: {str(e)}", e)
            raise
    
    def _handle_causal_graph(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle causal graph analysis request"""
        try:
            # Extract parameters
            data = pd.DataFrame(request.get('data', []))
            methods = request.get('methods', ['correlation', 'granger_causality'])
            variables = request.get('variables', None)
            
            # Validate input
            if data.empty:
                raise ValueError("Data is required for causal graph analysis")
            
            # Perform comprehensive causal analysis
            causal_result = self.causal_engine.perform_causal_analysis(
                data=data,
                methods=methods,
                variables=variables
            )
            
            # Prepare response
            response = {
                'success': True,
                'type': 'causal_graph',
                'causal_graph': causal_result.causal_graph,
                'strength_matrix': causal_result.strength_matrix.to_dict(),
                'confidence_matrix': causal_result.confidence_matrix.to_dict(),
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
                'analysis_params': {
                    'methods': methods,
                    'variables': variables or list(data.columns),
                    'significance_level': self.analytics_config.causal_inference.significance_level
                },
                'timestamp': causal_result.timestamp.isoformat()
            }
            
            return response
            
        except Exception as e:
            self.error_handler.handle_error(f"Error in causal graph analysis: {str(e)}", e)
            raise
    
    def _handle_comprehensive_causal_analysis(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle comprehensive causal analysis request"""
        try:
            # Extract parameters
            data = pd.DataFrame(request.get('data', []))
            methods = request.get('methods', [
                'correlation', 'granger_causality', 'conditional_independence'
            ])
            variables = request.get('variables', None)
            
            # Validate input
            if data.empty:
                raise ValueError("Data is required for comprehensive causal analysis")
            
            # Perform comprehensive causal analysis
            causal_result = self.causal_engine.perform_causal_analysis(
                data=data,
                methods=methods,
                variables=variables
            )
            
            # Get strongest relationships
            strongest_relationships = self.causal_engine.get_strongest_relationships(
                min_strength=0.5,
                min_confidence=0.7
            )
            
            # Prepare response
            response = {
                'success': True,
                'type': 'comprehensive_causal_analysis',
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
                'strongest_relationships': [
                    {
                        'cause': rel.cause,
                        'effect': rel.effect,
                        'strength': rel.strength,
                        'confidence': rel.confidence,
                        'method': rel.method
                    }
                    for rel in strongest_relationships
                ],
                'causal_graph': causal_result.causal_graph,
                'strength_matrix': causal_result.strength_matrix.to_dict(),
                'confidence_matrix': causal_result.confidence_matrix.to_dict(),
                'analysis_summary': self.causal_engine.get_analysis_summary(),
                'analysis_params': {
                    'methods': methods,
                    'variables': variables or list(data.columns),
                    'significance_level': self.analytics_config.causal_inference.significance_level,
                    'min_relationship_strength': self.analytics_config.causal_inference.min_relationship_strength
                },
                'timestamp': causal_result.timestamp.isoformat()
            }
            
            return response
            
        except Exception as e:
            self.error_handler.handle_error(f"Error in comprehensive causal analysis: {str(e)}", e)
            raise
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            'name': self.agent_name,
            'version': self.agent_version,
            'description': self.agent_description,
            'capabilities': [
                'granger_causality',
                'correlation_analysis',
                'conditional_independence',
                'causal_paths',
                'causal_graph',
                'comprehensive_causal_analysis'
            ],
            'config': self.analytics_config.causal_inference.__dict__
        }
    
    def export_results(self, filepath: str) -> None:
        """Export agent results"""
        try:
            export_data = {
                'agent_info': self.get_agent_info(),
                'causal_analysis_history': [
                    {
                        'timestamp': result.timestamp.isoformat(),
                        'methods': result.analysis_methods,
                        'relationship_count': len(result.relationships)
                    }
                    for result in self.causal_engine.analysis_history
                ],
                'strongest_relationships': [
                    {
                        'cause': rel.cause,
                        'effect': rel.effect,
                        'strength': rel.strength,
                        'confidence': rel.confidence,
                        'method': rel.method
                    }
                    for rel in self.causal_engine.get_strongest_relationships()
                ],
                'analysis_summary': self.causal_engine.get_analysis_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Agent results exported to {filepath}")
            
        except Exception as e:
            self.error_handler.handle_error(f"Error exporting results: {str(e)}", e)
