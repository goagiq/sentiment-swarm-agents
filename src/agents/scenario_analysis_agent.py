"""
Scenario Analysis Agent

Coordinates scenario analysis components including scenario building, impact analysis,
risk assessment, and scenario comparison.
"""

import logging
from typing import Dict, List, Optional, Any
from src.agents.base_agent import StrandsBaseAgent
from src.core.models import AnalysisRequest, AnalysisResult, DataType, SentimentResult
from src.core.scenario_analysis import (
    ScenarioBuilder, ImpactAnalyzer, RiskAssessor, ScenarioComparator
)
from src.core.scenario_analysis.scenario_builder import Scenario, ScenarioType, ParameterType

logger = logging.getLogger(__name__)


class ScenarioAnalysisAgent(StrandsBaseAgent):
    """
    Agent for scenario analysis and what-if analysis capabilities.
    
    Provides:
    - Scenario creation and management
    - Impact analysis and quantification
    - Risk assessment and scoring
    - Multi-scenario comparison
    """
    
    def __init__(self):
        super().__init__()
        self.scenario_builder = ScenarioBuilder()
        self.impact_analyzer = ImpactAnalyzer()
        self.risk_assessor = RiskAssessor()
        self.scenario_comparator = ScenarioComparator()
        
        logger.info("Scenario Analysis Agent initialized")
    
    def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the request."""
        # Check if the request is for scenario analysis
        try:
            if isinstance(request.content, str):
                import json
                content_data = json.loads(request.content)
            elif isinstance(request.content, dict):
                content_data = request.content
            else:
                return False
            
            analysis_type = content_data.get('analysis_type')
            return analysis_type in [
                'scenario_building',
                'impact_analysis', 
                'risk_assessment',
                'scenario_comparison',
                'what_if_analysis'
            ]
        except:
            return False
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Process scenario analysis request.
        
        Args:
            request: Analysis request containing scenario analysis parameters
            
        Returns:
            Analysis result with scenario analysis outcomes
        """
        # Extract content data
        try:
            if isinstance(request.content, str):
                import json
                content_data = json.loads(request.content)
            elif isinstance(request.content, dict):
                content_data = request.content
            else:
                raise ValueError("Unsupported content type")
        except Exception as e:
            logger.error(f"Error parsing request content: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="uncertain",
                    confidence=0.0,
                    reasoning=f"Error parsing request content: {str(e)}"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={"error": str(e), "agent": "scenario_analysis_agent"}
            )
        
        # Get analysis type from content
        analysis_type = content_data.get('analysis_type')
        logger.info(f"Processing scenario analysis request: {analysis_type}")
        
        try:
            if analysis_type == 'scenario_building':
                result = await self._handle_scenario_building(request, content_data)
            elif analysis_type == 'impact_analysis':
                result = await self._handle_impact_analysis(request, content_data)
            elif analysis_type == 'risk_assessment':
                result = await self._handle_risk_assessment(request, content_data)
            elif analysis_type == 'scenario_comparison':
                result = await self._handle_scenario_comparison(request, content_data)
            elif analysis_type == 'what_if_analysis':
                result = await self._handle_what_if_analysis(request, content_data)
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing scenario analysis request: {e}")
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
                metadata={"error": str(e), "agent": "scenario_analysis_agent"}
            )
    
    async def _handle_scenario_building(self, request: AnalysisRequest, content_data: Dict[str, Any]) -> AnalysisResult:
        """Handle scenario building requests."""
        try:
            # Extract scenario parameters from request
            scenario_data = content_data.get('scenario', {})
            scenario_name = scenario_data.get('name', 'New Scenario')
            scenario_description = scenario_data.get('description', '')
            scenario_type = ScenarioType(scenario_data.get('type', 'custom'))
            tags = scenario_data.get('tags', [])
            
            # Create scenario
            scenario = self.scenario_builder.create_scenario(
                name=scenario_name,
                description=scenario_description,
                scenario_type=scenario_type,
                tags=tags
            )
            
            # Add parameters if provided
            parameters_data = scenario_data.get('parameters', {})
            for param_name, param_data in parameters_data.items():
                self.scenario_builder.add_parameter(
                    scenario.id,
                    param_name,
                    ParameterType(param_data.get('type', 'numerical')),
                    param_data.get('current_value', 0),
                    param_data.get('scenario_value', 0),
                    min_value=param_data.get('min_value'),
                    max_value=param_data.get('max_value'),
                    description=param_data.get('description', ''),
                    unit=param_data.get('unit'),
                    confidence_level=param_data.get('confidence_level', 0.95)
                )
            
            # Validate scenario
            validation = self.scenario_builder.validate_scenario(scenario.id)
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.9,
                    reasoning="Scenario building completed successfully"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={
                    'agent': 'scenario_analysis_agent',
                    'scenario_id': scenario.id,
                    'parameter_count': len(scenario.parameters),
                    'result': {
                        'scenario': scenario.to_dict(),
                        'validation': validation
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error in scenario building: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="uncertain",
                    confidence=0.0,
                    reasoning=f"Scenario building error: {str(e)}"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={"error": str(e), "agent": "scenario_analysis_agent"}
            )
    
    async def _handle_impact_analysis(self, request: AnalysisRequest, content_data: Dict[str, Any]) -> AnalysisResult:
        """Handle impact analysis requests."""
        try:
            # Get scenario ID from request
            scenario_id = content_data.get('scenario_id')
            if not scenario_id:
                raise ValueError("scenario_id is required for impact analysis")
            
            # Get scenario
            scenario = self.scenario_builder.get_scenario(scenario_id)
            if not scenario:
                raise ValueError(f"Scenario {scenario_id} not found")
            
            # Get baseline data if provided
            baseline_data = content_data.get('baseline_data')
            
            # Perform impact analysis
            impact_analysis = self.impact_analyzer.analyze_impact(
                scenario, baseline_data
            )
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.9,
                    reasoning="Impact analysis completed successfully"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={
                    'agent': 'scenario_analysis_agent',
                    'scenario_id': scenario_id,
                    'overall_impact_score': impact_analysis.overall_impact_score,
                    'overall_risk_level': impact_analysis.overall_risk_level,
                    'result': {
                        'impact_analysis': impact_analysis.to_dict(),
                        'scenario': scenario.to_dict()
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error in impact analysis: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="uncertain",
                    confidence=0.0,
                    reasoning=f"Impact analysis error: {str(e)}"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={"error": str(e), "agent": "scenario_analysis_agent"}
            )
    
    async def _handle_risk_assessment(self, request: AnalysisRequest, content_data: Dict[str, Any]) -> AnalysisResult:
        """Handle risk assessment requests."""
        try:
            # Get scenario ID from request
            scenario_id = content_data.get('scenario_id')
            if not scenario_id:
                raise ValueError("scenario_id is required for risk assessment")
            
            # Get scenario
            scenario = self.scenario_builder.get_scenario(scenario_id)
            if not scenario:
                raise ValueError(f"Scenario {scenario_id} not found")
            
            # Get baseline data if provided
            baseline_data = content_data.get('baseline_data')
            
            # Perform risk assessment
            risk_assessment = self.risk_assessor.assess_risk(
                scenario, baseline_data
            )
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.9,
                    reasoning="Risk assessment completed successfully"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={
                    'agent': 'scenario_analysis_agent',
                    'scenario_id': scenario_id,
                    'overall_risk_score': risk_assessment.overall_risk_score,
                    'overall_risk_level': risk_assessment.overall_risk_level.value,
                    'high_risk_factors_count': len(risk_assessment.high_risk_factors),
                    'result': {
                        'risk_assessment': risk_assessment.to_dict(),
                        'scenario': scenario.to_dict()
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="uncertain",
                    confidence=0.0,
                    reasoning=f"Risk assessment error: {str(e)}"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={"error": str(e), "agent": "scenario_analysis_agent"}
            )
    
    async def _handle_scenario_comparison(self, request: AnalysisRequest, content_data: Dict[str, Any]) -> AnalysisResult:
        """Handle scenario comparison requests."""
        try:
            # Get scenario IDs from request
            scenario_ids = content_data.get('scenario_ids', [])
            if len(scenario_ids) < 2:
                raise ValueError("At least 2 scenario IDs required for comparison")
            
            # Get scenarios
            scenarios = []
            for scenario_id in scenario_ids:
                scenario = self.scenario_builder.get_scenario(scenario_id)
                if not scenario:
                    raise ValueError(f"Scenario {scenario_id} not found")
                scenarios.append(scenario)
            
            # Get comparison type
            comparison_type = content_data.get('comparison_type', 'pairwise')
            
            # Perform comparison
            comparison_result = self.scenario_comparator.compare_scenarios(
                scenarios, comparison_type=comparison_type
            )
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.9,
                    reasoning="Scenario comparison completed successfully"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={
                    'agent': 'scenario_analysis_agent',
                    'scenarios_compared': len(scenarios),
                    'comparison_type': comparison_type,
                    'comparison_id': comparison_result.comparison_id,
                    'result': {
                        'comparison_result': comparison_result.to_dict(),
                        'scenarios': [s.to_dict() for s in scenarios]
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error in scenario comparison: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="uncertain",
                    confidence=0.0,
                    reasoning=f"Scenario comparison error: {str(e)}"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={"error": str(e), "agent": "scenario_analysis_agent"}
            )
    
    async def _handle_what_if_analysis(self, request: AnalysisRequest, content_data: Dict[str, Any]) -> AnalysisResult:
        """Handle comprehensive what-if analysis requests."""
        try:
            # Get scenario ID from request
            scenario_id = content_data.get('scenario_id')
            if not scenario_id:
                raise ValueError("scenario_id is required for what-if analysis")
            
            # Get scenario
            scenario = self.scenario_builder.get_scenario(scenario_id)
            if not scenario:
                raise ValueError(f"Scenario {scenario_id} not found")
            
            # Get baseline data if provided
            baseline_data = content_data.get('baseline_data')
            
            # Perform comprehensive analysis
            impact_analysis = self.impact_analyzer.analyze_impact(
                scenario, baseline_data
            )
            
            risk_assessment = self.risk_assessor.assess_risk(
                scenario, baseline_data
            )
            
            # Get similar scenarios for comparison
            similar_scenarios = self.scenario_builder.list_scenarios(
                scenario_type=scenario.scenario_type,
                tags=scenario.tags
            )
            
            comparison_result = None
            if len(similar_scenarios) > 1:
                comparison_result = self.scenario_comparator.compare_scenarios(
                    similar_scenarios, comparison_type='pairwise'
                )
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.9,
                    reasoning="What-if analysis completed successfully"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={
                    'agent': 'scenario_analysis_agent',
                    'scenario_id': scenario_id,
                    'overall_impact_score': impact_analysis.overall_impact_score,
                    'overall_risk_level': risk_assessment.overall_risk_level.value,
                    'analysis_comprehensive': True,
                    'result': {
                        'scenario': scenario.to_dict(),
                        'impact_analysis': impact_analysis.to_dict(),
                        'risk_assessment': risk_assessment.to_dict(),
                        'comparison_result': comparison_result.to_dict() if comparison_result else None,
                        'similar_scenarios_count': len(similar_scenarios)
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error in what-if analysis: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="uncertain",
                    confidence=0.0,
                    reasoning=f"What-if analysis error: {str(e)}"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={"error": str(e), "agent": "scenario_analysis_agent"}
            )
    
    def get_available_scenarios(self, scenario_type: Optional[str] = None,
                              tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get available scenarios with optional filtering."""
        try:
            scenarios = self.scenario_builder.list_scenarios(
                scenario_type=ScenarioType(scenario_type) if scenario_type else None,
                tags=tags
            )
            
            return [scenario.to_dict() for scenario in scenarios]
            
        except Exception as e:
            logger.error(f"Error getting available scenarios: {e}")
            return []
    
    def delete_scenario(self, scenario_id: str) -> bool:
        """Delete a scenario."""
        try:
            return self.scenario_builder.delete_scenario(scenario_id)
        except Exception as e:
            logger.error(f"Error deleting scenario {scenario_id}: {e}")
            return False
    
    def export_scenario(self, scenario_id: str, format: str = 'json') -> Optional[str]:
        """Export a scenario."""
        try:
            return self.scenario_builder.export_scenario(scenario_id, format)
        except Exception as e:
            logger.error(f"Error exporting scenario {scenario_id}: {e}")
            return None
    
    def import_scenario(self, data: str, format: str = 'json') -> Optional[Dict[str, Any]]:
        """Import a scenario."""
        try:
            scenario = self.scenario_builder.import_scenario(data, format)
            return scenario.to_dict()
        except Exception as e:
            logger.error(f"Error importing scenario: {e}")
            return None
