"""
Scenario Builder Component

Provides interactive scenario creation and management capabilities for what-if analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


class ScenarioType(Enum):
    """Types of scenarios that can be created."""
    BASELINE = "baseline"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    CUSTOM = "custom"
    STRESS_TEST = "stress_test"
    SENSITIVITY = "sensitivity"


class ParameterType(Enum):
    """Types of parameters that can be modified in scenarios."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TIME_SERIES = "time_series"
    PERCENTAGE = "percentage"


@dataclass
class ScenarioParameter:
    """Represents a parameter that can be modified in a scenario."""
    name: str
    parameter_type: ParameterType
    current_value: Any
    scenario_value: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    description: str = ""
    unit: Optional[str] = None
    confidence_level: float = 0.95


@dataclass
class Scenario:
    """Represents a complete scenario configuration."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    scenario_type: ScenarioType = ScenarioType.CUSTOM
    parameters: Dict[str, ScenarioParameter] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'scenario_type': self.scenario_type.value,
            'parameters': {
                name: {
                    'name': param.name,
                    'parameter_type': param.parameter_type.value,
                    'current_value': param.current_value,
                    'scenario_value': param.scenario_value,
                    'min_value': param.min_value,
                    'max_value': param.max_value,
                    'description': param.description,
                    'unit': param.unit,
                    'confidence_level': param.confidence_level
                }
                for name, param in self.parameters.items()
            },
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat(),
            'tags': self.tags,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Scenario':
        """Create scenario from dictionary."""
        scenario = cls(
            id=data.get('id', str(uuid.uuid4())),
            name=data['name'],
            description=data['description'],
            scenario_type=ScenarioType(data['scenario_type']),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {})
        )
        
        # Parse parameters
        for name, param_data in data.get('parameters', {}).items():
            scenario.parameters[name] = ScenarioParameter(
                name=param_data['name'],
                parameter_type=ParameterType(param_data['parameter_type']),
                current_value=param_data['current_value'],
                scenario_value=param_data['scenario_value'],
                min_value=param_data.get('min_value'),
                max_value=param_data.get('max_value'),
                description=param_data.get('description', ''),
                unit=param_data.get('unit'),
                confidence_level=param_data.get('confidence_level', 0.95)
            )
        
        return scenario


class ScenarioBuilder:
    """
    Interactive scenario creation and management system.
    
    Provides capabilities for:
    - Creating custom scenarios
    - Managing scenario parameters
    - Template-based scenario generation
    - Scenario validation and optimization
    """
    
    def __init__(self):
        self.scenarios: Dict[str, Scenario] = {}
        self.templates: Dict[str, Dict[str, Any]] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default scenario templates."""
        self.templates = {
            'baseline': {
                'name': 'Baseline Scenario',
                'description': 'Current state without modifications',
                'scenario_type': ScenarioType.BASELINE,
                'parameters': {}
            },
            'optimistic': {
                'name': 'Optimistic Scenario',
                'description': 'Best-case scenario with favorable conditions',
                'scenario_type': ScenarioType.OPTIMISTIC,
                'parameters': {}
            },
            'pessimistic': {
                'name': 'Pessimistic Scenario',
                'description': 'Worst-case scenario with unfavorable conditions',
                'scenario_type': ScenarioType.PESSIMISTIC,
                'parameters': {}
            },
            'stress_test': {
                'name': 'Stress Test Scenario',
                'description': 'Extreme conditions to test system resilience',
                'scenario_type': ScenarioType.STRESS_TEST,
                'parameters': {}
            }
        }
    
    def create_scenario(self, name: str, description: str = "", 
                       scenario_type: ScenarioType = ScenarioType.CUSTOM,
                       tags: Optional[List[str]] = None) -> Scenario:
        """
        Create a new scenario.
        
        Args:
            name: Scenario name
            description: Scenario description
            scenario_type: Type of scenario
            tags: Optional tags for categorization
            
        Returns:
            Created scenario object
        """
        scenario = Scenario(
            name=name,
            description=description,
            scenario_type=scenario_type,
            tags=tags or []
        )
        
        self.scenarios[scenario.id] = scenario
        logger.info(f"Created scenario: {name} (ID: {scenario.id})")
        return scenario
    
    def create_scenario_from_template(self, template_name: str, 
                                    custom_name: Optional[str] = None,
                                    **kwargs) -> Scenario:
        """
        Create a scenario from a predefined template.
        
        Args:
            template_name: Name of the template to use
            custom_name: Optional custom name for the scenario
            **kwargs: Additional parameters to override template defaults
            
        Returns:
            Created scenario object
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name].copy()
        template.update(kwargs)
        
        name = custom_name or template['name']
        scenario = self.create_scenario(
            name=name,
            description=template['description'],
            scenario_type=template['scenario_type'],
            tags=template.get('tags', [])
        )
        
        # Add template parameters
        for param_name, param_data in template.get('parameters', {}).items():
            self.add_parameter(
                scenario.id, param_name, param_data['parameter_type'],
                param_data['current_value'], param_data['scenario_value'],
                **{k: v for k, v in param_data.items() 
                   if k not in ['parameter_type', 'current_value', 'scenario_value']}
            )
        
        return scenario
    
    def add_parameter(self, scenario_id: str, name: str, 
                     parameter_type: ParameterType,
                     current_value: Any, scenario_value: Any,
                     **kwargs) -> ScenarioParameter:
        """
        Add a parameter to a scenario.
        
        Args:
            scenario_id: ID of the scenario
            name: Parameter name
            parameter_type: Type of parameter
            current_value: Current/baseline value
            scenario_value: Value for this scenario
            **kwargs: Additional parameter attributes
            
        Returns:
            Created parameter object
        """
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_id}' not found")
        
        parameter = ScenarioParameter(
            name=name,
            parameter_type=parameter_type,
            current_value=current_value,
            scenario_value=scenario_value,
            **kwargs
        )
        
        self.scenarios[scenario_id].parameters[name] = parameter
        self.scenarios[scenario_id].modified_at = datetime.now()
        
        logger.info(f"Added parameter '{name}' to scenario '{scenario_id}'")
        return parameter
    
    def update_parameter(self, scenario_id: str, name: str, 
                        **kwargs) -> ScenarioParameter:
        """
        Update a parameter in a scenario.
        
        Args:
            scenario_id: ID of the scenario
            name: Parameter name
            **kwargs: Parameters to update
            
        Returns:
            Updated parameter object
        """
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_id}' not found")
        
        if name not in self.scenarios[scenario_id].parameters:
            raise ValueError(f"Parameter '{name}' not found in scenario")
        
        parameter = self.scenarios[scenario_id].parameters[name]
        
        for key, value in kwargs.items():
            if hasattr(parameter, key):
                setattr(parameter, key, value)
        
        self.scenarios[scenario_id].modified_at = datetime.now()
        
        logger.info(f"Updated parameter '{name}' in scenario '{scenario_id}'")
        return parameter
    
    def remove_parameter(self, scenario_id: str, name: str) -> bool:
        """
        Remove a parameter from a scenario.
        
        Args:
            scenario_id: ID of the scenario
            name: Parameter name
            
        Returns:
            True if parameter was removed, False if not found
        """
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_id}' not found")
        
        if name in self.scenarios[scenario_id].parameters:
            del self.scenarios[scenario_id].parameters[name]
            self.scenarios[scenario_id].modified_at = datetime.now()
            logger.info(f"Removed parameter '{name}' from scenario '{scenario_id}'")
            return True
        
        return False
    
    def get_scenario(self, scenario_id: str) -> Optional[Scenario]:
        """Get a scenario by ID."""
        return self.scenarios.get(scenario_id)
    
    def list_scenarios(self, scenario_type: Optional[ScenarioType] = None,
                      tags: Optional[List[str]] = None) -> List[Scenario]:
        """
        List scenarios with optional filtering.
        
        Args:
            scenario_type: Filter by scenario type
            tags: Filter by tags (scenario must have all specified tags)
            
        Returns:
            List of matching scenarios
        """
        scenarios = list(self.scenarios.values())
        
        if scenario_type:
            scenarios = [s for s in scenarios if s.scenario_type == scenario_type]
        
        if tags:
            scenarios = [s for s in scenarios 
                        if all(tag in s.tags for tag in tags)]
        
        return scenarios
    
    def delete_scenario(self, scenario_id: str) -> bool:
        """
        Delete a scenario.
        
        Args:
            scenario_id: ID of the scenario to delete
            
        Returns:
            True if scenario was deleted, False if not found
        """
        if scenario_id in self.scenarios:
            del self.scenarios[scenario_id]
            logger.info(f"Deleted scenario '{scenario_id}'")
            return True
        
        return False
    
    def validate_scenario(self, scenario_id: str) -> Dict[str, Any]:
        """
        Validate a scenario for completeness and consistency.
        
        Args:
            scenario_id: ID of the scenario to validate
            
        Returns:
            Validation results
        """
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_id}' not found")
        
        scenario = self.scenarios[scenario_id]
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'parameter_count': len(scenario.parameters),
            'missing_required_fields': []
        }
        
        # Check required fields
        if not scenario.name:
            validation_result['errors'].append("Scenario name is required")
            validation_result['is_valid'] = False
        
        # Validate parameters
        for param_name, parameter in scenario.parameters.items():
            # Check parameter bounds
            if parameter.parameter_type == ParameterType.NUMERICAL:
                if parameter.min_value is not None and parameter.max_value is not None:
                    if parameter.min_value > parameter.max_value:
                        validation_result['errors'].append(
                            f"Parameter '{param_name}': min_value > max_value"
                        )
                        validation_result['is_valid'] = False
                    
                    if not (parameter.min_value <= parameter.scenario_value <= parameter.max_value):
                        validation_result['warnings'].append(
                            f"Parameter '{param_name}': scenario_value outside bounds"
                        )
            
            # Check confidence level
            if not (0 <= parameter.confidence_level <= 1):
                validation_result['errors'].append(
                    f"Parameter '{param_name}': confidence_level must be between 0 and 1"
                )
                validation_result['is_valid'] = False
        
        return validation_result
    
    def export_scenario(self, scenario_id: str, format: str = 'json') -> str:
        """
        Export a scenario to various formats.
        
        Args:
            scenario_id: ID of the scenario to export
            format: Export format ('json', 'yaml')
            
        Returns:
            Exported scenario as string
        """
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario '{scenario_id}' not found")
        
        scenario = self.scenarios[scenario_id]
        
        if format.lower() == 'json':
            return json.dumps(scenario.to_dict(), indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def import_scenario(self, data: Union[str, Dict[str, Any]], 
                       format: str = 'json') -> Scenario:
        """
        Import a scenario from various formats.
        
        Args:
            data: Scenario data (string or dict)
            format: Import format ('json', 'yaml')
            
        Returns:
            Imported scenario object
        """
        if format.lower() == 'json':
            if isinstance(data, str):
                scenario_data = json.loads(data)
            else:
                scenario_data = data
            
            scenario = Scenario.from_dict(scenario_data)
            self.scenarios[scenario.id] = scenario
            
            logger.info(f"Imported scenario: {scenario.name} (ID: {scenario.id})")
            return scenario
        else:
            raise ValueError(f"Unsupported import format: {format}")
