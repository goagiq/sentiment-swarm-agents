"""
Mitigation Planner Module

This module provides risk reduction strategies and action planning capabilities
for developing comprehensive risk mitigation plans.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import numpy as np

from .risk_identifier import Risk
from .risk_quantifier import RiskImpact

logger = logging.getLogger(__name__)


@dataclass
class MitigationAction:
    """Mitigation action data structure."""
    id: str
    risk_id: str
    action_type: str  # 'preventive', 'detective', 'corrective', 'compensating'
    title: str
    description: str
    priority: str  # 'low', 'medium', 'high', 'critical'
    effort_level: str  # 'low', 'medium', 'high'
    estimated_cost: float
    estimated_duration: int  # days
    success_probability: float  # 0.0 to 1.0
    risk_reduction: float  # 0.0 to 1.0
    dependencies: List[str]
    resources_required: List[str]
    created_at: datetime


@dataclass
class MitigationPlan:
    """Complete mitigation plan data structure."""
    plan_id: str
    risk_id: str
    risk_description: str
    current_risk_level: str
    target_risk_level: str
    actions: List[MitigationAction]
    total_cost: float
    total_duration: int  # days
    expected_risk_reduction: float
    success_probability: float
    created_at: datetime
    status: str  # 'draft', 'approved', 'in_progress', 'completed'


class MitigationPlanner:
    """
    Risk mitigation planning system that generates comprehensive
    risk reduction strategies and action plans.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the mitigation planner with configuration."""
        self.config = config or {}
        self.mitigation_strategies = self._load_mitigation_strategies()
        self.action_templates = self._load_action_templates()
        self.logger = logging.getLogger(__name__)
        
    def _load_mitigation_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined mitigation strategies for different risk categories."""
        return {
            'data_quality': {
                'preventive': [
                    'implement_data_validation',
                    'establish_data_governance',
                    'automate_data_quality_checks'
                ],
                'detective': [
                    'deploy_data_monitoring',
                    'implement_anomaly_detection',
                    'regular_data_audits'
                ],
                'corrective': [
                    'data_cleansing_procedures',
                    'backup_restoration_processes',
                    'data_recovery_protocols'
                ]
            },
            'performance': {
                'preventive': [
                    'capacity_planning',
                    'performance_monitoring',
                    'load_balancing'
                ],
                'detective': [
                    'real_time_monitoring',
                    'performance_alerting',
                    'bottleneck_analysis'
                ],
                'corrective': [
                    'performance_optimization',
                    'infrastructure_scaling',
                    'code_optimization'
                ]
            },
            'security': {
                'preventive': [
                    'access_controls',
                    'encryption_implementation',
                    'security_training'
                ],
                'detective': [
                    'intrusion_detection',
                    'security_monitoring',
                    'vulnerability_assessments'
                ],
                'corrective': [
                    'incident_response',
                    'security_patches',
                    'forensic_analysis'
                ]
            },
            'business': {
                'preventive': [
                    'market_analysis',
                    'customer_feedback_systems',
                    'competitive_intelligence'
                ],
                'detective': [
                    'kpi_monitoring',
                    'trend_analysis',
                    'early_warning_systems'
                ],
                'corrective': [
                    'strategy_adjustment',
                    'process_improvement',
                    'customer_retention_programs'
                ]
            }
        }
    
    def _load_action_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load action templates with cost and duration estimates."""
        return {
            'implement_data_validation': {
                'title': 'Implement Data Validation Framework',
                'description': 'Deploy automated data validation rules and checks',
                'effort_level': 'medium',
                'estimated_cost': 15000,
                'estimated_duration': 14,
                'success_probability': 0.8,
                'risk_reduction': 0.6
            },
            'establish_data_governance': {
                'title': 'Establish Data Governance Framework',
                'description': 'Create data governance policies and procedures',
                'effort_level': 'high',
                'estimated_cost': 25000,
                'estimated_duration': 30,
                'success_probability': 0.7,
                'risk_reduction': 0.7
            },
            'deploy_data_monitoring': {
                'title': 'Deploy Data Quality Monitoring',
                'description': 'Implement real-time data quality monitoring',
                'effort_level': 'medium',
                'estimated_cost': 12000,
                'estimated_duration': 10,
                'success_probability': 0.9,
                'risk_reduction': 0.5
            },
            'capacity_planning': {
                'title': 'Implement Capacity Planning',
                'description': 'Develop capacity planning and scaling strategies',
                'effort_level': 'medium',
                'estimated_cost': 20000,
                'estimated_duration': 21,
                'success_probability': 0.8,
                'risk_reduction': 0.6
            },
            'performance_monitoring': {
                'title': 'Enhance Performance Monitoring',
                'description': 'Deploy comprehensive performance monitoring',
                'effort_level': 'medium',
                'estimated_cost': 18000,
                'estimated_duration': 12,
                'success_probability': 0.9,
                'risk_reduction': 0.5
            },
            'access_controls': {
                'title': 'Strengthen Access Controls',
                'description': 'Implement multi-factor authentication and access controls',
                'effort_level': 'medium',
                'estimated_cost': 22000,
                'estimated_duration': 18,
                'success_probability': 0.8,
                'risk_reduction': 0.7
            },
            'security_training': {
                'title': 'Security Awareness Training',
                'description': 'Conduct comprehensive security training for staff',
                'effort_level': 'low',
                'estimated_cost': 8000,
                'estimated_duration': 7,
                'success_probability': 0.7,
                'risk_reduction': 0.4
            },
            'market_analysis': {
                'title': 'Conduct Market Analysis',
                'description': 'Perform comprehensive market and competitive analysis',
                'effort_level': 'medium',
                'estimated_cost': 15000,
                'estimated_duration': 14,
                'success_probability': 0.8,
                'risk_reduction': 0.5
            },
            'kpi_monitoring': {
                'title': 'Implement KPI Monitoring',
                'description': 'Deploy real-time KPI monitoring and alerting',
                'effort_level': 'medium',
                'estimated_cost': 16000,
                'estimated_duration': 12,
                'success_probability': 0.9,
                'risk_reduction': 0.6
            }
        }
    
    def _calculate_action_priority(self, risk: Risk, action_type: str) -> str:
        """Calculate priority for a mitigation action."""
        # Base priority on risk severity
        severity_priority = {
            'critical': 'critical',
            'high': 'high',
            'medium': 'medium',
            'low': 'low'
        }
        
        base_priority = severity_priority.get(risk.severity, 'medium')
        
        # Adjust based on action type
        if action_type == 'preventive':
            return base_priority
        elif action_type == 'detective':
            # Detective actions are slightly lower priority
            priority_map = {
                'critical': 'high',
                'high': 'medium',
                'medium': 'low',
                'low': 'low'
            }
            return priority_map.get(base_priority, 'medium')
        else:
            return base_priority
    
    def _generate_action_id(self, risk_id: str, action_type: str, index: int) -> str:
        """Generate unique action ID."""
        return f"{risk_id}_{action_type}_{index}_{datetime.now().strftime('%Y%m%d')}"
    
    def create_mitigation_actions(self, risk: Risk, 
                                risk_impact: RiskImpact) -> List[MitigationAction]:
        """Create mitigation actions for a specific risk."""
        actions = []
        strategies = self.mitigation_strategies.get(risk.category, {})
        
        # Generate actions for each strategy type
        for action_type, strategy_list in strategies.items():
            for i, strategy in enumerate(strategy_list):
                if strategy in self.action_templates:
                    template = self.action_templates[strategy]
                    
                    action = MitigationAction(
                        id=self._generate_action_id(risk.id, action_type, i),
                        risk_id=risk.id,
                        action_type=action_type,
                        title=template['title'],
                        description=template['description'],
                        priority=self._calculate_action_priority(risk, action_type),
                        effort_level=template['effort_level'],
                        estimated_cost=template['estimated_cost'],
                        estimated_duration=template['estimated_duration'],
                        success_probability=template['success_probability'],
                        risk_reduction=template['risk_reduction'],
                        dependencies=[],
                        resources_required=self._get_required_resources(strategy),
                        created_at=datetime.now()
                    )
                    actions.append(action)
        
        # Add custom actions based on risk specifics
        custom_actions = self._generate_custom_actions(risk, risk_impact)
        actions.extend(custom_actions)
        
        return actions
    
    def _get_required_resources(self, strategy: str) -> List[str]:
        """Get required resources for a strategy."""
        resource_mapping = {
            'implement_data_validation': ['data_engineer', 'software_developer'],
            'establish_data_governance': ['data_architect', 'business_analyst'],
            'deploy_data_monitoring': ['devops_engineer', 'data_engineer'],
            'capacity_planning': ['system_architect', 'operations_manager'],
            'performance_monitoring': ['devops_engineer', 'performance_engineer'],
            'access_controls': ['security_engineer', 'system_administrator'],
            'security_training': ['security_specialist', 'hr_coordinator'],
            'market_analysis': ['business_analyst', 'market_researcher'],
            'kpi_monitoring': ['business_analyst', 'data_engineer']
        }
        
        return resource_mapping.get(strategy, ['general_staff'])
    
    def _generate_custom_actions(self, risk: Risk, 
                               risk_impact: RiskImpact) -> List[MitigationAction]:
        """Generate custom actions based on specific risk characteristics."""
        custom_actions = []
        
        # Generate custom actions based on risk category and impact
        if risk.category == 'data_quality' and risk_impact.financial_impact > 50000:
            custom_actions.append(MitigationAction(
                id=f"{risk.id}_custom_1_{datetime.now().strftime('%Y%m%d')}",
                risk_id=risk.id,
                action_type='corrective',
                title='Implement Data Quality Dashboard',
                description='Create executive dashboard for data quality monitoring',
                priority='high',
                effort_level='medium',
                estimated_cost=10000,
                estimated_duration=7,
                success_probability=0.8,
                risk_reduction=0.4,
                dependencies=[],
                resources_required=['data_engineer', 'ui_developer'],
                created_at=datetime.now()
            ))
        
        elif risk.category == 'security' and risk_impact.compliance_impact > 0.7:
            custom_actions.append(MitigationAction(
                id=f"{risk.id}_custom_1_{datetime.now().strftime('%Y%m%d')}",
                risk_id=risk.id,
                action_type='preventive',
                title='Compliance Audit and Gap Analysis',
                description='Conduct comprehensive compliance audit',
                priority='critical',
                effort_level='high',
                estimated_cost=30000,
                estimated_duration=21,
                success_probability=0.9,
                risk_reduction=0.8,
                dependencies=[],
                resources_required=['compliance_officer', 'security_auditor'],
                created_at=datetime.now()
            ))
        
        return custom_actions
    
    def create_mitigation_plan(self, risk: Risk, 
                             risk_impact: RiskImpact,
                             target_risk_level: str = 'low') -> MitigationPlan:
        """Create a comprehensive mitigation plan for a risk."""
        
        # Generate mitigation actions
        actions = self.create_mitigation_actions(risk, risk_impact)
        
        # Calculate plan metrics
        total_cost = sum(action.estimated_cost for action in actions)
        total_duration = max(action.estimated_duration for action in actions) if actions else 0
        
        # Calculate expected risk reduction
        if actions:
            # Weighted average of risk reduction based on success probability
            total_weighted_reduction = sum(
                action.risk_reduction * action.success_probability 
                for action in actions
            )
            expected_risk_reduction = min(total_weighted_reduction, 1.0)
        else:
            expected_risk_reduction = 0.0
        
        # Calculate overall success probability
        if actions:
            success_probability = np.mean([action.success_probability for action in actions])
        else:
            success_probability = 0.0
        
        plan = MitigationPlan(
            plan_id=f"plan_{risk.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            risk_id=risk.id,
            risk_description=risk.description,
            current_risk_level=risk.severity,
            target_risk_level=target_risk_level,
            actions=actions,
            total_cost=total_cost,
            total_duration=total_duration,
            expected_risk_reduction=expected_risk_reduction,
            success_probability=success_probability,
            created_at=datetime.now(),
            status='draft'
        )
        
        self.logger.info(f"Created mitigation plan for risk {risk.id} with {len(actions)} actions")
        return plan
    
    def optimize_mitigation_plan(self, plan: MitigationPlan, 
                               budget_constraint: Optional[float] = None,
                               timeline_constraint: Optional[int] = None) -> MitigationPlan:
        """Optimize mitigation plan based on constraints."""
        
        if not budget_constraint and not timeline_constraint:
            return plan
        
        # Sort actions by cost-effectiveness (risk reduction per dollar)
        cost_effective_actions = sorted(
            plan.actions,
            key=lambda x: x.risk_reduction / x.estimated_cost if x.estimated_cost > 0 else 0,
            reverse=True
        )
        
        optimized_actions = []
        total_cost = 0
        max_duration = 0
        
        for action in cost_effective_actions:
            # Check budget constraint
            if budget_constraint and total_cost + action.estimated_cost > budget_constraint:
                continue
            
            # Check timeline constraint
            if timeline_constraint and action.estimated_duration > timeline_constraint:
                continue
            
            optimized_actions.append(action)
            total_cost += action.estimated_cost
            max_duration = max(max_duration, action.estimated_duration)
        
        # Recalculate plan metrics
        if optimized_actions:
            expected_risk_reduction = min(
                sum(action.risk_reduction * action.success_probability 
                    for action in optimized_actions), 1.0
            )
            success_probability = np.mean([action.success_probability for action in optimized_actions])
        else:
            expected_risk_reduction = 0.0
            success_probability = 0.0
        
        # Create optimized plan
        optimized_plan = MitigationPlan(
            plan_id=f"{plan.plan_id}_optimized",
            risk_id=plan.risk_id,
            risk_description=plan.risk_description,
            current_risk_level=plan.current_risk_level,
            target_risk_level=plan.target_risk_level,
            actions=optimized_actions,
            total_cost=total_cost,
            total_duration=max_duration,
            expected_risk_reduction=expected_risk_reduction,
            success_probability=success_probability,
            created_at=datetime.now(),
            status='draft'
        )
        
        self.logger.info(f"Optimized plan: {len(optimized_actions)} actions, "
                        f"cost: ${total_cost:,.2f}, duration: {max_duration} days")
        
        return optimized_plan
    
    def get_mitigation_summary(self, plans: List[MitigationPlan]) -> Dict[str, Any]:
        """Generate a summary of mitigation plans."""
        if not plans:
            return {'total_plans': 0, 'total_cost': 0}
        
        total_cost = sum(plan.total_cost for plan in plans)
        total_duration = max(plan.total_duration for plan in plans) if plans else 0
        avg_risk_reduction = np.mean([plan.expected_risk_reduction for plan in plans])
        avg_success_probability = np.mean([plan.success_probability for plan in plans])
        
        # Plan distribution by priority
        priority_distribution = {}
        for plan in plans:
            for action in plan.actions:
                priority = action.priority
                priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
        
        return {
            'total_plans': len(plans),
            'total_cost': total_cost,
            'total_duration': total_duration,
            'avg_risk_reduction': avg_risk_reduction,
            'avg_success_probability': avg_success_probability,
            'priority_distribution': priority_distribution
        }
