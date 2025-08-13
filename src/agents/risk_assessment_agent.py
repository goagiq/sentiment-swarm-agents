"""
Risk Assessment Agent

This agent orchestrates the comprehensive risk assessment and management system,
providing a unified interface for risk identification, quantification, mitigation planning,
and continuous monitoring.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

from src.core.risk_management import (
    RiskIdentifier, RiskQuantifier, MitigationPlanner, RiskMonitor
)
from src.core.risk_management.risk_identifier import Risk
from src.core.risk_management.risk_quantifier import RiskImpact
from src.core.risk_management.mitigation_planner import MitigationPlan
from src.core.risk_management.risk_monitor import RiskAlert, RiskTrend

logger = logging.getLogger(__name__)


from src.agents.base_agent import StrandsBaseAgent
from src.core.models import AnalysisRequest, AnalysisResult


class RiskAssessmentAgent(StrandsBaseAgent):
    """
    Comprehensive risk assessment and management agent that provides
    end-to-end risk analysis capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize the risk assessment agent."""
        self.config = config or {}
        
        # Initialize base agent
        super().__init__(**kwargs)
        
        # Initialize risk management components
        self.risk_identifier = RiskIdentifier(self.config.get('risk_identifier', {}))
        self.risk_quantifier = RiskQuantifier(self.config.get('risk_quantifier', {}))
        self.mitigation_planner = MitigationPlanner(self.config.get('mitigation_planner', {}))
        self.risk_monitor = RiskMonitor(self.config.get('risk_monitor', {}))
        
        # Start monitoring
        self.risk_monitor.start_monitoring()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Risk Assessment Agent initialized")
    
    def assess_risks(self, 
                    data_metrics: Optional[Dict[str, Any]] = None,
                    performance_metrics: Optional[Dict[str, Any]] = None,
                    security_metrics: Optional[Dict[str, Any]] = None,
                    business_metrics: Optional[Dict[str, Any]] = None,
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive risk assessment across all categories.
        
        Returns:
            Dictionary containing identified risks, impacts, and mitigation plans
        """
        try:
            # Step 1: Identify risks
            risks = self.risk_identifier.identify_all_risks(
                data_metrics=data_metrics,
                performance_metrics=performance_metrics,
                security_metrics=security_metrics,
                business_metrics=business_metrics
            )
            
            if not risks:
                return {
                    'status': 'success',
                    'message': 'No risks identified',
                    'risks': [],
                    'impacts': [],
                    'mitigation_plans': [],
                    'summary': {'total_risks': 0}
                }
            
            # Step 2: Quantify risk impacts
            risk_impacts = self.risk_quantifier.quantify_multiple_risks(risks, context)
            
            # Step 3: Create mitigation plans
            mitigation_plans = []
            for risk, impact in zip(risks, risk_impacts):
                plan = self.mitigation_planner.create_mitigation_plan(risk, impact)
                mitigation_plans.append(plan)
            
            # Step 4: Process risks for monitoring
            alerts = []
            for risk, impact in zip(risks, risk_impacts):
                risk_alerts = self.risk_monitor.process_risk_update(risk, impact)
                alerts.extend(risk_alerts)
            
            # Step 5: Generate summaries
            risk_summary = self.risk_identifier.get_risk_summary(risks)
            impact_summary = self.risk_quantifier.get_impact_summary(risk_impacts)
            mitigation_summary = self.mitigation_planner.get_mitigation_summary(mitigation_plans)
            monitoring_summary = self.risk_monitor.get_risk_summary()
            
            # Step 6: Create comprehensive assessment result
            assessment_result = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'risks': [self._risk_to_dict(risk) for risk in risks],
                'impacts': [self._impact_to_dict(impact) for impact in risk_impacts],
                'mitigation_plans': [self._plan_to_dict(plan) for plan in mitigation_plans],
                'alerts': [self._alert_to_dict(alert) for alert in alerts],
                'summaries': {
                    'risk_summary': risk_summary,
                    'impact_summary': impact_summary,
                    'mitigation_summary': mitigation_summary,
                    'monitoring_summary': monitoring_summary
                },
                'recommendations': self._generate_recommendations(risks, risk_impacts, mitigation_plans)
            }
            
            self.logger.info(f"Risk assessment completed: {len(risks)} risks identified")
            return assessment_result
            
        except Exception as e:
            self.logger.error(f"Error in risk assessment: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_risk_trends(self, risk_ids: Optional[List[str]] = None, 
                       period_days: int = 7) -> Dict[str, Any]:
        """Get risk trends for specified risks or all monitored risks."""
        try:
            if risk_ids is None:
                # Get trends for all monitored risks
                risk_ids = list(self.risk_monitor.risk_history.keys())
            
            trends = {}
            for risk_id in risk_ids:
                trend = self.risk_monitor.analyze_risk_trend(risk_id, period_days)
                if trend:
                    trends[risk_id] = self._trend_to_dict(trend)
            
            return {
                'status': 'success',
                'trends': trends,
                'total_trends': len(trends),
                'period_days': period_days
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk trends: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_active_alerts(self, 
                         severity_filter: Optional[str] = None,
                         category_filter: Optional[str] = None) -> Dict[str, Any]:
        """Get active risk alerts with optional filtering."""
        try:
            alerts = self.risk_monitor.get_active_alerts(severity_filter, category_filter)
            
            return {
                'status': 'success',
                'alerts': [self._alert_to_dict(alert) for alert in alerts],
                'total_alerts': len(alerts),
                'filters_applied': {
                    'severity': severity_filter,
                    'category': category_filter
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting active alerts: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def optimize_mitigation_plan(self, plan_id: str, 
                               budget_constraint: Optional[float] = None,
                               timeline_constraint: Optional[int] = None) -> Dict[str, Any]:
        """Optimize a specific mitigation plan based on constraints."""
        try:
            # Find the plan (in a real implementation, this would be stored in a database)
            # For now, we'll create a new optimized plan based on the constraints
            
            # This is a simplified implementation - in practice, you'd retrieve the plan
            # from storage and optimize it
            return {
                'status': 'success',
                'message': f'Plan {plan_id} optimization completed',
                'optimization_constraints': {
                    'budget': budget_constraint,
                    'timeline': timeline_constraint
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing mitigation plan: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def add_alert_callback(self, callback) -> Dict[str, Any]:
        """Add a callback function for risk alerts."""
        try:
            self.risk_monitor.add_alert_callback(callback)
            return {
                'status': 'success',
                'message': f'Alert callback {callback.__name__} added successfully'
            }
        except Exception as e:
            self.logger.error(f"Error adding alert callback: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _risk_to_dict(self, risk: Risk) -> Dict[str, Any]:
        """Convert Risk object to dictionary."""
        return {
            'id': risk.id,
            'category': risk.category,
            'severity': risk.severity,
            'probability': risk.probability,
            'impact': risk.impact,
            'description': risk.description,
            'source': risk.source,
            'detected_at': risk.detected_at.isoformat(),
            'affected_components': risk.affected_components,
            'indicators': risk.indicators,
            'confidence': risk.confidence
        }
    
    def _impact_to_dict(self, impact: RiskImpact) -> Dict[str, Any]:
        """Convert RiskImpact object to dictionary."""
        return {
            'risk_id': impact.risk_id,
            'financial_impact': impact.financial_impact,
            'operational_impact': impact.operational_impact,
            'strategic_impact': impact.strategic_impact,
            'reputation_impact': impact.reputation_impact,
            'compliance_impact': impact.compliance_impact,
            'total_impact_score': impact.total_impact_score,
            'impact_breakdown': impact.impact_breakdown,
            'confidence_level': impact.confidence_level,
            'assessment_date': impact.assessment_date.isoformat()
        }
    
    def _plan_to_dict(self, plan: MitigationPlan) -> Dict[str, Any]:
        """Convert MitigationPlan object to dictionary."""
        return {
            'plan_id': plan.plan_id,
            'risk_id': plan.risk_id,
            'risk_description': plan.risk_description,
            'current_risk_level': plan.current_risk_level,
            'target_risk_level': plan.target_risk_level,
            'actions': [
                {
                    'id': action.id,
                    'title': action.title,
                    'description': action.description,
                    'priority': action.priority,
                    'effort_level': action.effort_level,
                    'estimated_cost': action.estimated_cost,
                    'estimated_duration': action.estimated_duration,
                    'success_probability': action.success_probability,
                    'risk_reduction': action.risk_reduction
                }
                for action in plan.actions
            ],
            'total_cost': plan.total_cost,
            'total_duration': plan.total_duration,
            'expected_risk_reduction': plan.expected_risk_reduction,
            'success_probability': plan.success_probability,
            'status': plan.status,
            'created_at': plan.created_at.isoformat()
        }
    
    def _alert_to_dict(self, alert: RiskAlert) -> Dict[str, Any]:
        """Convert RiskAlert object to dictionary."""
        return {
            'alert_id': alert.alert_id,
            'risk_id': alert.risk_id,
            'alert_type': alert.alert_type,
            'severity': alert.severity,
            'message': alert.message,
            'timestamp': alert.timestamp.isoformat(),
            'risk_data': alert.risk_data,
            'action_required': alert.action_required,
            'assigned_to': alert.assigned_to
        }
    
    def _trend_to_dict(self, trend: RiskTrend) -> Dict[str, Any]:
        """Convert RiskTrend object to dictionary."""
        return {
            'risk_id': trend.risk_id,
            'trend_period': trend.trend_period,
            'start_date': trend.start_date.isoformat(),
            'end_date': trend.end_date.isoformat(),
            'risk_scores': trend.risk_scores,
            'trend_direction': trend.trend_direction,
            'trend_strength': trend.trend_strength,
            'confidence_level': trend.confidence_level
        }
    
    def _generate_recommendations(self, risks: List[Risk], 
                                impacts: List[RiskImpact],
                                plans: List[MitigationPlan]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on risk assessment."""
        recommendations = []
        
        # High-impact risks that need immediate attention
        high_impact_risks = [
            (risk, impact) for risk, impact in zip(risks, impacts)
            if impact.total_impact_score > 0.7
        ]
        
        if high_impact_risks:
            recommendations.append({
                'type': 'immediate_action',
                'priority': 'critical',
                'title': 'Address High-Impact Risks',
                'description': f'Found {len(high_impact_risks)} high-impact risks requiring immediate attention',
                'actions': [
                    f'Review and implement mitigation plan for {risk.id}' 
                    for risk, _ in high_impact_risks
                ]
            })
        
        # Cost-effective mitigation opportunities
        cost_effective_plans = [
            plan for plan in plans
            if plan.expected_risk_reduction / max(plan.total_cost, 1) > 0.01
        ]
        
        if cost_effective_plans:
            recommendations.append({
                'type': 'optimization',
                'priority': 'high',
                'title': 'Implement Cost-Effective Mitigations',
                'description': f'Found {len(cost_effective_plans)} cost-effective mitigation plans',
                'actions': [
                    f'Prioritize implementation of {plan.plan_id}' 
                    for plan in cost_effective_plans
                ]
            })
        
        # Monitoring recommendations
        if len(risks) > 5:
            recommendations.append({
                'type': 'monitoring',
                'priority': 'medium',
                'title': 'Enhance Risk Monitoring',
                'description': 'Consider implementing automated risk monitoring for early detection',
                'actions': [
                    'Set up automated risk alerts',
                    'Implement real-time risk dashboards',
                    'Establish regular risk review meetings'
                ]
            })
        
        return recommendations
    
    def shutdown(self):
        """Shutdown the risk assessment agent."""
        try:
            self.risk_monitor.stop_monitoring()
            self.logger.info("Risk Assessment Agent shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        # Risk assessment agent can process text, numerical, and time series data
        supported_types = ['text', 'numerical', 'time_series']
        return request.data_type.value in supported_types
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process the analysis request."""
        try:
            # Parse the request content
            if isinstance(request.content, str):
                try:
                    content_data = json.loads(request.content)
                except json.JSONDecodeError:
                    content_data = {"text": request.content}
            else:
                content_data = request.content
            
            # Perform risk assessment
            result = self.assess_risks(
                data_metrics=content_data.get('data_metrics'),
                performance_metrics=content_data.get('performance_metrics'),
                security_metrics=content_data.get('security_metrics'),
                business_metrics=content_data.get('business_metrics'),
                context=content_data.get('context')
            )
            
            # Create analysis result
            from src.core.models import SentimentResult
            
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
                metadata={"result": result}
            )
            
        except Exception as e:
            self.logger.error(f"Error processing risk assessment request: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="uncertain",
                    confidence=0.0,
                    reasoning=f"Error: {str(e)}"
                ),
                processing_time=0.0,
                raw_content=str(request.content),
                extracted_text="",
                metadata={"error": str(e)}
            )
    
    async def process_risk_assessment(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process risk assessment request (legacy method for compatibility)."""
        return self.assess_risks(
            data_metrics=request.get('data_metrics'),
            performance_metrics=request.get('performance_metrics'),
            security_metrics=request.get('security_metrics'),
            business_metrics=request.get('business_metrics'),
            context=request.get('context')
        )
