"""
Risk Quantifier Module

This module provides risk impact measurement and quantification capabilities
for assessing the financial, operational, and strategic impact of risks.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

from .risk_identifier import Risk

logger = logging.getLogger(__name__)


@dataclass
class RiskImpact:
    """Risk impact assessment data structure."""
    risk_id: str
    financial_impact: float  # Monetary value
    operational_impact: float  # 0.0 to 1.0 scale
    strategic_impact: float  # 0.0 to 1.0 scale
    reputation_impact: float  # 0.0 to 1.0 scale
    compliance_impact: float  # 0.0 to 1.0 scale
    total_impact_score: float  # 0.0 to 1.0 scale
    impact_breakdown: Dict[str, float]
    confidence_level: float  # 0.0 to 1.0
    assessment_date: datetime


class RiskQuantifier:
    """
    Risk impact measurement and quantification system that assesses
    the financial, operational, strategic, and compliance impact of risks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the risk quantifier with configuration."""
        self.config = config or {}
        self.impact_weights = self._load_impact_weights()
        self.financial_models = self._load_financial_models()
        self.logger = logging.getLogger(__name__)
        
    def _load_impact_weights(self) -> Dict[str, float]:
        """Load impact assessment weights for different risk categories."""
        return {
            'data_quality': {
                'financial': 0.3,
                'operational': 0.4,
                'strategic': 0.2,
                'reputation': 0.1,
                'compliance': 0.2
            },
            'performance': {
                'financial': 0.4,
                'operational': 0.5,
                'strategic': 0.3,
                'reputation': 0.2,
                'compliance': 0.1
            },
            'security': {
                'financial': 0.5,
                'operational': 0.3,
                'strategic': 0.4,
                'reputation': 0.6,
                'compliance': 0.8
            },
            'business': {
                'financial': 0.8,
                'operational': 0.4,
                'strategic': 0.7,
                'reputation': 0.3,
                'compliance': 0.2
            }
        }
    
    def _load_financial_models(self) -> Dict[str, Dict[str, Any]]:
        """Load financial impact calculation models."""
        return {
            'data_quality': {
                'cost_per_incident': 5000,  # USD
                'time_to_resolve': 24,  # hours
                'productivity_loss': 0.15  # 15% productivity loss
            },
            'performance': {
                'cost_per_minute_downtime': 1000,  # USD
                'user_experience_impact': 0.25,  # 25% user satisfaction drop
                'revenue_loss_percentage': 0.1  # 10% revenue loss
            },
            'security': {
                'breach_cost_per_record': 150,  # USD
                'regulatory_fine_base': 50000,  # USD
                'reputation_damage_cost': 100000  # USD
            },
            'business': {
                'revenue_impact_multiplier': 1.5,
                'customer_acquisition_cost': 200,  # USD
                'lifetime_value_loss': 0.3  # 30% LTV loss
            }
        }
    
    def calculate_financial_impact(self, risk: Risk, 
                                 business_context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate the financial impact of a risk."""
        base_models = self.financial_models.get(risk.category, {})
        context = business_context or {}
        
        if risk.category == 'data_quality':
            incidents_per_month = context.get('data_incidents_per_month', 2)
            team_size = context.get('affected_team_size', 10)
            hourly_rate = context.get('avg_hourly_rate', 50)
            
            direct_cost = base_models['cost_per_incident'] * incidents_per_month
            productivity_loss = (team_size * hourly_rate * 
                               base_models['time_to_resolve'] * 
                               base_models['productivity_loss'])
            
            return direct_cost + productivity_loss
            
        elif risk.category == 'performance':
            downtime_minutes = context.get('avg_downtime_minutes', 30)
            monthly_revenue = context.get('monthly_revenue', 100000)
            
            downtime_cost = (base_models['cost_per_minute_downtime'] * 
                           downtime_minutes)
            revenue_loss = (monthly_revenue * 
                          base_models['revenue_loss_percentage'])
            
            return downtime_cost + revenue_loss
            
        elif risk.category == 'security':
            records_affected = context.get('records_affected', 1000)
            regulatory_multiplier = context.get('regulatory_multiplier', 1.0)
            
            breach_cost = base_models['breach_cost_per_record'] * records_affected
            regulatory_fine = (base_models['regulatory_fine_base'] * 
                             regulatory_multiplier)
            reputation_cost = base_models['reputation_damage_cost']
            
            return breach_cost + regulatory_fine + reputation_cost
            
        elif risk.category == 'business':
            monthly_revenue = context.get('monthly_revenue', 100000)
            customers_affected = context.get('customers_affected', 100)
            
            revenue_impact = (monthly_revenue * 
                            base_models['revenue_impact_multiplier'] * 
                            risk.probability)
            customer_loss = (customers_affected * 
                           base_models['customer_acquisition_cost'])
            
            return revenue_impact + customer_loss
        
        return 0.0
    
    def calculate_operational_impact(self, risk: Risk, 
                                   operational_context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate the operational impact of a risk."""
        context = operational_context or {}
        
        # Base operational impact based on risk severity
        severity_impact = {
            'low': 0.2,
            'medium': 0.4,
            'high': 0.7,
            'critical': 0.9
        }
        
        base_impact = severity_impact.get(risk.severity, 0.5)
        
        # Adjust based on affected components
        component_multiplier = len(risk.affected_components) * 0.1
        adjusted_impact = min(base_impact + component_multiplier, 1.0)
        
        # Consider operational context
        if context.get('business_critical_hours', False):
            adjusted_impact *= 1.3
        
        if context.get('peak_usage_period', False):
            adjusted_impact *= 1.2
        
        return min(adjusted_impact, 1.0)
    
    def calculate_strategic_impact(self, risk: Risk, 
                                 strategic_context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate the strategic impact of a risk."""
        context = strategic_context or {}
        
        # Base strategic impact
        strategic_impact = risk.probability * risk.impact
        
        # Adjust based on strategic importance
        if context.get('core_business_function', False):
            strategic_impact *= 1.5
        
        if context.get('competitive_advantage', False):
            strategic_impact *= 1.3
        
        if context.get('regulatory_requirement', False):
            strategic_impact *= 1.4
        
        return min(strategic_impact, 1.0)
    
    def calculate_reputation_impact(self, risk: Risk, 
                                  reputation_context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate the reputation impact of a risk."""
        context = reputation_context or {}
        
        # Base reputation impact based on risk category
        category_reputation_impact = {
            'data_quality': 0.3,
            'performance': 0.4,
            'security': 0.8,
            'business': 0.5
        }
        
        base_impact = category_reputation_impact.get(risk.category, 0.5)
        
        # Adjust based on public visibility
        if context.get('public_facing', False):
            base_impact *= 1.5
        
        if context.get('media_attention', False):
            base_impact *= 1.8
        
        if context.get('customer_impact', False):
            base_impact *= 1.3
        
        return min(base_impact * risk.probability, 1.0)
    
    def calculate_compliance_impact(self, risk: Risk, 
                                  compliance_context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate the compliance impact of a risk."""
        context = compliance_context or {}
        
        # Base compliance impact
        compliance_impact = 0.0
        
        if risk.category == 'security':
            compliance_impact = 0.8
        elif risk.category == 'data_quality':
            compliance_impact = 0.6
        elif risk.category == 'business':
            compliance_impact = 0.4
        elif risk.category == 'performance':
            compliance_impact = 0.3
        
        # Adjust based on regulatory requirements
        if context.get('gdpr_applicable', False):
            compliance_impact *= 1.4
        
        if context.get('sox_compliance', False):
            compliance_impact *= 1.3
        
        if context.get('industry_regulations', False):
            compliance_impact *= 1.2
        
        return min(compliance_impact * risk.probability, 1.0)
    
    def calculate_total_impact_score(self, risk: Risk, 
                                   financial_impact: float,
                                   operational_impact: float,
                                   strategic_impact: float,
                                   reputation_impact: float,
                                   compliance_impact: float) -> float:
        """Calculate the total weighted impact score."""
        weights = self.impact_weights.get(risk.category, {
            'financial': 0.3,
            'operational': 0.3,
            'strategic': 0.2,
            'reputation': 0.1,
            'compliance': 0.1
        })
        
        # Normalize financial impact to 0-1 scale (assuming max $1M impact)
        normalized_financial = min(financial_impact / 1000000, 1.0)
        
        total_score = (
            weights['financial'] * normalized_financial +
            weights['operational'] * operational_impact +
            weights['strategic'] * strategic_impact +
            weights['reputation'] * reputation_impact +
            weights['compliance'] * compliance_impact
        )
        
        return min(total_score, 1.0)
    
    def quantify_risk_impact(self, risk: Risk, 
                           business_context: Optional[Dict[str, Any]] = None,
                           operational_context: Optional[Dict[str, Any]] = None,
                           strategic_context: Optional[Dict[str, Any]] = None,
                           reputation_context: Optional[Dict[str, Any]] = None,
                           compliance_context: Optional[Dict[str, Any]] = None) -> RiskImpact:
        """Quantify the comprehensive impact of a risk."""
        
        # Calculate individual impact components
        financial_impact = self.calculate_financial_impact(risk, business_context)
        operational_impact = self.calculate_operational_impact(risk, operational_context)
        strategic_impact = self.calculate_strategic_impact(risk, strategic_context)
        reputation_impact = self.calculate_reputation_impact(risk, reputation_context)
        compliance_impact = self.calculate_compliance_impact(risk, compliance_context)
        
        # Calculate total impact score
        total_impact_score = self.calculate_total_impact_score(
            risk, financial_impact, operational_impact, strategic_impact,
            reputation_impact, compliance_impact
        )
        
        # Create impact breakdown
        impact_breakdown = {
            'financial': financial_impact,
            'operational': operational_impact,
            'strategic': strategic_impact,
            'reputation': reputation_impact,
            'compliance': compliance_impact
        }
        
        # Calculate confidence level based on data quality
        confidence_level = risk.confidence * 0.8 + 0.2  # Base confidence
        
        return RiskImpact(
            risk_id=risk.id,
            financial_impact=financial_impact,
            operational_impact=operational_impact,
            strategic_impact=strategic_impact,
            reputation_impact=reputation_impact,
            compliance_impact=compliance_impact,
            total_impact_score=total_impact_score,
            impact_breakdown=impact_breakdown,
            confidence_level=confidence_level,
            assessment_date=datetime.now()
        )
    
    def quantify_multiple_risks(self, risks: List[Risk], 
                              context: Optional[Dict[str, Any]] = None) -> List[RiskImpact]:
        """Quantify the impact of multiple risks."""
        context = context or {}
        
        risk_impacts = []
        for risk in risks:
            impact = self.quantify_risk_impact(
                risk,
                business_context=context.get('business', {}),
                operational_context=context.get('operational', {}),
                strategic_context=context.get('strategic', {}),
                reputation_context=context.get('reputation', {}),
                compliance_context=context.get('compliance', {})
            )
            risk_impacts.append(impact)
        
        self.logger.info(f"Quantified impact for {len(risk_impacts)} risks")
        return risk_impacts
    
    def get_impact_summary(self, risk_impacts: List[RiskImpact]) -> Dict[str, Any]:
        """Generate a summary of risk impacts."""
        if not risk_impacts:
            return {'total_financial_impact': 0, 'avg_impact_score': 0}
        
        total_financial = sum(impact.financial_impact for impact in risk_impacts)
        avg_impact_score = np.mean([impact.total_impact_score for impact in risk_impacts])
        avg_confidence = np.mean([impact.confidence_level for impact in risk_impacts])
        
        # Impact distribution by category
        impact_by_category = {}
        for impact in risk_impacts:
            category = impact.risk_id.split('_')[0]  # Extract category from risk ID
            if category not in impact_by_category:
                impact_by_category[category] = []
            impact_by_category[category].append(impact.total_impact_score)
        
        category_avg_impacts = {
            category: np.mean(scores) 
            for category, scores in impact_by_category.items()
        }
        
        return {
            'total_financial_impact': total_financial,
            'avg_impact_score': avg_impact_score,
            'avg_confidence': avg_confidence,
            'category_avg_impacts': category_avg_impacts,
            'total_risks_assessed': len(risk_impacts)
        }
