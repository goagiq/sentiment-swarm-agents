"""
Risk Identifier Module

This module provides automated risk detection capabilities for identifying
various types of risks in business operations, data, and system performance.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Risk:
    """Risk data structure for storing identified risks."""
    id: str
    category: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    probability: float  # 0.0 to 1.0
    impact: float  # 0.0 to 1.0
    description: str
    source: str
    detected_at: datetime
    affected_components: List[str]
    indicators: List[str]
    confidence: float  # 0.0 to 1.0


class RiskIdentifier:
    """
    Automated risk detection system that identifies various types of risks
    in business operations, data quality, system performance, and security.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the risk identifier with configuration."""
        self.config = config or {}
        self.risk_patterns = self._load_risk_patterns()
        self.thresholds = self._load_thresholds()
        self.logger = logging.getLogger(__name__)
        
    def _load_risk_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined risk patterns and indicators."""
        return {
            'data_quality': {
                'missing_data': {'threshold': 0.1, 'weight': 0.8},
                'duplicate_data': {'threshold': 0.05, 'weight': 0.6},
                'inconsistent_data': {'threshold': 0.15, 'weight': 0.7},
                'outdated_data': {'threshold': 7, 'weight': 0.5}  # days
            },
            'performance': {
                'response_time': {'threshold': 2.0, 'weight': 0.9},  # seconds
                'error_rate': {'threshold': 0.05, 'weight': 0.8},
                'resource_usage': {'threshold': 0.8, 'weight': 0.7},
                'throughput_decline': {'threshold': 0.2, 'weight': 0.6}
            },
            'security': {
                'failed_logins': {'threshold': 5, 'weight': 0.9},
                'suspicious_activity': {'threshold': 3, 'weight': 0.8},
                'data_access_patterns': {'threshold': 0.1, 'weight': 0.7},
                'encryption_issues': {'threshold': 0.0, 'weight': 1.0}
            },
            'business': {
                'revenue_decline': {'threshold': 0.1, 'weight': 0.9},
                'customer_churn': {'threshold': 0.05, 'weight': 0.8},
                'operational_efficiency': {'threshold': 0.15, 'weight': 0.7},
                'compliance_violations': {'threshold': 0.0, 'weight': 1.0}
            }
        }
    
    def _load_thresholds(self) -> Dict[str, float]:
        """Load risk detection thresholds."""
        return {
            'low_risk': 0.3,
            'medium_risk': 0.5,
            'high_risk': 0.7,
            'critical_risk': 0.9
        }
    
    def identify_data_quality_risks(self, data_metrics: Dict[str, Any]) -> List[Risk]:
        """Identify data quality related risks."""
        risks = []
        
        # Check for missing data
        if data_metrics.get('missing_rate', 0) > self.risk_patterns['data_quality']['missing_data']['threshold']:
            risks.append(Risk(
                id=f"dq_missing_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                category='data_quality',
                severity=self._calculate_severity(data_metrics['missing_rate']),
                probability=data_metrics['missing_rate'],
                impact=0.7,
                description=f"High missing data rate: {data_metrics['missing_rate']:.2%}",
                source='data_analysis',
                detected_at=datetime.now(),
                affected_components=['data_pipeline', 'analytics'],
                indicators=['missing_rate', 'data_completeness'],
                confidence=0.8
            ))
        
        # Check for duplicate data
        if data_metrics.get('duplicate_rate', 0) > self.risk_patterns['data_quality']['duplicate_data']['threshold']:
            risks.append(Risk(
                id=f"dq_duplicate_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                category='data_quality',
                severity=self._calculate_severity(data_metrics['duplicate_rate']),
                probability=data_metrics['duplicate_rate'],
                impact=0.5,
                description=f"High duplicate data rate: {data_metrics['duplicate_rate']:.2%}",
                source='data_analysis',
                detected_at=datetime.now(),
                affected_components=['data_storage', 'analytics'],
                indicators=['duplicate_rate', 'data_integrity'],
                confidence=0.7
            ))
        
        return risks
    
    def identify_performance_risks(self, performance_metrics: Dict[str, Any]) -> List[Risk]:
        """Identify system performance related risks."""
        risks = []
        
        # Check response time
        if performance_metrics.get('avg_response_time', 0) > self.risk_patterns['performance']['response_time']['threshold']:
            risks.append(Risk(
                id=f"perf_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                category='performance',
                severity=self._calculate_severity(performance_metrics['avg_response_time'] / 5.0),
                probability=min(performance_metrics['avg_response_time'] / 5.0, 1.0),
                impact=0.8,
                description=f"High response time: {performance_metrics['avg_response_time']:.2f}s",
                source='system_monitoring',
                detected_at=datetime.now(),
                affected_components=['api', 'database', 'processing'],
                indicators=['response_time', 'user_experience'],
                confidence=0.9
            ))
        
        # Check error rate
        if performance_metrics.get('error_rate', 0) > self.risk_patterns['performance']['error_rate']['threshold']:
            risks.append(Risk(
                id=f"perf_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                category='performance',
                severity=self._calculate_severity(performance_metrics['error_rate']),
                probability=performance_metrics['error_rate'],
                impact=0.9,
                description=f"High error rate: {performance_metrics['error_rate']:.2%}",
                source='system_monitoring',
                detected_at=datetime.now(),
                affected_components=['system_reliability', 'user_experience'],
                indicators=['error_rate', 'system_stability'],
                confidence=0.8
            ))
        
        return risks
    
    def identify_security_risks(self, security_metrics: Dict[str, Any]) -> List[Risk]:
        """Identify security related risks."""
        risks = []
        
        # Check for failed login attempts
        if security_metrics.get('failed_logins', 0) > self.risk_patterns['security']['failed_logins']['threshold']:
            risks.append(Risk(
                id=f"sec_logins_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                category='security',
                severity='high',
                probability=min(security_metrics['failed_logins'] / 10.0, 1.0),
                impact=0.9,
                description=f"Multiple failed login attempts: {security_metrics['failed_logins']}",
                source='security_monitoring',
                detected_at=datetime.now(),
                affected_components=['authentication', 'user_accounts'],
                indicators=['failed_logins', 'brute_force_attempts'],
                confidence=0.9
            ))
        
        # Check for suspicious activity
        if security_metrics.get('suspicious_activity_count', 0) > self.risk_patterns['security']['suspicious_activity']['threshold']:
            risks.append(Risk(
                id=f"sec_suspicious_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                category='security',
                severity='medium',
                probability=min(security_metrics['suspicious_activity_count'] / 5.0, 1.0),
                impact=0.7,
                description=f"Suspicious activity detected: {security_metrics['suspicious_activity_count']} events",
                source='security_monitoring',
                detected_at=datetime.now(),
                affected_components=['security', 'data_protection'],
                indicators=['suspicious_activity', 'anomaly_detection'],
                confidence=0.7
            ))
        
        return risks
    
    def identify_business_risks(self, business_metrics: Dict[str, Any]) -> List[Risk]:
        """Identify business related risks."""
        risks = []
        
        # Check for revenue decline
        if business_metrics.get('revenue_change', 0) < -self.risk_patterns['business']['revenue_decline']['threshold']:
            risks.append(Risk(
                id=f"biz_revenue_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                category='business',
                severity=self._calculate_severity(abs(business_metrics['revenue_change'])),
                probability=min(abs(business_metrics['revenue_change']), 1.0),
                impact=0.9,
                description=f"Revenue decline detected: {business_metrics['revenue_change']:.2%}",
                source='business_analytics',
                detected_at=datetime.now(),
                affected_components=['revenue', 'business_operations'],
                indicators=['revenue_change', 'financial_performance'],
                confidence=0.8
            ))
        
        # Check for customer churn
        if business_metrics.get('churn_rate', 0) > self.risk_patterns['business']['customer_churn']['threshold']:
            risks.append(Risk(
                id=f"biz_churn_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                category='business',
                severity=self._calculate_severity(business_metrics['churn_rate']),
                probability=business_metrics['churn_rate'],
                impact=0.8,
                description=f"High customer churn rate: {business_metrics['churn_rate']:.2%}",
                source='business_analytics',
                detected_at=datetime.now(),
                affected_components=['customer_retention', 'business_growth'],
                indicators=['churn_rate', 'customer_satisfaction'],
                confidence=0.7
            ))
        
        return risks
    
    def _calculate_severity(self, risk_score: float) -> str:
        """Calculate risk severity based on risk score."""
        if risk_score >= self.thresholds['critical_risk']:
            return 'critical'
        elif risk_score >= self.thresholds['high_risk']:
            return 'high'
        elif risk_score >= self.thresholds['medium_risk']:
            return 'medium'
        else:
            return 'low'
    
    def identify_all_risks(self, 
                          data_metrics: Optional[Dict[str, Any]] = None,
                          performance_metrics: Optional[Dict[str, Any]] = None,
                          security_metrics: Optional[Dict[str, Any]] = None,
                          business_metrics: Optional[Dict[str, Any]] = None) -> List[Risk]:
        """Identify all types of risks from provided metrics."""
        all_risks = []
        
        if data_metrics:
            all_risks.extend(self.identify_data_quality_risks(data_metrics))
        
        if performance_metrics:
            all_risks.extend(self.identify_performance_risks(performance_metrics))
        
        if security_metrics:
            all_risks.extend(self.identify_security_risks(security_metrics))
        
        if business_metrics:
            all_risks.extend(self.identify_business_risks(business_metrics))
        
        self.logger.info(f"Identified {len(all_risks)} risks across all categories")
        return all_risks
    
    def get_risk_summary(self, risks: List[Risk]) -> Dict[str, Any]:
        """Generate a summary of identified risks."""
        if not risks:
            return {'total_risks': 0, 'risk_distribution': {}}
        
        risk_distribution = {
            'critical': len([r for r in risks if r.severity == 'critical']),
            'high': len([r for r in risks if r.severity == 'high']),
            'medium': len([r for r in risks if r.severity == 'medium']),
            'low': len([r for r in risks if r.severity == 'low'])
        }
        
        category_distribution = {}
        for risk in risks:
            category_distribution[risk.category] = category_distribution.get(risk.category, 0) + 1
        
        return {
            'total_risks': len(risks),
            'risk_distribution': risk_distribution,
            'category_distribution': category_distribution,
            'avg_confidence': np.mean([r.confidence for r in risks]),
            'avg_impact': np.mean([r.impact for r in risks]),
            'avg_probability': np.mean([r.probability for r in risks])
        }
